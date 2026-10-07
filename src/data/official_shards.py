"""Materialize the frozen official-39 recipe without loading the dataset in RAM.

The audited index assigns each digest to its first source file. A sequential
index pass writes small selection files; only one source file's selection is
loaded at a time, with an explicit cap. Raw CSVs and the audit DB stay read-only.
All 39 features remain float64 so fingerprints can be checked after writing.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sqlite3
import struct

import numpy as np
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_protocol import split_for_digest
from src.data.official_quality import fingerprints

RECORD = struct.Struct('<32sQQ')  # feature digest, class mask, raw-label mask
SPLITS = ('train', 'val', 'test')


def save_json(path, value):
    # Replace the receipt atomically: a killed process cannot leave half JSON.
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def selection(path, cap):
    if path.stat().st_size % RECORD.size or path.stat().st_size // RECORD.size > cap:
        raise ValueError('Invalid selection size or per-file memory cap exceeded')
    result = {}
    with path.open('rb') as stream:
        while record := stream.read(RECORD.size):
            digest, class_mask, raw_mask = RECORD.unpack(record)
            if digest in result or class_mask <= 0 or class_mask & (class_mask - 1):
                raise ValueError('Duplicate selection or ambiguous class')
            result[digest] = (class_mask.bit_length() - 1, raw_mask)
    return result


def export_selections(database, directory, file_count):
    directory.mkdir()
    # Group a bounded cursor batch before writing: random digest order would
    # otherwise reopen a file almost once per vector on low-descriptor systems.
    connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        cursor = connection.execute(
                'SELECT digest,class_mask,raw_mask,first_file FROM vectors '
                'WHERE (class_mask & (class_mask-1))=0')
        while rows := cursor.fetchmany(65536):
            buffers = {}
            for digest, mask, raw_mask, file_id in rows:
                if not 0 <= file_id < file_count or not 0 < mask < (1 << len(CLASSES)):
                    raise ValueError('Invalid audited selection metadata')
                buffers.setdefault(file_id, bytearray()).extend(RECORD.pack(digest, mask, raw_mask))
            for file_id, buffer in buffers.items():
                with (directory / f'{file_id:04d}.bin').open('ab') as stream:
                    stream.write(buffer)
    finally:
        connection.close()
    for file_id in range(file_count):
        (directory / f'{file_id:04d}.bin').touch(exist_ok=True)


def verify_file(output, record, features, cap):
    """Read shards back and consume the exact eligible set once.

    The audited primary key is globally unique, and selections partition it by
    first_file. Exhausting each selection exactly once proves both within-split
    uniqueness and cross-split disjointness without a second giant hash set.
    """
    expected = selection(output / record['selection'], cap)
    if sha256(output / record['selection']) != record['selection_sha256']:
        raise ValueError('Selection checksum changed')
    counts = {split: Counter({name: 0 for name in CLASSES}) for split in SPLITS}
    for shard in record['shards']:
        path = output / shard['path']
        if sha256(path) != shard['sha256']:
            raise ValueError(f'Shard checksum changed: {path}')
        rows = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
            values = np.column_stack([batch.column(name).to_numpy() for name in features])
            hashes = fingerprints(values)
            stored = batch.column('_digest').to_pylist()
            classes = batch.column('_class_id').to_pylist()
            masks = batch.column('_raw_label_mask').to_pylist()
            file_ids = batch.column('_source_file_id').to_pylist()
            source_rows = batch.column('_source_record').to_pylist()
            for digest, saved, target, mask, file_id, row in zip(hashes, stored, classes, masks, file_ids, source_rows):
                if (digest != saved or expected.pop(digest, None) != (target, mask)
                        or split_for_digest(digest) != shard['split']
                        or file_id != record['file_id'] or not 0 <= row < record['rows']):
                    raise ValueError('Shard content/provenance/split mismatch or duplicate')
                counts[shard['split']][CLASSES[target]] += 1
            rows += len(batch)
        if rows != shard['rows']:
            raise ValueError('Shard row count mismatch')
    if expected:
        raise ValueError('Eligible vectors missing from shards')
    return counts


def materialize(source, audit_dir, protocol_path, output, cap=400000, resume=False):
    source, audit_dir, output = Path(source).resolve(), Path(audit_dir).resolve(), Path(output).resolve()
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    summary_path, database = audit_dir / 'summary.json', audit_dir / 'vectors.sqlite'
    audit = json.loads(summary_path.read_text())
    if (protocol['audit_summary_sha256'] != sha256(summary_path)
            or protocol['duplicate_index_sha256'] != sha256(database)
            or protocol['protocol_builder_sha256'] != sha256(Path(__file__).with_name('official_protocol.py'))
            or protocol['features'] != audit['features'] or protocol['classes'] != CLASSES):
        raise ValueError('Frozen protocol or audited inputs changed')
    if cap < 1 or output.is_relative_to(source) or output.is_relative_to(audit_dir):
        raise ValueError('Invalid cap or output inside immutable inputs')
    if max(record['rows'] for record in audit['files']) > cap:
        raise ValueError('Source exceeds per-file selection cap; use a reviewed larger cap')
    files = audit['files']
    if [r['path'] for r in files] != sorted(p.relative_to(source).as_posix() for p in source.rglob('*.csv')):
        raise ValueError('Source listing changed')
    manifest_path = output / 'manifest.json'
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if (manifest['status'] != 'incomplete' or manifest['builder_sha256'] != sha256(__file__)
                or manifest['protocol_sha256'] != sha256(protocol_path) or manifest['selection_cap'] != cap):
            raise ValueError('Resume requires matching incomplete builder/recipe/cap')
    else:
        output.mkdir(parents=True, exist_ok=False)
        print('Exporting immutable index selections...', flush=True)
        export_selections(database, output / 'selections', len(files))
        shutil.copyfile(protocol_path, output / 'protocol.json')
        shutil.copyfile(__file__, output / 'builder_source.py')
        manifest = {'status': 'incomplete', 'builder_sha256': sha256(__file__),
                    'protocol_sha256': sha256(protocol_path), 'features': protocol['features'],
                    'classes': CLASSES, 'selection_cap': cap, 'files': [],
                    'test_evaluated': False, 'training_ready': False,
                    'source_record_definition': 'zero-based successfully parsed CSV record; malformed records excluded'}
        save_json(manifest_path, manifest)
    features = protocol['features']
    completed = len(manifest['files'])
    for file_id, receipt in enumerate(files):
        path = source / receipt['path']
        if sha256(path) != receipt['sha256']:
            raise ValueError(f'Source changed: {path}')
        before = path.stat()
        if file_id < completed:
            saved = manifest['files'][file_id]
            if saved['file_id'] != file_id or saved['source_sha256'] != receipt['sha256']:
                raise ValueError('Resume source prefix mismatch')
            verify_file(output, saved, features, cap)
            continue
        if shutil.disk_usage(output).free < 10 * 1024**3:
            raise RuntimeError('Less than 10 GiB free; stopped safely')
        selected_path = output / 'selections' / f'{file_id:04d}.bin'
        selected = selection(selected_path, cap)
        record = {'file_id': file_id, 'source': receipt['path'], 'source_sha256': receipt['sha256'],
                  'selection': selected_path.relative_to(output).as_posix(),
                  'selection_sha256': sha256(selected_path), 'rows': 0, 'nonfinite_rows': 0,
                  'malformed': 0, 'shards': []}

        def bad_row(row):
            record['malformed'] += 1
            return 'skip'

        reader = pc.open_csv(path, read_options=pc.ReadOptions(block_size=8*1024*1024, use_threads=False),
                             parse_options=pc.ParseOptions(invalid_row_handler=bad_row),
                             convert_options=pc.ConvertOptions(column_types={name: pa.float64() for name in features}))
        if reader.schema.names != features:
            raise ValueError('CSV feature order changed')
        for batch_id, batch in enumerate(reader):
            values = np.column_stack([batch.column(name).to_numpy(zero_copy_only=False) for name in features])
            finite_indices = np.flatnonzero(np.isfinite(values).all(axis=1))
            groups = {split: [] for split in SPLITS}
            for row, digest in zip(finite_indices, fingerprints(values[finite_indices])):
                metadata = selected.pop(digest, None)
                if metadata is not None:
                    groups[split_for_digest(digest)].append((int(row), digest, *metadata))
            for split, kept in groups.items():
                if not kept:
                    continue
                indices, digests, targets, masks = zip(*kept)
                table = pa.Table.from_batches([batch]).take(pa.array(indices, type=pa.int64()))
                table = table.append_column('_digest', pa.array(digests, type=pa.binary(32)))
                table = table.append_column('_class_id', pa.array(targets, type=pa.int8()))
                table = table.append_column('_raw_label_mask', pa.array(masks, type=pa.uint64()))
                table = table.append_column('_source_file_id', pa.array([file_id]*len(kept), type=pa.int32()))
                table = table.append_column('_source_record', pa.array([record['rows']+i for i in indices], type=pa.int64()))
                relative = f'{split}/source-{file_id:04d}-batch-{batch_id:04d}.parquet'
                destination = output / relative
                destination.parent.mkdir(exist_ok=True)
                # Only a current, uncommitted file prefix may be replaced on resume.
                pq.write_table(table, destination, compression='zstd', row_group_size=8192)
                record['shards'].append({'path': relative, 'split': split, 'rows': len(kept), 'sha256': sha256(destination)})
            record['rows'] += len(values)
            record['nonfinite_rows'] += len(values) - len(finite_indices)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during materialization')
        if selected or any(record[key] != receipt[key] for key in ('rows', 'malformed', 'nonfinite_rows')):
            raise ValueError('Source counts or selected vector coverage changed')
        record['counts'] = verify_file(output, record, features, cap)
        manifest['files'].append(record)
        save_json(manifest_path, manifest)
        print(f"{file_id+1}/{len(files)} verified {receipt['path']}", flush=True)
    counts = {split: {name: sum(r['counts'][split][name] for r in manifest['files']) for name in CLASSES} for split in SPLITS}
    if counts != protocol['counts_by_split_and_class']:
        raise ValueError('Materialized counts disagree with frozen recipe')
    manifest.update(status='complete', counts_by_split_and_class=counts,
                    exact_digest_disjointness_verified=True, data_materialized=True,
                    next_gate='bounded-memory loader parity, subset selection and trainer integration; no model evaluation yet')
    save_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-cap', type=int, default=400000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = materialize(args.source, args.audit, args.protocol, args.output, args.selection_cap, args.resume)
    print(json.dumps(result['counts_by_split_and_class'], indent=2))


if __name__ == '__main__':
    main()
