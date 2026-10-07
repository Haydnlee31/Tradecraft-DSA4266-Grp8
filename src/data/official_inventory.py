"""Bounded-memory CSV inventory and optional *unsplit* Parquet staging.

This extension never replaces the original splits or assigns train/val/test.
Staging is not training approval: overlap, duplicate and split audits come next.
PyArrow reads a bounded block; no full file or full class is collected in RAM.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as arrow_csv
import pyarrow.parquet as pq

from src.data.label_map import CLASSES, LABEL_TO_CLASS


def sha256(path):
    """Hash even very large inputs without materializing them in memory."""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inventory(source, output, features, block_bytes=8 * 1024 * 1024,
              shard_rows=65536, write_shards=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if block_bytes < 1024 or shard_rows < 1:
        raise ValueError('block_bytes >= 1024 and shard_rows >= 1 required')
    if len(features) != 46 or len(set(features)) != 46 or 'label' in features:
        raise ValueError('Expected exactly 46 distinct numeric feature names')
    if output.exists():
        raise FileExistsError('Use a new output directory; existing evidence is preserved')
    if output.is_relative_to(source):
        raise ValueError('Output must be outside the raw source directory')
    files = sorted(source.rglob('*.csv'))
    if not files:
        raise FileNotFoundError('No CSV inputs; obtain the official CSV release first')
    # Header inspection is cheap; fail before creating outputs if schemas drift.
    for path in files:
        if not path.resolve().is_relative_to(source):
            raise ValueError('CSV symlink escapes source directory')
        with path.open(newline='', encoding='utf-8-sig') as stream:
            header = next(csv.reader(stream), [])
        if len(header) != 47 or set(header) != {*features, 'label'}:
            raise ValueError(f'Unexpected schema: {path}; do not silently rename features')
    output.mkdir(parents=True)
    receipt = {'status': 'incomplete', 'source_root': str(source),
               'feature_columns': features, 'classes': CLASSES,
               'block_bytes': block_bytes, 'max_shard_rows': shard_rows,
               'split_assigned': False, 'training_ready': False,
               'duplicates_checked': False, 'mirror_overlap_checked': False,
               'official_origin_verified': False, 'files': [], 'shards': []}
    status_path = output / 'inventory.json'
    status_path.write_text(json.dumps(receipt, indent=2) + '\n')
    totals = Counter()
    for file_id, path in enumerate(files):
        before = path.stat()
        source_hash = sha256(path)
        counts, invalid, offset = Counter(), Counter(), 0
        reader = arrow_csv.open_csv(
            path, read_options=arrow_csv.ReadOptions(block_size=block_bytes, use_threads=False),
            convert_options=arrow_csv.ConvertOptions(
                column_types={**{name: pa.float64() for name in features}, 'label': pa.string()},
                include_columns=[*features, 'label']))
        for batch in reader:
            labels = batch.column('label').to_pylist()
            unknown = set(labels) - set(LABEL_TO_CLASS)
            if unknown:
                raise ValueError(f'Unknown/null labels in {path}: {unknown}')
            counts.update(labels)
            for name in features:
                values = batch.column(name).to_numpy(zero_copy_only=False)
                invalid[name] += int((~np.isfinite(values)).sum())
            if write_shards:
                # Cleaning rules are a separate study decision. Never silently
                # discard, impute or change invalid rows while staging inputs.
                if any(invalid.values()):
                    raise ValueError(f'Nonfinite features in {path}; run inventory-only and review cleaning')
                for start in range(0, batch.num_rows, shard_rows):
                    piece = batch.slice(start, shard_rows)
                    table = pa.Table.from_batches([piece])
                    table = table.append_column('source_file_id', pa.array([file_id] * len(table), type=pa.int32()))
                    table = table.append_column('source_row', pa.array(
                        range(offset + start, offset + start + len(table)), type=pa.int64()))
                    name = f'shard-{len(receipt["shards"]):06d}.parquet'
                    pq.write_table(table, output / name, compression='zstd', row_group_size=shard_rows)
                    receipt['shards'].append({'path': name, 'rows': len(table),
                                               'sha256': sha256(output / name)})
            offset += batch.num_rows
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Source changed during inventory: {path}')
        totals.update(counts)
        receipt['files'].append({'file_id': file_id, 'path': path.relative_to(source).as_posix(),
                                 'sha256': source_hash, 'bytes': before.st_size, 'rows': offset,
                                 'raw_label_counts': dict(sorted(counts.items())),
                                 'nonfinite_by_feature': dict(invalid)})
    receipt['raw_label_counts'] = {name: totals[name] for name in LABEL_TO_CLASS}
    receipt['class_counts'] = {name: sum(n for raw, n in totals.items()
                                        if LABEL_TO_CLASS[raw] == name) for name in CLASSES}
    receipt['rows'] = sum(totals.values())
    receipt['status'] = 'complete'
    # A complete receipt is written last. Failed staging directories remain
    # visibly incomplete for inspection and cannot be consumed by the reader.
    status_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def iter_staged_batches(directory, batch_rows=4096):
    """Read bounded unsplit Arrow batches for audits, NOT a training DataLoader.

No shuffling/scaling is performed. The future training adapter must use frozen
split/client assignments and must never expose provenance columns as features.
"""
    if batch_rows < 1:
        raise ValueError('batch_rows must be positive')
    directory = Path(directory).resolve()
    receipt = json.loads((directory / 'inventory.json').read_text())
    if receipt['status'] != 'complete' or not receipt['shards']:
        raise ValueError('Complete sharded staging required')
    for item in receipt['shards']:
        path = (directory / item['path']).resolve()
        if not path.is_relative_to(directory) or sha256(path) != item['sha256']:
            raise ValueError('Invalid shard path or hash')
        with pq.ParquetFile(path) as parquet:
            if parquet.metadata.num_rows != item['rows']:
                raise ValueError('Shard row count mismatch')
            yield from parquet.iter_batches(batch_size=batch_rows, use_threads=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-metadata', type=Path,
                        default=Path('data/splits/feature_scaler.json'),
                        help='Read feature names only; never reuse the old scaler coefficients')
    parser.add_argument('--block-mib', type=int, default=8)
    parser.add_argument('--shard-rows', type=int, default=65536)
    parser.add_argument('--write-shards', action='store_true')
    args = parser.parse_args()
    features = json.loads(args.feature_metadata.read_text())['feature_columns']
    result = inventory(args.source, args.output, features, args.block_mib * 1024 * 1024,
                       args.shard_rows, args.write_shards)
    print(json.dumps({'rows': result['rows'], 'class_counts': result['class_counts'],
                      'training_ready': result['training_ready']}, indent=2))


if __name__ == '__main__':
    main()
