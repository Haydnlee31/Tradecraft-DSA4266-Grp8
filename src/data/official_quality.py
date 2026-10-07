"""Read-only official-39 quality and disk-backed duplicate audit.

No train/validation/test data are written, and no imputation is invented. Exact
finite numeric feature vectors are fingerprinted with SHA256 independently of
their labels, so conflicting labels cannot hide behind different label hashes.
This does not prove absence of near-duplicates, session leakage or mirror overlap.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import pyarrow as pa
import pyarrow.csv as pc

from src.data.label_map import CLASSES, LABEL_TO_CLASS, official_raw_label
from src.data.official_inventory import sha256


def fingerprints(values):
    """Canonical little-endian float64; +0 and -0 denote the same value.

    Caller excludes all nonfinite rows before hashing. Hashes refer to parsed
    numerical equality, not byte-identical CSV text. Hash collision risk is not
    mathematically zero; this is a SHA256-based duplicate audit.
    """
    array = np.array(values, dtype='<f8', order='C', copy=True)
    if not np.isfinite(array).all():
        raise ValueError('Fingerprinting requires finite values')
    array[array == 0] = 0
    return [hashlib.sha256(row.tobytes()).digest() for row in array]


def open_index(path, create=True, cache_mib=512):
    if not 16 <= cache_mib <= 1024:
        raise ValueError('SQLite cache must be between 16 and 1024 MiB')
    connection = sqlite3.connect(path)
    connection.execute(f'PRAGMA cache_size=-{cache_mib * 1024}')
    connection.execute('PRAGMA temp_store=FILE')
    if create:
        connection.execute('''CREATE TABLE vectors (
        digest BLOB PRIMARY KEY, n INTEGER NOT NULL, raw_mask INTEGER NOT NULL,
        class_mask INTEGER NOT NULL, first_file INTEGER NOT NULL,
        cross_file INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID''')
    return connection


def add_vectors(connection, hashes, raw_mask, class_mask, file_id):
    # Batch-local aggregation reduces SQLite work without changing counts.
    counts = Counter(hashes)
    connection.executemany('''INSERT INTO vectors
        (digest,n,raw_mask,class_mask,first_file) VALUES (?,?,?,?,?)
        ON CONFLICT(digest) DO UPDATE SET n=n+excluded.n,
        raw_mask=raw_mask|excluded.raw_mask, class_mask=class_mask|excluded.class_mask,
        cross_file=cross_file|(first_file!=excluded.first_file)''',
        ((key, n, raw_mask, class_mask, file_id) for key, n in counts.items()))


def summarize_index(connection):
    names = ['unique_vectors', 'finite_rows', 'duplicate_excess_rows',
             'cross_file_vectors', 'raw_label_conflict_vectors', 'class_conflict_vectors']
    row = connection.execute('''SELECT count(*),coalesce(sum(n),0),
        coalesce(sum(n-1),0),coalesce(sum(cross_file),0),
        coalesce(sum((raw_mask & (raw_mask-1))!=0),0),
        coalesce(sum((class_mask & (class_mask-1))!=0),0) FROM vectors''').fetchone()
    result = dict(zip(names, row))
    result['unambiguous_unique_per_class'] = {}
    for mask, count in connection.execute('SELECT class_mask,count(*) FROM vectors GROUP BY class_mask'):
        if mask & (mask - 1) == 0:
            result['unambiguous_unique_per_class'][CLASSES[mask.bit_length()-1]] = count
    return result


def run(source, schema_receipt, output, resume=False, cache_mib=512):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and not resume:
        raise FileExistsError('Use a new audit directory; do not overwrite evidence')
    if resume and not output.exists():
        raise FileNotFoundError('Resume requires an existing incomplete audit')
    if not 16 <= cache_mib <= 1024:
        raise ValueError('SQLite cache must be between 16 and 1024 MiB')
    if output.is_relative_to(source):
        raise ValueError('Audit output must be outside raw inputs')
    prior = json.loads(Path(schema_receipt).read_text())
    profiles = prior['schema_profiles']
    if len(profiles) != 1 or len(profiles[0]['columns']) != 39:
        raise ValueError('Expected the audited single 39-feature layout')
    features = profiles[0]['columns']
    expected = {record['path']: record for record in prior['files']}
    actual = {p.relative_to(source).as_posix() for p in source.rglob('*.csv')}
    if set(expected) != actual:
        raise ValueError('Source file listing differs from schema audit')
    labels = list(LABEL_TO_CLASS)
    for name in expected:
        relative = Path(name)
        if len(relative.parts) != 2 or not (source / name).resolve().is_relative_to(source):
            raise ValueError('Expected attack-folder/file.csv layout')
        official_raw_label(relative.parent.name)
    output.mkdir(parents=True, exist_ok=resume)
    result = {'status': 'incomplete', 'training_ready': False, 'split_frozen': False,
              'mirror_overlap_checked': False, 'near_duplicates_checked': False,
              'source_manifest_sha256': sha256(schema_receipt), 'builder_sha256': sha256(__file__),
              'label_map_sha256': sha256(Path(__file__).with_name('label_map.py')),
              'features': features, 'files': [], 'label_mapping': {
                  Path(name).parent.name: official_raw_label(Path(name).parent.name) for name in expected}}
    status = output / 'summary.json'
    if resume:
        result = json.loads(status.read_text())
        if (result['status'] != 'incomplete' or result['features'] != features or
                result['source_manifest_sha256'] != sha256(schema_receipt) or
                result['label_map_sha256'] != sha256(Path(__file__).with_name('label_map.py'))):
            raise ValueError('Only matching incomplete audit checkpoints can resume')
        if not (output / 'vectors.sqlite').exists():
            raise FileNotFoundError('Missing duplicate index')
        result['resume_builder_sha256'] = sha256(__file__)
    completed = len(result['files'])
    # Preserve the exact implementation of each execution segment for provenance.
    shutil.copyfile(__file__, output / f'audit_source_{sha256(__file__)[:12]}.py')
    connection = open_index(output / 'vectors.sqlite', create=not resume, cache_mib=cache_mib)
    if resume:
        committed = connection.execute('SELECT coalesce(sum(n),0) FROM vectors').fetchone()[0]
        recorded = sum(r['rows'] - r['nonfinite_rows'] for r in result['files'])
        if committed != recorded:
            connection.close()
            raise ValueError('Database/checkpoint boundary mismatch; refuse unsafe resume')
    result['sqlite_cache_mib'] = cache_mib
    result['timing_scope'] = 'final execution segment only' if resume else 'entire audit'
    status.write_text(json.dumps(result, indent=2) + '\n')
    started = time.monotonic()
    try:
        for file_id, (name, receipt) in enumerate(sorted(expected.items())):
            if shutil.disk_usage(output).free < 10 * 1024**3:
                raise RuntimeError('Less than 10 GiB free; stopped without altering raw inputs')
            path = source / name
            before = path.stat()
            if sha256(path) != receipt['sha256']:
                raise ValueError(f'Source hash changed: {name}')
            if file_id < completed:
                saved = result['files'][file_id]
                if saved['path'] != name or saved['sha256'] != receipt['sha256']:
                    raise ValueError('Completed-file prefix differs from source manifest')
                continue
            with path.open(newline='', encoding='utf-8-sig') as stream:
                if next(csv.reader(stream)) != features:
                    raise ValueError(f'Schema changed: {name}')
            label = official_raw_label(path.parent.name)
            record = {'path': name, 'sha256': receipt['sha256'], 'raw_label': label,
                      'class': LABEL_TO_CLASS[label], 'rows': 0, 'malformed': 0,
                      'nonfinite_rows': 0, 'float32_overflow_rows': 0,
                      'nonfinite_by_feature': {name: 0 for name in features}}

            def bad_row(row):
                record['malformed'] += 1
                return 'skip'  # Counted quarantine candidate, not silent cleaning.

            reader = pc.open_csv(path, read_options=pc.ReadOptions(block_size=8*1024*1024, use_threads=False),
                                 parse_options=pc.ParseOptions(invalid_row_handler=bad_row),
                                 convert_options=pc.ConvertOptions(column_types={name: pa.float64() for name in features}))
            for batch in reader:
                values = np.column_stack([batch.column(name).to_numpy(zero_copy_only=False) for name in features])
                invalid = ~np.isfinite(values)
                finite = ~invalid.any(axis=1)
                record['rows'] += len(values)
                record['nonfinite_rows'] += int((~finite).sum())
                record['float32_overflow_rows'] += int((finite & (np.abs(values) > np.finfo(np.float32).max).any(axis=1)).sum())
                for feature, n in zip(features, invalid.sum(axis=0)):
                    record['nonfinite_by_feature'][feature] += int(n)
                add_vectors(connection, fingerprints(values[finite]), 1 << labels.index(label),
                            1 << CLASSES.index(LABEL_TO_CLASS[label]), file_id)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'Source changed during audit: {name}')
            if record['rows'] != receipt['rows'] or record['malformed'] != receipt['malformed_records']['count']:
                raise ValueError(f'Parser counts changed: {name}')
            connection.commit()
            result['files'].append(record)
            status.write_text(json.dumps(result, indent=2) + '\n')
            print(f"{file_id+1}/{len(expected)} {name}: rows={record['rows']} invalid={record['nonfinite_rows']} elapsed={time.monotonic()-started:.1f}s", flush=True)
        result['duplicates'] = summarize_index(connection)
    finally:
        connection.close()
    result['totals'] = {key: sum(record[key] for record in result['files']) for key in
                        ('rows', 'malformed', 'nonfinite_rows', 'float32_overflow_rows')}
    result['single_file_raw_labels'] = sorted(label for label, n in
        Counter(record['raw_label'] for record in result['files']).items() if n == 1)
    result['elapsed_seconds'] = time.monotonic() - started
    result['status'] = 'complete'
    status.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--schema-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cache-mib', type=int, default=512)
    args = parser.parse_args()
    result = run(args.source, args.schema_receipt, args.output, args.resume, args.cache_mib)
    print(json.dumps({'totals': result['totals'], 'duplicates': result['duplicates']}, indent=2))


if __name__ == '__main__':
    main()
