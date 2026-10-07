"""Audit unfamiliar CSV layouts without assuming they match the training schema.

Counts source records, not unique flows. Folder names are recorded verbatim, not
silently converted into ground-truth labels. Only the first column is converted
to strings; this is a structural audit, not numerical cleaning or model input.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as arrow_csv

from src.data.official_inventory import sha256


def audit(source, features):
    source = Path(source).resolve()
    files = sorted(source.rglob('*.csv'))
    if not files:
        raise ValueError('No CSV files found')
    profiles, records, folders = {}, [], Counter()
    for index, path in enumerate(files):
        if not path.resolve().is_relative_to(source):
            raise ValueError('Input path escapes source')
        before = path.stat()
        with path.open(newline='', encoding='utf-8-sig') as stream:
            header = next(csv.reader(stream))
        if not header or len(set(header)) != len(header):
            raise ValueError(f'Empty or duplicate header: {path}')
        key = tuple(header)
        if key not in profiles:
            profiles[key] = {
                'id': len(profiles), 'columns': header, 'files': 0, 'rows': 0,
                'missing_model_features': sorted(set(features) - set(header)),
                'extra_columns': sorted(set(header) - set(features) - {'label'}),
                'label_columns': [name for name in header if name.lower() == 'label'],
                'exact_training_schema': set(header) == {*features, 'label'},
            }
        malformed = {'count': 0, 'examples': []}

        def record_malformed(row):
            # Audit only: count structural failures, but do not write cleaned
            # data or call these records valid training examples.
            malformed['count'] += 1
            if len(malformed['examples']) < 3:
                malformed['examples'].append({'row_number': row.number,
                                              'expected_columns': row.expected_columns,
                                              'actual_columns': row.actual_columns})
            return 'skip'

        reader = arrow_csv.open_csv(
            path, read_options=arrow_csv.ReadOptions(block_size=8 * 1024 * 1024, use_threads=False),
            parse_options=arrow_csv.ParseOptions(invalid_row_handler=record_malformed),
            convert_options=arrow_csv.ConvertOptions(include_columns=[header[0]],
                                                     column_types={header[0]: pa.string()}))
        rows = sum(batch.num_rows for batch in reader)
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Input changed: {path}')
        profile = profiles[key]
        profile['files'] += 1
        profile['rows'] += rows
        relative = path.relative_to(source)
        folder = relative.parent.as_posix()
        folders[folder] += rows
        records.append({'path': relative.as_posix(), 'rows': rows, 'bytes': after.st_size,
                        'sha256': digest, 'schema_id': profile['id'], 'malformed_records': malformed})
        if (index + 1) % 25 == 0:
            print(f'Audited {index + 1}/{len(files)} CSV files', flush=True)
    return {'status': 'complete', 'training_ready': False, 'unique_rows_audited': False,
            'numeric_values_audited': False, 'labels_assigned': False,
            'source_root': str(source), 'files': records, 'schema_profiles': list(profiles.values()),
            'rows': sum(folders.values()), 'row_count_scope': 'structurally parseable records only',
            'malformed_records': sum(r['malformed_records']['count'] for r in records),
            'folder_rows': dict(sorted(folders.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-metadata', type=Path, default=Path('data/splits/feature_scaler.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    features = json.loads(args.feature_metadata.read_text())['feature_columns']
    result = audit(args.source, features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('rows', 'malformed_records', 'schema_profiles', 'folder_rows')}, indent=2))


if __name__ == '__main__':
    main()
