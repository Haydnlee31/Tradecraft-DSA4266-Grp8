"""Bounded, read-only proxy and exact model-input collision preflight.

The raw release was grouped at float64 precision. Training uses scaled float32,
so we audit that representation separately. SQLite keys contain complete row
bytes, not hashes: removing Number cannot hide behind a hash collision. Input
arrays remain read-only and only train/validation filenames are ever opened.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_shards import save_json


PROJECTIONS = {'full39': [], 'without_number': ['Number']}
FLAGS = ('test_opened', 'model_trained', 'split_changed', 'cloud_authorized')
SETTINGS = dict(batch_size=8192, sqlite_cache_mib=128,
                number_bin_tolerance=1e-4, proxy_absolute_tolerance=1e-3,
                proxy_relative_tolerance=1e-5, denominator_floor=1e-4,
                window_group_cutoff=55.)


def validate_plan(plan):
    if (plan.get('protocol') != 'official39-shortcut-preflight-v1'
            or plan.get('projections') != PROJECTIONS
            or plan.get('settings') != SETTINGS
            or plan.get('rows') != {'train': 2000000, 'val': 2059284}
            or any(plan.get(k) is not False for k in FLAGS)):
        raise ValueError('Unreviewed shortcut audit scope/settings')
    digest = plan.get('packed_manifest_sha256', '')
    if digest != 'dfd1caf01cbe233b52ce55083941001f390ec251dd89b37543630a2bc53935be':
        raise ValueError('Unreviewed packed cohort')


@contextmanager
def packed_arrays(root, manifest):
    """Validate every named input before mmap, and close handles on failure too."""
    names = [f'{split}_{key}.npy' for split in ('train', 'val') for key in ('x', 'y')]
    for name in names:
        if sha256(root/name) != manifest['files'][name]:
            raise ValueError(f'Packed checksum changed: {name}')
    opened, arrays = [], {}
    try:
        for split in ('train', 'val'):
            for key in ('x', 'y'):
                a = np.load(root/f'{split}_{key}.npy', mmap_mode='r', allow_pickle=False)
                opened.append(a)
            x, y = opened[-2:]
            if (x.shape != (manifest[f'{split}_rows'], 39) or y.shape != (len(x),)
                    or x.dtype != np.float32 or y.dtype != np.int64):
                raise ValueError('Packed shape/dtype changed')
            arrays[split] = (x, y)
        yield arrays
    finally:
        for a in opened:
            a._mmap.close()


def canonical(values):
    array = np.array(values, dtype='<f4', order='C', copy=True)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite feature input')
    array[array == 0] = 0  # +0 and -0 have the same numerical meaning.
    return [row.tobytes() for row in array]


def check_labels(y):
    if y.dtype != np.int64 or np.any((y < 0) | (y >= len(CLASSES))):
        raise ValueError('Invalid class labels')


def proxy_scan(x, y, scaler, features, settings):
    """A prespecified arithmetic diagnostic, not a learned attack classifier."""
    j = features.index
    selected = [j(name) for name in ('Number', 'Tot sum', 'AVG', 'Tot size')]
    means, scales = np.asarray(scaler['mean'])[selected], np.asarray(scaler['scale'])[selected]
    result = {}
    for denominator in ('AVG', 'Tot size'):
        result[denominator] = {name: dict(rows=0, valid_denominator=0, within_tolerance=0,
                                        abs_error_sum=0., max_abs_error=0.,
                                        near_10_or_100=0, window_group_agreement=0)
                               for name in CLASSES}
    for start in range(0, len(y), settings['batch_size']):
        labels = y[start:start+settings['batch_size']]
        check_labels(labels)
        xb = x[start:start+len(labels)]
        if not np.isfinite(xb).all():
            raise ValueError('Nonfinite feature input')
        raw = np.asarray(xb[:, selected], dtype=np.float64) * scales + means
        number, total = raw[:, 0], raw[:, 1]
        for index, denominator in ((2, 'AVG'), (3, 'Tot size')):
            valid = raw[:, index] > settings['denominator_floor']
            ratio = np.divide(total, raw[:, index], out=np.zeros(len(labels)), where=valid)
            error = np.abs(ratio-number)
            passing = valid & (error <= settings['proxy_absolute_tolerance']
                               + settings['proxy_relative_tolerance'] * np.abs(number))
            known = ((np.abs(number-10) <= settings['number_bin_tolerance'])
                     | (np.abs(number-100) <= settings['number_bin_tolerance']))
            same_group = valid & known & ((ratio >= settings['window_group_cutoff'])
                                         == (number >= settings['window_group_cutoff']))
            for label, name in enumerate(CLASSES):
                take = labels == label
                r = result[denominator][name]
                for key, mask in [('rows', take), ('valid_denominator', take & valid),
                                  ('within_tolerance', take & passing), ('near_10_or_100', take & known),
                                  ('window_group_agreement', take & same_group)]:
                    r[key] += int(mask.sum())
                errors = error[take & valid]
                r['abs_error_sum'] += float(errors.sum())
                r['max_abs_error'] = max(r['max_abs_error'], float(errors.max(initial=0)))
    for classes in result.values():
        for r in classes.values():
            r['mean_abs_error_valid'] = (r.pop('abs_error_sum') / r['valid_denominator']
                                         if r['valid_denominator'] else None)
    return result


def open_index(path, cache_mib):
    if path.exists():
        raise FileExistsError('Use a fresh collision database')
    db = sqlite3.connect(path)
    db.execute(f'PRAGMA cache_size=-{cache_mib*1024}')
    db.execute('PRAGMA temp_store=FILE')
    counts = ','.join(f'{s}{i} INTEGER NOT NULL' for s in ('t', 'v') for i in range(len(CLASSES)))
    for table in PROJECTIONS:
        db.execute(f'CREATE TABLE {table} (k BLOB PRIMARY KEY, nt INTEGER NOT NULL, '
                   f'nv INTEGER NOT NULL, mt INTEGER NOT NULL, mv INTEGER NOT NULL, '
                   f'{counts}, baseline_shared INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID')
    return db


def add_rows(db, table, keys, y, split):
    if table not in PROJECTIONS or split not in ('train', 'val'):
        raise ValueError('Unsupported collision scope')
    check_labels(y)
    if len(keys) != len(y):
        raise ValueError('Key/label count mismatch')
    # Aggregate a batch's repeated keys before touching the disk index.
    grouped = {}
    for key, label in zip(keys, y):
        counts = grouped.setdefault(key, [0]*len(CLASSES))
        counts[int(label)] += 1
    columns = ['nt', 'nv', 'mt', 'mv'] + [f'{s}{i}' for s in ('t', 'v') for i in range(len(CLASSES))]
    rows = []
    for key, counts in grouped.items():
        n, mask = sum(counts), sum(1 << i for i, count in enumerate(counts) if count)
        if split == 'train':
            rows.append([key, n, 0, mask, 0, *counts, *([0]*len(CLASSES))])
        else:
            rows.append([key, 0, n, 0, mask, *([0]*len(CLASSES)), *counts])
    updates = ','.join(f'{c}={c}{"|" if c in ("mt", "mv") else "+"}excluded.{c}' for c in columns)
    db.executemany(f'INSERT INTO {table} (k,{",".join(columns)}) VALUES '
                   f'({",".join(["?"]*(len(columns)+1))}) ON CONFLICT(k) DO UPDATE SET {updates}', rows)


def summarize(db, table):
    if table not in PROJECTIONS:
        raise ValueError('Unsupported collision table')
    # Build one aggregate query instead of rereading the large index for each
    # counter. No feature matrix or global key set is materialized in memory.
    expressions = []
    def single(expression):
        expressions.append(expression)
        return len(expressions)-1
    cross = 'nt>0 AND nv>0'
    r = dict(unique_vectors=single('1'), train_rows=single('nt'), val_rows=single('nv'),
             train_unique_vectors=single('nt>0'), val_unique_vectors=single('nv>0'))
    r['cross_split_groups'] = single(f'({cross})')
    r['cross_split_train_rows'] = single(f'CASE WHEN {cross} THEN nt ELSE 0 END')
    r['cross_split_val_rows'] = single(f'CASE WHEN {cross} THEN nv ELSE 0 END')
    r['cross_split_row_pairs'] = single('nt*nv')
    same_pairs = '+'.join(f't{i}*v{i}' for i in range(len(CLASSES)))
    r['cross_split_different_label_pairs'] = single(f'nt*nv-({same_pairs})')
    r['conflicting_label_groups'] = single('((mt|mv)&((mt|mv)-1))!=0')
    r['cross_split_conflicting_label_groups'] = single(f'({cross}) AND (((mt|mv)&((mt|mv)-1))!=0)')
    for split, mask in [('train', 'mt'), ('val', 'mv')]:
        r[f'{split}_conflicting_label_groups'] = single(f'({mask}&({mask}-1))!=0')
    r['cross_split_rows_by_class'] = {
        name: {split: single(f'CASE WHEN {cross} THEN {s}{i} ELSE 0 END')
               for s, split in [('t', 'train'), ('v', 'val')]}
        for i, name in enumerate(CLASSES)}
    values = db.execute('SELECT ' + ','.join(f'coalesce(sum({e}),0)' for e in expressions)
                        + f' FROM {table}').fetchone()
    def resolve(item):
        return {k: resolve(v) for k, v in item.items()} if isinstance(item, dict) else int(values[item])
    r = resolve(r)
    r['train_duplicate_excess_rows'] = r['train_rows']-r['train_unique_vectors']
    r['val_duplicate_excess_rows'] = r['val_rows']-r['val_unique_vectors']
    return r


def collision_scan(arrays, features, database, settings):
    keep = [i for i, name in enumerate(features) if name != 'Number']
    db = open_index(database, settings['sqlite_cache_mib'])
    try:
        for split in ('train', 'val'):
            x, y = arrays[split]
            for start in range(0, len(y), settings['batch_size']):
                xb, yb = x[start:start+settings['batch_size']], y[start:start+settings['batch_size']]
                add_rows(db, 'full39', canonical(xb), yb, split)
                add_rows(db, 'without_number', canonical(xb[:, keep]), yb, split)
                db.commit()
            print(f'{split}: {len(y):,} rows indexed for both representations', flush=True)
        # Mark projected groups containing a baseline cross-split vector. This
        # distinguishes wholly new groups from expansions of existing overlap.
        cursor = db.execute('SELECT k FROM full39 WHERE nt>0 AND nv>0')
        while records := cursor.fetchmany(settings['batch_size']):
            keys = [(np.frombuffer(k, dtype='<f4')[keep].tobytes(),) for (k,) in records]
            db.executemany('UPDATE without_number SET baseline_shared=1 WHERE k=?', keys)
        db.commit()
        results = {table: summarize(db, table) for table in PROJECTIONS}
        results['new_overlap'] = {
            'train_rows': results['without_number']['cross_split_train_rows']-results['full39']['cross_split_train_rows'],
            'val_rows': results['without_number']['cross_split_val_rows']-results['full39']['cross_split_val_rows'],
            'projected_groups_without_any_baseline_shared_vector': int(db.execute(
                'SELECT count(*) FROM without_number WHERE nt>0 AND nv>0 AND baseline_shared=0').fetchone()[0])}
        return results
    finally:
        db.close()


def run(plan_path, packed, output):
    plan_path, packed, output = map(Path, (plan_path, packed, output))
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if sha256(plan_path.parent/'explanation-sensitivity-results.json') != plan['prior_sensitivity_results_sha256']:
        raise ValueError('Parent sensitivity evidence changed')
    if sha256(packed/'manifest.json') != plan['packed_manifest_sha256']:
        raise ValueError('Packed manifest checksum changed')
    m = json.loads((packed/'manifest.json').read_text())
    if (m['status'] != 'complete' or m['classes'] != CLASSES or m['test_opened'] is not False
            or len(m['features']) != 39 or len(set(m['features'])) != 39
            or any(m[f'{s}_rows'] != plan['rows'][s] for s in ('train', 'val'))
            or sha256(packed/'scaler.json') != m['scaler_sha256']):
        raise ValueError('Unreviewed packed data/scaler')
    scaler = json.loads((packed/'scaler.json').read_text())
    if (scaler['features'] != m['features'] or scaler['fit_split'] != 'train'
            or scaler['n_samples_seen'] != m['train_rows']):
        raise ValueError('Scaler does not match training cohort')
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='incomplete', plan_sha256=sha256(plan_path),
                   auditor_sha256=sha256(__file__), packed_manifest_sha256=sha256(packed/'manifest.json'),
                   input_files_sha256=m['files'], scaler_sha256=m['scaler_sha256'],
                   **dict.fromkeys(FLAGS, False))
    save_json(output/'receipt.json', receipt)
    with packed_arrays(packed, m) as arrays:
        receipt['proxy'] = {s: proxy_scan(*arrays[s], scaler, m['features'], plan['settings'])
                            for s in ('train', 'val')}
        receipt['collisions'] = collision_scan(arrays, m['features'], output/'vectors.sqlite', plan['settings'])
    receipt['status'] = 'complete'
    save_json(output/'receipt.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--packed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    r = run(args.plan, args.packed, args.output)
    print(json.dumps({'status': r['status'], 'collisions': r['collisions']}, indent=2))


if __name__ == '__main__':
    main()
