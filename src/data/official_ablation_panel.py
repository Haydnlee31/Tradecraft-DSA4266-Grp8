"""Freeze a label-blind shared validation panel for the three ablation arms.

All masks are nested, so matching after removing Number AND Tot sum contains
every match under either smaller mask. One exact 37-feature index therefore
implements the union rule. The source arrays and original splits never change.
Only row indices (O(N) small integers), not the full feature matrix, enter RAM.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data import official_shortcut_audit as prior


DROP = ['Number', 'Tot sum']
FLAGS = ('test_opened', 'model_trained', 'model_scored', 'split_changed', 'cloud_authorized')
FILES = ('retained_rows.npy', 'excluded_rows.npy')


def write_json(path, value):
    """Fixed UTF-8/LF bytes make independently generated receipts portable."""
    tmp = path.with_suffix('.json.tmp')
    tmp.write_bytes((json.dumps(value, indent=2)+'\n').encode('utf-8'))
    tmp.replace(path)


def validate_plan(plan):
    if (plan.get('protocol') != 'official39-ablation-panel-v1'
            or plan.get('drop_features') != DROP or plan.get('artifacts') != list(FILES)
            or plan.get('rows') != {'train': 2000000, 'val': 2059284}
            or plan.get('batch_size') != 8192 or plan.get('sqlite_cache_mib') != 128
            or any(plan.get(k) is not False for k in FLAGS)
            or plan.get('packed_manifest_sha256') != 'dfd1caf01cbe233b52ce55083941001f390ec251dd89b37543630a2bc53935be'):
        raise ValueError('Unreviewed panel plan')


def open_index(path, cache_mib):
    if path.exists():
        raise FileExistsError('Use a fresh panel database')
    db = sqlite3.connect(path)
    db.execute(f'PRAGMA cache_size=-{int(cache_mib)*1024}')
    db.execute('PRAGMA temp_store=FILE')
    cols = ','.join(f'{s}{i} INTEGER NOT NULL' for s in ('t', 'v') for i in range(len(CLASSES)))
    db.execute(f'CREATE TABLE vectors (k BLOB PRIMARY KEY, nt INTEGER NOT NULL, '
               f'nv INTEGER NOT NULL, mt INTEGER NOT NULL, mv INTEGER NOT NULL, {cols}) WITHOUT ROWID')
    db.execute('CREATE TEMP TABLE candidates (i INTEGER PRIMARY KEY, k BLOB NOT NULL)')
    return db


def add_rows(db, keys, y, split):
    if split not in ('train', 'val') or len(keys) != len(y):
        raise ValueError('Invalid index scope/shape')
    prior.check_labels(y)
    grouped = {}
    for key, label in zip(keys, y):
        grouped.setdefault(key, Counter())[int(label)] += 1
    columns = ['nt', 'nv', 'mt', 'mv'] + [f'{s}{i}' for s in ('t', 'v') for i in range(len(CLASSES))]
    records = []
    for key, counts in grouped.items():
        mask = sum(1 << label for label in counts)
        vector = [counts[i] for i in range(len(CLASSES))]
        if split == 'train':
            records.append([key, sum(vector), 0, mask, 0, *vector, *([0]*len(CLASSES))])
        else:
            records.append([key, 0, sum(vector), 0, mask, *([0]*len(CLASSES)), *vector])
    updates = ','.join(f'{c}={c}{"|" if c in ("mt", "mv") else "+"}excluded.{c}' for c in columns)
    db.executemany(f'INSERT INTO vectors (k,{",".join(columns)}) VALUES '
                   f'({",".join(["?"]*(len(columns)+1))}) ON CONFLICT(k) DO UPDATE SET {updates}', records)


def matching_rows(db, keys):
    """Membership reads only feature keys and whether training has that key."""
    db.execute('DELETE FROM candidates')
    db.executemany('INSERT INTO candidates VALUES (?,?)', enumerate(keys))
    return np.asarray([r[0] for r in db.execute(
        'SELECT i FROM candidates JOIN vectors USING(k) WHERE nt>0 ORDER BY i')], dtype=np.int64)


def summarize(db):
    cross = 'nt>0 AND nv>0'
    expr = dict(unique_vectors='count(*)', train_rows='sum(nt)', val_rows='sum(nv)',
                train_unique_vectors='sum(nt>0)', val_unique_vectors='sum(nv>0)',
                cross_split_groups=f'sum({cross})', cross_split_row_pairs='sum(nt*nv)',
                cross_split_train_rows=f'sum(CASE WHEN {cross} THEN nt ELSE 0 END)',
                cross_split_val_rows=f'sum(CASE WHEN {cross} THEN nv ELSE 0 END)',
                conflicting_label_groups='sum(((mt|mv)&((mt|mv)-1))!=0)',
                cross_split_conflicting_label_groups=f'sum(({cross}) AND (((mt|mv)&((mt|mv)-1))!=0))',
                cross_split_different_label_pairs='sum(nt*nv-('+ '+'.join(f't{i}*v{i}' for i in range(8))+'))')
    for i, name in enumerate(CLASSES):
        for s, split in [('t', 'train'), ('v', 'val')]:
            expr[f'{split}/{name}'] = f'sum(CASE WHEN {cross} THEN {s}{i} ELSE 0 END)'
    values = db.execute('SELECT '+','.join(f'coalesce({e},0)' for e in expr.values())+' FROM vectors').fetchone()
    result = dict(zip(expr, map(int, values)))
    result['cross_split_rows_by_class'] = {name: {s: result.pop(f'{s}/{name}') for s in ('train', 'val')} for name in CLASSES}
    for split in ('train', 'val'):
        result[f'{split}_duplicate_excess_rows'] = result[f'{split}_rows']-result[f'{split}_unique_vectors']
    return result


def build(arrays, features, output, batch_size=8192, cache_mib=128):
    """Low-level builder also supports tiny synthetic fixtures for portable CI."""
    if len(features) != 39 or len(set(features)) != 39 or any(name not in features for name in DROP) or batch_size < 1:
        raise ValueError('Expected the named 39-feature inputs')
    keep = [i for i, name in enumerate(features) if name not in DROP]
    excluded_mask = np.zeros(len(arrays['val'][1]), dtype=bool)
    db = open_index(output/'vectors.sqlite', cache_mib)
    try:
        for split in ('train', 'val'):
            x, y = arrays[split]
            for start in range(0, len(y), batch_size):
                xb, yb = x[start:start+batch_size], y[start:start+batch_size]
                if not np.isfinite(xb).all():
                    raise ValueError('Nonfinite source input, including masked columns')
                keys = prior.canonical(xb[:, keep])
                add_rows(db, keys, yb, split)
                if split == 'val':
                    # Labels contribute diagnostics but never the decision.
                    excluded_mask[start+matching_rows(db, keys)] = True
                db.commit()
            print(f'{split}: {len(y):,} joint-projection rows indexed', flush=True)
        stats = summarize(db)
    finally:
        db.close()
    rows = {'retained_rows.npy': np.flatnonzero(~excluded_mask).astype('<i8'),
            'excluded_rows.npy': np.flatnonzero(excluded_mask).astype('<i8')}
    for name, indices in rows.items():
        np.save(output/name, indices, allow_pickle=False)
    y = arrays['val'][1]
    retained_counts = np.bincount(y[rows['retained_rows.npy']], minlength=len(CLASSES))
    excluded_counts = np.bincount(y[rows['excluded_rows.npy']], minlength=len(CLASSES))
    if int(excluded_mask.sum()) != stats['cross_split_val_rows']:
        raise ValueError('Panel membership and index counts disagree')
    for i, name in enumerate(CLASSES):
        if int(excluded_counts[i]) != stats['cross_split_rows_by_class'][name]['val']:
            raise ValueError('Panel class counts disagree')
    return dict(joint_projection=stats, retained_rows=len(rows['retained_rows.npy']),
                excluded_rows=len(rows['excluded_rows.npy']),
                retained_class_counts=dict(zip(CLASSES, map(int, retained_counts))),
                excluded_class_counts=dict(zip(CLASSES, map(int, excluded_counts))),
                all_classes_retained=bool(np.all(retained_counts > 0)),
                files={name: sha256(output/name) for name in FILES})


def run(plan_path, packed, output):
    plan_path, packed, output = map(Path, (plan_path, packed, output))
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    for name, key in [('shortcut-ablation-plan.json', 'ablation_design_sha256'),
                      ('shortcut-audit-results.json', 'prior_audit_results_sha256')]:
        if sha256(plan_path.parent/name) != plan[key]:
            raise ValueError('Parent evidence changed')
    if sha256(packed/'manifest.json') != plan['packed_manifest_sha256']:
        raise ValueError('Packed cohort changed')
    m = json.loads((packed/'manifest.json').read_text())
    if (m['status'] != 'complete' or m['classes'] != CLASSES or m['test_opened'] is not False
            or sha256(packed/'scaler.json') != m['scaler_sha256']
            or any(m[f'{s}_rows'] != plan['rows'][s] for s in ('train', 'val'))):
        raise ValueError('Invalid packed manifest/scaler')
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(protocol=plan['protocol'], status='incomplete', plan_sha256=sha256(plan_path),
                   builder_sha256=sha256(__file__), canonical_reader_sha256=sha256(prior.__file__),
                   packed_manifest_sha256=sha256(packed/'manifest.json'),
                   input_files_sha256=m['files'], scaler_sha256=m['scaler_sha256'],
                   drop_features=DROP, classes=CLASSES, source_validation_rows=m['val_rows'],
                   primary_endpoint='final_at_budget_on_shared_panel', **dict.fromkeys(FLAGS, False))
    write_json(output/'receipt.json', receipt)
    with prior.packed_arrays(packed, m) as arrays:
        receipt.update(build(arrays, m['features'], output, plan['batch_size'], plan['sqlite_cache_mib']))
    previous = json.loads((plan_path.parent/'shortcut-audit-results.json').read_text())['collisions']['full39']
    receipt['additional_overlap_rows_vs_full39'] = {s: receipt['joint_projection'][f'cross_split_{s}_rows']-previous[f'cross_split_{s}_rows']
                                                   for s in ('train', 'val')}
    if min(receipt['additional_overlap_rows_vs_full39'].values()) < 0:
        raise ValueError('Coarser projection cannot reduce overlapping rows')
    receipt['status'] = 'prepared_not_scored'
    write_json(output/'receipt.json', receipt)
    return receipt


def load_panel(root, data):
    """Validate portable artifacts before any model training or output write.

    Receipt identity is included in the runner's environment. Changing even a
    well-formed panel cannot silently resume an old experiment. The reviewed
    production receipt hash is additionally checked by the cloud pilot gate.
    """
    root = Path(root)
    receipt_hash = sha256(root/'receipt.json')
    r = json.loads((root/'receipt.json').read_text())
    if (r['protocol'] != 'official39-ablation-panel-v1' or r['status'] != 'prepared_not_scored'
            or r['packed_manifest_sha256'] != sha256(data.root/'manifest.json')
            or r['drop_features'] != DROP or r['classes'] != CLASSES
            or not r['all_classes_retained'] or any(r[k] is not False for k in FLAGS)
            or set(r['files']) != set(FILES) or r['source_validation_rows'] != data.manifest['val_rows']):
        raise ValueError('Invalid panel scope or cohort')
    arrays = []
    for name in FILES:
        if sha256(root/name) != r['files'][name]:
            raise ValueError('Panel index checksum changed')
        a = np.load(root/name, allow_pickle=False)
        if a.ndim != 1 or a.dtype != np.int64 or np.any(a[1:] <= a[:-1]) or np.any((a < 0) | (a >= r['source_validation_rows'])):
            raise ValueError('Panel rows must be sorted unique valid int64 indices')
        arrays.append(a)
    retained, excluded = arrays
    # O(N) boolean bookkeeping is small (~2 MiB for this validation cohort).
    counts = np.zeros(r['source_validation_rows'], dtype=np.uint8)
    counts[retained] += 1
    counts[excluded] += 1
    if not np.all(counts == 1) or len(retained) != r['retained_rows'] or len(excluded) != r['excluded_rows']:
        raise ValueError('Panel and exclusions do not cover validation exactly once')
    for name, indices in [('retained', retained), ('excluded', excluded)]:
        observed = np.bincount(data.arrays['val'][1][indices], minlength=len(CLASSES))
        if dict(zip(CLASSES, map(int, observed))) != r[f'{name}_class_counts']:
            raise ValueError('Panel class counts changed')
    if sha256(root/'receipt.json') != receipt_hash:
        raise ValueError('Panel receipt changed while loading')
    mask = np.zeros(r['source_validation_rows'], dtype=bool)
    mask[retained] = True
    return mask, dict(receipt_sha256=receipt_hash, files=r['files'], retained_rows=len(retained),
                      excluded_rows=len(excluded), retained_class_counts=r['retained_class_counts'])


def publish(first, replay, plan_path, output):
    first, replay, plan_path, output = map(Path, (first, replay, plan_path, output))
    if first.resolve() == replay.resolve() or (first/'receipt.json').read_bytes() != (replay/'receipt.json').read_bytes():
        raise ValueError('Require separate identical panel replay receipts')
    if output.exists():
        raise FileExistsError('Do not overwrite published evidence')
    r = json.loads((first/'receipt.json').read_text())
    if (r['status'] != 'prepared_not_scored' or r['plan_sha256'] != sha256(plan_path)
            or r['builder_sha256'] != sha256(__file__) or any(r[k] is not False for k in FLAGS)):
        raise ValueError('Incomplete/changed panel receipt')
    for name in FILES:
        if sha256(first/name) != r['files'][name] or sha256(replay/name) != r['files'][name]:
            raise ValueError('Panel replay indices differ')
    r['source_receipt_sha256'] = sha256(first/'receipt.json')
    r['independent_replay_receipt_sha256'] = sha256(replay/'receipt.json')
    write_json(output, r)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build_parser = sub.add_parser('build')
    for name in ('plan', 'packed', 'output'):
        build_parser.add_argument('--'+name, type=Path, required=True)
    publish_parser = sub.add_parser('publish')
    for name in ('first', 'replay', 'plan', 'output'):
        publish_parser.add_argument('--'+name, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    args['plan_path'] = args.pop('plan')
    r = run(**args) if action == 'build' else publish(**args)
    print(json.dumps({k: r[k] for k in ('status', 'retained_rows', 'excluded_rows', 'retained_class_counts', 'all_classes_retained')}, indent=2))


if __name__ == '__main__':
    main()
