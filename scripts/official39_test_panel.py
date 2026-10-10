"""Explicitly gated test packing and label-blind, disk-backed overlap screening.

Importing or testing this module does not open official data. Its low-level
indexer accepts synthetic arrays; the production entry point checks explicit
approval before constructing any test reader. It never loads model weights.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts import official39_final_evaluation as evaluation
from src.data.label_map import CLASSES
from src.data.official_loader import OfficialBatches
from src.data.official_shortcut_audit import canonical, packed_arrays
from src.eval.official_evidence import digest, require
from src.models.official_ablation import ARMS

BATCH_SIZE = 8192


def class_counts(y):
    require(y.ndim == 1 and y.dtype == np.int64 and np.all((y >= 0) & (y < len(CLASSES))), 'Invalid class IDs')
    return dict(zip(CLASSES, map(int, np.bincount(y, minlength=len(CLASSES)))))


def screening(references, x, y, features, output, batch_size=BATCH_SIZE):
    """Keep membership independent of labels; count labels only for reporting.

    Each SQLite key is the full canonical projected vector, not a lossy hash.
    Flags distinguish training (1) and validation (2) membership. Test labels
    never enter the index or its inclusion queries. O(N) boolean/index vectors
    are retained in RAM, but features and exact-key indices remain disk-backed.
    """
    require(set(references) == {'train', 'val'} and len(features) == len(set(features)) == 39
            and all(name in features for name in ('Number', 'Tot sum')) and batch_size > 0, 'Invalid screening schema')
    require(x.shape == (len(y), 39) and x.dtype == np.float32, 'Invalid held-out input array')
    class_counts(y)
    output = Path(output)
    excluded, results = np.zeros(len(y), bool), {}
    for arm, removed in ARMS.items():
        columns = [i for i, name in enumerate(features) if name not in removed]
        database = output / f'{arm}.sqlite'
        require(not database.exists(), 'Preserve existing overlap index')
        db = sqlite3.connect(database)
        try:
            db.execute('PRAGMA cache_size=-131072')
            db.execute('PRAGMA temp_store=FILE')
            db.execute('CREATE TABLE vectors (k BLOB PRIMARY KEY, flags INTEGER NOT NULL, ntest INTEGER NOT NULL) WITHOUT ROWID')
            db.execute('CREATE TEMP TABLE query_rows (i INTEGER PRIMARY KEY, k BLOB NOT NULL)')
            for split, flag in [('train', 1), ('val', 2)]:
                source = references[split]
                require(source.ndim == 2 and source.shape[1] == 39 and source.dtype == np.float32, 'Invalid reference input array')
                for start in range(0, len(source), batch_size):
                    raw = source[start:start + batch_size]
                    require(np.isfinite(raw).all(), 'Nonfinite reference input, including masked columns')
                    keys = canonical(raw[:, columns])
                    db.executemany('INSERT INTO vectors VALUES (?,?,0) ON CONFLICT(k) DO UPDATE SET flags=flags|excluded.flags',
                                   ((k, flag) for k in keys))
                    db.commit()
            matched = {name: np.zeros(len(y), bool) for name in ('train', 'val', 'either')}
            for start in range(0, len(y), batch_size):
                raw = x[start:start + batch_size]
                require(np.isfinite(raw).all(), 'Nonfinite held-out input, including masked columns')
                keys = canonical(raw[:, columns])
                db.execute('DELETE FROM query_rows')
                db.executemany('INSERT INTO query_rows VALUES (?,?)', enumerate(keys))
                for i, flag in db.execute('SELECT i,flags FROM query_rows JOIN vectors USING(k) WHERE flags>0 ORDER BY i'):
                    matched['train'][start+i] = bool(flag & 1)
                    matched['val'][start+i] = bool(flag & 2)
                    matched['either'][start+i] = True
                db.executemany('INSERT INTO vectors VALUES (?,0,1) ON CONFLICT(k) DO UPDATE SET ntest=ntest+1', ((k,) for k in keys))
                db.commit()
            unique, duplicated_groups, excess = db.execute(
                'SELECT count(*),coalesce(sum(ntest>1),0),coalesce(sum(ntest-1),0) FROM vectors WHERE ntest>0').fetchone()
            results[arm] = dict(
                matched_rows={name: int(mask.sum()) for name, mask in matched.items()},
                matched_class_counts={name: class_counts(y[mask]) for name, mask in matched.items()},
                held_out_unique_projected_vectors=int(unique), duplicate_groups=int(duplicated_groups),
                duplicate_excess_rows=int(excess))
            excluded |= matched['either']
        finally:
            db.close()
    retained = ~excluded
    np.save(output / 'retained.npy', retained, allow_pickle=False)
    counts = class_counts(y[retained])
    require(int(excluded.sum()) == results['number_total_masked']['matched_rows']['either'],
            'Nested-mask union disagrees with joint projection')
    return dict(retained_rows=int(retained.sum()), excluded_rows=int(excluded.sum()),
                retained_class_counts=counts, excluded_class_counts=class_counts(y[excluded]),
                all_classes_retained=all(n > 0 for n in counts.values()),
                projections=results, label_blind_membership=True)


def scaler_from_record(record, features):
    require(record['features'] == features and record['fit_split'] == 'train'
            and record['n_samples_seen'] == 2000000, 'Scaler must be fitted on the fixed 2M training subset')
    scaler = StandardScaler()
    scaler.mean_, scaler.var_, scaler.scale_ = (np.asarray(record[k], dtype=np.float64) for k in ('mean', 'var', 'scale'))
    require(all(v.shape == (39,) and np.isfinite(v).all() for v in (scaler.mean_, scaler.var_, scaler.scale_))
            and np.all(scaler.scale_ > 0), 'Invalid saved scaler values')
    scaler.n_features_in_, scaler.n_samples_seen_ = 39, 2000000
    return scaler


def prepare(shards, packed, output, allow_test_preparation=False):
    # This guard precedes every production data read, including manifests.
    require(allow_test_preparation is True, 'Explicit test-preparation approval required; test stays closed')
    plan = evaluation.load_plan()
    manifest = evaluation.verify_pack(packed, plan)
    root, packed, output = Path(shards), Path(packed), Path(output)
    identity = plan['data_identity']
    require(digest(root / 'manifest.json') == identity['parent_shards_manifest_sha256']
            and digest(root / 'protocol.json') == identity['split_protocol_sha256'], 'Held-out source identity changed')
    loader = OfficialBatches(root, 'test', batch_size=BATCH_SIZE, allow_test=True)
    require(loader.rows == identity['test_rows_before_overlap_filter']
            and loader.features == identity['features'], 'Held-out population changed')
    scaler = scaler_from_record(evaluation.read(packed / 'scaler.json'), loader.features)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(protocol='official39-test-panel-v1', status='incomplete', plan_sha256=digest(evaluation.PLAN),
                   source_sha256=evaluation.source_identity(), explicit_test_preparation_opt_in=True,
                   test_opened=True, test_evaluated=False, model_trained=False,
                   parent_manifest_sha256=identity['parent_shards_manifest_sha256'],
                   scaler_sha256=identity['scaler_sha256'], rows=loader.rows, features=loader.features, classes=CLASSES)
    evaluation.save(output / 'receipt.json', receipt)
    x = np.lib.format.open_memmap(output / 'x.npy', mode='w+', dtype=np.float32, shape=(loader.rows, 39))
    y = np.lib.format.open_memmap(output / 'y.npy', mode='w+', dtype=np.int64, shape=(loader.rows,))
    try:
        offset = 0
        for xb, yb in loader.batches(scaler):
            require(offset + len(yb) <= len(y), 'Source iterator exceeds declared rows')
            x[offset:offset+len(yb)], y[offset:offset+len(yb)] = xb, yb
            offset += len(yb)
        require(offset == len(y), 'Source iterator did not cover held-out rows')
        x.flush()
        y.flush()
        receipt['class_counts'] = class_counts(y)
        require(receipt['class_counts'] == identity['test_class_counts_before_overlap_filter'], 'Held-out class counts changed')
        # Independent source replay checks that packing did not change precision,
        # row order or labels. This is data QA, not model evaluation.
        offset = 0
        for xb, yb in loader.batches(scaler):
            require(np.array_equal(x[offset:offset+len(yb)], xb)
                    and np.array_equal(y[offset:offset+len(yb)], yb), 'Packed data differs from source replay')
            offset += len(yb)
        require(offset == len(y), 'Replay row coverage changed')
    finally:
        x._mmap.close()
        y._mmap.close()
    x, y = (np.load(output / name, mmap_mode='r', allow_pickle=False) for name in ('x.npy', 'y.npy'))
    try:
        with packed_arrays(packed, manifest) as reference:
            receipt['panel'] = screening({split: reference[split][0] for split in ('train', 'val')},
                                         x, y, loader.features, output)
    finally:
        x._mmap.close()
        y._mmap.close()
    receipt['files'] = {name: digest(output / name) for name in ('x.npy', 'y.npy', 'retained.npy')}
    receipt['status'] = 'prepared_not_scored' if receipt['panel']['all_classes_retained'] else 'blocked_empty_class'
    evaluation.save(output / 'receipt.json', receipt)
    return receipt


@contextmanager
def prepared_arrays(root, plan):
    """Only the separately approved scorer calls this production data loader."""
    root, opened = Path(root), []
    r = evaluation.read(root / 'receipt.json')
    require(r['protocol'] == 'official39-test-panel-v1' and r['status'] == 'prepared_not_scored'
            and r['plan_sha256'] == digest(evaluation.PLAN) and r['source_sha256'] == evaluation.source_identity(),
            'Prepared test identity changed or gate incomplete')
    require(r['explicit_test_preparation_opt_in'] is True and r['test_opened'] is True
            and r['test_evaluated'] is False and r['model_trained'] is False, 'Wrong preparation scope')
    identity = plan['data_identity']
    require(r['parent_manifest_sha256'] == identity['parent_shards_manifest_sha256']
            and r['scaler_sha256'] == identity['scaler_sha256'] and r['features'] == identity['features']
            and r['classes'] == CLASSES and r['rows'] == identity['test_rows_before_overlap_filter']
            and r['class_counts'] == identity['test_class_counts_before_overlap_filter'], 'Prepared population changed')
    require(set(r['files']) == {'x.npy', 'y.npy', 'retained.npy'} and r['panel']['all_classes_retained'] is True
            and r['panel']['label_blind_membership'] is True, 'Invalid panel receipt')
    try:
        for name in ('x.npy', 'y.npy', 'retained.npy'):
            require(digest(root / name) == r['files'][name], 'Prepared input checksum changed')
            opened.append(np.load(root / name, mmap_mode='r', allow_pickle=False))
        x, y, retained = opened
        require(x.shape == (r['rows'], 39) and x.dtype == np.float32 and y.shape == (r['rows'],)
                and y.dtype == np.int64 and retained.shape == y.shape and retained.dtype == bool, 'Prepared array schema changed')
        require(class_counts(y) == r['class_counts'] and class_counts(y[retained]) == r['panel']['retained_class_counts']
                and class_counts(y[~retained]) == r['panel']['excluded_class_counts'], 'Prepared class supports changed')
        require(int(retained.sum()) == r['panel']['retained_rows']
                and int((~retained).sum()) == r['panel']['excluded_rows'], 'Prepared membership counts changed')
        require(all(v > 0 for v in r['panel']['retained_class_counts'].values()), 'A shared-panel class is empty')
        yield x, y, retained, r
    finally:
        for array in opened:
            array._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards', type=Path, required=True)
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-test-preparation', action='store_true')
    args = parser.parse_args()
    r = prepare(args.shards, args.packed, args.output, args.allow_test_preparation)
    print(f'{r["status"]}: retained={r["panel"]["retained_rows"]}; test_evaluated=False')


if __name__ == '__main__':
    main()
