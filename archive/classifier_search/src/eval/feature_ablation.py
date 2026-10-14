"""Feature removal with one label-blind, collision-safe comparison population.

Constant standardized columns remove information without changing input slots.
All masks are nested: an exact match on the 37 remaining columns includes every
possible match on 38 or 39. This gives one conservative panel for all arms.
The database stores exact feature bytes, not probabilistic hash matches.
"""
from pathlib import Path
import sqlite3

import numpy as np

from src.data.official_shortcut_audit import canonical
from src.eval.official_evidence import require

ARMS = {'full39': [], 'number_masked': ['Number'],
        'number_total_masked': ['Number', 'Tot sum']}


def columns(features, arm):
    require(arm in ARMS and len(features) == len(set(features)) == 39
            and all(c in features for c in ARMS['number_total_masked']), 'Wrong feature schema/mask')
    return [features.index(c) for c in ARMS[arm]]


def project(x, features, arm):
    """Copy before masking: never modify source arrays or the other arms.

    Zero is the old training mean in standardized space, not a raw count of
    zero. The existing scaler stays fixed. Even unused inputs must be finite.
    """
    drop = columns(features, arm)
    require(x.ndim == 2 and x.shape[1] == 39 and len(x) > 0
            and np.isfinite(x).all(), 'Invalid projection input')
    result = np.array(x, dtype=np.float64, order='C', copy=True)
    result[:, drop] = 0.
    return result


def rows_ok(rows, size):
    require(rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer) and len(rows) > 0
            and len(np.unique(rows)) == len(rows)
            and np.all((rows >= 0) & (rows < size)), 'Invalid or repeated row indices')


def collision_panel(x, features, population, sampled, audit, database, batch=8192):
    """Exclude matches to the ENTIRE fit partition, before looking at labels.

    Some fit-partition rows supply only class-prior counts, not training
    features. Excluding their matches too is deliberately conservative. The
    separate sampled-match count makes that distinction visible. No row is
    reassigned, no fitting row is removed, and calibration is not consulted.
    """
    database = Path(database)
    require(not database.exists(), 'Preserve existing collision index')
    require(type(batch) is int and batch > 0 and x.ndim == 2 and x.shape[1] == 39,
            'Invalid collision scan')
    drop = columns(features, 'number_total_masked')
    keep = [i for i in range(39) if i not in drop]
    for rows in (population, sampled, audit):
        rows_ok(rows, len(x))
    require(np.isin(sampled, population).all() and not np.isin(audit, population).any(),
            'Fit/audit row identities overlap or sample escaped fit')
    selected = np.zeros(len(x), dtype=np.uint8)
    selected[sampled] = 1
    matched = np.zeros(len(audit), dtype=bool)
    sampled_matches = np.zeros(len(audit), dtype=bool)
    db = sqlite3.connect(database)
    try:
        db.execute('PRAGMA cache_size=-131072')  # 128 MiB, disk-backed keys.
        db.execute('PRAGMA temp_store=FILE')
        db.execute('CREATE TABLE vectors (k BLOB PRIMARY KEY, sampled INTEGER NOT NULL) WITHOUT ROWID')
        db.execute('CREATE TEMP TABLE candidates (i INTEGER PRIMARY KEY, k BLOB NOT NULL)')
        for start in range(0, len(population), batch):
            rows = population[start:start+batch]
            xb = np.array(x[rows])
            require(np.isfinite(xb).all(), 'Nonfinite collision input')
            keys = canonical(xb[:, keep])
            db.executemany('INSERT INTO vectors VALUES (?,?) ON CONFLICT(k) DO UPDATE '
                           'SET sampled=max(sampled,excluded.sampled)',
                           ((k, int(s)) for k, s in zip(keys, selected[rows])))
            db.commit()
        unique = int(db.execute('SELECT count(*) FROM vectors').fetchone()[0])
        print(f'PANEL: indexed {len(population):,} fit rows using exact 37-feature keys', flush=True)
        for start in range(0, len(audit), batch):
            rows = audit[start:start+batch]
            xb = np.array(x[rows])
            require(np.isfinite(xb).all(), 'Nonfinite collision input')
            keys = canonical(xb[:, keep])
            db.execute('DELETE FROM candidates')
            db.executemany('INSERT INTO candidates VALUES (?,?)', enumerate(keys))
            for i, s in db.execute('SELECT i,sampled FROM candidates JOIN vectors USING(k) ORDER BY i'):
                matched[start+i] = True
                sampled_matches[start+i] = bool(s)
    finally:
        db.close()
    retained, excluded = audit[~matched], audit[matched]
    return retained, excluded, dict(fit_population_rows=len(population), fit_sample_rows=len(sampled),
        original_audit_rows=len(audit), retained_rows=len(retained), excluded_rows=len(excluded),
        matches_to_sampled_fit_rows=int(sampled_matches.sum()),
        matches_only_to_unsampled_fit_rows=int((matched & ~sampled_matches).sum()),
        joint_projection_unique_fit_keys=unique, drop_features=ARMS['number_total_masked'],
        labels_used_for_membership=False, split_reassigned=False)


def masked_splits(model, features, arm):
    """Accounting guard only; prediction still uses sklearn's public API."""
    drop = columns(features, arm)
    return sum(int(np.count_nonzero((tree.nodes['is_leaf'] == 0)
        & np.isin(tree.nodes['feature_idx'], drop)))
        for stage in model._predictors for tree in stage)
