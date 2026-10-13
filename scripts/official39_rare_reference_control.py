"""Post-hoc count-sensitivity control; not a classifier or new training run.

The first diagnostic used every FIT row, so nearest-distance comparisons
confound geometry with reference abundance. This extension fixes reference
counts at 760 unique vectors per class, with three prespecified draws. Their
spread is sensitivity to reference selection, not a confidence interval.
"""
import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scripts import official39_rare_error_diagnostic as previous
from src.data.label_map import CLASSES
from src.eval.official_evidence import digest, require

prior, frozen, core = previous.prior, previous.frozen, previous.core
PLAN = ROOT/'reports/risk_aware_routing/rare-reference-plan.json'
REFERENCE = ROOT/'reports/risk_aware_routing/rare-error-results.json'
PREVIOUS_OUTPUT = ROOT/'outputs/official39-rare-error-v1'


def unique_candidates(x, y):
    """Unique numeric vectors WITHIN each class, retaining conflicting groups.

    Global first-label deduplication would silently change class membership.
    Return original FIT positions, sorted before seeded sampling.
    """
    core.labels(y, len(x))
    require(x.ndim == 2 and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid vectors')
    result = []
    for k in range(len(CLASSES)):
        positions = np.flatnonzero(y == k)
        _, first = np.unique(x[positions], axis=0, return_index=True)
        result.append(np.sort(positions[first]))
    return result


def balanced_indices(candidates, per_class, seed):
    require(type(per_class) is int and per_class > 0 and len(candidates) == len(CLASSES)
            and all(a.ndim == 1 and np.issubdtype(a.dtype, np.integer)
                    and len(a) >= per_class and (a >= 0).all()
                    and len(np.unique(a)) == len(a) for a in candidates), 'Invalid candidate groups')
    flat = np.concatenate(candidates)
    require(len(np.unique(flat)) == len(flat), 'Overlapping candidate positions')
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(a, per_class, replace=False) for a in candidates]))


def identity():
    plan = frozen.read(PLAN)
    require(plan['reference_sha256'] == digest(REFERENCE) and plan['seeds'] == [7, 17, 27]
            and plan['unique_vectors_per_class'] == 760 and plan['model_trained'] is False
            and plan['test_opened'] is False and plan['audit_scored'] is False, 'Control plan changed')
    return dict(plan=plan, plan_sha256=digest(PLAN), source_sha256=digest(Path(__file__)),
                upstream_identity=previous.identity())


def run(output):
    require(not output.exists(), 'Preserve existing control')
    ident = identity()
    initial = previous.verify(PREVIOUS_OUTPUT)
    require(all(frozen.read(REFERENCE)[k] == v for k, v in initial.items()), 'Reference changed')
    receipts = previous.inputs()
    started = time.perf_counter()
    output.mkdir()
    records, files = [], {}
    with prior.data() as d:
        predictions = previous.aligned_predictions(d, receipts, initial['errors']['models'])
        manifest = frozen.read(prior.PACKED/'manifest.json')
        keep, _ = prior.headroom.load_panel(prior.PANEL,
            SimpleNamespace(root=prior.PACKED, arrays={'val': (d.x, d.y)}, manifest=manifest))
        fit_rows = np.flatnonzero(keep)[np.load(prior.CACHE/'partitions.npy', allow_pickle=False) == 0]
        require(prior.neural.vector_hash(fit_rows) == initial['fitting_rows_sha256'], 'FIT population changed')
        ids = prior.rep.retained(d.features)
        x, y = np.array(d.x[fit_rows][:, ids], dtype=np.float32), np.array(d.y[fit_rows])
        candidates = unique_candidates(x, y)
        require([len(a) for a in candidates] == [initial['exact_fitting_overlap']['classes'][c]['unique_vectors']
                                              for c in CLASSES], 'Unique candidate counts changed')
        calibration_y = np.array(d.y[d.calibration])
        rare = np.isin(calibration_y, [CLASSES.index(n) for n in core.RARE])
        rare_rows = d.calibration[rare]
        require(prior.neural.vector_hash(rare_rows) == initial['rare_rows_sha256'], 'Queries changed')
        query = np.array(d.x[rare_rows][:, ids], dtype=np.float32)
        edges = frozen.read(previous.INPUTS['contrastive']/'edges.json')
        for seed in ident['plan']['seeds']:
            chosen = balanced_indices(candidates, 760, seed)
            counts = np.bincount(y[chosen], minlength=8).tolist()
            require(counts == [760]*8, 'Unbalanced reference counts')
            filename = f'seed{seed}-rows.npy'
            np.save(output/filename, fit_rows[chosen], allow_pickle=False)
            files[filename] = digest(output/filename)
            geometry = {}
            for name in initial['identity']['plan']['geometries']:
                q, r = (query, x[chosen]) if name == 'packed_standardized_euclidean' else (
                    previous.coordinates(query, edges), previous.coordinates(x[chosen], edges))
                values = core.nearest_by_class(q, r, y[chosen])
                geometry[name] = core.proximity(values, calibration_y[rare],
                                                {n: p[rare] for n, p in predictions.items()})
                filename = f'seed{seed}-{name}.npy'
                np.save(output/filename, values, allow_pickle=False)
                files[filename] = digest(output/filename)
            records.append(dict(seed=seed, reference_counts=counts, proximity=geometry))
            print(f'PASS: balanced-reference draw {seed}, both geometries', flush=True)
    require(identity() == ident and previous.verify(PREVIOUS_OUTPUT) == initial, 'Source evidence changed')
    frozen.save(output/'receipt.json', dict(status='complete', identity=ident, files=files, draws=records,
        candidate_counts=[len(a) for a in candidates], rare_rows_sha256=initial['rare_rows_sha256'],
        model_trained=False, test_opened=False, test_evaluated=False, audit_scored=False, cloud_used=False,
        independent_confirmation=False, deployment_authorized=False, elapsed_seconds=time.perf_counter()-started))


def publish(first, replay, destination):
    require(first.resolve() != replay.resolve() and not destination.exists(), 'Fresh independent replay required')
    runs = []
    for path in [first, replay]:
        r = frozen.read(path/'receipt.json')
        require(r['identity'] == identity() and r['status'] == 'complete'
                and len(r['draws']) == 3 and len(r['files']) == 9, 'Control changed')
        require(not any(r[k] for k in ['model_trained', 'test_opened', 'test_evaluated', 'audit_scored',
                                      'cloud_used', 'independent_confirmation', 'deployment_authorized']), 'Scope changed')
        for name, sha in r['files'].items():
            require(Path(name).name == name and digest(path/name) == sha, 'Control artifact changed')
        runs.append(r)
    require(prior.without_timing(runs[0]) == prior.without_timing(runs[1]), 'Control replay mismatch')
    frozen.save(destination, dict(runs[0], non_timing_replay_exact=True,
        first_sha256=digest(first/'receipt.json'), replay_sha256=digest(replay/'receipt.json')))
    print('PASS: balanced-reference selection, distances and counts replay exactly', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('publish')
    for n in ['first', 'replay', 'destination']:
        p.add_argument('--'+n, type=Path, required=True)
    args = vars(ap.parse_args())
    {'run': run, 'publish': publish}[args.pop('command')](**args)
