"""Read verified frozen predictions and FIT features; never train or open test."""
import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scripts import official39_contrastive as previous
from src.data.label_map import CLASSES
from src.eval import rare_error_diagnostic as core
from src.eval.numeric_retrieval import coordinates
from src.eval.official_evidence import digest, require

prior, frozen = previous.prior, previous.frozen
PLAN = ROOT/'reports/risk_aware_routing/rare-error-plan.json'
REFERENCES = {name: ROOT/f'reports/risk_aware_routing/{name}-results.json'
              for name in ['contrastive', 'retrieval']}
INPUTS = {name: ROOT/f'outputs/official39-{stem}-v1' for name, stem in
          [('contrastive', 'contrastive'), ('retrieval', 'numeric-retrieval')]}
SOURCES = ['src/eval/rare_error_diagnostic.py', 'scripts/official39_rare_error_diagnostic.py',
           'src/eval/numeric_retrieval.py', 'src/data/label_map.py']


def identity():
    plan = frozen.read(PLAN)
    require(all(plan[n+'_reference_sha256'] == digest(p) for n, p in REFERENCES.items()),
            'Reference changed')
    require(not any(plan[k] for k in ['model_trained', 'test_opened', 'audit_scored',
                                     'cloud_authorized', 'deployment_authorized']), 'Scope changed')
    return dict(plan=plan, plan_sha256=digest(PLAN), sources={n: digest(ROOT/n) for n in SOURCES})


def inputs():
    receipts = dict(contrastive=previous.verify(INPUTS['contrastive']),
                    retrieval=previous.previous.verify(INPUTS['retrieval']))
    for name, receipt in receipts.items():
        public = frozen.read(REFERENCES[name])
        require(all(public[k] == v for k, v in receipt.items()), 'Published receipt mismatch')
    return receipts


def aligned_predictions(d, receipts, names):
    """Match actual row order and labels, not only supports or saved scores."""
    result = {}
    for name, path in INPUTS.items():
        for field, actual in [('fit', d.fit), ('calibration', d.calibration)]:
            require(np.array_equal(np.load(path/(field+'-rows.npy'), allow_pickle=False), actual),
                    'Row alignment changed')
    require(np.array_equal(np.load(INPUTS['contrastive']/'calibration-labels.npy', allow_pickle=False),
                           d.y[d.calibration]), 'Labels changed')
    for name in names:
        origin = 'contrastive' if name in ['tree', 'ple']+previous.ARMS else 'retrieval'
        pred = np.load(INPUTS[origin]/(name+'-calibration.npy'), allow_pickle=False)
        observed = prior.old.prior.prior.scored(d.y[d.calibration], pred, d.pop)
        require(observed == receipts[origin]['scores'][name], 'Saved scores/labels changed')
        result[name] = pred
    for name in ['tree', 'ple']:
        require(np.array_equal(result[name], np.load(INPUTS['retrieval']/(name+'-calibration.npy'),
                                                    allow_pickle=False)), 'Shared comparator changed')
    return result


def run(output):
    require(not output.exists(), 'Preserve existing diagnostic')
    ident, receipts = identity(), inputs()
    start = time.perf_counter()
    output.mkdir()
    files = {}
    with prior.data() as d:
        predictions = aligned_predictions(d, receipts, ident['plan']['models'])
        y = np.array(d.y[d.calibration])
        errors = core.overlap(y, predictions)
        print('PASS: six frozen predictors aligned; shared errors counted', flush=True)
        # Recover the complete FIT population from the same audited group split.
        # This includes unselected majority-class FIT examples, but no audit rows.
        manifest = frozen.read(prior.PACKED/'manifest.json')
        keep, _ = prior.headroom.load_panel(prior.PANEL,
            SimpleNamespace(root=prior.PACKED, arrays={'val': (d.x, d.y)}, manifest=manifest))
        panel_rows = np.flatnonzero(keep)
        split = np.load(prior.CACHE/'partitions.npy', allow_pickle=False)
        fit_rows = panel_rows[split == 0]
        require(len(fit_rows) == ident['plan']['full_fitting_rows']
                and np.array_equal(np.bincount(d.y[fit_rows], minlength=8), d.pop)
                and not np.intersect1d(fit_rows, d.calibration).size, 'Invalid fitting population')
        ids = prior.rep.retained(d.features)
        fit_x = np.array(d.x[fit_rows][:, ids], dtype=np.float32)
        fit_y = np.array(d.y[fit_rows])
        conflicts = core.exact_conflicts(fit_x, fit_y)
        print('PASS: complete FIT exact-vector ambiguity counted', flush=True)
        rare_mask = np.isin(y, [CLASSES.index(n) for n in core.RARE])
        rare_rows, rare_y = d.calibration[rare_mask], y[rare_mask]
        require(len(rare_rows) == 772, 'Rare cohort changed')
        query = np.array(d.x[rare_rows][:, ids], dtype=np.float32)
        rare_predictions = {n: p[rare_mask] for n, p in predictions.items()}
        edges = frozen.read(INPUTS['contrastive']/'edges.json')
        geometry = {}
        for name in ident['plan']['geometries']:
            q, r = (query, fit_x) if name == 'packed_standardized_euclidean' else (
                coordinates(query, edges), coordinates(fit_x, edges))
            values = core.nearest_by_class(q, r, fit_y)
            geometry[name] = core.proximity(values, rare_y, rare_predictions)
            np.save(output/(name+'.npy'), values, allow_pickle=False)
            files[name+'.npy'] = digest(output/(name+'.npy'))
            print('PASS: nearest-class distances: '+name, flush=True)
        # Store only small private distance matrices and aggregate public counts.
        result = dict(status='complete', identity=ident,
            source_receipts={n: digest(p/'receipt.json') for n, p in INPUTS.items()},
            runtime=d.runtime, files=files, fitting_rows_sha256=prior.neural.vector_hash(fit_rows),
            rare_rows_sha256=prior.neural.vector_hash(rare_rows),
            calibration_rows_sha256=prior.neural.vector_hash(d.calibration),
            retained_features=[d.features[j] for j in ids], full_fitting_counts=d.pop.tolist(),
            calibration_counts=np.bincount(y, minlength=8).tolist(),
            scores={n: receipts['contrastive' if n in ['tree', 'ple']+previous.ARMS else 'retrieval']['scores'][n]
                    for n in predictions}, errors=errors, exact_fitting_overlap=conflicts, proximity=geometry,
            model_trained=False, test_opened=False, test_evaluated=False, audit_scored=False,
            cloud_used=False, independent_confirmation=False, deployment_authorized=False)
    require(identity() == ident and inputs() == receipts, 'Inputs or implementation changed during diagnostic')
    result['elapsed_seconds'] = time.perf_counter()-start
    frozen.save(output/'receipt.json', result)
    print('PASS: diagnostic complete; no training, audit scoring, test or cloud', flush=True)


def verify(output):
    r = frozen.read(output/'receipt.json')
    require(r['status'] == 'complete' and r['identity'] == identity(), 'Invalid diagnostic identity')
    require(not any(r[k] for k in ['model_trained', 'test_opened', 'test_evaluated', 'audit_scored',
                                  'cloud_used', 'independent_confirmation', 'deployment_authorized']), 'Invalid scope')
    require(set(r['files']) == {n+'.npy' for n in r['identity']['plan']['geometries']}, 'Wrong distance artifacts')
    receipts = inputs()
    label = np.load(INPUTS['contrastive']/'calibration-labels.npy', allow_pickle=False)
    predictions = {}
    for name in r['identity']['plan']['models']:
        origin = 'contrastive' if name in ['tree', 'ple']+previous.ARMS else 'retrieval'
        predictions[name] = np.load(INPUTS[origin]/(name+'-calibration.npy'), allow_pickle=False)
        score = prior.old.prior.prior.scored(label, predictions[name], r['full_fitting_counts'])
        require(score == r['scores'][name] == receipts[origin]['scores'][name], 'Diagnostic score changed')
    require(r['errors'] == core.overlap(label, predictions), 'Error counts changed')
    require(r['calibration_counts'] == np.bincount(label, minlength=8).tolist()
            and r['full_fitting_counts'] == receipts['contrastive']['recipe']['fit_population_counts'],
            'Population counts changed')
    rare = np.isin(label, [CLASSES.index(n) for n in core.RARE])
    for name, checksum in r['files'].items():
        require(digest(output/name) == checksum, 'Distance artifact changed')
        values = np.load(output/name, allow_pickle=False)
        require(values.shape == (772, 8) and values.dtype == np.float64
                and np.isfinite(values).all() and (values >= 0).all(), 'Invalid stored distances')
        require(r['proximity'][Path(name).stem] == core.proximity(values, label[rare],
                {n: p[rare] for n, p in predictions.items()}), 'Proximity aggregate changed')
    require(r['source_receipts'] == {n: digest(p/'receipt.json') for n, p in INPUTS.items()},
            'Upstream evidence changed')
    return r


def publish(first, replay, destination):
    require(first.resolve() != replay.resolve() and not destination.exists(), 'Fresh independent replay required')
    a, b = verify(first), verify(replay)
    require(prior.without_timing(a) == prior.without_timing(b), 'Diagnostic replay changed')
    frozen.save(destination, dict(a, non_timing_replay_exact=True,
        first_sha256=digest(first/'receipt.json'), replay_sha256=digest(replay/'receipt.json')))
    print('PASS: distances and aggregate diagnostic replay exactly', flush=True)


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
