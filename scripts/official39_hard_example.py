"""Bounded local hard-example comparison, with frozen provenance and bridges."""
import argparse
import io
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts import official39_contrastive as previous
from src.eval import hard_example_flow as core
from src.eval.official_evidence import digest, require

prior, frozen, neural, rep = previous.prior, previous.frozen, previous.neural, previous.rep
PLAN = ROOT/'reports/risk_aware_routing/hard-example-plan.json'
REFERENCE = ROOT/'reports/risk_aware_routing/contrastive-results.json'
DIAGNOSTIC = ROOT/'reports/risk_aware_routing/rare-reference-results.json'
PREVIOUS = ROOT/'outputs/official39-contrastive-v1'
ARMS, COMPARATORS = core.ARMS, ['tree', 'ple']
SOURCES = ['src/eval/hard_example_flow.py', 'scripts/official39_hard_example.py',
           'src/eval/contrastive_flow.py', 'scripts/official39_contrastive.py',
           'src/eval/matched_neural.py', 'src/eval/flow_representation.py',
           'src/models/architectures.py', 'src/eval/numerical_embedding.py']


def identity():
    p = frozen.read(PLAN)
    require(p['historical_reference_sha256'] == digest(REFERENCE)
            and p['diagnostic_reference_sha256'] == digest(DIAGNOSTIC)
            and p['training'] == core.SETTINGS and p['arms'] == ARMS
            and not any(p[k] for k in ['test_opened', 'audit_scored', 'cloud_authorized', 'deployment_authorized']),
            'Unreviewed hard-example plan')
    return dict(plan=p, plan_sha256=digest(PLAN), sources={n: digest(ROOT/n) for n in SOURCES},
                historical_identity=previous.identity())


def run(output):
    require(not output.exists(), 'Preserve existing run')
    ident, historical = identity(), previous.verify(PREVIOUS)
    public = frozen.read(REFERENCE)
    require(all(public[k] == v for k,v in historical.items()) and public['non_timing_replay_exact'],
            'Historical reference changed')
    start = time.perf_counter()
    output.mkdir()
    files, records, models = {}, [], {}
    with prior.data() as d:
        for field, rows in [('fit', d.fit), ('calibration', d.calibration)]:
            require(np.array_equal(rows, np.load(PREVIOUS/(field+'-rows.npy'), allow_pickle=False)),
                    'Cohort changed')
        ids = rep.retained(d.features)
        raw = np.array(d.x[d.fit][:, ids], dtype=np.float32)
        y = np.array(d.y[d.fit], dtype=np.int64)
        edges = frozen.read(PREVIOUS/'edges.json')
        require(rep.fit_edges(raw) == edges, 'FIT representation changed')
        x = rep.encode(raw, edges)
        _, groups = np.unique(raw, axis=0, return_inverse=True)
        require(np.array_equal(groups, np.load(PREVIOUS/'fit-groups.npy', allow_pickle=False)),
                'Duplicate grouping changed')
        for name, value in [('fit-rows.npy', d.fit), ('calibration-rows.npy', d.calibration)]:
            files[name] = previous.previous.save_array(output, name, value)
        frozen.save(output/'edges.json', edges)
        files['edges.json'] = digest(output/'edges.json')
        recipe = dict(identity=ident, runtime=d.runtime, files=dict(files),
            fit_population_counts=d.pop.tolist(), fit_sample_counts=np.bincount(y, minlength=8).tolist(),
            ple_dimensions=x.shape[1], fit_rows_sha256=neural.vector_hash(d.fit),
            calibration_rows_sha256=neural.vector_hash(d.calibration),
            historical_receipt_sha256=digest(PREVIOUS/'receipt.json'),
            test_opened=False, audit_scored=False)
        frozen.save(output/'frozen-recipe.json', recipe)
        for arm in ARMS:
            begin = time.perf_counter()
            model, record, _ = core.fit(x, y, groups, d.pop, arm)
            require(record['examples_processed'] == 16220160 and record['optimizer_steps'] == 31680
                    and record['mining_examples_scored'] == 16219560
                    and record['sampled_class_counts'] == [2027520]*8, 'Work budget changed')
            filename = arm+'.pt'
            (output/filename).write_bytes(neural.serialize(model))
            files[filename] = digest(output/filename)
            model.load_state_dict(torch.load(io.BytesIO((output/filename).read_bytes()),
                                            map_location='cpu', weights_only=True), strict=True)
            require(neural.state_hash(model) == record['final_state_sha256'], 'Reload changed weights')
            record.update(file=filename, sha256=files[filename], seconds=time.perf_counter()-begin)
            records.append(record)
            models[arm] = model
            if arm == ARMS[0]:
                old_control = historical['models'][0]
                require(all(record[k] == old_control[k] for k in
                        ['initial_state_sha256', 'final_state_sha256', 'class_weights', 'parameters',
                         'deployment_parameters', 'sampled_class_counts', 'examples_processed', 'optimizer_steps']),
                        'Historical control weight/work bridge failed')
                print('PASS: ordinary control matches historical CE weights exactly', flush=True)
        frozen.save(output/'models-frozen.json', records)
        print('PASS: both models frozen before calibration scoring', flush=True)
        labels = np.array(d.y[d.calibration], dtype=np.int64)
        require(np.array_equal(labels, np.load(PREVIOUS/'calibration-labels.npy', allow_pickle=False)),
                'Historical calibration labels changed')
        files['calibration-labels.npy'] = previous.previous.save_array(output, 'calibration-labels.npy', labels)
        cx = rep.encode(np.array(d.x[d.calibration][:, ids]), edges)
        predictions = {n: np.load(PREVIOUS/(n+'-calibration.npy'), allow_pickle=False) for n in COMPARATORS}
        for arm, model in models.items():
            predictions[arm] = previous.core.predict(model, cx)
        require(np.array_equal(predictions[ARMS[0]],
                np.load(PREVIOUS/'balanced_ce_control-calibration.npy', allow_pickle=False)),
                'Historical control prediction bridge failed')
        scores = {}
        for name, pred in predictions.items():
            files[name+'-calibration.npy'] = previous.previous.save_array(output, name+'-calibration.npy', pred)
            scores[name] = prior.old.prior.prior.scored(labels, pred, d.pop)
        require(all(scores[n] == historical['scores'][n] for n in COMPARATORS)
                and scores[ARMS[0]] == historical['scores']['balanced_ce_control'], 'Score bridge changed')
        checks = {n: rep.gate(scores[ARMS[1]]['metrics'], scores[n]['metrics']) for n in ['tree', ARMS[0]]}
        result = dict(status='complete', recipe=recipe, files=files, models=records, scores=scores,
            control_historical_bridge_exact=True, continuation_checks=checks,
            continue_to_confirmation=all(checks.values()), examples_processed=32440320,
            optimizer_steps=63360, mining_examples_scored=32439120,
            test_opened=False, test_evaluated=False, audit_scored=False,
            cloud_used=False, independent_confirmation=False, deployment_authorized=False)
    require(identity() == ident and previous.verify(PREVIOUS) == historical, 'Source/evidence changed during fit')
    result['elapsed_seconds'] = time.perf_counter()-start
    frozen.save(output/'receipt.json', result)
    print({n: round(s['metrics']['macro_f1'],6) for n,s in scores.items()}, flush=True)
    print({'continue_to_confirmation': result['continue_to_confirmation']}, flush=True)


def verify(output):
    r = frozen.read(output/'receipt.json')
    require(r['status'] == 'complete' and r['recipe']['identity'] == identity()
            and r['control_historical_bridge_exact'] is True
            and not any(r[k] for k in ['test_opened', 'test_evaluated', 'audit_scored', 'cloud_used',
                                      'independent_confirmation', 'deployment_authorized'])
            and r['examples_processed'] == 32440320 and r['optimizer_steps'] == 63360
            and r['mining_examples_scored'] == 32439120, 'Invalid run receipt')
    require(frozen.read(output/'frozen-recipe.json') == r['recipe']
            and frozen.read(output/'models-frozen.json') == r['models'], 'Frozen artifacts changed')
    expected = {'fit-rows.npy', 'calibration-rows.npy', 'edges.json', 'calibration-labels.npy'}
    expected.update(a+'.pt' for a in ARMS)
    expected.update(a+'-calibration.npy' for a in ARMS+COMPARATORS)
    require(set(r['files']) == expected and set(r['scores']) == set(ARMS+COMPARATORS)
            and [m['arm'] for m in r['models']] == ARMS, 'Missing arms/artifacts')
    for name, sha in r['files'].items():
        require(digest(output/name) == sha, 'Artifact changed')
    require(all(r['files'][n] == sha for n,sha in r['recipe']['files'].items()), 'Recipe artifacts changed')
    y = np.load(output/'calibration-labels.npy', allow_pickle=False)
    require(y.shape == (409918,) and np.issubdtype(y.dtype, np.integer)
            and np.isin(y, np.arange(8)).all(), 'Invalid calibration labels')
    for name, score in r['scores'].items():
        pred = np.load(output/(name+'-calibration.npy'), allow_pickle=False)
        require(pred.shape == y.shape and pred.dtype == np.uint8 and (pred < 8).all()
                and score == prior.old.prior.prior.scored(y,pred,r['recipe']['fit_population_counts']),
                'Invalid predictions/scores')
    for model, forced in zip(r['models'], [0, 7434240]):
        require(model['settings'] == core.SETTINGS and model['completed_epochs'] == 60
                and model['sha256'] == r['files'][model['file']] and model['examples_processed'] == 16220160
                and model['optimizer_steps'] == 31680 and model['mining_examples_scored'] == 16219560
                and model['sampled_class_counts'] == [2027520]*8 and model['forced_hard_positions'] == forced
                and [m['epoch'] for m in model['mining']] == list(range(1,61)), 'Model/work receipt changed')
    for key in ['initial_state_sha256', 'class_weights', 'parameters', 'deployment_parameters', 'sampled_class_counts']:
        require(r['models'][0][key] == r['models'][1][key], 'Unmatched control')
    require(r['models'][0]['observations'][:2] == r['models'][1]['observations'][:2]
            and r['models'][0]['mining'][:6] == r['models'][1]['mining'][:6], 'Warmup changed')
    historical = frozen.read(REFERENCE)
    require(r['models'][0]['final_state_sha256'] == historical['models'][0]['final_state_sha256']
            and r['scores'][ARMS[0]] == historical['scores']['balanced_ce_control']
            and all(r['scores'][n] == historical['scores'][n] for n in COMPARATORS), 'Historical bridge changed')
    checks = {n: rep.gate(r['scores'][ARMS[1]]['metrics'],r['scores'][n]['metrics']) for n in ['tree',ARMS[0]]}
    require(r['continuation_checks'] == checks and r['continue_to_confirmation'] == all(checks.values()),
            'Continuation decision changed')
    return r


def publish(first, replay, destination):
    require(first.resolve() != replay.resolve() and not destination.exists(), 'Fresh independent replay required')
    a,b = verify(first),verify(replay)
    require(prior.without_timing(a) == prior.without_timing(b), 'Hard-example replay mismatch')
    frozen.save(destination, dict(a, non_timing_replay_exact=True,
        first_sha256=digest(first/'receipt.json'), replay_sha256=digest(replay/'receipt.json')))
    print('PASS: both models, mining losses/pools, sampling and predictions replay exactly', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--output',type=Path,required=True)
    p = sub.add_parser('publish')
    for n in ['first','replay','destination']:
        p.add_argument('--'+n,type=Path,required=True)
    args = vars(ap.parse_args())
    {'run':run,'publish':publish}[args.pop('command')](**args)
