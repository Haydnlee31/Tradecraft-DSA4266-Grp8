"""Fixed local SupCon screen with a sampler/objective-matched CE control."""
import argparse
from dataclasses import replace
import io
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts import official39_numeric_retrieval as previous
from src.eval import contrastive_flow as core
from src.eval.official_evidence import digest, require
from src.models.architectures import MLPClassifier

prior = previous.prior
frozen, neural, rep = prior.frozen, prior.neural, prior.rep
PLAN = ROOT/'reports/risk_aware_routing/contrastive-plan.json'
REFERENCE = ROOT/'reports/risk_aware_routing/retrieval-results.json'
PREVIOUS_OUTPUT = ROOT/'outputs/official39-numeric-retrieval-v1'
SOURCES = ['src/eval/contrastive_flow.py', 'scripts/official39_contrastive.py',
           'scripts/official39_numeric_retrieval.py', 'scripts/official39_representation_pilot.py',
           'src/eval/flow_representation.py', 'src/eval/numerical_embedding.py',
           'src/eval/matched_neural.py', 'src/eval/score_fusion.py', 'src/models/architectures.py']
ARMS = ['balanced_ce_control', 'balanced_ce_supcon']
COMPARATORS = ['tree', 'ple']


def identity():
    p = frozen.read(PLAN)
    require(p['reference_sha256'] == digest(REFERENCE) and p['training'] == core.SETTINGS
            and p['arms'] == ARMS and p['fit_rows'] == 270326 and p['calibration_rows'] == 409918
            and not any(p[k] for k in ['test_opened', 'audit_scored', 'cloud_authorized', 'deployment_authorized'])
            and p['query_examples_per_fit'] == 16220160 and p['optimizer_steps_per_fit'] == 31680,
            'Unreviewed contrastive plan')
    return dict(plan=p, plan_sha256=digest(PLAN), reference_sha256=digest(REFERENCE),
                sources={n: digest(ROOT/n) for n in SOURCES})


def run(output):
    require(not output.exists(), 'Preserve existing output')
    ident = identity()
    reference = frozen.read(REFERENCE)
    previous_result = previous.verify(PREVIOUS_OUTPUT)
    require(reference['training_and_predictions_replay_exact']
            and all(reference[k] == v for k, v in previous_result.items()), 'Reference changed')
    historical = prior.verify_numerical(previous.NUMERICAL)
    start = time.perf_counter()
    output.mkdir()
    files, models, records = {}, {}, []
    with prior.data() as d:
        require(not np.intersect1d(d.fit, d.calibration).size, 'Fitting/calibration overlap')
        ids = rep.retained(d.features)
        raw = np.array(d.x[d.fit][:, ids], dtype=np.float32)
        y = np.array(d.y[d.fit], dtype=np.int64)
        edges = rep.fit_edges(raw)
        require(edges == frozen.read(previous.NUMERICAL/'edges.json'), 'PLE representation changed')
        x = rep.encode(raw, edges)
        _, groups = np.unique(raw, axis=0, return_inverse=True)
        labels = np.array(d.y[d.calibration], dtype=np.int64)
        for name, value in [('fit-rows.npy', d.fit), ('calibration-rows.npy', d.calibration),
                            ('fit-groups.npy', groups), ('calibration-labels.npy', labels)]:
            files[name] = previous.save_array(output, name, value)
        frozen.save(output/'edges.json', edges)
        files['edges.json'] = digest(output/'edges.json')
        recipe = dict(identity=ident, runtime=d.runtime, files=dict(files),
            fit_population_counts=d.pop.tolist(), fit_sample_counts=np.bincount(y, minlength=8).tolist(),
            calibration_supports=dict(zip(prior.CLASSES, np.bincount(labels, minlength=8).tolist())),
            ple_dimensions=x.shape[1], fit_rows_sha256=neural.vector_hash(d.fit),
            calibration_rows_sha256=neural.vector_hash(d.calibration),
            test_opened=False, audit_scored=False)
        frozen.save(output/'frozen-recipe.json', recipe)
        for arm, coefficient in zip(ARMS, [0., .1]):
            begin = time.perf_counter()
            model, record = core.fit(x, y, groups, d.pop, coefficient)
            require(record['examples_processed'] == 16220160 and record['optimizer_steps'] == 31680
                    and record['sampled_class_counts'] == [2027520]*8, 'Training budget changed')
            name = arm+'.pt'
            (output/name).write_bytes(neural.serialize(model))
            files[name] = digest(output/name)
            model.load_state_dict(torch.load(io.BytesIO((output/name).read_bytes()), weights_only=True,
                                            map_location='cpu'), strict=True)
            require(neural.state_hash(model) == record['final_state_sha256'], 'Checkpoint reload changed state')
            record.update(arm=arm, file=name, sha256=files[name], seconds=time.perf_counter()-begin)
            records.append(record)
            models[arm] = model
        for field in ['initial_state_sha256', 'batch_order_sha256', 'sampled_class_counts', 'class_weights']:
            require(records[0][field] == records[1][field], 'Unmatched contrastive control: '+field)
        frozen.save(output/'models-frozen.json', records)
        print('PASS: both fits frozen; identical initialization, sampled rows/order and CE weights', flush=True)
        cx = rep.encode(np.array(d.x[d.calibration][:, ids]), edges)
        tree = prior.tree_model(d, 7)
        predictions = {'tree': prior.ablation.predict_rows(tree, d.x, d.calibration, d.features, neural.ARM)}
        record = next(m for m in historical['models'] if m['arm'] == 'ple' and m['seed'] == 7)
        require(digest(previous.NUMERICAL/record['file']) == record['sha256'], 'Historical PLE changed')
        ple = MLPClassifier(x.shape[1], 8, replace(neural.CONFIG, hidden_dims=(64, 32)))
        ple.load_state_dict(torch.load(previous.NUMERICAL/record['file'], weights_only=True, map_location='cpu'))
        require(neural.state_hash(ple) == record['final_state_sha256'], 'Historical PLE state changed')
        predictions['ple'] = prior.numerical.predict(ple, cx)
        inference_seconds = {}
        for arm, model in models.items():
            begin = time.perf_counter()
            predictions[arm] = core.predict(model, cx)
            inference_seconds[arm] = time.perf_counter()-begin
        scores = {}
        for name, pred in predictions.items():
            files[name+'-calibration.npy'] = previous.save_array(output, name+'-calibration.npy', pred)
            scores[name] = prior.old.prior.prior.scored(labels, pred, d.pop)
        require(all(scores[n] == reference['scores'][n] for n in COMPARATORS), 'Historical calibration bridge changed')
        checks = {name: rep.gate(scores[ARMS[1]]['metrics'], scores[name]['metrics'])
                  for name in ['tree', ARMS[0]]}
        result = dict(status='complete', recipe=recipe, files=files, models=records, scores=scores,
            continuation_checks=checks, continue_to_confirmation=all(checks.values()),
            examples_processed=sum(m['examples_processed'] for m in records),
            optimizer_steps=sum(m['optimizer_steps'] for m in records), inference_seconds=inference_seconds,
            test_opened=False, test_evaluated=False, audit_scored=False,
            independent_confirmation=False, deployment_authorized=False)
    require(identity() == ident, 'Implementation changed during experiment')
    result['elapsed_seconds'] = time.perf_counter()-start
    frozen.save(output/'receipt.json', result)
    print({name: round(score['metrics']['macro_f1'], 6) for name, score in scores.items()}, flush=True)
    print({'continue_to_confirmation': result['continue_to_confirmation']}, flush=True)


def verify(output):
    r = frozen.read(output/'receipt.json')
    require(r['status'] == 'complete' and r['recipe']['identity'] == identity()
            and not any(r[k] for k in ['test_opened', 'test_evaluated', 'audit_scored',
                                      'independent_confirmation', 'deployment_authorized'])
            and r['examples_processed'] == 32440320 and r['optimizer_steps'] == 63360,
            'Invalid contrastive receipt')
    require(frozen.read(output/'frozen-recipe.json') == r['recipe']
            and frozen.read(output/'models-frozen.json') == r['models'], 'Frozen artifacts changed')
    expected_files = {'fit-rows.npy', 'calibration-rows.npy', 'fit-groups.npy', 'calibration-labels.npy', 'edges.json'}
    expected_files.update(arm+'.pt' for arm in ARMS)
    expected_files.update(name+'-calibration.npy' for name in ARMS+COMPARATORS)
    require(set(r['files']) == expected_files and set(r['scores']) == set(ARMS+COMPARATORS)
            and [m['arm'] for m in r['models']] == ARMS, 'Missing artifacts or arms')
    for name, sha in r['files'].items():
        require(digest(output/name) == sha, 'Artifact changed')
    require(all(r['files'][n] == sha for n, sha in r['recipe']['files'].items()), 'Recipe artifacts changed')
    y = np.load(output/'calibration-labels.npy', allow_pickle=False)
    require(y.shape == (409918,) and np.issubdtype(y.dtype, np.integer) and np.isin(y, np.arange(8)).all(),
            'Invalid calibration labels')
    supports = dict(zip(prior.CLASSES, np.bincount(y, minlength=8).tolist()))
    require(supports == r['recipe']['calibration_supports'], 'Support classes changed')
    for name, score in r['scores'].items():
        p = np.load(output/(name+'-calibration.npy'), allow_pickle=False)
        require(p.shape == y.shape and p.dtype == np.uint8 and (p < 8).all(), 'Invalid predictions')
        require(score == prior.old.prior.prior.scored(y, p, r['recipe']['fit_population_counts']), 'Scores changed')
    for model in r['models']:
        require(model['settings'] == core.SETTINGS and model['sha256'] == r['files'][model['file']]
                and model['examples_processed'] == 16220160 and model['optimizer_steps'] == 31680
                and model['sampled_class_counts'] == [2027520]*8
                and model['eligible_contrastive_anchors']+model['excluded_contrastive_anchors'] == 16220160,
                'Model/budget changed')
    require([m['coefficient'] for m in r['models']] == [0., .1], 'Wrong loss arms')
    for key in ['initial_state_sha256', 'batch_order_sha256', 'class_weights', 'sampled_class_counts']:
        require(r['models'][0][key] == r['models'][1][key], 'Unmatched control')
    checks = {n: rep.gate(r['scores'][ARMS[1]]['metrics'], r['scores'][n]['metrics']) for n in ['tree', ARMS[0]]}
    require(r['continuation_checks'] == checks and r['continue_to_confirmation'] == all(checks.values()),
            'Continuation decision changed')
    return r


def publish(first, replay, destination):
    require(first.resolve() != replay.resolve() and not destination.exists(), 'Fresh independent replay required')
    a, b = verify(first), verify(replay)
    require(prior.without_timing(a) == prior.without_timing(b), 'Contrastive replay mismatch')
    frozen.save(destination, dict(a, non_timing_replay_exact=True,
        first_sha256=digest(first/'receipt.json'), replay_sha256=digest(replay/'receipt.json')))
    print('PASS: both fits, sampling and four prediction arrays replay exactly; audit/test closed', flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--output', required=True, type=Path)
    p = sub.add_parser('publish')
    for name in ['first', 'replay', 'destination']:
        p.add_argument('--'+name, required=True, type=Path)
    args = vars(ap.parse_args())
    {'run': run, 'publish': publish}[args.pop('command')](**args)


if __name__ == '__main__':
    main()
