"""One predeclared local retrieval screen on reused development calibration.

No test interface, cloud interface, automatic sweep, or deployment action.
The original byte-pinned experiment modules remain unchanged.
"""
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
from scripts import official39_representation_pilot as prior
from src.eval import numeric_retrieval as retrieval
from src.eval.official_evidence import digest, require
from src.models.architectures import MLPClassifier

frozen, neural, rep = prior.frozen, prior.neural, prior.rep
PLAN = ROOT/'reports/risk_aware_routing/retrieval-plan.json'
REFERENCE = ROOT/'reports/risk_aware_routing/numerical-embedding-results.json'
NUMERICAL = ROOT/'outputs/official39-numerical-embedding-v1'
SOURCES = ['src/eval/numeric_retrieval.py', 'scripts/official39_numeric_retrieval.py',
           'scripts/official39_representation_pilot.py', 'src/eval/flow_representation.py',
           'src/eval/numerical_embedding.py', 'src/eval/matched_neural.py']


def identity():
    p = frozen.read(PLAN)
    require(p['reference_sha256'] == digest(REFERENCE) and not p['test_opened']
            and not p['audit_scored'] and not p['cloud_authorized'] and p['seeds'] == [7]
            and p['fit_rows'] == 270326 and p['calibration_rows'] == 409918
            and p['bank']['class_cap'] == 2048 and p['bank']['sample_seed'] == 20261013
            and p['retrieval']['neighbours'] == 8
            and p['training']['epochs'] == 60 and p['training']['batch_size'] == 512,
            'Unreviewed retrieval plan')
    return dict(plan=p, plan_sha256=digest(PLAN), reference_sha256=digest(REFERENCE),
                sources={n: digest(ROOT/n) for n in SOURCES})


def save_array(root, name, array):
    np.save(root/name, array, allow_pickle=False)
    return digest(root/name)


def run(output):
    require(not output.exists(), 'Preserve existing run')
    ident = identity()
    reference = frozen.read(REFERENCE)
    historical = prior.verify_numerical(NUMERICAL)
    require(reference['non_timing_replay_exact'] and all(reference[k] == v for k, v in historical.items()),
            'Numerical reference changed')
    start = time.perf_counter()
    output.mkdir()
    files, models, records = {}, {}, []
    with prior.data() as d:
        ids = rep.retained(d.features)
        fit_raw = np.array(d.x[d.fit][:, ids], dtype=np.float32)
        fit_y = np.array(d.y[d.fit], dtype=np.int64)
        edges = rep.fit_edges(fit_raw)
        require(edges == frozen.read(NUMERICAL/'edges.json'), 'PLE basis changed')
        xx = rep.encode(fit_raw, edges)
        selected, groups = retrieval.bank(fit_raw, fit_y)
        rr, ry = xx[selected], fit_y[selected]
        coords = retrieval.coordinates(fit_raw, edges)
        print(f'BANK: {len(selected):,} unique fitting references; searching training contexts', flush=True)
        begin = time.perf_counter()
        train_context = retrieval.neighbours(coords, coords[selected], query_groups=groups,
                                             reference_groups=groups[selected])
        search_fit_seconds = time.perf_counter()-begin
        require(not np.any(groups[:, None] == groups[selected][train_context]), 'Duplicate leak')
        for name, value in [('bank-fit-indices.npy', selected), ('bank-row-ids.npy', d.fit[selected]),
                            ('fit-context.npy', train_context), ('fit-rows.npy', d.fit),
                            ('calibration-rows.npy', d.calibration)]:
            files[name] = save_array(output, name, value)
        frozen.save(output/'edges.json', edges)
        files['edges.json'] = digest(output/'edges.json')
        recipe = dict(identity=ident, runtime=d.runtime, fit_population_counts=d.pop.tolist(),
            fit_sample_counts=np.bincount(fit_y, minlength=8).tolist(),
            bank_class_counts=np.bincount(ry, minlength=8).tolist(), bank_rows=len(selected),
            ple_dimensions=xx.shape[1], self_or_duplicate_contexts=0,
            fit_rows_sha256=neural.vector_hash(d.fit), calibration_rows_sha256=neural.vector_hash(d.calibration),
            files=dict(files), test_opened=False, audit_scored=False)
        frozen.save(output/'frozen-recipe.json', recipe)
        for arm in ident['plan']['training']['arms']:
            begin = time.perf_counter()
            model, record = retrieval.fit(xx, fit_y, d.pop, rr, ry, train_context, arm == 'retrieval')
            require(record['examples_processed'] == 16219560 and record['optimizer_steps'] == 31680,
                    'Fit budget changed')
            name = arm+'.pt'
            (output/name).write_bytes(neural.serialize(model))
            files[name] = digest(output/name)
            model.load_state_dict(torch.load(io.BytesIO((output/name).read_bytes()), weights_only=True,
                                            map_location='cpu'), strict=True)
            require(neural.state_hash(model) == record['final_state_sha256'], 'Checkpoint reload mismatch')
            record.update(arm=arm, file=name, sha256=files[name], seconds=time.perf_counter()-begin)
            records.append(record)
            models[arm] = model
        frozen.save(output/'models-frozen.json', records)
        print('PASS: both models frozen before calibration scoring; audit/test closed', flush=True)
        raw = np.array(d.x[d.calibration][:, ids], dtype=np.float32)
        calibration_x = rep.encode(raw, edges)
        begin = time.perf_counter()
        cc = retrieval.neighbours(retrieval.coordinates(raw, edges), coords[selected])
        search_calibration_seconds = time.perf_counter()-begin
        files['calibration-context.npy'] = save_array(output, 'calibration-context.npy', cc)
        tree = prior.tree_model(d, 7)
        before = time.perf_counter()
        predictions = {'tree': prior.ablation.predict_rows(tree, d.x, d.calibration, d.features, neural.ARM)}
        timings = {'tree': time.perf_counter()-before}
        before = time.perf_counter()
        predictions['knn'] = retrieval.vote(cc, ry)
        timings['knn'] = time.perf_counter()-before
        for arm, model in models.items():
            before = time.perf_counter()
            predictions[arm] = retrieval.predict(model, calibration_x, rr, ry, cc)
            timings[arm] = time.perf_counter()-before
        # Historical PLE is evaluated on the SAME calibration rows, not compared
        # against its earlier audit score. Only its owned tensor checkpoint loads.
        record = next(m for m in historical['models'] if m['arm'] == 'ple' and m['seed'] == 7)
        require(digest(NUMERICAL/record['file']) == record['sha256'], 'PLE checkpoint changed')
        ple = MLPClassifier(xx.shape[1], 8, replace(neural.CONFIG, hidden_dims=(64, 32)))
        ple.load_state_dict(torch.load(NUMERICAL/record['file'], weights_only=True, map_location='cpu'))
        require(neural.state_hash(ple) == record['final_state_sha256'], 'PLE state changed')
        before = time.perf_counter()
        predictions['ple'] = prior.numerical.predict(ple, calibration_x)
        timings['ple'] = time.perf_counter()-before
        scores = {}
        for name, pred in predictions.items():
            files[name+'-calibration.npy'] = save_array(output, name+'-calibration.npy', pred)
            scores[name] = prior.old.prior.prior.scored(d.y[d.calibration], pred, d.pop)
        checks = {name: rep.gate(scores['retrieval']['metrics'], scores[name]['metrics'])
                  for name in ['tree', 'self_context_control']}
        result = dict(status='complete', recipe=recipe, models=records, scores=scores, files=files,
            calibration_supports=np.bincount(d.y[d.calibration], minlength=8).tolist(),
            continuation_checks=checks, continue_to_confirmation=all(checks.values()),
            test_opened=False, test_evaluated=False, audit_scored=False,
            independent_confirmation=False, deployment_authorized=False,
            examples_processed=sum(m['examples_processed'] for m in records),
            optimizer_steps=sum(m['optimizer_steps'] for m in records),
            search_fit_seconds=search_fit_seconds, search_calibration_seconds=search_calibration_seconds,
            prediction_seconds=timings)
    require(identity() == ident, 'Source/plan changed mid-run')
    result['elapsed_seconds'] = time.perf_counter()-start
    frozen.save(output/'receipt.json', result)
    print({name: round(s['metrics']['macro_f1'], 6) for name, s in scores.items()}, flush=True)
    print({'continue_to_confirmation': result['continue_to_confirmation']}, flush=True)


def verify(output):
    r = frozen.read(output/'receipt.json')
    require(r['status'] == 'complete' and r['recipe']['identity'] == identity()
            and all(r[k] is False for k in ['test_opened', 'test_evaluated', 'audit_scored',
                                           'independent_confirmation', 'deployment_authorized'])
            and r['examples_processed'] == 32439120 and r['optimizer_steps'] == 63360,
            'Invalid receipt')
    require(frozen.read(output/'frozen-recipe.json') == r['recipe']
            and frozen.read(output/'models-frozen.json') == r['models'], 'Frozen recipe/models changed')
    require(set(r['scores']) == {'tree', 'knn', 'ple', 'self_context_control', 'retrieval'}
            and [m['arm'] for m in r['models']] == ['self_context_control', 'retrieval'], 'Missing arms')
    for name, checksum in r['files'].items():
        require(Path(name).name == name and digest(output/name) == checksum, 'Artifact changed')
    for model in r['models']:
        require(model['sha256'] == r['files'][model['file']]
                and model['examples_processed'] == 16219560 and model['optimizer_steps'] == 31680,
                'Invalid checkpoint or budget')
    for name, score in r['scores'].items():
        prior.old.prior.prior.verify_score(score, r['calibration_supports'], r['recipe']['fit_population_counts'])
        pred = np.load(output/(name+'-calibration.npy'), allow_pickle=False)
        require(pred.shape == (409918,) and pred.dtype == np.uint8 and (pred < 8).all()
                and neural.vector_hash(pred) == score['prediction_sha256'], 'Predictions changed')
    expected = {name: rep.gate(r['scores']['retrieval']['metrics'], r['scores'][name]['metrics'])
                for name in ['tree', 'self_context_control']}
    require(r['continuation_checks'] == expected and r['continue_to_confirmation'] == all(expected.values()),
            'Continuation decision changed')
    return r


def publish(first, replay, destination):
    require(first.resolve() != replay.resolve() and not destination.exists(), 'Fresh independent replay required')
    a, b = verify(first), verify(replay)
    require(prior.without_timing(a) == prior.without_timing(b), 'Retrieval replay mismatch')
    frozen.save(destination, dict(a, non_timing_replay_exact=True,
        first_sha256=digest(first/'receipt.json'), replay_sha256=digest(replay/'receipt.json')))
    print('PASS: bank, contexts, both fits and all five predictions replay exactly; audit/test closed', flush=True)


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
