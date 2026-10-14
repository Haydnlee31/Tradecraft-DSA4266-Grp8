"""Bounded text screen, then a fixed numerical-embedding fallback.

Reuses the already audited development cohorts. This is not an independent
confirmation and has no interface for the original test or cloud execution.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits
from scripts import official39_score_adjustment as old
from src.eval import flow_representation as rep
from src.eval import matched_neural as neural
from src.eval import numerical_embedding as numerical
from src.eval.official_evidence import digest, require
from src.data.label_map import CLASSES

frozen, expert, ablation, headroom = old.frozen, old.expert, old.ablation, old.headroom
PLAN = ROOT/'reports/risk_aware_routing/representation-plan.json'
REFERENCE = ROOT/'reports/risk_aware_routing/score-adjustment-results.json'
SOURCES = ['src/eval/flow_representation.py', 'scripts/official39_representation_pilot.py',
           'scripts/official39_laya_inference.py', 'src/eval/numerical_embedding.py']
PACKED = ROOT/'outputs/official39-packed-2m-v1'
PANEL = ROOT/'outputs/official39-ablation-panel-v1'
CACHE = ROOT/'outputs/official39-routing-gate-v2'
TREES = ROOT/'outputs/official39-feature-ablation-v1'
ADJUSTMENT = ROOT/'outputs/official39-score-adjustment-v1'


def identity():
    p = frozen.read(PLAN)
    require(p['reference_sha256'] == digest(REFERENCE) and p['test_opened'] is False
            and p['cloud_authorized'] is False, 'Unreviewed reference')
    return dict(plan=p, plan_sha256=digest(PLAN), reference_sha256=digest(REFERENCE),
                sources={n: digest(ROOT/n) for n in SOURCES})


@contextmanager
def data():
    ref = frozen.read(REFERENCE)
    old.audit_record(ref)
    old.verify_output(ADJUSTMENT, ref)
    tree_ref = frozen.read(old.prior.prior.REFERENCE)
    ablation.verify_output(TREES, tree_ref)
    cache_ref = frozen.read(expert.previous.PREVIOUS)
    expert.previous.verify_private(CACHE, cache_ref)
    runtime = frozen.configure('cpu')
    require(runtime == ref['identity']['runtime'], 'Historical base runtime changed')
    manifest = frozen.verify_pack(PACKED, frozen.load_plan())
    with threadpool_limits(limits=2), headroom.validation_only(PACKED, manifest) as (x, y):
        keep, info = headroom.load_panel(PANEL, SimpleNamespace(root=PACKED, arrays={'val': (x, y)}, manifest=manifest))
        require(info == ref['validation_panel'], 'Validation panel changed')
        rows = np.flatnonzero(keep)
        split, chosen, keys = [np.load(CACHE/n, allow_pickle=False)
                              for n in ('partitions.npy', 'fit_rows.npy', 'group_keys.npy')]
        expert.previous.fusion.check_split(rows, split, chosen, keys)
        fit = rows[chosen]
        audit = np.load(TREES/'retained_rows.npy', allow_pickle=False)
        calibration = np.load(ADJUSTMENT/'calibration-rows.npy', allow_pickle=False)
        require(np.array_equal(audit, rows[split == 2]) and len(fit) == 270326
                and len(calibration) == 409918 and len(audit) == 408027, 'Cohort changed')
        pop = np.bincount(y[rows[split == 0]], minlength=8)
        require(pop.tolist() == ref['fit_population_counts'], 'Fitting prior changed')
        yield SimpleNamespace(x=x, y=y, fit=fit, audit=audit, calibration=calibration,
            features=manifest['features'], pop=pop, tree_ref=tree_ref, runtime=runtime)
    require(manifest == frozen.verify_pack(PACKED, frozen.load_plan()), 'Packed inputs mutated')
    old.verify_output(ADJUSTMENT, ref)
    expert.previous.verify_private(CACHE, cache_ref)
    ablation.verify_output(TREES, tree_ref)


def tree_model(d, seed):
    spec = next(m for m in d.tree_ref['models'] if m['seed'] == seed and m['arm'] == old.prior.ARM)
    return expert.load_owned(TREES, spec)


def prepare(output, checkpoint):
    require(not output.exists(), 'Preserve existing preparation')
    ident = identity()
    spec = ident['plan']['laya']
    commit = subprocess.check_output(['git', '-C', str(ROOT/'outputs/laya-upstream-review'),
                                      'rev-parse', 'HEAD'], text=True).strip()
    require(commit == spec['source_commit'], 'Wrong Laya source revision')
    files = {str(p.relative_to(checkpoint)): digest(p) for p in sorted(checkpoint.rglob('*'))
             if p.is_file() and '.cache' not in p.parts}
    require('model.safetensors' in files, 'Checkpoint download incomplete')
    with data() as d:
        ids = rep.retained(d.features)
        edges = rep.fit_edges(d.x[d.fit][:, ids])
        rows = rep.stratified_rows(d.calibration, d.y[d.calibration], spec['rows_per_class'], spec['sample_seed'])
        started = time.perf_counter()
        states = {mode: rep.serialize(np.array(d.x[rows]), d.features, edges, mode)
                  for mode in spec['serializers']}
        serialize_seconds = time.perf_counter()-started
        request = dict(plan=ident['plan'], classes=CLASSES, checkpoint_files=files,
            test_opened=False, states=states, questions={'attack': dict(type='choice',
                instructions='Classify this IoT network flow into exactly one security category.',
                criteria=dict(zip(CLASSES, ['normal traffic', 'distributed denial of service',
                    'denial of service', 'reconnaissance or scanning', 'web application attack',
                    'password guessing', 'network impersonation', 'Mirai botnet attack'])))})
        labels = np.array(d.y[rows])
        pop = np.bincount(d.y[d.calibration], minlength=8)
        tree = tree_model(d, 7)
        pred = ablation.predict_rows(tree, d.x, rows, d.features, old.prior.ARM)
        output.mkdir()
        np.save(output/'rows.npy', rows, allow_pickle=False)
        np.save(output/'labels.npy', labels, allow_pickle=False)
        frozen.save(output/'inputs.json', request)
        frozen.save(output/'edges.json', edges)
        frozen.save(output/'preparation.json', dict(identity=ident, rows=len(rows),
            sample_counts=np.bincount(labels, minlength=8).tolist(), calibration_population_counts=pop.tolist(),
            baseline=old.prior.prior.scored(labels, pred, pop), serialization_seconds=serialize_seconds,
            files={n: digest(output/n) for n in ['rows.npy', 'labels.npy', 'inputs.json', 'edges.json']},
            test_opened=False, audit_scored=False, labels_in_model_input=False))
    print('PASS: fixed label-free Laya inputs prepared; test and audit unscored', flush=True)


def laya_review(prepared, first, replay, destination):
    require(not destination.exists() and first.resolve() != replay.resolve(), 'Fresh result and independent replay required')
    p = frozen.read(prepared/'preparation.json')
    require(p['identity'] == identity(), 'Preparation source changed')
    require(all(digest(prepared/n) == sha for n, sha in p['files'].items()), 'Prepared data changed')
    runs = [frozen.read(path) for path in [first, replay]]
    require(without_timing(runs[0]) == without_timing(runs[1]), 'Laya replay changed')
    r = runs[0]
    require(r['status'] == 'complete' and r['test_opened'] is False and r['fine_tuned'] is False
            and r['cpu_fallback_count'] == 0 and r['input_sha256'] == digest(prepared/'inputs.json')
            and r['source_sha256'] == digest(ROOT/'scripts/official39_laya_inference.py')
            and set(r['arms']) == set(rep.SERIALIZERS), 'Wrong Laya inference receipt')
    labels = np.load(prepared/'labels.npy', allow_pickle=False)
    scores, passing = {}, []
    for name, arm in r['arms'].items():
        prediction = np.asarray(arm['predictions'], dtype=np.uint8)
        scores[name] = old.prior.prior.scored(labels, prediction, p['calibration_population_counts'])
        if rep.gate(scores[name]['prior_standardized'], p['baseline']['prior_standardized']):
            passing.append(name)
    result = dict(status='complete', identity=identity(), preparation=p, inference=r,
        first_sha256=digest(first), replay_sha256=digest(replay), replay_exact=True,
        scores=scores, passing_arms=passing, pivot_authorized=not passing,
        test_opened=False, audit_scored=False,
        interpretation='Balanced calibration screen reweighted to calibration prior; not independent or full-population measurement')
    frozen.save(destination, result)
    print({'passing_arms': passing, 'pivot_authorized': not passing}, flush=True)


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items()
                if not k.endswith('_seconds') and k != 'seconds'}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def numerical_run(laya_result, output):
    require(not output.exists(), 'Preserve existing numerical run')
    ident = identity()
    laya = frozen.read(laya_result)
    require(laya['identity'] == ident and laya['status'] == 'complete'
            and laya['pivot_authorized'] is True and laya['passing_arms'] == [], 'Pivot not authorized by screen')
    spec = ident['plan']['numerical_pivot']
    require(spec['bins'] == 16 and spec['seeds'] == [7, 17, 27]
            and spec['epochs'] == 60 and spec['batch_size'] == 512, 'Unexpected numerical recipe')
    started = time.perf_counter()
    output.mkdir()
    models, model_records = {}, []
    with data() as d:
        fit_x = neural.inputs(d.x[d.fit], d.features)
        ids = rep.retained(d.features)
        edges = rep.fit_edges(fit_x[:, ids], spec['bins'])
        ple_x = rep.encode(fit_x[:, ids], edges)
        width = numerical.raw_width(ple_x.shape[1])
        frozen.save(output/'edges.json', edges)
        # Freeze representation, row IDs and all hyperparameters before fits.
        freeze = dict(identity=ident, laya_result_sha256=digest(laya_result), runtime=d.runtime,
            fit_rows_sha256=neural.vector_hash(d.fit), audit_rows_sha256=neural.vector_hash(d.audit),
            fit_population_counts=d.pop.tolist(), fit_sample_counts=np.bincount(d.y[d.fit], minlength=8).tolist(),
            edges_sha256=digest(output/'edges.json'), ple_dimensions=ple_x.shape[1], raw_first_width=width,
            test_opened=False, no_audit_selection=True)
        frozen.save(output/'frozen-recipe.json', freeze)
        for arm in spec['arms']:
            xx, first_width = (ple_x, 64) if arm == 'ple' else (fit_x, width)
            for seed in spec['seeds']:
                begin = time.perf_counter()
                model, record = numerical.fit(xx, np.array(d.y[d.fit], dtype=np.int64), d.pop, seed, first_width)
                require(record['examples_processed'] == 16219560 and record['optimizer_steps'] == 31680,
                        'Training budget changed')
                name = f'{arm}-seed{seed}.pt'
                (output/name).write_bytes(neural.serialize(model))
                record.update(arm=arm, file=name, sha256=digest(output/name), seconds=time.perf_counter()-begin)
                # Reload tensor-only artifacts before any audit inference.
                import io
                import torch
                model.load_state_dict(torch.load(io.BytesIO((output/name).read_bytes()),
                                                map_location='cpu', weights_only=True), strict=True)
                require(neural.state_hash(model) == record['final_state_sha256'], 'Checkpoint reload changed weights')
                models[(arm, seed)] = model
                model_records.append(record)
        frozen.save(output/'models-frozen.json', model_records)
        print('PASS: all six models frozen before audit scoring', flush=True)
        audit_x = neural.inputs(d.x[d.audit], d.features)
        audit_ple = rep.encode(audit_x[:, ids], edges)
        cases = []
        historical = frozen.read(REFERENCE)
        for seed in spec['seeds']:
            tree = tree_model(d, seed)
            tree_pred = ablation.predict_rows(tree, d.x, d.audit, d.features, old.prior.ARM)
            baseline = old.prior.prior.scored(d.y[d.audit], tree_pred, d.pop)
            expected = next(c for c in historical['audit'] if c['seed'] == seed)['scores']['baseline']
            require(baseline == expected, 'Historical tree audit bridge changed')
            scores = {'tree': baseline}
            for arm, xx in [('ple', audit_ple), ('raw_capacity_control', audit_x)]:
                pred = numerical.predict(models[(arm, seed)], xx)
                np.save(output/f'{arm}-seed{seed}-audit.npy', pred, allow_pickle=False)
                scores[arm] = old.prior.prior.scored(d.y[d.audit], pred, d.pop)
            # Gate uses observed audit metrics, not reweighted fitting diagnostics.
            case = dict(seed=seed, scores=scores,
                ple_passes_tree=rep.gate(scores['ple']['metrics'], baseline['metrics']),
                ple_passes_capacity_control=rep.gate(scores['ple']['metrics'], scores['raw_capacity_control']['metrics']))
            cases.append(case)
            print(f'AUDIT seed={seed}: PLE={scores["ple"]["metrics"]["macro_f1"]:.6f}; raw control={scores["raw_capacity_control"]["metrics"]["macro_f1"]:.6f}', flush=True)
        result = dict(status='complete', recipe=freeze, models=model_records, cases=cases,
            models_frozen_sha256=digest(output/'models-frozen.json'),
            all_seeds_pass_tree=all(c['ple_passes_tree'] for c in cases),
            all_seeds_pass_capacity_control=all(c['ple_passes_capacity_control'] for c in cases),
            test_opened=False, test_evaluated=False, deployment_authorized=False, independent_confirmation=False,
            examples_processed=sum(m['examples_processed'] for m in model_records),
            optimizer_steps=sum(m['optimizer_steps'] for m in model_records))
    require(identity() == ident, 'Experiment source changed mid-run')
    result['elapsed_seconds'] = time.perf_counter()-started
    frozen.save(output/'receipt.json', result)


def verify_numerical(output):
    r = frozen.read(output/'receipt.json')
    require(r['status'] == 'complete' and r['recipe']['identity'] == identity()
            and not r['test_opened'] and not r['test_evaluated'] and not r['deployment_authorized']
            and r['examples_processed'] == 97317360 and r['optimizer_steps'] == 190080
            and len(r['models']) == 6 and len(r['cases']) == 3, 'Wrong numerical result')
    require(frozen.read(output/'frozen-recipe.json') == r['recipe']
            and digest(output/'edges.json') == r['recipe']['edges_sha256']
            and digest(output/'models-frozen.json') == r['models_frozen_sha256']
            and frozen.read(output/'models-frozen.json') == r['models'], 'Frozen records changed')
    supports = frozen.read(REFERENCE)['audit_panel']['retained_classes']
    for model in r['models']:
        require(digest(output/model['file']) == model['sha256'], 'Checkpoint changed')
    for case in r['cases']:
        for arm, score in case['scores'].items():
            old.prior.prior.verify_score(score, supports, r['recipe']['fit_population_counts'])
            if arm != 'tree':
                pred = np.load(output/f'{arm}-seed{case["seed"]}-audit.npy', allow_pickle=False)
                require(neural.vector_hash(pred) == score['prediction_sha256'], 'Saved predictions changed')
    return r


def publish(first, replay, destination):
    require(first.resolve() != replay.resolve() and not destination.exists(), 'Independent replay required')
    a, b = verify_numerical(first), verify_numerical(replay)
    # The frozen-model JSON also contains measured durations; compare its
    # substantive content rather than requiring identical stopwatch bytes.
    def comparable(r):
        return without_timing({k: v for k, v in r.items() if k != 'models_frozen_sha256'})
    require(comparable(a) == comparable(b), 'Numerical training/inference replay changed')
    frozen.save(destination, dict(a, non_timing_replay_exact=True,
        first_sha256=digest(first/'receipt.json'), replay_sha256=digest(replay/'receipt.json')))
    print('PASS: all six fits replay exactly; historical tree bridge exact; test untouched', flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p = sub.add_parser('review-laya')
    for name in ['prepared', 'first', 'replay', 'destination']:
        p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('numerical')
    p.add_argument('--laya-result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('publish')
    for name in ['first', 'replay', 'destination']:
        p.add_argument('--'+name, type=Path, required=True)
    args = vars(ap.parse_args())
    command = args.pop('command')
    {'prepare': prepare, 'review-laya': laya_review, 'numerical': numerical_run, 'publish': publish}[command](**args)


if __name__ == '__main__':
    main()
