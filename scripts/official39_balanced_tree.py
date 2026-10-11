"""Equal-class loss on the fixed masked tree, with an exact historical bridge.

Only development fit/audit rows are accessed. No test, calibration, threshold
search, new neural training, remote API or cloud launch is part of this stage.
"""
import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from scripts import official39_matched_neural as prior
from src.data.label_map import WEB_BASED, BRUTE_FORCE
from src.eval import balanced_tree as balanced
from src.eval.official_evidence import describe, digest, require

ablation, expert, frozen, headroom = prior.prior, prior.older, prior.frozen, prior.headroom
PLAN = ROOT/'reports/risk_aware_routing/balanced-tree-plan.json'
REFERENCE = ROOT/'reports/risk_aware_routing/matched-neural-results.json'
ARM = prior.neural.ARM
FLAGS = (*prior.FLAGS, 'threshold_search', 'calibration_used')
SOURCES = ['src/eval/balanced_tree.py', 'scripts/official39_balanced_tree.py']
METHODS = ('sqrt_population', 'equal_class')


def validate_plan(p):
    require(p['protocol'] == 'official39-balanced-tree-v1'
            and p['scope'] == 'reused_validation_exploratory_internal_audit'
            and p['reference_sha256'] == digest(REFERENCE) and p['seeds'] == [7, 17, 27]
            and p['arm'] == ARM and p['weighting'] == 'equal_class'
            and p['classifier'] == balanced.expert.CONFIG
            and p['device'] == 'cpu' and p['threads'] == 2
            and p['fit_rows'] == 270326 and p['audit_rows'] == 408027
            and all(p[k] is False for k in FLAGS), 'Unreviewed class-weight experiment')


def fit_one(output, x, y, rows, population, features, seed, old):
    name = f'seed{seed}-equal-class.joblib'
    require(not (output/name).exists(), 'Preserve existing model')
    started = time.perf_counter()
    z = ablation.ablation.project(x[rows], features, ARM)
    require(ablation.hash_vector(z) == old['fit_inputs_sha256']
            and ablation.hash_vector(rows.astype('<i8')) == old['original_fit_rows_sha256'],
            'Fitting inputs or row identities differ')
    model, spec = balanced.fit(z, np.array(y[rows]), population, seed)
    for key in ('fit_rows', 'fit_sample_counts', 'fit_population_counts', 'config'):
        require(spec[key] == old[key], 'A factor besides weights changed')
    raw = balanced.expert.model_bytes(model)
    with (output/name).open('xb') as stream:
        stream.write(raw)
    restored = balanced.expert.restore_owned(raw, spec['model_sha256'], 'features', seed)
    with threadpool_limits(limits=2):
        prediction = balanced.expert.predict(restored, z)
    require(ablation.hash_vector(prediction) == spec['training_prediction_sha256']
            and ablation.ablation.masked_splits(restored, features, ARM) == 0,
            'Serialization changed predictions or mask leaked')
    spec.update(file=name, model_bytes=len(raw), serialization_bridge_exact=True,
        original_fit_rows_sha256=old['original_fit_rows_sha256'], fit_inputs_sha256=old['fit_inputs_sha256'],
        arm=ARM, dropped_features=ablation.ablation.ARMS[ARM], masked_feature_splits=0,
        seconds=time.perf_counter()-started)
    print(f'FROZEN: equal-class seed={seed}; 800 trees; {spec["seconds"]:.1f}s', flush=True)
    return spec


def screen(case):
    a, b = [case['audit'][m]['metrics'] for m in METHODS]
    return dict(seed=case['seed'], macro_f1_higher=b['macro_f1'] > a['macro_f1'],
        rare_f1_higher={c: b['per_class'][c]['f1'] > a['per_class'][c]['f1'] for c in (WEB_BASED, BRUTE_FORCE)},
        benign_false_alerts_not_higher=b['benign_false_alert_rate'] <= a['benign_false_alert_rate'])


def summarize(cases):
    require([c['seed'] for c in cases] == [7, 17, 27], 'Need all three paired seeds')
    out = {}
    for cohort in ('fit', 'audit'):
        out[cohort] = {}
        for method in METHODS:
            values = [c[cohort][method] for c in cases]
            out[cohort][method] = {view: dict(
                **{k: describe([v[view][k] for v in values]) for k in ('macro_f1', 'benign_false_alert_rate')},
                per_class={name: {k: describe([v[view]['per_class'][name][k] for v in values])
                           for k in ('precision', 'recall', 'f1')} for name in balanced.CLASSES})
                for view in ('metrics', 'prior_standardized')}
    out['equal_minus_sqrt_audit'] = {k: describe([
        c['audit']['equal_class']['metrics'][k]-c['audit']['sqrt_population']['metrics'][k] for c in cases])
        for k in ('macro_f1', 'benign_false_alert_rate')}
    out['standardized_audit_minus_fit'] = {m: describe([
        c['audit'][m]['prior_standardized']['macro_f1']-c['fit'][m]['prior_standardized']['macro_f1']
        for c in cases]) for m in METHODS}
    out['screens'] = [screen(c) for c in cases]
    out['continuation_screen_passed'] = all(s['macro_f1_higher'] and s['benign_false_alerts_not_higher']
        and all(s['rare_f1_higher'].values()) for s in out['screens'])
    return out


def paired(y, a, b):
    # Relabel the reused generic correctness counter explicitly: there is no
    # neural classifier in this comparison. The oracle is diagnostic only.
    raw = prior.neural.paired_correctness(y, a, b)
    return {c: dict(support=v['support'], both_correct=v['both_correct'],
        sqrt_only_correct=v['tree_only_correct'], equal_only_correct=v['neural_only_correct'],
        neither_correct=v['neither_correct'], oracle_recall_upper_bound=v['oracle_recall_upper_bound'])
        for c, v in raw.items()}


def verify_pair(p, a, b):
    # Use the already tested marginal checks without copying their logic.
    mapped = {c: dict(support=v['support'], both_correct=v['both_correct'],
        tree_only_correct=v['sqrt_only_correct'], neural_only_correct=v['equal_only_correct'],
        neither_correct=v['neither_correct'], oracle_recall_upper_bound=v['oracle_recall_upper_bound'])
        for c, v in p.items()}
    prior.verify_pair(mapped, a, b)


def audit_record(r):
    require(r['status'] == 'complete' and all(r[k] is False for k in FLAGS), 'Wrong completed scope')
    validate_plan(r['identity']['plan'])
    ref, tree_ref = frozen.read(REFERENCE), frozen.read(prior.REFERENCE)
    prior.audit_record(ref)
    sources = dict(ref['identity']['source_sha256'])
    sources.update({n: digest(ROOT/n) for n in SOURCES})
    require(r['identity']['plan_sha256'] == digest(PLAN) and r['identity']['source_sha256'] == sources
            and r['identity']['runtime'] == ref['identity']['runtime']
            and r['common_panel'] == ref['common_panel'] and r['split'] == ref['split']
            and r['validation_panel'] == ref['validation_panel'], 'Identity changed')
    require([s['seed'] for s in r['models']] == [7, 17, 27], 'Missing fixed-budget fits')
    for spec in r['models']:
        old = next(s for s in tree_ref['models'] if s['seed'] == spec['seed'] and s['arm'] == ARM)
        for key in ('mode', 'inputs', 'classes', 'config', 'fit_rows', 'fit_sample_counts', 'fit_population_counts',
                    'original_fit_rows_sha256', 'fit_inputs_sha256', 'arm', 'dropped_features'):
            require(spec[key] == old[key], 'Non-weight factor changed')
        require(spec['weighting'] == 'equal_class' and spec['iterations'] == 100 and spec['trees'] == 800
                and 800 <= spec['leaves'] <= 12000 and spec['nodes'] == 2*spec['leaves']-800
                and 1 <= spec['maximum_leaves_in_tree'] <= 15 and spec['masked_feature_splits'] == 0
                and spec['serialization_bridge_exact'] and 0 < spec['model_bytes'] < 20_000_000
                and spec['file'] == f'seed{spec["seed"]}-equal-class.joblib', 'Fit work or mask changed')
        require(np.asarray(spec['weighted_class_mass']).shape == (8,)
                and np.allclose(spec['weighted_class_mass'], np.full(8, spec['fit_rows']/8), rtol=1e-10, atol=1e-8),
                'Class loss mass is not equal')
    require(len({s['sample_weights_sha256'] for s in r['models']}) == 1, 'Seed changed sample weights')
    require([c['seed'] for c in r['cases']] == [7, 17, 27], 'Missing evaluations')
    for case, spec in zip(r['cases'], r['models']):
        old = next(c for c in ref['cases'] if c['seed'] == case['seed'])
        for cohort, supports in [('fit', dict(zip(balanced.CLASSES, spec['fit_sample_counts']))),
                                  ('audit', ref['common_panel']['retained_classes'])]:
            require(set(case[cohort]) == set(METHODS), 'Method coverage changed')
            for value in case[cohort].values():
                prior.verify_score(value, supports, spec['fit_population_counts'])
            require(case[cohort]['sqrt_population'] == old[cohort]['tree'], 'Historical predictions/metrics differ')
            verify_pair(case[cohort+'_paired_correctness'], case[cohort]['sqrt_population'], case[cohort]['equal_class'])
        require(case['fit']['equal_class']['prediction_sha256'] == spec['training_prediction_sha256'], 'Training bridge differs')
    require(r['summary'] == summarize(r['cases']), 'Summary changed')
    return dict(status='verified', fits=3, trees_fitted=2400, fit_rows=270326, audit_rows=408027,
                continuation_screen_passed=r['summary']['continuation_screen_passed'], test_opened=False)


def verify_output(output, r):
    require(digest(output/'models-frozen.json') == r['models_frozen_sha256']
            and frozen.read(output/'models-frozen.json') == dict(status='all_models_frozen_before_audit',
                identity=r['identity'], models=r['models']), 'Model freeze changed')
    for s in r['models']:
        require(Path(s['file']).name == s['file'] and digest(output/s['file']) == s['model_sha256']
                and (output/s['file']).stat().st_size == s['model_bytes'], 'Owned model changed')


def run(cache, packed, panel, trees, output):
    cache, packed, panel, trees, output = map(Path, (cache, packed, panel, trees, output))
    require(not output.exists(), 'Preserve existing output; use a fresh directory')
    p, ref, tree_ref = frozen.read(PLAN), frozen.read(REFERENCE), frozen.read(prior.REFERENCE)
    validate_plan(p)
    prior.audit_record(ref)
    ablation.verify_output(trees, tree_ref)
    require(digest(trees/'receipt.json') == tree_ref['source_receipt_sha256'], 'Wrong original tree receipt')
    cache_ref = frozen.read(expert.previous.PREVIOUS)
    expert.previous.verify_private(cache, cache_ref)
    runtime, parent = frozen.configure('cpu'), frozen.load_plan()
    require(runtime == ref['identity']['runtime'], 'Use historical local CPU runtime')
    manifest = frozen.verify_pack(packed, parent)
    sources = dict(ref['identity']['source_sha256'])
    sources.update({n: digest(ROOT/n) for n in SOURCES})
    identity = dict(plan=p, plan_sha256=digest(PLAN), source_sha256=sources, runtime=runtime)
    started = time.perf_counter()
    with frozen.session(output, identity, resume=False), threadpool_limits(limits=2):
        r = dict(status='fitting', identity=identity, models=[], common_panel=ref['common_panel'],
            validation_panel=ref['validation_panel'], split=ref['split'], **dict.fromkeys(FLAGS, False))
        frozen.save(output/'receipt.json', r)
        with headroom.validation_only(packed, manifest) as (x, y):
            features = manifest['features']
            keep, info = headroom.load_panel(panel, SimpleNamespace(root=packed, arrays={'val': (x, y)}, manifest=manifest))
            require(info == ref['validation_panel'], 'Validation cohort changed')
            rows = np.flatnonzero(keep)
            split, chosen, keys = [np.load(cache/n, allow_pickle=False)
                                   for n in ('partitions.npy', 'fit_rows.npy', 'group_keys.npy')]
            expert.previous.fusion.check_split(rows, split, chosen, keys)
            fit_rows, population = rows[chosen], np.bincount(y[rows[split == 0]], minlength=8)
            audit_rows = np.load(trees/'retained_rows.npy', allow_pickle=False)
            require(np.array_equal(audit_rows, rows[split == 2]) and len(audit_rows) == 408027,
                    'Common audit population changed')
            for seed in p['seeds']:
                old = next(m for m in tree_ref['models'] if m['seed'] == seed and m['arm'] == ARM)
                r['models'].append(fit_one(output, x, y, fit_rows, population, features, seed, old))
                frozen.save(output/'receipt.json', r)
            frozen.save(output/'models-frozen.json', dict(status='all_models_frozen_before_audit', identity=identity, models=r['models']))
            r['models_frozen_sha256'] = digest(output/'models-frozen.json')
            r['cases'] = []
            for spec in r['models']:
                seed = spec['seed']
                model = expert.load_owned(output, spec)
                old = next(m for m in tree_ref['models'] if m['seed'] == seed and m['arm'] == ARM)
                control = expert.load_owned(trees, old)
                case = dict(seed=seed)
                for cohort, ids in [('fit', fit_rows), ('audit', audit_rows)]:
                    a = ablation.predict_rows(control, x, ids, features, ARM)
                    b = ablation.predict_rows(model, x, ids, features, ARM)
                    labels = np.array(y[ids])
                    case[cohort] = {METHODS[0]: prior.scored(labels, a, population),
                                    METHODS[1]: prior.scored(labels, b, population)}
                    case[cohort+'_paired_correctness'] = paired(labels, a, b)
                r['cases'].append(case)
                print(f'PASS: seed={seed} fixed fit/audit scores recorded', flush=True)
            r['summary'] = summarize(r['cases'])
        expert.previous.verify_private(cache, cache_ref)
        ablation.verify_output(trees, tree_ref)
        require(frozen.verify_pack(packed, parent) == manifest
                and digest(panel/'receipt.json') == info['receipt_sha256']
                and all(digest(panel/n) == h for n, h in info['files'].items()), 'Inputs changed')
        verify_output(output, r)
        r.update(status='complete', elapsed_seconds=time.perf_counter()-started)
        audit_record(r)
        frozen.save(output/'receipt.json', r)
    return audit_record(r)


def publish(first, replay, destination):
    first, replay, destination = map(Path, (first, replay, destination))
    require(first.resolve() != replay.resolve() and not destination.exists(), 'Require independent replay and fresh publication')
    a, b = [frozen.read(p/'receipt.json') for p in (first, replay)]
    for path, record in ((first, a), (replay, b)):
        audit_record(record)
        verify_output(path, record)
    require(expert.without_timing(a) == expert.without_timing(b), 'Non-timing replay differs')
    result = dict(a, non_timing_replay_exact=True, source_receipt_sha256=digest(first/'receipt.json'),
                  replay_receipt_sha256=digest(replay/'receipt.json'))
    frozen.save(destination, result)
    return audit_record(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('run')
    for name in ('cache', 'packed', 'panel', 'trees', 'output'):
        fit.add_argument('--'+name, type=Path, required=True)
    pub = commands.add_parser('publish')
    for name in ('first', 'replay', 'destination'):
        pub.add_argument('--'+name, type=Path, required=True)
    check = commands.add_parser('audit')
    check.add_argument('--record', type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    result = run(**args) if command == 'run' else publish(**args) if command == 'publish' else audit_record(frozen.read(args['record']))
    import json
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
