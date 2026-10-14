"""Bounded local neural control using the boosted tree's data and objective.

No final-test loader, cloud mode, hyperparameter selection or deployment path.
The historical classifiers remain untouched. Three new neural models are
frozen before audit scoring; an independent full replay precedes publication.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from scripts import official39_feature_ablation as prior
from src.eval import matched_neural as neural
from src.eval.official_evidence import describe, digest, metrics_from_counts, require

frozen, older, headroom = prior.frozen, prior.prior, prior.headroom
PLAN = ROOT/'reports/risk_aware_routing/matched-neural-plan.json'
REFERENCE = ROOT/'reports/risk_aware_routing/feature-ablation-results.json'
FLAGS = ('test_opened', 'test_evaluated', 'cloud_authorized', 'hyperparameter_search',
         'human_review_required', 'deployment_authorized', 'independent_confirmation')
SOURCES = ['src/eval/matched_neural.py', 'scripts/official39_matched_neural.py']


def validate_plan(p):
    require(p['protocol'] == 'official39-matched-neural-v1'
            and p['scope'] == 'reused_validation_exploratory_internal_audit'
            and p['reference_sha256'] == digest(REFERENCE) and p['seeds'] == [7, 17, 27]
            and p['arm'] == neural.ARM and p['device'] == 'cpu' and p['threads'] == 2
            and p['fit_rows'] == 270326 and p['audit_rows'] == 408027
            and p['settings'] == neural.SETTINGS
            and p['architecture'] == dict(asdict(neural.CONFIG), hidden_dims=[64, 32])
            and all(p[k] is False for k in FLAGS), 'Unreviewed neural scope or recipe')


def scored(y, pred, population):
    value = prior.evaluate(y, pred)
    value['prior_standardized'] = neural.prior_standardized(value['metrics']['confusion_matrix'], population)
    return value


def fit_one(output, x, y, rows, population, features, seed, tree_spec):
    start = time.perf_counter()
    z64 = prior.ablation.project(x[rows], features, neural.ARM)
    require(prior.hash_vector(z64) == tree_spec['fit_inputs_sha256']
            and neural.vector_hash(rows.astype('<i8')) == tree_spec['original_fit_rows_sha256'],
            'Tree/neural fitting inputs differ')
    z, labels = z64.astype(np.float32), np.array(y[rows])
    model, spec = neural.fit(z, labels, population, seed)
    for key in ('fit_rows', 'fit_sample_counts', 'fit_population_counts', 'weighted_class_mass'):
        require(spec[key] == tree_spec[key], 'Tree/neural objective differs')
    name = f'seed{seed}-matched-neural.pt'
    with (output/name).open('xb') as stream:
        stream.write(neural.serialize(model))
    checksum = digest(output/name)
    restored = neural.restore((output/name).read_bytes(), checksum, spec['final_state_sha256'])
    require(neural.vector_hash(neural.predict(restored, z)) == spec['training_prediction_sha256'],
            'Neural serialization changed predictions')
    spec.update(file=name, file_sha256=checksum, file_bytes=(output/name).stat().st_size,
        original_fit_rows_sha256=tree_spec['original_fit_rows_sha256'],
        fit_inputs_sha256=tree_spec['fit_inputs_sha256'], serialization_bridge_exact=True,
        seconds=time.perf_counter()-start)
    print(f'FROZEN: neural seed={seed}; {spec["optimizer_steps"]:,} updates; {spec["seconds"]:.1f}s', flush=True)
    return spec


def predict_rows(model, x, rows, features):
    result = np.empty(len(rows), dtype=np.uint8)
    for start in range(0, len(rows), 8192):
        take = rows[start:start+8192]
        result[start:start+len(take)] = neural.predict(model, neural.inputs(x[take], features))
    return result


def summarize(cases):
    require([c['seed'] for c in cases] == [7, 17, 27], 'Require all three paired seeds')
    summaries = {}
    for cohort in ('fit', 'audit'):
        summaries[cohort] = {}
        for method in ('tree', 'neural'):
            values = [c[cohort][method] for c in cases]
            summaries[cohort][method] = {
                view: dict(**{k: describe([v[view][k] for v in values])
                              for k in ('macro_f1', 'benign_false_alert_rate')},
                    per_class={name: {k: describe([v[view]['per_class'][name][k] for v in values])
                               for k in ('precision', 'recall', 'f1')} for name in neural.CLASSES})
                for view in ('metrics', 'prior_standardized')}
    summaries['paired_audit_neural_minus_tree'] = {
        k: describe([c['audit']['neural']['metrics'][k]-c['audit']['tree']['metrics'][k] for c in cases])
        for k in ('macro_f1', 'benign_false_alert_rate')}
    summaries['standardized_audit_minus_fit'] = {method: {
        k: describe([c['audit'][method]['prior_standardized'][k]-c['fit'][method]['prior_standardized'][k]
                     for c in cases]) for k in ('macro_f1', 'benign_false_alert_rate')}
        for method in ('tree', 'neural')}
    return summaries


def verify_score(value, supports, population):
    m, e = metrics_from_counts(value['metrics']['confusion_matrix'], supports)
    require(value['metrics'] == m and value['errors'] == e
            and value['prior_standardized'] == neural.prior_standardized(m['confusion_matrix'], population),
            'Score/prior-standardization changed')


def verify_pair(pair, tree, network):
    require(set(pair) == set(neural.CLASSES), 'Missing paired class')
    for name in neural.CLASSES:
        p = pair[name]
        require(all(type(p[k]) is int and p[k] >= 0 for k in
                    ('support', 'both_correct', 'tree_only_correct', 'neural_only_correct', 'neither_correct'))
                and p['support'] > 0
                and p['both_correct']+p['tree_only_correct']+p['neural_only_correct']+p['neither_correct'] == p['support'],
                'Invalid paired counts')
        i = neural.CLASSES.index(name)
        require(p['both_correct']+p['tree_only_correct'] == tree['metrics']['confusion_matrix'][i][i]
                and p['both_correct']+p['neural_only_correct'] == network['metrics']['confusion_matrix'][i][i]
                and p['support'] == tree['metrics']['per_class'][name]['support']
                and p['oracle_recall_upper_bound'] == (p['support']-p['neither_correct'])/p['support'], 'Paired marginals changed')


def audit_record(r):
    require(r['status'] == 'complete' and all(r[k] is False for k in FLAGS), 'Wrong neural experiment scope')
    p = r['identity']['plan']
    validate_plan(p)
    ref = frozen.read(REFERENCE)
    prior.audit_record(ref)
    sources = dict(ref['identity']['source_sha256'])
    sources.update({n: digest(ROOT/n) for n in SOURCES})
    require(r['identity']['source_sha256'] == sources and r['identity']['plan_sha256'] == digest(PLAN)
            and r['identity']['runtime'] == ref['identity']['runtime']
            and r['common_panel'] == ref['common_panel']
            and r['validation_panel'] == ref['validation_panel'] and r['split'] == ref['split'], 'Identity changed')
    require([m['seed'] for m in r['models']] == [7, 17, 27], 'Missing neural models')
    for spec in r['models']:
        old = next(m for m in ref['models'] if m['seed'] == spec['seed'] and m['arm'] == neural.ARM)
        for key in ('fit_rows', 'fit_sample_counts', 'fit_population_counts', 'weighted_class_mass',
                    'original_fit_rows_sha256', 'fit_inputs_sha256'):
            require(spec[key] == old[key], 'Tree/neural rows or objective changed')
        require(spec['settings'] == neural.SETTINGS and spec['model_config'] == p['architecture']
                and spec['parameters'] == 5096 and spec['parameter_bytes'] == 20384
                and spec['examples_processed'] == 16219560 and spec['optimizer_steps'] == 31680
                and spec['serialization_bridge_exact'] and 0 < spec['file_bytes'] < 100000
                and spec['file'] == f'seed{spec["seed"]}-matched-neural.pt', 'Neural budget/schema changed')
        require([o['epoch'] for o in spec['observations']] == p['settings']['report_epochs'], 'Training curve changed')
        for o in spec['observations']:
            supports = dict(zip(neural.CLASSES, spec['fit_sample_counts']))
            metrics_from_counts(o['training_confusion_matrix'], supports)
            require(o['train_prior_standardized'] == neural.prior_standardized(
                        o['training_confusion_matrix'], spec['fit_population_counts'])
                    and o['examples_processed'] == 270326*o['epoch']
                    and o['optimizer_steps'] == 528*o['epoch']
                    and np.isfinite(o['train_online_weighted_ce']) and o['train_online_weighted_ce'] >= 0,
                    'Training observation changed')
        require(spec['observations'][-1]['state_sha256'] == spec['final_state_sha256'], 'Wrong endpoint')
    require([c['seed'] for c in r['cases']] == [7, 17, 27], 'Missing paired evaluation')
    for case, spec in zip(r['cases'], r['models']):
        old_spec = next(m for m in ref['models'] if m['seed'] == case['seed'] and m['arm'] == neural.ARM)
        old = next(c for c in ref['cases'] if c['seed'] == case['seed'])
        pop = spec['fit_population_counts']
        for cohort, supports in [('fit', dict(zip(neural.CLASSES, spec['fit_sample_counts']))),
                                  ('audit', ref['common_panel']['retained_classes'])]:
            require(set(case[cohort]) == {'tree', 'neural'}, 'Comparison method changed')
            for value in case[cohort].values():
                verify_score(value, supports, pop)
            verify_pair(case[cohort+'_paired_correctness'], case[cohort]['tree'], case[cohort]['neural'])
        require(case['fit']['tree']['prediction_sha256'] == old_spec['training_prediction_sha256']
                and case['fit']['neural']['prediction_sha256'] == spec['training_prediction_sha256']
                and case['fit']['neural']['metrics']['confusion_matrix'] == spec['observations'][-1]['training_confusion_matrix'],
                'Fitting prediction bridge changed')
        require(all(case['audit']['tree'][k] == old['primary'][neural.ARM][k]
                    for k in ('metrics', 'errors', 'prediction_sha256'))
                and case['context'] == {k: old['primary'][k] for k in ('centralized_reference', 'full39')},
                'Historical tree/context bridge changed')
    require(r['summary'] == summarize(r['cases']), 'Summary changed')
    return dict(status='verified', models=3, examples_processed=48658680, optimizer_steps=95040,
                fit_rows=270326, audit_rows=408027, test_opened=False)


def verify_output(output, r):
    require(digest(output/'models-frozen.json') == r['models_frozen_sha256']
            and frozen.read(output/'models-frozen.json') == dict(status='all_models_frozen_before_audit',
                identity=r['identity'], models=r['models']), 'Frozen neural models changed')
    for spec in r['models']:
        require(Path(spec['file']).name == spec['file']
                and digest(output/spec['file']) == spec['file_sha256']
                and (output/spec['file']).stat().st_size == spec['file_bytes'], 'Saved neural weights changed')


def run(cache, packed, panel, trees, output):
    cache, packed, panel, trees, output = map(Path, (cache, packed, panel, trees, output))
    require(not output.exists(), 'Preserve existing output; use a fresh directory')
    p, ref = frozen.read(PLAN), frozen.read(REFERENCE)
    validate_plan(p)
    prior.audit_record(ref)
    prior.verify_output(trees, ref)
    require(digest(trees/'receipt.json') == ref['source_receipt_sha256'], 'Wrong tree source receipt')
    cache_ref = frozen.read(older.previous.PREVIOUS)
    older.previous.verify_private(cache, cache_ref)
    runtime, parent = frozen.configure('cpu'), frozen.load_plan()
    require(runtime == ref['identity']['runtime'], 'Use the existing local CPU runtime')
    manifest = frozen.verify_pack(packed, parent)
    sources = dict(ref['identity']['source_sha256'])
    sources.update({n: digest(ROOT/n) for n in SOURCES})
    identity = dict(plan=p, plan_sha256=digest(PLAN), source_sha256=sources, runtime=runtime)
    started = time.perf_counter()
    with frozen.session(output, identity, resume=False), threadpool_limits(limits=2):
        r = dict(status='fitting', identity=identity, common_panel=ref['common_panel'],
                 validation_panel=ref['validation_panel'], split=ref['split'], models=[], **dict.fromkeys(FLAGS, False))
        frozen.save(output/'receipt.json', r)
        with headroom.validation_only(packed, manifest) as (x, y):
            features = manifest['features']
            keep, info = headroom.load_panel(panel, SimpleNamespace(root=packed, arrays={'val': (x, y)}, manifest=manifest))
            require(info == ref['validation_panel'], 'Validation panel changed')
            rows = np.flatnonzero(keep)
            split, chosen, keys = [np.load(cache/n, allow_pickle=False)
                                   for n in ('partitions.npy', 'fit_rows.npy', 'group_keys.npy')]
            older.previous.fusion.check_split(rows, split, chosen, keys)
            fit_rows, population = rows[chosen], np.bincount(y[rows[split == 0]], minlength=8)
            audit_rows = np.load(trees/'retained_rows.npy', allow_pickle=False)
            # This fixed protocol reuses the completed zero-collision panel.
            require(np.array_equal(audit_rows, rows[split == 2]) and len(audit_rows) == 408027,
                    'Common audit population changed')
            for seed in p['seeds']:
                tree_spec = next(m for m in ref['models'] if m['seed'] == seed and m['arm'] == neural.ARM)
                r['models'].append(fit_one(output, x, y, fit_rows, population, features, seed, tree_spec))
                frozen.save(output/'receipt.json', r)
            frozen.save(output/'models-frozen.json', dict(status='all_models_frozen_before_audit', identity=identity, models=r['models']))
            r['models_frozen_sha256'] = digest(output/'models-frozen.json')
            r['cases'] = []
            for spec in r['models']:
                seed = spec['seed']
                model = neural.restore((output/spec['file']).read_bytes(), spec['file_sha256'], spec['final_state_sha256'])
                tree_spec = next(m for m in ref['models'] if m['seed'] == seed and m['arm'] == neural.ARM)
                tree = older.load_owned(trees, tree_spec)
                case = dict(seed=seed)
                for cohort, ids in [('fit', fit_rows), ('audit', audit_rows)]:
                    tp = prior.predict_rows(tree, x, ids, features, neural.ARM)
                    npred = predict_rows(model, x, ids, features)
                    labels = np.array(y[ids])
                    case[cohort] = dict(tree=scored(labels, tp, population), neural=scored(labels, npred, population))
                    case[cohort+'_paired_correctness'] = neural.paired_correctness(labels, tp, npred)
                old = next(c for c in ref['cases'] if c['seed'] == seed)
                case['context'] = {k: old['primary'][k] for k in ('centralized_reference', 'full39')}
                r['cases'].append(case)
                print(f'PASS: seed={seed} fixed-endpoint fit/audit scores recorded', flush=True)
            r['summary'] = summarize(r['cases'])
        older.previous.verify_private(cache, cache_ref)
        prior.verify_output(trees, ref)
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
    require(first.resolve() != replay.resolve() and not destination.exists(), 'Need fresh publication and independent replay')
    a, b = [frozen.read(p/'receipt.json') for p in (first, replay)]
    for path, record in ((first, a), (replay, b)):
        audit_record(record)
        verify_output(path, record)
    require(older.without_timing(a) == older.without_timing(b), 'Non-timing replay differs')
    record = dict(a, non_timing_replay_exact=True, source_receipt_sha256=digest(first/'receipt.json'),
                  replay_receipt_sha256=digest(replay/'receipt.json'))
    frozen.save(destination, record)
    return audit_record(record)


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
