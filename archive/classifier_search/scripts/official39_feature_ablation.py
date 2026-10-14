"""Local feature-only ablations with exact projection-overlap screening.

The old neural checkpoints and feature-expert evidence remain unchanged. This
runner has no final-test mode, neural-score inputs, API calls or cloud path.
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

from scripts import official39_feature_expert as prior
from src.eval import feature_ablation as ablation
from src.eval.official_evidence import describe, digest, metrics_from_counts, require

expert, frozen, headroom = prior.expert, prior.frozen, prior.headroom
PLAN = ROOT/'reports/risk_aware_routing/feature-ablation-plan.json'
REFERENCE = ROOT/'reports/risk_aware_routing/feature-expert-results.json'
FLAGS = ('neural_scores_used', 'neural_models_trained', 'hyperparameter_search',
         'human_review_required', 'test_opened', 'test_evaluated', 'cloud_authorized',
         'external_api_called', 'deployment_authorized', 'independent_confirmation')
SOURCES = ['src/eval/feature_ablation.py', 'scripts/official39_feature_ablation.py',
           'src/data/official_shortcut_audit.py']
hash_vector = prior.previous.prior.vector_hash


def validate_plan(p):
    require(p['protocol'] == 'official39-feature-ablation-v1'
            and p['scope'] == 'reused_validation_exploratory_internal_audit'
            and p['seeds'] == [7, 17, 27] and p['arms'] == ablation.ARMS
            and p['device'] == 'cpu' and p['threads'] == 2 and p['batch_size'] == 8192
            and all(p[k] is False for k in FLAGS), 'Unreviewed ablation scope')
    require(p['feature_expert_plan_sha256'] == digest(prior.PLAN)
            and p['feature_expert_results_sha256'] == digest(REFERENCE), 'Prior expert evidence changed')


def counts(y):
    return dict(zip(expert.CLASSES, map(int, np.bincount(y, minlength=len(expert.CLASSES)))))


def fit_one(root, x, y, rows, population_counts, features, arm, seed, old_spec):
    name = f'seed{seed}-{arm}.joblib'
    require(not (root/name).exists(), 'Preserve fitted model')
    started = time.perf_counter()
    z = ablation.project(x[rows], features, arm)
    model, spec = expert.fit(z, np.array(y[rows]), population_counts, 'features', seed)
    raw = expert.model_bytes(model)
    with (root/name).open('xb') as stream:
        stream.write(raw)
    restored = expert.restore_owned(raw, spec['model_sha256'], 'features', seed)
    with threadpool_limits(limits=2):
        replay = expert.predict(restored, z)
    require(hash_vector(replay) == spec['training_prediction_sha256'], 'Model serialization changed predictions')
    spec.update(file=name, model_bytes=len(raw), serialization_bridge_exact=True,
        original_fit_rows_sha256=hash_vector(rows.astype('<i8')), fit_inputs_sha256=hash_vector(z))
    if arm == 'full39':
        require(all(spec[k] == v for k, v in old_spec.items() if k not in ('file', 'seconds')),
                'Historical full39 fitted model differs')
    used = ablation.masked_splits(model, features, arm)
    require(used == 0, 'A masked feature appears in tree splits')
    spec.update(arm=arm, dropped_features=ablation.ARMS[arm], masked_feature_splits=used,
        historical_model_exact=arm == 'full39', seconds=time.perf_counter()-started)
    print(f'FROZEN: seed={seed} {arm}; 800 trees; {spec["seconds"]:.1f}s', flush=True)
    return spec


def predict_rows(model, x, rows, features, arm):
    before = expert.state_hash(model)
    result = np.empty(len(rows), dtype=np.uint8)
    for start in range(0, len(rows), 8192):
        selected = rows[start:start+8192]
        result[start:start+len(selected)] = expert.predict(model, ablation.project(x[selected], features, arm))
    require(expert.state_hash(model) == before and ablation.masked_splits(model, features, arm) == 0,
            'Scoring changed the model or leaked a masked feature')
    return result


def evaluate(y, pred):
    return dict(**headroom.metrics(y, pred), prediction_sha256=hash_vector(pred))


def summarize(cases):
    require([p['seed'] for p in cases] == [7, 17, 27], 'Require every paired seed')
    result = {}
    for population in ('primary', 'original_audit'):
        records = [c[population] for c in cases]
        methods = {}
        for name in ['centralized_reference', *ablation.ARMS]:
            values = [r[name]['metrics'] for r in records]
            methods[name] = {k: describe([m[k] for m in values])
                            for k in ('macro_f1', 'benign_false_alert_rate')}
            methods[name]['per_class'] = {c: {k: describe([m['per_class'][c][k] for m in values])
                for k in ('precision', 'recall', 'f1')} for c in expert.CLASSES}
        paired = {arm: {k: describe([r[arm]['metrics'][k]-r['full39']['metrics'][k] for r in records])
                  for k in ('macro_f1', 'benign_false_alert_rate')} for arm in list(ablation.ARMS)[1:]}
        screens = [dict(seed=case['seed'], arm=arm,
            **prior.meets_reference(case[population][arm]['metrics'],
                                    case[population]['centralized_reference']['metrics']))
                   for case in cases for arm in ablation.ARMS]
        result[population] = dict(methods=methods, masked_minus_full39=paired, versus_centralized=screens)
    return result


def verify_panel_record(p, reference):
    old = reference['split']
    require(p['fit_population_rows'] == old['partitions']['fit']['rows']
            and p['fit_sample_rows'] == 270326 and p['original_audit_rows'] == 408027
            and p['retained_rows']+p['excluded_rows'] == 408027 and p['retained_rows'] > 0
            and p['matches_to_sampled_fit_rows']+p['matches_only_to_unsampled_fit_rows'] == p['excluded_rows']
            and min(p[k] for k in ('excluded_rows', 'matches_to_sampled_fit_rows', 'matches_only_to_unsampled_fit_rows')) >= 0
            and 0 < p['joint_projection_unique_fit_keys'] <= p['fit_population_rows']
            and p['drop_features'] == ablation.ARMS['number_total_masked']
            and not p['labels_used_for_membership'] and not p['split_reassigned'], 'Invalid common panel')
    require(set(p['files']) == {'retained_rows.npy', 'excluded_rows.npy'}, 'Wrong panel files')
    for c in expert.CLASSES:
        require(p['retained_classes'][c] > 0 and p['excluded_classes'][c] >= 0
                and p['retained_classes'][c]+p['excluded_classes'][c] == old['partitions']['audit']['classes'][c],
                'Panel class coverage changed')
    require(sum(p['retained_classes'].values()) == p['retained_rows']
            and sum(p['excluded_classes'].values()) == p['excluded_rows'], 'Panel support changed')


def audit_record(r):
    require(r['status'] == 'complete' and all(r[k] is False for k in FLAGS), 'Wrong completed scope')
    validate_plan(r['identity']['plan'])
    ref = frozen.read(REFERENCE)
    prior.audit_record(ref)
    expected_sources = dict(ref['identity']['source_sha256'])
    expected_sources.update({n: digest(ROOT/n) for n in SOURCES})
    require(r['identity']['plan_sha256'] == digest(PLAN)
            and r['identity']['source_sha256'] == expected_sources
            and r['identity']['runtime'] == ref['identity']['runtime']
            and r['identity']['sklearn'] == ref['identity']['sklearn']
            and r['identity']['joblib'] == ref['identity']['joblib'], 'Runtime/source identity changed')
    require(r['split'] == ref['split'] and r['validation_panel'] == ref['validation_panel'], 'Original cohort changed')
    verify_panel_record(r['common_panel'], ref)
    expected = [(s, a) for s in (7, 17, 27) for a in ablation.ARMS]
    require([(s['seed'], s['arm']) for s in r['models']] == expected, 'Missing fixed-budget fits')
    for spec in r['models']:
        old = next(p['models']['features'] for p in ref['prepared_pairs'] if p['seed'] == spec['seed'])
        require(spec['config'] == expert.CONFIG and spec['mode'] == 'features' and spec['inputs'] == 39
                and spec['classes'] == expert.CLASSES and spec['fit_rows'] == 270326
                and spec['dropped_features'] == ablation.ARMS[spec['arm']]
                and spec['masked_feature_splits'] == 0 and spec['serialization_bridge_exact']
                and spec['iterations'] == 100 and spec['trees'] == 800
                and 800 <= spec['leaves'] <= 12000 and spec['nodes'] == 2*spec['leaves']-800
                and 1 <= spec['maximum_leaves_in_tree'] <= 15 and 0 < spec['model_bytes'] < 20_000_000
                and spec['historical_model_exact'] == (spec['arm'] == 'full39'), 'Fit work/mask changed')
        for k in ('fit_sample_counts', 'fit_population_counts', 'weighted_class_mass', 'original_fit_rows_sha256'):
            require(spec[k] == old[k], 'Fitting population/objective changed')
        if spec['arm'] == 'full39':
            require(all(spec[k] == v for k, v in old.items() if k not in ('file', 'seconds')), 'Historical model changed')
    require([c['seed'] for c in r['cases']] == [7, 17, 27], 'Missing paired scores')
    for case in r['cases']:
        old = next(p for p in ref['pairs'] if p['seed'] == case['seed'] and p['lane'] == 'iid')
        for cohort, support in [('primary', r['common_panel']['retained_classes']),
                                ('original_audit', ref['split']['partitions']['audit']['classes'])]:
            require(set(case[cohort]) == {'centralized_reference', *ablation.ARMS}, 'Missing scoring arm')
            for value in case[cohort].values():
                m, errors = metrics_from_counts(value['metrics']['confusion_matrix'], support)
                require(value['metrics'] == m and value['errors'] == errors, 'Class metrics changed')
        require(case['historical_audit_exact'], 'Missing full-audit bridge')
        for name, old_name in [('full39', 'features'), ('centralized_reference', 'centralized_reference')]:
            require(all(case['original_audit'][name][k] == old['methods'][old_name][k]
                        for k in ('metrics', 'errors', 'prediction_sha256')), 'Historical audit differs')
    require(r['summary'] == summarize(r['cases']), 'Summary changed')
    return dict(status='verified', fits=9, trees_fitted=7200,
                primary_rows=r['common_panel']['retained_rows'], excluded_rows=r['common_panel']['excluded_rows'],
                test_opened=False)


def verify_output(output, r):
    locked = frozen.read(output/'models-frozen.json')
    panel = frozen.read(output/'panel-frozen.json')
    require(digest(output/'models-frozen.json') == r['models_frozen_sha256']
            and locked == dict(status='all_models_frozen_before_audit', identity=r['identity'],
                               common_panel=r['common_panel'], models=r['models'])
            and digest(output/'panel-frozen.json') == r['panel_frozen_sha256']
            and panel == r['common_panel'], 'Frozen model/panel record changed')
    for name, value in r['common_panel']['files'].items():
        require(Path(name).name == name and digest(output/name) == value, 'Common population changed')
    for s in r['models']:
        require(Path(s['file']).name == s['file'] and (output/s['file']).stat().st_size == s['model_bytes']
                and digest(output/s['file']) == s['model_sha256'], 'Private fitted model changed')


def run(cache, packed, panel, output):
    cache, packed, panel, output = map(Path, (cache, packed, panel, output))
    require(not output.exists(), 'Preserve existing run; use fresh output')
    p, ref = frozen.read(PLAN), frozen.read(REFERENCE)
    validate_plan(p)
    prior.audit_record(ref)
    cache_ref = frozen.read(prior.previous.PREVIOUS)
    prior.previous.verify_private(cache, cache_ref)
    runtime, parent = frozen.configure('cpu'), frozen.load_plan()
    require(runtime == ref['identity']['runtime'] and prior.sklearn.__version__ == ref['identity']['sklearn']
            and prior.joblib.__version__ == ref['identity']['joblib'], 'Use historical local runtime')
    manifest = frozen.verify_pack(packed, parent)
    sources = dict(ref['identity']['source_sha256'])
    sources.update({n: digest(ROOT/n) for n in SOURCES})
    identity = dict(plan=p, plan_sha256=digest(PLAN), runtime=runtime,
        sklearn=prior.sklearn.__version__, joblib=prior.joblib.__version__, source_sha256=sources)
    start = time.perf_counter()
    with frozen.session(output, identity, resume=False), threadpool_limits(limits=2):
        r = dict(status='preparing', identity=identity, **dict.fromkeys(FLAGS, False))
        frozen.save(output/'receipt.json', r)
        with headroom.validation_only(packed, manifest) as (x, y):
            features = manifest['features']
            keep, info = headroom.load_panel(panel, SimpleNamespace(root=packed, arrays={'val': (x, y)}, manifest=manifest))
            require(info == ref['validation_panel'], 'Original panel changed')
            rows = np.flatnonzero(keep)
            split, chosen, keys = [np.load(cache/n, allow_pickle=False)
                                   for n in ('partitions.npy', 'fit_rows.npy', 'group_keys.npy')]
            prior.previous.fusion.check_split(rows, split, chosen, keys)
            fit_rows, population, audit = rows[chosen], rows[split == 0], rows[split == 2]
            population_counts = np.bincount(y[population], minlength=8)
            require(population_counts.tolist() == list(ref['split']['partitions']['fit']['classes'].values()), 'Fit prior changed')
            retained, excluded, common = ablation.collision_panel(x, features, population, fit_rows, audit, output/'vectors.sqlite')
            common.update(retained_classes=counts(y[retained]), excluded_classes=counts(y[excluded]), files={})
            for name, indices in [('retained_rows.npy', retained), ('excluded_rows.npy', excluded)]:
                np.save(output/name, indices.astype('<i8'), allow_pickle=False)
                common['files'][name] = digest(output/name)
            verify_panel_record(common, ref)
            frozen.save(output/'panel-frozen.json', common)
            r.update(common_panel=common, panel_frozen_sha256=digest(output/'panel-frozen.json'),
                     split=ref['split'], validation_panel=info, models=[])
            print(f'FROZEN PANEL: {len(retained):,} retained; {len(excluded):,} excluded; labels did not choose membership', flush=True)
            for seed in p['seeds']:
                old = next(case['models']['features'] for case in ref['prepared_pairs'] if case['seed'] == seed)
                for arm in ablation.ARMS:
                    r['models'].append(fit_one(output, x, y, fit_rows, population_counts, features, arm, seed, old))
                    frozen.save(output/'receipt.json', r)
            frozen.save(output/'models-frozen.json', dict(status='all_models_frozen_before_audit',
                identity=identity, common_panel=common, models=r['models']))
            r['models_frozen_sha256'] = digest(output/'models-frozen.json')
            r['cases'] = []
            selected = np.isin(audit, retained)
            for seed in p['seeds']:
                predictions = {'centralized_reference': prior.previous.prior.get_predictions(cache, seed, 'light', audit)}
                for spec in [s for s in r['models'] if s['seed'] == seed]:
                    model = prior.load_owned(output, spec)
                    predictions[spec['arm']] = predict_rows(model, x, audit, features, spec['arm'])
                case = dict(seed=seed,
                    original_audit={n: evaluate(np.array(y[audit]), v) for n, v in predictions.items()},
                    primary={n: evaluate(np.array(y[retained]), v[selected]) for n, v in predictions.items()})
                old = next(c for c in ref['pairs'] if c['seed'] == seed and c['lane'] == 'iid')
                require(all(case['original_audit']['full39'][k] == old['methods']['features'][k]
                            for k in ('metrics', 'errors', 'prediction_sha256')), 'Full39 audit bridge failed')
                case['historical_audit_exact'] = True
                r['cases'].append(case)
                print(f'PASS: seed={seed} full39 historical bridge; all arms scored on both fixed populations', flush=True)
            r['summary'] = summarize(r['cases'])
        prior.previous.verify_private(cache, cache_ref)
        require(frozen.verify_pack(packed, parent) == manifest and digest(panel/'receipt.json') == info['receipt_sha256']
                and all(digest(panel/n) == h for n, h in info['files'].items()), 'Input cohort changed')
        verify_output(output, r)
        r.update(status='complete', elapsed_seconds=time.perf_counter()-start)
        audit_record(r)
        frozen.save(output/'receipt.json', r)
    return audit_record(r)


def publish(first, replay, destination):
    first, replay, destination = map(Path, (first, replay, destination))
    require(first.resolve() != replay.resolve() and not destination.exists(), 'Require independent replay and fresh publication')
    a, b = frozen.read(first/'receipt.json'), frozen.read(replay/'receipt.json')
    for root, r in ((first, a), (replay, b)):
        audit_record(r)
        verify_output(root, r)
    require(prior.without_timing(a) == prior.without_timing(b), 'Non-timing replay differs')
    result = dict(a, non_timing_replay_exact=True, source_receipt_sha256=digest(first/'receipt.json'),
                  replay_receipt_sha256=digest(replay/'receipt.json'))
    frozen.save(destination, result)
    return audit_record(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('run')
    for name in ('cache', 'packed', 'panel', 'output'):
        fit.add_argument('--'+name, type=Path, required=True)
    pub = commands.add_parser('publish')
    for name in ('first', 'replay', 'destination'):
        pub.add_argument('--'+name, type=Path, required=True)
    check = commands.add_parser('audit')
    check.add_argument('--record', type=Path, required=True)
    args = vars(parser.parse_args())
    cmd = args.pop('command')
    value = run(**args) if cmd == 'run' else publish(**args) if cmd == 'publish' else audit_record(frozen.read(args['record']))
    import json
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
