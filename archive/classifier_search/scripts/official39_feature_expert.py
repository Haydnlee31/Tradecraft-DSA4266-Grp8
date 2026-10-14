"""Bounded local feature/score comparison; all models frozen before audit.

The public result is aggregate evidence. Private fitted estimators and any flow
arrays stay outside git. This is a centralized supervised extension, not a new
federated training algorithm, independent confirmation, or deployment approval.
"""
import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import sklearn
import torch
from threadpoolctl import threadpool_limits

from scripts import official39_score_fusion as previous
from src.eval import feature_expert as expert
from src.eval.official_evidence import describe, digest, metrics_from_counts, require

frozen, headroom = previous.frozen, previous.headroom
PLAN = ROOT/'reports/risk_aware_routing/feature-expert-plan.json'
REFERENCE = ROOT/'reports/risk_aware_routing/fusion-results.json'
DIVERSITY = ROOT/'reports/full_data_extension/diversity-results.json'
METHODS = previous.METHODS + list(expert.MODES)
TIMED = ['base', 'centralized_reference', 'fusion', 'features', 'scores', 'hybrid']
FALSE_FLAGS = ('neural_models_trained', 'human_review_required', 'human_review_performed',
               'test_opened', 'test_evaluated', 'cloud_authorized', 'external_api_called',
               'deployment_authorized', 'independent_confirmation')
SOURCES = ['src/eval/feature_expert.py', 'scripts/official39_feature_expert.py']


def validate_plan(p):
    require(p['protocol'] == 'official39-feature-expert-v1'
            and p['scope'] == 'reused_validation_exploratory_internal_audit'
            and p['seeds'] == [7, 17, 27] and p['base_lanes'] == ['iid', 'dirichlet']
            and p['modes'] == expert.MODES and p['classifier'] == expert.CONFIG
            and p['device'] == 'cpu' and p['threads'] == 2 and p['rare_class_expert_trained'] is True
            and all(p[k] is False for k in FALSE_FLAGS), 'Unreviewed feature-expert scope')
    require(p['fusion_plan_sha256'] == digest(previous.PLAN)
            and p['fusion_results_sha256'] == digest(REFERENCE)
            and p['prior_diversity_results_sha256'] == digest(DIVERSITY), 'Prior evidence changed')


def sample_inputs(mode, x, base_path, expert_path, rows):
    require(rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer)
            and len(rows) > 0 and np.all((rows >= 0) & (rows < len(x))), 'Invalid source indices')
    if mode == 'features':
        return expert.inputs(mode, x=np.array(x[rows]))
    with previous.mmap(base_path) as bp, previous.mmap(expert_path) as ep:
        require(bp.shape == ep.shape == (len(x), 8) and bp.dtype == ep.dtype == np.float32, 'Score cohort changed')
        return expert.inputs(mode, x=np.array(x[rows]) if mode == 'hybrid' else None,
                             base=np.array(bp[rows]), specialist=np.array(ep[rows]))


def fit_one(output, name, mode, seed, x, y, fit_rows, counts, bp, ep):
    require(not (output/(name+'.joblib')).exists(), 'Preserve fitted model')
    start = time.perf_counter()
    z = sample_inputs(mode, x, bp, ep, fit_rows)
    model, spec = expert.fit(z, np.array(y[fit_rows]), counts, mode, seed)
    raw = expert.model_bytes(model)
    path = output/(name+'.joblib')
    # These are generated private model artifacts, not source-code edits.
    with path.open('xb') as stream:
        stream.write(raw)
    require(digest(path) == spec['model_sha256'], 'Serialized model hash changed')
    restored = expert.restore_owned(path.read_bytes(), spec['model_sha256'], mode, seed)
    with threadpool_limits(limits=2):
        replay = expert.predict(restored, z)
    require(previous.prior.vector_hash(replay) == spec['training_prediction_sha256'], 'Saved-model prediction bridge differs')
    spec.update(file=path.name, model_bytes=len(raw), serialization_bridge_exact=True,
        original_fit_rows_sha256=previous.prior.vector_hash(fit_rows.astype('<i8')),
        fit_inputs_sha256=previous.prior.vector_hash(z), seconds=time.perf_counter()-start)
    print(f'FROZEN: {name}; {spec["trees"]} trees, {spec["inputs"]} inputs; {spec["seconds"]:.1f}s', flush=True)
    return spec


def load_owned(output, spec):
    require(Path(spec['file']).name == spec['file'] and spec['file'].endswith('.joblib'), 'Unsafe owned model name')
    path = output/spec['file']
    require(path.stat().st_size == spec['model_bytes'], 'Owned model size changed')
    return expert.restore_owned(path.read_bytes(), spec['model_sha256'], spec['mode'], spec['seed'])


def predict_rows(model, mode, x, rows, bp, ep):
    result = np.empty(len(rows), dtype=np.uint8)
    for start in range(0, len(rows), 8192):
        selected = rows[start:start+8192]
        z = sample_inputs(mode, x, bp, ep, selected)
        result[start:start+len(selected)] = expert.predict(model, z)
    return result


def score_pair(pair, reference, cache, fusion_root, output, x, y, rows):
    seed, lane = pair['seed'], pair['lane']
    old_pair = next(p for p in reference['prepared_pairs'] if p['seed'] == seed and p['lane'] == lane)
    # Recompute all seven historical controls from their pinned saved heads
    # and score caches. No fitting routine is called after the freeze.
    old = previous.evaluate_pair(old_pair, cache, fusion_root, rows, y, frozen.read(previous.PREVIOUS))
    expected = next(p for p in reference['pairs'] if p['seed'] == seed and p['lane'] == lane)
    require(old == expected, 'Linear-fusion historical comparison changed')
    methods = dict(old['methods'])
    base = previous.prior.get_predictions(cache, seed, lane, rows)
    balanced = previous.prior.get_predictions(cache, seed, 'balanced_panel', rows)
    ay = np.array(y[rows])
    bp, ep = cache/f'seed{seed}-{lane}-probabilities.npy', fusion_root/f'seed{seed}-specialist-probabilities.npy'
    for mode, spec in pair['models'].items():
        model = load_owned(output, spec)
        before = expert.state_hash(model)
        pred = predict_rows(model, mode, x, rows, bp, ep)
        require(expert.state_hash(model) == before, 'Audit modified expert state')
        methods[mode] = dict(**headroom.metrics(ay, pred),
            prediction_sha256=previous.prior.vector_hash(pred),
            changes=previous.change_counts(ay, base, balanced, pred))
    return dict(seed=seed, lane=lane, rows=len(rows), historical_controls_exact=True, methods=methods)


def meets_reference(treatment, reference):
    return dict(macro_f1_higher=treatment['macro_f1'] > reference['macro_f1'],
        both_rare_f1_higher=all(treatment['per_class'][c]['f1'] > reference['per_class'][c]['f1']
                              for c in ('Web-based', 'Brute Force')),
        benign_false_alerts_not_higher=treatment['benign_false_alert_rate'] <= reference['benign_false_alert_rate'])


def summarize(pairs):
    lanes, checks = {}, []
    expected = [(s, lane) for s in (7, 17, 27) for lane in ('iid', 'dirichlet')]
    require([(p['seed'], p['lane']) for p in pairs] == expected, 'Require six canonical seed/lane pairs')
    for lane in ('iid', 'dirichlet'):
        cases = [p for p in pairs if p['lane'] == lane]
        lanes[lane] = {}
        for name in METHODS:
            values = [p['methods'][name]['metrics'] for p in cases]
            lanes[lane][name] = dict(
                **{k: describe([v[k] for v in values]) for k in ('macro_f1', 'benign_false_alert_rate')},
                per_class={c: {k: describe([v['per_class'][c][k] for v in values])
                              for k in ('precision', 'recall', 'f1')} for c in expert.CLASSES})
    for p in pairs:
        metrics = {k: v['metrics'] for k, v in p['methods'].items()}
        checks.append(dict(seed=p['seed'], lane=p['lane'],
            hybrid_over_feature_macro_f1=metrics['hybrid']['macro_f1']-metrics['features']['macro_f1'],
            hybrid_over_score_macro_f1=metrics['hybrid']['macro_f1']-metrics['scores']['macro_f1'],
            hybrid_vs_centralized=meets_reference(metrics['hybrid'], metrics['centralized_reference']),
            features_vs_centralized=meets_reference(metrics['features'], metrics['centralized_reference'])))
    return dict(lanes=lanes, checks=checks,
        hybrid_screen_passed=all(c['hybrid_over_feature_macro_f1'] > 0 and c['hybrid_over_score_macro_f1'] > 0
                                and all(c['hybrid_vs_centralized'].values()) for c in checks),
        feature_only_screen_passed=all(all(c['features_vs_centralized'].values()) for c in checks),
        deployment_authorized=False, independent_confirmation=False)


def live_prediction(method, xb, neural, models, linear_heads):
    base, balanced, central = neural
    if method in ('base', 'centralized_reference', 'fusion'):
        return previous.predict_live(method, xb, base, balanced, central, linear_heads)
    if method == 'features':
        return expert.predict(models[method], expert.inputs(method, x=xb.numpy()))
    with torch.no_grad():
        bp, ep = base(xb).softmax(1).numpy(), balanced(xb).softmax(1).numpy()
    z = expert.inputs(method, x=xb.numpy() if method == 'hybrid' else None, base=bp, specialist=ep)
    return expert.predict(models[method], z)


def measure_latency(pair, x, fit_rows, neural, models, linear_heads):
    before = [frozen.state_digest(m) for m in neural]+[expert.state_hash(models[m]) for m in expert.MODES]
    for m in neural:
        m.eval()
    batches = []
    for size in (1, 512):
        xb = torch.from_numpy(np.array(x[fit_rows[:size]], dtype=np.float32))
        for _ in range(3):
            for method in TIMED:
                live_prediction(method, xb, neural, models, linear_heads)
        samples = {m: [] for m in TIMED}
        for repeat in range(30):
            order = TIMED[repeat % len(TIMED):]+TIMED[:repeat % len(TIMED)]
            for method in order:
                start = time.perf_counter_ns()
                result = live_prediction(method, xb, neural, models, linear_heads)
                duration = time.perf_counter_ns()-start
                require(result.shape == (size,), 'Wrong live output coverage')
                samples[method].append(duration/1e6)
        batches.append(dict(batch_size=size, methods={m: dict(samples_ms=v,
            median_batch_ms=float(np.median(v)), p95_batch_ms=float(np.percentile(v, 95)),
            median_us_per_row=float(np.median(v)*1000/size)) for m, v in samples.items()}))
    after = [frozen.state_digest(m) for m in neural]+[expert.state_hash(models[m]) for m in expert.MODES]
    require(before == after, 'Timing modified model state')
    return dict(seed=pair['seed'], lane=pair['lane'], weights_unchanged=True, batches=batches)


def verify_input_artifacts(cache, fusion_root, reference):
    require(digest(fusion_root/'receipt.json') == reference['source_receipt_sha256'], 'Original fusion receipt changed')
    previous.verify_output(fusion_root, reference)
    previous.verify_private(cache, frozen.read(previous.PREVIOUS))


def verify_output(output, r):
    locked = frozen.read(output/'models-frozen.json')
    require(digest(output/'models-frozen.json') == r['models_frozen_sha256']
            and locked['status'] == 'all_models_frozen_before_audit' and 'pairs' not in locked
            and locked['identity'] == r['identity'] and locked['prepared_pairs'] == r['prepared_pairs'], 'Model freeze changed')
    for p in r['prepared_pairs']:
        for spec in p['models'].values():
            require(Path(spec['file']).name == spec['file'] and (output/spec['file']).stat().st_size == spec['model_bytes']
                    and digest(output/spec['file']) == spec['model_sha256'], 'Owned model artifact changed')


def audit_record(r):
    require(r['status'] == 'complete' and r['rare_class_expert_trained'] is True
            and r['scope'] == 'reused_validation_exploratory_internal_audit'
            and all(r[k] is False for k in FALSE_FLAGS), 'Wrong expert experiment scope')
    validate_plan(r['identity']['plan'])
    require(r['identity']['plan_sha256'] == digest(PLAN)
            and all(digest(ROOT/n) == h for n, h in r['identity']['source_sha256'].items()), 'Expert source changed')
    ref = frozen.read(REFERENCE)
    previous.audit_record(ref)
    require(r['split'] == ref['split'] and r['validation_panel'] == ref['validation_panel'], 'Split changed')
    expected = [(s, lane) for s in (7, 17, 27) for lane in ('iid', 'dirichlet')]
    require([(p['seed'], p['lane']) for p in r['pairs']] == expected
            and [(p['seed'], p['lane']) for p in r['prepared_pairs']] == expected, 'Incomplete comparisons')
    unique, supports = {}, r['split']['partitions']['audit']['classes']
    for pair, prepared, old in zip(r['pairs'], r['prepared_pairs'], ref['pairs']):
        require(pair['rows'] == 408027 and pair['historical_controls_exact']
                and set(pair['methods']) == set(METHODS) and set(prepared['models']) == set(expert.MODES), 'Incomplete methods')
        for mode, spec in prepared['models'].items():
            require(spec['seed'] == pair['seed'] and spec['mode'] == mode and spec['config'] == expert.CONFIG
                    and spec['classes'] == expert.CLASSES and spec['inputs'] == expert.MODES[mode]
                    and spec['fit_rows'] == 270326 and spec['serialization_bridge_exact']
                    and spec['fit_sample_counts'] == list(r['split']['fit_sample_classes'].values())
                    and spec['fit_population_counts'] == list(r['split']['partitions']['fit']['classes'].values())
                    and spec['original_fit_rows_sha256'] == ref['prepared_pairs'][0]['heads']['fusion']['original_fit_rows_sha256']
                    and spec['iterations'] == 100 and spec['trees'] == 800
                    and 800 <= spec['leaves'] <= 12000 and spec['nodes'] == 2*spec['leaves']-800
                    and 1 <= spec['maximum_leaves_in_tree'] <= 15 and 0 < spec['model_bytes'] < 20_000_000, 'Fit protocol changed')
            mass = np.asarray(spec['weighted_class_mass'])
            target = np.sqrt(spec['fit_population_counts'])
            require(np.allclose(mass/mass.sum(), target/target.sum(), rtol=1e-9, atol=1e-12)
                    and np.isclose(mass.sum(), spec['fit_rows']), 'Fit objective changed')
            if spec['file'] in unique:
                require(unique[spec['file']] == spec, 'Shared feature-only model changed across lanes')
            unique[spec['file']] = spec
        for name in previous.METHODS:
            require(pair['methods'][name] == old['methods'][name], 'Historical control changed')
        base_cm = np.asarray(pair['methods']['base']['metrics']['confusion_matrix'])
        for name, value in pair['methods'].items():
            m, e = metrics_from_counts(value['metrics']['confusion_matrix'], supports)
            require(value['metrics'] == m and value['errors'] == e, 'Metrics changed')
            c = value['changes']
            require(sum(v['corrected'] for v in c['per_class'].values()) == c['corrected']
                    and sum(v['damaged'] for v in c['per_class'].values()) == c['damaged']
                    and 0 <= c['corrected']+c['damaged'] <= c['changed'] <= pair['rows']
                    and 0 <= c['third_label_correct'] <= c['third_label'] <= c['changed'], 'Correction/harm totals changed')
            for i, label in enumerate(expert.CLASSES):
                d = c['per_class'][label]
                require(0 <= d['corrected'] <= supports[label]-base_cm[i, i] and 0 <= d['damaged'] <= base_cm[i, i]
                        and m['confusion_matrix'][i][i] == base_cm[i, i]+d['corrected']-d['damaged'], 'Class corrections changed')
    require(len(unique) == 15 and r['summary'] == summarize(r['pairs']), 'Fit count/summary changed')
    require([(p['seed'], p['lane']) for p in r['latency']] == expected, 'Missing timing cases')
    for case in r['latency']:
        require(case['weights_unchanged'] and [b['batch_size'] for b in case['batches']] == [1, 512], 'Timing scope changed')
        for b in case['batches']:
            require(set(b['methods']) == set(TIMED), 'Missing timing control')
            for v in b['methods'].values():
                t = np.asarray(v['samples_ms'])
                require(t.shape == (30,) and np.isfinite(t).all() and np.all(t > 0)
                        and v['median_batch_ms'] == float(np.median(t))
                        and v['p95_batch_ms'] == float(np.percentile(t, 95))
                        and v['median_us_per_row'] == float(np.median(t)*1000/b['batch_size']), 'Timing aggregation changed')
    return dict(status='verified', unique_fits=15, trees_fitted=12000, audit_rows=408027, test_opened=False)


def run(cache, fusion_root, packed, panel, specialists, archives, output):
    cache, fusion_root, packed, panel, specialists, archives, output = map(Path,
        (cache, fusion_root, packed, panel, specialists, archives, output))
    require(not output.exists(), 'Preserve existing run; use fresh output')
    p, ref = frozen.read(PLAN), frozen.read(REFERENCE)
    validate_plan(p)
    previous.audit_record(ref)
    verify_input_artifacts(cache, fusion_root, ref)
    runtime, parent = frozen.configure('cpu'), frozen.load_plan()
    require(runtime == ref['identity']['runtime'] and sklearn.__version__ == ref['identity']['sklearn'], 'Use recorded local runtime')
    manifest = frozen.verify_pack(packed, parent)
    sp = frozen.read(previous.specialist.PLAN)
    previous.specialist.validate_plan(sp)
    old_specialists = previous.specialist.specialist_reference(specialists,
        dict(sp, packed_manifest_sha256=parent['data_identity']['packed_manifest_sha256']), manifest)
    sources = dict(ref['identity']['source_sha256'])
    sources.update({n: digest(ROOT/n) for n in SOURCES})
    identity = dict(plan=p, plan_sha256=digest(PLAN), runtime=runtime,
        sklearn=sklearn.__version__, joblib=joblib.__version__, source_sha256=sources)
    started = time.perf_counter()
    with frozen.session(output, identity, resume=False), threadpool_limits(limits=2):
        r = dict(status='preparing', scope=p['scope'], identity=identity,
            rare_class_expert_trained=True, **dict.fromkeys(FALSE_FLAGS, False))
        frozen.save(output/'receipt.json', r)
        with headroom.validation_only(packed, manifest) as (x, y):
            keep, info = headroom.load_panel(panel, SimpleNamespace(root=packed, arrays={'val': (x, y)}, manifest=manifest))
            require(info == ref['validation_panel'], 'Shared panel changed')
            rows = np.flatnonzero(keep)
            split = np.load(cache/'partitions.npy', allow_pickle=False)
            chosen = np.load(cache/'fit_rows.npy', allow_pickle=False)
            keys = np.load(cache/'group_keys.npy', allow_pickle=False)
            previous.fusion.check_split(rows, split, chosen, keys)
            fit_rows, audit_rows = rows[chosen], rows[split == 2]
            counts = np.bincount(y[rows[split == 0]], minlength=8)
            require(counts.tolist() == list(ref['split']['partitions']['fit']['classes'].values()), 'Fit distribution changed')
            r.update(split=ref['split'], validation_panel=info, prepared_pairs=[])
            print('PASS: same fit rows, weights, duplicate grouping and audit population; no final test', flush=True)
            for seed in p['seeds']:
                feature_spec = fit_one(output, f'seed{seed}-features', 'features', seed, x, y, fit_rows, counts, None, None)
                for lane in p['base_lanes']:
                    models = dict(features=feature_spec)
                    bp, ep = cache/f'seed{seed}-{lane}-probabilities.npy', fusion_root/f'seed{seed}-specialist-probabilities.npy'
                    for mode in ('scores', 'hybrid'):
                        models[mode] = fit_one(output, f'seed{seed}-{lane}-{mode}', mode, seed, x, y, fit_rows, counts, bp, ep)
                    r['prepared_pairs'].append(dict(seed=seed, lane=lane, models=models))
                    frozen.save(output/'receipt.json', r)
            r['status'] = 'all_models_frozen_before_audit'
            frozen.save(output/'models-frozen.json', r)
            locked = frozen.read(output/'models-frozen.json')
            r['models_frozen_sha256'] = digest(output/'models-frozen.json')
            r['pairs'] = []
            for pair in locked['prepared_pairs']:
                r['pairs'].append(score_pair(pair, ref, cache, fusion_root, output, x, y, audit_rows))
                print(f'PASS: scored frozen seed={pair["seed"]} {pair["lane"]}, historical controls exact', flush=True)
            r['summary'] = summarize(r['pairs'])
            candidates = headroom.select_candidates(parent)
            store = frozen.Artifacts(archives, candidates)
            r['latency'] = []
            try:
                for pair, old_pair in zip(locked['prepared_pairs'], ref['prepared_pairs']):
                    seed, lane = pair['seed'], pair['lane']
                    base = store.load(next(c for c in candidates if c['seed'] == seed and c['lane'] == lane), parent, 'cpu')[0]
                    central = store.load(next(c for c in candidates if c['seed'] == seed and c['lane'] == 'light'), parent, 'cpu')[0]
                    balanced = previous.specialist.load_specialist(specialists,
                        next(s for s in sp['specialists'] if s['seed'] == seed), old_specialists, manifest['features'])[0]
                    models = {m: load_owned(output, spec) for m, spec in pair['models'].items()}
                    linear = {m: previous.load_head(fusion_root, spec) for m, spec in old_pair['heads'].items()}
                    r['latency'].append(measure_latency(pair, x, fit_rows, (base, balanced, central), models, linear))
            finally:
                store.close()
        verify_input_artifacts(cache, fusion_root, ref)
        verify_output(output, r)
        require(frozen.verify_pack(packed, parent) == manifest and digest(panel/'receipt.json') == info['receipt_sha256']
                and all(digest(panel/n) == h for n, h in info['files'].items()), 'Input metadata changed')
        require(digest(specialists/'receipt.json') == sp['specialist_receipt_sha256']
                and all(digest(specialists/s['checkpoint']) == s['sha256'] for s in sp['specialists']), 'Frozen specialist changed')
        for name, h in {(c['artifact']['archive'], c['artifact']['archive_sha256']) for c in candidates}:
            require(digest(archives/name) == h, 'Archive changed')
        r.update(status='complete', elapsed_seconds=time.perf_counter()-started)
        audit_record(r)
        frozen.save(output/'receipt.json', r)
    return r


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k not in ('seconds', 'elapsed_seconds', 'latency', 'models_frozen_sha256')}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def publish(first, replay, destination):
    first, replay, destination = map(Path, (first, replay, destination))
    require(first.resolve() != replay.resolve() and not destination.exists(), 'Require separate replay and fresh publication')
    a, b = frozen.read(first/'receipt.json'), frozen.read(replay/'receipt.json')
    for root, r in ((first, a), (replay, b)):
        audit_record(r)
        verify_output(root, r)
    require(without_timing(a) == without_timing(b), 'Non-timing replay differs')
    result = dict(a, non_timing_replay_exact=True, source_receipt_sha256=digest(first/'receipt.json'),
                  replay_receipt_sha256=digest(replay/'receipt.json'))
    frozen.save(destination, result)
    return audit_record(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit_parser = commands.add_parser('run')
    for name in ('cache', 'fusion-root', 'packed', 'panel', 'specialists', 'archives', 'output'):
        fit_parser.add_argument('--'+name, type=Path, required=True)
    pub = commands.add_parser('publish')
    for name in ('first', 'replay', 'destination'):
        pub.add_argument('--'+name, type=Path, required=True)
    check = commands.add_parser('audit')
    check.add_argument('--record', type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    r = run(**args) if command == 'run' else publish(**args) if command == 'publish' else audit_record(frozen.read(args['record']))
    import json
    print(json.dumps(r['summary']['checks'] if command == 'run' else r, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
