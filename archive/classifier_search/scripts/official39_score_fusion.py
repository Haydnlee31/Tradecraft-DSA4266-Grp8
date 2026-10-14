"""Local automatic fusion of frozen generalist and specialist scores.

This extension is exploratory assisted inference, not a replacement for the
closed official39 test study. It has no test, cloud, API, abstention or review
mode. New heads and caches stay private; only aggregate evidence is published.
"""
import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
import torch
from threadpoolctl import threadpool_limits

from scripts import official39_routing_gate as prior
from scripts import official39_routing_gate_report as prior_report
from scripts.official39_policy_tradeoffs import verify_private
from src.eval import score_fusion as fusion
from src.eval.official_evidence import describe, digest, metrics_from_counts, require

frozen, headroom, specialist = prior.frozen, prior.headroom, prior.specialist
PLAN = ROOT/'reports/risk_aware_routing/fusion-plan.json'
PREVIOUS = ROOT/'reports/risk_aware_routing/router-results.json'
METHODS = ['base', 'specialist', 'mean_probability', 'base_recalibrated',
           'specialist_recalibrated', 'centralized_reference', 'fusion']
FALSE_FLAGS = ('neural_models_trained', 'human_review_required', 'human_review_performed',
               'abstention', 'test_opened', 'test_evaluated', 'cloud_authorized',
               'external_api_called', 'deployment_authorized')
SOURCES = ['src/eval/score_fusion.py', 'scripts/official39_score_fusion.py',
           'scripts/official39_policy_tradeoffs.py', 'scripts/official39_routing_gate_report.py']


def validate_plan(p):
    expected = dict(protocol='official39-score-fusion-v1',
        scope='reused_validation_exploratory_internal_audit', seeds=[7, 17, 27],
        base_lanes=['iid', 'dirichlet'], device='cpu', threads=2, neural_batch_size=512,
        classifier=fusion.CONFIG, controls=METHODS[:-1], fusion_heads_trained=True)
    require(all(p.get(k) == v for k, v in expected.items())
            and all(p.get(k) is False for k in FALSE_FLAGS), 'Unreviewed fusion scope')
    require(p['router_plan_sha256'] == digest(prior.PLAN)
            and p['router_results_sha256'] == digest(PREVIOUS)
            and p['specialist_plan_sha256'] == digest(specialist.PLAN), 'Prior evidence changed')


@contextmanager
def mmap(path):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        yield a
    finally:
        a._mmap.close()


def selected_scores(base_path, expert_path, rows, mode):
    require(mode in ('base', 'specialist', 'fusion'), 'Invalid head mode')
    paths = [base_path] if mode == 'base' else [expert_path] if mode == 'specialist' else [base_path, expert_path]
    with ExitStack() as stack:
        arrays = [stack.enter_context(mmap(p)) for p in paths]
        require(all(a.ndim == 2 and a.shape[1] == 8 and a.dtype == np.float32
                    and np.all((rows >= 0) & (rows < len(a))) for a in arrays), 'Wrong score cache schema')
        return fusion.inputs(*(np.array(a[rows]) for a in arrays))


def scan_specialist(model, x, y, features, output, seed, cache):
    """New soft-score cache, bridged to every previous hard prediction."""
    name = f'seed{seed}-specialist-probabilities.npy'
    path = output/name
    require(not path.exists(), 'Preserve score cache')
    before = frozen.state_digest(model)
    model.eval()
    p = np.lib.format.open_memmap(path, mode='w+', dtype='<f4', shape=(len(y), 8))
    offset, started = 0, time.perf_counter()
    try:
        with mmap(cache/f'seed{seed}-balanced_panel-predictions.npy') as old, torch.no_grad():
            require(old.shape == y.shape and old.dtype == np.uint8, 'Wrong specialist prediction cache')
            for xb, yb in frozen.batches(x, y, features, 'full39'):
                logits = model(torch.from_numpy(xb))
                end = offset+len(yb)
                p[offset:end] = logits.softmax(1).numpy()
                fusion.inputs(np.asarray(p[offset:end]))
                require(np.array_equal(logits.argmax(1).numpy(), old[offset:end])
                        and np.array_equal(p[offset:end].argmax(1), old[offset:end]), 'Specialist hard prediction changed')
                offset = end
        require(offset == len(y) and frozen.state_digest(model) == before, 'Neural coverage or state changed')
        p.flush()
    finally:
        p._mmap.close()
    return dict(seed=seed, file=name, sha256=digest(path), rows=len(y),
        model_state_sha256=before, hard_prediction_bridge_exact=True, weights_unchanged=True,
        seconds=time.perf_counter()-started)


def fit_head(output, name, base_path, expert_path, rows, y, counts, mode):
    started = time.perf_counter()
    z = selected_scores(base_path, expert_path, rows, mode)
    h = fusion.fit(z, np.array(y[rows]), counts)
    h.update(mode=mode, original_fit_rows_sha256=prior.vector_hash(rows.astype('<i8')))
    filename = name+'.json'
    frozen.save(output/filename, h)
    require(frozen.read(output/filename) == h, 'Head serialization changed')
    print(f'FROZEN: {name}; {h["iterations"]} iterations, {h["learned_parameters"]} learned parameters', flush=True)
    return dict(file=filename, sha256=digest(output/filename), mode=mode,
        inputs=h['inputs'], learned_parameters=h['learned_parameters'], iterations=h['iterations'],
        fit_rows=h['fit_rows'], original_fit_rows_sha256=h['original_fit_rows_sha256'],
        fit_sample_counts=h['fit_sample_counts'], fit_population_counts=h['fit_population_counts'],
        weighted_class_mass=h['weighted_class_mass'], sklearn_prediction_bridge_exact=True,
        seconds=time.perf_counter()-started)


def load_head(output, spec):
    require(Path(spec['file']).name == spec['file'] and digest(output/spec['file']) == spec['sha256'], 'Frozen head changed')
    h = frozen.read(output/spec['file'])
    fusion.validate_head(h)
    return h


def change_counts(y, base, expert, prediction):
    repaired, damaged = (base != y) & (prediction == y), (base == y) & (prediction != y)
    third = (prediction != base) & (prediction != expert)
    return dict(corrected=int(repaired.sum()), damaged=int(damaged.sum()),
        changed=int((base != prediction).sum()), third_label=int(third.sum()),
        third_label_correct=int((third & (prediction == y)).sum()),
        per_class={c: dict(corrected=int((repaired & (y == i)).sum()),
                           damaged=int((damaged & (y == i)).sum())) for i, c in enumerate(fusion.CLASSES)})


def evaluate_pair(pair, cache, output, audit_rows, y, historical):
    seed, lane = pair['seed'], pair['lane']
    with mmap(cache/f'seed{seed}-{lane}-probabilities.npy') as bp, \
            mmap(output/f'seed{seed}-specialist-probabilities.npy') as ep:
        base, expert = np.array(bp[audit_rows]), np.array(ep[audit_rows])
    ay = np.array(y[audit_rows])
    predictions = dict(base=base.argmax(1).astype(np.uint8), specialist=expert.argmax(1).astype(np.uint8),
        mean_probability=fusion.mean_prediction(base, expert),
        centralized_reference=prior.get_predictions(cache, seed, 'light', audit_rows))
    for mode, method in [('base', 'base_recalibrated'), ('specialist', 'specialist_recalibrated'), ('fusion', 'fusion')]:
        h = load_head(output, pair['heads'][mode])
        z = fusion.inputs(base) if mode == 'base' else fusion.inputs(expert) if mode == 'specialist' else fusion.inputs(base, expert)
        predictions[method] = fusion.predict(h, z)
    results = {name: dict(**headroom.metrics(ay, predictions[name]),
        prediction_sha256=prior.vector_hash(predictions[name]),
        changes=change_counts(ay, predictions['base'], predictions['specialist'], predictions[name])) for name in METHODS}
    old = next(r for r in historical['pairs'] if r['seed'] == seed and r['lane'] == lane)
    for name, old_result in [('base', old['base']), ('specialist', old['always']['balanced_panel']),
                             ('centralized_reference', old['always']['light'])]:
        require(results[name]['metrics'] == old_result['metrics'] and results[name]['errors'] == old_result['errors'],
                'Historical audit population/prediction bridge differs')
    return dict(seed=seed, lane=lane, rows=len(ay), historical_controls_exact=True, methods=results)


def summarize(pairs):
    """Equal-weight seed means; retain every paired value, never select a seed."""
    summary = {}
    for lane in ('iid', 'dirichlet'):
        cases = [r for r in pairs if r['lane'] == lane]
        require([r['seed'] for r in cases] == [7, 17, 27], 'Incomplete paired seeds')
        summary[lane] = {}
        for name in METHODS:
            metrics = [r['methods'][name]['metrics'] for r in cases]
            summary[lane][name] = dict(
                **{k: describe([m[k] for m in metrics]) for k in ('macro_f1', 'benign_false_alert_rate')},
                per_class={c: {k: describe([m['per_class'][c][k] for m in metrics])
                              for k in ('precision', 'recall', 'f1')} for c in fusion.CLASSES})
        summary[lane]['fusion_minus_control'] = {name: {
            k: describe([r['methods']['fusion']['metrics'][k]-r['methods'][name]['metrics'][k] for r in cases])
            for k in ('macro_f1', 'benign_false_alert_rate')} for name in METHODS[:-1]}
    checks = []
    for r in pairs:
        m, b = r['methods']['fusion']['metrics'], r['methods']['base']['metrics']
        checks.append(dict(seed=r['seed'], lane=r['lane'],
            macro_f1_beats_every_control=all(m['macro_f1'] > r['methods'][c]['metrics']['macro_f1'] for c in METHODS[:-1]),
            both_rare_f1_beat_base=all(m['per_class'][c]['f1'] > b['per_class'][c]['f1'] for c in ('Web-based', 'Brute Force')),
            false_alert_change=m['benign_false_alert_rate']-b['benign_false_alert_rate']))
    return dict(lanes=summary, screen=checks,
        continuation_screen_passed=all(c['macro_f1_beats_every_control'] and c['both_rare_f1_beat_base'] for c in checks),
        deployment_authorized=False, independent_confirmation=False)


def predict_live(method, x, base, expert, centralized, heads):
    """Actual in-memory inference path, not arithmetic on saved predictions."""
    def scores(model):
        return model(x).softmax(1).numpy()
    with torch.no_grad():
        if method == 'centralized_reference':
            return centralized(x).argmax(1).numpy()
        if method in ('base', 'specialist'):
            return (base if method == 'base' else expert)(x).argmax(1).numpy()
        if method == 'base_recalibrated':
            return fusion.predict(heads['base'], fusion.inputs(scores(base)))
        if method == 'specialist_recalibrated':
            return fusion.predict(heads['specialist'], fusion.inputs(scores(expert)))
        bp, ep = scores(base), scores(expert)
        if method == 'mean_probability':
            return fusion.mean_prediction(bp, ep)
        require(method == 'fusion', 'Unknown inference method')
        return fusion.predict(heads['fusion'], fusion.inputs(bp, ep))


def latency(pair, x, fit_rows, models, heads):
    """Measured local CPU path; explicitly not edge or network latency.

    Round-robin method ordering rotates between repeats to reduce ordering
    bias. Input copies, checkpoint I/O and external preprocessing are outside
    the timer. Validation of head and soft-score schema remains inside it.
    """
    base, expert, centralized = models
    before = [frozen.state_digest(m) for m in models]
    for model in models:
        model.eval()
    records = []
    for batch in (1, 512):
        xb = torch.from_numpy(np.array(x[fit_rows[:batch]], dtype=np.float32))
        for _ in range(3):
            for method in METHODS:
                predict_live(method, xb, base, expert, centralized, heads)
        times = {m: [] for m in METHODS}
        for repeat in range(30):
            order = METHODS[repeat % len(METHODS):]+METHODS[:repeat % len(METHODS)]
            for method in order:
                start = time.perf_counter_ns()
                result = predict_live(method, xb, base, expert, centralized, heads)
                elapsed = time.perf_counter_ns()-start
                require(result.shape == (batch,), 'Inference output coverage differs')
                times[method].append(elapsed/1e6)
        records.append(dict(batch_size=batch, methods={m: dict(
            median_batch_ms=float(np.median(v)), p95_batch_ms=float(np.percentile(v, 95)),
            median_us_per_row=float(np.median(v)*1000/batch), samples_ms=v) for m, v in times.items()}))
    require([frozen.state_digest(m) for m in models] == before, 'Timing changed frozen weights')
    return dict(seed=pair['seed'], lane=pair['lane'], weights_unchanged=True, batches=records)


def audit_record(r):
    require(r['status'] == 'complete' and r['fusion_heads_trained'] is True
            and r['scope'] == 'reused_validation_exploratory_internal_audit'
            and all(r[k] is False for k in FALSE_FLAGS), 'Wrong experiment scope')
    validate_plan(r['identity']['plan'])
    require(r['identity']['plan_sha256'] == digest(PLAN)
            and all(digest(ROOT/n) == h for n, h in r['identity']['source_sha256'].items()), 'Fusion source identity changed')
    previous = frozen.read(PREVIOUS)
    prior_report.audit(previous)
    require(r['split'] == previous['split'] and r['validation_panel'] == previous['validation_panel'], 'Cohort changed')
    require(len(r['cache_checks']) == 3 and [c['seed'] for c in r['cache_checks']] == [7, 17, 27]
            and all(c['hard_prediction_bridge_exact'] and c['weights_unchanged'] for c in r['cache_checks']), 'Missing specialist bridges')
    expected = [(s, lane) for s in (7, 17, 27) for lane in ('iid', 'dirichlet')]
    require([(p['seed'], p['lane']) for p in r['pairs']] == expected
            and [(p['seed'], p['lane']) for p in r['prepared_pairs']] == expected, 'Incomplete pairs')
    supports = r['split']['partitions']['audit']['classes']
    unique_heads = {}
    for prepared, pair in zip(r['prepared_pairs'], r['pairs']):
        require(pair['rows'] == 408027 and pair['historical_controls_exact']
                and set(prepared['heads']) == {'base', 'specialist', 'fusion'} and set(pair['methods']) == set(METHODS), 'Incomplete comparison')
        for mode, h in prepared['heads'].items():
            require(h['mode'] == mode and h['fit_rows'] == 270326 and h['sklearn_prediction_bridge_exact']
                    and h['inputs'] == (16 if mode == 'fusion' else 8)
                    and h['learned_parameters'] == (136 if mode == 'fusion' else 72)
                    and 0 < h['iterations'] <= fusion.CONFIG['max_iter']
                    and h['fit_sample_counts'] == list(r['split']['fit_sample_classes'].values())
                    and h['fit_population_counts'] == list(r['split']['partitions']['fit']['classes'].values()), 'Head fit scope changed')
            if h['file'] in unique_heads:
                require(unique_heads[h['file']] == h, 'Shared specialist head differs between lanes')
            unique_heads[h['file']] = h
        b = pair['methods']['base']['metrics']['confusion_matrix']
        old = next(o for o in previous['pairs'] if o['seed'] == pair['seed'] and o['lane'] == pair['lane'])
        for name, value in pair['methods'].items():
            m, e = metrics_from_counts(value['metrics']['confusion_matrix'], supports)
            require(m == value['metrics'] and e == value['errors'], 'Metrics changed')
            c = value['changes']
            require(sum(v['corrected'] for v in c['per_class'].values()) == c['corrected']
                    and sum(v['damaged'] for v in c['per_class'].values()) == c['damaged']
                    and 0 <= c['corrected']+c['damaged'] <= c['changed'] <= pair['rows']
                    and 0 <= c['third_label_correct'] <= c['third_label'] <= c['changed'], 'Invalid change totals')
            for i, label in enumerate(fusion.CLASSES):
                d = c['per_class'][label]
                require(0 <= d['corrected'] <= supports[label]-b[i][i] and 0 <= d['damaged'] <= b[i][i]
                        and m['confusion_matrix'][i][i] == b[i][i]+d['corrected']-d['damaged'], 'Per-class correction counts changed')
        for name, old_result in [('base', old['base']), ('specialist', old['always']['balanced_panel']),
                                 ('centralized_reference', old['always']['light'])]:
            require(pair['methods'][name]['metrics'] == old_result['metrics'], 'Historical control changed')
    require(len(unique_heads) == 15 and r['summary'] == summarize(r['pairs']), 'Head count or summary changed')
    require([(p['seed'], p['lane']) for p in r['latency']] == expected, 'Missing inference measurements')
    for case in r['latency']:
        require(case['weights_unchanged'] and [b['batch_size'] for b in case['batches']] == [1, 512], 'Wrong timing scope')
        for b in case['batches']:
            require(set(b['methods']) == set(METHODS), 'Missing latency control')
            for v in b['methods'].values():
                t = np.asarray(v['samples_ms'])
                require(t.shape == (30,) and np.isfinite(t).all() and np.all(t > 0)
                        and v['median_batch_ms'] == float(np.median(t))
                        and v['p95_batch_ms'] == float(np.percentile(t, 95))
                        and v['median_us_per_row'] == float(np.median(t)*1000/b['batch_size']), 'Latency aggregation changed')
    return dict(status='verified', pairs=6, fitted_heads=15, audit_rows=408027, test_opened=False)


def verify_output(output, r):
    locked = frozen.read(output/'heads-frozen.json')
    require(digest(output/'heads-frozen.json') == r['heads_frozen_sha256']
            and locked['status'] == 'all_heads_frozen_before_audit' and 'pairs' not in locked
            and locked['prepared_pairs'] == r['prepared_pairs'], 'Pre-audit head freeze changed')
    for pair in r['prepared_pairs']:
        for spec in pair['heads'].values():
            load_head(output, spec)
    for c in r['cache_checks']:
        require(Path(c['file']).name == c['file'] and digest(output/c['file']) == c['sha256'], 'Expert score cache changed')


def run(cache, packed, panel, specialists, archives, output):
    cache, packed, panel, specialists, archives, output = map(Path, (cache, packed, panel, specialists, archives, output))
    require(not output.exists(), 'Preserve existing evidence; use a fresh output')
    p, previous = frozen.read(PLAN), frozen.read(PREVIOUS)
    validate_plan(p)
    prior_report.audit(previous)
    verify_private(cache, previous)
    runtime, parent = frozen.configure('cpu'), frozen.load_plan()
    require(runtime == previous['identity']['runtime'], 'Use the recorded local CPU runtime')
    manifest = frozen.verify_pack(packed, parent)
    sp = frozen.read(specialist.PLAN)
    specialist.validate_plan(sp)
    old = specialist.specialist_reference(specialists, dict(sp,
        packed_manifest_sha256=parent['data_identity']['packed_manifest_sha256']), manifest)
    sources = dict(previous['identity']['source_sha256'])
    sources.update({n: digest(ROOT/n) for n in SOURCES})
    identity = dict(plan=p, plan_sha256=digest(PLAN), source_sha256=sources,
        runtime=runtime, sklearn=sklearn.__version__, processor=platform.processor())
    start = time.perf_counter()
    with frozen.session(output, identity, resume=False), threadpool_limits(limits=2):
        r = dict(status='preparing', scope=p['scope'], identity=identity,
            fusion_heads_trained=True, **dict.fromkeys(FALSE_FLAGS, False))
        frozen.save(output/'receipt.json', r)
        with headroom.validation_only(packed, manifest) as (x, y):
            keep, info = headroom.load_panel(panel, SimpleNamespace(root=packed, arrays={'val': (x, y)}, manifest=manifest))
            require(info == previous['validation_panel'], 'Validation panel changed')
            rows = np.flatnonzero(keep)
            split = np.load(cache/'partitions.npy', allow_pickle=False)
            chosen = np.load(cache/'fit_rows.npy', allow_pickle=False)
            keys = np.load(cache/'group_keys.npy', allow_pickle=False)
            fusion.check_split(rows, split, chosen, keys)
            fit_rows, audit_rows = rows[chosen], rows[split == 2]
            counts = np.bincount(y[rows[split == 0]], minlength=8)
            require(counts.tolist() == list(previous['split']['partitions']['fit']['classes'].values()), 'Fit cohort changed')
            r.update(split=previous['split'], validation_panel=info, cache_checks=[], prepared_pairs=[])
            print('PASS: unchanged grouped fit/audit split; no neural training or human review', flush=True)
            for seed in p['seeds']:
                model, _ = specialist.load_specialist(specialists, next(s for s in sp['specialists'] if s['seed'] == seed), old, manifest['features'])
                r['cache_checks'].append(scan_specialist(model, x, y, manifest['features'], output, seed, cache))
                print(f'PASS: seed={seed} specialist probability scan reproduces all historical predictions', flush=True)
                ep = output/f'seed{seed}-specialist-probabilities.npy'
                expert_head = fit_head(output, f'seed{seed}-specialist-head', ep, ep, fit_rows, y, counts, 'specialist')
                for lane in p['base_lanes']:
                    bp = cache/f'seed{seed}-{lane}-probabilities.npy'
                    heads = dict(specialist=expert_head)
                    for mode in ('base', 'fusion'):
                        heads[mode] = fit_head(output, f'seed{seed}-{lane}-{mode}-head', bp, ep, fit_rows, y, counts, mode)
                    r['prepared_pairs'].append(dict(seed=seed, lane=lane, heads=heads))
                del model
            r['status'] = 'all_heads_frozen_before_audit'
            frozen.save(output/'heads-frozen.json', r)
            locked = frozen.read(output/'heads-frozen.json')
            r['heads_frozen_sha256'] = digest(output/'heads-frozen.json')
            r['pairs'] = [evaluate_pair(pair, cache, output, audit_rows, y, previous) for pair in locked['prepared_pairs']]
            r['summary'] = summarize(r['pairs'])
            print('PASS: all six comparisons evaluated with frozen heads; starting local latency measurements', flush=True)
            candidates = headroom.select_candidates(parent)
            store = frozen.Artifacts(archives, candidates)
            r['latency'] = []
            try:
                for pair in locked['prepared_pairs']:
                    seed = pair['seed']
                    base = store.load(next(c for c in candidates if c['seed'] == seed and c['lane'] == pair['lane']), parent, 'cpu')[0]
                    central = store.load(next(c for c in candidates if c['seed'] == seed and c['lane'] == 'light'), parent, 'cpu')[0]
                    expert = specialist.load_specialist(specialists, next(s for s in sp['specialists'] if s['seed'] == seed), old, manifest['features'])[0]
                    hs = {m: load_head(output, spec) for m, spec in pair['heads'].items()}
                    r['latency'].append(latency(pair, x, fit_rows, (base, expert, central), hs))
            finally:
                store.close()
        verify_private(cache, previous)
        verify_output(output, r)
        require(frozen.verify_pack(packed, parent) == manifest
                and digest(panel/'receipt.json') == info['receipt_sha256']
                and all(digest(panel/n) == h for n, h in info['files'].items()), 'Input metadata changed')
        require(digest(specialists/'receipt.json') == sp['specialist_receipt_sha256']
                and all(digest(specialists/s['checkpoint']) == s['sha256'] for s in sp['specialists']), 'Specialist changed')
        for name, h in {(c['artifact']['archive'], c['artifact']['archive_sha256']) for c in candidates}:
            require(digest(archives/name) == h, 'Checkpoint archive changed')
        r.update(status='complete', elapsed_seconds=time.perf_counter()-start)
        audit_record(r)
        frozen.save(output/'receipt.json', r)
    return r


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k not in ('seconds', 'elapsed_seconds', 'latency', 'heads_frozen_sha256')}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def publish(first, replay, destination):
    first, replay, destination = map(Path, (first, replay, destination))
    require(first.resolve() != replay.resolve() and not destination.exists(), 'Use independent replay and new publication')
    a, b = frozen.read(first/'receipt.json'), frozen.read(replay/'receipt.json')
    for root, r in ((first, a), (replay, b)):
        audit_record(r)
        verify_output(root, r)
    require(without_timing(a) == without_timing(b), 'Non-timing replay differs')
    export = dict(a, non_timing_replay_exact=True, source_receipt_sha256=digest(first/'receipt.json'),
                  replay_receipt_sha256=digest(replay/'receipt.json'))
    frozen.save(destination, export)
    return audit_record(export)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run_parser = commands.add_parser('run')
    for name in ('cache', 'packed', 'panel', 'specialists', 'archives', 'output'):
        run_parser.add_argument('--'+name, type=Path, required=True)
    pub = commands.add_parser('publish')
    for name in ('first', 'replay', 'destination'):
        pub.add_argument('--'+name, type=Path, required=True)
    check = commands.add_parser('audit')
    check.add_argument('--record', type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    result = run(**args) if command == 'run' else publish(**args) if command == 'publish' else audit_record(frozen.read(args['record']))
    import json
    print(json.dumps(result['summary']['screen'] if command == 'run' else result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
