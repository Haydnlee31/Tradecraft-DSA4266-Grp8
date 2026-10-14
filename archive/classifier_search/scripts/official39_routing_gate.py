"""Local, single-design router fit with calibration frozen before internal audit.

The final test and cloud have no entry point here. Base/expert checkpoints stay
frozen. Reused validation is partitioned only for this exploratory gate study;
it does not become a new untouched test set by assigning it different names.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sklearn
import torch

from scripts import official39_routing_specialist as specialist
from src.eval import routing_gate as gate
from src.eval.official_evidence import describe, digest, metrics_from_counts, require

headroom, frozen = specialist.headroom, specialist.frozen
PLAN = ROOT/'reports/risk_aware_routing/router-plan.json'
PREVIOUS = ROOT/'reports/risk_aware_routing/specialist-results.json'
FALSE_FLAGS = ('base_models_trained', 'test_opened', 'test_evaluated', 'external_api_called',
               'cloud_authorized', 'deployment_authorized')


def validate_plan(p):
    expected = dict(protocol='official39-local-router-v1', seeds=[7, 17, 27], base_lanes=['iid', 'dirichlet'],
        fallback='balanced_panel', device='cpu', split_seed=20261011, minimum_rows_per_class_per_partition=25,
        fit_cap_per_class=50000, fit_sampling_seed=20261012, gate=gate.GATE_CONFIG,
        budgets_percent=[1, 5, 10], calibration_grid_basis_points=10, random_seed=20261014,
        guarded_controls=['confidence_balanced', 'random_balanced', 'confidence_centralized', 'random_centralized'],
        quota_only_controls=['confidence_balanced', 'confidence_centralized'])
    require(all(p.get(k) == v for k, v in expected.items()) and all(p.get(k) is False for k in FALSE_FLAGS)
            and p.get('model_trained') is True and p.get('learned_gate_trained') is True, 'Unreviewed router scope')
    for key, path in [('specialist_plan_sha256', specialist.PLAN), ('specialist_results_sha256', PREVIOUS),
                      ('specialist_runner_sha256', Path(specialist.__file__))]:
        require(p[key] == digest(path), 'Specialist evidence changed')


def vector_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def split_panel(x, y, rows, output, p):
    """O(N) small index vectors; no full feature matrix enters RAM."""
    keys = np.empty(len(rows), dtype='<u8')
    for start in range(0, len(rows), 8192):
        keys[start:start+8192] = gate.group_keys(x[rows[start:start+8192]], p['split_seed'])
    split = gate.partitions(keys)
    py = np.asarray(y[rows])
    info = {}
    for i, name in enumerate(('fit', 'calibration', 'audit')):
        selected = np.flatnonzero(split == i)
        counts = np.bincount(py[selected], minlength=8)
        require(np.all(counts >= p['minimum_rows_per_class_per_partition']), 'Insufficient class support; do not resplit')
        info[name] = dict(rows=len(selected), unique_groups=len(np.unique(keys[selected])),
                         classes=dict(zip(gate.CLASSES, map(int, counts))), indices_sha256=vector_hash(selected))
    chosen = gate.fit_rows(py, split, p['fit_cap_per_class'], p['fit_sampling_seed'])
    require(np.all(split[chosen] == 0), 'Fit sampler crossed partition')
    # Keep the deterministic split and actual capped fit indices as private
    # reproducibility artifacts. Never modify the source dataset or panel.
    for name, a in [('partitions.npy', split), ('fit_rows.npy', chosen), ('group_keys.npy', keys)]:
        np.save(output/name, a, allow_pickle=False)
    return split, chosen, dict(partitions=info, exact_vector_groups_stay_together=True,
        fit_sample_rows=len(chosen), fit_sample_classes=dict(zip(gate.CLASSES, map(int, np.bincount(py[chosen], minlength=8)))),
        files={n: digest(output/n) for n in ('partitions.npy', 'fit_rows.npy', 'group_keys.npy')})


def scan_cache(model, x, y, feature_names, output, name, prior, keep_probabilities):
    """Cache inference, not outcomes; all original predictions must replay."""
    before = frozen.state_digest(model)
    model.eval()
    pred = np.empty(len(y), np.uint8)
    path = output/f'{name}-probabilities.npy'
    probs = np.lib.format.open_memmap(path, mode='w+', dtype='<f4', shape=(len(y), 8)) if keep_probabilities else None
    offset = 0
    try:
        with torch.no_grad():
            for xb, yb in frozen.batches(x, y, feature_names, 'full39'):
                logits = model(torch.from_numpy(xb))
                require(torch.isfinite(logits).all().item(), 'Nonfinite frozen-model output')
                end = offset+len(yb)
                pred[offset:end] = logits.argmax(1).numpy()
                if probs is not None:
                    probs[offset:end] = logits.softmax(1).numpy()
                offset = end
        require(offset == len(y) and before == frozen.state_digest(model) == prior['model_state_sha256']
                and vector_hash(pred) == prior['prediction_sha256'], 'Frozen model replay failed')
    finally:
        if probs is not None:
            probs.flush()
            probs._mmap.close()
    np.save(output/f'{name}-predictions.npy', pred, allow_pickle=False)
    return dict(name=name, prediction_replay_exact=True, weights_unchanged=True,
                files={n: digest(output/n) for n in [f'{name}-predictions.npy']+
                       ([path.name] if keep_probabilities else [])})


def make_features(x, probs, original_rows):
    result = np.empty((len(original_rows), 47), dtype=np.float32)
    for start in range(0, len(original_rows), 8192):
        take = original_rows[start:start+8192]
        result[start:start+8192] = gate.features(x[take], probs[take])
    return result


def score_rows(tree, x, probs, original_rows):
    score = np.empty(len(original_rows), dtype=float)
    for start in range(0, len(original_rows), 8192):
        rows = original_rows[start:start+8192]
        score[start:start+8192] = gate.gate_score(tree, gate.features(x[rows], probs[rows]))
    return score


def get_predictions(output, seed, lane, rows):
    a = np.load(output/f'seed{seed}-{lane}-predictions.npy', mmap_mode='r', allow_pickle=False)
    try:
        return np.array(a[rows], copy=True)
    finally:
        a._mmap.close()


def random_scores(rows, seed):
    # Independent random-control score assigned before any gate fit/calibration.
    # Row position is not a gate input. This control is paired across fallbacks.
    return np.random.default_rng(seed).random(rows)


def prepare_pair(seed, lane, x, y, rows, split, chosen, output, p):
    name = f'seed{seed}-{lane}'
    fit, cal = rows[chosen], rows[split == 1]
    probs = np.load(output/f'{name}-probabilities.npy', mmap_mode='r', allow_pickle=False)
    try:
        z = make_features(x, probs, fit)
        tree = gate.fit_gate(z, np.array(y[fit]), get_predictions(output, seed, lane, fit),
                             get_predictions(output, seed, 'balanced_panel', fit))
        del z
        tree['feature_names'] = frozen.read(frozen.PLAN)['data_identity']['features'] + [f'base_softmax_{c}' for c in gate.CLASSES]
        frozen.save(output/f'{name}-gate.json', tree)
        scores = dict(learned=score_rows(tree, x, probs, cal), confidence=1-probs[cal].max(1),
                      random=random_scores(len(rows), p['random_seed']+seed)[split == 1])
    finally:
        probs._mmap.close()
    base, cy = get_predictions(output, seed, lane, cal), np.array(y[cal])
    policies = []
    for fallback, expert_lane in [('balanced', 'balanced_panel'), ('centralized', 'light')]:
        expert = get_predictions(output, seed, expert_lane, cal)
        for rule in (('learned', 'confidence', 'random') if fallback == 'balanced' else ('confidence', 'random')):
            result = gate.calibrate(scores[rule], cy, base, expert)
            for choice in result['policies']:
                policies.append(dict(rule=f'{rule}_{fallback}', kind='guarded', fallback=expert_lane,
                    budget_percent=choice['budget_percent'], selection=choice['guarded'],
                    calibration_rows=result['rows'], calibration_base_false_alerts=result['base_false_alerts']))
                if rule == 'confidence':
                    policies.append(dict(rule=f'{rule}_{fallback}', kind='quota_only', fallback=expert_lane,
                        budget_percent=choice['budget_percent'], selection=choice['quota'],
                        calibration_rows=result['rows'], calibration_base_false_alerts=result['base_false_alerts']))
    return dict(seed=seed, lane=lane, gate=f'{name}-gate.json', gate_sha256=digest(output/f'{name}-gate.json'),
                nodes=tree['nodes'], depth=tree['depth'], target_counts=tree['target_counts'], policies=policies)


def audit_pair(pair, x, y, rows, split, output, p):
    """The only caller that computes router audit outcomes; no fitting here."""
    seed, lane = pair['seed'], pair['lane']
    take = rows[split == 2]
    require(digest(output/pair['gate']) == pair['gate_sha256'], 'Frozen gate changed')
    tree = frozen.read(output/pair['gate'])
    probs = np.load(output/f'seed{seed}-{lane}-probabilities.npy', mmap_mode='r', allow_pickle=False)
    try:
        scores = dict(learned=score_rows(tree, x, probs, take), confidence=1-probs[take].max(1),
                      random=random_scores(len(rows), p['random_seed']+seed)[split == 2])
    finally:
        probs._mmap.close()
    ay, base = np.array(y[take]), get_predictions(output, seed, lane, take)
    expert = {key: get_predictions(output, seed, key, take) for key in ('balanced_panel', 'light')}
    result = dict(seed=seed, lane=lane, base=headroom.metrics(ay, base),
                  always={key: headroom.metrics(ay, val) for key, val in expert.items()}, policies=[])
    for policy in pair['policies']:
        rule = policy['rule'].split('_')[0]
        record = gate.evaluate_policy(ay, base, expert[policy['fallback']], scores[rule],
                                      policy['selection']['threshold'], policy['budget_percent'])
        metrics, errors = metrics_from_counts(record['confusion_matrix'], dict(zip(gate.CLASSES, map(int, np.bincount(ay, minlength=8)))))
        record.update(rule=policy['rule'], kind=policy['kind'], budget_percent=policy['budget_percent'],
                      threshold=policy['selection']['threshold'], metrics=metrics, errors=errors)
        result['policies'].append(record)
    return result


def summarize(pairs):
    result = {}
    for lane in ('iid', 'dirichlet'):
        rows = [p for p in pairs if p['lane'] == lane]
        require([p['seed'] for p in rows] == [7, 17, 27], 'Require every paired seed')
        combined = []
        for i, ref in enumerate(rows[0]['policies']):
            records = [r['policies'][i] for r in rows]
            require(all((r['rule'], r['kind'], r['budget_percent']) == (ref['rule'], ref['kind'], ref['budget_percent']) for r in records), 'Policy alignment changed')
            combined.append(dict(rule=ref['rule'], kind=ref['kind'], budget_percent=ref['budget_percent'],
                macro_f1=describe([r['metrics']['macro_f1'] for r in records]),
                benign_false_alert_rate=describe([r['metrics']['benign_false_alert_rate'] for r in records]),
                escalation_rate=describe([r['escalation_rate'] for r in records]),
                budget_respected=[r['budget_respected'] for r in records], benign_guard_respected=[r['benign_guard_respected'] for r in records],
                per_class_recall={c: describe([r['metrics']['per_class'][c]['recall'] for r in records]) for c in gate.CLASSES}))
        screen = []
        for budget in (1, 5, 10):
            passed = []
            for row in rows:
                at = {r['rule']: r for r in row['policies'] if r['kind'] == 'guarded' and r['budget_percent'] == budget}
                learned = at['learned_balanced']
                passed.append(bool(learned['budget_respected'] and learned['benign_guard_respected']
                    and learned['metrics']['macro_f1'] > row['base']['metrics']['macro_f1']
                    and all(learned['metrics']['per_class'][c]['recall'] > row['base']['metrics']['per_class'][c]['recall'] for c in ('Web-based', 'Brute Force'))
                    and all(learned['metrics']['macro_f1'] >= at[r]['metrics']['macro_f1'] for r in ('confidence_balanced', 'random_balanced'))))
            screen.append(dict(budget_percent=budget, paired_seed_passes=passed, all_seeds_pass=all(passed)))
        result[lane] = dict(policies=combined, continuation_screen=screen,
            base={k: describe([r['base']['metrics'][k] for r in rows]) for k in ('macro_f1', 'benign_false_alert_rate')})
    return result


def run(archives, packed, panel, specialists, output):
    output = Path(output)
    require(not output.exists(), 'Preserve existing run; use a fresh output')
    p, previous = frozen.read(PLAN), frozen.read(PREVIOUS)
    validate_plan(p)
    runtime, parent = frozen.configure('cpu'), frozen.load_plan()
    require(runtime == previous['identity']['runtime'], 'Use the recorded local CPU runtime')
    manifest = frozen.verify_pack(packed, parent)
    sp = frozen.read(specialist.PLAN)
    specialist.validate_plan(sp)
    old = specialist.specialist_reference(specialists, dict(sp, packed_manifest_sha256=parent['data_identity']['packed_manifest_sha256']), manifest)
    sources = dict(previous['identity']['source_sha256'])
    require(all(digest(ROOT/n) == h for n, h in sources.items()), 'Pinned source changed')
    sources.update({n: digest(ROOT/n) for n in ('scripts/official39_routing_gate.py', 'src/eval/routing_gate.py')})
    identity = dict(plan=p, plan_sha256=digest(PLAN), source_sha256=sources,
                    runtime=runtime, sklearn=sklearn.__version__, **dict.fromkeys(FALSE_FLAGS, False))
    started = time.perf_counter()
    with frozen.session(output, identity, resume=False):
        receipt = dict(status='preparing', identity=identity, scope='reused_validation_exploratory_internal_audit',
                       model_trained=True, learned_gate_trained=True, **dict.fromkeys(FALSE_FLAGS, False))
        frozen.save(output/'receipt.json', receipt)
        with headroom.validation_only(packed, manifest) as (x, y):
            keep, info = headroom.load_panel(panel, SimpleNamespace(root=Path(packed), arrays={'val': (x, y)}, manifest=manifest))
            require(info == previous['validation_panel'], 'Panel changed')
            rows = np.flatnonzero(keep)
            split, chosen, split_info = split_panel(x, y, rows, output, p)
            receipt.update(validation_panel=info, split=split_info, cache_checks=[], prepared_pairs=[])
            print('PASS: label-blind grouped split and bounded fit sample', flush=True)
            candidates = headroom.select_candidates(parent)
            store = frozen.Artifacts(archives, candidates)
            try:
                for seed in p['seeds']:
                    for lane in ('light', 'iid', 'dirichlet', 'balanced_panel'):
                        if lane == 'balanced_panel':
                            model, _ = specialist.load_specialist(specialists, next(s for s in sp['specialists'] if s['seed'] == seed), old, manifest['features'])
                        else:
                            candidate = next(c for c in candidates if c['seed'] == seed and c['lane'] == lane)
                            model, env, _ = store.load(candidate, parent, 'cpu')
                            require(env['input_transform']['panel'] == info, 'Model cohort changed')
                        prior = next(c for c in previous['model_checks'] if c['seed'] == seed and c['lane'] == lane)
                        receipt['cache_checks'].append(scan_cache(model, x, y, manifest['features'], output,
                            f'seed{seed}-{lane}', prior, lane in p['base_lanes']))
                        del model
                    for lane in p['base_lanes']:
                        receipt['prepared_pairs'].append(prepare_pair(seed, lane, x, y, rows, split, chosen, output, p))
                        print(f'FROZEN: seed={seed} {lane}, gate and calibration thresholds; no audit outcomes yet', flush=True)
            finally:
                store.close()
            receipt['status'] = 'frozen_before_internal_audit'
            frozen.save(output/'calibration.json', receipt)
            calibration_hash = digest(output/'calibration.json')
            # Read back the frozen record. Audit receives no estimator or fit
            # routine and does not modify thresholds after seeing its labels.
            locked = frozen.read(output/'calibration.json')
            require(len(locked['prepared_pairs']) == 6, 'Incomplete calibration freeze')
            for check in locked['cache_checks']:
                require(all(digest(output/n) == h for n, h in check['files'].items()), 'Inference cache changed')
            pairs = [audit_pair(pair, x, y, rows, split, output, p) for pair in locked['prepared_pairs']]
            receipt.update(pairs=pairs, summary=summarize(pairs), calibration_sha256=calibration_hash)
            require(digest(output/'calibration.json') == calibration_hash, 'Calibration changed during audit')
            for pair in locked['prepared_pairs']:
                require(digest(output/pair['gate']) == pair['gate_sha256'], 'Gate changed during audit')
            for check in locked['cache_checks']:
                require(all(digest(output/n) == h for n, h in check['files'].items()), 'Inference cache changed during audit')
        require(digest(PLAN) == identity['plan_sha256'] and all(digest(ROOT/n) == h for n, h in sources.items()), 'Plan/source changed')
        validate_plan(frozen.read(PLAN))
        require(all(digest(output/n) == h for n, h in split_info['files'].items()), 'Split changed')
        require(digest(Path(packed)/'manifest.json') == parent['data_identity']['packed_manifest_sha256']
                and digest(Path(packed)/'scaler.json') == parent['data_identity']['scaler_sha256']
                and digest(Path(panel)/'receipt.json') == info['receipt_sha256'], 'Input metadata changed')
        require(all(digest(Path(panel)/n) == h for n, h in info['files'].items()), 'Panel membership changed')
        require(digest(Path(specialists)/'receipt.json') == sp['specialist_receipt_sha256']
                and all(digest(Path(specialists)/s['checkpoint']) == s['sha256'] for s in sp['specialists']), 'Specialist artifacts changed')
        for name, value in {(c['artifact']['archive'], c['artifact']['archive_sha256']) for c in candidates}:
            require(digest(Path(archives)/name) == value, 'Checkpoint archive changed')
        receipt.update(status='complete', elapsed_seconds=time.perf_counter()-started)
        frozen.save(output/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archives', 'packed', 'panel', 'specialists', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    run(a.archives, a.packed, a.panel, a.specialists, a.output)
    print('COMPLETE: local router/internal audit; no cloud, external API or final test')


if __name__ == '__main__':
    main()
