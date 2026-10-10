"""Frozen-model policy sensitivity, with automatic decisions and review separate.

This is explicitly AFTER the earlier internal audit. Its zero-extra-alert
failure remains a failure. New alert tolerances are illustrative research
scenarios, not permission to deploy or to claim an independent confirmation.
Only saved predictions and trees are used: there is no model-fitting call,
neural forward pass, final-test input or cloud action in this runner.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts import official39_routing_gate as prior
from scripts import official39_routing_gate_report as prior_report
from src.eval.official_evidence import describe, digest, metrics_from_counts, require

gate, frozen, headroom = prior.gate, prior.frozen, prior.headroom
PLAN = ROOT/'reports/risk_aware_routing/tradeoff-plan.json'
PREVIOUS = ROOT/'reports/risk_aware_routing/router-results.json'
RULES = ['learned_balanced', 'confidence_balanced', 'random_balanced', 'confidence_centralized', 'random_centralized']
FLAGS = ('model_trained', 'learned_gate_trained', 'neural_models_executed', 'test_opened', 'test_evaluated',
         'external_api_called', 'cloud_authorized', 'human_review_performed', 'deployment_authorized', 'operational_budget_approved')


def validate_plan(p):
    expected = dict(protocol='official39-policy-tradeoffs-v1', seeds=[7, 17, 27], base_lanes=['iid', 'dirichlet'],
        device='cpu', budgets_percent=[1, 5, 10], extra_false_alert_basis_points=[0, 50, 100, 200],
        automatic_rules=RULES, review_rules=['learned', 'confidence', 'random'], calibration_thresholds_selected=True)
    require(all(p.get(k) == v for k, v in expected.items()) and all(p.get(k) is False for k in FLAGS), 'Unreviewed policy scope')
    require(p['router_plan_sha256'] == digest(prior.PLAN) and p['router_results_sha256'] == digest(PREVIOUS), 'Prior evidence changed')


def candidate_grid(scores, y, base, expert):
    """Reuse the historical grid without editing its byte-pinned implementation.

    Prefix confusion updates are bounded by the grid size, not one full
    rescore for every threshold. Whole ties are never split by row identifiers.
    Synthetic tests and a real zero-slack bridge guard semantic equivalence.
    """
    gate.labels_ok(y, base, expert)
    require(scores.shape == y.shape and np.isfinite(scores).all(), 'Invalid calibration scores')
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    options, quotas = {0: None}, {}
    for bp in range(10, 1001, 10):
        cap = len(y)*bp//10000
        count, threshold = 0, None
        if cap:
            boundary = sorted_scores[cap-1]
            end = int(np.searchsorted(-sorted_scores, -boundary, side='right'))
            count = end if end <= cap else int(np.searchsorted(-sorted_scores, -boundary, side='left'))
            threshold = float(sorted_scores[count-1]) if count else None
        options[count] = threshold
        if bp in (100, 500, 1000):
            # JSON object keys are strings; keep the in-memory receipt in
            # the same form so publication can verify an exact round trip.
            quotas[str(bp//100)] = count
    cm = gate.confusion(y, base)
    result = dict(rows=len(y), benign_rows=int((y == gate.BENIGN_ID).sum()), base_false_alerts=gate.false_alerts(cm),
                  base_macro_f1=gate.macro_f1(cm), candidates=[], quotas=quotas)
    require(result['benign_rows'] > 0, 'Calibration must contain benign flows')
    offset = 0
    for count, threshold in sorted(options.items()):
        added = order[offset:count]
        if len(added):
            cm += gate.confusion(y[added], expert[added])-gate.confusion(y[added], base[added])
        offset = count
        result['candidates'].append(dict(threshold=threshold, calls=count, macro_f1=gate.macro_f1(cm), false_alerts=gate.false_alerts(cm)))
    return result


def choose(grid, budget, extra_bp):
    require(budget in (1, 5, 10) and extra_bp in (0, 50, 100, 200), 'Unreviewed budget')
    allowance = grid['benign_rows']*extra_bp//10000
    allowed = [r for r in grid['candidates'] if r['calls'] <= grid['rows']*budget//100
               and r['false_alerts'] <= grid['base_false_alerts']+allowance]
    require(allowed, 'Zero-call candidate missing')
    return dict(max(allowed, key=lambda r: (r['macro_f1'], -r['calls'])))


def review_choice(grid, budget):
    # Grid counts depend on scores only. Unlike automatic relabelling, review
    # quota selection must not optimize against any calibration labels.
    require(budget in (1, 5, 10), 'Unreviewed review budget')
    record = next(r for r in grid['candidates'] if r['calls'] == grid['quotas'][str(budget)])
    return dict(threshold=record['threshold'], calls=record['calls'])


def review_counts(y, base, scores, threshold, budget):
    """A referral is unresolved, not a correct attack prediction.

    The saved classifier labels are NOT replaced with ground truth or with an
    expert label. We count queue composition and which existing mistakes enter
    it; any actual reviewer success, time or security outcome is unknown.
    """
    gate.labels_ok(y, base)
    require(scores.shape == y.shape and budget in (1, 5, 10), 'Invalid review inputs')
    route = gate.decisions(scores, threshold)
    calls = int(route.sum())
    per_class = {}
    for c, name in enumerate(gate.CLASSES):
        actual = y == c
        selected = actual & route
        support = int(actual.sum())
        require(support > 0, 'All classes required for review reporting')
        per_class[name] = dict(support=support, referred=int(selected.sum()),
            review_coverage=float(selected.sum()/support),
            base_errors_referred=int((selected & (base != y)).sum()),
            base_errors_not_referred=int((actual & ~route & (base != y)).sum()),
            attacks_predicted_benign_referred=int((selected & (base == gate.BENIGN_ID)).sum()) if c != gate.BENIGN_ID else 0,
            attacks_predicted_benign_not_referred=int((actual & ~route & (base == gate.BENIGN_ID)).sum()) if c != gate.BENIGN_ID else 0)
    return dict(rows=len(y), calls=calls, review_rate=calls/len(y), reviews_per_10000_flows=10000*calls/len(y),
        budget_respected=calls <= len(y)*budget//100,
        base_errors_referred=sum(v['base_errors_referred'] for v in per_class.values()),
        benign_referrals=per_class[gate.BENIGN]['referred'], per_class=per_class,
        classification_labels_changed=False, human_review_performed=False, successful_reviews=None)


def verify_private(root, previous):
    root = Path(root)
    require(digest(root/'receipt.json') == previous['source_receipt_sha256']
            and digest(root/'calibration.json') == previous['calibration_sha256'], 'Prior private receipt changed')
    for check in previous['cache_checks']:
        for name, value in check['files'].items():
            require(Path(name).name == name and digest(root/name) == value, 'Prediction cache changed')
    for pair in previous['prepared_pairs']:
        require(pair['gate'] == f'seed{pair["seed"]}-{pair["lane"]}-gate.json'
                and digest(root/pair['gate']) == pair['gate_sha256'], 'Frozen tree changed')
    for name, value in previous['split']['files'].items():
        require(name in ('partitions.npy', 'fit_rows.npy', 'group_keys.npy') and digest(root/name) == value, 'Original grouping changed')


def scores_for(pair, x, rows, split, part, root, old_plan):
    seed, lane = pair['seed'], pair['lane']
    probs = np.load(root/f'seed{seed}-{lane}-probabilities.npy', mmap_mode='r', allow_pickle=False)
    take = rows[split == part]
    try:
        tree = frozen.read(root/pair['gate'])
        return dict(learned=prior.score_rows(tree, x, probs, take), confidence=1-probs[take].max(1),
                    random=prior.random_scores(len(rows), old_plan['random_seed']+seed)[split == part])
    finally:
        probs._mmap.close()


def prepare_pair(pair, x, y, rows, split, root, old_plan):
    seed, lane = pair['seed'], pair['lane']
    take = rows[split == 1]
    scores = scores_for(pair, x, rows, split, 1, root, old_plan)
    cy, base = np.array(y[take]), prior.get_predictions(root, seed, lane, take)
    policies, reviews, grids = [], [], {}
    for rule in RULES:
        method, fallback = rule.split('_')
        expert_lane = 'balanced_panel' if fallback == 'balanced' else 'light'
        grid = candidate_grid(scores[method], cy, base, prior.get_predictions(root, seed, expert_lane, take))
        grids[rule] = grid
        for budget in (1, 5, 10):
            old = next(p for p in pair['policies'] if p['rule'] == rule and p['kind'] == 'guarded' and p['budget_percent'] == budget)
            require(choose(grid, budget, 0) == old['selection'], 'Zero-slack calibration bridge changed')
            for extra_bp in (0, 50, 100, 200):
                policies.append(dict(rule=rule, fallback=expert_lane, budget_percent=budget, extra_false_alert_bp=extra_bp,
                    selection=choose(grid, budget, extra_bp), calibration_benign_rows=grid['benign_rows'],
                    calibration_rows=grid['rows'], calibration_base_false_alerts=grid['base_false_alerts']))
        if fallback == 'balanced':
            for budget in (1, 5, 10):
                reviews.append(dict(rule=method, budget_percent=budget, selection=review_choice(grid, budget)))
    return dict(seed=seed, lane=lane, zero_slack_calibration_exact=True, grids=grids, policies=policies, review_policies=reviews)


def score_pair(locked, original, x, y, rows, split, root, old_plan, old_pair):
    seed, lane = locked['seed'], locked['lane']
    take = rows[split == 2]
    scores = scores_for(original, x, rows, split, 2, root, old_plan)
    ay, base = np.array(y[take]), prior.get_predictions(root, seed, lane, take)
    expert = {k: prior.get_predictions(root, seed, k, take) for k in ('balanced_panel', 'light')}
    supports = dict(zip(gate.CLASSES, map(int, np.bincount(ay, minlength=8))))
    result = dict(seed=seed, lane=lane, base=headroom.metrics(ay, base), automatic=[], reviews=[], zero_slack_audit_exact=True)
    require(result['base'] == old_pair['base'], 'Base audit population changed')
    for p in locked['policies']:
        method = p['rule'].split('_')[0]
        r = gate.evaluate_policy(ay, base, expert[p['fallback']], scores[method], p['selection']['threshold'], p['budget_percent'])
        m, e = metrics_from_counts(r['confusion_matrix'], supports)
        if p['extra_false_alert_bp'] == 0:
            old = next(o for o in old_pair['policies'] if o['rule'] == p['rule'] and o['kind'] == 'guarded' and o['budget_percent'] == p['budget_percent'])
            require(all(old[k] == v for k, v in r.items()) and m == old['metrics'] and e == old['errors'], 'Zero-slack audit bridge changed')
        extra = r['false_alerts']-old_pair['base']['errors'][gate.BENIGN]['false_alerts']
        result['automatic'].append(dict(rule=p['rule'], budget_percent=p['budget_percent'], extra_false_alert_bp=p['extra_false_alert_bp'],
            threshold=p['selection']['threshold'], metrics=m, errors=e, calls=r['calls'], rows=r['rows'],
            escalation_rate=r['escalation_rate'], budget_respected=r['budget_respected'],
            extra_false_alerts=extra, extra_false_alerts_per_10000_benign=10000*extra/supports[gate.BENIGN],
            alert_limit_respected=extra <= supports[gate.BENIGN]*p['extra_false_alert_bp']//10000,
            corrected_rows=r['corrected_rows'], damaged_rows=r['damaged_rows'], per_class_routing=r['per_class_routing']))
    for p in locked['review_policies']:
        result['reviews'].append(dict(rule=p['rule'], budget_percent=p['budget_percent'], threshold=p['selection']['threshold'],
            **review_counts(ay, base, scores[p['rule']], p['selection']['threshold'], p['budget_percent'])))
    return result


def summarize(pairs):
    require([(r['seed'], r['lane']) for r in pairs] ==
            [(s, lane) for s in (7, 17, 27) for lane in ('iid', 'dirichlet')], 'Incomplete paired scenarios')
    summary = {}
    for lane in ('iid', 'dirichlet'):
        rows = [r for r in pairs if r['lane'] == lane]
        require([r['seed'] for r in rows] == [7, 17, 27], 'All paired seeds required')
        auto, review = [], []
        require(all(len(r['automatic']) == 60 and len(r['reviews']) == 9 for r in rows), 'Missing scenarios')
        for i, reference in enumerate(rows[0]['automatic']):
            records = [r['automatic'][i] for r in rows]
            key = ('rule', 'budget_percent', 'extra_false_alert_bp')
            require(all(all(r[k] == reference[k] for k in key) for r in records), 'Unpaired automatic policies')
            auto.append(dict(**{k: reference[k] for k in key},
                **{k: describe([r['metrics'][k] for r in records]) for k in ('macro_f1', 'benign_false_alert_rate')},
                **{k: describe([r[k] for r in records]) for k in ('escalation_rate', 'extra_false_alerts_per_10000_benign', 'corrected_rows', 'damaged_rows')},
                budget_respected=[r['budget_respected'] for r in records], alert_limit_respected=[r['alert_limit_respected'] for r in records],
                per_class_recall={c: describe([r['metrics']['per_class'][c]['recall'] for r in records]) for c in gate.CLASSES}))
        for i, reference in enumerate(rows[0]['reviews']):
            records = [r['reviews'][i] for r in rows]
            require(all((r['rule'], r['budget_percent']) == (reference['rule'], reference['budget_percent']) for r in records), 'Unpaired review policies')
            review.append(dict(rule=reference['rule'], budget_percent=reference['budget_percent'],
                **{k: describe([r[k] for r in records]) for k in ('review_rate', 'reviews_per_10000_flows', 'base_errors_referred', 'benign_referrals')},
                budget_respected=[r['budget_respected'] for r in records],
                per_class_review_coverage={c: describe([r['per_class'][c]['review_coverage'] for r in records]) for c in gate.CLASSES}))
        summary[lane] = dict(automatic=auto, review=review)
    return summary


def audit(record):
    """Recompute aggregate metrics and workloads without opening flow arrays."""
    p, previous = frozen.read(PLAN), frozen.read(PREVIOUS)
    validate_plan(p)
    prior_report.audit(previous)
    require(record['status'] == 'complete' and record['scope'] == p['scope']
            and all(record[k] is False for k in FLAGS) and record['calibration_thresholds_selected'] is True
            and record['operating_point_selected'] is False and record['zero_slack_bridge_exact'] is True, 'Wrong result scope')
    identity = record['identity']
    expected_sources = dict(previous['identity']['source_sha256'])
    expected_sources.update({n: digest(ROOT/n) for n in ('scripts/official39_routing_gate_report.py', 'scripts/official39_policy_tradeoffs.py')})
    require(identity['plan'] == p and identity['plan_sha256'] == digest(PLAN)
            and identity['source_sha256'] == expected_sources and identity['runtime'] == previous['identity']['runtime']
            and identity['prior_private_receipt_sha256'] == previous['source_receipt_sha256'], 'Plan/source identity changed')
    require(record['split'] == previous['split'] and record['validation_panel'] == previous['validation_panel'], 'Population changed')
    expected = [(s, lane) for s in (7, 17, 27) for lane in ('iid', 'dirichlet')]
    require([(r['seed'], r['lane']) for r in record['prepared']] == expected
            and [(r['seed'], r['lane']) for r in record['pairs']] == expected, 'Incomplete pairs')
    cal = record['split']['partitions']['calibration']
    supports = record['split']['partitions']['audit']['classes']
    rows = sum(supports.values())
    keys = [(r, b, e) for r in RULES for b in (1, 5, 10) for e in (0, 50, 100, 200)]
    review_keys = [(r, b) for r in ('learned', 'confidence', 'random') for b in (1, 5, 10)]
    for locked, pair, old, old_locked in zip(record['prepared'], record['pairs'], previous['pairs'], previous['prepared_pairs']):
        require(locked['zero_slack_calibration_exact'] is True and pair['zero_slack_audit_exact'] is True
                and pair['base'] == old['base'], 'Historical bridge changed')
        require([(r['rule'], r['budget_percent'], r['extra_false_alert_bp']) for r in locked['policies']] == keys
                and [(r['rule'], r['budget_percent'], r['extra_false_alert_bp']) for r in pair['automatic']] == keys
                and [(r['rule'], r['budget_percent']) for r in locked['review_policies']] == review_keys
                and [(r['rule'], r['budget_percent']) for r in pair['reviews']] == review_keys, 'Missing scenarios')
        base = np.asarray(pair['base']['metrics']['confusion_matrix'])
        for selection, r in zip(locked['policies'], pair['automatic']):
            original_selection = next(o for o in old_locked['policies'] if o['kind'] == 'guarded'
                and (o['rule'], o['budget_percent']) == (r['rule'], r['budget_percent']))
            chosen = selection['selection']
            require(selection['calibration_rows'] == cal['rows'] and selection['calibration_benign_rows'] == cal['classes'][gate.BENIGN]
                    and selection['calibration_base_false_alerts'] == original_selection['calibration_base_false_alerts']
                    and 0 <= chosen['calls'] <= cal['rows']*r['budget_percent']//100
                    and chosen['false_alerts'] <= selection['calibration_base_false_alerts']+cal['classes'][gate.BENIGN]*r['extra_false_alert_bp']//10000,
                    'Calibration constraint changed')
            m, e = metrics_from_counts(r['metrics']['confusion_matrix'], supports)
            extra = e[gate.BENIGN]['false_alerts']-gate.false_alerts(base)
            require(r['metrics'] == m and r['errors'] == e and r['rows'] == rows
                    and r['threshold'] == chosen['threshold'] and 0 <= r['calls'] <= rows
                    and r['escalation_rate'] == r['calls']/rows and r['budget_respected'] == (r['calls'] <= rows*r['budget_percent']//100)
                    and r['extra_false_alerts'] == extra and r['extra_false_alerts_per_10000_benign'] == 10000*extra/supports[gate.BENIGN]
                    and r['alert_limit_respected'] == (extra <= supports[gate.BENIGN]*r['extra_false_alert_bp']//10000), 'Automatic metrics/limits changed')
            routing = r['per_class_routing']
            for k, total in [('escalated', 'calls'), ('corrected', 'corrected_rows'), ('damaged', 'damaged_rows')]:
                require(sum(v[k] for v in routing.values()) == r[total], 'Routing totals differ')
            for c, name in enumerate(gate.CLASSES):
                v = routing[name]
                require(v['support'] == supports[name] and 0 <= v['corrected'] and 0 <= v['damaged']
                        and v['corrected']+v['damaged'] <= v['escalated'] <= v['support']
                        and m['confusion_matrix'][c][c] == base[c, c]+v['corrected']-v['damaged'], 'Correction/harm changed')
            if r['extra_false_alert_bp'] == 0:
                original = next(o for o in old['policies'] if o['kind'] == 'guarded'
                    and (o['rule'], o['budget_percent']) == (r['rule'], r['budget_percent']))
                require(chosen == original_selection['selection'] and all(r[k] == original[k] for k in
                    ('threshold', 'metrics', 'errors', 'calls', 'rows', 'escalation_rate', 'budget_respected', 'corrected_rows', 'damaged_rows', 'per_class_routing')),
                    'Zero-slack bridge changed')
        for selection, r in zip(locked['review_policies'], pair['reviews']):
            require(0 <= selection['selection']['calls'] <= cal['rows']*r['budget_percent']//100
                    and r['threshold'] == selection['selection']['threshold'] and r['rows'] == rows
                    and r['classification_labels_changed'] is False and r['human_review_performed'] is False
                    and r['successful_reviews'] is None, 'Review incorrectly credited or recalibrated')
            counts = r['per_class']
            require(r['calls'] == sum(v['referred'] for v in counts.values())
                    and r['base_errors_referred'] == sum(v['base_errors_referred'] for v in counts.values())
                    and r['benign_referrals'] == counts[gate.BENIGN]['referred']
                    and r['review_rate'] == r['calls']/rows and r['reviews_per_10000_flows'] == 10000*r['calls']/rows
                    and r['budget_respected'] == (r['calls'] <= rows*r['budget_percent']//100), 'Review totals changed')
            for c, name in enumerate(gate.CLASSES):
                v = counts[name]
                require(v['support'] == supports[name] and 0 <= v['referred'] <= v['support']
                        and v['review_coverage'] == v['referred']/v['support']
                        and 0 <= v['base_errors_referred'] <= v['referred']
                        and 0 <= v['base_errors_not_referred'] <= v['support']-v['referred']
                        and v['base_errors_referred']+v['base_errors_not_referred'] == supports[name]-base[c, c], 'Review errors changed')
                require(0 <= v['attacks_predicted_benign_referred'] <= v['base_errors_referred']
                        and 0 <= v['attacks_predicted_benign_not_referred'] <= v['base_errors_not_referred']
                        and v['attacks_predicted_benign_referred']+v['attacks_predicted_benign_not_referred'] ==
                        (int(base[c, gate.BENIGN_ID]) if c != gate.BENIGN_ID else 0), 'Missed-attack workload changed')
    require(record['summary'] == summarize(record['pairs']), 'Summary changed')
    return dict(status='verified', pairs=6, automatic_scenarios=360, review_scenarios=54,
                test_opened=False, scope='Aggregate provenance and count audit; not independent confirmation')


def run(router_run, packed, panel, output):
    root, output = Path(router_run), Path(output)
    require(not output.exists(), 'Preserve existing run')
    p, previous = frozen.read(PLAN), frozen.read(PREVIOUS)
    validate_plan(p)
    prior_report.audit(previous)
    verify_private(root, previous)
    runtime = frozen.configure('cpu')
    require(runtime == previous['identity']['runtime'], 'Use recorded local CPU runtime')
    parent = frozen.load_plan()
    manifest = frozen.verify_pack(packed, parent)
    sources = dict(previous['identity']['source_sha256'])
    sources.update({n: digest(ROOT/n) for n in ('scripts/official39_routing_gate_report.py', 'scripts/official39_policy_tradeoffs.py')})
    identity = dict(plan=p, plan_sha256=digest(PLAN), source_sha256=sources, runtime=runtime,
                    prior_private_receipt_sha256=previous['source_receipt_sha256'])
    started = time.perf_counter()
    with frozen.session(output, identity, resume=False):
        receipt = dict(status='preparing', scope=p['scope'], identity=identity,
                       calibration_thresholds_selected=True, **dict.fromkeys(FLAGS, False))
        frozen.save(output/'receipt.json', receipt)
        with headroom.validation_only(packed, manifest) as (x, y):
            keep, info = headroom.load_panel(panel, SimpleNamespace(root=Path(packed), arrays={'val': (x, y)}, manifest=manifest))
            require(info == previous['validation_panel'], 'Validation panel changed')
            rows = np.flatnonzero(keep)
            split = np.load(root/'partitions.npy', allow_pickle=False)
            require(split.shape == rows.shape and split.dtype == np.uint8 and np.isin(split, [0, 1, 2]).all(), 'Wrong saved split')
            for i, name in enumerate(('fit', 'calibration', 'audit')):
                indices = np.flatnonzero(split == i)
                prior_partition = previous['split']['partitions'][name]
                require(prior.vector_hash(indices) == prior_partition['indices_sha256']
                        and dict(zip(gate.CLASSES, map(int, np.bincount(y[rows[indices]], minlength=8)))) == prior_partition['classes'], 'Grouping/cohort changed')
            receipt.update(validation_panel=info, split=previous['split'], prepared=[])
            old_plan = previous['identity']['plan']
            for pair in previous['prepared_pairs']:
                receipt['prepared'].append(prepare_pair(pair, x, y, rows, split, root, old_plan))
                print(f'FROZEN: {pair["lane"]} seed={pair["seed"]}; zero-slack calibration bridge exact', flush=True)
            receipt['status'] = 'frozen_before_repeat_audit'
            frozen.save(output/'calibration.json', receipt)
            locked_hash = digest(output/'calibration.json')
            locked = frozen.read(output/'calibration.json')
            # The partition was already examined in the earlier study. This
            # freeze prevents within-run feedback, not post-audit selection bias.
            pairs = [score_pair(a, b, x, y, rows, split, root, old_plan, c)
                     for a, b, c in zip(locked['prepared'], previous['prepared_pairs'], previous['pairs'])]
            receipt.update(pairs=pairs, summary=summarize(pairs), calibration_sha256=locked_hash)
            require(digest(output/'calibration.json') == locked_hash, 'Threshold freeze changed')
        verify_private(root, previous)
        validate_plan(frozen.read(PLAN))
        require(digest(PLAN) == identity['plan_sha256'] and all(digest(ROOT/n) == h for n, h in sources.items()), 'Source/plan changed')
        require(digest(Path(packed)/'manifest.json') == parent['data_identity']['packed_manifest_sha256']
                and digest(Path(packed)/'scaler.json') == parent['data_identity']['scaler_sha256']
                and digest(Path(panel)/'receipt.json') == info['receipt_sha256']
                and all(digest(Path(panel)/n) == h for n, h in info['files'].items()), 'Input metadata changed')
        receipt.update(status='complete', elapsed_seconds=time.perf_counter()-started,
                       zero_slack_bridge_exact=True, operating_point_selected=False)
        frozen.save(output/'receipt.json', receipt)
    return receipt


def publish(receipt, output, destination):
    destination, output = Path(destination), Path(output)
    require(not destination.exists(), 'Preserve published evidence')
    require(receipt == frozen.read(output/'receipt.json') and receipt['status'] == 'complete'
            and all(receipt[k] is False for k in FLAGS) and receipt['operating_point_selected'] is False, 'Wrong result scope')
    require(digest(output/'calibration.json') == receipt['calibration_sha256']
            and receipt['summary'] == summarize(receipt['pairs']), 'Frozen result changed')
    locked = frozen.read(output/'calibration.json')
    require(locked['status'] == 'frozen_before_repeat_audit' and 'pairs' not in locked
            and locked['prepared'] == receipt['prepared'], 'Missing threshold freeze')
    audit(receipt)
    # Publish aggregates, not flow rows, probability caches or tree parameters.
    # Full candidate grids remain in the private calibration receipt.
    export = {k: v for k, v in receipt.items() if k not in ('prepared',)}
    export['prepared'] = [{k: v for k, v in r.items() if k != 'grids'} for r in receipt['prepared']]
    export['source_receipt_sha256'] = digest(output/'receipt.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(export, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('router-run', 'packed', 'panel', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--publish-summary', type=Path)
    parser.add_argument('--audit-summary', type=Path)
    a = parser.parse_args()
    if a.audit_summary:
        require(all(v is None for k, v in vars(a).items() if k != 'audit_summary'), 'Audit does not accept run arguments')
        print(audit(frozen.read(a.audit_summary)))
        return
    require(all(getattr(a, k) is not None for k in ('router_run', 'packed', 'panel', 'output')), 'All run input/output paths required')
    require(a.publish_summary is None or not a.publish_summary.exists(), 'Preserve published evidence')
    result = run(a.router_run, a.packed, a.panel, a.output)
    if a.publish_summary:
        publish(result, a.output, a.publish_summary)
    print('COMPLETE: exploratory trade-offs only; no training, test, cloud or operational budget approval')


if __name__ == '__main__':
    main()
