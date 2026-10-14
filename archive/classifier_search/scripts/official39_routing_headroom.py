"""Check whether existing models are complementary before building a router.

This is a post-closeout DEVELOPMENT diagnostic, not a new final-test result.
Only the named validation arrays are opened. All nine historical full39 models
are fixed; no masked-arm selection, retraining, API access or test CLI exists.

The label-informed oracle is deliberately separated from implementable control
rules. It answers whether the expert can correct a mistake at all. It cannot
tell us whether a router can recognise that mistake without knowing its label.
"""
import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts import official39_final_evaluation as frozen
from src.data.label_map import BENIGN, CLASSES
from src.data.official_ablation_panel import load_panel
from src.eval.official_evidence import describe, digest, metrics_from_counts, require

PLAN = ROOT / 'reports/risk_aware_routing/headroom-plan.json'
N = len(CLASSES)
BENIGN_ID = CLASSES.index(BENIGN)
FLAGS = ('test_opened', 'test_evaluated', 'model_trained', 'learned_gate_trained',
         'external_api_called', 'deployment_authorized')


def validate_plan(plan):
    expected = dict(protocol='official39-routing-headroom-v1', seeds=[7, 17, 27],
                    base_lanes=['iid', 'dirichlet'], expert_lane='light', arm='full39',
                    endpoint='last.pt_at_step_20', device='cpu', batch_size=512,
                    budgets_percent=[1, 5, 10],
                    controls=['lowest_base_confidence', 'random_same_budget'],
                    confidence='uncalibrated_max_softmax',
                    budget_rule='floor(primary_population_rows * percent / 100)',
                    ties='original_validation_row_order')
    require(all(plan.get(k) == v for k, v in expected.items()), 'Unreviewed headroom scope')
    require(all(plan.get(k) is False for k in FLAGS), 'Headroom cannot train or access test/API')
    require(plan['parent_plan_sha256'] == digest(frozen.PLAN), 'Frozen parent plan changed')
    require(plan['validation_panel_receipt_sha256'] ==
            'd2ae8eb1cdbd37974dd2e1c5c880e56858adc69181121760b16ea2f641309e77',
            'Unreviewed validation population')


def select_candidates(parent):
    candidates = [c for c in parent['candidates'] if c['arm'] == 'full39']
    expected = {(lane, seed) for lane in ('light', 'iid', 'dirichlet') for seed in (7, 17, 27)}
    require(len(candidates) == 9 and {(c['lane'], c['seed']) for c in candidates} == expected
            and all(c['role'] == 'primary_reference' for c in candidates),
            'Require all nine fixed full39 references, not a selected seed or model')
    return candidates


@contextmanager
def validation_only(root, manifest):
    """No train/test arrays, globs or arbitrary manifest-driven filenames.

    Hashes are checked before mmap and again on exit. The matrix remains on
    disk; only small prediction vectors and bounded feature batches enter RAM.
    """
    root, opened = Path(root), []
    names = ('val_x.npy', 'val_y.npy')
    for name in names:
        require(digest(root / name) == manifest['files'][name], 'Validation checksum changed')
    try:
        for name in names:
            opened.append(np.load(root / name, mmap_mode='r', allow_pickle=False))
        x, y = opened
        require(x.shape == (manifest['val_rows'], 39) and y.shape == (len(x),)
                and x.dtype == np.float32 and y.dtype == np.int64, 'Wrong validation arrays')
        yield x, y
    finally:
        for array in opened:
            array._mmap.close()
    for name in names:
        require(digest(root / name) == manifest['files'][name], 'Validation changed during run')


def labels_ok(*arrays):
    require(all(a.ndim == 1 and np.issubdtype(a.dtype, np.integer)
                and np.all((a >= 0) & (a < N)) for a in arrays), 'Invalid class indices')
    require(len({len(a) for a in arrays}) == 1 and len(arrays[0]) > 0, 'Unaligned or empty predictions')


def metrics(y, predicted):
    labels_ok(y, predicted)
    # Cast before multiplication: uint8 predictions alone would overflow for
    # the three-axis joint table below. Canonical class IDs come from label_map.
    counts = np.bincount(y.astype(np.int64) * N + predicted, minlength=N*N).reshape(N, N)
    result, errors = metrics_from_counts(counts.tolist(), dict(zip(CLASSES, map(int, counts.sum(1)))))
    return dict(metrics=result, errors=errors)


def scan(model, x, y, features):
    """Exactly the frozen batch shape and argmax, with uncalibrated confidence.

    Confidence is a routing diagnostic only, never a claimed probability that
    this domain's prediction is correct. We do not fit a temperature here.
    """
    before = frozen.state_digest(model)
    model.eval()
    predicted = np.empty(len(y), dtype=np.uint8)
    confidence = np.empty(len(y), dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for xb, yb in frozen.batches(x, y, features, 'full39'):
            logits = model(torch.from_numpy(xb))
            require(logits.shape == (len(yb), N) and torch.isfinite(logits).all().item(),
                    'Invalid logits')
            end = offset + len(yb)
            predicted[offset:end] = logits.argmax(1).numpy()
            confidence[offset:end] = logits.softmax(1).max(1).values.numpy()
            offset = end
    require(offset == len(y) and frozen.state_digest(model) == before, 'Coverage or weights changed')
    return predicted, confidence


def joint_counts(y, base, expert):
    labels_ok(y, base, expert)
    key = (y.astype(np.int64) * N + base) * N + expert
    return np.bincount(key, minlength=N**3).reshape(N, N, N)


def pair_analysis(y, base, expert, confidence, seed, budgets):
    """Paired rows, not differences between unrelated aggregate confusions.

    Every routed result covers the SAME full population. Review referrals are
    not fabricated true positives. The budgeted controls substitute the actual
    expert's prediction, including cases where it damages a correct prediction.
    """
    labels_ok(y, base, expert)
    require(confidence.shape == y.shape and np.isfinite(confidence).all()
            and np.all((confidence >= 0) & (confidence <= 1)), 'Invalid confidence')
    require(budgets and all(type(p) is int and 0 <= p <= 100 for p in budgets)
            and len(set(budgets)) == len(budgets), 'Invalid routing budgets')
    base_ok, expert_ok = base == y, expert == y
    rescue, harm = ~base_ok & expert_ok, base_ok & ~expert_ok
    counts = joint_counts(y, base, expert)
    # This oracle only chooses existing predictions; it never replaces an
    # example that BOTH models miss with its true label. Its per-class recall
    # is the union ceiling for these two classifiers, not an F1 optimum.
    oracle = np.where(rescue, expert, base)
    classes = {}
    for c, name in enumerate(CLASSES):
        mask, matrix = y == c, counts[c]
        support = int(mask.sum())
        require(support > 0, 'Every class must remain in the population')
        both = int(matrix[c, c])
        base_only = int(matrix[c, :].sum()) - both
        expert_only = int(matrix[:, c].sum()) - both
        record = dict(support=support, both_correct=both, base_only_correct=base_only,
                      expert_only_correct=expert_only,
                      neither_correct=support-both-base_only-expert_only,
                      union_recall_ceiling=(both+base_only+expert_only)/support)
        if c != BENIGN_ID:
            missed = mask & (base == BENIGN_ID)
            record['base_benign_misses'] = dict(
                total=int(missed.sum()), corrected_category=int((missed & expert_ok).sum()),
                flagged_wrong_attack=int((missed & (expert != BENIGN_ID) & ~expert_ok).sum()),
                still_benign=int((missed & (expert == BENIGN_ID)).sum()))
        classes[name] = record

    # Stable sorting resolves ties by the original validation row order. This
    # is an offline, whole-cohort quota, NOT a fitted threshold deployable on a
    # single arriving flow. Both orders are entirely independent of labels.
    orders = dict(lowest_base_confidence=np.argsort(confidence, kind='stable'),
                  random_same_budget=np.random.default_rng(seed).permutation(len(y)))
    controls, bounds = [], []
    for percent in budgets:
        budget = len(y) * percent // 100
        bounds.append(dict(budget_percent=percent, max_rows=budget,
                           max_correctable_rows=min(budget, int(rescue.sum())),
                           scope='label-informed exact-correctness count bound; not joint class recall or F1'))
        for name, order in orders.items():
            take = order[:budget]
            routed = base.copy()
            routed[take] = expert[take]
            rows = {}
            for c, label in enumerate(CLASSES):
                chosen = take[y[take] == c]
                rows[label] = dict(escalated=int(len(chosen)),
                                   corrected=int(rescue[chosen].sum()),
                                   damaged=int(harm[chosen].sum()))
            controls.append(dict(rule=name, budget_percent=percent,
                escalated_rows=budget, escalation_rate=budget/len(y),
                selected_rows_sha256=hashlib.sha256(take.astype('<i8').tobytes()).hexdigest(),
                corrected_rows=int(rescue[take].sum()), damaged_rows=int(harm[take].sum()),
                per_class_routing=rows, **metrics(y, routed)))
    return dict(rows=len(y), joint_axes=['true', 'base', 'expert'], joint_counts=counts.tolist(),
                base=metrics(y, base), always_expert=metrics(y, expert), per_class=classes,
                oracle=dict(uses_true_labels=True, implementable=False,
                    macro_f1_upper_bound=False, corrected_rows=int(rescue.sum()),
                    required_escalation_rate=float(rescue.mean()), **metrics(y, oracle)),
                budget_correctness_bounds=bounds, controls=controls)


def summarize(pairs):
    """Equal-weight seed means, never pooled seed predictions or best-seed picks."""
    result = {}
    for lane in ('iid', 'dirichlet'):
        rows = [p for p in pairs if p['base_lane'] == lane]
        require([p['seed'] for p in rows] == [7, 17, 27], 'Incomplete paired seeds')
        summary = {}
        for strategy in ('base', 'always_expert', 'oracle'):
            summary[strategy] = {key: describe([r[strategy]['metrics'][key] for r in rows])
                                 for key in ('macro_f1', 'benign_false_alert_rate')}
            summary[strategy]['per_class_recall'] = {
                c: describe([r[strategy]['metrics']['per_class'][c]['recall'] for r in rows]) for c in CLASSES}
        summary['oracle']['required_escalation_rate'] = describe([r['oracle']['required_escalation_rate'] for r in rows])
        summary['controls'] = []
        for index, reference in enumerate(rows[0]['controls']):
            records = [r['controls'][index] for r in rows]
            require(all((r['rule'], r['budget_percent']) == (reference['rule'], reference['budget_percent'])
                        for r in records), 'Control order changed')
            summary['controls'].append(dict(rule=reference['rule'], budget_percent=reference['budget_percent'],
                **{key: describe([r['metrics'][key] for r in records])
                   for key in ('macro_f1', 'benign_false_alert_rate')},
                per_class_recall={c: describe([r['metrics']['per_class'][c]['recall'] for r in records]) for c in CLASSES}))
        result[lane] = summary
    return result


def publish_summary(receipt, receipt_path, destination):
    """Small teammate-facing aggregate; never copy flow rows or checkpoints.

    Full confusion matrices and joint counts remain in the private run receipt.
    This export retains per-seed values, every class's recall, correction/harm
    counts and provenance. It is not an independent replay of raw predictions.
    """
    destination = Path(destination)
    require(not destination.exists(), 'Summary exists; preserve it')
    require(receipt == frozen.read(receipt_path) and receipt['status'] == 'complete'
            and all(receipt[k] is False for k in FLAGS), 'Require completed validation-only receipt')
    require(len(receipt['model_checks']) == 9 and
            all(m['same_runtime_reference_exact'] and m['weights_unchanged'] for m in receipt['model_checks']),
            'Incomplete model checks')
    require(receipt['summary'] == summarize(receipt['pairs']), 'Summary differs from paired records')
    export = {key: receipt[key] for key in ('status', 'scope', 'identity', 'model_checks', 'summary', *FLAGS)}
    export['source_receipt_sha256'] = digest(receipt_path)
    export['pairs'] = []
    for pair in receipt['pairs']:
        export['pairs'].append(dict(seed=pair['seed'], base_lane=pair['base_lane'],
            expert_lane=pair['expert_lane'], rows=pair['rows'], per_class=pair['per_class'],
            budget_correctness_bounds=pair['budget_correctness_bounds'],
            controls=[{key: c[key] for key in ('rule', 'budget_percent', 'escalated_rows',
                        'corrected_rows', 'damaged_rows', 'per_class_routing')} for c in pair['controls']]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a second publisher racing the
    # preflight check. A failed write is not permission to overwrite evidence.
    import json
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(export, handle, indent=2, allow_nan=False)
        handle.write('\n')


def run(archives, packed, panel, output, plan_path=PLAN):
    output = Path(output)
    require(not output.exists(), 'Output exists; preserve prior evidence')
    plan = frozen.read(plan_path)
    validate_plan(plan)
    parent, runtime = frozen.load_plan(), frozen.configure('cpu')
    candidates = select_candidates(parent)
    manifest = frozen.verify_pack(packed, parent)
    sources = frozen.source_identity()
    for name in ('scripts/official39_routing_headroom.py', 'src/data/official_inventory.py'):
        sources[name] = digest(ROOT / name)
    identity = dict(protocol=plan['protocol'], plan_sha256=digest(plan_path), plan=plan,
                    parent_plan_sha256=digest(frozen.PLAN), runtime=runtime, source_sha256=sources,
                    candidate_ids=[c['id'] for c in candidates], **dict.fromkeys(FLAGS, False))
    started = time.perf_counter()
    with validation_only(packed, manifest) as (x, y):
        keep, panel_info = load_panel(panel, SimpleNamespace(root=Path(packed), arrays={'val': (x, y)}, manifest=manifest))
        require(panel_info['receipt_sha256'] == plan['validation_panel_receipt_sha256'], 'Validation panel changed')
        identity['validation_panel'] = panel_info
        with frozen.session(output, identity, resume=False) as folder:
            receipt = dict(status='incomplete', scope='exploratory_reused_validation_not_independent_test',
                           session_sha256=digest(folder/'session.json'), identity=identity,
                           model_checks=[], pairs=[], **dict.fromkeys(FLAGS, False))
            frozen.save(folder/'receipt.json', receipt)
            store = frozen.Artifacts(archives, candidates)
            try:
                py = np.array(y[keep], copy=True)
                for seed in plan['seeds']:
                    predictions, confidences = {}, {}
                    for lane in ('light', 'iid', 'dirichlet'):
                        candidate = next(c for c in candidates if c['seed'] == seed and c['lane'] == lane)
                        model, env, historical = store.load(candidate, parent, 'cpu')
                        require(env['input_transform']['panel'] == panel_info, 'Checkpoint panel mismatch')
                        pred, conf = scan(model, x, y, manifest['features'])
                        # Independent frozen evaluator, same runtime: differences
                        # against historical CUDA are reported, never concealed.
                        reference = frozen.score(model, frozen.batches(x, y, manifest['features'], 'full39'), 'cpu', keep)
                        for pop_y, pop_pred, name in ((y, pred, 'all'), (py, pred[keep], 'panel')):
                            require(metrics(pop_y, pop_pred)['metrics']['confusion_matrix'] ==
                                    reference[name]['metrics']['confusion_matrix'], 'Same-runtime reference mismatch')
                        receipt['model_checks'].append(dict(candidate=candidate,
                            model_state_sha256=reference['model_state_sha256'], weights_unchanged=True,
                            same_runtime_reference_exact=True, historical_comparison=frozen.compare_archive(reference, historical),
                            validation_prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest()))
                        predictions[lane], confidences[lane] = pred[keep], conf[keep]
                        print(f'PASS: seed={seed} {lane}, full and panel CPU reference exact', flush=True)
                        del model, pred, conf
                    for lane in plan['base_lanes']:
                        result = pair_analysis(py, predictions[lane], predictions['light'], confidences[lane],
                                               seed, plan['budgets_percent'])
                        result.update(seed=seed, base_lane=lane, expert_lane='light', arm='full39')
                        receipt['pairs'].append(result)
                        print(f'PAIR: seed={seed} {lane}, base F1={result["base"]["metrics"]["macro_f1"]:.6f}, '
                              f'label-informed oracle F1={result["oracle"]["metrics"]["macro_f1"]:.6f}', flush=True)
                    frozen.save(folder/'receipt.json', receipt)
            finally:
                store.close()
        # Rehash only pinned validation inputs, archive files and source files.
        # No final-test archive, array or metrics are opened by this diagnostic.
    require(digest(plan_path) == identity['plan_sha256'] and digest(frozen.PLAN) == identity['parent_plan_sha256']
            and all(digest(ROOT/name) == value for name, value in sources.items()), 'Plan/source changed')
    require(digest(Path(packed)/'manifest.json') == parent['data_identity']['packed_manifest_sha256']
            and digest(Path(packed)/'scaler.json') == parent['data_identity']['scaler_sha256']
            and digest(Path(panel)/'receipt.json') == plan['validation_panel_receipt_sha256'], 'Data metadata changed')
    for name, expected in panel_info['files'].items():
        require(digest(Path(panel)/name) == expected, 'Panel membership changed')
    for name, expected in {(c['artifact']['archive'], c['artifact']['archive_sha256']) for c in candidates}:
        require(digest(Path(archives)/name) == expected, 'Checkpoint archive changed')
    receipt.update(status='complete', summary=summarize(receipt['pairs']),
                   elapsed_seconds=time.perf_counter()-started,
                   next='Review complementarity before fitting a gate; independent final evaluation still required')
    frozen.save(output/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('archives', 'packed', 'panel', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--publish-summary', type=Path,
                        help='Optional fresh aggregate JSON destination; no flow data or weights')
    args = parser.parse_args()
    if args.publish_summary:
        require(not args.publish_summary.exists(), 'Summary exists; preserve it')
    result = run(args.archives, args.packed, args.panel, args.output)
    if args.publish_summary:
        publish_summary(result, args.output/'receipt.json', args.publish_summary)
    print(f'COMPLETE: {len(result["pairs"])} paired development comparisons; no test access, training or APIs')


if __name__ == '__main__':
    main()
