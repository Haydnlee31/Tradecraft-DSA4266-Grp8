"""Small numerical gate with explicit fit, calibration and audit boundaries.

A gate predicts whether a consultation helps; it does NOT see that expert's
prediction as an input. True labels and both predictions are allowed only to
construct training targets or evaluate a fixed decision. All helpers support
small synthetic fixtures, so the safety rules can run in CI without flow data.
"""
import hashlib

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from src.data.label_map import BENIGN, CLASSES
from src.eval.official_evidence import require

N = len(CLASSES)
BENIGN_ID = CLASSES.index(BENIGN)
GATE_CONFIG = dict(max_depth=5, min_samples_leaf=128, random_state=20261013, criterion='gini')


def group_keys(x, seed=20261011):
    """Labels cannot influence membership; equal vectors include signed zero.

    Using 64 bits for bookkeeping can conservatively co-locate a hash collision;
    it cannot separate identical inputs. Nonfinite features fail closed.
    """
    a = np.array(x, dtype='<f4', order='C', copy=True)
    require(a.ndim == 2 and a.shape[1] == 39 and np.isfinite(a).all(), 'Invalid grouping inputs')
    a[a == 0] = 0
    prefix = f'tradecraft-router-split-{seed}:'.encode()
    return np.array([int.from_bytes(hashlib.sha256(prefix+row.tobytes()).digest()[:8], 'little')
                     for row in a], dtype='<u8')


def partitions(keys):
    require(keys.ndim == 1 and keys.dtype == np.uint64, 'Expected group hashes')
    bucket = keys % 10
    return np.where(bucket < 6, 0, np.where(bucket < 8, 1, 2)).astype(np.uint8)


def fit_rows(y, split, cap=50000, seed=20261012):
    """Only fit-partition labels drive this bounded stratified sampling step."""
    labels_ok(y)
    require(split.shape == y.shape and np.isin(split, [0, 1, 2]).all() and cap > 0, 'Invalid split or cap')
    rng, chosen = np.random.default_rng(seed), []
    for c in range(N):
        rows = np.flatnonzero((split == 0) & (y == c))
        require(len(rows) > 0, 'Missing fit class')
        if len(rows) > cap:
            rows = rng.choice(rows, cap, replace=False)
        chosen.append(rows)
    return np.sort(np.concatenate(chosen)).astype('<i8')


def labels_ok(*arrays):
    require(arrays and all(a.ndim == 1 and np.issubdtype(a.dtype, np.integer)
                          and np.all((a >= 0) & (a < N)) for a in arrays)
            and len({len(a) for a in arrays}) == 1 and len(arrays[0]) > 0, 'Invalid labels or alignment')


def features(x, base_probability):
    require(x.ndim == 2 and x.shape[1] == 39 and base_probability.shape == (len(x), N)
            and np.isfinite(x).all() and np.isfinite(base_probability).all()
            and np.all((base_probability >= 0) & (base_probability <= 1))
            and np.allclose(base_probability.sum(1), 1, rtol=0, atol=2e-6), 'Invalid gate features')
    # Labels, the expert's prediction and group/row identifiers are deliberately
    # absent from this function's interface.
    return np.column_stack((x, base_probability)).astype(np.float32)


def fit_gate(z, y, base, expert):
    labels_ok(y, base, expert)
    require(z.shape == (len(y), 39+N) and z.dtype == np.float32 and np.isfinite(z).all(), 'Invalid fit inputs')
    counts = np.bincount(y, minlength=N)
    require(np.all(counts > 0), 'Every fit class is required')
    target = np.zeros(len(y), np.uint8)
    target[(base != y) & (expert == y)] = 1
    target[(base == y) & (expert != y)] = 2
    weight = (1. / counts)[y]
    weight[y == BENIGN_ID] *= 2
    weight /= weight.mean()
    model = DecisionTreeClassifier(**GATE_CONFIG).fit(z, target, sample_weight=weight)
    t = model.tree_
    # Store each leaf's actual predict_proba output. Some sklearn versions
    # store already-normalized values internally, others store counts. Taking
    # the public prediction once avoids a second normalization that can move
    # scores by a few floating-point bits and break an exact threshold replay.
    fitted_probability = model.predict_proba(z)
    leaf_ids = model.apply(z)
    value = t.value[:, 0, :].copy()
    for leaf in np.unique(leaf_ids):
        value[leaf] = fitted_probability[np.flatnonzero(leaf_ids == leaf)[0]]
    # JSON is auditable and avoids unpickling a model at inference time.
    record = dict(protocol='repair-harm-tree-v1', inputs=39+N, config=GATE_CONFIG,
        children_left=t.children_left.tolist(), children_right=t.children_right.tolist(),
        feature=t.feature.tolist(), threshold=t.threshold.tolist(), classes=model.classes_.tolist(),
        value=value.tolist(), nodes=int(t.node_count), depth=int(model.get_depth()),
        fit_rows=len(y), fit_class_counts=dict(zip(CLASSES, map(int, counts))),
        target_counts=np.bincount(target, minlength=3).tolist())
    expected = np.zeros((len(y), 3))
    expected[:, model.classes_] = fitted_probability
    require(np.array_equal(tree_probabilities(record, z), expected), 'JSON tree does not reproduce sklearn')
    return record


def tree_probabilities(tree, z):
    require(tree['protocol'] == 'repair-harm-tree-v1' and tree['inputs'] == 39+N
            and z.ndim == 2 and z.shape[1] == tree['inputs'] and z.dtype == np.float32
            and np.isfinite(z).all(), 'Wrong inference schema')
    left, right = np.asarray(tree['children_left']), np.asarray(tree['children_right'])
    column, threshold = np.asarray(tree['feature']), np.asarray(tree['threshold'])
    value, classes = np.asarray(tree['value'], dtype=float), np.asarray(tree['classes'], dtype=int)
    n = tree['nodes']
    require(0 < n <= 63 and 0 <= tree['depth'] <= 5 and all(a.shape == (n,) for a in (left, right, column, threshold))
            and value.shape == (n, len(classes)) and np.isfinite(value).all() and np.all(value >= 0)
            and np.all(value.sum(1) > 0) and np.isfinite(threshold).all()
            and len(np.unique(classes)) == len(classes) and np.isin(classes, [0, 1, 2]).all(), 'Malformed tree')
    leaf = left == -1
    require(np.all(right[leaf] == -1) and np.all((left[~leaf] > np.flatnonzero(~leaf)) & (left[~leaf] < n))
            and np.all((right[~leaf] > np.flatnonzero(~leaf)) & (right[~leaf] < n))
            and np.all((column[~leaf] >= 0) & (column[~leaf] < 39+N)), 'Invalid tree edges')
    nodes = np.zeros(len(z), dtype=int)
    for _ in range(5):
        active = ~leaf[nodes]
        ids, current = np.flatnonzero(active), nodes[active]
        nodes[ids] = np.where(z[ids, column[current]] <= threshold[current], left[current], right[current])
    require(np.all(leaf[nodes]), 'Tree exceeds depth bound')
    p = value[nodes].copy()
    require(np.allclose(p.sum(1), 1, rtol=0, atol=1e-10), 'Invalid leaf probabilities')
    result = np.zeros((len(z), 3))
    result[:, classes] = p
    return result


def gate_score(tree, z):
    p = tree_probabilities(tree, z)
    return p[:, 1]-p[:, 2]


def confusion(y, pred):
    labels_ok(y, pred)
    return np.bincount(y.astype(np.int64)*N+pred, minlength=N*N).reshape(N, N)


def macro_f1(cm):
    denominator = cm.sum(0)+cm.sum(1)
    return float(np.divide(2*np.diag(cm), denominator, out=np.zeros(N), where=denominator > 0).mean())


def false_alerts(cm):
    return int(cm[BENIGN_ID].sum()-cm[BENIGN_ID, BENIGN_ID])


def decisions(scores, threshold):
    require(scores.ndim == 1 and np.isfinite(scores).all(), 'Invalid routing score')
    if threshold is None:
        return np.zeros(len(scores), dtype=bool)
    require(np.isfinite(threshold), 'Invalid threshold')
    return scores >= threshold


def calibrate(scores, y, base, expert, budgets=(1, 5, 10)):
    """Choose using calibration labels only; never accept audit inputs here.

    The grid has 0.1 percentage-point increments up to 10%. At a tie boundary,
    omit the whole tie group rather than splitting identical score decisions
    by row ID. A None threshold means zero consultations on ANY later cohort.
    """
    labels_ok(y, base, expert)
    require(scores.shape == y.shape and np.isfinite(scores).all() and list(budgets) == [1, 5, 10], 'Invalid calibration')
    order = np.argsort(-scores, kind='stable')
    ordered = scores[order]
    options = {0: None}
    quota_n = {}
    for bp in range(10, 1001, 10):
        cap = len(y)*bp//10000
        if cap == 0:
            take, threshold = 0, None
        else:
            boundary = ordered[cap-1]
            end = int(np.searchsorted(-ordered, -boundary, side='right'))
            take = end if end <= cap else int(np.searchsorted(-ordered, -boundary, side='left'))
            threshold = float(ordered[take-1]) if take else None
        options[take] = threshold
        if bp in (100, 500, 1000):
            quota_n[bp//100] = take
    cm = confusion(y, base)
    base_f1, base_alerts = macro_f1(cm), false_alerts(cm)
    candidates, offset = [], 0
    for take, threshold in sorted(options.items()):
        added = order[offset:take]
        if len(added):
            cm -= confusion(y[added], base[added])
            cm += confusion(y[added], expert[added])
        offset = take
        candidates.append(dict(threshold=threshold, calls=take, macro_f1=macro_f1(cm), false_alerts=false_alerts(cm)))
    chosen = []
    for budget in budgets:
        allowed = [c for c in candidates if c['calls'] <= len(y)*budget//100
                   and c['false_alerts'] <= base_alerts]
        best = max(allowed, key=lambda c: (c['macro_f1'], -c['calls']))
        chosen.append(dict(budget_percent=budget, guarded=dict(best),
            quota=next(dict(c) for c in candidates if c['calls'] == quota_n[budget])))
    return dict(rows=len(y), base_macro_f1=base_f1, base_false_alerts=base_alerts,
                candidate_count=len(candidates), policies=chosen)


def evaluate_policy(y, base, expert, score, threshold, budget):
    """Apply a fixed threshold without sorting or retuning on the audit set."""
    labels_ok(y, base, expert)
    require(score.shape == y.shape, 'Wrong audit alignment')
    route = decisions(score, threshold)
    pred = np.where(route, expert, base)
    cm = confusion(y, pred)
    per_class = {}
    for c, name in enumerate(CLASSES):
        selected = route & (y == c)
        per_class[name] = dict(support=int((y == c).sum()), escalated=int(selected.sum()),
            corrected=int((selected & (base != y) & (expert == y)).sum()),
            damaged=int((selected & (base == y) & (expert != y)).sum()))
    return dict(confusion_matrix=cm.tolist(), macro_f1=macro_f1(cm), false_alerts=false_alerts(cm),
        calls=int(route.sum()), escalation_rate=float(route.mean()), rows=len(y),
        budget_respected=bool(int(route.sum()) <= len(y)*budget//100),
        benign_guard_respected=bool(false_alerts(cm) <= false_alerts(confusion(y, base))),
        corrected_rows=int((route & (base != y) & (expert == y)).sum()),
        damaged_rows=int((route & (base == y) & (expert != y)).sum()), per_class_routing=per_class)
