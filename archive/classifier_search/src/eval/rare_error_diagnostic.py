"""Descriptive frozen-model error and feature-overlap diagnostics.

These functions do not fit models, recommend thresholds or estimate Bayes
error. In particular, proximity depends on both the metric and class density.
The label-conditioned distance comparison is post-hoc, not a deployable rule.
"""
import numpy as np
from sklearn.neighbors import KDTree

from src.data.label_map import CLASSES, WEB_BASED, BRUTE_FORCE
from src.eval.official_evidence import require

RARE = [WEB_BASED, BRUTE_FORCE]


def labels(y, size=None):
    require(y.ndim == 1 and np.issubdtype(y.dtype, np.integer)
            and len(y) > 0 and np.isin(y, np.arange(len(CLASSES))).all()
            and (size is None or len(y) == size), 'Invalid labels')


def overlap(y, predictions):
    """Which errors are shared, and what could an impossible perfect selector do?

    The oracle chooses ONLY among existing hard predictions using the true
    label. It is not an ensemble score, generalization bound or new-model limit.
    """
    labels(y)
    require('tree' in predictions and len(predictions) >= 2, 'Missing comparators')
    for pred in predictions.values():
        labels(pred, len(y))
    names = list(predictions)
    p = np.stack(list(predictions.values()), axis=1)
    correct = p == y[:, None]
    tree = predictions['tree'] == y
    result = {}
    for k, name in enumerate(CLASSES):
        rows = y == k
        count = correct[rows].sum(1)
        all_miss = rows & ~correct.any(1)
        unanimous_wrong = all_miss & (p == p[:, :1]).all(1)
        arms = {}
        for arm, pred in predictions.items():
            good = pred == y
            arms[arm] = dict(correct=int((rows & good).sum()),
                recovered_tree_misses=int((rows & ~tree & good).sum()),
                lost_tree_hits=int((rows & tree & ~good).sum()),
                false_negative_destinations=dict(zip(CLASSES, np.bincount(
                    pred[rows & ~good], minlength=len(CLASSES)).tolist())),
                false_positive_sources=dict(zip(CLASSES, np.bincount(
                    y[(~rows) & (pred == k)], minlength=len(CLASSES)).tolist())))
        result[name] = dict(support=int(rows.sum()),
            correct_model_count_histogram=np.bincount(count, minlength=len(names)+1).tolist(),
            all_models_miss=int(all_miss.sum()),
            any_model_correct=int((rows & correct.any(1)).sum()),
            oracle_recall=float(correct[rows].any(1).mean()) if rows.any() else None,
            unanimous_wrong_destinations=dict(zip(CLASSES, np.bincount(
                p[unanimous_wrong, 0], minlength=len(CLASSES)).tolist())), arms=arms)
    return dict(models=names, classes=result)


def exact_conflicts(x, y):
    """Exact packed float32 vectors, NOT equality in original CSV precision.

    All duplicate rows remain counted. Mixed-label groups establish ambiguity
    in this finite fitting representation only; their absence proves nothing
    about separability of previously unseen examples.
    """
    labels(y, len(x))
    require(x.ndim == 2 and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid vectors')
    _, inverse = np.unique(x, axis=0, return_inverse=True)
    groups = int(inverse.max())+1
    counts = np.zeros((groups, len(CLASSES)), dtype=np.int64)
    np.add.at(counts, (inverse, y), 1)
    mixed = (counts > 0).sum(1) > 1
    result = {}
    for k, name in enumerate(CLASSES):
        present = counts[:, k] > 0
        result[name] = dict(rows=int(counts[:, k].sum()), unique_vectors=int(present.sum()),
            rows_in_mixed_label_groups=int(counts[mixed, k].sum()),
            conflicting_groups_by_other_class={other: int((present & (counts[:, j] > 0)).sum())
                for j, other in enumerate(CLASSES) if j != k})
    return dict(rows=len(x), unique_vectors=groups, mixed_label_groups=int(mixed.sum()),
        minimum_total_fitting_errors_from_exact_conflicts=int((counts.sum(1)-counts.max(1)).sum()),
        classes=result)


def nearest_by_class(query, reference, reference_y):
    """Distance only, to one nearest FIT vector per class; no votes or fitting.

    Duplicate references cannot change nearest distance. All classes use ALL
    their fitting rows rather than an enriched/capped bank. This still leaves
    natural density differences, which must be retained as an interpretation limit.
    """
    labels(reference_y, len(reference))
    require(query.ndim == reference.ndim == 2 and query.shape[1] == reference.shape[1]
            and len(query) > 0 and np.isfinite(query).all() and np.isfinite(reference).all(),
            'Invalid geometry')
    result = np.empty((len(query), len(CLASSES)), dtype=np.float64)
    for k in range(len(CLASSES)):
        values = reference[reference_y == k]
        require(len(values) > 0, 'Missing reference class')
        tree = KDTree(values, leaf_size=40)
        result[:, k] = tree.query(query, k=1, return_distance=True)[0][:, 0]
    return result


def proximity(distances, y, predictions):
    labels(y)
    require(distances.shape == (len(y), len(CLASSES)) and np.isfinite(distances).all()
            and (distances >= 0).all(), 'Invalid distances')
    require('tree' in predictions, 'Missing tree')
    for pred in predictions.values():
        labels(pred, len(y))
    correct = np.stack([v == y for v in predictions.values()], axis=1)
    tree = predictions['tree'] == y
    strata = dict(all=np.ones(len(y), bool), tree_correct=tree,
                  tree_missed_other_model_correct=(~tree) & correct.any(1),
                  all_models_missed=~correct.any(1))
    result = {}
    for name in RARE:
        k = CLASSES.index(name)
        groups = {}
        for group, mask in strata.items():
            dd = distances[(y == k) & mask]
            own = dd[:, k]
            if not len(own):
                groups[group] = dict(rows=0)
                continue
            other = np.delete(dd, k, axis=1).min(1)
            groups[group] = dict(rows=len(own),
                same_class_distance_quartiles=np.quantile(own, [.25, .5, .75]).tolist(),
                other_class_strictly_closer=int((other < own).sum()),
                nearest_other_equal_distance=int((other == own).sum()),
                zero_same_class_distance=int((own == 0).sum()),
                zero_any_class_distance=int((dd.min(1) == 0).sum()),
                comparisons={label: dict(other_closer=int((dd[:, j] < own).sum()),
                    equal_distance=int((dd[:, j] == own).sum()),
                    own_closer=int((own < dd[:, j]).sum()))
                    for j, label in enumerate(CLASSES) if j != k})
        result[name] = groups
    return result
