"""Small, fully automatic score fusion; no referral or true label at inference.

The two neural classifiers stay frozen. A multinomial logistic head learns
from their soft scores, so it can predict a third class that neither model's
argmax selected. Separate one-model heads test whether combining models adds
anything beyond supervised recalibration. This is stacking, not new FL training.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.data.label_map import CLASSES
from src.eval.official_evidence import require
from src.eval.routing_gate import labels_ok

N = len(CLASSES)
EPSILON = 1e-6
CONFIG = dict(C=1.0, solver='lbfgs', max_iter=1000, tol=1e-5, fit_intercept=True)


def inputs(*probabilities):
    """One or two eight-class score vectors, in canonical class order.

    Clip zeros before log, not before argmax of standalone controls. Log scores
    preserve small rare-class signals. We do not call them calibrated risks.
    Labels, row IDs and review decisions are absent from this interface.
    """
    require(len(probabilities) in (1, 2), 'Require one or two models')
    for p in probabilities:
        require(p.ndim == 2 and p.shape[1] == N and len(p) > 0
                and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
                and np.allclose(p.sum(1), 1, atol=2e-6, rtol=0), 'Invalid probabilities')
    require(len({len(p) for p in probabilities}) == 1, 'Unaligned probabilities')
    return np.log(np.maximum(np.column_stack(probabilities).astype(np.float64), EPSILON))


def fit_weights(y, population_counts):
    """Correct the capped sample, then apply square-root class balancing.

    With n_c full FIT rows and m_c sampled FIT rows, each retained row gets
    sqrt(n_c)/m_c. Thus the class's total loss weight is proportional to
    sqrt(n_c), as in full-fit inverse-sqrt-frequency weighting. This avoids
    accidentally applying full inverse-frequency weighting a second time to
    the class-capped sample. Mean-one normalization fixes the regularization
    scale across the three heads. No calibration/audit counts enter the fit.
    """
    labels_ok(y)
    n = np.asarray(population_counts)
    m = np.bincount(y, minlength=N)
    require(n.shape == (N,) and np.issubdtype(n.dtype, np.integer)
            and np.all(n >= m) and np.all(m > 0), 'Invalid fit population counts')
    w = (np.sqrt(n)/m)[y]
    return w/w.mean()


def validate_head(head):
    require(head.get('protocol') == 'log-score-fusion-v1' and head.get('classes') == CLASSES
            and head.get('epsilon') == EPSILON and head.get('config') == CONFIG
            and head.get('inputs') in (N, 2*N), 'Wrong fusion schema')
    d = head['inputs']
    mean, scale = np.asarray(head['mean']), np.asarray(head['scale'])
    coef, intercept = np.asarray(head['coef']), np.asarray(head['intercept'])
    require(mean.shape == scale.shape == (d,) and coef.shape == (N, d)
            and intercept.shape == (N,) and all(np.isfinite(a).all() for a in (mean, scale, coef, intercept))
            and np.all(scale > 0), 'Malformed fusion parameters')
    return mean, scale, coef, intercept


def predict(head, z):
    mean, scale, coef, intercept = validate_head(head)
    require(z.ndim == 2 and z.shape[1] == head['inputs'] and np.isfinite(z).all(), 'Wrong fusion inputs')
    # Export plain JSON parameters rather than executable pickle objects.
    # Class order is fixed and ties consistently select the first class.
    return (((z-mean)/scale) @ coef.T + intercept).argmax(1).astype(np.uint8)


def fit(z, y, population_counts):
    labels_ok(y)
    require(z.shape in ((len(y), N), (len(y), 2*N)) and np.isfinite(z).all(), 'Invalid fit scores')
    weights = fit_weights(y, population_counts)
    mean, scale = z.mean(0), z.std(0)
    scale[scale < 1e-12] = 1.
    standardized = (z-mean)/scale
    # All hyperparameters are fixed before the internal audit. Failure to
    # converge stops the experiment; it does not trigger a hidden search.
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = LogisticRegression(**CONFIG).fit(standardized, y, sample_weight=weights)
        require(np.array_equal(model.classes_, np.arange(N)), 'Missing fitted class')
        expected = model.predict(standardized)
    head = dict(protocol='log-score-fusion-v1', classes=CLASSES, epsilon=EPSILON,
        config=CONFIG, inputs=z.shape[1], mean=mean.tolist(), scale=scale.tolist(),
        coef=model.coef_.tolist(), intercept=model.intercept_.tolist(),
        iterations=int(model.n_iter_.max()), fit_rows=len(y),
        fit_sample_counts=np.bincount(y, minlength=N).tolist(),
        fit_population_counts=np.asarray(population_counts).tolist(),
        weighted_class_mass=np.bincount(y, weights=weights, minlength=N).tolist(),
        learned_parameters=int(model.coef_.size+model.intercept_.size))
    with threadpool_limits(limits=2):
        require(np.array_equal(predict(head, z), expected), 'JSON head does not reproduce sklearn')
    head['sklearn_prediction_bridge_exact'] = True
    return head


def mean_prediction(base, expert):
    inputs(base, expert)  # Validate before applying the no-fit control.
    return ((base.astype(np.float64)+expert)/2).argmax(1).astype(np.uint8)


def check_split(rows, split, chosen, keys):
    """Do not permit duplicated fit rows, overlap, or moved duplicate groups."""
    require(rows.ndim == split.ndim == keys.ndim == chosen.ndim == 1
            and np.issubdtype(rows.dtype, np.integer) and np.issubdtype(chosen.dtype, np.integer)
            and rows.shape == split.shape == keys.shape and len(rows) > 0 and len(chosen) > 0
            and np.all(rows >= 0) and np.all(np.diff(rows) > 0)
            and np.isin(split, [0, 1, 2]).all() and set(np.unique(split)) == {0, 1, 2}
            and np.all((chosen >= 0) & (chosen < len(rows))) and np.all(np.diff(chosen) > 0), 'Invalid partition indices')
    require(np.all(split[chosen] == 0), 'Fit overlaps calibration or audit')
    from src.eval.routing_gate import partitions
    require(np.array_equal(split, partitions(keys)), 'Duplicate groups changed partition')
