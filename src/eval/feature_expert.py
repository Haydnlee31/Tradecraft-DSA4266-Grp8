"""Fixed nonlinear controls for automatic eight-class feature-aware inference.

The same small histogram-boosting recipe sees flow features, frozen neural
scores, or both. That separates access to features from merely replacing a
linear score head with a nonlinear one. There is no review queue or rare-only
binary target: every method predicts the original eight classes.
"""
import hashlib
import io

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from src.data.label_map import CLASSES
from src.eval import score_fusion
from src.eval.official_evidence import require
from src.eval.routing_gate import labels_ok

MODES = {'features': 39, 'scores': 16, 'hybrid': 55}
CONFIG = dict(loss='log_loss', learning_rate=.1, max_iter=100,
    max_leaf_nodes=15, max_depth=None, min_samples_leaf=30, l2_regularization=1.,
    max_bins=255, categorical_features=None, early_stopping=False,
    validation_fraction=None, class_weight=None)


def inputs(mode, x=None, base=None, specialist=None):
    """Only inference-available inputs; labels/row identifiers are not features.

    Flow values already have the historical train-fitted scaling. Trees need
    no newly fitted scaler. Score clipping/logging exactly matches fusion v1.
    Require absent unused inputs, so the feature-only model cannot accidentally
    consume neural outputs, nor the score-only control raw flow values.
    """
    require(mode in MODES, 'Unknown feature-expert mode')
    if mode == 'scores':
        require(x is None, 'Score-only control cannot see flow features')
    else:
        require(x is not None and x.ndim == 2 and x.shape[1] == 39
                and len(x) > 0 and np.isfinite(x).all(), 'Invalid flow features')
    if mode == 'features':
        require(base is None and specialist is None, 'Feature-only control cannot see neural scores')
        return np.ascontiguousarray(x, dtype=np.float64)
    require(base is not None and specialist is not None, 'Both frozen score vectors required')
    scores = score_fusion.inputs(base, specialist)
    if mode == 'scores':
        return scores
    require(len(x) == len(scores), 'Unaligned flow features and scores')
    return np.column_stack((x, scores)).astype(np.float64, copy=False)


def model_bytes(model):
    """Serialize ONLY a model fitted locally by this experiment.

    Joblib is not safe for arbitrary downloaded models. The runner pins the
    bytes it just wrote, loads only its own named output, and never accepts an
    external model file. Aggregate public reports do not include these objects.
    """
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=0, protocol=5)
    return buffer.getvalue()


def state_hash(model):
    return hashlib.sha256(model_bytes(model)).hexdigest()


def validate_model(model, mode, seed):
    require(mode in MODES and type(model) is HistGradientBoostingClassifier, 'Wrong expert type')
    params = model.get_params(deep=False)
    require(all(params[k] == v for k, v in CONFIG.items()) and params['random_state'] == seed,
            'Expert settings changed')
    require(model.n_features_in_ == MODES[mode] and model.n_iter_ == CONFIG['max_iter']
            and not model.do_early_stopping_ and model.n_trees_per_iteration_ == len(CLASSES)
            and np.array_equal(model.classes_, np.arange(len(CLASSES))), 'Expert schema/budget changed')


def structure(model):
    # These arrays are used only for accounting. Inference uses sklearn's
    # public predict API, not our own implementation of its private tree format.
    trees = [t for stage in model._predictors for t in stage]
    leaves = [int(np.count_nonzero(t.nodes['is_leaf'])) for t in trees]
    require(len(trees) == 800 and all(1 <= n <= 15 for n in leaves), 'Unreviewed tree work')
    return dict(iterations=int(model.n_iter_), trees=len(trees),
        nodes=sum(len(t.nodes) for t in trees), leaves=sum(leaves),
        maximum_leaves_in_tree=max(leaves), node_array_bytes=sum(t.nodes.nbytes for t in trees))


def fit(z, y, population_counts, mode, seed):
    labels_ok(y)
    require(mode in MODES and z.shape == (len(y), MODES[mode]) and np.isfinite(z).all()
            and type(seed) is int and seed >= 0, 'Invalid expert fit input')
    weights = score_fusion.fit_weights(y, population_counts)
    # Same fit labels, sampled row identities and inverse-sqrt class objective
    # as the previous linear fusion experiment. No class_weight='balanced',
    # internal validation split, best iteration, or audit-based early stopping.
    with threadpool_limits(limits=2):
        model = HistGradientBoostingClassifier(**CONFIG, random_state=seed).fit(z, y, sample_weight=weights)
        prediction = model.predict(z).astype(np.uint8)
    validate_model(model, mode, seed)
    return model, dict(mode=mode, seed=seed, inputs=MODES[mode], classes=CLASSES,
        config=CONFIG, fit_rows=len(y), fit_sample_counts=np.bincount(y, minlength=8).tolist(),
        fit_population_counts=np.asarray(population_counts).tolist(),
        weighted_class_mass=np.bincount(y, weights=weights, minlength=8).tolist(),
        training_prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),
        model_sha256=state_hash(model), **structure(model))


def predict(model, z):
    require(z.ndim == 2 and z.shape[1] == model.n_features_in_ and np.isfinite(z).all(), 'Wrong expert inference inputs')
    result = model.predict(z).astype(np.uint8)
    labels_ok(result)
    return result


def restore_owned(raw, expected_hash, mode, seed):
    """Called only on locally produced, receipt-bound model bytes, never uploads."""
    require(hashlib.sha256(raw).hexdigest() == expected_hash and 0 < len(raw) < 20_000_000,
            'Owned model checksum/size changed')
    model = joblib.load(io.BytesIO(raw))
    validate_model(model, mode, seed)
    return model
