"""One-factor class-priority control for the existing eight-class tree.

Equal total class weight changes the objective, not the sample or architecture.
It does not promise higher precision: a rare-class prediction still needs to
be judged against the natural-prevalence audit, not a balanced evaluation set.
"""
import numpy as np
from threadpoolctl import threadpool_limits

from src.data.label_map import CLASSES
from src.eval import feature_expert as expert
from src.eval.matched_neural import vector_hash
from src.eval.official_evidence import require
from src.eval.routing_gate import labels_ok


def weights_for(y, population_counts, weighting='equal_class'):
    """Mean-one weights keep the overall loss/L2 scale fixed.

    The sqrt_population path is a test bridge to the historical recipe.
    Only equal_class is run on real data here. With m_c sampled examples of
    class c, N/(K*m_c) gives each class total weight N/K. Do not multiply by
    inverse full-population frequency again: that would overcorrect the cap.
    Population counts validate the sample but do not set equal-class weights.
    """
    labels_ok(y)
    require(weighting in ('equal_class', 'sqrt_population'), 'Unknown class objective')
    old = expert.score_fusion.fit_weights(y, population_counts)
    if weighting == 'sqrt_population':
        return old
    sample_counts = np.bincount(y, minlength=len(CLASSES))
    return (len(y)/(len(CLASSES)*sample_counts))[y].astype(np.float64)


def fit(z, y, population_counts, seed, weighting='equal_class'):
    require(z.shape == (len(y), 39) and z.dtype == np.float64 and np.isfinite(z).all()
            and type(seed) is int and seed >= 0, 'Invalid fitting inputs')
    weights = weights_for(y, population_counts, weighting)
    # The recipe, regularization, rows and seeds are unchanged. Avoid sklearn
    # class_weight as a second multiplier, resampling, early stopping or a
    # validation-fraction setting that could hide a different fitting budget.
    with threadpool_limits(limits=2):
        model = expert.HistGradientBoostingClassifier(**expert.CONFIG, random_state=seed).fit(z, y, sample_weight=weights)
        prediction = expert.predict(model, z)
    expert.validate_model(model, 'features', seed)
    return model, dict(mode='features', seed=seed, inputs=39, classes=CLASSES,
        config=expert.CONFIG, weighting=weighting, fit_rows=len(y),
        fit_sample_counts=np.bincount(y, minlength=len(CLASSES)).tolist(),
        fit_population_counts=np.asarray(population_counts).tolist(),
        weighted_class_mass=np.bincount(y, weights=weights, minlength=len(CLASSES)).tolist(),
        sample_weights_sha256=vector_hash(weights.astype('<f8')),
        training_prediction_sha256=vector_hash(prediction), model_sha256=expert.state_hash(model),
        **expert.structure(model))
