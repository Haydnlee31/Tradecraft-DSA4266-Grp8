"""A data/objective-matched neural control, not a hyperparameter search.

Reuse the light LayerNorm architecture and the tree's sampled-row weights.
There is deliberately no validation argument to fit(), and no best checkpoint.
Both training curves and the returned model refer to the fixed epoch budget.
"""
from dataclasses import asdict, replace
import hashlib
import io

import numpy as np
import torch
from torch.nn import functional as F

from src.data.label_map import BENIGN, CLASSES
from src.eval.feature_ablation import project
from src.eval.official_evidence import require
from src.eval.score_fusion import fit_weights
from src.models.architectures import LIGHT_CONFIG, MLPClassifier

ARM = 'number_total_masked'
CONFIG = replace(LIGHT_CONFIG, normalization='layer')
SETTINGS = dict(epochs=60, batch_size=512, lr=.001, weight_decay=.0001,
                report_epochs=[1, 5, 20, 40, 60], optimizer='Adam',
                loss='mean_of_sample_weight_times_cross_entropy')


def vector_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        a = value.detach().cpu().numpy().astype('<f4', copy=False)
        h.update(name.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def inputs(x, features):
    # The tree used float64 copies of packed float32 values. No information
    # changes on this cast back to the same float32 values for PyTorch.
    return project(x, features, ARM).astype(np.float32)


def weighted_loss(logits, y, weights):
    """Fixed batch-size denominator, NOT the random batch's weight sum.

    The global mean-one sample weights are exactly the tree objective. Keeping
    them in float64 avoids changing their definition when passing to PyTorch;
    network activations and parameter gradients remain float32. Do not add a
    second class_weight or weighted sampler: that would double-compensate.
    """
    require(logits.shape == (len(y), len(CLASSES)) and weights.shape == y.shape
            and len(y) > 0 and torch.isfinite(weights).all().item()
            and (weights > 0).all().item(), 'Invalid weighted batch')
    return (F.cross_entropy(logits, y, reduction='none')*weights).mean()


def predict(model, x, batch_size=8192):
    require(x.ndim == 2 and x.shape[1] == 39 and x.dtype == np.float32
            and len(x) > 0 and np.isfinite(x).all() and batch_size > 0, 'Wrong neural inputs')
    before = state_hash(model)
    model.eval()
    result = np.empty(len(x), np.uint8)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(torch.from_numpy(np.array(x[start:start+batch_size], copy=True)))
            require(torch.isfinite(logits).all().item(), 'Nonfinite neural prediction')
            result[start:start+len(logits)] = logits.argmax(1).numpy()
    require(state_hash(model) == before, 'Prediction changed neural weights')
    return result


def confusion(y, prediction):
    require(y.shape == prediction.shape and y.ndim == 1 and len(y) > 0
            and all(np.issubdtype(a.dtype, np.integer) and np.isin(a, np.arange(8)).all()
                    for a in (y, prediction)), 'Invalid labels/predictions')
    return np.bincount(y.astype(np.int64)*8+prediction, minlength=64).reshape(8, 8)


def prior_standardized(matrix, population_counts):
    """Comparable fit/audit metrics at the same FIT-population class prior.

    The sample deliberately caps common classes. Its raw precision/F1 is not
    directly comparable to the uncapped audit's. Preserve each class's error
    proportions, then give both matrices the same original FIT class mass.
    These are reweighted diagnostics, not observed row counts or test metrics.
    """
    cm, pop = np.asarray(matrix, dtype=np.float64), np.asarray(population_counts)
    require(cm.shape == (8, 8) and np.isfinite(cm).all() and (cm >= 0).all()
            and (cm.sum(1) > 0).all() and pop.shape == (8,) and np.isfinite(pop).all()
            and (pop > 0).all(), 'Every class needs positive support')
    mass = cm/cm.sum(1, keepdims=True)*(pop/pop.sum())[:, None]
    per = {}
    for i, name in enumerate(CLASSES):
        recall = cm[i, i]/cm[i].sum()
        precision = mass[i, i]/mass[:, i].sum() if mass[:, i].sum() else 0.
        per[name] = dict(precision=float(precision), recall=float(recall),
                        f1=float(2*precision*recall/(precision+recall)) if precision+recall else 0.)
    return dict(macro_f1=float(np.mean([v['f1'] for v in per.values()])),
                benign_false_alert_rate=1.-per[BENIGN]['recall'], per_class=per,
                class_prior=(pop/pop.sum()).tolist())


def fit(x, y, population_counts, seed, settings=None):
    """Fixed final endpoint; smaller settings are for synthetic tests only."""
    s = dict(SETTINGS if settings is None else settings)
    require(type(seed) is int and seed >= 0 and x.shape == (len(y), 39)
            and x.dtype == np.float32 and np.isfinite(x).all()
            and s['optimizer'] == 'Adam' and s['loss'] == SETTINGS['loss']
            and 1 <= s['epochs'] <= 60 and s['batch_size'] >= 2
            and 0 < s['lr'] <= .001 and 0 <= s['weight_decay'] <= .0001
            and s['report_epochs'] == sorted(set(s['report_epochs']))
            and s['report_epochs'][0] >= 1 and s['report_epochs'][-1] == s['epochs'], 'Invalid fitting recipe')
    weights = fit_weights(y, population_counts)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = MLPClassifier(39, len(CLASSES), CONFIG)
    initial = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=s['lr'], weight_decay=s['weight_decay'])
    xx, yy, ww = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(weights)
    examples = steps = 0
    observations = []
    for epoch in range(1, s['epochs']+1):
        order = np.random.default_rng(seed+epoch).permutation(len(y))
        model.train()
        loss_sum = 0.
        for start in range(0, len(y), s['batch_size']):
            take = order[start:start+s['batch_size']]
            # LayerNorm supports a singleton tail, so no row is dropped or
            # duplicated. Real cohort: 528 updates per complete epoch.
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_loss(model(xx[take]), yy[take], ww[take])
            require(torch.isfinite(loss).item(), 'Nonfinite training loss')
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item()
                        for p in model.parameters()), 'Nonfinite gradient')
            optimizer.step()
            loss_sum += loss.item()*len(take)
            examples += len(take)
            steps += 1
        if epoch in s['report_epochs']:
            pred = predict(model, x)
            cm = confusion(y, pred).tolist()
            standardized = prior_standardized(cm, population_counts)
            observations.append(dict(epoch=epoch, examples_processed=examples, optimizer_steps=steps,
                train_online_weighted_ce=loss_sum/len(y), training_confusion_matrix=cm,
                train_prior_standardized=standardized, state_sha256=state_hash(model)))
            print(f'NEURAL seed={seed} epoch={epoch}: fit-prior macro-F1={standardized["macro_f1"]:.6f}', flush=True)
    pred = predict(model, x)
    return model, dict(seed=seed, settings=s, model_config=dict(asdict(CONFIG), hidden_dims=list(CONFIG.hidden_dims)),
        parameters=model.num_parameters(), parameter_bytes=model.parameter_bytes(), fit_rows=len(y),
        fit_sample_counts=np.bincount(y, minlength=8).tolist(),
        fit_population_counts=np.asarray(population_counts).tolist(),
        weighted_class_mass=np.bincount(y, weights=weights, minlength=8).tolist(),
        sample_weights_sha256=vector_hash(weights.astype('<f8')), initial_state_sha256=initial,
        final_state_sha256=state_hash(model), training_prediction_sha256=vector_hash(pred),
        examples_processed=examples, optimizer_steps=steps, observations=observations)


def serialize(model):
    # Only tensor weights; no executable estimator is required for reload.
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return stream.getvalue()


def restore(raw, checksum, expected_state):
    require(len(raw) < 100_000 and hashlib.sha256(raw).hexdigest() == checksum, 'Neural file checksum changed')
    # Initialization for reload must not perturb a caller's training RNG.
    with torch.random.fork_rng(devices=[]):
        model = MLPClassifier(39, len(CLASSES), CONFIG)
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True), strict=True)
    require(state_hash(model) == expected_state, 'Restored tensor state changed')
    return model


def paired_correctness(y, tree, neural):
    """Label-informed headroom only; NOT a deployable prediction selector."""
    confusion(y, tree)
    confusion(y, neural)
    result = {}
    for i, name in enumerate(CLASSES):
        use = y == i
        a, b = tree[use] == i, neural[use] == i
        require(len(a) > 0, 'Missing paired class')
        result[name] = dict(support=len(a), both_correct=int((a & b).sum()),
            tree_only_correct=int((a & ~b).sum()), neural_only_correct=int((~a & b).sum()),
            neither_correct=int((~a & ~b).sum()), oracle_recall_upper_bound=float((a | b).mean()))
    return result
