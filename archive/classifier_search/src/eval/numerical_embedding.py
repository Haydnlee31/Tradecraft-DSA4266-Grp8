"""Fixed-budget PLE versus parameter-matched raw MLP.

The source data, sample weights and optimizer budget match the prior MLP.
We intentionally do not tune bins, widths, loss or stopping on audit labels.
"""
from dataclasses import replace
import numpy as np
import torch

from src.eval import matched_neural as old
from src.eval.official_evidence import require
from src.models.architectures import MLPClassifier


def parameter_count(inputs, first):
    # Linear + LayerNorm for two hidden layers, followed by eight logits.
    return inputs*first + 3*first + first*32 + 3*32 + 32*8 + 8


def raw_width(ple_dimensions):
    target = parameter_count(ple_dimensions, 64)
    return min(range(1, target+1), key=lambda w: (abs(parameter_count(39, w)-target), w))


def predict(model, x, batch=8192):
    require(x.ndim == 2 and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid numerical inputs')
    model.eval()
    before = old.state_hash(model)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits = model(torch.from_numpy(np.array(x[start:start+batch], copy=True)))
            require(torch.isfinite(logits).all().item(), 'Nonfinite logits')
            predictions.append(logits.argmax(1).numpy().astype(np.uint8))
    require(old.state_hash(model) == before, 'Prediction mutated model')
    return np.concatenate(predictions)


def fit(x, y, population, seed, first, epochs=60, batch=512):
    require(x.ndim == 2 and x.dtype == np.float32 and len(x) == len(y) > 0
            and np.isfinite(x).all() and 1 <= epochs <= 60 and batch >= 2, 'Invalid fitting recipe')
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = MLPClassifier(x.shape[1], 8, replace(old.CONFIG, hidden_dims=(first, 32)))
    require(model.num_parameters() == parameter_count(x.shape[1], first), 'Capacity accounting error')
    weights = old.fit_weights(y, population)
    xx, yy, ww = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    examples = steps = 0
    observations = []
    initial = old.state_hash(model)
    for epoch in range(1, epochs+1):
        order = np.random.default_rng(seed+epoch).permutation(len(y))
        model.train()
        for start in range(0, len(y), batch):
            take = order[start:start+batch]
            optimizer.zero_grad(set_to_none=True)
            loss = old.weighted_loss(model(xx[take]), yy[take], ww[take])
            require(torch.isfinite(loss).item(), 'Nonfinite loss')
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), 'Nonfinite gradient')
            optimizer.step()
            examples += len(take)
            steps += 1
        if epoch in [1, 5, 20, 40, epochs]:
            cm = old.confusion(y, predict(model, x))
            score = old.prior_standardized(cm, population)
            observations.append(dict(epoch=epoch, confusion_matrix=cm.tolist(), prior_standardized=score))
            print(f'NUMERIC input={x.shape[1]} hidden={first} seed={seed} epoch={epoch}: fit macro-F1={score["macro_f1"]:.6f}', flush=True)
    return model, dict(seed=seed, input_dimensions=x.shape[1], hidden_dims=[first, 32],
        parameters=model.num_parameters(), initial_state_sha256=initial, final_state_sha256=old.state_hash(model),
        examples_processed=examples, optimizer_steps=steps, epochs=epochs, batch_size=batch,
        sample_weights_sha256=old.vector_hash(weights.astype('<f8')), observations=observations)
