"""Single-view supervised contrastive learning for numeric flows.

This is an adaptation of SupCon, not its image/two-augmentation recipe. Other
distinct FIT flows of the same class are positives. No fabricated packet
features, reference labels at inference, or evaluation labels are needed.
"""
from dataclasses import replace
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data.label_map import CLASSES
from src.eval import matched_neural as old
from src.eval.official_evidence import require
from src.models.architectures import MLPClassifier

N = len(CLASSES)
SETTINGS = dict(seed=7, epochs=60, steps_per_epoch=528, per_class=64,
                lr=.001, weight_decay=.0001, temperature=.1, coefficient=.1,
                dropout=.2, hidden_dims=[64, 32], projection_dims=[32, 16])


def class_weights(population):
    """Balanced sampling must NOT reuse the old per-row inverse-count weights.

    Target class mass is sqrt(n_c)/sum sqrt(n_c). The sampler supplies 1/8
    mass per class, so multiplying CE by 8*sqrt(n_c)/sum sqrt(n_c) preserves
    that historical objective in expectation. No batch-weight denominator.
    """
    n = np.asarray(population)
    require(n.shape == (N,) and np.issubdtype(n.dtype, np.integer) and (n > 0).all(),
            'Invalid fitting population')
    return N*np.sqrt(n)/np.sqrt(n).sum()


def batches(y, seed, epoch, steps=528, per_class=64):
    require(y.ndim == 1 and np.issubdtype(y.dtype, np.integer)
            and np.isin(y, np.arange(N)).all() and steps > 0 and per_class >= 2,
            'Invalid sampler')
    pools = [np.flatnonzero(y == c) for c in range(N)]
    require(all(len(p) >= per_class for p in pools), 'Insufficient class support')
    rng = np.random.default_rng(seed+epoch)
    for _ in range(steps):
        # No row is repeated within a batch. Across batches, independent draws
        # intentionally revisit rare examples; record that exposure honestly.
        take = np.concatenate([rng.choice(p, per_class, replace=False) for p in pools])
        yield rng.permutation(take)


def supcon(embedding, labels, groups, temperature=.1):
    """Mean positive log-probability, excluding self AND exact duplicate groups.

    Exclusion applies to both the positive set and the denominator, including
    any conflicting-label duplicate. Missing-positive anchors contribute no
    contrastive term. The training controller logs their count.
    """
    require(embedding.ndim == 2 and labels.shape == groups.shape == (len(embedding),)
            and 0 < temperature <= 1 and torch.isfinite(embedding).all().item(), 'Invalid SupCon input')
    z = F.normalize(embedding, dim=1)
    allowed = groups[:, None] != groups[None, :]
    positive = (labels[:, None] == labels[None, :]) & allowed
    count = positive.sum(1)
    eligible = count > 0
    used = int(eligible.sum())
    if used == 0:
        return embedding.sum()*0., 0
    logits = (z[eligible] @ z.T)/temperature
    log_denom = torch.logsumexp(logits.masked_fill(~allowed[eligible], -torch.inf), dim=1)
    mean_positive = (logits*positive[eligible]).sum(1)/count[eligible]
    return (log_denom-mean_positive).mean(), used


class FlowModel(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.base = MLPClassifier(dimensions, N, replace(old.CONFIG, hidden_dims=(64, 32)))
        # Both arms instantiate and execute the same projector. Its loss has
        # coefficient zero in the control; it is discarded for inference.
        self.projector = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16))

    def forward(self, x):
        h = self.base.net[:-1](x)
        return self.base.net[-1](h), self.projector(h)


def fit(x, y, groups, population, coefficient, settings=None):
    s = dict(SETTINGS if settings is None else settings)
    require(x.ndim == 2 and x.dtype == np.float32 and np.isfinite(x).all()
            and y.shape == groups.shape == (len(x),) and np.issubdtype(groups.dtype, np.integer)
            and np.isin(y, np.arange(N)).all() and coefficient in (0., .1)
            and 1 <= s['epochs'] <= 60 and 0 < s['steps_per_epoch'] <= 528,
            'Invalid training recipe')
    torch.manual_seed(s['seed'])
    torch.use_deterministic_algorithms(True)
    model = FlowModel(x.shape[1])
    initial = old.state_hash(model)
    weights = class_weights(population)
    xx, yy, gg, ww = map(torch.from_numpy, (x, y, groups, weights))
    optimizer = torch.optim.Adam(model.parameters(), lr=s['lr'], weight_decay=s['weight_decay'])
    order_hash = hashlib.sha256()
    seen = np.zeros(len(y), dtype=bool)
    class_seen = np.zeros(N, dtype=np.int64)
    examples = steps = eligible = 0
    observations = []
    for epoch in range(1, s['epochs']+1):
        model.train()
        ce_sum = con_sum = 0.
        for take in batches(y, s['seed'], epoch, s['steps_per_epoch'], s['per_class']):
            order_hash.update(take.astype('<i8').tobytes())
            seen[take] = True
            class_seen += np.bincount(y[take], minlength=N)
            optimizer.zero_grad(set_to_none=True)
            logits, projection = model(xx[take])
            ce = old.weighted_loss(logits, yy[take], ww[yy[take]])
            con, anchors = supcon(projection, yy[take], gg[take], s['temperature'])
            loss = ce+coefficient*con
            require(torch.isfinite(loss).item(), 'Nonfinite training loss')
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),
                    'Nonfinite gradient')
            optimizer.step()
            steps += 1
            examples += len(take)
            eligible += anchors
            ce_sum += float(ce.detach())
            con_sum += float(con.detach())
        if epoch in [1, 5, 20, 40, s['epochs']]:
            observations.append(dict(epoch=epoch, ce=ce_sum/s['steps_per_epoch'],
                supcon=con_sum/s['steps_per_epoch'], state_sha256=old.state_hash(model)))
            print(f'CONTRAST coefficient={coefficient} epoch={epoch} CE={ce_sum/s["steps_per_epoch"]:.6f}'
                  f' SupCon={con_sum/s["steps_per_epoch"]:.6f}', flush=True)
    return model, dict(settings=s, coefficient=coefficient,
        parameters=sum(p.numel() for p in model.parameters()), deployment_parameters=model.base.num_parameters(),
        initial_state_sha256=initial, final_state_sha256=old.state_hash(model),
        class_weights=weights.tolist(), examples_processed=examples, optimizer_steps=steps,
        sampled_class_counts=class_seen.tolist(), unique_rows_seen=int(seen.sum()),
        unique_class_counts=np.bincount(y[seen], minlength=N).tolist(),
        batch_order_sha256=order_hash.hexdigest(), eligible_contrastive_anchors=eligible,
        excluded_contrastive_anchors=examples-eligible, observations=observations)


def predict(model, x, batch=8192):
    before = old.state_hash(model)
    prediction = old_numeric_predict(model.base, x, batch)
    require(old.state_hash(model) == before, 'Prediction mutated model')
    return prediction


def old_numeric_predict(model, x, batch):
    # Keep inference identical to the tested PLE MLP, without a retrieval bank
    # or contrastive projection. Import locally to avoid a circular dependency.
    from src.eval.numerical_embedding import predict as implementation
    return implementation(model, x, batch)
