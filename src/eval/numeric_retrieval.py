"""A bounded, TabR-inspired experiment, NOT a reproduction of TabR.

TabR learns neighbour search as well as attention. Here the expensive candidate
search is frozen in fitting-quantile coordinates. A small network learns how
to weight those candidates using keys, label embeddings and key differences.
This makes the limitation explicit and keeps the local experiment affordable.
"""
import numpy as np
import torch
from torch import nn
from sklearn.neighbors import KDTree

from src.data.label_map import CLASSES
from src.eval import matched_neural as old
from src.eval.official_evidence import require


def coordinates(x, edges):
    """One clipped rank-like coordinate per feature, using FIT edges only.

    Collapsed quantiles are evenly spaced here, so these are not exact
    empirical percentiles. Binary features remain binary; constants are zero.
    """
    require(x.ndim == 2 and x.shape[1] == len(edges) and np.isfinite(x).all(),
            'Invalid retrieval inputs')
    result = np.zeros(x.shape, np.float64)
    for j, e in enumerate(edges):
        e = np.asarray(e, dtype=np.float64)
        require(len(e) > 0 and np.isfinite(e).all() and (np.diff(e) > 0).all(), 'Bad edges')
        if len(e) > 1:
            result[:, j] = np.interp(x[:, j], e, np.linspace(0, 1, len(e)))
    return result


def bank(x, y, cap=2048, seed=20261013):
    """Unique FIT vectors only; never select references from evaluation rows.

    For conflicting labels on the exact same retained vector, keep the first
    FIT row (not a label-informed favourite). Then cap each class. Reference
    balance is a design choice, not a new training prior or a calibrated risk.
    """
    require(x.ndim == 2 and x.dtype == np.float32 and np.isfinite(x).all()
            and y.shape == (len(x),) and np.isin(y, np.arange(8)).all()
            and type(cap) is int and cap > 0, 'Invalid bank input')
    # Numeric uniqueness treats +0 and -0 as the same value.
    _, first, groups = np.unique(x, axis=0, return_index=True, return_inverse=True)
    first = np.sort(first)
    rng = np.random.default_rng(seed)
    chosen = []
    for label in range(len(CLASSES)):
        eligible = first[y[first] == label]
        chosen.extend(rng.choice(eligible, min(cap, len(eligible)), replace=False))
    chosen = np.sort(np.asarray(chosen, np.int64))
    require(len(chosen) > 8, 'Reference bank too small')
    return chosen, groups


def neighbours(query, reference, k=8, query_groups=None, reference_groups=None):
    """Bounded exact KD-tree search with self/duplicate exclusion.

    The bank has at most one row per original vector, hence one extra result
    suffices to exclude its group. Search ties follow sklearn's fixed tree;
    returned candidates are ordered by distance then bank index. No labels
    enter search. Callers must separately bind row provenance.
    """
    require(query.ndim == reference.ndim == 2 and query.shape[1] == reference.shape[1]
            and np.isfinite(query).all() and np.isfinite(reference).all()
            and type(k) is int and 0 < k < len(reference), 'Invalid search')
    exclude = query_groups is not None
    require(exclude == (reference_groups is not None), 'Incomplete exclusion groups')
    if exclude:
        require(query_groups.shape == (len(query),) and reference_groups.shape == (len(reference),)
                and len(np.unique(reference_groups)) == len(reference), 'Bank groups must be unique')
    tree = KDTree(reference, leaf_size=40)
    result = np.empty((len(query), k), np.int32)
    for start in range(0, len(query), 4096):
        distance, index = tree.query(query[start:start+4096], k=k+1, sort_results=True)
        if exclude:
            distance[query_groups[start:start+len(index), None] == reference_groups[index]] = np.inf
        order = np.lexsort((index, distance), axis=1)[:, :k]
        index = np.take_along_axis(index, order, axis=1)
        require(np.isfinite(np.take_along_axis(distance, order, axis=1)).all(), 'Insufficient neighbours')
        if exclude:
            require(not np.any(query_groups[start:start+len(index), None] == reference_groups[index]),
                    'Self or duplicate label leak')
        result[start:start+len(index)] = index
    return result


def vote(context, bank_y):
    """Unweighted neighbour voting control; no threshold/temperature search."""
    require(context.ndim == 2 and np.issubdtype(context.dtype, np.integer)
            and (context >= 0).all() and (context < len(bank_y)).all()
            and np.isin(bank_y, np.arange(8)).all(), 'Invalid vote')
    votes = np.eye(8, dtype=np.int16)[bank_y[context]].sum(1)
    return votes.argmax(1).astype(np.uint8)


class RetrievalModel(nn.Module):
    def __init__(self, dimensions, retrieval=True):
        super().__init__()
        self.retrieval = retrieval
        self.encoder = nn.Sequential(nn.Linear(dimensions, 64), nn.LayerNorm(64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU())
        self.key = nn.Linear(32, 32)
        self.delta = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        self.labels = nn.Embedding(8, 32) if retrieval else None
        self.head = nn.Sequential(nn.LayerNorm(32), nn.ReLU(), nn.Dropout(.2), nn.Linear(32, 8))

    def forward(self, x, reference=None, reference_y=None, context=None):
        h = self.encoder(x)
        key = self.key(h)
        if self.retrieval:
            # Encode each distinct reference once per batch, retaining gradients
            # to the shared encoder. Evaluation never supplies query labels.
            unique, inverse = torch.unique(context, sorted=True, return_inverse=True)
            reference_key = self.key(self.encoder(reference[unique]))[inverse]
            difference = key[:, None] - reference_key
            weight = torch.softmax(-difference.square().sum(-1), dim=1)
            values = self.labels(reference_y[context]) + self.delta(difference)
            h = h + (weight[..., None]*values).sum(1)
        else:
            # All key/delta parameters remain active in the no-reference control.
            # Only the 8x32 label embedding (256 parameters) is absent.
            h = h + self.delta(key)
        return self.head(h)


def predict(model, x, reference, reference_y, context, batch=2048):
    before = old.state_hash(model)
    model.eval()
    reference, reference_y = torch.from_numpy(reference), torch.from_numpy(reference_y)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xx = torch.from_numpy(x[start:start+batch])
            cc = torch.from_numpy(context[start:start+batch].astype(np.int64))
            logits = model(xx, reference, reference_y, cc)
            require(torch.isfinite(logits).all().item(), 'Nonfinite prediction')
            predictions.append(logits.argmax(1).numpy().astype(np.uint8))
    require(old.state_hash(model) == before, 'Inference mutated weights')
    return np.concatenate(predictions)


def fit(x, y, population, reference, reference_y, context, retrieval, seed=7, epochs=60, batch=512):
    require(x.ndim == 2 and x.dtype == reference.dtype == np.float32
            and reference.shape[1] == x.shape[1] and y.shape == (len(x),)
            and reference_y.shape == (len(reference),) and context.shape[0] == len(x)
            and np.isfinite(x).all() and np.isfinite(reference).all()
            and (context >= 0).all() and (context < len(reference)).all()
            and 1 <= epochs <= 60 and batch >= 2, 'Invalid fit')
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = RetrievalModel(x.shape[1], retrieval)
    weights = old.fit_weights(y, population)
    xx, yy, ww = map(torch.from_numpy, (x, y, weights))
    rr, ry = map(torch.from_numpy, (reference, reference_y))
    cc = torch.from_numpy(context.astype(np.int64))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    initial = old.state_hash(model)
    examples = steps = contexts = 0
    observations = []
    for epoch in range(1, epochs+1):
        order = np.random.default_rng(seed+epoch).permutation(len(y))
        model.train()
        loss_sum = 0.
        for start in range(0, len(y), batch):
            take = order[start:start+batch]
            optimizer.zero_grad(set_to_none=True)
            loss = old.weighted_loss(model(xx[take], rr, ry, cc[take]), yy[take], ww[take])
            require(torch.isfinite(loss).item(), 'Nonfinite training loss')
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),
                    'Nonfinite gradient')
            optimizer.step()
            examples += len(take)
            steps += 1
            contexts += len(take)*context.shape[1] if retrieval else 0
            loss_sum += float(loss.detach())*len(take)
        if epoch in [1, 5, 20, 40, epochs]:
            observations.append(dict(epoch=epoch, weighted_ce=loss_sum/len(y), state_sha256=old.state_hash(model)))
            print(f'RETRIEVAL enabled={retrieval} epoch={epoch} fit_loss={loss_sum/len(y):.6f}', flush=True)
    return model, dict(seed=seed, retrieval=retrieval, epochs=epochs, batch_size=batch,
        parameters=sum(p.numel() for p in model.parameters()), initial_state_sha256=initial,
        final_state_sha256=old.state_hash(model), examples_processed=examples, optimizer_steps=steps,
        reference_uses=contexts, sample_weights_sha256=old.vector_hash(weights.astype('<f8')),
        observations=observations)
