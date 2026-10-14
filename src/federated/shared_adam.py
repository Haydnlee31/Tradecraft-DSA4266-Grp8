"""CPU numerical control for population-weighted gradients and server Adam.

This is synchronous gradient aggregation, NOT FedAvg, FedAdam on client deltas,
or MIME. Clients perform no local optimizer step. One server optimizer keeps its
moments across synchronization steps. Raw batches appear only in the independent
central reference; the server aggregation function receives gradient tensors.
"""
import copy
import math

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parameters(model):
    ps = tuple(model.parameters())
    require(bool(ps) and all(p.requires_grad and p.device.type == 'cpu' and
                            p.dtype in (torch.float32, torch.float64) for p in ps),
            'Trainable CPU float32/float64 parameters required')
    require(len({p.dtype for p in ps}) == 1, 'Mixed parameter dtypes unsupported')
    # Training BatchNorm would mutate client buffers and add a separate
    # aggregation policy. This bridge deliberately uses buffer-free LayerNorm.
    require(not list(model.buffers()), 'Stateful model buffers unsupported')
    return ps


def population_weights(counts):
    require(len(counts) > 0 and all(type(n) is int and n > 0 for n in counts),
            'Positive integer client population counts required')
    total = sum(counts)
    return [n / total for n in counts]


def check_gradients(grads, refs):
    require(len(grads) == len(refs), 'Gradient tensor count differs')
    for g, p in zip(grads, refs):
        require(g.shape == p.shape and g.dtype == p.dtype and g.device == p.device
                and bool(torch.isfinite(g).all()), 'Invalid gradient payload')


def client_gradient(model, batch, criterion, dropout_seed):
    """Return a detached gradient message without changing model/RNG state.

    A separate seed for EACH matched microbatch replays the same dropout masks
    centrally. Seeding once at startup would not establish mask correspondence.
    fork_rng restores the caller's CPU RNG after this isolated forward/backward.
    """
    ps = parameters(model)
    x, y = batch
    require(len(x) == len(y) and len(y) > 0, 'Nonempty matching batch required')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(dropout_seed)
        loss = criterion(model(x), y)
        require(loss.ndim == 0 and bool(torch.isfinite(loss)), 'Nonfinite/scalar loss required')
        grads = tuple(g.detach().clone() for g in torch.autograd.grad(loss, ps))
    check_gradients(grads, ps)
    return grads


def aggregate_gradients(messages, counts):
    """Server receives gradients plus assigned population sizes, NOT raw rows.

    With equal client minibatches but unequal populations, uniform concatenated
    averaging would target a DIFFERENT sampled objective. Weight by population,
    not by this step's equal processed-example counts.
    """
    weights = population_weights(counts)
    require(len(messages) == len(weights) and bool(messages[0]), 'Missing client messages')
    refs = messages[0]
    for message in messages:
        check_gradients(message, refs)
    total = [torch.zeros_like(g) for g in refs]
    for message, weight in zip(messages, weights):
        for accumulated, g in zip(total, message):
            accumulated.add_(g, alpha=weight)
    check_gradients(total, refs)
    return tuple(total)


def central_accumulated_gradient(model, batches, counts, criterion, seeds):
    """Independent autograd reference: weight LOSSES before differentiating.

    Federated aggregation instead weights independently computed GRADIENTS.
    Neither path performs an optimizer step between client microbatches.
    Sequential minibatch Adam is intentionally not the equivalence target.
    """
    ps = parameters(model)
    weights = population_weights(counts)
    require(len(batches) == len(seeds) == len(weights), 'Missing reference batches/seeds')
    model.zero_grad(set_to_none=True)
    try:
        for (x, y), weight, seed in zip(batches, weights, seeds):
            require(len(x) == len(y) and len(y) > 0, 'Nonempty matching batch required')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                loss = criterion(model(x), y) * weight
                require(loss.ndim == 0 and bool(torch.isfinite(loss)), 'Invalid reference loss')
                loss.backward()
        grads = tuple(p.grad.detach().clone() for p in ps)
    finally:
        model.zero_grad(set_to_none=True)
    check_gradients(grads, ps)
    return grads


def make_optimizer(model, recipe):
    # Explicit single-tensor Adam path: bias correction, persistent m/v, coupled
    # L2 weight decay. This is not AdamW or MIME's lagged-moment base updater.
    return torch.optim.Adam(parameters(model), lr=recipe['lr'],
                            betas=tuple(recipe['betas']), eps=recipe['eps'],
                            weight_decay=recipe['weight_decay'], amsgrad=False,
                            foreach=False, fused=False)


def apply_gradient(model, optimizer, grads, clip_norm=None):
    ps = parameters(model)
    require([id(p) for g in optimizer.param_groups for p in g['params']] == [id(p) for p in ps],
            'Optimizer is attached to a different model')
    check_gradients(grads, ps)
    require(clip_norm is None or math.isfinite(clip_norm) and clip_norm > 0, 'Invalid clip norm')
    optimizer.zero_grad(set_to_none=True)
    for p, g in zip(ps, grads):
        p.grad = g.detach().clone()
    # Clip AFTER aggregation. Adam adds its coupled L2 term afterwards; the
    # central reference uses exactly the same order and calls Adam only once.
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(ps, clip_norm, error_if_nonfinite=True, foreach=False)
    applied = tuple(p.grad.clone() for p in ps)
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in ps), 'Nonfinite parameters')
    optimizer.zero_grad(set_to_none=True)
    return applied


def optimizer_tensors(model, optimizer):
    """Expose moments in parameter order for equality/recovery audits."""
    ps = parameters(model)
    return {key: tuple(optimizer.state[p][key].detach().clone() for p in ps)
            for key in ('step', 'exp_avg', 'exp_avg_sq')}


def restore(initial_model, snapshot, recipe):
    """Restore optimizer moments AND counters, never only model weights."""
    model = copy.deepcopy(initial_model)
    model.load_state_dict(snapshot['model'], strict=True)
    optimizer = make_optimizer(model, recipe)
    optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    return model, optimizer


def residual(left, right, atol, rtol):
    require(len(left) == len(right) and len(left) > 0, 'Missing comparison tensors')
    require(all(a.shape == b.shape for a, b in zip(left, right)), 'Comparison shape mismatch')
    a = torch.cat([t.detach().reshape(-1).double() for t in left])
    b = torch.cat([t.detach().reshape(-1).double() for t in right])
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), 'Nonfinite audit values')
    d = a-b
    return {'within_tolerance': bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
            'bit_exact': bool(torch.equal(a, b)), 'max_abs': float(d.abs().max()),
            'l2': float(torch.linalg.vector_norm(d))}
