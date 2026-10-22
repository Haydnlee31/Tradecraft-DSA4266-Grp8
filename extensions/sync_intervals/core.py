"""Local SGD with normalized messages and persistent server Adam.

FedOpt-inspired adaptation: g_i = mean of local path gradients, mathematically
(theta_start - theta_end)/(local_lr*K) for plain SGD. Sending accumulated
gradients avoids subtractive cancellation in float32 parameter differences.
K=1 directly uses the original gradient helper, giving a bit-exact bridge.
No momentum, decay, clipping, compression or client Adam in interval arms.
"""
import copy
import math

import torch

from src.federated import shared_adam as shared


def interval_step(model, optimizer, counts, batch, criterion, steps, local_lr, seed):
    shared.require(type(steps) is int and steps > 0, 'Positive integer interval required')
    shared.require(math.isfinite(local_lr) and local_lr > 0, 'Positive finite local_lr required')
    shared.parameters(model)
    shared.population_weights(counts)
    model.train()
    messages = []
    examples = 0
    for client in range(len(counts)):
        if steps == 1:
            xy = batch(client, 0)
            messages.append(shared.client_gradient(model, xy, criterion, seed))
            examples += len(xy[1])
            continue
        local = copy.deepcopy(model).train()
        mean = [torch.zeros_like(p) for p in local.parameters()]
        for step in range(steps):
            xy = batch(client, step)
            grads = shared.client_gradient(local, xy, criterion, seed)
            with torch.no_grad():
                for p, g, accumulated in zip(local.parameters(), grads, mean):
                    accumulated.add_(g, alpha=1/steps)
                    p.add_(g, alpha=-local_lr)
            shared.require(all(bool(torch.isfinite(p).all()) for p in local.parameters()),
                           'Nonfinite local SGD parameters')
            examples += len(xy[1])
        messages.append(tuple(mean))
    aggregate = shared.aggregate_gradients(messages, counts)
    shared.apply_gradient(model, optimizer, aggregate)
    return examples


def fedavg_step(model, counts, batch, criterion, steps, local_lr, seed, adam_recipe=None):
    """Independent optimizer controls: plain local SGD or reset client Adam."""
    shared.require(type(steps) is int and steps > 0, 'Positive integer interval required')
    shared.require(math.isfinite(local_lr) and local_lr > 0, 'Invalid local rate')
    shared.parameters(model)
    shared.population_weights(counts)
    model.train()
    messages, examples = [], 0
    for client in range(len(counts)):
        local = copy.deepcopy(model)
        optimizer = shared.make_optimizer(local, adam_recipe) if adam_recipe is not None else None
        for step in range(steps):
            xy = batch(client, step)
            grads = shared.client_gradient(local, xy, criterion, seed)
            if optimizer is not None:
                shared.apply_gradient(local, optimizer, grads)
            else:
                with torch.no_grad():
                    for p, g in zip(local.parameters(), grads):
                        p.add_(g, alpha=-local_lr)
                shared.require(all(bool(torch.isfinite(p).all()) for p in local.parameters()),
                               'Nonfinite SGD control parameters')
            examples += len(xy[1])
        messages.append(tuple(p.detach()-q.detach() for p, q in zip(local.parameters(), model.parameters())))
    with torch.no_grad():
        for p, delta in zip(model.parameters(), shared.aggregate_gradients(messages, counts)):
            p.add_(delta)
    shared.require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite aggregate')
    return examples
