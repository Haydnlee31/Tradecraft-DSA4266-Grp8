"""Population-weighted MimeLite with full round-start anchor gradients.

Algorithm 1 and Table 5: https://arxiv.org/html/2008.03606
Unlike full MIME, local updates use uncorrected minibatch gradients. They still
read the SAME frozen old server moments; the server updates those moments once
from full gradients at the round-start model, never from local-path gradients.
This is the paper's lagged-v base, not bias-corrected torch.optim.Adam.
"""
import copy

import torch

from src.federated import mime_adam as mime
from src.federated import shared_adam as shared


def lite_round(model, state, counts, full_batches, local_batch, criterion, recipe, local_steps):
    ps = shared.parameters(model)
    shared.population_weights(counts)
    mime.validate(state, ps, recipe)
    shared.require(type(local_steps) is int and local_steps > 0, 'Positive local step count required')
    anchors, deltas = [], []
    work = dict(anchor_examples=0, local_examples=0, backward_examples=0,
                backward_batches=0, local_updates=0, server_state_updates=1)
    for client, count in enumerate(counts):
        # The provider streams the complete owned fit subset exactly once. It
        # remains responsible for row identity/coverage, not only the count.
        accumulated, seen = [torch.zeros_like(p) for p in ps], 0
        for x, y, seed in full_batches(client):
            g = shared.client_gradient(model, (x, y), criterion, seed)
            for total, value in zip(accumulated, g):
                total.add_(value, alpha=len(y)/count)
            seen += len(y)
            work['backward_batches'] += 1
        shared.require(seen == count, 'Full anchor row count differs from assigned population')
        anchors.append(tuple(accumulated))
        work['anchor_examples'] += seen

        local = copy.deepcopy(model)
        for step in range(local_steps):
            x, y, seed = local_batch(client, step)
            gradient = shared.client_gradient(local, (x, y), criterion, seed)
            direction = mime.directions(gradient, state, recipe)
            with torch.no_grad():
                for p, d in zip(local.parameters(), direction):
                    p.add_(d, alpha=-recipe['lr'])
            shared.require(all(bool(torch.isfinite(p).all()) for p in local.parameters()), 'Nonfinite local model')
            work['local_examples'] += len(y)
            work['backward_batches'] += 1
            work['local_updates'] += 1
        deltas.append(tuple(p.detach()-q.detach() for p, q in zip(local.parameters(), ps)))

    result = copy.deepcopy(model)
    with torch.no_grad():
        for p, d in zip(result.parameters(), shared.aggregate_gradients(deltas, counts)):
            p.add_(d)
    moments = mime.update_state(shared.aggregate_gradients(anchors, counts), state, recipe)
    work['backward_examples'] = work['anchor_examples']+work['local_examples']
    # Down: model, m, v. Up: full anchor gradient, local model. Anchor uploads
    # need no pre-local server broadcast, so they share one round-end barrier.
    return result, moments, {'work': work,
        'estimated_payload_bytes': 5*len(counts)*sum(p.numel()*p.element_size() for p in ps),
        'logical_barriers': 1, 'measured_network_traffic': False,
        'transport_overhead_included': False}
