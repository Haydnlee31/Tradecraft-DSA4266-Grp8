"""Weighted full-MIME numerical kernel, not an authorized training pipeline.

Algorithm 1 / Table 5: https://arxiv.org/html/2008.03606
Local direction = ((1-beta1)*corrected_gradient + beta1*old_m)/(sqrt(old_v)+eps).
The SAME old moments remain frozen throughout every client's local block.
Only afterwards does the server update moments using the common anchor gradient.

This lagged-v decomposition has NO bias correction, clipping, or weight decay.
It is NOT ordinary PyTorch Adam and must have its own matched central control.
The caller must explicitly choose epsilon and initial v before an experiment:
zero v plus a tiny epsilon can make the first step dangerously large.
"""
import copy
import math

import torch

from src.federated.shared_adam import parameters, require, check_gradients, population_weights, client_gradient, aggregate_gradients


def initialize(model, initial_v=0.):
    require(math.isfinite(initial_v) and initial_v >= 0, 'Invalid initial second moment')
    ps = parameters(model)
    return {'m': tuple(torch.zeros_like(p) for p in ps),
            'v': tuple(torch.full_like(p, initial_v) for p in ps), 'round': 0}


def validate(state, refs, recipe):
    require(type(state['round']) is int and state['round'] >= 0, 'Invalid round counter')
    for key in ('m', 'v'):
        check_gradients(state[key], refs)
    require(all(bool((v >= 0).all()) for v in state['v']), 'Negative second moment')
    require(all(math.isfinite(recipe[k]) and 0 <= recipe[k] < 1 for k in ('beta1', 'beta2')),
            'Invalid momentum coefficient')
    require(all(math.isfinite(recipe[k]) and recipe[k] > 0 for k in ('lr', 'eps')),
            'Positive finite learning rate and epsilon required')


def directions(gradient, state, recipe):
    """Read old moments; deliberately do not mutate them or apply bias correction."""
    validate(state, gradient, recipe)
    result = tuple(((1-recipe['beta1'])*g+recipe['beta1']*m)/(v.sqrt()+recipe['eps'])
                   for g, m, v in zip(gradient, state['m'], state['v']))
    check_gradients(result, gradient)
    return result


def update_state(anchor_gradient, state, recipe):
    validate(state, anchor_gradient, recipe)
    new = {'m': tuple(recipe['beta1']*m+(1-recipe['beta1'])*g for m,g in zip(state['m'],anchor_gradient)),
           'v': tuple(recipe['beta2']*v+(1-recipe['beta2'])*g.square() for v,g in zip(state['v'],anchor_gradient)),
           'round': state['round']+1}
    validate(new, anchor_gradient, recipe)
    return new


def full_round(model, state, counts, full_batches, local_batch, criterion, recipe, local_steps):
    """Full participation, population-weighted anchors and final displacements.

    full_batches(client) yields (x, y, dropout_seed) covering each assigned row
    once. local_batch(client, step) returns that triple for a sampled minibatch.
    Paired gradients reuse the exact same rows AND stochastic masks. Full-data
    anchors with dropout enabled are sampled-mask estimates, not exact gradients
    of the loss averaged over all possible dropout masks.

    Providers own the row provenance. This kernel verifies counts, not row IDs;
    a real runner must separately verify exact coverage and client ownership.
    The common global class-weight denominator is supplied by the criterion.
    """
    ps = parameters(model)
    weights = population_weights(counts)
    validate(state, ps, recipe)
    require(type(local_steps) is int and local_steps > 0, 'Positive local step count required')
    anchors = []
    work = {'anchor_examples': 0, 'local_examples': 0, 'backward_examples': 0,
            'backward_batches': 0, 'local_updates': 0, 'server_state_updates': 1}
    # Stream each complete local dataset; do not concatenate all client rows.
    for client, count in enumerate(counts):
        total, seen = [torch.zeros_like(p) for p in ps], 0
        for x, y, seed in full_batches(client):
            gradient = client_gradient(model, (x,y), criterion, seed)
            for accumulated, g in zip(total, gradient):
                accumulated.add_(g, alpha=len(y)/count)
            seen += len(y)
            work['backward_batches'] += 1
        require(seen == count, 'Full anchor row count differs from assigned population')
        anchors.append(tuple(total))
        work['anchor_examples'] += seen
    common = aggregate_gradients(anchors, counts)
    deltas = []
    first_step_cancellation_exact = True
    for client in range(len(counts)):
        local = copy.deepcopy(model)
        for step in range(local_steps):
            x, y, seed = local_batch(client, step)
            current = client_gradient(local, (x,y), criterion, seed)
            at_anchor = client_gradient(model, (x,y), criterion, seed)
            if step == 0:
                first_step_cancellation_exact &= all(torch.equal(g,h) for g,h in zip(current,at_anchor))
            corrected = tuple(g-h+c for g,h,c in zip(current, at_anchor, common))
            direction = directions(corrected, state, recipe)
            with torch.no_grad():
                for p, d in zip(local.parameters(), direction):
                    p.add_(d, alpha=-recipe['lr'])
            require(all(bool(torch.isfinite(p).all()) for p in local.parameters()), 'Nonfinite local weights')
            work['local_examples'] += len(y)
            work['backward_batches'] += 2
            work['local_updates'] += 1
        deltas.append(tuple(p.detach()-start.detach() for p,start in zip(local.parameters(),ps)))
    require(first_step_cancellation_exact, 'Same-mask first-step correction failed')
    result = copy.deepcopy(model)
    average_delta = aggregate_gradients(deltas, counts)
    with torch.no_grad():
        for p, delta in zip(result.parameters(), average_delta):
            p.add_(delta)
    work['backward_examples'] = work['anchor_examples']+2*work['local_examples']
    # Same clients provide anchor and local work; cache their anchor uploads.
    # Down: x, m, v, c. Up: anchor gradient, final model. No protocol overhead.
    parameter_bytes = sum(p.numel()*p.element_size() for p in ps)
    return result, update_state(common, state, recipe), {
        'work': work, 'first_step_cancellation_exact': first_step_cancellation_exact,
        'estimated_payload_bytes': 6*parameter_bytes*len(weights),
        'measured_network_traffic': False, 'transport_overhead_included': False,
        'full_participation': True, 'local_steps': local_steps,
        'optimizer': 'Table5 lagged-v Adam, frozen local state, no bias correction/decay/clipping',
    }
