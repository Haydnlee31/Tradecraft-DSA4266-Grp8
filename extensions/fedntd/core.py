"""FedNTD adaptation to the project's population-weighted, reset-Adam control.

Reference: Lee et al., NeurIPS 2022, https://arxiv.org/abs/2106.03097.
The supervised criterion is supplied by the caller (fixed-training-mean weighted
CE in this experiment). Distillation is unweighted batch-mean KL(teacher ||
student), after removing each row's true-class logit, multiplied by T**2.
"""
import copy
import math

import torch
from torch.nn import functional as F

from src.federated import shared_adam as shared


def validate_settings(beta, temperature):
    shared.require(math.isfinite(beta) and beta >= 0, 'beta must be finite and nonnegative')
    shared.require(math.isfinite(temperature) and temperature > 0,
                   'temperature must be finite and positive')


def not_true_kl(student, teacher, target, temperature=1.0):
    """Detach the teacher even if the caller passes a differentiable tensor."""
    validate_settings(0., temperature)
    shared.require(student.ndim == 2 and student.shape == teacher.shape
                   and student.shape[0] > 0 and student.shape[1] > 1,
                   'Nonempty matching batch-by-class logits required')
    shared.require(student.is_floating_point() and student.dtype == teacher.dtype
                   and student.device == teacher.device == target.device,
                   'Logit dtype/device mismatch')
    shared.require(target.shape == (len(student),) and target.dtype == torch.long
                   and bool(((target >= 0) & (target < student.shape[1])).all()),
                   'Targets must be valid class indices')
    shared.require(bool(torch.isfinite(student).all() and torch.isfinite(teacher).all()),
                   'Nonfinite logits')
    mask = torch.ones_like(student, dtype=torch.bool).scatter_(1, target[:, None], False)
    s = student[mask].reshape(len(student), -1) / temperature
    t = teacher.detach()[mask].reshape(len(student), -1) / temperature
    # log_target avoids probability underflow for confident teacher predictions.
    loss = F.kl_div(F.log_softmax(s, dim=1), F.log_softmax(t, dim=1),
                    reduction='batchmean', log_target=True) * temperature**2
    shared.require(bool(torch.isfinite(loss)), 'Nonfinite distillation loss')
    return loss


def frozen_teacher(model):
    teacher = copy.deepcopy(model).eval()
    teacher.requires_grad_(False)
    return teacher


def client_gradient(model, batch, criterion, dropout_seed, teacher, beta, temperature):
    validate_settings(beta, temperature)
    if beta == 0:
        # Exact original path: no teacher copy/forward or extra RNG consumption.
        return shared.client_gradient(model, batch, criterion, dropout_seed)
    shared.require(teacher is not None and not teacher.training
                   and all(not p.requires_grad for p in teacher.parameters()),
                   'A frozen eval-mode round-start teacher is required')
    ps = shared.parameters(model)
    x, y = batch
    shared.require(len(x) == len(y) and len(y) > 0, 'Nonempty matching batch required')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(dropout_seed)
        logits = model(x)
        with torch.no_grad():
            teacher_logits = teacher(x)
        loss = criterion(logits, y) + beta * not_true_kl(logits, teacher_logits, y, temperature)
        shared.require(loss.ndim == 0 and bool(torch.isfinite(loss)), 'Invalid combined loss')
        grads = tuple(g.detach().clone() for g in torch.autograd.grad(loss, ps))
    shared.check_gradients(grads, ps)
    return grads


def round_update(model, counts, batch, criterion, recipe, local_steps, seed,
                 beta=0.1, temperature=1.0):
    """Return a new global model and work counters; leave input model unchanged.

    batch(client, step) supplies only that client's fit rows. CPU, buffer-free
    models only, matching shared_adam's explicit numerical contract. All clients
    use the same frozen global teacher throughout the round. Optimizers reset
    per client and round. No local optimizer/teacher state is communicated.
    """
    validate_settings(beta, temperature)
    shared.parameters(model)
    shared.population_weights(counts)
    shared.require(type(local_steps) is int and local_steps > 0, 'Positive local_steps required')
    teacher = frozen_teacher(model) if beta else None
    deltas = []
    examples = 0
    for client in range(len(counts)):
        local = copy.deepcopy(model).train()
        optimizer = shared.make_optimizer(local, recipe)
        for step in range(local_steps):
            xy = batch(client, step)
            gradient = client_gradient(local, xy, criterion, seed, teacher, beta, temperature)
            shared.apply_gradient(local, optimizer, gradient)
            examples += len(xy[1])
        deltas.append(tuple(p.detach()-q.detach() for p, q in zip(local.parameters(), model.parameters())))
    updated = copy.deepcopy(model).train()
    with torch.no_grad():
        for p, delta in zip(updated.parameters(), shared.aggregate_gradients(deltas, counts)):
            p.add_(delta)
    shared.require(all(bool(torch.isfinite(p).all()) for p in updated.parameters()), 'Nonfinite model')
    return updated, {
        'backward_examples': examples,
        'backward_batches': len(counts)*local_steps,
        'optimizer_steps': len(counts)*local_steps,
        'teacher_forward_examples': examples if beta else 0,
        'teacher_forward_batches': len(counts)*local_steps if beta else 0,
        'synchronization_rounds': 1,
        'logical_barriers': 1,
        'estimated_payload_bytes': 2*len(counts)*sum(p.numel()*p.element_size() for p in model.parameters()),
    }
