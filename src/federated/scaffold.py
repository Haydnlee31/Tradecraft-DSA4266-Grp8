"""Full-participation, data-size-weighted SCAFFOLD with plain SGD.

Reference: Karimireddy et al., ICML 2020, Algorithm 1, Option II.
For p_i = n_i / N, maintain c = sum_i p_i c_i. All clients take the SAME
number K of local steps. No momentum, Adam, clipping or partial participation
is hidden in this implementation. The weighted extension and sampling policy
are explicit; this is not a claim to reproduce the paper's experiments.
"""
import copy
import hashlib

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.eval.federated_class_safeguard import require, vector, state_digest
from src.federated.partition import check_exact_cover
from src.models.official_losses import build_official_criterion
from src.utils.seed import set_seed, client_seed


def parameters(model):
    result = {k: p.detach().clone() for k, p in model.named_parameters()}
    require(set(result) == set(model.state_dict()), 'Only buffer-free models are supported')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in result.values()),
            'This bounded implementation supports CPU FP32 parameters only')
    return result


def initialize_controls(model, clients):
    require(clients > 0, 'Positive client count required')
    zero = {k: torch.zeros_like(p) for k, p in parameters(model).items()}
    return {'server': copy.deepcopy(zero), 'clients': [copy.deepcopy(zero) for _ in range(clients)]}


def aggregate(states, sizes):
    """Flower weights by assigned population, NOT equal sampled work counts."""
    from flwr.app import ArrayRecord, MetricRecord, RecordDict
    from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords
    require(len(states) == len(sizes) and len(states) > 0 and all(n > 0 for n in sizes), 'Invalid weights')
    replies = [RecordDict({'arrays': ArrayRecord(s), 'metrics': MetricRecord({'assigned-examples': int(n)})})
               for s, n in zip(states, sizes)]
    return aggregate_arrayrecords(replies, 'assigned-examples').to_torch_state_dict()


def control_digest(controls):
    h = hashlib.sha256()
    for state in [controls['server']]+controls['clients']:
        for key, p in state.items():
            h.update(key.encode())
            h.update(str(tuple(p.shape)).encode())
            h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def verify_controls(model, controls, sizes):
    """Check shape/finiteness and the weighted server invariant at every round."""
    expected = parameters(model)
    require(len(controls['clients']) == len(sizes), 'Wrong number of client controls')
    for state in [controls['server']]+controls['clients']:
        require(list(state) == list(expected), 'Control parameter keys changed')
        for key, p in state.items():
            require(p.shape == expected[key].shape and p.dtype == torch.float32
                    and p.device.type == 'cpu' and bool(torch.isfinite(p).all()), 'Invalid control tensor')
    weights = np.asarray(sizes, dtype=np.float64)/sum(sizes)
    residual, scale = 0., 1.
    for key in expected:
        mean = sum(float(w)*c[key].double() for w, c in zip(weights, controls['clients']))
        residual = max(residual, float((mean-controls['server'][key].double()).abs().max()))
        scale = max(scale, float(mean.abs().max()))
    require(residual <= 1e-6*scale, 'Weighted server control invariant violated')
    return residual


def option_two(start, end, ci, c, steps, lr):
    """c_i+ = c_i - c + (x - y_i)/(K*lr), including the L2 gradient.

    Compute differences in FP64, then store the actual FP32 control that will
    be sent/used. The formula assumes constant lr and plain un-clipped SGD.
    """
    require(steps > 0 and np.isfinite(lr) and lr > 0, 'Invalid Option II step scale')
    return {k: (ci[k].double()-c[k].double()+(start[k].double()-end[k].double())/(steps*lr)).float()
            for k in start}


def sampled_batches(data, rows, batch_size, steps, seed):
    """Uniform WITH replacement within a client; pooled uniform for central.

    Equal K avoids confounding SCAFFOLD with unequal optimizer-step counts.
    Do not claim one full epoch or unique-row coverage. Each sampled row index
    is hashed in order and class counts are recorded without exporting rows.
    """
    rows = np.asarray(rows, dtype=np.int64)
    x, y = data.arrays['train']
    require(rows.ndim == 1 and len(rows) > 0 and np.all((rows >= 0) & (rows < len(y)))
            and batch_size >= 2 and steps > 0, 'Invalid training sampler')
    meter = {'examples_processed': 0, 'optimizer_steps': 0, 'sampled_class_counts': [0]*len(CLASSES),
             'sampled_rows_sha256': None}
    def batches():
        rng, digest = np.random.default_rng(seed), hashlib.sha256()
        counts = np.zeros(len(CLASSES), dtype=np.int64)
        for _ in range(steps):
            take = rows[rng.integers(0, len(rows), size=batch_size)]
            digest.update(take.astype('<i8', copy=False).tobytes())
            counts += np.bincount(y[take], minlength=len(CLASSES))
            meter.update(examples_processed=meter['examples_processed']+len(take),
                         optimizer_steps=meter['optimizer_steps']+1,
                         sampled_class_counts=counts.tolist(), sampled_rows_sha256=digest.hexdigest())
            yield np.array(x[take], copy=True), np.array(y[take], copy=True)
    return batches(), meter


def local_sgd(model, batches, criterion, lr, weight_decay, ci=None, c=None):
    """Return a fresh local model; preserve caller model/control tensors.

    L2 is added BEFORE the control correction, so Option II estimates the
    average stochastic gradient of the regularized local objective. Accumulated
    raw gradients below audit the telescoping identity, not extra backprop.
    """
    require(np.isfinite(lr) and lr > 0 and np.isfinite(weight_decay) and weight_decay >= 0,
            'Invalid SGD settings')
    require((ci is None) == (c is None), 'Both controls or neither required')
    local = copy.deepcopy(model).train()
    optimizer = torch.optim.SGD(local.parameters(), lr=lr, momentum=0, weight_decay=0)
    raw_sum = {k: torch.zeros_like(p, dtype=torch.float64) for k, p in local.named_parameters()}
    correction = {k: c[k]-ci[k] for k in c} if c is not None else None
    total, examples, steps = 0., 0, 0
    for x, y in batches:
        loss = criterion(local(torch.from_numpy(x)), torch.from_numpy(y))
        require(bool(torch.isfinite(loss)), 'Nonfinite training loss; stop without retuning')
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for key, p in local.named_parameters():
                p.grad.add_(p, alpha=weight_decay)
                raw_sum[key] += p.grad.double()
                if correction is not None:
                    p.grad.add_(correction[key])
                require(bool(torch.isfinite(p.grad).all()), 'Nonfinite corrected gradient')
        optimizer.step()
        total += float(loss.detach())*len(y)
        examples += len(y)
        steps += 1
    require(steps > 0 and all(bool(torch.isfinite(p).all()) for p in local.parameters()), 'Invalid local result')
    return local, {'mean_task_loss': total/examples, 'examples_processed': examples, 'optimizer_steps': steps}, {
        k: value/steps for k, value in raw_sum.items()}


def federated_round(model, data, parts, recipe, step, controls=None):
    check_exact_cover(parts, data.manifest['train_rows'])
    sizes = list(map(len, parts))
    if controls is not None:
        verify_controls(model, controls, sizes)
    start, states, next_ci, clients, deltas = parameters(model), [], [], [], []
    criterion = build_official_criterion('sqrt_weighted_ce', data.manifest['train_class_counts'], 'fixed_train_mean')
    for client, rows in enumerate(parts):
        seed = client_seed(recipe['seed'], client, step)
        set_seed(seed)
        torch.use_deterministic_algorithms(True)
        batches, meter = sampled_batches(data, rows, recipe['batch_size'], recipe['local_steps'], seed)
        ci, c = (controls['clients'][client], controls['server']) if controls is not None else (None, None)
        local, work, mean_raw = local_sgd(model, batches, criterion, recipe['lr'], recipe['weight_decay'], ci, c)
        state = parameters(local)
        states.append(state)
        deltas.append(vector(local)-vector(model))
        require(work['examples_processed'] == meter['examples_processed']
                and work['optimizer_steps'] == meter['optimizer_steps'], 'Sample/work mismatch')
        row = {'client': client, 'assigned_examples': len(rows), 'seed': seed, **work, **meter}
        if controls is not None:
            updated = option_two(start, state, ci, c, recipe['local_steps'], recipe['lr'])
            row['option_two_identity_residual_l2'] = float(torch.sqrt(sum(
                ((updated[k].double()-mean_raw[k])**2).sum() for k in start)))
            next_ci.append(updated)
        clients.append(row)
    result = copy.deepcopy(model)
    result.load_state_dict(aggregate(states, sizes))
    new_controls = None
    if controls is not None:
        # Full participation lets us rebuild the weighted mean instead of
        # accumulating roundoff in c += sum p_i (c_i+ - c_i).
        new_controls = {'server': aggregate(next_ci, sizes), 'clients': next_ci}
    weights = np.asarray(sizes)/sum(sizes)
    ds = np.stack(deltas)
    mean = weights @ ds
    return result, new_controls, {
        'clients': clients, 'model_sha256': state_digest(result),
        'aggregate_delta_norm': float(np.linalg.norm(vector(result)-vector(model))),
        'client_update_dispersion_rms': float(np.sqrt(weights @ np.sum((ds-mean)**2, axis=1))),
        'control_sha256': control_digest(new_controls) if new_controls else None,
        'server_invariant_residual_linf': verify_controls(result, new_controls, sizes) if new_controls else None,
    }


def central_round(model, data, clients, recipe, step):
    # A separate reproducible stream; pooled uniform sampling represents the
    # SAME population objective, not the same realized samples as FL clients.
    seed = client_seed(recipe['seed'], clients, step)
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    batches, meter = sampled_batches(data, np.arange(data.manifest['train_rows']), recipe['batch_size'],
                                     clients*recipe['local_steps'], seed)
    criterion = build_official_criterion('sqrt_weighted_ce', data.manifest['train_class_counts'], 'fixed_train_mean')
    local, work, _ = local_sgd(model, batches, criterion, recipe['lr'], recipe['weight_decay'])
    require(work['examples_processed'] == meter['examples_processed'], 'Central sample/work mismatch')
    return local, {'seed': seed, **work, **meter, 'model_sha256': state_digest(local)}
