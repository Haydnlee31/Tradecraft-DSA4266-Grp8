"""A fixed epoch-wise hard-example sampling test, not an OHEM reproduction.

Only FIT losses determine difficulty. Class counts and weights stay fixed;
within-class exposure deliberately changes. There is no inverse-probability
correction: this is a changed training objective, not an unbiased estimator
of the old one. Half of each class batch remains uniformly sampled.
"""
import copy
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F

from src.eval import contrastive_flow as historical
from src.eval.official_evidence import require

old = historical.old
N = historical.N
ARMS = ['ordinary_control', 'hard_mixture']
SETTINGS = dict(seed=7, epochs=60, warmup_epochs=5, steps_per_epoch=528,
                per_class=64, hard_fraction=.25, lr=.001, weight_decay=.0001,
                scoring_batch=8192)


def difficulty(model, x, y, batch=8192):
    """One no-gradient unweighted CE value per FIT row, dropout disabled.

    Mining must not change model weights or consume the training RNG stream.
    The ordinary arm does the same scoring work but ignores the ranking.
    """
    require(x.ndim == 2 and y.shape == (len(x),) and batch > 0, 'Invalid scoring input')
    before, rng, training = old.state_hash(model), torch.get_rng_state().clone(), model.training
    result = np.empty(len(y), np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(y), batch):
            logits = model.base(x[start:start+batch])
            value = F.cross_entropy(logits, y[start:start+batch], reduction='none')
            require(torch.isfinite(value).all().item(), 'Nonfinite mining loss')
            result[start:start+len(value)] = value.numpy()
    model.train(training)
    require(old.state_hash(model) == before and torch.equal(rng, torch.get_rng_state()),
            'Mining mutated weights or training RNG')
    return result


def hard_pools(y, losses, fraction=.25):
    require(y.ndim == 1 and losses.shape == y.shape and len(y) > 0
            and np.issubdtype(y.dtype, np.integer) and np.isin(y, np.arange(N)).all()
            and np.isfinite(losses).all() and (losses >= 0).all() and fraction == .25,
            'Invalid difficulty ranking')
    result = []
    for label in range(N):
        rows = np.flatnonzero(y == label)
        require(len(rows) > 0, 'Missing FIT class')
        # Stable tie break uses FIT position, never calibration information.
        order = np.lexsort((rows, -losses[rows]))
        result.append(np.sort(rows[order[:math.ceil(len(rows)*fraction)]]))
    return result


def mixed_batches(y, pools, seed, epoch, steps, per_class):
    require(len(pools) == N and type(per_class) is int and per_class >= 2
            and per_class % 2 == 0 and type(steps) is int and steps > 0, 'Invalid batch recipe')
    classes = [np.flatnonzero(y == c) for c in range(N)]
    for c, p in enumerate(pools):
        require(p.ndim == 1 and np.issubdtype(p.dtype, np.integer)
                and len(p) >= per_class and len(np.unique(p)) == len(p)
                and np.isin(p, classes[c]).all(), 'Invalid hard pool')
    rng = np.random.default_rng(seed+epoch)
    for _ in range(steps):
        chunks = []
        for c in range(N):
            uniform = rng.choice(classes[c], per_class//2, replace=False)
            # Draw 64 unique hard proposals, remove at most 32 uniform rows,
            # then take the first 32. This is uniform sampling from the hard
            # pool excluding the uniform half, with no within-batch duplicates.
            proposals = rng.choice(pools[c], per_class, replace=False)
            hard = proposals[~np.isin(proposals, uniform)][:per_class//2]
            require(len(hard) == per_class//2, 'Insufficient distinct hard rows')
            chunks.extend([uniform, hard])
        yield rng.permutation(np.concatenate(chunks))


def fit(x, y, groups, population, arm, settings=None, resume=None, stop_after=None):
    """CPU-only fitting with exact recovery at completed epoch boundaries.

    A recovery state includes Adam moments, RNG, exposure counters and the
    chained batch digest. Difficulty is recomputed for the next epoch from
    restored weights; no stale hard-pool cache is reused. Synthetic tests may
    shorten the run, but the public runner accepts only the fixed recipe.
    """
    s = dict(SETTINGS if settings is None else settings)
    require(set(s) == set(SETTINGS) and arm in ARMS and x.ndim == 2
            and x.dtype == np.float32 and np.isfinite(x).all()
            and y.shape == groups.shape == (len(x),) and y.dtype == np.int64
            and np.issubdtype(groups.dtype, np.integer) and (groups >= 0).all()
            and np.isin(y, np.arange(N)).all()
            and 1 <= s['epochs'] <= 60 and 0 <= s['warmup_epochs'] <= s['epochs']
            and 0 < s['steps_per_epoch'] <= 528 and s['per_class'] >= 2 and s['per_class'] % 2 == 0
            and s['hard_fraction'] == .25 and s['lr'] == .001 and s['weight_decay'] == .0001
            and s['scoring_batch'] > 0 and (np.bincount(y, minlength=N) >= 4*s['per_class']).all(),
            'Invalid fitting recipe')
    weights = historical.class_weights(population)
    binding = dict(settings=s, arm=arm, shape=list(x.shape),
        x_sha256=old.vector_hash(x), y_sha256=old.vector_hash(y),
        groups_sha256=old.vector_hash(groups), class_weights=weights.tolist())
    torch.manual_seed(s['seed'])
    torch.use_deterministic_algorithms(True)
    model = historical.FlowModel(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=s['lr'], weight_decay=s['weight_decay'])
    record = dict(settings=s, arm=arm, completed_epochs=0,
        parameters=sum(p.numel() for p in model.parameters()), deployment_parameters=model.base.num_parameters(),
        initial_state_sha256=old.state_hash(model), class_weights=weights.tolist(),
        examples_processed=0, optimizer_steps=0, mining_examples_scored=0, forced_hard_positions=0,
        sampled_class_counts=[0]*N, selected_hard_pool_positions=0,
        batch_chain_sha256=hashlib.sha256(b'').hexdigest(), observations=[], mining=[])
    seen = np.zeros(len(y), bool)
    if resume is not None:
        require(resume['binding'] == binding, 'Recovery inputs/settings changed')
        record = copy.deepcopy(resume['record'])
        require(0 < record['completed_epochs'] < s['epochs'], 'Invalid recovery boundary')
        model.load_state_dict(resume['model'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['rng'])
        seen = resume['seen'].numpy().copy()
        require(seen.shape == (len(y),) and seen.dtype == np.bool_
                and old.state_hash(model) == record['final_state_sha256'], 'Invalid recovery state')
    endpoint = s['epochs'] if stop_after is None else stop_after
    require(type(endpoint) is int and record['completed_epochs'] < endpoint <= s['epochs'], 'Invalid stop epoch')
    xx, yy, gg, ww = map(torch.from_numpy, (x, y, groups, weights))
    for epoch in range(record['completed_epochs']+1, endpoint+1):
        losses = difficulty(model, xx, yy, s['scoring_batch'])
        pools = hard_pools(y, losses, s['hard_fraction'])
        is_hard = np.zeros(len(y), bool)
        is_hard[np.concatenate(pools)] = True
        record['mining_examples_scored'] += len(y)
        record['mining'].append(dict(epoch=epoch, losses_sha256=old.vector_hash(losses),
            pool_sha256=old.vector_hash(np.concatenate(pools).astype('<i8')),
            pool_class_counts=[len(p) for p in pools]))
        enabled = arm == ARMS[1] and epoch > s['warmup_epochs']
        batches = (mixed_batches(y, pools, s['seed'], epoch, s['steps_per_epoch'], s['per_class']) if enabled else
                   historical.batches(y, s['seed'], epoch, s['steps_per_epoch'], s['per_class']))
        model.train()
        ce_sum = 0.
        for take in batches:
            record['batch_chain_sha256'] = hashlib.sha256(bytes.fromhex(record['batch_chain_sha256'])
                                                       + take.astype('<i8').tobytes()).hexdigest()
            seen[take] = True
            record['sampled_class_counts'] = (np.asarray(record['sampled_class_counts'])
                                             + np.bincount(y[take], minlength=N)).tolist()
            record['selected_hard_pool_positions'] += int(is_hard[take].sum())
            record['forced_hard_positions'] += len(take)//2 if enabled else 0
            optimizer.zero_grad(set_to_none=True)
            logits, projection = model(xx[take])
            ce = old.weighted_loss(logits, yy[take], ww[yy[take]])
            # Keep the zero-coefficient historical projector computation in
            # BOTH arms for an exact bridge to the prior CE control. This is
            # not a contrastive training objective or a new inference layer.
            con, _ = historical.supcon(projection, yy[take], gg[take])
            loss = ce+0.*con
            require(torch.isfinite(loss).item(), 'Nonfinite loss')
            loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),
                    'Nonfinite gradient')
            optimizer.step()
            record['optimizer_steps'] += 1
            record['examples_processed'] += len(take)
            ce_sum += float(ce.detach())
        record['completed_epochs'] = epoch
        if epoch in [1, 5, 20, 40, s['epochs']]:
            record['observations'].append(dict(epoch=epoch, ce=ce_sum/s['steps_per_epoch'],
                                               state_sha256=old.state_hash(model)))
            print(f'HARD arm={arm} epoch={epoch} CE={ce_sum/s["steps_per_epoch"]:.6f}', flush=True)
    record.update(final_state_sha256=old.state_hash(model), unique_rows_seen=int(seen.sum()),
                  unique_class_counts=np.bincount(y[seen], minlength=N).tolist())
    state = dict(binding=binding, record=copy.deepcopy(record), model=copy.deepcopy(model.state_dict()),
                 optimizer=copy.deepcopy(optimizer.state_dict()), rng=torch.get_rng_state().clone(),
                 seen=torch.from_numpy(seen.copy()))
    return model, record, state
