"""Validation-only disk-backed centralized and simulated Flower FedAvg runner.

One central step is an epoch; one FL step is a full-participation round with one
local epoch and freshly initialized local Adam state. These are NOT matched update
budgets. Recovery is at completed epoch/round boundaries, not mid-batch. This new
39-feature study never opens test data or alters the historical runner.

An opt-in local batch cap supports shorter client updates. An optional aggregate
optimizer-step budget makes the final round shorter for ALL clients equally.
Matching update counts does not match examples, communication, Adam resets or
per-client exposure; report those differences rather than claiming equivalence.
"""
import argparse
from dataclasses import asdict, replace
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.federated.partition import dirichlet_split, check_exact_cover
from src.models.architectures import MLPClassifier, config_for_variant
from src.models.losses import build_criterion
from src.models.research import atomic_save, atomic_json, resolve_device, rng_state, restore_rng, memory_metrics
from src.utils.seed import client_seed, set_seed


def partitions(labels, lane, clients, seed, alpha):
    if clients < 1 or len(labels) < clients*2:
        raise ValueError('Each simulated client needs at least two rows')
    if lane == 'iid':
        # Shuffle first: packed rows retain source ordering, so contiguous chunks
        # without this permutation would not be an IID control.
        parts = [np.sort(v) for v in np.array_split(np.random.default_rng(seed).permutation(len(labels)), clients)]
        attempts = 1
    elif lane == 'dirichlet':
        # Only labels/indices are materialized, never all feature rows. Reuse the
        # legacy partition algorithm so the non-IID policy is not silently changed.
        names = np.array(CLASSES, dtype=object)[labels]
        parts, attempts = dirichlet_split(names, clients, alpha, seed, min_partition_size=2)
    else:
        raise ValueError('Unknown federated partition lane')
    check_exact_cover(parts, len(labels))
    return parts, attempts


def train_epoch(model, optimizer, criterion, batches, device):
    model.train()
    total, examples, updates = 0., 0, 0
    for x, y in batches:
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()*len(y)
        examples += len(y)
        updates += 1
    if not examples:
        raise ValueError('Empty training iterator')
    return total/examples, examples, updates


def local_round_limits(sizes, batch_size, cap, budget, epochs):
    """Freeze a full-participation schedule before training.

    Zero cap is the historical full local epoch. A positive exact budget must
    divide by the client count: no client is silently omitted from the last
    round. Every client must have enough batches for the requested cap. Shuffling
    restarts each round; capped training is sampling, not guaranteed row coverage.
    """
    if cap < 0 or budget < 0:
        raise ValueError('Local batch cap and update budget must be nonnegative')
    if not budget:
        return [cap] * epochs
    if not cap or budget % len(sizes):
        raise ValueError('Exact update budget requires a cap and divisibility by client count')
    # PackedData merges a singleton tail into the preceding batch.
    available = [(n+batch_size-1)//batch_size - int(n > batch_size and n % batch_size == 1)
                 for n in sizes]
    if min(available) < cap:
        raise ValueError('Client has too few batches for the exact-budget cap')
    per_client = budget // len(sizes)
    full_rounds, remainder = divmod(per_client, cap)
    limits = [cap] * full_rounds + ([remainder] if remainder else [])
    if len(limits) != epochs:
        raise ValueError(f'Exact update budget requires epochs={len(limits)} (rounds), got {epochs}')
    return limits


def confusion_metrics(cm):
    # Fixed eight-class reduction, including classes with no correct predictions.
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    diagonal = cm.diagonal()
    recall = np.divide(diagonal, support, out=np.zeros(8), where=support != 0)
    precision = np.divide(diagonal, predicted, out=np.zeros(8), where=predicted != 0)
    f1 = np.divide(2*diagonal, support+predicted, out=np.zeros(8), where=(support+predicted) != 0)
    benign = CLASSES.index('Benign')
    return {'macro_f1': float(f1.mean()), 'accuracy': float(diagonal.sum()/cm.sum()),
            'per_class': {name: {'precision': float(precision[i]), 'recall': float(recall[i]),
                                  'f1': float(f1[i]), 'support': int(support[i])} for i, name in enumerate(CLASSES)},
            'benign_false_alert_rate': float(1-recall[benign]) if support[benign] else None,
            'confusion_matrix': cm.tolist()}


def evaluate(model, batches, device):
    model.eval()
    cm = np.zeros((8, 8), dtype=np.int64)
    total_loss = 0.
    with torch.no_grad():
        for x, y in batches:
            logits = model(torch.from_numpy(x).to(device))
            loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y).to(device), reduction='sum')
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite validation loss')
            total_loss += loss.item()
            predictions = logits.argmax(1).cpu().numpy()
            cm += np.bincount(y*8+predictions, minlength=64).reshape(8, 8)
    if not cm.sum():
        raise ValueError('Empty validation iterator')
    return total_loss/int(cm.sum()), confusion_metrics(cm)


def run(data_root, output, lane='light', epochs=2, seed=7, clients=20,
        partition_seed=7, alpha=.5, batch_size=512, lr=.001, weight_decay=1e-5,
        loss='sqrt_weighted_ce', normalization='batch', device='cpu', threads=2,
        resume=False, stop_after=0, local_max_batches=0, total_update_budget=0):
    if lane not in ('light', 'heavy', 'iid', 'dirichlet') or epochs < 1 or batch_size < 2 or threads < 1:
        raise ValueError('Invalid lane, epochs, batch size or threads')
    if min(seed, partition_seed, stop_after) < 0 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Invalid seed or alpha')
    if not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('Invalid optimizer settings')
    if min(local_max_batches, total_update_budget) < 0:
        raise ValueError('Local batch cap and update budget must be nonnegative')
    if (local_max_batches or total_update_budget) and lane not in ('iid', 'dirichlet'):
        raise ValueError('Local batch cap and update budget are federated-only')
    device = resolve_device(device)
    data, output = PackedData(data_root), Path(output)
    if data.manifest['train_rows'] > 5000000:
        raise ValueError('Reviewed index-memory cap exceeded')
    torch.set_num_threads(threads)
    set_seed(seed)
    # Fail rather than silently using a nondeterministic operation in this runner.
    torch.use_deterministic_algorithms(True)
    config = replace(config_for_variant('heavy' if lane == 'heavy' else 'light'), normalization=normalization)
    settings = dict(lane=lane, epochs=epochs, seed=seed, clients=clients, partition_seed=partition_seed,
                    alpha=alpha, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                    loss=loss, normalization=normalization, device=device, threads=threads,
                    local_max_batches=local_max_batches, total_update_budget=total_update_budget)
    source_root = Path(__file__).parents[1]
    env = {'settings': settings, 'packed_manifest_sha256': sha256(Path(data_root)/'manifest.json'),
           'source_sha256': {str(p.relative_to(source_root)): sha256(p) for p in sorted(source_root.rglob('*.py'))},
           'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'flwr', 'scikit-learn')},
           'platform': platform.platform(), 'model_config': asdict(config),
           'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
           'device_name': torch.cuda.get_device_name() if device == 'cuda' else platform.processor(),
           'classes': CLASSES, 'features': data.manifest['features'], 'test_evaluated': False,
           'optimizer_policy': 'persistent central; reset each client-round',
           'tail_policy': 'merge singleton tail; all selected rows processed',
           'local_sampling_policy': 'fresh seeded client shuffle per round; cap takes prefix, no guaranteed full coverage' if local_max_batches else 'full local epoch',
           'aggregation_policy': 'original assigned client row counts, even when capped',
           'checkpoint_selection': 'strict maximum validation macro-F1, beginning at step 1'}
    # JSON normalizes tuples to lists before comparing recovery provenance.
    env = json.loads(json.dumps(env))
    if resume:
        if json.loads((output/'environment.json').read_text()) != env:
            raise ValueError('Refusing resume: data/code/environment/settings changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(env, output/'environment.json')
    parts = None
    if lane in ('iid', 'dirichlet'):
        if resume:
            info = json.loads((output/'partition.json').read_text())
            if info['sha256'] != sha256(output/'assignments.npz'):
                raise ValueError('Client assignments changed')
            with np.load(output/'assignments.npz', allow_pickle=False) as saved:
                parts = [saved[f'client_{i}'] for i in range(clients)]
            check_exact_cover(parts, data.manifest['train_rows'])
        else:
            parts, attempts = partitions(data.arrays['train'][1], lane, clients, partition_seed, alpha)
            np.savez_compressed(output/'assignments.npz', **{f'client_{i}': rows for i, rows in enumerate(parts)})
            atomic_json({'sha256': sha256(output/'assignments.npz'), 'attempts': attempts,
                         'simulated_clients_not_physical_devices': True,
                         'counts': [np.bincount(data.arrays['train'][1][rows], minlength=8).tolist() for rows in parts]}, output/'partition.json')
    round_limits = local_round_limits([len(p) for p in parts], batch_size, local_max_batches,
                                     total_update_budget, epochs) if parts is not None else [0]*epochs
    model = MLPClassifier(39, 8, config).to(device)
    counts = data.manifest['train_class_counts']
    criterion = build_criterion(loss, counts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history, best, best_score, best_step, start = [], None, -1., 0, 1
    if resume:
        checkpoint = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        if checkpoint['environment_sha256'] != sha256(output/'environment.json'):
            raise ValueError('Checkpoint belongs to a different environment')
        if checkpoint['partition_sha256'] != (sha256(output/'assignments.npz') if parts is not None else None):
            raise ValueError('Checkpoint client assignments changed')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        restore_rng(checkpoint['rng'])
        history, best = checkpoint['history'], checkpoint['best']
        best_score, best_step, start = checkpoint['best_score'], checkpoint['best_step'], checkpoint['step']+1
    for step in range(start, epochs+1):
        started = time.perf_counter()
        if parts is None:
            epoch_seed = client_seed(seed, 0, step)
            set_seed(epoch_seed)
            torch.use_deterministic_algorithms(True)
            train_loss, examples, updates = train_epoch(model, optimizer, criterion,
                data.batches('train', batch_size, epoch_seed), device)
        else:
            from flwr.app import ArrayRecord, MetricRecord, RecordDict
            from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords
            replies, examples, updates, total = [], 0, 0, 0.
            client_work = []
            for client, rows in enumerate(parts):
                local_seed = client_seed(seed, client, step)
                set_seed(local_seed)
                torch.use_deterministic_algorithms(True)
                local = MLPClassifier(39, 8, config).to(device)
                local.load_state_dict(model.state_dict())
                local_optimizer = torch.optim.Adam(local.parameters(), lr=lr, weight_decay=weight_decay)
                batches = data.batches('train', batch_size, local_seed, rows)
                if round_limits[step-1]:
                    batches = itertools.islice(batches, round_limits[step-1])
                local_loss, n, steps = train_epoch(local, local_optimizer, criterion, batches, device)
                examples, updates, total = examples+n, updates+steps, total+local_loss*n
                client_work.append({'client': client, 'assigned_rows': len(rows),
                                    'examples_processed': n, 'optimizer_steps': steps})
                # Full-epoch behavior is unchanged because n == len(rows). With
                # a cap, using n would introduce a second treatment by changing
                # the aggregation weights toward uniform client weighting.
                replies.append(RecordDict({'arrays': ArrayRecord(local.cpu().state_dict()),
                                            'metrics': MetricRecord({'num-examples': len(rows)})}))
            model.load_state_dict(aggregate_arrayrecords(replies, 'num-examples').to_torch_state_dict())
            train_loss = total/examples
        val_loss, metrics = evaluate(model, data.batches('val', batch_size), device)
        if device == 'cuda':
            torch.cuda.synchronize()
        history.append({'step': step, 'validation_metrics': metrics, 'val_loss': val_loss,
                        'train_batch_mean_task_loss': train_loss, 'examples_processed': examples,
                        'optimizer_steps': updates, 'elapsed_seconds': time.perf_counter()-started})
        if parts is not None:
            history[-1]['client_work'] = client_work
            history[-1]['local_batch_cap'] = round_limits[step-1]
        if metrics['macro_f1'] > best_score:
            best_score, best_step = metrics['macro_f1'], step
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        atomic_save({'step': step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'rng': rng_state(), 'history': history, 'best': best, 'best_score': best_score,
                     'partition_sha256': sha256(output/'assignments.npz') if parts is not None else None,
                     'best_step': best_step, 'environment_sha256': sha256(output/'environment.json')}, output/'last.pt')
        print(f'{lane} step={step} val_macro_f1={metrics["macro_f1"]:.5f}', flush=True)
        if stop_after and step >= stop_after and step < epochs:
            atomic_json({'status': 'paused', 'step': step}, output/'status.json')
            return
    atomic_save({'model': best, 'config': asdict(config), 'features': data.manifest['features'],
                 'classes': CLASSES, 'best_step': best_step}, output/'best.pt')
    if total_update_budget and sum(h['optimizer_steps'] for h in history) != total_update_budget:
        raise ValueError('Executed optimizer steps do not match the exact budget')
    result = {'lane': lane, 'best_step': best_step, 'history': history,
              'validation_metrics': history[best_step-1]['validation_metrics'], 'test_metrics': None,
              'final_validation_metrics': history[-1]['validation_metrics'],
              'total_update_budget': total_update_budget,
              'examples_processed': sum(h['examples_processed'] for h in history),
              'optimizer_steps': sum(h['optimizer_steps'] for h in history),
              'num_parameters': model.num_parameters(), 'edge_hardware_measurements': False,
              **memory_metrics(device)}
    atomic_json(result, output/'result.json')
    atomic_json({'status': 'complete', 'step': len(history)}, output/'status.json')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lane', choices=['light', 'heavy', 'iid', 'dirichlet'], required=True)
    for name, default in [('epochs', 2), ('seed', 7), ('clients', 20), ('partition-seed', 7), ('batch-size', 512), ('threads', 2), ('stop-after', 0)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--loss', choices=['ce', 'sqrt_weighted_ce', 'weighted_ce'], default='sqrt_weighted_ce')
    p.add_argument('--normalization', choices=['batch', 'layer'], default='batch')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--local-max-batches', type=int, default=0,
                   help='Federated-only local batch cap; zero preserves full epochs')
    p.add_argument('--total-update-budget', type=int, default=0,
                   help='Exact sum of client optimizer steps; requires cap, full participation and matching round count')
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
