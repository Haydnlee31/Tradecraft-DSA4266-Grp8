"""CPU-only, one-round client/aggregation diagnostic, never a production resume.

Replay from a trusted archived LayerNorm checkpoint with either a full local
epoch or a small batch cap. Both start from the SAME weights and use the SAME
client assignments and sampling seeds. A cap changes training exposure, so this
is a mechanism probe, not a matched-compute model-quality comparison.
"""
import argparse
import copy
import hashlib
import io
import itertools
import json
from pathlib import Path
import platform
import tarfile

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.federated.partition import check_exact_cover
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.losses import build_criterion
from src.models.official_streaming import train_epoch, evaluate
from src.models.research import atomic_json
from src.utils.seed import client_seed, set_seed


def probe_rows(labels, per_class=200, seed=426639):
    """Fixed validation-only probe; stratification changes precision/F1 prevalence.

    Used to compare client and aggregate behavior on identical rows, NOT as the
    headline validation score. Final aggregate also gets full validation below.
    Only labels/indices are resident; full feature arrays remain disk-backed.
    """
    if per_class < 1:
        raise ValueError('Probe size must be positive')
    rng = np.random.default_rng(seed)
    selected = []
    for code in range(len(CLASSES)):
        rows = np.flatnonzero(labels == code)
        selected.extend(rng.choice(rows, min(per_class, len(rows)), replace=False))
    return np.sort(np.asarray(selected, dtype=np.int64))


def parameter_distance(model, reference):
    """Euclidean distance of learnable parameters, excluding normalization buffers."""
    return float(sum((p.detach().double()-q.detach().double()).square().sum()
                     for p, q in zip(model.parameters(), reference.parameters())).sqrt())


def reduction_audit(counts, weights):
    """Composition-only denominator audit, not a measured gradient or causal proof.

    Weighted CE divides by SUM of target weights in each batch. In a single-class
    batch the class weight cancels exactly. The full-client mean below is only a
    proxy: actual minibatch denominators vary, and Adam further changes dynamics.
    """
    counts = np.asarray(counts, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    sizes = counts.sum(axis=1)
    if counts.ndim != 2 or np.any(sizes <= 0) or np.any(weights <= 0):
        raise ValueError('Nonempty clients and positive weights required')
    means = counts @ weights / sizes
    global_mean = counts.sum(axis=0) @ weights / sizes.sum()
    return {'client_mean_target_weight': means.tolist(),
            'pooled_mean_target_weight': float(global_mean),
            'client_to_pooled_gradient_scale_proxy': (global_mean/means).tolist(),
            'interpretation': 'Composition denominator proxy only, not actual minibatch gradients or Adam updates'}


def branch(data, model, parts, settings, rows, round_number, cap):
    """Do one round and inspect every local model BEFORE Flower aggregation."""
    from flwr.app import ArrayRecord, MetricRecord, RecordDict
    from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

    if cap < 0:
        raise ValueError('Batch cap must be nonnegative; zero means full local epoch')
    global_model = copy.deepcopy(model)
    criterion = build_criterion(settings['loss'], data.manifest['train_class_counts'])
    replies, clients = [], []
    total_examples = total_updates = 0
    for client, indices in enumerate(parts):
        seed = client_seed(settings['seed'], client, round_number)
        set_seed(seed)
        torch.use_deterministic_algorithms(True)
        # Construct exactly as the runner does: initialization consumes RNG before
        # loading global weights, which determines subsequent dropout draws.
        local = MLPClassifier(39, len(CLASSES), model.config)
        local.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(local.parameters(), lr=settings['lr'],
                                     weight_decay=settings['weight_decay'])
        batches = data.batches('train', settings['batch_size'], seed, indices)
        if cap:
            batches = itertools.islice(batches, cap)
        loss, n, updates = train_epoch(local, optimizer, criterion, batches, 'cpu')
        _, metrics = evaluate(local, data.batches('val', settings['batch_size'], rows=rows), 'cpu')
        clients.append({'client': client, 'assigned_rows': len(indices),
                        'examples_processed': n, 'optimizer_steps': updates,
                        'task_loss': loss, 'parameter_delta_l2': parameter_distance(local, model),
                        'probe_metrics': metrics})
        total_examples += n
        total_updates += updates
        # Keep original partition-size weights in BOTH arms. Weighting by capped
        # examples would silently introduce nearly uniform client weighting.
        replies.append(RecordDict({'arrays': ArrayRecord(local.state_dict()),
                                   'metrics': MetricRecord({'num-examples': len(indices)})}))
    global_model.load_state_dict(aggregate_arrayrecords(replies, 'num-examples').to_torch_state_dict())
    _, probe = evaluate(global_model, data.batches('val', settings['batch_size'], rows=rows), 'cpu')
    _, full = evaluate(global_model, data.batches('val', settings['batch_size']), 'cpu')
    return {'cap': cap, 'clients': clients, 'examples_processed': total_examples,
            'optimizer_steps': total_updates, 'aggregation_weights': [len(p) for p in parts],
            'aggregate_parameter_delta_l2': parameter_distance(global_model, model),
            'aggregate_probe_metrics': probe, 'aggregate_full_validation_metrics': full}


def diagnose(data_root, archive, member_root, output, cap=5, per_class=200):
    if cap < 1:
        raise ValueError('Capped diagnostic needs at least one batch')
    data = PackedData(data_root)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    # Read explicit archive members, never extract paths or execute attached code.
    with tarfile.open(archive) as bundle:
        def member(name):
            return bundle.extractfile(member_root.rstrip('/')+'/'+name).read()
        env_bytes = member('environment.json')
        env = json.loads(env_bytes)
        checkpoint_bytes = member('last.pt')
        saved = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
        assignment_bytes = member('assignments.npz')
        info = json.loads(member('partition.json'))
        if (saved['environment_sha256'] != hashlib.sha256(env_bytes).hexdigest()
                or saved['partition_sha256'] != info['sha256']
                or info['sha256'] != hashlib.sha256(assignment_bytes).hexdigest()
                or env['packed_manifest_sha256'] != sha256(Path(data_root)/'manifest.json')):
            raise ValueError('Checkpoint/data/assignment provenance mismatch')
        with np.load(io.BytesIO(assignment_bytes), allow_pickle=False) as arrays:
            parts = [arrays[f'client_{i}'].copy() for i in range(env['settings']['clients'])]
    settings = env['settings']
    if settings['normalization'] != 'layer' or settings['lane'] not in ('iid', 'dirichlet'):
        raise ValueError('This diagnostic is restricted to federated LayerNorm runs')
    check_exact_cover(parts, data.manifest['train_rows'])
    counts = [np.bincount(data.arrays['train'][1][p], minlength=len(CLASSES)).tolist() for p in parts]
    if counts != info['counts']:
        raise ValueError('Client counts do not match the packed labels')
    torch.set_num_threads(2)
    set_seed(settings['seed'])
    torch.use_deterministic_algorithms(True)
    model = MLPClassifier(39, len(CLASSES), MLPConfig(**env['model_config']))
    model.load_state_dict(saved['model'])
    rows = probe_rows(data.arrays['val'][1], per_class)
    _, before_probe = evaluate(model, data.batches('val', settings['batch_size'], rows=rows), 'cpu')
    _, before_full = evaluate(model, data.batches('val', settings['batch_size']), 'cpu')
    criterion = build_criterion(settings['loss'], data.manifest['train_class_counts'])
    weights = criterion.weight.numpy() if criterion.weight is not None else np.ones(len(CLASSES))
    result = {'status': 'incomplete', 'purpose': 'One-round CPU mechanism probe, NOT a GPU resume or model selection',
              'archive_sha256': sha256(archive), 'member_root': member_root,
              'checkpoint_sha256': hashlib.sha256(checkpoint_bytes).hexdigest(),
              'packed_manifest_sha256': env['packed_manifest_sha256'],
              'source_sha256': {str(p.relative_to(Path(__file__).parents[1])): sha256(p)
                                for p in sorted(Path(__file__).parents[1].rglob('*.py'))},
              'torch': torch.__version__, 'numpy': np.__version__, 'platform': platform.platform(),
              'settings': settings, 'checkpoint_step': saved['step'], 'device': 'cpu',
              'test_evaluated': False, 'matched_compute': False,
              'probe_rows': rows.tolist(), 'probe_seed': 426639,
              'probe_warning': 'Class-stratified validation probe; precision/F1 are not population estimates',
              'before_probe_metrics': before_probe, 'before_full_validation_metrics': before_full,
              'client_counts': counts, 'loss_reduction_audit': reduction_audit(counts, weights), 'branches': {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(result, output)
    for name, limit in [('full_epoch', 0), ('capped', cap)]:
        result['branches'][name] = branch(data, model, parts, settings, rows, saved['step']+1, limit)
        atomic_json(result, output)
        print(f'{name}: diagnostic complete', flush=True)
    result['status'] = 'complete'
    atomic_json(result, output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--member-root', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cap', type=int, default=5)
    p.add_argument('--per-class', type=int, default=200)
    diagnose(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
