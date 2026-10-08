"""One CPU round distinguishes local predictions from two aggregation rules.

Start from one frozen cloud checkpoint. Train each client ONCE with the existing
recipe, then reuse those same local models for parameter averaging and weighted
probability averaging. The ensemble is a diagnostic, not a proposed single-model
deployment: it needs all client models and roughly twenty times the inference
work. No hyperparameter search, production resume, or test access occurs here.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import tarfile

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.eval.official_client_diagnostic import parameter_distance
from src.federated.partition import check_exact_cover
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_losses import build_official_criterion
from src.models.official_streaming import train_epoch, confusion_metrics
from src.models.research import atomic_json
from src.utils.seed import client_seed, set_seed

TARGETS = ('Web-based', 'Brute Force', 'Spoofing')


def common_probe(labels, per_other=2000, seed=426639):
    """All three failing classes, bounded seeded samples of the other classes."""
    if per_other < 1:
        raise ValueError('Positive probe cap required')
    rng = np.random.default_rng(seed)
    selected = []
    for c, name in enumerate(CLASSES):
        rows = np.flatnonzero(labels == c)
        selected.extend(rows if name in TARGETS else rng.choice(rows, min(per_other, len(rows)), replace=False))
    return np.sort(np.asarray(selected, dtype=np.int64))


def probabilities(model, data, split, rows):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()
                               for x, _ in data.batches(split, 2048, rows=rows)])


def conditional_summary(predictions, labels):
    """No precision or macro-F1 on the class-stratified probe population."""
    matrix = np.bincount(labels*len(CLASSES)+predictions,
                         minlength=len(CLASSES)**2).reshape(len(CLASSES), len(CLASSES))
    return {'confusion_matrix': matrix.tolist(), 'per_class': {
        name: {'rows': int(matrix[c].sum()),
               'recall': float(matrix[c, c]/matrix[c].sum()) if matrix[c].sum() else None}
        for c, name in enumerate(CLASSES)}}


def own_training_recall(model, data, indices):
    """Evaluation-mode recall on the client's own examples, not generalization."""
    labels = data.arrays['train'][1][indices]
    result = {}
    for name in TARGETS:
        c = CLASSES.index(name)
        selected = indices[labels == c]
        result[name] = {'rows': len(selected), 'recall': None}
        if len(selected):
            p = probabilities(model, data, 'train', selected)
            result[name]['recall'] = float((p.argmax(axis=1) == c).mean())
    return result


def weighted_probabilities(probabilities_by_client, weights):
    """Use the SAME original row-count weights as parameter aggregation."""
    w = np.asarray(weights, dtype=np.float64)
    if (w.ndim != 1 or not w.size or len(w) != len(probabilities_by_client)
            or not np.isfinite(w).all() or np.any(w <= 0)):
        raise ValueError('Finite positive weights for every client required')
    # Scaling first avoids overflow for otherwise valid very large weights.
    w = w/w.max()
    w = w/w.sum()
    shape = probabilities_by_client[0].shape
    if len(shape) != 2 or shape[1] != len(CLASSES):
        raise ValueError('Expected rows by eight classes')
    average = np.zeros(shape, dtype=np.float64)
    for p, weight in zip(probabilities_by_client, w):
        if p.shape != shape or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
            raise ValueError('Invalid client probability matrix')
        average += weight*p
    return average


def full_metrics(models, weights, data):
    """Stream the natural full validation population: headline precision is valid.

    Accumulate only one batch of predictions and an 8x8 confusion matrix. No
    feature matrix or full-validation probability matrix is loaded into RAM.
    """
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for model in models:
        model.eval()
    with torch.inference_mode():
        for x, y in data.batches('val', 4096):
            xt = torch.from_numpy(x)
            ps = [torch.softmax(model(xt), dim=1).numpy() for model in models]
            pred = weighted_probabilities(ps, weights).argmax(axis=1)
            cm += np.bincount(y*len(CLASSES)+pred, minlength=len(CLASSES)**2).reshape(cm.shape)
    return confusion_metrics(cm)


def one_round(data, model, parts, settings, rows, round_number):
    """Train once, inspect local results, and compare the two averaging rules."""
    from flwr.app import ArrayRecord, RecordDict, MetricRecord
    from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

    if model.config.normalization != 'layer':
        raise ValueError('Probe is restricted to LayerNorm')
    check_exact_cover(parts, data.manifest['train_rows'])
    labels = np.array(data.arrays['val'][1][rows], copy=True)
    initial = probabilities(model, data, 'val', rows)
    criterion = build_official_criterion(settings['loss'], data.manifest['train_class_counts'],
                                         settings['loss_reduction'])
    locals_, replies, ps, clients = [], [], [], []
    for client, indices in enumerate(parts):
        # Match the runner's RNG order: seed, construct, load global weights,
        # then train. Evaluation passes have no dropout or RNG consumption.
        seed = client_seed(settings['seed'], client, round_number)
        set_seed(seed)
        torch.use_deterministic_algorithms(True)
        local = MLPClassifier(len(data.manifest['features']), len(CLASSES), model.config)
        local.load_state_dict(model.state_dict())
        before = own_training_recall(local, data, indices)
        optimizer = torch.optim.Adam(local.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
        loss, examples, updates = train_epoch(local, optimizer, criterion,
            data.batches('train', settings['batch_size'], seed, indices), 'cpu')
        if examples != len(indices):
            raise ValueError('Client did not process all assigned rows')
        p = probabilities(local, data, 'val', rows)
        ps.append(p)
        locals_.append(local)
        clients.append({'client': client, 'assigned_rows': len(indices), 'examples_processed': examples,
                        'optimizer_steps': updates, 'task_loss': loss,
                        'train_class_counts': np.bincount(data.arrays['train'][1][indices], minlength=len(CLASSES)).tolist(),
                        'parameter_delta_l2': parameter_distance(local, model),
                        'own_train_before': before, 'own_train_after': own_training_recall(local, data, indices),
                        'common_probe': conditional_summary(p.argmax(axis=1), labels)})
        replies.append(RecordDict({'arrays': ArrayRecord(local.state_dict()),
                                   'metrics': MetricRecord({'num-examples': len(indices)})}))
        print(f'client {client+1}/{len(parts)}: inspected', flush=True)
    aggregate = copy.deepcopy(model)
    aggregate.load_state_dict(aggregate_arrayrecords(replies, 'num-examples').to_torch_state_dict())
    averaged = probabilities(aggregate, data, 'val', rows)
    weights = list(map(len, parts))
    ensemble = weighted_probabilities(ps, weights)
    pred = np.stack([p.argmax(axis=1) for p in ps])
    # Oracle union is deliberately labelled: selecting a correct client using
    # the TRUE label is impossible at deployment. It only locates learned signal.
    union = np.any(pred == labels[None, :], axis=0)
    result = {'clients': clients, 'aggregation_weights': weights,
              'examples_processed': sum(c['examples_processed'] for c in clients),
              'optimizer_steps': sum(c['optimizer_steps'] for c in clients),
              'probe_before': conditional_summary(initial.argmax(axis=1), labels),
              'probe_parameter_average': conditional_summary(averaged.argmax(axis=1), labels),
              'probe_probability_average': conditional_summary(ensemble.argmax(axis=1), labels),
              'oracle_any_client_recall_not_deployable': {
                  name: float(union[labels == c].mean()) if np.any(labels == c) else None
                  for c, name in enumerate(CLASSES)},
              'full_validation_before': full_metrics([model], [1], data),
              'full_validation_parameter_average': full_metrics([aggregate], [1], data),
              'full_validation_probability_average': full_metrics(locals_, weights, data)}
    return result


def diagnose(data_root, archive, member_root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data = PackedData(data_root)
    archive_hash = sha256(archive)
    with tarfile.open(archive) as t:
        def raw(name):
            return t.extractfile(member_root.rstrip('/')+'/'+name).read()
        eb, cb, ab = (raw(n) for n in ('environment.json', 'last.pt', 'assignments.npz'))
        env, partition = json.loads(eb), json.loads(raw('partition.json'))
        status, result = json.loads(raw('status.json')), json.loads(raw('result.json'))
        saved = torch.load(io.BytesIO(cb), map_location='cpu', weights_only=True)
        if (saved['environment_sha256'] != hashlib.sha256(eb).hexdigest()
                or saved['partition_sha256'] != hashlib.sha256(ab).hexdigest()
                or partition['sha256'] != saved['partition_sha256']
                or env['packed_manifest_sha256'] != sha256(Path(data_root)/'manifest.json')):
            raise ValueError('Checkpoint/data/assignment provenance mismatch')
        with np.load(io.BytesIO(ab), allow_pickle=False) as arrays:
            parts = [arrays[f'client_{i}'].copy() for i in range(env['settings']['clients'])]
    s = env['settings']
    control = env.get('partition_control', {})
    if (s['normalization'] != 'layer' or s['lane'] != 'dirichlet' or saved['step'] != 20
            or s['local_max_batches'] or s['total_update_budget'] or env['test_evaluated']
            or data.manifest['train_rows'] != 2000000 or len(parts) != 20
            or status != {'status': 'complete', 'step': 20}
            or result['test_metrics'] is not None
            or control.get('policy') != 'nested-anchor-class-shares-v1'
            or control.get('cohort') != '2m'
            or control.get('assignment_sha256') != saved['partition_sha256']):
        raise ValueError('Expected the completed full-epoch 2M controlled non-IID checkpoint')
    check_exact_cover(parts, data.manifest['train_rows'])
    if [np.bincount(data.arrays['train'][1][p], minlength=8).tolist() for p in parts] != partition['counts']:
        raise ValueError('Partition counts differ from packed data')
    for name, digest in env['source_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha256(Path(__file__).parents[1]/relative) != digest:
            raise ValueError('Archived training source no longer matches')
    torch.set_num_threads(2)
    set_seed(s['seed'])
    torch.use_deterministic_algorithms(True)
    model = MLPClassifier(39, len(CLASSES), MLPConfig(**env['model_config']))
    model.load_state_dict(saved['model'])
    rows = common_probe(data.arrays['val'][1])
    receipt = {'status': 'incomplete', 'purpose': 'Single CPU mechanism probe, not a CUDA resume or model selection',
               'device': 'cpu', 'test_evaluated': False, 'round': 21,
               'archive_sha256': archive_hash, 'checkpoint_sha256': hashlib.sha256(cb).hexdigest(),
               'packed_manifest_sha256': env['packed_manifest_sha256'],
               'partition_sha256': saved['partition_sha256'], 'partition_control': control, 'settings': s,
               'script_sha256': sha256(__file__), 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'flwr')},
               'probe_rows': rows.tolist(), 'probe_seed': 426639,
               'limitations': 'Probe recalls are conditional; no probe precision. Full validation metrics use natural prevalence. Ensemble uses all 20 models. One endpoint/seed/round only.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt, output)
    receipt['observations'] = one_round(data, model, parts, s, rows, 21)
    observed = receipt['observations']
    if observed['examples_processed'] != 2000000 or observed['optimizer_steps'] != 3917:
        raise ValueError('Unexpected diagnostic work budget')
    # This is a CPU mechanism probe, not a promise of CPU/GPU training identity.
    # Nonetheless, establish the SAME starting predictions before interpreting
    # differences. Read the final history entry, not a selected best checkpoint.
    if observed['full_validation_before'] != result['history'][-1]['validation_metrics']:
        raise ValueError('CPU starting metrics differ from archived GPU endpoint')
    if sha256(archive) != archive_hash:
        raise ValueError('Source archive changed during diagnosis')
    receipt['status'] = 'complete'
    receipt['starting_validation_metrics_exact'] = True
    receipt['source_archive_unchanged'] = True
    atomic_json(receipt, output)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'archive', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--member-root', required=True)
    diagnose(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
