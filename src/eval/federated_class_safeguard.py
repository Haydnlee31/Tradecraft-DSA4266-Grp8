"""Training-only, one-round diagnostic of class-risk-constrained FedAvg.

This does NOT install a new training strategy. Three preselected frozen models
each produce one ordinary client round and one counterfactual server update.
No corrected model is fed into another round; validation/test arrays never open.
Class gradients use deterministic eval-mode CE on the assigned TRAIN examples.
They are loss surrogates, not guarantees about recall, false alerts or privacy.
"""
import argparse
import ast
import copy
import hashlib
import importlib.metadata
import io
import itertools
import json
from pathlib import Path
import platform
import tarfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.federated.partition import check_exact_cover
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_losses import build_official_criterion
from src.models.official_streaming import train_epoch, confusion_metrics
from src.models.research import atomic_json
from src.utils.seed import client_seed, set_seed

PROTECTED = ('Benign', 'Web-based', 'Brute Force')
SOURCE_FILES = ('models/architectures.py', 'models/official_streaming.py',
                'models/official_losses.py', 'models/losses.py', 'data/official_packed.py',
                'data/label_map.py', 'utils/seed.py', 'federated/partition.py')

# Reviewed git diff 1d767e2 -> 5cc8206: optional feature-mask paths were added,
# but the default local loop and these imported functions were unchanged. Keep
# the exact file pair AND AST guards; do not waive source checks generically.
RUNNER_BRIDGE = (
    '82eebe2e342c727fb79811278c9847e9debd0daeb9c2ff7debf7c29988a29646',
    'cf7085916f6794867485802236662131c3465d1f40564d692a398c64b7ffe496',
)
RUNNER_FUNCTIONS = {
    'train_epoch': '201c79d33cd8fa37cfd26efef1eb9a1b316cf517207641e04e0704694fdd1254',
    'confusion_metrics': '8a3c6e903211cd4dd32a5f3dcb54c9e8bff21e2441528e10d30e85fd1bb43c96',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_source(name, expected):
    path = Path(__file__).parents[1]/name
    current = sha256(path)
    if current == expected:
        return {'archived_sha256': expected, 'current_sha256': current, 'exact_file': True}
    require(name == 'models/official_streaming.py' and (expected, current) == RUNNER_BRIDGE,
            'Historical source changed: '+name)
    tree = ast.parse(path.read_text())
    for function, digest in RUNNER_FUNCTIONS.items():
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        require(hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == digest,
                'Historical function changed: '+function)
    return {'archived_sha256': expected, 'current_sha256': current, 'exact_file': False,
            'reviewed_bridge': '1d767e2 to 5cc8206 optional ablation; imported functions AST-exact',
            'imported_function_ast_sha256': RUNNER_FUNCTIONS}


class TrainOnlyData(PackedData):
    """Reuse historical batching without its loader opening validation arrays."""
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        m = self.manifest
        require(m['status'] == 'complete' and len(m['features']) == 39
                and m['classes'] == CLASSES and 0 < m['train_rows'] <= 5000000,
                'Completed bounded official39 training cohort required')
        require(sha256(self.root/'scaler.json') == m['scaler_sha256'], 'Scaler changed')
        pair = []
        for key in ('x', 'y'):
            name = f'train_{key}.npy'
            require(sha256(self.root/name) == m['files'][name], 'Training array changed')
            pair.append(np.load(self.root/name, mmap_mode='r', allow_pickle=False))
        x, y = pair
        require(x.shape == (m['train_rows'], 39) and y.shape == (len(x),)
                and x.dtype == np.float32 and y.dtype == np.int64, 'Wrong training shape/dtype')
        require(np.all((y >= 0) & (y < len(CLASSES))), 'Invalid training label')
        require(np.bincount(y, minlength=8).tolist() ==
                [m['train_class_counts'][c] for c in CLASSES], 'Training counts changed')
        self.arrays = {'train': pair}

    def batches(self, split, batch_size, seed=None, rows=None):
        require(split == 'train', 'Only training rows may be opened')
        yield from super().batches(split, batch_size, seed, rows)


def vector(model):
    return np.concatenate([p.detach().cpu().numpy().reshape(-1).astype(np.float64)
                           for p in model.parameters()])


def moved_model(model, displacement):
    """Apply in the real parameter dtype, then measure the rounded step too."""
    result = copy.deepcopy(model)
    target = vector(model) + displacement
    require(target.shape == vector(model).shape and np.isfinite(target).all(), 'Invalid step')
    offset = 0
    with torch.no_grad():
        for p in result.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(target[offset:offset+n].reshape(tuple(p.shape))).to(p.dtype))
            offset += n
    return result


def state_digest(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def project_delta(delta, gradients):
    """Exact small active-set QP: closest d satisfying every g @ d <= 0.

    With three safeguards there are only eight active sets. Normalizing each
    nonzero row improves conditioning without changing the halfspaces. KKT
    checks are essential: returning zero merely because it is feasible would
    silently replace the closest-update algorithm with a different one.
    """
    delta, gradients = np.asarray(delta, dtype=np.float64), np.asarray(gradients, dtype=np.float64)
    require(delta.ndim == 1 and gradients.ndim == 2 and gradients.shape[1] == len(delta)
            and len(gradients) <= 3 and np.isfinite(delta).all()
            and np.isfinite(gradients).all(), 'Invalid projection inputs')
    norms = np.linalg.norm(gradients, axis=1)
    require(np.isfinite(norms).all() and np.isfinite(np.linalg.norm(delta)), 'Projection norm overflow')
    g = gradients[norms > 0] / norms[norms > 0, None]
    tol = 1e-10 * max(1., np.linalg.norm(delta))
    if not len(g) or np.all(g @ delta <= 0):
        return delta.copy(), {'active_constraints': [], 'kkt_residual': 0., 'changed': False}
    candidates = []
    for n in range(1, len(g)+1):
        for active in itertools.combinations(range(len(g)), n):
            a = g[list(active)]
            lam = np.linalg.lstsq(a @ a.T, a @ delta, rcond=1e-12)[0]
            d = delta - a.T @ lam
            residual = max(float(np.max(g @ d)), float(np.max(np.abs(a @ d))),
                           float(max(0., -np.min(lam))))
            if residual <= tol:
                candidates.append((float(np.dot(d-delta, d-delta)), d, active, residual))
    require(bool(candidates), 'Projection could not certify KKT conditions')
    _, d, active, residual = min(candidates, key=lambda v: v[0])
    require(np.linalg.norm(d) <= np.linalg.norm(delta)+tol, 'Cone projection expanded norm')
    return d, {'active_constraints': list(active), 'kkt_residual': residual, 'changed': True}


def class_gradients(model, data, parts, batch_size=2048):
    """Pool class gradient SUMS by class counts, never average client means.

    Compute all eight classes for diagnosis. The proposed three-constraint
    training method would only upload the three protected summaries. All eight
    logits and every model parameter participate even in a single-class batch.
    Eval mode avoids consuming dropout RNG or changing the saved global model.
    """
    work = copy.deepcopy(model).eval()
    counts = np.zeros((len(parts), 8), dtype=np.int64)
    sums = np.zeros((len(parts), 8, len(vector(model))), dtype=np.float64)
    losses = np.zeros((len(parts), 8), dtype=np.float64)
    labels = data.arrays['train'][1]
    batches = 0
    for client, rows in enumerate(parts):
        for c in range(8):
            selected = rows[labels[rows] == c]
            counts[client, c] = len(selected)
            for x, y in data.batches('train', batch_size, rows=selected):
                loss = F.cross_entropy(work(torch.from_numpy(x)), torch.from_numpy(y), reduction='sum')
                require(bool(torch.isfinite(loss)), 'Nonfinite gradient loss')
                grads = torch.autograd.grad(loss, tuple(work.parameters()))
                sums[client, c] += np.concatenate([p.detach().numpy().reshape(-1).astype(np.float64)
                                                  for p in grads])
                losses[client, c] += loss.item()
                batches += 1
    total = counts.sum(axis=0)
    require(np.all(total > 0), 'Every class must have training examples')
    return sums.sum(axis=0)/total[:, None], {
        'client_counts': counts.tolist(), 'class_counts': total.tolist(),
        'mean_class_ce': (losses.sum(axis=0)/total).tolist(),
        'gradient_examples': int(total.sum()), 'gradient_backward_batches': batches,
    }


def training_evaluation(model, data, batch_size=2048):
    """Natural training prevalence; metrics are fitting diagnostics, not validation."""
    work = copy.deepcopy(model).eval()
    cm, losses = np.zeros((8, 8), dtype=np.int64), np.zeros(8, dtype=np.float64)
    with torch.inference_mode():
        for x, y in data.batches('train', batch_size):
            logits = work(torch.from_numpy(x))
            terms = F.cross_entropy(logits, torch.from_numpy(y), reduction='none').double().numpy()
            require(np.isfinite(terms).all(), 'Nonfinite evaluation loss')
            losses += np.bincount(y, weights=terms, minlength=8)
            cm += np.bincount(y*8+logits.argmax(1).numpy(), minlength=64).reshape(8, 8)
    return {'mean_class_ce': (losses/cm.sum(axis=1)).tolist(), 'metrics': confusion_metrics(cm)}


def fedavg_round(model, data, parts, settings, step):
    """Exact historical client RNG/Adam/Flower path; no corrected continuation."""
    from flwr.app import ArrayRecord, RecordDict, MetricRecord
    from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords
    check_exact_cover(parts, data.manifest['train_rows'])
    criterion = build_official_criterion(settings['loss'], data.manifest['train_class_counts'],
                                         settings['loss_reduction'])
    replies, work, deltas = [], [], []
    start = vector(model)
    for client, rows in enumerate(parts):
        seed = client_seed(settings['seed'], client, step)
        set_seed(seed)
        torch.use_deterministic_algorithms(True)
        local = MLPClassifier(39, 8, model.config)
        local.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(local.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
        loss, examples, updates = train_epoch(local, optimizer, criterion,
            data.batches('train', settings['batch_size'], seed, rows), 'cpu')
        require(examples == len(rows), 'Incomplete client epoch')
        work.append({'client': client, 'examples_processed': examples, 'optimizer_steps': updates,
                     'task_loss': loss})
        deltas.append(vector(local)-start)
        replies.append(RecordDict({'arrays': ArrayRecord(local.state_dict()),
                                   'metrics': MetricRecord({'num-examples': len(rows)})}))
    result = copy.deepcopy(model)
    result.load_state_dict(aggregate_arrayrecords(replies, 'num-examples').to_torch_state_dict())
    return result, np.stack(deltas), work


def checkpoint(archive, root, step, data):
    """Read bounded named members, never extract or execute archive content."""
    with tarfile.open(archive) as t:
        def raw(name):
            member = t.getmember(root+'/'+name)
            require(member.isfile() and member.size < 100_000_000, 'Invalid archive member')
            return t.extractfile(member).read()
        eb, cb, ab = (raw(n) for n in ('environment.json', 'last.pt', 'assignments.npz'))
        env, partition = json.loads(eb), json.loads(raw('partition.json'))
        saved = torch.load(io.BytesIO(cb), map_location='cpu', weights_only=True)
        require(saved['step'] == step and saved['environment_sha256'] == hashlib.sha256(eb).hexdigest()
                and saved['partition_sha256'] == hashlib.sha256(ab).hexdigest()
                and partition['sha256'] == saved['partition_sha256'], 'Checkpoint provenance mismatch')
        require(env['packed_manifest_sha256'] == sha256(data.root/'manifest.json'), 'Wrong cohort')
        s, control = env['settings'], env.get('partition_control', {})
        require(s['lane'] == 'dirichlet' and s['normalization'] == 'layer'
                and not s['local_max_batches'] and not s['total_update_budget']
                and not env['test_evaluated'] and control.get('cohort') == '2m'
                and control.get('policy') == 'nested-anchor-class-shares-v1'
                and control['assignment_sha256'] == saved['partition_sha256'], 'Wrong checkpoint recipe')
        with np.load(io.BytesIO(ab), allow_pickle=False) as z:
            parts = [z[f'client_{i}'].copy() for i in range(s['clients'])]
    source_checks = {name: verify_source(name, env['source_sha256'][name]) for name in SOURCE_FILES}
    check_exact_cover(parts, data.manifest['train_rows'])
    require([np.bincount(data.arrays['train'][1][p], minlength=8).tolist() for p in parts]
            == partition['counts'], 'Client counts changed')
    model = MLPClassifier(39, 8, MLPConfig(**env['model_config']))
    model.load_state_dict(saved['model'])
    return model, parts, s, {'checkpoint_sha256': hashlib.sha256(cb).hexdigest(),
                            'environment_sha256': hashlib.sha256(eb).hexdigest(),
                            'partition_sha256': saved['partition_sha256'], 'source_checks': source_checks}


def inspect_step(model, data, parts, settings, step):
    original = state_digest(model)
    gradients, gradient_work = class_gradients(model, data, parts)
    ordinary, client_deltas, client_work = fedavg_round(model, data, parts, settings, step+1)
    delta = vector(ordinary)-vector(model)
    ids = [CLASSES.index(c) for c in PROTECTED]
    projected, solver = project_delta(delta, gradients[ids])
    corrected = moved_model(model, projected)
    applied = vector(corrected)-vector(model)
    before, avg, corrected_eval = [training_evaluation(m, data) for m in (model, ordinary, corrected)]
    weights = np.asarray(list(map(len, parts)), dtype=np.float64)/data.manifest['train_rows']
    weighted = weights @ client_deltas
    # Positive dot means predicted INCREASE because these are parameter moves,
    # not gradients to be subtracted by an optimizer.
    per_class = {}
    for c, name in enumerate(CLASSES):
        per_class[name] = {
            'gradient_norm': float(np.linalg.norm(gradients[c])),
            'fedavg_predicted_ce_change': float(gradients[c] @ delta),
            'projected_predicted_ce_change': float(gradients[c] @ applied),
            'fedavg_actual_ce_change': avg['mean_class_ce'][c]-before['mean_class_ce'][c],
            'projected_actual_ce_change': corrected_eval['mean_class_ce'][c]-before['mean_class_ce'][c],
            'weighted_client_directional_contributions': (weights*(client_deltas @ gradients[c])).tolist(),
        }
    criterion = build_official_criterion(settings['loss'], data.manifest['train_class_counts'], 'fixed_train_mean')
    mass = np.asarray(gradient_work['class_counts'])*criterion.weight.numpy()
    pooled = mass @ gradients / mass.sum()
    # This shared-weight gradient is a REFERENCE direction, not another fitted
    # model. With dropout disabled it describes the global fixed-denominator CE,
    # not the historical minibatch-denominator Adam update.
    require(state_digest(model) == original, 'Frozen source model was mutated')
    return {'source_step': step, 'diagnostic_round': step+1, 'source_model_sha256': original,
            'fedavg_model_sha256': state_digest(ordinary), 'corrected_model_sha256': state_digest(corrected),
            'settings': settings, 'gradient_work': gradient_work, 'client_work': client_work,
            'ordinary_delta_norm': float(np.linalg.norm(delta)),
            'corrected_delta_norm': float(np.linalg.norm(applied)),
            'retained_norm_fraction': float(np.linalg.norm(applied)/np.linalg.norm(delta)) if np.linalg.norm(delta) else 0.,
            'correction_norm': float(np.linalg.norm(applied-delta)),
            'flower_rounding_delta_norm': float(np.linalg.norm(delta-weighted)),
            'projection_rounding_delta_norm': float(np.linalg.norm(applied-projected)),
            'pooled_fixed_objective_dot_fedavg': float(pooled @ delta),
            'cosine_with_pooled_descent': float(-pooled @ delta/(np.linalg.norm(pooled)*np.linalg.norm(delta)))
                if np.linalg.norm(pooled)*np.linalg.norm(delta) else None,
            'solver': solver, 'per_class': per_class, 'before': before, 'fedavg': avg,
            'corrected': corrected_eval, 'source_model_unchanged': True,
            'training_forward_evaluation_examples': 3*data.manifest['train_rows'],
            'hypothetical_extra_upload_bytes_per_client': len(ids)*model.parameter_bytes(),
            'diagnostic_all_class_upload_bytes_per_client': len(CLASSES)*model.parameter_bytes()}


def gate(cases, plan):
    """A strict prospective screen, NOT a validated deployment criterion."""
    tol, useful, safe = plan['loss_tolerance'], [], []
    for case in cases:
        pc = case['per_class']
        harmed = [c for c in PROTECTED[1:] if pc[c]['fedavg_predicted_ce_change'] > tol
                  and pc[c]['fedavg_actual_ce_change'] > tol]
        safe_case = (case['retained_norm_fraction'] >= plan['min_retained_norm_fraction']
                     and all(pc[c]['projected_actual_ce_change'] <= tol for c in PROTECTED))
        helps = any(pc[c]['projected_actual_ce_change'] < pc[c]['fedavg_actual_ce_change']-tol for c in harmed)
        useful.append(bool(harmed and helps and safe_case))
        safe.append(safe_case)
    passed = sum(useful) >= plan['min_useful_checkpoints'] and all(safe)
    return {'decision': 'eligible_for_separately_planned_pilot' if passed else 'stop_no_full_training',
            'useful_checkpoint_flags': useful, 'safe_checkpoint_flags': safe,
            'full_training_launched': False, 'not_a_performance_or_generalization_claim': True}


def run(data_root, archive, plan_path, output):
    output, plan_path, archive = Path(output), Path(plan_path), Path(archive)
    require(not output.exists(), 'Output exists; preserve it')
    plan = json.loads(plan_path.read_text())
    require(plan['protected_classes'] == list(PROTECTED), 'Changed protected classes')
    require(sha256(archive) == plan['archive_sha256'], 'Archive differs from frozen plan')
    require(sha256(Path(data_root)/'manifest.json') == plan['packed_manifest_sha256'], 'Wrong data')
    data = TrainOnlyData(data_root)
    torch.set_num_threads(2)
    set_seed(7)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'purpose': 'Training-only class-gradient diagnostic, not model selection',
               'plan_sha256': sha256(plan_path), 'script_sha256': sha256(__file__),
               'archive_sha256': plan['archive_sha256'], 'packed_manifest_sha256': plan['packed_manifest_sha256'],
               'device': 'cpu', 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'flwr')},
               'validation_rows_opened': False, 'test_opened': False, 'test_evaluated': False,
               'class_order': CLASSES, 'cases': []}
    atomic_json(receipt, output/'result.json')
    for choice in plan['checkpoints']:
        started = time.perf_counter()
        print(f"Inspecting frozen step {choice['step']} on training rows only", flush=True)
        model, parts, settings, provenance = checkpoint(archive, choice['member_root'], choice['step'], data)
        case = inspect_step(model, data, parts, settings, choice['step'])
        case['provenance'] = provenance
        case['elapsed_seconds'] = time.perf_counter()-started
        receipt['cases'].append(case)
        atomic_json(receipt, output/'result.json')
        print(f"step={choice['step']} retained_norm={case['retained_norm_fraction']:.4f}", flush=True)
    require(len({c['provenance']['partition_sha256'] for c in receipt['cases']}) == 1, 'Partitions differ')
    require(sha256(archive) == plan['archive_sha256'], 'Source archive changed')
    receipt['work'] = {
        'client_training_examples': sum(w['examples_processed'] for c in receipt['cases'] for w in c['client_work']),
        'client_optimizer_steps': sum(w['optimizer_steps'] for c in receipt['cases'] for w in c['client_work']),
        'gradient_examples': sum(c['gradient_work']['gradient_examples'] for c in receipt['cases']),
        'training_forward_evaluation_examples': sum(c['training_forward_evaluation_examples'] for c in receipt['cases']),
    }
    require(receipt['work'] == plan['work'], 'Diagnostic budget differs from the frozen plan')
    receipt['gate'] = gate(receipt['cases'], plan)
    receipt['status'] = 'complete'
    receipt['source_archive_unchanged'] = True
    atomic_json(receipt, output/'result.json')
    print(json.dumps(receipt['gate'], indent=2), flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--plan-path', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
