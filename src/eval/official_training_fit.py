"""Train-only fitting diagnostic on a bounded, balanced official-39 panel.

This is deliberately NOT a validation experiment. Can the existing light MLP
learn the training examples when each of eight classes gets equal exposure?
A fixed boosted-tree fit provides a nonlinear reference on precisely those rows.
Good scores here can reflect memorization; they are never deployment evidence.
Only train arrays and train-fitted scaler metadata are opened. The usual
PackedData loader also opens validation, so we intentionally do not use it here.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.models.architectures import LIGHT_CONFIG, MLPClassifier
from src.models.official_losses import build_official_criterion
from src.models.official_streaming import confusion_metrics, evaluate, train_epoch
from src.models.research import atomic_json
from src.utils.seed import set_seed


def balanced_rows(labels, per_class, seed):
    """Sample without replacement; keep the selected source row IDs for replay."""
    y = np.asarray(labels)
    if (y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or not len(y)
            or np.any(y < 0) or np.any(y >= len(CLASSES))
            or not isinstance(per_class, int) or not 1 <= per_class <= 1269):
        raise ValueError('Integer eight-class labels and a bounded positive cap required')
    rng = np.random.default_rng(seed)
    selected = []
    for c in range(len(CLASSES)):
        candidates = np.flatnonzero(y == c)
        if len(candidates) < per_class:
            raise ValueError('Every class must supply the panel count without replacement')
        selected.append(rng.choice(candidates, per_class, replace=False))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def load_training_panel(root, per_class, seed):
    """Hash/map only training files; materialize at most 10,152 x 39 features.

    The existing scaler was fitted on the full training cohort. Reusing it is
    intentional: refitting on the balanced panel would change another factor.
    Full-file hashes bind the selected examples to the original packed cohort.
    """
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest['status'] != 'complete' or manifest['classes'] != CLASSES
            or len(manifest['features']) != 39 or len(set(manifest['features'])) != 39
            or manifest['test_opened'] is not False):
        raise ValueError('Completed official-39 training provenance required')
    if sha256(root/'scaler.json') != manifest['scaler_sha256']:
        raise ValueError('Scaler checksum mismatch')
    scaler = json.loads((root/'scaler.json').read_text())
    if (scaler['fit_split'] != 'train' or scaler['n_samples_seen'] != manifest['train_rows']
            or scaler['features'] != manifest['features']):
        raise ValueError('Scaler must be fitted on this training cohort')
    for name in ('train_x.npy', 'train_y.npy'):
        if sha256(root/name) != manifest['files'][name]:
            raise ValueError('Training array checksum mismatch')
    # Do not iterate over manifest['files']: it also lists sealed split files.
    x = np.load(root/'train_x.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(root/'train_y.npy', mmap_mode='r', allow_pickle=False)
    if (x.shape != (manifest['train_rows'], 39) or y.shape != (len(x),)
            or x.dtype != np.float32 or y.dtype != np.int64):
        raise ValueError('Training shape or dtype mismatch')
    rows = balanced_rows(y, per_class, seed)
    actual = dict(zip(CLASSES, map(int, np.bincount(y, minlength=len(CLASSES)))))
    if actual != manifest['train_class_counts']:
        raise ValueError('Training label counts mismatch')
    panel_x, panel_y = np.array(x[rows], copy=True), np.array(y[rows], copy=True)
    if not np.isfinite(panel_x).all():
        raise ValueError('Nonfinite panel feature')
    return panel_x, panel_y, rows, manifest


def panel_batches(x, y, batch_size, seed=None):
    """Every panel row once per epoch; same singleton-tail rule as the runner."""
    if batch_size < 2:
        raise ValueError('Batch size must be at least two')
    order = np.arange(len(y))
    if seed is not None:
        np.random.default_rng(seed).shuffle(order)
    start = 0
    while start < len(order):
        end = min(start+batch_size, len(order))
        if len(order)-end == 1:
            end += 1
        take = order[start:end]
        yield np.array(x[take], copy=True), np.array(y[take], copy=True)
        start = end


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().astype('<f4', copy=False).tobytes())
    return digest.hexdigest()


def fit_mlp(x, y, settings, seed, *, dropout=None, prefix_reference=None, hidden_dims=None):
    """Fresh initialization; balanced counts make square-root CE uniform CE.

    Do NOT reuse the original imbalanced cohort weights on balanced examples:
    that would double-compensate for frequency and change this fitting question.
    Widths, dropout, LayerNorm, Adam and batch size match the light model recipe.
    No early stopping or best-checkpoint selection is performed.

    The separate dropout-only diagnostic may explicitly disable dropout. The
    default path stays unchanged so it can be bridged to the archived fitting
    receipt exactly. This option never changes the production architecture.

    A duration control may supply an earlier fixed-budget receipt. Before the
    next epoch, compare the entire prefix (including parameter hash and work).
    The SAME optimizer then continues; neither Adam nor the RNG is restarted.

    The bounded width diagnostic can request (128, 64) instead of (64, 32).
    This pairs seeds and initialization policy, NOT identical starting tensors
    across widths. Defaults and the production architecture remain unchanged.
    """
    counts = dict(zip(CLASSES, map(int, np.bincount(y, minlength=len(CLASSES)))))
    if len(set(counts.values())) != 1 or min(counts.values()) <= 0:
        raise ValueError('The fitting panel must be balanced across all eight classes')
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    if dropout is not None and dropout not in (0., LIGHT_CONFIG.dropout):
        raise ValueError('Only the default dropout or the zero-dropout control is supported')
    widths = LIGHT_CONFIG.hidden_dims if hidden_dims is None else tuple(hidden_dims)
    if widths not in ((64, 32), (128, 64)):
        raise ValueError('Only the reference or the bounded wider control is supported')
    config = replace(LIGHT_CONFIG, normalization=settings['normalization'],
                     dropout=LIGHT_CONFIG.dropout if dropout is None else dropout,
                     hidden_dims=widths)
    model = MLPClassifier(x.shape[1], len(CLASSES), config)
    initial = state_hash(model)
    criterion = build_official_criterion(settings['loss'], counts, settings['loss_reduction'])
    # Floating-point normalization can yield 0.99999994 instead of exactly 1;
    # equality ACROSS classes is what makes weighted-mean CE uniform here.
    if (not torch.equal(criterion.weight, criterion.weight[0].expand(len(CLASSES)))
            or not torch.allclose(criterion.weight, torch.ones(len(CLASSES)))):
        raise ValueError('Balanced-panel weights must be uniform')
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    observations, examples, updates = [], 0, 0
    start = time.perf_counter()
    prefix_epoch = None
    if prefix_reference is not None:
        prefix_epoch = prefix_reference['observations'][-1]['epoch']
        if (not 1 <= prefix_epoch < settings['epochs']
                or [e for e in settings['report_epochs'] if e <= prefix_epoch]
                != [o['epoch'] for o in prefix_reference['observations']]):
            raise ValueError('Duration control requires the unchanged earlier observation schedule')

    def snapshot():
        return {'seed': seed, 'model_config': asdict(config), 'num_parameters': model.num_parameters(),
                'initial_state_sha256': initial, 'final_state_sha256': state_hash(model),
                'examples_processed': examples, 'optimizer_steps': updates,
                'class_weights': criterion.weight.tolist(), 'observations': observations,
                'elapsed_seconds': time.perf_counter()-start}

    for epoch in range(1, settings['epochs']+1):
        loss, seen, steps = train_epoch(model, optimizer, criterion,
            panel_batches(x, y, settings['batch_size'], seed+epoch), 'cpu')
        if seen != len(y):
            raise ValueError('Incomplete panel exposure')
        examples += seen
        updates += steps
        if epoch in settings['report_epochs']:
            ce, metrics = evaluate(model, panel_batches(x, y, 2048), 'cpu')
            observations.append({'epoch': epoch, 'train_batch_mean_loss': loss,
                                 'train_eval_cross_entropy': ce, 'training_panel_metrics': metrics})
            print(f'mlp seed={seed} epoch={epoch}: TRAIN macro-F1={metrics["macro_f1"]:.6f}', flush=True)
        if epoch == prefix_epoch:
            # Normalize tuples to JSON lists, but do not round numbers. Timing
            # alone is excluded. A failed bridge stops BEFORE additional work.
            def numerical(record):
                return json.loads(json.dumps({k: v for k, v in record.items() if k != 'elapsed_seconds'}))
            if numerical(snapshot()) != numerical(prefix_reference):
                raise ValueError('Training prefix differs from the reference; extension stopped')
    batches_per_epoch = ((len(y)+settings['batch_size']-1)//settings['batch_size']
                         - int(len(y) > settings['batch_size'] and len(y) % settings['batch_size'] == 1))
    if (examples != len(y)*settings['epochs']
            or updates != batches_per_epoch*settings['epochs']):
        raise ValueError('Fixed neural training budget did not match')
    result = snapshot()
    if prefix_reference is not None:
        result['prefix_bridge_exact'] = True
        result['prefix_epoch'] = prefix_epoch
        result['prefix_state_sha256'] = prefix_reference['final_state_sha256']
    return result


def fit_tree(x, y, settings):
    """Fixed boosting budget, no internal validation holdout or early stopping."""
    if settings['early_stopping'] is not False or settings['validation_fraction'] is not None:
        raise ValueError('No validation or early stopping allowed in this fitting diagnostic')
    model = HistGradientBoostingClassifier(**settings)
    start = time.perf_counter()
    model.fit(x, y)
    pred = model.predict(x)
    if (model.n_iter_ != settings['max_iter'] or model.do_early_stopping_
            or not np.array_equal(model.classes_, np.arange(len(CLASSES)))):
        raise ValueError('Unexpected tree budget or label mapping')
    matrix = np.bincount(y*len(CLASSES)+pred, minlength=len(CLASSES)**2).reshape(len(CLASSES), len(CLASSES))
    return {'settings': settings, 'iterations': int(model.n_iter_),
            'trees': int(model.n_iter_*model.n_trees_per_iteration_),
            'training_panel_metrics': confusion_metrics(matrix),
            'training_prediction_sha256': hashlib.sha256(pred.astype('<i8').tobytes()).hexdigest(),
            'elapsed_seconds': time.perf_counter()-start}


def validate_plan(plan):
    """Fail closed on expansion beyond the bounded, training-only protocol."""
    m, p, t = plan['mlp'], plan['panel'], plan['hist_gradient_boosting']
    if (not m['report_epochs'] or not m['seeds']
            or plan['protocol'] != 'official39-balanced-training-fit-v1'
            or plan['validation_opened'] is not False or plan['test_opened'] is not False
            or p['replacement'] is not False or not 1 <= p['per_class'] <= 1269
            or not 1 <= m['epochs'] <= 100 or m['report_epochs'][-1] != m['epochs']
            or sorted(set(m['report_epochs'])) != m['report_epochs'] or m['report_epochs'][0] < 1
            or not 1 <= len(m['seeds']) <= 3 or len(set(m['seeds'])) != len(m['seeds'])
            or min(m['seeds']) < 0 or not 1 <= plan['threads'] <= 2
            or m['normalization'] != 'layer' or m['loss'] != 'sqrt_weighted_ce'
            or m['loss_reduction'] != 'batch_weight_sum' or m['class_weight_counts'] != 'balanced_panel_only'
            or not 1 <= t['max_iter'] <= 100 or not 2 <= t['max_leaf_nodes'] <= 31
            or t['early_stopping'] is not False or t['validation_fraction'] is not None):
        raise ValueError('Plan exceeds the bounded training-only protocol')


def diagnose(data_root, plan_path, output):
    output, data_root = Path(output), Path(data_root)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(Path(plan_path).read_text())
    validate_plan(plan)
    if sha256(data_root/'manifest.json') != plan['packed_manifest_sha256']:
        raise ValueError('Plan and packed manifest differ')
    x, y, rows, manifest = load_training_panel(data_root, plan['panel']['per_class'], plan['panel']['seed'])
    if manifest['train_rows'] != plan['expected_training_rows']:
        raise ValueError('Unexpected training cohort')
    source_root = Path(__file__).parents[1]
    # Pin the shared loop/architecture/loss helpers as well as this diagnostic.
    sources = ('eval/official_training_fit.py', 'models/architectures.py', 'models/losses.py',
               'models/official_losses.py', 'models/official_streaming.py', 'models/research.py',
               'data/label_map.py', 'data/official_inventory.py', 'utils/seed.py')
    receipt = {'status': 'incomplete', 'purpose': plan['purpose'], 'device': 'cpu',
               'validation_opened': False, 'test_opened': False, 'model_promoted': False,
               'plan': plan, 'plan_sha256': sha256(plan_path),
               'source_sha256': {p: sha256(source_root/p) for p in sources},
               'packed_manifest_sha256': plan['packed_manifest_sha256'],
               'training_file_sha256': {n: manifest['files'][n] for n in ('train_x.npy', 'train_y.npy')},
               'scaler_sha256': manifest['scaler_sha256'], 'features': manifest['features'],
               'classes': CLASSES, 'panel_rows': rows.tolist(),
               'panel_sha256': hashlib.sha256(rows.astype('<i8').tobytes()+x.astype('<f4').tobytes()+y.astype('<i8').tobytes()).hexdigest(),
               'panel_counts': dict(zip(CLASSES, map(int, np.bincount(y, minlength=len(CLASSES))))),
               'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scikit-learn', 'threadpoolctl')},
               'mlp_fits': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt, output)
    torch.set_num_threads(plan['threads'])
    with threadpool_limits(limits=plan['threads']):
        for seed in plan['mlp']['seeds']:
            receipt['mlp_fits'].append(fit_mlp(x, y, plan['mlp'], seed))
            atomic_json(receipt, output)
        receipt['tree_fit'] = fit_tree(x, y, plan['hist_gradient_boosting'])
    if sha256(plan_path) != receipt['plan_sha256']:
        raise ValueError('Plan changed during fitting')
    if any(sha256(source_root/p) != digest for p, digest in receipt['source_sha256'].items()):
        raise ValueError('Source changed during fitting')
    receipt['status'] = 'complete'
    atomic_json(receipt, output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--plan-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    diagnose(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
