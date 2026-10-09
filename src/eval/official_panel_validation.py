"""Validate frozen panel-trained models, never tune them or open the final test.

All training replays must match before validation files are even hashed. The
small final models are kept in memory; the large validation matrix stays on
disk. Validation cannot influence training, endpoint choice or alert thresholds.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.official_dropout_fit import comparable, check_pair
from src.eval.official_duration_fit import CONDITIONS
from src.eval.official_training_fit import fit_mlp, fit_tree, load_training_panel, state_hash, validate_plan
from src.eval.official_width_fit import check_width_pair, prefix_from_duration
from src.models.official_streaming import evaluate, confusion_metrics
from src.models.research import atomic_json, atomic_save


def check_protocol(plan, reference, tree_reference):
    """Allow evaluation of exactly the archived fits, not a new model search."""
    if (plan['protocol'] != 'official39-frozen-panel-validation-v1'
            or plan['hidden_dims'] != [[64, 32], [128, 64]]
            or plan['dropout_values'] != [.2, 0.]
            or plan['prediction_rule'] != 'argmax' or plan['include_fixed_tree'] is not True
            or plan['require_all_training_bridges_before_validation'] is not True
            or plan['test_opened'] is not False or plan['model_promoted'] is not False
            or not 1 <= plan['epochs'] <= 300 or not 1 <= len(plan['seeds']) <= 3
            or not 1 <= plan['evaluation_batch_size'] <= 8192
            or set(plan['validation_class_counts']) != set(CLASSES)
            or min(plan['validation_class_counts'].values()) <= 0
            or sum(plan['validation_class_counts'].values()) != plan['validation_rows']):
        raise ValueError('Plan exceeds the fixed validation comparison')
    if (reference['status'] != 'complete' or reference['all_baseline_bridges_exact'] is not True
            or reference['device'] != 'cpu' or reference['validation_opened'] is not False
            or reference['test_opened'] is not False or reference['model_promoted'] is not False
            or reference['plan_sha256'] != plan['reference_plan_sha256']
            or reference['panel_sha256'] != plan['panel_sha256']
            or reference['source_sha256']['eval/official_training_fit.py'] != plan['previous_fitter_sha256']):
        raise ValueError('Reference is not the approved completed width comparison')
    validate_plan(reference['base_plan'])
    settings = reference['mlp_settings']
    if settings['epochs'] != plan['epochs'] or settings['seeds'] != plan['seeds']:
        raise ValueError('Frozen neural budget or seeds differ')
    for width in ('small_fits', 'wide_fits'):
        for key, dropout in CONDITIONS:
            fits = reference[width][key]
            if [f['seed'] for f in fits] != plan['seeds']:
                raise ValueError('Incomplete reference seed coverage')
            for fit in fits:
                if (fit['model_config']['dropout'] != dropout
                        or [o['epoch'] for o in fit['observations']] != settings['report_epochs']):
                    raise ValueError('Frozen dropout or observation schedule differs')
        for pair in zip(reference[width]['baseline_fits'], reference[width]['zero_dropout_fits']):
            check_pair(*pair)
    for key, _ in CONDITIONS:
        for pair in zip(reference['small_fits'][key], reference['wide_fits'][key]):
            check_width_pair(*pair)
    if (tree_reference['status'] != 'complete' or tree_reference['device'] != 'cpu'
            or tree_reference['validation_opened'] is not False or tree_reference['test_opened'] is not False
            or tree_reference['model_promoted'] is not False
            or tree_reference['plan'] != reference['base_plan']
            or tree_reference['panel_sha256'] != plan['panel_sha256']
            or tree_reference['packed_manifest_sha256'] != reference['packed_manifest_sha256']
            or tree_reference['tree_fit']['settings'] != reference['base_plan']['hist_gradient_boosting']):
        raise ValueError('Tree reference differs from the frozen panel recipe')


def validation_batches(x, y, batch_size):
    """Sequential bounded copies, with no shuffling or split-wide allocation."""
    if not 1 <= batch_size <= 8192:
        raise ValueError('Validation batches must be bounded')
    for start in range(0, len(y), batch_size):
        xb = np.array(x[start:start+batch_size], copy=True)
        yb = np.array(y[start:start+batch_size], copy=True)
        if (not np.isfinite(xb).all() or np.any(yb < 0) or np.any(yb >= len(CLASSES))):
            raise ValueError('Invalid validation values or labels')
        yield xb, yb


def load_validation(root, manifest, plan):
    """Explicit validation allowlist: never iterate over all manifest files."""
    root = Path(root)
    for name in ('val_x.npy', 'val_y.npy'):
        if sha256(root/name) != manifest['files'][name]:
            raise ValueError('Validation checksum mismatch')
    x = np.load(root/'val_x.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(root/'val_y.npy', mmap_mode='r', allow_pickle=False)
    if (manifest['val_rows'] != plan['validation_rows']
            or x.shape != (plan['validation_rows'], 39) or y.shape != (len(x),)
            or x.dtype != np.float32 or y.dtype != np.int64):
        raise ValueError('Validation shape or dtype mismatch')
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    for _, labels in validation_batches(x, y, plan['evaluation_batch_size']):
        counts += np.bincount(labels, minlength=len(CLASSES))
    if dict(zip(CLASSES, map(int, counts))) != plan['validation_class_counts']:
        raise ValueError('Validation class counts differ from the plan')
    return x, y


def evaluate_tree(model, batches):
    """The frozen boosted tree uses its usual highest-probability class rule."""
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    digest = hashlib.sha256()
    for x, y in batches:
        pred = model.predict(x).astype(np.int64)
        digest.update(pred.astype('<i8').tobytes())
        matrix += np.bincount(y*len(CLASSES)+pred, minlength=len(CLASSES)**2).reshape(matrix.shape)
    if not matrix.sum():
        raise ValueError('Empty validation iterator')
    return confusion_metrics(matrix), digest.hexdigest()


def summarize(records):
    """Report all training seeds, not a chosen winner or a confidence interval."""
    def stats(values):
        return {'mean': statistics.mean(values),
                'sample_sd': statistics.stdev(values) if len(values) > 1 else None}
    result = {}
    for condition in dict.fromkeys(r['condition'] for r in records):
        rows = [r['validation_metrics'] for r in records if r['condition'] == condition]
        result[condition] = {key: stats([m[key] for m in rows])
                            for key in ('macro_f1', 'benign_false_alert_rate')}
        result[condition]['per_class'] = {c: {key: stats([m['per_class'][c][key] for m in rows])
                                              for key in ('precision', 'recall', 'f1')} for c in CLASSES}
    return result


def diagnose(data_root, reference_path, tree_reference_path, plan_path, output):
    output, data_root = Path(output), Path(data_root)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(Path(plan_path).read_text())
    if (sha256(reference_path) != plan['reference_receipt_sha256']
            or sha256(tree_reference_path) != plan['tree_reference_sha256']):
        raise ValueError('Reference checksum mismatch')
    reference = json.loads(Path(reference_path).read_text())
    tree_reference = json.loads(Path(tree_reference_path).read_text())
    check_protocol(plan, reference, tree_reference)
    base, settings = reference['base_plan'], reference['mlp_settings']
    if sha256(data_root/'manifest.json') != reference['packed_manifest_sha256']:
        raise ValueError('Packed manifest differs from the frozen reference')
    x, y, rows, manifest = load_training_panel(data_root, base['panel']['per_class'], base['panel']['seed'])
    panel_hash = hashlib.sha256(rows.astype('<i8').tobytes()+x.astype('<f4').tobytes()+y.astype('<i8').tobytes()).hexdigest()
    if panel_hash != plan['panel_sha256'] or manifest['train_rows'] != base['expected_training_rows']:
        raise ValueError('Training panel differs from the reference')
    source_root, sources = Path(__file__).parents[1], {}
    for name, old_hash in reference['source_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid source path')
        sources[name] = sha256(source_root/path)
        # Retaining the trained objects is the only fitter change. Every full
        # training record must still pass its numerical bridge below.
        if name != 'eval/official_training_fit.py' and sources[name] != old_hash:
            raise ValueError('An unchanged shared source differs from the reference')
    sources['eval/official_panel_validation.py'] = sha256(__file__)
    receipt = {'status': 'incomplete', 'purpose': plan['purpose'], 'device': 'cpu',
               'plan': plan, 'plan_sha256': sha256(plan_path), 'panel_sha256': panel_hash,
               'packed_manifest_sha256': reference['packed_manifest_sha256'],
               'source_sha256': sources, 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scikit-learn', 'threadpoolctl')},
               'all_training_bridges_exact': False, 'validation_opened': False,
               'test_opened': False, 'model_promoted': False, 'neural_fits': [], 'validation_results': []}
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = output/'receipt.json'
    atomic_json(receipt, receipt_path)

    def check_inputs():
        if (sha256(plan_path) != receipt['plan_sha256']
                or sha256(reference_path) != plan['reference_receipt_sha256']
                or sha256(tree_reference_path) != plan['tree_reference_sha256']
                or sha256(data_root/'manifest.json') != receipt['packed_manifest_sha256']
                or any(sha256(source_root/p) != digest for p, digest in sources.items())):
            raise ValueError('Plan, reference, manifest or source changed during comparison')

    torch.set_num_threads(base['threads'])
    models = []
    with threadpool_limits(limits=base['threads']):
        for width in ('small_fits', 'wide_fits'):
            for key, dropout in CONDITIONS:
                for old in reference[width][key]:
                    prefix = prefix_from_duration(old) if width == 'small_fits' else None
                    fit, model = fit_mlp(x, y, settings, old['seed'], dropout=dropout,
                        hidden_dims=old['model_config']['hidden_dims'], prefix_reference=prefix, return_model=True)
                    if comparable(fit) != comparable(old) or state_hash(model) != old['final_state_sha256']:
                        raise ValueError('Neural training bridge failed; validation remains sealed')
                    condition = ('small' if width == 'small_fits' else 'wide') + ('_dropout' if dropout else '_no_dropout')
                    name = f'{condition}_seed{old["seed"]}'
                    # Diagnostic final weights, not best checkpoints or production
                    # promotion. Keep them so future analysis can reuse these fits.
                    checkpoint = output/(name+'.pt')
                    atomic_save({'state_dict': model.state_dict(), 'model_config': fit['model_config'],
                                 'panel_sha256': panel_hash, 'features': manifest['features'],
                                 'classes': CLASSES, 'seed': old['seed'], 'epochs': settings['epochs']}, checkpoint)
                    receipt['neural_fits'].append({'condition': condition, 'name': name, 'training_fit': fit,
                                                 'checkpoint': checkpoint.name, 'checkpoint_sha256': sha256(checkpoint)})
                    models.append((condition, name, old['seed'], model, fit['final_state_sha256']))
                    atomic_json(receipt, receipt_path)
        tree_fit, tree_model = fit_tree(x, y, base['hist_gradient_boosting'], return_model=True)
        if comparable(tree_fit) != comparable(tree_reference['tree_fit']):
            raise ValueError('Tree training bridge failed; validation remains sealed')
        receipt['tree_fit'] = tree_fit
        receipt['all_training_bridges_exact'] = True
        check_inputs()
        # This flag is persisted before attempting any validation file access.
        receipt['validation_opened'] = True
        atomic_json(receipt, receipt_path)
        vx, vy = load_validation(data_root, manifest, plan)
        receipt['validation_file_sha256'] = {n: manifest['files'][n] for n in ('val_x.npy', 'val_y.npy')}
        receipt['validation_class_counts'] = plan['validation_class_counts']
        for condition, name, seed, model, final_hash in models:
            start = time.perf_counter()
            loss, metrics = evaluate(model, validation_batches(vx, vy, plan['evaluation_batch_size']), 'cpu')
            if state_hash(model) != final_hash:
                raise ValueError('Evaluation modified final model parameters')
            if {c: metrics['per_class'][c]['support'] for c in CLASSES} != plan['validation_class_counts']:
                raise ValueError('Incomplete validation coverage')
            receipt['validation_results'].append({'condition': condition, 'name': name, 'seed': seed,
                'validation_cross_entropy': loss, 'validation_metrics': metrics,
                'elapsed_seconds': time.perf_counter()-start})
            print(f'{name}: VALIDATION macro-F1={metrics["macro_f1"]:.6f} FAR={metrics["benign_false_alert_rate"]:.4%}', flush=True)
            atomic_json(receipt, receipt_path)
        start = time.perf_counter()
        metrics, prediction_hash = evaluate_tree(tree_model, validation_batches(vx, vy, plan['evaluation_batch_size']))
        if {c: metrics['per_class'][c]['support'] for c in CLASSES} != plan['validation_class_counts']:
            raise ValueError('Incomplete tree validation coverage')
        receipt['validation_results'].append({'condition': 'fixed_tree', 'name': 'fixed_tree_seed7',
            'seed': base['hist_gradient_boosting']['random_state'], 'validation_metrics': metrics,
            'prediction_sha256': prediction_hash, 'elapsed_seconds': time.perf_counter()-start})
        print(f'fixed_tree: VALIDATION macro-F1={metrics["macro_f1"]:.6f} FAR={metrics["benign_false_alert_rate"]:.4%}', flush=True)
    check_inputs()
    # Hash only the four allowed arrays, never a generic split glob or test path.
    for name in ('train_x.npy', 'train_y.npy', 'val_x.npy', 'val_y.npy'):
        if sha256(data_root/name) != manifest['files'][name]:
            raise ValueError('Data changed during comparison')
    receipt['summary'] = summarize(receipt['validation_results'])
    receipt['status'] = 'complete'
    atomic_json(receipt, receipt_path)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'reference-path', 'tree-reference-path', 'plan-path', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    diagnose(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
