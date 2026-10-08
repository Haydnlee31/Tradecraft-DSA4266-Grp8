"""One width-only training-fit control, gated by six exact historical bridges.

The smaller fits are replayed first, including their archived 100-epoch prefix
checks. Only after EVERY bridge passes do we fit the wider model. Both sizes
see the same rows and updates; neither FLOPs nor wall time is matched. No
validation/test loader or production architecture change belongs in this module.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from threadpoolctl import threadpool_limits
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.official_dropout_fit import check_pair, comparable
from src.eval.official_duration_fit import CONDITIONS
from src.eval.official_training_fit import fit_mlp, load_training_panel, validate_plan
from src.models.research import atomic_json


def prefix_from_duration(fit):
    """Reconstruct the archived short prefix without touching model or RNG state.

    Duration receipts retain observations through the short endpoint and its
    exact weight hash. Work is constant per epoch, so integer division recovers
    the earlier counters. The replay must match this prefix AND the final fit.
    """
    result = comparable(fit)
    epoch = result.pop('prefix_epoch')
    if result.pop('prefix_bridge_exact') is not True:
        raise ValueError('Reference has no verified training prefix')
    final_epoch = result['observations'][-1]['epoch']
    if not 1 <= epoch < final_epoch:
        raise ValueError('Invalid prefix endpoint')
    result['final_state_sha256'] = result.pop('prefix_state_sha256')
    for key in ('examples_processed', 'optimizer_steps'):
        if result[key] <= 0 or result[key] % final_epoch:
            raise ValueError('Reference work is not a constant integer per epoch')
        result[key] = result[key] // final_epoch * epoch
    result['observations'] = [o for o in result['observations'] if o['epoch'] <= epoch]
    if not result['observations'] or result['observations'][-1]['epoch'] != epoch:
        raise ValueError('Missing prefix observation')
    return result


def check_width_pair(small, wide):
    """Same update budget and recipe; width-dependent initial tensors may differ."""
    for key in ('seed', 'examples_processed', 'optimizer_steps', 'class_weights'):
        if small[key] != wide[key]:
            raise ValueError(f'Width pair differs in {key}')
    a, b = (comparable(f)['model_config'] for f in (small, wide))
    if a.pop('hidden_dims') != [64, 32] or b.pop('hidden_dims') != [128, 64] or a != b:
        raise ValueError('Configurations differ beyond the width control')
    if small['num_parameters'] != 5096 or wide['num_parameters'] != 14280:
        raise ValueError('Unexpected official-39 parameter counts')
    if [o['epoch'] for o in small['observations']] != [o['epoch'] for o in wide['observations']]:
        raise ValueError('Width pair observation endpoints differ')


def check_protocol(plan, reference):
    if (plan['protocol'] != 'official39-width-only-training-fit-v1'
            or plan['reference_hidden_dims'] != [64, 32]
            or plan['treatment_hidden_dims'] != [128, 64]
            or plan['dropout_values'] != [.2, 0.]
            or not 1 <= plan['epochs'] <= 300
            or plan['require_all_exact_baseline_bridges_before_treatment'] is not True
            or plan['validation_opened'] is not False or plan['test_opened'] is not False):
        raise ValueError('Plan exceeds the bounded width-only control')
    if (reference['status'] != 'complete' or reference['all_prefixes_exact'] is not True
            or reference['device'] != 'cpu' or reference['validation_opened'] is not False
            or reference['test_opened'] is not False or reference['model_promoted'] is not False
            or reference['plan_sha256'] != plan['reference_plan_sha256']
            or reference['panel_sha256'] != plan['panel_sha256']
            or reference['source_sha256']['eval/official_training_fit.py'] != plan['previous_fitter_sha256']):
        raise ValueError('Reference is not the approved completed duration diagnostic')
    validate_plan(reference['base_plan'])
    base, settings = reference['base_plan']['mlp'], reference['extended_mlp_settings']
    unchanged = copy.deepcopy(settings)
    unchanged['epochs'], unchanged['report_epochs'] = base['epochs'], base['report_epochs']
    if (unchanged != base or settings['epochs'] != plan['epochs']
            or settings['report_epochs'] != sorted(set(settings['report_epochs']))
            or settings['report_epochs'][-1] != plan['epochs']):
        raise ValueError('Duration settings differ beyond the approved extended budget')
    rows = reference['base_plan']['panel']['per_class'] * len(CLASSES)
    batch = settings['batch_size']
    batches = (rows+batch-1)//batch - int(rows > batch and rows % batch == 1)
    for key, dropout in CONDITIONS:
        if [f['seed'] for f in reference[key]] != base['seeds']:
            raise ValueError('Reference seed coverage mismatch')
        for fit in reference[key]:
            if (fit['model_config']['dropout'] != dropout
                    or list(fit['model_config']['hidden_dims']) != [64, 32]
                    or fit['model_config']['normalization'] != 'layer'
                    or fit['num_parameters'] != 5096
                    or [o['epoch'] for o in fit['observations']] != settings['report_epochs']
                    or fit['examples_processed'] != rows*settings['epochs']
                    or fit['optimizer_steps'] != batches*settings['epochs']):
                raise ValueError('Reference condition, observation or work mismatch')
            short = prefix_from_duration(fit)
            if [o['epoch'] for o in short['observations']] != base['report_epochs']:
                raise ValueError('Reference prefix differs from its base plan')
    for pair in zip(reference['baseline_fits'], reference['zero_dropout_fits']):
        check_pair(*pair)


def width_change(small, wide):
    check_width_pair(small, wide)
    a, b = (f['observations'][-1]['training_panel_metrics'] for f in (small, wide))
    return {'seed': small['seed'], 'dropout': small['model_config']['dropout'],
            'macro_f1_change': b['macro_f1']-a['macro_f1'],
            'benign_false_alert_rate_change': b['benign_false_alert_rate']-a['benign_false_alert_rate'],
            'per_class_recall_change': {c: b['per_class'][c]['recall']-a['per_class'][c]['recall'] for c in CLASSES}}


def diagnose(data_root, reference_path, plan_path, output):
    output, data_root = Path(output), Path(data_root)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(Path(plan_path).read_text())
    if sha256(reference_path) != plan['reference_receipt_sha256']:
        raise ValueError('Reference checksum mismatch')
    reference = json.loads(Path(reference_path).read_text())
    check_protocol(plan, reference)
    base, settings = reference['base_plan'], reference['extended_mlp_settings']
    if sha256(data_root/'manifest.json') != reference['packed_manifest_sha256']:
        raise ValueError('Training cohort differs from the reference')
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
        # Only the opt-in width argument changes. The mandatory replay below
        # verifies every original fit, not just its final aggregate score.
        if name != 'eval/official_training_fit.py' and sources[name] != old_hash:
            raise ValueError('An unchanged shared source differs from the reference')
    sources['eval/official_width_fit.py'] = sha256(__file__)
    receipt = {'status': 'incomplete', 'purpose': plan['purpose'], 'device': 'cpu',
               'plan': plan, 'plan_sha256': sha256(plan_path),
               'reference_receipt_sha256': plan['reference_receipt_sha256'],
               'base_plan': base, 'mlp_settings': settings,
               'panel_sha256': panel_hash, 'panel_counts': reference['panel_counts'],
               'packed_manifest_sha256': reference['packed_manifest_sha256'],
               'source_sha256': sources, 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scikit-learn', 'threadpoolctl')},
               'validation_opened': False, 'test_opened': False, 'model_promoted': False,
               'all_baseline_bridges_exact': False,
               'small_fits': {key: [] for key, _ in CONDITIONS},
               'wide_fits': {key: [] for key, _ in CONDITIONS}, 'width_changes': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt, output)
    torch.set_num_threads(base['threads'])
    with threadpool_limits(limits=base['threads']):
        for key, dropout in CONDITIONS:
            for old in reference[key]:
                fit = fit_mlp(x, y, settings, old['seed'], dropout=dropout,
                              prefix_reference=prefix_from_duration(old))
                if comparable(fit) != comparable(old):
                    raise ValueError('Baseline bridge failed; width treatments were not launched')
                receipt['small_fits'][key].append(fit)
                atomic_json(receipt, output)
        receipt['all_baseline_bridges_exact'] = True
        atomic_json(receipt, output)
        # This loop cannot begin until all seeds AND dropout conditions bridge.
        for key, dropout in CONDITIONS:
            for small in receipt['small_fits'][key]:
                fit = fit_mlp(x, y, settings, small['seed'], dropout=dropout,
                              hidden_dims=plan['treatment_hidden_dims'])
                receipt['width_changes'].append(width_change(small, fit))
                receipt['wide_fits'][key].append(fit)
                atomic_json(receipt, output)
    for pair in zip(receipt['wide_fits']['baseline_fits'], receipt['wide_fits']['zero_dropout_fits']):
        check_pair(*pair)
    if (sha256(plan_path) != receipt['plan_sha256']
            or sha256(reference_path) != receipt['reference_receipt_sha256']
            or any(sha256(source_root/p) != digest for p, digest in sources.items())):
        raise ValueError('Plan, reference or source changed during diagnosis')
    receipt['status'] = 'complete'
    atomic_json(receipt, output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'reference-path', 'plan-path', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    diagnose(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
