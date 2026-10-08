"""Paired dropout-only follow-up to the frozen balanced training-fit receipt.

First reproduce EVERY historical neural fit with the new helper's default path.
Only then disable dropout, preserving the panel, initialization, optimizer,
epochs and shuffling. Both conditions evaluate training rows only. No validation
or test data are opened, and no deployment setting or checkpoint is modified.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.official_training_fit import load_training_panel, fit_mlp, validate_plan
from src.models.research import atomic_json


def comparable(fit):
    """Ignore wall time, but compare every numerical observation and weight hash.

    JSON normalization aligns tuple-valued configs with archived JSON lists.
    It does not round any floats or discard seed, work, or configuration fields.
    """
    return json.loads(json.dumps({k: v for k, v in fit.items() if k != 'elapsed_seconds'}))


def check_pair(reference, treatment):
    """Identical starting tensors and settings except for the dropout module."""
    for key in ('seed', 'initial_state_sha256', 'num_parameters', 'examples_processed',
                'optimizer_steps', 'class_weights'):
        if reference[key] != treatment[key]:
            raise ValueError(f'Paired control differs in {key}')
    a, b = (comparable(f)['model_config'] for f in (reference, treatment))
    if a.pop('dropout') != .2 or b.pop('dropout') != 0. or a != b:
        raise ValueError('Model configurations differ beyond the dropout control')
    if [o['epoch'] for o in reference['observations']] != [o['epoch'] for o in treatment['observations']]:
        raise ValueError('Observation endpoints differ')


def paired_changes(reference, treatment):
    check_pair(reference, treatment)
    a, b = (f['observations'][-1]['training_panel_metrics'] for f in (reference, treatment))
    return {'seed': reference['seed'], 'epoch': reference['observations'][-1]['epoch'],
            'macro_f1_change': b['macro_f1']-a['macro_f1'],
            'benign_false_alert_rate_change': b['benign_false_alert_rate']-a['benign_false_alert_rate'],
            'per_class_recall_change': {c: b['per_class'][c]['recall']-a['per_class'][c]['recall'] for c in CLASSES}}


def check_protocol(plan, reference):
    if (plan['protocol'] != 'official39-dropout-only-training-fit-v1'
            or plan['dropout_values'] != [.2, 0.]
            or plan['inherit_panel_and_neural_settings_from_reference'] is not True
            or plan['require_exact_baseline_bridge'] is not True
            or plan['require_identical_initial_weights'] is not True
            or plan['validation_opened'] is not False or plan['test_opened'] is not False):
        raise ValueError('Only the frozen paired dropout control is supported')
    if (reference['status'] != 'complete' or reference['device'] != 'cpu'
            or reference['validation_opened'] is not False or reference['test_opened'] is not False
            or reference['plan_sha256'] != plan['reference_plan_sha256']
            or reference['panel_sha256'] != plan['panel_sha256']
            or reference['source_sha256']['eval/official_training_fit.py'] != plan['previous_fitter_sha256']):
        raise ValueError('Reference is not the approved completed training-only fit')
    validate_plan(reference['plan'])
    if [f['seed'] for f in reference['mlp_fits']] != reference['plan']['mlp']['seeds']:
        raise ValueError('Reference seed coverage differs from its plan')


def diagnose(data_root, reference_path, plan_path, output):
    output, data_root = Path(output), Path(data_root)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(Path(plan_path).read_text())
    if sha256(reference_path) != plan['reference_receipt_sha256']:
        raise ValueError('Reference receipt checksum mismatch')
    reference = json.loads(Path(reference_path).read_text())
    check_protocol(plan, reference)
    base = reference['plan']
    if sha256(data_root/'manifest.json') != reference['packed_manifest_sha256']:
        raise ValueError('Training cohort differs from the reference')
    x, y, rows, manifest = load_training_panel(data_root, base['panel']['per_class'], base['panel']['seed'])
    panel_hash = hashlib.sha256(rows.astype('<i8').tobytes()+x.astype('<f4').tobytes()+y.astype('<i8').tobytes()).hexdigest()
    if (panel_hash != reference['panel_sha256'] or rows.tolist() != reference['panel_rows']
            or manifest['train_rows'] != base['expected_training_rows']):
        raise ValueError('The selected panel differs from the previous diagnostic')
    source_root = Path(__file__).parents[1]
    sources = {}
    for name, old_hash in reference['source_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid source path in reference')
        sources[name] = sha256(source_root/path)
        # Only the explicit dropout option was added to this helper. A strict
        # numerical baseline bridge below verifies its unchanged default path.
        if name != 'eval/official_training_fit.py' and sources[name] != old_hash:
            raise ValueError('An unchanged shared source differs from the reference')
    sources['eval/official_dropout_fit.py'] = sha256(__file__)
    receipt = {'status': 'incomplete', 'purpose': plan['purpose'], 'device': 'cpu',
               'plan': plan, 'plan_sha256': sha256(plan_path),
               'reference_receipt_sha256': plan['reference_receipt_sha256'],
               'base_plan': base, 'panel_sha256': panel_hash,
               'panel_counts': reference['panel_counts'], 'packed_manifest_sha256': reference['packed_manifest_sha256'],
               'source_sha256': sources, 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scikit-learn', 'threadpoolctl')},
               'validation_opened': False, 'test_opened': False, 'model_promoted': False,
               'baseline_bridge_exact': False, 'baseline_fits': [], 'zero_dropout_fits': [], 'paired_changes': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt, output)
    torch.set_num_threads(base['threads'])
    with threadpool_limits(limits=base['threads']):
        for old in reference['mlp_fits']:
            fit = fit_mlp(x, y, base['mlp'], old['seed'])
            if comparable(fit) != comparable(old):
                raise ValueError('Baseline bridge failed; do not interpret or launch the treatment')
            receipt['baseline_fits'].append(fit)
            atomic_json(receipt, output)
        receipt['baseline_bridge_exact'] = True
        atomic_json(receipt, output)
        for baseline in receipt['baseline_fits']:
            fit = fit_mlp(x, y, base['mlp'], baseline['seed'], dropout=0.)
            change = paired_changes(baseline, fit)
            receipt['zero_dropout_fits'].append(fit)
            receipt['paired_changes'].append(change)
            atomic_json(receipt, output)
    if (sha256(plan_path) != receipt['plan_sha256']
            or sha256(reference_path) != receipt['reference_receipt_sha256']
            or any(sha256(source_root/p) != digest for p, digest in sources.items())):
        raise ValueError('Source, reference or plan changed during diagnosis')
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
