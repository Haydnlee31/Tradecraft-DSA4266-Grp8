"""Bounded duration-only control for both pre-existing dropout conditions.

Recompute from the original initialization to 300 epochs. Each fit must match
its archived 100-epoch prefix BEFORE continuing, without resetting Adam. Final
training metrics, not the best observed epoch, are the comparison endpoint.
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
from src.eval.official_dropout_fit import check_pair
from src.eval.official_training_fit import load_training_panel, fit_mlp, validate_plan
from src.models.research import atomic_json

CONDITIONS = (('baseline_fits', .2), ('zero_dropout_fits', 0.))


def check_protocol(plan, reference):
    """Allow a longer budget, not a changed recipe or a newly selected seed."""
    if (plan['protocol'] != 'official39-duration-only-training-fit-v1'
            or plan['validation_opened'] is not False or plan['test_opened'] is not False
            or plan['require_exact_prefix'] is not True or plan['dropout_values'] != [.2, 0.]
            or not 1 <= plan['reference_epochs'] < plan['epochs'] <= 300
            or not plan['additional_report_epochs']
            or plan['additional_report_epochs'] != sorted(set(plan['additional_report_epochs']))
            or plan['additional_report_epochs'][0] <= plan['reference_epochs']
            or plan['additional_report_epochs'][-1] != plan['epochs']):
        raise ValueError('Plan exceeds the bounded duration-only control')
    if (reference['status'] != 'complete' or reference['baseline_bridge_exact'] is not True
            or reference['device'] != 'cpu' or reference['validation_opened'] is not False
            or reference['test_opened'] is not False or reference['model_promoted'] is not False
            or reference['plan_sha256'] != plan['reference_plan_sha256']
            or reference['panel_sha256'] != plan['panel_sha256']
            or reference['source_sha256']['eval/official_training_fit.py'] != plan['previous_fitter_sha256']):
        raise ValueError('Reference is not the approved completed dropout diagnostic')
    validate_plan(reference['base_plan'])
    base = reference['base_plan']['mlp']
    if base['epochs'] != plan['reference_epochs']:
        raise ValueError('Reference training budget mismatch')
    for key, dropout in CONDITIONS:
        if [f['seed'] for f in reference[key]] != base['seeds']:
            raise ValueError('Reference seed coverage mismatch')
        for fit in reference[key]:
            if (fit['model_config']['dropout'] != dropout
                    or [o['epoch'] for o in fit['observations']] != base['report_epochs']):
                raise ValueError('Reference condition or observation schedule mismatch')
    for pair in zip(reference['baseline_fits'], reference['zero_dropout_fits']):
        check_pair(*pair)


def duration_change(short, long):
    a, b = (f['observations'][-1]['training_panel_metrics'] for f in (short, long))
    return {'seed': short['seed'], 'dropout': short['model_config']['dropout'],
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
    base = reference['base_plan']
    if sha256(data_root/'manifest.json') != reference['packed_manifest_sha256']:
        raise ValueError('Training cohort differs from the reference')
    x, y, rows, manifest = load_training_panel(data_root, base['panel']['per_class'], base['panel']['seed'])
    panel_hash = hashlib.sha256(rows.astype('<i8').tobytes()+x.astype('<f4').tobytes()+y.astype('<i8').tobytes()).hexdigest()
    if panel_hash != reference['panel_sha256'] or manifest['train_rows'] != base['expected_training_rows']:
        raise ValueError('Training panel differs from the reference')
    source_root, sources = Path(__file__).parents[1], {}
    for name, old_hash in reference['source_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid source path')
        sources[name] = sha256(source_root/path)
        if name != 'eval/official_training_fit.py' and sources[name] != old_hash:
            raise ValueError('An unchanged shared source differs from the reference')
    sources['eval/official_duration_fit.py'] = sha256(__file__)
    settings = copy.deepcopy(base['mlp'])
    settings['epochs'] = plan['epochs']
    settings['report_epochs'] += plan['additional_report_epochs']
    receipt = {'status': 'incomplete', 'purpose': plan['purpose'], 'device': 'cpu',
               'plan': plan, 'plan_sha256': sha256(plan_path),
               'reference_receipt_sha256': plan['reference_receipt_sha256'],
               'base_plan': base, 'extended_mlp_settings': settings,
               'panel_sha256': panel_hash, 'panel_counts': reference['panel_counts'],
               'packed_manifest_sha256': reference['packed_manifest_sha256'],
               'source_sha256': sources, 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scikit-learn', 'threadpoolctl')},
               'validation_opened': False, 'test_opened': False, 'model_promoted': False,
               'all_prefixes_exact': False, 'baseline_fits': [], 'zero_dropout_fits': [], 'duration_changes': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt, output)
    torch.set_num_threads(base['threads'])
    with threadpool_limits(limits=base['threads']):
        for key, dropout in CONDITIONS:
            for old in reference[key]:
                fit = fit_mlp(x, y, settings, old['seed'], dropout=dropout, prefix_reference=old)
                if fit.get('prefix_bridge_exact') is not True:
                    raise ValueError('Missing exact prefix verification')
                receipt[key].append(fit)
                receipt['duration_changes'].append(duration_change(old, fit))
                atomic_json(receipt, output)
    for pair in zip(receipt['baseline_fits'], receipt['zero_dropout_fits']):
        check_pair(*pair)
    if (sha256(plan_path) != receipt['plan_sha256']
            or sha256(reference_path) != receipt['reference_receipt_sha256']
            or any(sha256(source_root/p) != digest for p, digest in sources.items())):
        raise ValueError('Plan, reference or source changed during diagnosis')
    receipt['all_prefixes_exact'] = True
    receipt['status'] = 'complete'
    atomic_json(receipt, output)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'reference-path', 'plan-path', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    diagnose(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
