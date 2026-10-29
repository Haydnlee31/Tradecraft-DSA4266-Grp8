"""Seed replication and matched-capacity raw control for compressed shared Adam with PLE.

Reuses the screen's codec loop and PLE encoder unchanged. No confirmation,
validation or test rows are opened.
"""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.shared_optimizer_data import ROOT, PLAN as BASE_PLAN, DevelopmentData
from src.eval.federated_class_safeguard import state_digest
from src.eval.ple_shared_adam_screen import PLE, train_compressed, logits_for, primary
from src.eval.shared_optimizer_alarms import grouped_split, evaluate_scores
from src.federated.shared_adam import require
from src.models.official_losses import build_official_criterion
from src.models.research import atomic_json

PLAN = ROOT/'reports/federated_ple/followup-plan.json'
REPLICATION = ROOT/'reports/federated_shared_optimizer/block-replication-results.json'


class Raw:
    def __init__(self, features):
        self.dim = features

    def __call__(self, x):
        return x


def run(data_root, output):
    plan, base = json.loads(PLAN.read_text()), json.loads(BASE_PLAN.read_text())
    output = Path(output)
    require(not output.exists(), 'Use a new output folder')
    torch.set_num_threads(plan['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = DevelopmentData(data_root, base)
    criterion = build_official_criterion('sqrt_weighted_ce', data.manifest['counts']['fit'], 'fixed_train_mean')
    encoders = {'ple': PLE(data.arrays['fit'][0], plan['bins']), 'raw': Raw(data.arrays['fit'][0].shape[1])}
    x, y = data.arrays['development']
    y = np.asarray(y)
    calibration, _ = grouped_split(x, plan['alarm_split_domain'])
    output.mkdir(parents=True)
    np.save(output/'calibration_mask.npy', calibration, allow_pickle=False)
    require(sha256(output/'calibration_mask.npy') == plan['alarm_calibration_mask_sha256'], 'Calibration split differs')
    refs = {r['seed']: r for r in json.loads(REPLICATION.read_text())['summary']['results'] if r['arm'] == 'block128'}
    require(sorted(refs) == plan['seeds'], 'Missing historical block128 seed')
    t = plan['primary_target_basis_points']
    results = []
    for seed in plan['seeds']:
        for arm, spec in plan['arms'].items():
            encoder = encoders[spec['encoding']]
            arm_plan = {'seed': seed, 'architecture': {'hidden_dims': spec['hidden_dims'], **plan['architecture_fixed']}}
            settings = {'learning_rate': plan['learning_rate'], 'synchronizations': plan['synchronizations']}
            print(f'START {arm} seed={seed}', flush=True)
            started = time.perf_counter()
            model, receipt = train_compressed(data, {**base, 'seed': seed}, arm_plan, encoder, criterion, settings)
            require(receipt['sample_chain_sha256'] == refs[seed]['sample_chain_sha256'], 'Sample chain differs')
            result = {'arm': arm, 'seed': seed, 'input_dim': encoder.dim, 'parameters': model.num_parameters(),
                      'training_seconds': time.perf_counter()-started, 'final_model_sha256': state_digest(model),
                      **receipt, **evaluate_scores(logits_for(model, encoder, x, plan['inference_batch_size']), y,
                                                  calibration, plan['false_alert_targets_basis_points'])}
            atomic_json(result, output/f'{arm}-seed{seed}.json')
            results.append(result)
            m = primary(result, t)
            print(f'END {arm} seed={seed}: macro-F1={m["macro_f1"]:.4f} FAR={m["benign_false_alert_rate"]:.2%} '
                  f'params={result["parameters"]}', flush=True)
    per_seed = []
    for seed in plan['seeds']:
        m = primary(next(r for r in results if r['arm'] == 'compressed_ple' and r['seed'] == seed), t)
        b = primary(refs[seed], t)
        q = {'macro_f1_gain': m['macro_f1']-b['macro_f1'],
             'absolute_far_difference': abs(m['benign_false_alert_rate']-b['benign_false_alert_rate']),
             'largest_class_recall_drop': max(b['per_class'][n]['recall']-m['per_class'][n]['recall'] for n in CLASSES)}
        per_seed.append({'seed': seed, **q, 'passes': q['macro_f1_gain'] >= .01 and q['absolute_far_difference'] <= .005
                         and q['largest_class_recall_drop'] <= .05})
    mean = {arm: float(np.mean([primary(r, t)['macro_f1'] for r in results if r['arm'] == arm])) for arm in plan['arms']}
    mean['block128'] = float(np.mean([primary(refs[s], t)['macro_f1'] for s in plan['seeds']]))
    decision = {'replication_passes': all(p['passes'] for p in per_seed),
                'capacity_margin': mean['compressed_ple']-mean['compressed_raw_wide'],
                'capacity_passes': mean['compressed_ple']-mean['compressed_raw_wide'] >= .01}
    decision['keep_ple'] = decision['replication_passes'] and decision['capacity_passes']
    summary = {'status': 'complete', 'plan': plan, 'plan_sha256': sha256(PLAN), 'results': results,
               'per_seed': per_seed, 'mean_primary_macro_f1': mean, 'decision': decision,
               'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__},
               'confirmation_scored': False, 'original_validation_opened': False, 'test_opened': False}
    atomic_json(summary, output/'summary.json')
    print(json.dumps({'per_seed': per_seed, 'mean': mean, 'decision': decision}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == '__main__':
    main()
