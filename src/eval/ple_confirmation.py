"""Freeze and score PLE16 + block128 compressed shared Adam once on the confirmation cohort.

Models are retrained with the committed PLE code and must reproduce their
recorded digests; thresholds are the development-calibrated ones.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.shared_optimizer_data import ROOT, PLAN as BASE_PLAN, DevelopmentData
from src.eval import shared_optimizer_alarms as alarms
from src.eval.federated_class_safeguard import state_digest
from src.eval.ple_shared_adam_screen import PLE, initial_model, train_compressed
from src.federated.shared_adam import require
from src.models.official_losses import build_official_criterion

PLAN = ROOT/'reports/federated_ple/confirmation-plan.json'
FREEZE = ROOT/'reports/federated_ple/confirmation-freeze.json'
RESULTS = ROOT/'reports/federated_ple/confirmation-results.json'
FOLLOWUP = ROOT/'outputs/ple-followup-v1'
CONFIRMATION = ROOT/'outputs/shared-optimizer-confirmation-v1/data'
CONFIRMATION_MANIFEST_SHA256 = '1f54dffeb692fc4fa020f5d6abd3d2f51192a6f66cabde47df06a258ab31d536'
FOLLOWUP_HASHES = {7: '0afc6c0135663be261c1b36c70d5edc565e7d74dab7117caedd5f69ed1ccda31',
                   17: '362d9c0e8fa59e23607a69e28d5b8ab77f1c82d73e515ba0fe5074c55130f387',
                   27: 'f034baac076af79aebd282479628d7c01e2f0767dc42211096ed5281ec2f9f1f'}
ARCHITECTURE = {'hidden_dims': [64, 32], 'normalization': 'layer', 'dropout': 0.0}
SEEDS = (7, 17, 27)


def setup():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def freeze(data_root, models):
    require(not FREEZE.exists(), 'Freeze receipt already exists')
    setup()
    base = json.loads(BASE_PLAN.read_text())
    data = DevelopmentData(data_root, base)
    criterion = build_official_criterion('sqrt_weighted_ce', data.manifest['counts']['fit'], 'fixed_train_mean')
    ple = PLE(data.arrays['fit'][0], 16)
    models = Path(models)
    models.mkdir(parents=True, exist_ok=True)
    receipt = {'plan_sha256': sha256(PLAN), 'ple_input_dim': ple.dim, 'seeds': {}}
    for seed in SEEDS:
        path = FOLLOWUP/f'compressed_ple-seed{seed}.json'
        require(sha256(path) == FOLLOWUP_HASHES[seed], f'Follow-up result {seed} differs')
        recorded = json.loads(path.read_text())
        model, receipt_run = train_compressed(data, {**base, 'seed': seed}, {'seed': seed, 'architecture': ARCHITECTURE},
                                              ple, criterion, {'learning_rate': .003, 'synchronizations': 2160})
        require(state_digest(model) == recorded['final_model_sha256'], f'Seed {seed} does not reproduce its model')
        out = models/f'ple-seed{seed}.pt'
        torch.save(model.state_dict(), out)
        receipt['seeds'][str(seed)] = {
            'model_digest': recorded['final_model_sha256'], 'state_dict_sha256': sha256(out),
            'payload_bytes': receipt_run['payload_bytes'], 'parameters': model.num_parameters(),
            'thresholds': {str(p['calibration']['target_basis_points']): p['calibration']['threshold']
                           for p in recorded['operating_points']}}
        print(f'seed {seed}: reproduced model digest exactly', flush=True)
    FREEZE.write_text(json.dumps(receipt, indent=2)+'\n')


def score(data_root, models):
    require(not RESULTS.exists(), 'Confirmation already scored; preserve it')
    setup()
    receipt = json.loads(FREEZE.read_text())
    require(receipt['plan_sha256'] == sha256(PLAN), 'Plan changed after freeze')
    require(sha256(CONFIRMATION/'manifest.json') == CONFIRMATION_MANIFEST_SHA256, 'Confirmation receipt differs')
    manifest = json.loads((CONFIRMATION/'manifest.json').read_text())
    for name in ('confirmation_x.npy', 'confirmation_y.npy'):
        require(sha256(CONFIRMATION/name) == manifest['files'][name], f'{name} differs')
    x = np.load(CONFIRMATION/'confirmation_x.npy', allow_pickle=False)
    y = np.load(CONFIRMATION/'confirmation_y.npy', allow_pickle=False)
    base = json.loads(BASE_PLAN.read_text())
    ple = PLE(DevelopmentData(data_root, base).arrays['fit'][0], 16)
    require(ple.dim == receipt['ple_input_dim'], 'PLE encoding differs')
    reference = json.loads((ROOT/'reports/federated_shared_optimizer/confirmation-results.json').read_text())['summary']
    require(reference['data_receipt_sha256'] == CONFIRMATION_MANIFEST_SHA256, 'Reference used different data')
    block = {r['seed']: r for r in reference['results'] if r['arm'] == 'block128'}
    point = lambda r, t: next(p['confirmation_metrics'] for p in r['operating_points'] if p['calibration']['target_basis_points'] == t)
    per_seed, all_targets = [], {}
    for seed in SEEDS:
        frozen = receipt['seeds'][str(seed)]
        path = Path(models)/f'ple-seed{seed}.pt'
        require(sha256(path) == frozen['state_dict_sha256'], f'Frozen model {seed} differs')
        model = initial_model({'seed': seed, 'architecture': ARCHITECTURE}, ple.dim)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        require(state_digest(model) == frozen['model_digest'], 'Model digest differs')
        model.eval()
        logits = np.empty((len(x), len(CLASSES)), dtype='<f4')
        with torch.no_grad():
            for start in range(0, len(x), 8192):
                logits[start:start+8192] = model(ple(torch.from_numpy(np.array(x[start:start+8192], copy=True)))).numpy()
        s, attack = alarms.margins(logits)
        metrics = {t: alarms.metrics(y, alarms.decisions(s, attack, frozen['thresholds'][t])) for t in frozen['thresholds']}
        all_targets[seed] = {t: m['macro_f1'] for t, m in metrics.items()}
        m, b = metrics['500'], point(block[seed], 500)
        q = {'seed': seed, 'macro_f1': m['macro_f1'], 'far': m['benign_false_alert_rate'], 'block128': b['macro_f1'],
             'block128_far': b['benign_false_alert_rate'], 'gain': m['macro_f1']-b['macro_f1'],
             'far_diff': abs(m['benign_false_alert_rate']-b['benign_false_alert_rate']),
             'largest_class_drop': max(b['per_class'][c]['recall']-m['per_class'][c]['recall'] for c in CLASSES),
             'payload_bytes': frozen['payload_bytes'], 'recall': {c: m['per_class'][c]['recall'] for c in CLASSES}}
        q['passes'] = q['gain'] >= .01 and q['far_diff'] <= .005 and q['largest_class_drop'] <= .05
        per_seed.append(q)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items() if k != 'recall'}, flush=True)
    summary = {'plan_sha256': sha256(PLAN), 'freeze_sha256': sha256(FREEZE), 'per_seed': per_seed,
               'mean_macro_f1': float(np.mean([q['macro_f1'] for q in per_seed])),
               'block128_mean': float(np.mean([q['block128'] for q in per_seed])),
               'all_targets_mean': {t: float(np.mean([all_targets[s][t] for s in SEEDS])) for t in ('100', '500', '1000', '2000')},
               'passes': all(q['passes'] for q in per_seed), 'thresholds_refitted': False, 'confirmation_scored_once': True}
    RESULTS.write_text(json.dumps(summary, indent=2)+'\n')
    print(f'mean {summary["mean_macro_f1"]:.4f} vs block128 {summary["block128_mean"]:.4f}; passes={summary["passes"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'score'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=ROOT/'outputs/ple-confirmation/models')
    args = parser.parse_args()
    (freeze if args.action == 'freeze' else score)(args.data, args.models)


if __name__ == '__main__':
    main()
