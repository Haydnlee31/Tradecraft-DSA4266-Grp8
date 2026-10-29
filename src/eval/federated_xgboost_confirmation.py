"""Freeze and score FedLC cyclic XGBoost once on the reserved confirmation cohort.

`freeze` and `infer` import numpy and xgboost only; `score` runs in a separate
torch process with the shared-optimizer alarm helpers. Thresholds are the
development-calibrated ones; nothing is refitted on confirmation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.federated_xgboost_components import ROOT, file_sha256, load, require, stage1_configs
from src.eval.federated_xgboost_mechanisms import COMPONENT_PLAN, cyclic

PLAN = ROOT/'reports/federated_xgboost/confirmation-plan.json'
FREEZE = ROOT/'reports/federated_xgboost/confirmation-freeze.json'
MECHANISM_FINAL = ROOT/'outputs/fedxgb-mechanisms/final'
CONFIRMATION = ROOT/'outputs/shared-optimizer-confirmation-v1/data'
CONFIRMATION_MANIFEST_SHA256 = '1f54dffeb692fc4fa020f5d6abd3d2f51192a6f66cabde47df06a258ab31d536'
SEEDS = (7, 17, 27)


def freeze(data_root, models):
    import xgboost as xgb
    require(not FREEZE.exists(), 'Freeze receipt already exists')
    a = load(data_root, json.loads(COMPONENT_PLAN.read_text()))
    config = {**stage1_configs(json.loads(COMPONENT_PLAN.read_text()), {})['base'], 'weights': 'sqrt'}
    scores = json.loads((MECHANISM_FINAL/'scores.json').read_text())
    models = Path(models)
    models.mkdir(parents=True, exist_ok=True)
    receipt = {'plan_sha256': file_sha256(PLAN), 'xgboost': xgb.__version__, 'seeds': {}}
    for seed in SEEDS:
        booster, _, payload, size = cyclic(a, a['parts'], config, 'fedlc', seed, xgb)
        stored = MECHANISM_FINAL/f'fedlc-sqrt-seed{seed}-margins.npy'
        margins = booster.predict(xgb.DMatrix(a['development_x']), output_margin=True).astype('<f4')
        require(np.array_equal(margins, np.load(stored, allow_pickle=False)), f'Seed {seed} does not reproduce development margins')
        path = models/f'fedlc-seed{seed}.ubj'
        booster.save_model(path)
        thresholds = {str(p['calibration']['target_basis_points']): p['calibration']['threshold']
                      for p in scores[f'fedlc-sqrt-seed{seed}']['operating_points']}
        receipt['seeds'][str(seed)] = {'model_sha256': file_sha256(path), 'model_bytes': size, 'payload_bytes': payload,
                                       'development_margins_sha256': file_sha256(stored), 'thresholds': thresholds}
        print(f'seed {seed}: reproduced development margins exactly; model {file_sha256(path)[:12]}', flush=True)
    FREEZE.write_text(json.dumps(receipt, indent=2)+'\n')


def infer(models, output):
    import xgboost as xgb
    receipt = json.loads(FREEZE.read_text())
    require(receipt['plan_sha256'] == file_sha256(PLAN), 'Plan changed after freeze')
    require(file_sha256(CONFIRMATION/'manifest.json') == CONFIRMATION_MANIFEST_SHA256, 'Confirmation receipt differs')
    manifest = json.loads((CONFIRMATION/'manifest.json').read_text())
    require(file_sha256(CONFIRMATION/'confirmation_x.npy') == manifest['files']['confirmation_x.npy'], 'Confirmation x differs')
    x = np.load(CONFIRMATION/'confirmation_x.npy', allow_pickle=False)
    output = Path(output)
    require(not output.exists(), 'Confirmation inference already run; preserve it')
    output.mkdir(parents=True)
    for seed in SEEDS:
        path = Path(models)/f'fedlc-seed{seed}.ubj'
        require(file_sha256(path) == receipt['seeds'][str(seed)]['model_sha256'], f'Frozen model {seed} differs')
        booster = xgb.Booster(model_file=str(path))
        np.save(output/f'fedlc-seed{seed}-confirmation-margins.npy',
                booster.predict(xgb.DMatrix(x), output_margin=True).astype('<f4'), allow_pickle=False)
        print(f'seed {seed}: confirmation margins written', flush=True)


def score(output):
    from src.data.label_map import CLASSES
    from src.eval import shared_optimizer_alarms as alarms
    receipt = json.loads(FREEZE.read_text())
    require(receipt['plan_sha256'] == file_sha256(PLAN), 'Plan changed after freeze')
    manifest = json.loads((CONFIRMATION/'manifest.json').read_text())
    require(file_sha256(CONFIRMATION/'confirmation_y.npy') == manifest['files']['confirmation_y.npy'], 'Confirmation y differs')
    y = np.load(CONFIRMATION/'confirmation_y.npy', allow_pickle=False)
    reference = json.loads((ROOT/'reports/federated_shared_optimizer/confirmation-results.json').read_text())['summary']
    require(reference['data_receipt_sha256'] == CONFIRMATION_MANIFEST_SHA256, 'Reference used different confirmation data')
    ref = {(r['seed'], r['arm']): r for r in reference['results']}
    point = lambda r, t: next(p['confirmation_metrics'] for p in r['operating_points'] if p['calibration']['target_basis_points'] == t)
    output = Path(output)
    results, per_seed = {}, []
    for seed in SEEDS:
        frozen = receipt['seeds'][str(seed)]
        s, attack = alarms.margins(np.load(output/f'fedlc-seed{seed}-confirmation-margins.npy', allow_pickle=False))
        metrics = {t: alarms.metrics(y, alarms.decisions(s, attack, frozen['thresholds'][t])) for t in frozen['thresholds']}
        results[seed] = metrics
        m, b = metrics['500'], point(ref[(seed, 'block128')], 500)
        q = {'seed': seed, 'macro_f1': m['macro_f1'], 'far': m['benign_false_alert_rate'],
             'block128': b['macro_f1'], 'block128_far': b['benign_false_alert_rate'],
             'fedavg': point(ref[(seed, 'fedavg_adam')], 500)['macro_f1'],
             'central': point(ref[(seed, 'central_adam')], 500)['macro_f1'],
             'diff_vs_block128': m['macro_f1']-b['macro_f1'],
             'far_diff': abs(m['benign_false_alert_rate']-b['benign_false_alert_rate']),
             'payload_bytes': frozen['payload_bytes'], 'model_bytes': frozen['model_bytes'],
             'recall': {c: m['per_class'][c]['recall'] for c in CLASSES},
             'block128_recall': {c: b['per_class'][c]['recall'] for c in CLASSES}}
        q['passes'] = q['diff_vs_block128'] >= -.01 and q['far_diff'] <= .005 and q['payload_bytes'] <= 88058880
        per_seed.append(q)
    mean = float(np.mean([q['macro_f1'] for q in per_seed]))
    block_mean = float(np.mean([q['block128'] for q in per_seed]))
    summary = {'plan_sha256': file_sha256(PLAN), 'freeze_sha256': file_sha256(FREEZE), 'per_seed': per_seed,
               'mean_macro_f1': mean, 'block128_mean': block_mean,
               'all_targets_mean': {t: float(np.mean([results[s][t]['macro_f1'] for s in SEEDS])) for t in ('100', '500', '1000', '2000')},
               'block128_all_targets_mean': {t: float(np.mean([point(ref[(s, 'block128')], int(t))['macro_f1'] for s in SEEDS]))
                                             for t in ('100', '500', '1000', '2000')},
               'passes': all(q['passes'] for q in per_seed), 'superior_mean': mean > block_mean,
               'thresholds_refitted': False, 'confirmation_scored_once': True}
    (ROOT/'reports/federated_xgboost/confirmation-results.json').write_text(json.dumps(summary, indent=2)+'\n')
    for q in per_seed:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items() if 'recall' not in k})
    print(f'mean {mean:.4f} vs block128 {block_mean:.4f}; passes={summary["passes"]} superior_mean={summary["superior_mean"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'infer', 'score'))
    parser.add_argument('--data', type=Path)
    parser.add_argument('--models', type=Path, default=ROOT/'outputs/fedxgb-confirmation/models')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/fedxgb-confirmation/margins')
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.data, args.models)
    elif args.action == 'infer':
        infer(args.models, args.output)
    else:
        score(args.output)


if __name__ == '__main__':
    main()
