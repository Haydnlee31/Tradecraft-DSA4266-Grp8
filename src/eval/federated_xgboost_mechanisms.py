"""Federated-learning mechanisms ported to cyclic XGBoost: FedLC and local-prior correction.

Training imports numpy and xgboost only; scoring reuses the component study's
torch-process `score`. Bagging (FedXgbBagging analog) was dropped per the plan's amendment. Selection uses the component study's fit-row holdout;
development rows are used once in `final`. No confirmation/validation/test rows.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.eval.federated_xgboost_components import (CLASSES, ROOT, class_weights, file_sha256, load, params_of,
                                                   require, stage1_configs)

PLAN = ROOT/'reports/federated_xgboost/mechanism-plan.json'
COMPONENT_PLAN = ROOT/'reports/federated_xgboost/component-plan.json'
PASSES = 10


def offsets(kind, client_y, global_prior):
    n = np.bincount(client_y, minlength=CLASSES).astype(np.float64)
    if kind == 'fedlc':
        return -1.0*np.maximum(n, 1)**-.25
    if kind == 'local_prior':
        return np.log((n+1)/(n+1).sum())-np.log(global_prior)
    return np.zeros(CLASSES)


def matrices(a, parts, config, kind, xgb, intercept=None):
    """Plain client matrices, or with base_margin = fitted intercept + client offset."""
    y_all = np.concatenate([a['fit_y'][p] for p in parts])
    prior = np.bincount(y_all, minlength=CLASSES)/len(y_all)
    cw = class_weights(np.bincount(y_all, minlength=CLASSES).astype(np.float64), config['weights'])
    cw = cw/cw[y_all].mean()
    out = []
    for p in parts:
        y = a['fit_y'][p]
        d = xgb.DMatrix(a['fit_x'][p], y, weight=cw[y].astype(np.float32))
        if intercept is not None:
            margin = (intercept.astype(np.float64)+offsets(kind, y, prior)).astype(np.float32)
            d.set_base_margin(np.tile(margin, (len(p), 1)).ravel())
        out.append(d)
    return out, prior


def eta_at(config, pass_id):
    return config['eta']*(0.3+0.7*0.5*(1+np.cos(np.pi*pass_id/(PASSES-1))))


def fitted_intercept(booster):
    model = json.loads(booster.save_raw('json'))
    return np.array(json.loads(model['learner']['learner_model_param']['base_score']), dtype=np.float32)


def cyclic(a, parts, config, kind, seed, xgb):
    plain, prior = matrices(a, parts, config, kind, xgb)
    data = plain
    booster, size, uplink = None, 0, 0
    for pass_id in range(PASSES):
        order = np.random.default_rng(np.random.SeedSequence([seed, pass_id, 777])).permutation(len(parts))
        for client in order.tolist():
            booster = xgb.train(params_of(config, seed, eta_at(config, pass_id)), data[client], 1, xgb_model=booster)
            if kind != 'none' and data is plain:
                data, _ = matrices(a, parts, config, kind, xgb, fitted_intercept(booster))
            new = len(booster.save_raw('ubj'))
            uplink += new-size
            size = new
    require(booster.num_boosted_rounds() == config['rounds'], 'Round count differs')
    return booster, prior, uplink+(len(parts)-1)*size, size


def predict(booster, x, prior, xgb):
    return booster.predict(xgb.DMatrix(x), output_margin=True).astype('<f4')


def train_arm(a, parts, config, arm, seed, xgb):
    return cyclic(a, parts, config, {'reference': 'none'}.get(arm, arm), seed, xgb)


def tune(data_root, output, arms, weights_kind):
    import xgboost as xgb
    a = load(data_root, json.loads(COMPONENT_PLAN.read_text()))
    config = {**stage1_configs(json.loads(COMPONENT_PLAN.read_text()), {})['base'], 'weights': weights_kind}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output/'holdout_y.npy', a['fit_y'][a['holdout']], allow_pickle=False)
    parts = [p[~a['holdout'][p]] for p in a['parts']]
    receipts = json.loads((output/'receipts.json').read_text()) if (output/'receipts.json').exists() else {}
    for arm in arms:
        for seed in (7, 17):
            key = f'{arm}-{weights_kind}-seed{seed}'
            if key in receipts:
                continue
            started = time.perf_counter()
            booster, prior, payload, size = train_arm(a, parts, config, arm, seed, xgb)
            np.save(output/f'{key}-margins.npy', predict(booster, a['fit_x'][a['holdout']], prior, xgb), allow_pickle=False)
            receipts[key] = {'arm': arm, 'weights': weights_kind, 'seed': seed, 'payload_bytes': payload, 'model_bytes': size,
                             'seconds': time.perf_counter()-started}
            print(f'{key}: payload {payload/1e6:.1f} MB, {receipts[key]["seconds"]:.0f}s', flush=True)
            json.dump(receipts, open(output/'receipts.json', 'w'), indent=2)


def final(data_root, output, arms_weights):
    import xgboost as xgb
    a = load(data_root, json.loads(COMPONENT_PLAN.read_text()))
    base = stage1_configs(json.loads(COMPONENT_PLAN.read_text()), {})['base']
    output = Path(output)
    require(not output.exists(), 'Use a new output folder')
    output.mkdir(parents=True)
    receipts = {}
    for arm, weights_kind in arms_weights:
        for seed in (7, 17, 27):
            booster, prior, payload, size = train_arm(a, a['parts'], {**base, 'weights': weights_kind}, arm, seed, xgb)
            key = f'{arm}-{weights_kind}-seed{seed}'
            np.save(output/f'{key}-margins.npy', predict(booster, a['development_x'], prior, xgb), allow_pickle=False)
            receipts[key] = {'arm': arm, 'weights': weights_kind, 'seed': seed, 'payload_bytes': payload, 'model_bytes': size}
            print(f'{key}: payload {payload/1e6:.1f} MB', flush=True)
    json.dump({'plan_sha256': file_sha256(PLAN), 'xgboost': xgb.__version__, 'receipts': receipts},
              open(output/'receipts.json', 'w'), indent=2)


def score_final(data_root, output):
    from src.data.label_map import CLASSES as NAMES
    from src.eval import shared_optimizer_alarms as alarms
    import torch
    from src.data.shared_optimizer_data import PLAN as BASE_PLAN, DevelopmentData
    plan = json.loads(PLAN.read_text())
    output = Path(output)
    receipts = json.loads((output/'receipts.json').read_text())
    require(receipts['plan_sha256'] == file_sha256(PLAN), 'Plan changed')
    torch.set_num_threads(2)
    data = DevelopmentData(data_root, json.loads(BASE_PLAN.read_text()))
    x, y = data.arrays['development']
    y = np.asarray(y)
    calibration, _ = alarms.grouped_split(x, plan['alarm_split_domain'])
    np.save(output/'calibration_mask.npy', calibration, allow_pickle=False)
    require(file_sha256(output/'calibration_mask.npy') == plan['alarm_calibration_mask_sha256'], 'Calibration split differs')
    block = {r['seed']: r for r in json.loads((ROOT/'reports/federated_shared_optimizer/block-replication-results.json')
                                               .read_text())['summary']['results']}
    primary = lambda r: next(p['audit_metrics'] for p in r['operating_points'] if p['calibration']['target_basis_points'] == 500)
    rows = {}
    for key, r in receipts['receipts'].items():
        ev = alarms.evaluate_scores(np.load(output/f'{key}-margins.npy', allow_pickle=False), y, calibration, [100, 500, 1000, 2000])
        m, b = primary(ev), primary(block[r['seed']])
        rows[key] = {**r, 'macro_f1': m['macro_f1'], 'far': m['benign_false_alert_rate'], 'block128': b['macro_f1'],
                     'block128_far': b['benign_false_alert_rate'], 'recall': {n: m['per_class'][n]['recall'] for n in NAMES},
                     'operating_points': ev['operating_points']}
    json.dump(rows, open(output/'scores.json', 'w'), indent=2)
    for key, q in rows.items():
        print(f'{key:32s} F1={q["macro_f1"]:.4f} block128={q["block128"]:.4f} FAR={q["far"]:.2%} DoS={q["recall"]["DoS"]:.3f} '
              f'Web={q["recall"]["Web-based"]:.3f} Brute={q["recall"]["Brute Force"]:.3f} payload={q["payload_bytes"]/1e6:.1f}MB')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('tune', 'final', 'score-final'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', default=['reference', 'fedlc', 'local_prior'])
    parser.add_argument('--weights', default='sqrt')
    parser.add_argument('--final-arms', nargs='+', default=[], help='arm:weights pairs')
    args = parser.parse_args()
    if args.action == 'tune':
        tune(args.data, args.output, args.arms, args.weights)
    elif args.action == 'final':
        final(args.data, args.output, [tuple(s.split(':')) for s in args.final_arms])
    else:
        score_final(args.data, args.output)


if __name__ == '__main__':
    main()
