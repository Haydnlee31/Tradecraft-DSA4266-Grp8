"""Component study for cyclic federated XGBoost: base learner, cyclic scheme, final check.

Training subcommands import numpy and xgboost only (torch+xgboost segfaults on
this macOS build); `score` runs in a separate torch process. Stages 1-2 select
on a hashed 20% holdout of fit rows; development rows are used only in stage 3.
No confirmation, validation or test rows are opened.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).parents[2]
PLAN = ROOT/'reports/federated_xgboost/component-plan.json'
EVIDENCE = ROOT/'reports/federated_shared_optimizer'
FEDAVG_PAYLOAD_BYTES = 88058880
CLASSES = 8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(data_root, plan):
    root = Path(data_root)
    require(file_sha256(root/'manifest.json') == plan['data_manifest_sha256'], 'Data manifest differs')
    manifest = json.loads((root/'manifest.json').read_text())
    for name in ('fit_x.npy', 'fit_y.npy', 'fit_row_ids.npy', 'development_x.npy', 'clients.npz'):
        require(file_sha256(root/name) == manifest['files'][name], f'Data checksum differs: {name}')
    a = {n: np.load(root/f'{n}.npy', allow_pickle=False) for n in ('fit_x', 'fit_y', 'fit_row_ids', 'development_x')}
    with np.load(root/'clients.npz', allow_pickle=False) as z:
        a['parts'] = [np.sort(z[f'client_{i}']) for i in range(len(manifest['fit_client_counts']))]
    require([len(p) for p in a['parts']] == manifest['fit_client_counts'], 'Client ownership differs')
    a['holdout'] = np.array([hashlib.sha256(b'tradecraft-xgb-tuning-v1:'+int(r).to_bytes(8, 'little', signed=True)).digest()[0] < 51
                             for r in a['fit_row_ids']])
    return a


def class_weights(counts, kind):
    """Per-class weight n_c^-p: none p=0, sqrt p=0.5, balanced p=1, or 'powerX' for p=X."""
    p = {'none': 0., 'sqrt': .5, 'balanced': 1.}.get(kind)
    p = float(kind.removeprefix('power')) if p is None else p
    return np.maximum(counts, 1)**-p


def weights(y, kind):
    w = class_weights(np.bincount(y, minlength=CLASSES).astype(np.float64), kind)[y]
    return (w/w.mean()).astype(np.float32)


def params_of(config, seed, eta=None):
    p = {k: v for k, v in config.items() if k not in ('rounds', 'weights')}
    p['seed'] = seed
    if eta is not None:
        p['eta'] = eta
    return p


def stage1_configs(plan, extra):
    base = plan['stage1_central']['base']
    configs = {name: {**base, **override} for name, override in plan['stage1_central']['variants'].items()}
    for name, override in extra.items():
        configs[name] = {**base, **override}
    return configs


def central(data_root, output, extra):
    import xgboost as xgb
    plan = json.loads(PLAN.read_text())
    a = load(data_root, plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = ~a['holdout']
    x, y = a['fit_x'][train], a['fit_y'][train]
    hold = xgb.DMatrix(a['fit_x'][a['holdout']])
    np.save(output/'holdout_y.npy', a['fit_y'][a['holdout']], allow_pickle=False)
    receipts = json.loads((output/'receipts.json').read_text()) if (output/'receipts.json').exists() else {}
    for name, config in stage1_configs(plan, extra).items():
        key = f'central-{name}'
        if key in receipts:
            continue
        started = time.perf_counter()
        booster = xgb.train(params_of(config, 0), xgb.DMatrix(x, y, weight=weights(y, config['weights'])), config['rounds'])
        np.save(output/f'{key}-margins.npy', booster.predict(hold, output_margin=True).astype('<f4'), allow_pickle=False)
        receipts[key] = {'config': config, 'model_bytes': len(booster.save_raw('ubj')), 'seconds': time.perf_counter()-started}
        print(f'{key}: {receipts[key]["model_bytes"]/1e6:.2f} MB, {receipts[key]["seconds"]:.0f}s', flush=True)
        json.dump(receipts, open(output/'receipts.json', 'w'), indent=2)


def cyclic_train(a, rows_of_client, config, scheme, seed, xgb, passes=10):
    """Returns booster, payload bytes. One boosting round per visit."""
    parts = [rows_of_client(i) for i in range(len(a['parts']))]
    sizes = np.array([len(p) for p in parts], dtype=np.float64)
    y_all = np.concatenate([a['fit_y'][p] for p in parts])
    counts = np.bincount(y_all, minlength=CLASSES).astype(np.float64)
    cw = class_weights(counts, config['weights'])
    cw = cw/cw[y_all].mean()
    subsample = 0.8 if scheme == 'subsample0.8' else 1.0
    data = [xgb.DMatrix(a['fit_x'][p], a['fit_y'][p], weight=cw[a['fit_y'][p]].astype(np.float32)) for p in parts]
    total = config['rounds']
    per_pass = total//passes
    require(per_pass*passes == total and per_pass % len(parts) == 0, 'Visit budget mismatch')
    booster, size, uplink = None, 0, 0
    for pass_id in range(passes):
        rng = np.random.default_rng(np.random.SeedSequence([seed, pass_id, 777]))
        if scheme == 'size_weighted':
            quota = np.floor(sizes/sizes.sum()*per_pass).astype(int)
            for i in np.argsort(-(sizes/sizes.sum()*per_pass-quota))[:per_pass-quota.sum()]:
                quota[i] += 1
            order = rng.permutation(np.repeat(np.arange(len(parts)), quota))
        else:
            order = np.concatenate([rng.permutation(len(parts)) for _ in range(per_pass//len(parts))])
        eta = config['eta']
        if scheme == 'eta_decay':
            eta = config['eta']*(0.3+0.7*0.5*(1+np.cos(np.pi*pass_id/(passes-1))))
        for client in order.tolist():
            booster = xgb.train({**params_of(config, seed, eta), 'subsample': subsample}, data[client], 1, xgb_model=booster)
            new = len(booster.save_raw('ubj'))
            uplink += new-size
            size = new
    require(booster.num_boosted_rounds() == total, 'Round count differs')
    return booster, uplink+(len(parts)-1)*size, size


def cyclic(data_root, output, config_name, extra):
    import xgboost as xgb
    plan = json.loads(PLAN.read_text())
    a = load(data_root, plan)
    config = stage1_configs(plan, extra)[config_name]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hold = xgb.DMatrix(a['fit_x'][a['holdout']])
    if not (output/'holdout_y.npy').exists():
        np.save(output/'holdout_y.npy', a['fit_y'][a['holdout']], allow_pickle=False)
    receipts = json.loads((output/'receipts.json').read_text()) if (output/'receipts.json').exists() else {}
    for scheme in plan['stage2_cyclic']['variants']:
        for seed in plan['stage2_cyclic']['seeds']:
            key = f'cyclic-{config_name}-{scheme}-seed{seed}'
            if key in receipts:
                continue
            started = time.perf_counter()
            booster, payload, size = cyclic_train(a, lambda i: a['parts'][i][~a['holdout'][a['parts'][i]]], config, scheme, seed, xgb)
            np.save(output/f'{key}-margins.npy', booster.predict(hold, output_margin=True).astype('<f4'), allow_pickle=False)
            receipts[key] = {'config': config, 'scheme': scheme, 'seed': seed, 'payload_bytes': payload,
                             'model_bytes': size, 'seconds': time.perf_counter()-started}
            print(f'{key}: payload {payload/1e6:.1f} MB, model {size/1e6:.2f} MB, {receipts[key]["seconds"]:.0f}s', flush=True)
            json.dump(receipts, open(output/'receipts.json', 'w'), indent=2)


def final(data_root, output, config_name, scheme, extra):
    import xgboost as xgb
    plan = json.loads(PLAN.read_text())
    a = load(data_root, plan)
    config = stage1_configs(plan, extra)[config_name]
    output = Path(output)
    require(not output.exists(), 'Use a new output folder')
    output.mkdir(parents=True)
    dev = xgb.DMatrix(a['development_x'])
    receipts = {}
    for seed in (7, 17, 27):
        booster, payload, size = cyclic_train(a, lambda i: a['parts'][i], config, scheme, seed, xgb)
        np.save(output/f'final-seed{seed}-margins.npy', booster.predict(dev, output_margin=True).astype('<f4'), allow_pickle=False)
        receipts[f'final-seed{seed}'] = {'config': config, 'scheme': scheme, 'seed': seed, 'payload_bytes': payload, 'model_bytes': size}
        print(f'final seed {seed}: payload {payload/1e6:.1f} MB, model {size/1e6:.2f} MB', flush=True)
    json.dump({'plan_sha256': file_sha256(PLAN), 'xgboost': xgb.__version__, 'receipts': receipts},
              open(output/'receipts.json', 'w'), indent=2)


def score(data_root, output):
    from src.data.label_map import CLASSES as NAMES
    from src.eval import shared_optimizer_alarms as alarms
    output = Path(output)
    receipts = json.loads((output/'receipts.json').read_text())
    if 'plan_sha256' not in receipts:  # stages 1-2: tuning holdout, threshold from its own benign rows
        y = np.load(output/'holdout_y.npy', allow_pickle=False)
        rows = {}
        for key, r in receipts.items():
            s, attack = alarms.margins(np.load(output/f'{key}-margins.npy', allow_pickle=False))
            threshold = alarms.calibrate(s[y == alarms.BENIGN], 500)['threshold']
            m = alarms.metrics(y, alarms.decisions(s, attack, threshold))
            rows[key] = {'macro_f1': m['macro_f1'], 'far': m['benign_false_alert_rate'],
                         'recall': {n: m['per_class'][n]['recall'] for n in NAMES}, **{k: v for k, v in r.items() if k != 'config'}}
        json.dump(rows, open(output/'scores.json', 'w'), indent=2)
        for key, q in sorted(rows.items()):
            print(f'{key:45s} F1={q["macro_f1"]:.4f} FAR={q["far"]:.2%} DoS={q["recall"]["DoS"]:.3f} '
                  f'Web={q["recall"]["Web-based"]:.3f} Brute={q["recall"]["Brute Force"]:.3f} '
                  f'model={q["model_bytes"]/1e6:.2f}MB' + (f' payload={q["payload_bytes"]/1e6:.1f}MB' if 'payload_bytes' in q else ''))
        return
    plan = json.loads(PLAN.read_text())
    require(receipts['plan_sha256'] == file_sha256(PLAN), 'Plan changed')
    import torch
    from src.data.shared_optimizer_data import PLAN as BASE_PLAN, DevelopmentData
    torch.set_num_threads(2)
    data = DevelopmentData(data_root, json.loads(BASE_PLAN.read_text()))
    x, y = data.arrays['development']
    y = np.asarray(y)
    calibration, _ = alarms.grouped_split(x, plan['alarm_split_domain'])
    np.save(output/'calibration_mask.npy', calibration, allow_pickle=False)
    require(file_sha256(output/'calibration_mask.npy') == plan['alarm_calibration_mask_sha256'], 'Calibration split differs')
    block = {r['seed']: r for r in json.loads((EVIDENCE/'block-replication-results.json').read_text())['summary']['results']}
    primary = lambda r: next(p['audit_metrics'] for p in r['operating_points'] if p['calibration']['target_basis_points'] == 500)
    per_seed = []
    for seed in (7, 17, 27):
        r = receipts['receipts'][f'final-seed{seed}']
        ev = alarms.evaluate_scores(np.load(output/f'final-seed{seed}-margins.npy', allow_pickle=False), y, calibration, [100, 500, 1000, 2000])
        m, b = primary(ev), primary(block[seed])
        q = {'seed': seed, 'macro_f1': m['macro_f1'], 'block128': b['macro_f1'], 'diff': m['macro_f1']-b['macro_f1'],
             'far': m['benign_false_alert_rate'], 'far_diff': abs(m['benign_false_alert_rate']-b['benign_false_alert_rate']),
             'payload_bytes': r['payload_bytes'], 'model_bytes': r['model_bytes'],
             'recall': {n: m['per_class'][n]['recall'] for n in NAMES},
             'flagged': {n: b['per_class'][n]['recall']-m['per_class'][n]['recall'] for n in NAMES
                         if b['per_class'][n]['recall']-m['per_class'][n]['recall'] > .10}}
        q['passes'] = q['diff'] >= -.01 and q['far_diff'] <= .005 and r['payload_bytes'] <= FEDAVG_PAYLOAD_BYTES
        per_seed.append({**q, 'operating_points': ev['operating_points']})
    mean_ok = np.mean([q['macro_f1'] for q in per_seed]) >= np.mean([q['block128'] for q in per_seed])
    summary = {'per_seed': per_seed, 'mean_at_least_block128': bool(mean_ok),
               'passes': bool(mean_ok and all(q['passes'] for q in per_seed)),
               'confirmation_scored': False, 'original_validation_opened': False, 'test_opened': False}
    json.dump(summary, open(output/'summary.json', 'w'), indent=2)
    for q in per_seed:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items() if k not in ('operating_points', 'recall')})
    print('passes', summary['passes'], 'mean_ok', mean_ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('central', 'cyclic', 'final', 'score'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', default='base')
    parser.add_argument('--scheme', default='v2')
    parser.add_argument('--extra', default='{}', help='JSON {name: overrides} for stage-1 combinations')
    args = parser.parse_args()
    extra = json.loads(args.extra)
    if args.action == 'central':
        central(args.data, args.output, extra)
    elif args.action == 'cyclic':
        cyclic(args.data, args.output, args.config, extra)
    elif args.action == 'final':
        final(args.data, args.output, args.config, args.scheme, extra)
    else:
        score(args.data, args.output)


if __name__ == '__main__':
    main()
