"""Reproduce bounded extensions without changing frozen historical evidence.

XGBoost and PyTorch run in different processes: never import both native runtimes
in a worker. Development fits thresholds; confirmation replay only checks the
six previously frozen candidate endpoints, never selects new settings.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / 'studies/bounded-extensions/plan.json'


def digest(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def new_folder(path):
    require(not path.exists(), f'Preserve existing output: {path}')
    path.mkdir(parents=True)


def sources():
    paths = [Path(__file__).resolve(), PLAN]
    paths += list((ROOT/'src').rglob('*.py'))
    paths += list((ROOT/'extensions/fedlc').glob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def tree_train(data_root, out):
    import numpy as np
    import xgboost as xgb
    from src.eval import federated_xgboost_components as components
    from src.eval import federated_xgboost_mechanisms as mechanisms
    plan = read(PLAN)
    new_folder(out)
    require(xgb.__version__ == plan['xgboost_version'], 'Use recorded XGBoost version')
    a = components.load(data_root, read(components.PLAN))
    config = components.stage1_configs(read(components.PLAN), {})['base']
    historical = read(ROOT/'reports/federated_xgboost/confirmation-freeze.json')
    records = []
    for seed in plan['seeds']:
        for arm in ('central_xgb', 'cyclic_xgb', 'cyclic_fedlc'):
            started = time.perf_counter()
            full_wire = {'uplink_bytes': 0, 'downlink_bytes': 0, 'visits': 0}
            if arm == 'central_xgb':
                # Match final tree capacity and the cyclic learning-rate schedule.
                # Central trees see all fit rows: explicitly NOT equal row-work.
                d = xgb.DMatrix(a['fit_x'], a['fit_y'], weight=components.weights(a['fit_y'], 'sqrt'))
                booster = None
                for step in range(200):
                    booster = xgb.train(components.params_of(config, seed, mechanisms.eta_at(config, step//20)),
                                        d, 1, xgb_model=booster)
                size = len(booster.save_raw('ubj'))
                cached = 0
            else:
                class TracedXGB:
                    DMatrix = xgb.DMatrix

                    @staticmethod
                    def train(*args, **kwargs):
                        previous = kwargs.get('xgb_model')
                        full_wire['downlink_bytes'] += len(previous.save_raw('ubj')) if previous is not None else 0
                        trained = xgb.train(*args, **kwargs)
                        full_wire['uplink_bytes'] += len(trained.save_raw('ubj'))
                        full_wire['visits'] += 1
                        return trained
                booster, _, cached, size = mechanisms.cyclic(a, a['parts'], config,
                    'fedlc' if arm == 'cyclic_fedlc' else 'none', seed, TracedXGB)
            name = f'{arm}-seed{seed}'
            model_path = out/f'{name}.ubj'
            booster.save_model(model_path)
            margins = booster.predict(xgb.DMatrix(a['development_x']), output_margin=True).astype('<f4')
            np.save(out/f'{name}-logits.npy', margins, allow_pickle=False)
            record = {'arm': arm, 'seed': seed, 'model_bytes': size, 'boosting_rounds': booster.num_boosted_rounds(),
                      'fit_row_visits_per_boosting_round_sum': len(a['fit_y'])*(200 if arm == 'central_xgb' else 10),
                      'historical_cached_tree_payload_estimate_bytes': cached, 'full_model_serialized_transfer': full_wire,
                      'model_sha256': digest(model_path), 'logits_file_sha256': digest(out/f'{name}-logits.npy'),
                      'training_and_inference_seconds': time.perf_counter()-started}
            if arm == 'cyclic_fedlc':
                ref = historical['seeds'][str(seed)]
                record['historical_model_exact'] = record['model_sha256'] == ref['model_sha256']
                record['historical_development_logits_exact'] = record['logits_file_sha256'] == ref['development_margins_sha256']
                require(record['historical_model_exact'] and record['historical_development_logits_exact'], 'XGBoost historical bridge differs')
            records.append(record)
            write(out/'training.json', {'status': 'running', 'results': records})
            print(f'END {name}: historical bridge verified where applicable', flush=True)
    write(out/'training.json', {'status': 'complete', 'results': records, 'xgboost': xgb.__version__,
                              'data_manifest_sha256': digest(data_root/'manifest.json'), 'sources': sources()})


def evaluation_setup(data_root):
    import numpy as np
    import torch
    from src.data.shared_optimizer_data import DevelopmentData, PLAN as BASE_PLAN
    from src.eval import shared_optimizer_alarms as alarms
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    base = read(BASE_PLAN)
    data = DevelopmentData(data_root, base)
    x, y = data.arrays['development']
    calibration, _ = alarms.grouped_split(x, read(ROOT/'reports/federated_ple/followup-plan.json')['alarm_split_domain'])
    return np, torch, base, data, x, y, calibration, alarms


def tree_score(data_root, out):
    np, torch, base, data, x, y, calibration, alarms = evaluation_setup(data_root)
    require(not (out/'summary.json').exists(), 'Preserve existing summary')
    saved = read(out/'training.json')
    require(saved['status'] == 'complete' and saved['sources'] == sources(), 'Training source contract differs')
    require(saved['data_manifest_sha256'] == digest(data_root/'manifest.json'), 'Data differs')
    results = []
    for row in saved['results']:
        path = out/f"{row['arm']}-seed{row['seed']}-logits.npy"
        require(digest(path) == row['logits_file_sha256'], 'Logits changed')
        results.append({**row, 'evaluation': alarms.evaluate_scores(np.load(path, allow_pickle=False), y, calibration, [100,500,1000,2000])})
    write(out/'summary.json', {**saved, 'results': results, 'scope': 'development audit', 'confirmation_scored': False})


def ple_train(data_root, out):
    np, torch, base, data, x, y, calibration, alarms = evaluation_setup(data_root)
    from src.eval.ple_shared_adam_screen import PLE, train_central, train_compressed, logits_for
    from src.eval.ple_shared_adam_followup import Raw
    from src.eval.federated_class_safeguard import state_digest
    from src.models.official_losses import build_official_criterion
    new_folder(out)
    historical = read(ROOT/'reports/federated_ple/confirmation-freeze.json')
    reference = read(ROOT/'reports/federated_shared_optimizer/block-replication-results.json')['summary']
    chains = {r['seed']: r['sample_chain_sha256'] for r in reference['results'] if r['arm']=='block128'}
    criterion = build_official_criterion('sqrt_weighted_ce', data.manifest['counts']['fit'], 'fixed_train_mean')
    ple = PLE(data.arrays['fit'][0], 16)
    records = []
    for seed in read(PLAN)['seeds']:
        for arm in ('central_ple', 'compressed_ple', 'compressed_raw_wide'):
            started = time.perf_counter()
            encoder = Raw(39) if arm == 'compressed_raw_wide' else ple
            architecture = {'hidden_dims': [214,32] if arm == 'compressed_raw_wide' else [64,32], 'dropout': 0., 'normalization': 'layer'}
            settings = {'learning_rate': .001 if arm == 'central_ple' else .003, 'steps':43200, 'synchronizations':2160}
            trainer = train_central if arm == 'central_ple' else train_compressed
            model, receipt = trainer(data, {**base,'seed':seed}, {'seed':seed,'architecture':architecture}, encoder, criterion, settings)
            require(receipt['backward_examples'] == 21600000, 'PLE budget differs')
            if arm != 'central_ple':
                require(receipt['sample_chain_sha256'] == chains[seed], 'PLE sample stream differs')
            model_digest = state_digest(model)
            if arm == 'compressed_ple':
                require(model_digest == historical['seeds'][str(seed)]['model_digest'], 'PLE historical model differs')
            path = out/f'{arm}-seed{seed}.pt'
            torch.save(model.state_dict(), path)
            logits = logits_for(model, encoder, x, 8192)
            row = {'arm':arm, 'seed':seed, 'parameters':model.num_parameters(), 'model_digest':model_digest,
                   'checkpoint_sha256':digest(path), **receipt,
                   'evaluation':alarms.evaluate_scores(logits, y, calibration, [100,500,1000,2000]),
                   'training_and_inference_seconds':time.perf_counter()-started}
            if arm == 'compressed_ple':
                thresholds = {str(p['calibration']['target_basis_points']):p['calibration']['threshold'] for p in row['evaluation']['operating_points']}
                require(thresholds == historical['seeds'][str(seed)]['thresholds'], 'PLE frozen thresholds differ')
                row['historical_model_and_thresholds_exact'] = True
            records.append(row)
            write(out/'progress.json', {'results':records})
            print(f'END {arm} seed={seed}: budget and sample stream verified', flush=True)
    write(out/'summary.json', {'status':'complete','results':records,'sources':sources(),
          'scope':'development audit','confirmation_scored':False, 'data_manifest_sha256':digest(data_root/'manifest.json'),
          'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__}})


def replay_tree_confirmation(data_root, out, confirmation):
    # This worker must stay NumPy/XGBoost-only.
    import numpy as np
    import xgboost as xgb
    frozen = read(ROOT/'reports/federated_xgboost/confirmation-freeze.json')
    validate_confirmation(confirmation)
    x = np.load(confirmation/'confirmation_x.npy', allow_pickle=False)
    for seed in read(PLAN)['seeds']:
        path = out/f'cyclic_fedlc-seed{seed}.ubj'
        require(digest(path)==frozen['seeds'][str(seed)]['model_sha256'], 'Unfrozen tree')
        target = out/f'confirmation-seed{seed}.npy'
        require(not target.exists(), 'Preserve confirmation replay')
        model = xgb.Booster(model_file=str(path))
        np.save(target, model.predict(xgb.DMatrix(x),output_margin=True).astype('<f4'),allow_pickle=False)


def validate_confirmation(root):
    require(digest(root/'manifest.json')=='1f54dffeb692fc4fa020f5d6abd3d2f51192a6f66cabde47df06a258ab31d536','Confirmation manifest differs')
    for name, expected in read(root/'manifest.json')['files'].items():
        require(digest(root/name)==expected, f'Confirmation file differs: {name}')


def replay_score(data_root, out, confirmation, family):
    np, torch, base, data, x, y, calibration, alarms = evaluation_setup(data_root)
    validate_confirmation(confirmation)
    frozen = read(ROOT/f'reports/federated_{family}/confirmation-freeze.json')
    reference = read(ROOT/f'reports/federated_{family}/confirmation-results.json')
    cx = np.load(confirmation/'confirmation_x.npy',allow_pickle=False)
    cy = np.load(confirmation/'confirmation_y.npy',allow_pickle=False)
    replay = []
    for seed in read(PLAN)['seeds']:
        f = frozen['seeds'][str(seed)]
        if family == 'ple':
            from src.eval.ple_shared_adam_screen import PLE, initial_model, logits_for
            from src.eval.federated_class_safeguard import state_digest
            encoder = PLE(data.arrays['fit'][0],16)
            model = initial_model({'seed':seed,'architecture':{'hidden_dims':[64,32],'dropout':0.,'normalization':'layer'}},encoder.dim)
            model.load_state_dict(torch.load(out/f'compressed_ple-seed{seed}.pt',map_location='cpu',weights_only=True))
            require(state_digest(model)==f['model_digest'],'Unfrozen PLE model')
            logits = logits_for(model,encoder,cx,8192)
        else:
            logits = np.load(out/f'confirmation-seed{seed}.npy',allow_pickle=False)
        score, attack = alarms.margins(logits)
        metrics = {t:alarms.metrics(cy,alarms.decisions(score,attack,threshold)) for t,threshold in f['thresholds'].items()}
        ref = next(r for r in reference['per_seed'] if r['seed']==seed)
        m = metrics['500']
        require(m['macro_f1']==ref['macro_f1'] and m['benign_false_alert_rate']==ref['far'] and
                {c:v['recall'] for c,v in m['per_class'].items()}==ref['recall'],'Historical confirmation differs')
        replay.append({'seed':seed,'historical_endpoint_exact':True,'metrics':metrics})
    target = out/'confirmation-replay.json'
    require(not target.exists(),'Preserve replay receipt')
    write(target,{'status':'complete','results':replay,'thresholds_refitted':False,'new_holdout':False,
                  'purpose':'reproduction of already exposed confirmation endpoints; no selection'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['trees','ple','tree-train','tree-score','tree-confirmation','replay-score'])
    parser.add_argument('--data',type=Path,default=ROOT/'outputs/shared-optimizer-development-data-v1')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--confirmation',type=Path,default=ROOT/'outputs/shared-optimizer-confirmation-v1/data')
    parser.add_argument('--family',choices=['xgboost','ple'])
    args=parser.parse_args()
    args.data,args.output,args.confirmation = args.data.resolve(),args.output.resolve(),args.confirmation.resolve()
    if args.action in ('trees','ple'):
        actions=['tree-train','tree-score','tree-confirmation','replay-score'] if args.action=='trees' else ['ple-worker','replay-score']
        for action in actions:
            if action=='ple-worker':
                # Separate worker invocation avoids global torch thread-state reuse.
                code='import sys; from pathlib import Path; sys.path.insert(0,"scripts"); from bounded_extensions import ple_train; ple_train(Path(sys.argv[1]),Path(sys.argv[2]))'
                subprocess.run([sys.executable,'-u','-c',code,str(args.data),str(args.output)],cwd=ROOT,check=True)
            else:
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),action,'--data',str(args.data),'--output',str(args.output),
                    '--confirmation',str(args.confirmation),'--family','xgboost' if args.action=='trees' else 'ple'],cwd=ROOT,check=True)
    elif args.action=='tree-train': tree_train(args.data,args.output)
    elif args.action=='tree-score': tree_score(args.data,args.output)
    elif args.action=='tree-confirmation': replay_tree_confirmation(args.data,args.output,args.confirmation)
    else: replay_score(args.data,args.output,args.confirmation,args.family)


if __name__=='__main__':
    main()
