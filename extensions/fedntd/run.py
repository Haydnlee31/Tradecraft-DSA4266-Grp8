"""Opt-in CPU FedNTD development screen. Run from the repository root."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from extensions.fedntd.core import round_update, validate_settings
from extensions.fedntd.data import DevelopmentData, PARENT_PLAN, ROOT, sampled_rows, sha256
from src.data.label_map import CLASSES
from src.federated.shared_adam import require
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_losses import build_official_criterion

HERE = Path(__file__).resolve().parent
DEFAULT_PLAN = HERE/'plan.json'


def validate_plan(plan):
    for key in ('clients', 'fit_rows', 'features', 'local_steps', 'batch_size',
                'backward_examples_per_arm', 'threads'):
        require(type(plan[key]) is int and plan[key] > 0, f'Positive integer {key} required')
    require(type(plan['seed']) is int and plan['seed'] >= 0, 'Nonnegative integer seed required')
    require(plan['betas'] and len(set(plan['betas'])) == len(plan['betas'])
            and plan['betas'][0] == 0, 'Unique betas beginning with the FedAvg zero control required')
    for beta in plan['betas']:
        validate_settings(beta, plan['temperature'])
    require(math.isfinite(plan['learning_rate']) and plan['learning_rate'] > 0, 'Invalid learning rate')
    a = plan['architecture']
    require(a['normalization'] == 'layer' and a['dropout'] == 0
            and a['hidden_dims'] == [64, 32], 'Frozen layer-normalized light architecture required')
    adam = plan['adam']
    require(len(adam['betas']) == 2 and all(math.isfinite(b) and 0 <= b < 1 for b in adam['betas'])
            and math.isfinite(adam['eps']) and adam['eps'] > 0
            and math.isfinite(adam['weight_decay']) and adam['weight_decay'] >= 0, 'Invalid Adam recipe')
    rounds, remainder = divmod(plan['backward_examples_per_arm'],
                              plan['clients']*plan['local_steps']*plan['batch_size'])
    require(rounds > 0 and remainder == 0, 'Work must divide full rounds exactly')
    return rounds


def initial_model(plan):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan['seed'])
        a = plan['architecture']
        return MLPClassifier(plan['features'], len(CLASSES), MLPConfig(
            'fedntd-extension', tuple(a['hidden_dims']), a['dropout'], a['normalization'])).train()


def advance(state, data, plan, criterion, beta):
    def batch(client, step):
        rows = sampled_rows(data, plan, state['round'], client, step)
        state['sample_chain_sha256'] = hashlib.sha256(
            bytes.fromhex(state['sample_chain_sha256']) + rows.astype('<i8').tobytes()).hexdigest()
        return data.batch(rows)
    model, work = round_update(state['model'], [len(p) for p in data.parts], batch, criterion,
                              {**plan['adam'], 'lr': plan['learning_rate']}, plan['local_steps'],
                              plan['seed'], beta, plan['temperature'])
    state['model'] = model
    state['round'] += 1
    for key, value in work.items():
        state['work'][key] = state['work'].get(key, 0) + value


def evaluate(model, data):
    """Eight-class argmax metrics; no alarm threshold fitting or holdout selector."""
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    x, y = data.arrays['development']
    model.eval()
    with torch.no_grad():
        for start in range(0, len(y), 8192):
            logits = model(torch.from_numpy(np.array(x[start:start+8192], copy=True)))
            require(bool(torch.isfinite(logits).all()), 'Nonfinite evaluation logits')
            predicted = logits.argmax(1).numpy()
            cm += np.bincount(y[start:start+8192]*len(CLASSES)+predicted,
                              minlength=len(CLASSES)**2).reshape(cm.shape)
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp, support, alerts = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        precision, recall = tp/alerts if alerts else 0., tp/support if support else 0.
        per_class[name] = {'precision': precision, 'recall': recall,
                           'f1': 2*tp/(support+alerts) if support+alerts else 0., 'support': support}
    benign = CLASSES.index('Benign')
    require(cm[benign].sum() > 0, 'Development cohort must contain benign rows')
    return {'macro_f1': float(np.mean([v['f1'] for v in per_class.values()])),
            'accuracy': float(np.trace(cm)/cm.sum()), 'per_class': per_class,
            'benign_false_alert_rate': float(1-cm[benign, benign]/cm[benign].sum()),
            'confusion_matrix': cm.tolist(), 'decision_rule': 'argmax'}


def atomic_write(path, value, tensor=False):
    tmp = path.with_name(path.name+'.tmp')
    if tensor:
        torch.save(value, tmp)
    else:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def source_hashes():
    # Includes the extension itself AND existing source helpers without changing
    # any original runner's all-src provenance contract.
    paths = list((ROOT/'src').rglob('*.py')) + list(HERE.glob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def run(data_root, output, plan_path=DEFAULT_PLAN):
    plan = json.loads(Path(plan_path).read_text())
    rounds = validate_plan(plan)
    output = Path(output).resolve()
    runs = HERE/'runs'
    require(output != runs.resolve() and runs.resolve() in output.parents,
            'Output must be a subdirectory of extensions/fedntd/runs/')
    require(not runs.is_symlink() and HERE in output.parents, 'Output cannot escape the extension')
    data_root = Path(data_root).resolve()
    require(output != data_root and data_root not in output.parents and output not in data_root.parents,
            'Output and input data must be separate')
    torch.set_num_threads(plan['threads'])
    torch.use_deterministic_algorithms(True)
    data = DevelopmentData(data_root, plan)
    criterion = build_official_criterion('sqrt_weighted_ce', data.manifest['counts']['fit'], 'fixed_train_mean')
    contract = {'plan': plan, 'plan_sha256': sha256(plan_path), 'parent_plan_sha256': sha256(PARENT_PLAN),
                'data_manifest_sha256': sha256(data.root/'manifest.json'), 'files': data.file_hashes,
                'sources': source_hashes(), 'runtime': {'python': platform.python_version(),
                    'torch': str(torch.__version__), 'numpy': str(np.__version__), 'platform': platform.platform()}}
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive process lock, in addition to atomic writes. Remove a stale lock
    # manually only after verifying that its recorded process is no longer alive.
    lock = output/'RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        return run_locked(data, output, plan, rounds, criterion, contract)
    finally:
        lock.unlink()


def run_locked(data, output, plan, rounds, criterion, contract):
    contract_path = output/'contract.json'
    if contract_path.exists():
        require(json.loads(contract_path.read_text()) == contract, 'Run source/data/plan/runtime contract changed')
    else:
        require(set(p.name for p in output.iterdir()) <= {'RUNNING.lock'}, 'Unknown existing output files')
        atomic_write(contract_path, contract)
    require(not (output/'summary.json').exists(), 'Completed run exists; preserve it')
    results = []
    for beta in plan['betas']:
        destination = output/f'beta-{beta:g}'
        destination.mkdir(exist_ok=True)
        checkpoint, final = destination/'last.pt', destination/'result.json'
        if final.exists():
            result = json.loads(final.read_text())
            require(result['contract'] == contract and result['beta'] == beta, 'Result contract differs')
            require(sha256(checkpoint) == result['checkpoint_sha256'], 'Completed checkpoint changed')
            results.append(result)
            continue
        state = {'model': initial_model(plan), 'round': 0, 'work': {}, 'sample_chain_sha256': '0'*64}
        elapsed = 0.
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            require(saved['contract'] == contract and saved['beta'] == beta, 'Recovery contract differs')
            state['model'].load_state_dict(saved['model'], strict=True)
            for key in ('round', 'work', 'sample_chain_sha256'):
                state[key] = copy.deepcopy(saved[key])
            require(0 <= state['round'] <= rounds and state['work']['backward_examples'] ==
                    state['round']*plan['clients']*plan['local_steps']*plan['batch_size'], 'Recovery work differs')
            elapsed = saved['training_seconds_sim_host']
        print(f'START beta={beta:g}: round {state["round"]}/{rounds}', flush=True)
        while state['round'] < rounds:
            started = time.perf_counter()
            advance(state, data, plan, criterion, beta)
            elapsed += time.perf_counter()-started
            atomic_write(checkpoint, {'contract': contract, 'beta': beta,
                         'model': state['model'].state_dict(), 'round': state['round'], 'work': state['work'],
                         'sample_chain_sha256': state['sample_chain_sha256'],
                         'training_seconds_sim_host': elapsed}, tensor=True)
            print(f'  beta={beta:g}: round {state["round"]}/{rounds}', flush=True)
        require(state['work']['backward_examples'] == plan['backward_examples_per_arm'], 'Final work differs')
        metrics = evaluate(state['model'], data)
        result = {'status': 'complete', 'beta': beta, 'temperature': plan['temperature'],
                  'contract': contract, 'rounds': state['round'], 'work': state['work'],
                  'training_seconds_sim_host': elapsed, 'sample_chain_sha256': state['sample_chain_sha256'],
                  'checkpoint_sha256': sha256(checkpoint), 'development_metrics': metrics,
                  'num_parameters': state['model'].num_parameters(),
                  'parameter_bytes': state['model'].parameter_bytes(),
                  'test_opened': False, 'confirmation_scored': False, 'network_measured': False}
        atomic_write(final, result)
        results.append(result)
    require(source_hashes() == contract['sources'], 'Sources changed during run')
    require(len({r['sample_chain_sha256'] for r in results}) == 1, 'Arms did not use identical samples')
    baseline = results[0]['development_metrics']
    comparisons = [{'beta': r['beta'],
                    'macro_f1_delta': r['development_metrics']['macro_f1']-baseline['macro_f1'],
                    'far_delta': r['development_metrics']['benign_false_alert_rate']-baseline['benign_false_alert_rate'],
                    'per_class_recall_delta': {c: r['development_metrics']['per_class'][c]['recall']-
                                              baseline['per_class'][c]['recall'] for c in CLASSES}}
                   for r in results[1:]]
    summary = {'status': 'complete', 'scope': 'exploratory development only', 'contract': contract,
               'results': results, 'comparisons_to_paired_fedavg': comparisons,
               'test_opened': False, 'confirmation_scored': False}
    atomic_write(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Existing prepared optimizer fit/development directory')
    parser.add_argument('--output', type=Path, required=True, help='New subdirectory of extensions/fedntd/runs/')
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    run(args.data, args.output, args.plan)


if __name__ == '__main__':
    main()
