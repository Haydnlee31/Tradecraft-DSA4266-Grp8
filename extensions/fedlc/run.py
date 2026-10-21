"""Opt-in CPU FedLC development screen. Run from the repository root.

Trains, for every seed in plan.json, each method twice: the unchanged published
control and the same run with FedLC. The original FedAvg and block128 shared-Adam
loops are called unchanged; FedLC enters only through the criterion and a
client-routing data view (see core.py).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from extensions.fedlc.core import (ClientRoutedData, FedLCCriterion, client_class_counts,
                                   require, validate_settings)
from extensions.fedlc.data import DevelopmentData, PARENT_PLAN, ROOT, sha256
from src.data.label_map import CLASSES
from src.eval import shared_optimizer_alarms as alarms
from src.eval import shared_optimizer_performance as training
from src.eval import shared_optimizer_replication as rep
from src.eval.federated_class_safeguard import state_digest
from src.federated import block_shared_adam as codec
from src.models.official_losses import build_official_criterion

HERE = Path(__file__).resolve().parent
DEFAULT_PLAN = HERE/'plan.json'
EVIDENCE = ROOT/'reports/federated_shared_optimizer'
ALARM_PLAN = EVIDENCE/'alarm-plan.json'
REPLICATION_PLAN = EVIDENCE/'replication-plan.json'
PUBLISHED = EVIDENCE/'block-replication-results.json'
BASE_LOSSES = ('sqrt_weighted_ce_fixed_train_mean', 'ce')
BLOCK_SETTINGS = {'bits': 4, 'feedback': True, 'block_size': 128}


def validate_plan(plan):
    require(plan['seeds'] and len(set(plan['seeds'])) == len(plan['seeds'])
            and all(type(s) is int and s >= 0 for s in plan['seeds']), 'Unique nonnegative integer seeds required')
    require(list(plan['methods']) == ['fedavg_adam', 'block128'], 'Methods must be fedavg_adam then block128')
    for settings in plan['methods'].values():
        lr = settings['learning_rate']
        require(isinstance(lr, float) and math.isfinite(lr) and lr > 0, 'Invalid learning rate')
    block = plan['methods']['block128']
    require({k: block[k] for k in BLOCK_SETTINGS} == BLOCK_SETTINGS, 'Only the frozen block128 codec is supported')
    require(plan['base_loss'] in BASE_LOSSES, f'base_loss must be one of {BASE_LOSSES}')
    validate_settings(plan['fedlc']['tau'], plan['fedlc']['count_floor'])
    require(plan['fedlc']['tau'] > 0, 'A positive tau is required for the treatment arm')
    targets = plan['false_alert_targets_basis_points']
    require(targets and all(type(t) is int and 0 < t < 10000 for t in targets)
            and plan['primary_target_basis_points'] in targets, 'Invalid alarm targets')
    a = plan['assessment']
    require(a['target_classes'] and set(a['target_classes']) < set(CLASSES), 'Unknown target classes')
    for key in ('min_mean_target_recall_gain', 'min_mean_target_f1_gain', 'max_macro_f1_drop_per_seed',
                'max_abs_far_difference_per_seed', 'max_other_class_recall_drop_per_seed'):
        require(isinstance(a[key], (int, float)) and math.isfinite(a[key]), f'Invalid {key}')
    require(type(plan['require_published_bridge']) is bool, 'require_published_bridge must be boolean')
    for key in ('threads', 'inference_batch_size', 'expected_backward_examples_per_run'):
        require(type(plan[key]) is int and plan[key] > 0, f'Positive integer {key} required')


def endpoints(plan):
    """Fixed order: within each seed and method, the control runs first."""
    return [(seed, method, fedlc) for seed in plan['seeds']
            for method in plan['methods'] for fedlc in (False, True)]


def arm_name(seed, method, fedlc):
    return f'seed{seed}-{method}-{"fedlc" if fedlc else "control"}'


def units_for(base, method):
    return training.work_plan(base, 'fedavg_adam' if method == 'fedavg_adam' else 'shared_adam')


def make_state(method, base, lr):
    if method == 'fedavg_adam':
        return training.make_state(base, 'fedavg_adam', lr)
    return codec.make_state(base, BLOCK_SETTINGS['bits'], BLOCK_SETTINGS['feedback'], lr,
                            BLOCK_SETTINGS['block_size'])


def advance(method, state, data, base, lr, criterion):
    if method == 'fedavg_adam':
        training.advance(state, data, base, 'fedavg_adam', lr, criterion)
    else:
        codec.advance(state, data, base, lr, criterion)


def snapshot(method, state):
    return training.snapshot(state) if method == 'fedavg_adam' else codec.snapshot(state)


def restore(method, saved, base, lr):
    if method == 'fedavg_adam':
        return training.restore(saved, base, 'fedavg_adam', lr)
    return codec.restore(saved, base, lr)


def evaluated_model(method, state):
    # block128 is scored on the delivered client model, never the hidden server target.
    return state['model'] if method == 'fedavg_adam' else state['visible']


def base_criterion(plan, data):
    if plan['base_loss'] == 'ce':
        return torch.nn.CrossEntropyLoss()
    return build_official_criterion('sqrt_weighted_ce', data.manifest['counts']['fit'], 'fixed_train_mean')


def atomic_write(path, value, tensor=False):
    tmp = path.with_name(path.name+'.tmp')
    if tensor:
        torch.save(value, tmp)
    else:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def plain(value):
    """JSON round trip so stored and recomputed metrics compare exactly."""
    return json.loads(json.dumps(value, allow_nan=False))


def source_hashes():
    paths = list((ROOT/'src').rglob('*.py')) + list(HERE.glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def published_reference(method, seed):
    summary = json.loads(PUBLISHED.read_text())['summary']
    if method == 'fedavg_adam':
        matched = [r for r in summary['references'] if r['arm'] == 'fedavg_adam' and r['seed'] == seed]
        key = 'final_model_sha256'
    else:
        matched = [r for r in summary['results'] if r['arm'] == 'block128' and r['seed'] == seed]
        key = 'delivered_model_sha256'
    return (matched[0], key) if len(matched) == 1 else (None, key)


def bridge(method, seed, state, digest, evaluated):
    """Exact comparison of a control arm with the published same-seed endpoint."""
    reference, key = published_reference(method, seed)
    if reference is None:
        return {'available': False, 'reason': f'No published {method} endpoint for seed {seed}'}
    checks = {'model_exact': digest == reference[key],
              'sample_chain_exact': state['sample_chain_sha256'] == reference['sample_chain_sha256'],
              'work_exact': plain(state['work']) == reference['work'],
              'default_metrics_exact': plain(evaluated['full_development_default_metrics'])
                                       == reference['full_development_default_metrics'],
              'operating_points_exact': plain(evaluated['operating_points']) == reference['operating_points']}
    if method == 'block128':
        checks['wire_exact'] = plain(state['wire']) == reference['wire']
    return {'available': True, **checks, 'all_exact': all(checks.values())}


def train_endpoint(method, base, lr, data, criterion, destination, identity):
    checkpoint = destination/'last.pt'
    state, elapsed = make_state(method, base, lr), 0.
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(saved['identity'] == identity, 'Recovery identity differs')
        state = restore(method, saved['state'], base, lr)
        elapsed = saved['training_seconds_sim_host']
    units, per_unit = units_for(base, method)
    require(0 <= state['unit'] <= units and state['work']['backward_examples'] == state['unit']*per_unit,
            'Recovery work differs')
    interval = max(1, units//12)
    print(f'START {identity["arm"]}: unit {state["unit"]}/{units}', flush=True)
    while state['unit'] < units:
        started = time.perf_counter()
        advance(method, state, data, base, lr, criterion)
        elapsed += time.perf_counter()-started
        if state['unit'] % interval == 0 or state['unit'] == units:
            atomic_write(checkpoint, {'identity': identity, 'state': snapshot(method, state),
                                      'training_seconds_sim_host': elapsed}, tensor=True)
            print(f'  {identity["arm"]}: unit {state["unit"]}/{units}', flush=True)
    require(state['work']['backward_examples'] == base['backward_examples_per_arm'], 'Final work differs')
    return state, elapsed


def per_seed_comparison(control, treated, plan):
    """FedLC minus control at every alarm target, on the same audit rows."""
    rows = []
    for target in plan['false_alert_targets_basis_points']:
        c, t = [next(p['audit_metrics'] for p in r['evaluation']['operating_points']
                     if p['calibration']['target_basis_points'] == target) for r in (control, treated)]
        rows.append({'target_basis_points': target,
                     'macro_f1_delta': t['macro_f1']-c['macro_f1'],
                     'far_delta': t['benign_false_alert_rate']-c['benign_false_alert_rate'],
                     'per_class_recall_delta': {k: t['per_class'][k]['recall']-c['per_class'][k]['recall'] for k in CLASSES},
                     'per_class_precision_delta': {k: t['per_class'][k]['precision']-c['per_class'][k]['precision'] for k in CLASSES},
                     'per_class_f1_delta': {k: t['per_class'][k]['f1']-c['per_class'][k]['f1'] for k in CLASSES}})
    return rows


def assess(comparisons, plan):
    """Apply the predeclared decision rule at the primary target, per method."""
    a, primary = plan['assessment'], plan['primary_target_basis_points']
    targets = a['target_classes']
    others = [k for k in CLASSES if k not in targets]
    out = {}
    for method in plan['methods']:
        per_seed = []
        for seed in plan['seeds']:
            row = next(r for r in comparisons[method][str(seed)] if r['target_basis_points'] == primary)
            recall, f1 = row['per_class_recall_delta'], row['per_class_f1_delta']
            safeguards = {'macro_f1': row['macro_f1_delta'] >= -a['max_macro_f1_drop_per_seed'],
                          'far': abs(row['far_delta']) <= a['max_abs_far_difference_per_seed'],
                          'other_class_recall': all(recall[k] >= -a['max_other_class_recall_drop_per_seed'] for k in others)}
            per_seed.append({'seed': seed, 'macro_f1_delta': row['macro_f1_delta'], 'far_delta': row['far_delta'],
                             'mean_target_recall_delta': float(np.mean([recall[k] for k in targets])),
                             'mean_target_f1_delta': float(np.mean([f1[k] for k in targets])),
                             'safeguards': safeguards, 'safeguards_pass': all(safeguards.values())})
        recall_gain = float(np.mean([s['mean_target_recall_delta'] for s in per_seed]))
        f1_gain = float(np.mean([s['mean_target_f1_delta'] for s in per_seed]))
        out[method] = {'per_seed': per_seed, 'mean_target_recall_gain': recall_gain, 'mean_target_f1_gain': f1_gain,
                       'every_seed_passes_safeguards': all(s['safeguards_pass'] for s in per_seed),
                       'supports_fedlc': all(s['safeguards_pass'] for s in per_seed)
                       and recall_gain >= a['min_mean_target_recall_gain'] and f1_gain >= a['min_mean_target_f1_gain']}
    return out


def run(data_root, output, plan_path=DEFAULT_PLAN, parent_plan=PARENT_PLAN):
    plan_path, parent_plan = Path(plan_path).resolve(), Path(parent_plan).resolve()
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    parent = json.loads(parent_plan.read_text())
    require(parent['backward_examples_per_arm'] == plan['expected_backward_examples_per_run'],
            'Parent plan work budget differs from this plan')
    output = Path(output).resolve()
    runs = HERE/'runs'
    require(output != runs.resolve() and runs.resolve() in output.parents,
            'Output must be a subdirectory of extensions/fedlc/runs/')
    require(not runs.is_symlink(), 'Output cannot escape the extension')
    data_root = Path(data_root).resolve()
    require(output != data_root and data_root not in output.parents and output not in data_root.parents,
            'Output and input data must be separate')
    torch.set_num_threads(plan['threads'])
    torch.use_deterministic_algorithms(True)
    data = DevelopmentData(data_root, parent, parent_plan)
    alarm_plan = json.loads(ALARM_PLAN.read_text())
    published_data = json.loads(REPLICATION_PLAN.read_text())
    manifest_sha256 = sha256(data.root/'manifest.json')
    real = parent_plan == PARENT_PLAN.resolve()
    if real:
        require(manifest_sha256 == published_data['data_manifest_sha256'],
                'Prepared data differs from the published shared-optimizer bundle')
    contract = {'plan': plan, 'plan_sha256': sha256(plan_path), 'parent_plan_sha256': sha256(parent_plan),
                'alarm_plan_sha256': sha256(ALARM_PLAN), 'published_evidence_sha256': sha256(PUBLISHED),
                'data_manifest_sha256': manifest_sha256, 'files': data.file_hashes, 'real_study': real,
                'sources': source_hashes(), 'runtime': {'python': platform.python_version(),
                    'torch': str(torch.__version__), 'numpy': str(np.__version__), 'platform': platform.platform()}}
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive process lock, in addition to atomic writes. Remove a stale lock
    # manually only after verifying that its recorded process is no longer alive.
    lock = output/'RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        return run_locked(data, output, plan, parent, alarm_plan, published_data, contract)
    finally:
        lock.unlink()


def run_locked(data, output, plan, parent, alarm_plan, published_data, contract):
    contract_path = output/'contract.json'
    if contract_path.exists():
        require(json.loads(contract_path.read_text()) == contract, 'Run source/data/plan/runtime contract changed')
    else:
        require(set(p.name for p in output.iterdir()) <= {'RUNNING.lock'}, 'Unknown existing output files')
        atomic_write(contract_path, contract)
    require(not (output/'summary.json').exists(), 'Completed run exists; preserve it')

    x, y = data.arrays['development']
    calibration, split = alarms.grouped_split(x, alarm_plan['split_domain'])
    for mask in (calibration, ~calibration):
        require(bool((np.bincount(y[mask], minlength=len(CLASSES)) > 0).all()), 'Split omits a class')
    mask_path = output/'calibration_mask.npy'
    np.save(mask_path, calibration, allow_pickle=False)
    split['calibration_mask_sha256'] = sha256(mask_path)
    if contract['real_study']:
        require(split['calibration_mask_sha256'] == published_data['calibration_mask_sha256'],
                'Calibration/audit split differs from the published one')

    criterion = base_criterion(plan, data)
    counts = client_class_counts(data.arrays['fit'][1], data.parts)
    tau, floor = plan['fedlc']['tau'], plan['fedlc']['count_floor']
    results = {}
    for seed, method, fedlc in endpoints(plan):
        name = arm_name(seed, method, fedlc)
        destination = output/name
        destination.mkdir(exist_ok=True)
        final = destination/'result.json'
        if final.exists():
            result = json.loads(final.read_text())
            require(result['contract'] == contract and result['arm'] == name, 'Result contract differs')
            require(sha256(destination/'last.pt') == result['checkpoint_sha256'], 'Completed checkpoint changed')
            results[name] = result
            continue
        base = {**parent, 'seed': seed}
        lr = plan['methods'][method]['learning_rate']
        identity = {'contract': contract, 'arm': name, 'seed': seed, 'method': method,
                    'fedlc': fedlc, 'learning_rate': lr}
        if fedlc:
            arm_criterion = FedLCCriterion(criterion, counts, tau, floor)
            arm_data = ClientRoutedData(data, arm_criterion)
        else:
            # The control is the published code path: no wrapper of any kind.
            arm_criterion, arm_data = criterion, data
        state, elapsed = train_endpoint(method, base, lr, arm_data, arm_criterion, destination, identity)
        model = evaluated_model(method, state)
        digest = state_digest(model)
        logits = rep.infer(model, x, plan['inference_batch_size'])
        evaluated = alarms.evaluate_scores(logits, y, calibration, plan['false_alert_targets_basis_points'])
        result = {'status': 'complete', 'arm': name, 'seed': seed, 'method': method, 'fedlc': fedlc,
                  'tau': tau if fedlc else 0.0, 'count_floor': floor, 'base_loss': plan['base_loss'],
                  'learning_rate': lr, 'contract': contract, 'units': state['unit'], 'work': state['work'],
                  'wire': state.get('wire'), 'training_seconds_sim_host': elapsed,
                  'sample_chain_sha256': state['sample_chain_sha256'],
                  'initial_model_sha256': state_digest(training.initial_model(base)),
                  'model_sha256': digest, 'logits_sha256': hashlib.sha256(logits.tobytes()).hexdigest(),
                  'checkpoint_sha256': sha256(destination/'last.pt'), 'evaluation': plain(evaluated),
                  'client_margins': [m.tolist() for m in arm_criterion.margins] if fedlc else None,
                  'num_parameters': model.num_parameters(), 'parameter_bytes': model.parameter_bytes(),
                  'test_opened': False, 'confirmation_scored': False, 'network_measured': False}
        if not fedlc and contract['real_study'] and plan['base_loss'] == 'sqrt_weighted_ce_fixed_train_mean':
            result['published_bridge'] = bridge(method, seed, state, digest, evaluated)
            if not result['published_bridge'].get('all_exact', False):
                message = f'{name} does not exactly reproduce the published endpoint: {result["published_bridge"]}'
                require(not plan['require_published_bridge'], message)
                print(f'WARNING {message}', flush=True)
        primary = next(p['audit_metrics'] for p in evaluated['operating_points']
                       if p['calibration']['target_basis_points'] == plan['primary_target_basis_points'])
        print(f'END {name}: 5% macro-F1={primary["macro_f1"]:.6f} actual FAR={primary["benign_false_alert_rate"]:.4%} '
              f'Web={primary["per_class"]["Web-based"]["recall"]:.2%} '
              f'BruteForce={primary["per_class"]["Brute Force"]["recall"]:.2%} '
              f'Spoofing={primary["per_class"]["Spoofing"]["recall"]:.2%}', flush=True)
        atomic_write(final, result)
        results[name] = result

    require(source_hashes() == contract['sources'], 'Sources changed during run')
    comparisons = {method: {} for method in plan['methods']}
    for seed in plan['seeds']:
        for method in plan['methods']:
            control, treated = (results[arm_name(seed, method, f)] for f in (False, True))
            # Routing must not change sampling or initialization: arms are truly paired.
            require(control['sample_chain_sha256'] == treated['sample_chain_sha256']
                    and control['initial_model_sha256'] == treated['initial_model_sha256'],
                    f'Unpaired arms for seed {seed} {method}')
            comparisons[method][str(seed)] = per_seed_comparison(control, treated, plan)
    bridges = [r['published_bridge'] for r in results.values() if 'published_bridge' in r]
    summary = {'status': 'complete', 'scope': 'exploratory development only', 'contract': contract,
               'split': plain(split), 'results': [results[arm_name(*e)] for e in endpoints(plan)],
               'fedlc_minus_control': comparisons, 'assessment': assess(comparisons, plan),
               'published_bridge_checked': len(bridges),
               'published_bridge_all_exact': bool(bridges) and all(b.get('all_exact', False) for b in bridges),
               'test_opened': False, 'confirmation_scored': False}
    atomic_write(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='Existing prepared optimizer fit/development directory')
    parser.add_argument('--output', type=Path, required=True, help='New subdirectory of extensions/fedlc/runs/')
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    try:
        # Match the published runs: two intra-op threads and one inter-op thread.
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    run(args.data, args.output, args.plan)


if __name__ == '__main__':
    main()
