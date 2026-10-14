"""One predeclared, local, training-derived development screen of five updaters.

Equal backward-example work is NOT equal wall time or equal communication.
Full MIME pays for every anchor and both local gradients. Nothing here opens
the original validation/test splits or scores the reserved confirmation rows.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.shared_optimizer_data import PLAN, ROOT, DevelopmentData
from src.eval.federated_class_safeguard import state_digest
from src.federated import shared_adam as shared, mime_adam as mime
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_losses import build_official_criterion
from src.models.official_streaming import confusion_metrics
from src.models.research import atomic_json, atomic_save


def work_plan(plan, arm):
    n, b, c, k = plan['fit_rows'], plan['batch_size'], plan['clients'], plan['mime']['local_steps']
    if arm in ('central_adam', 'central_mime_base'):
        per_unit = b
    elif arm == 'shared_adam':
        per_unit = c*b
    elif arm == 'fedavg_adam':
        per_unit = c*k*b
    elif arm == 'mime':
        per_unit = n+2*c*k*b
    else:
        raise ValueError('Unregistered arm')
    units, remainder = divmod(plan['backward_examples_per_arm'], per_unit)
    shared.require(remainder == 0 and units > 0, 'Budget does not divide arm units exactly')
    if arm == 'mime':
        shared.require(units == plan['mime']['rounds'], 'MIME round budget differs')
    return units, per_unit


def initial_model(plan, features=39):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan['seed'])
        a = plan['architecture']
        return MLPClassifier(features, len(CLASSES), MLPConfig('optimizer-screen',
                tuple(a['hidden_dims']), a['dropout'], a['normalization'])).train()


def make_state(plan, arm, lr):
    model = initial_model(plan)
    recipe = {**plan['adam'], 'lr': lr}
    optimizer = shared.make_optimizer(model, recipe) if arm in ('central_adam', 'shared_adam') else None
    moments = mime.initialize(model, plan['mime']['initial_v']) if arm in ('mime', 'central_mime_base') else None
    return {'model': model, 'optimizer': optimizer, 'moments': moments, 'unit': 0,
            'work': {k: 0 for k in ('backward_examples', 'backward_batches', 'optimizer_steps',
                                    'anchor_examples', 'local_examples', 'synchronization_rounds',
                                    'logical_barriers', 'estimated_payload_bytes')},
            'sample_chain_sha256': '0'*64}


def sampled_rows(data, plan, unit, client=None, step=0):
    # Reconstructable RNG per unit/client, independent of training execution order.
    seed = [plan['seed'], unit, 999 if client is None else client, step]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    population = len(data.arrays['fit'][1]) if client is None else data.parts[client]
    return rng.choice(population, plan['batch_size'], replace=True)


def record_rows(state, rows):
    state['sample_chain_sha256'] = hashlib.sha256(bytes.fromhex(state['sample_chain_sha256'])
                                                 + rows.astype('<i8').tobytes()).hexdigest()


def advance(state, data, plan, arm, lr, criterion):
    """One restart boundary: minibatch, synchronized gradient step or local block."""
    model, u, work = state['model'], state['unit'], state['work']
    model.train()
    counts = [len(p) for p in data.parts]
    c, b, k = len(counts), plan['batch_size'], plan['mime']['local_steps']
    parameter_bytes = model.parameter_bytes()
    recipe = {**plan['mime'], 'lr': lr}

    def batch(client=None, step=0, round_id=u):
        rows = sampled_rows(data, plan, round_id, client, step)
        record_rows(state, rows)
        return data.batch(rows)

    if arm in ('central_adam', 'central_mime_base'):
        gradient = shared.client_gradient(model, batch(), criterion, plan['seed'])
        if arm == 'central_adam':
            shared.apply_gradient(model, state['optimizer'], gradient)
        else:
            # Published Table 5 base updater: read OLD moments, update weights,
            # then update moments from the same stochastic gradient once.
            direction = mime.directions(gradient, state['moments'], recipe)
            with torch.no_grad():
                for p, d in zip(model.parameters(), direction):
                    p.add_(d, alpha=-lr)
            state['moments'] = mime.update_state(gradient, state['moments'], recipe)
        work['backward_examples'] += b
        work['backward_batches'] += 1
        work['optimizer_steps'] += 1
    elif arm == 'shared_adam':
        # Every k synchronization steps use the same client samples as one
        # FedAvg/MIME block. They do NOT make the same number of optimizer steps.
        round_id, step = divmod(u, k)
        gradients = [shared.client_gradient(model, batch(i, step, round_id), criterion, plan['seed'])
                     for i in range(c)]
        shared.apply_gradient(model, state['optimizer'], shared.aggregate_gradients(gradients, counts))
        work['backward_examples'] += c*b
        work['backward_batches'] += c
        work['optimizer_steps'] += 1
        work['synchronization_rounds'] += 1
        work['logical_barriers'] += 1
        work['estimated_payload_bytes'] += 2*c*parameter_bytes
    elif arm == 'fedavg_adam':
        deltas = []
        for i in range(c):
            local = copy.deepcopy(model)
            # This deliberately reproduces the reset-client-Adam convention,
            # rather than silently retaining optimizer memory between rounds.
            optimizer = shared.make_optimizer(local, {**plan['adam'], 'lr': lr})
            for step in range(k):
                gradient = shared.client_gradient(local, batch(i, step), criterion, plan['seed'])
                shared.apply_gradient(local, optimizer, gradient)
            deltas.append(tuple(p.detach()-q.detach() for p, q in zip(local.parameters(), model.parameters())))
        with torch.no_grad():
            for p, delta in zip(model.parameters(), shared.aggregate_gradients(deltas, counts)):
                p.add_(delta)
        work['backward_examples'] += c*k*b
        work['backward_batches'] += c*k
        work['optimizer_steps'] += c*k
        work['local_examples'] += c*k*b
        work['synchronization_rounds'] += 1
        work['logical_barriers'] += 1
        work['estimated_payload_bytes'] += 2*c*parameter_bytes
    elif arm == 'mime':
        def full_batches(client):
            # Ownership was checked as an exact cover during data preparation.
            # Iterate each owned row exactly once, including its short last batch.
            owned = data.parts[client]
            for start in range(0, len(owned), plan['anchor_batch_size']):
                rows = owned[start:start+plan['anchor_batch_size']]
                record_rows(state, rows)
                yield (*data.batch(rows), plan['seed'])
        def local_batch(client, step):
            return (*batch(client, step), plan['seed'])
        model, moments, receipt = mime.full_round(model, state['moments'], counts, full_batches,
                                        local_batch, criterion, recipe, k)
        state.update(model=model, moments=moments)
        for key in ('backward_examples', 'backward_batches', 'anchor_examples', 'local_examples'):
            work[key] += receipt['work'][key]
        work['optimizer_steps'] += receipt['work']['local_updates']
        work['synchronization_rounds'] += 1
        # First aggregate anchor gradients, then aggregate updated local models.
        work['logical_barriers'] += 2
        work['estimated_payload_bytes'] += receipt['estimated_payload_bytes']
    else:
        raise ValueError('Unregistered arm')
    shared.require(all(bool(torch.isfinite(p).all()) for p in state['model'].parameters()), 'Nonfinite model')
    state['unit'] += 1
    shared.require(work['backward_examples'] == state['unit']*work_plan(plan, arm)[1], 'Work counter mismatch')


def snapshot(state):
    return {'model': copy.deepcopy(state['model'].state_dict()),
            'optimizer': copy.deepcopy(state['optimizer'].state_dict()) if state['optimizer'] else None,
            'moments': copy.deepcopy(state['moments']), 'unit': state['unit'],
            'work': copy.deepcopy(state['work']), 'sample_chain_sha256': state['sample_chain_sha256']}


def restore(saved, plan, arm, lr):
    state = make_state(plan, arm, lr)
    state['model'].load_state_dict(saved['model'])
    if state['optimizer']:
        state['optimizer'].load_state_dict(saved['optimizer'])
    for key in ('moments', 'unit', 'work', 'sample_chain_sha256'):
        state[key] = copy.deepcopy(saved[key])
    return state


def evaluate_development(model, data):
    x, y = data.arrays['development']
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(y), 8192):
            logits = model(torch.from_numpy(np.array(x[start:start+8192], copy=True)))
            shared.require(bool(torch.isfinite(logits).all()), 'Nonfinite development logits')
            pred = logits.argmax(1).numpy()
            cm += np.bincount(y[start:start+8192]*len(CLASSES)+pred,
                              minlength=len(CLASSES)**2).reshape(cm.shape)
    shared.require(int(cm.sum()) == len(y), 'Development row coverage differs')
    return confusion_metrics(cm)


def screen(selected, plan):
    result = {}
    central, fedavg = (selected[a]['development_metrics'] for a in ('central_adam', 'fedavg_adam'))
    thresholds = plan['screen']
    for arm in ('shared_adam', 'mime'):
        m = selected[arm]['development_metrics']
        quantities = {'macro_f1_gain_over_fedavg': m['macro_f1']-fedavg['macro_f1'],
                      'macro_f1_gap_to_central': central['macro_f1']-m['macro_f1'],
                      'far_above_central': m['benign_false_alert_rate']-central['benign_false_alert_rate'],
                      'far_above_fedavg': m['benign_false_alert_rate']-fedavg['benign_false_alert_rate'],
                      'largest_class_recall_drop_from_fedavg': max(fedavg['per_class'][n]['recall']-
                                           m['per_class'][n]['recall'] for n in CLASSES)}
        checks = {key: quantities[key.removesuffix('_min')] >= value if key.endswith('_min') else
                  quantities['largest_class_recall_drop_from_fedavg' if key == 'any_class_recall_drop_from_fedavg_max'
                             else key.removesuffix('_max')] <= value for key, value in thresholds.items()}
        result[arm] = {'passes_development_screen': all(checks.values()), 'checks': checks, **quantities}
    return result


def sources():
    names = ['src/eval/shared_optimizer_performance.py', 'src/data/shared_optimizer_data.py',
             'src/federated/shared_adam.py', 'src/federated/mime_adam.py',
             'src/models/architectures.py', 'src/models/official_losses.py', 'src/models/losses.py']
    return {n: sha256(ROOT/n) for n in names}


def run(data_root, output):
    plan = json.loads(PLAN.read_text())
    shared.require(plan['architecture']['dropout'] == 0., 'This screen requires deterministic full anchors')
    shared.require(plan['total_backward_examples'] == len(plan['arms'])*len(plan['learning_rates'])*
                   plan['backward_examples_per_arm'], 'Total budget differs')
    torch.set_num_threads(plan['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = DevelopmentData(data_root, plan)
    shared.require(len(data.arrays['fit'][1]) == plan['fit_rows'], 'Fit size differs')
    criterion = build_official_criterion('sqrt_weighted_ce', data.manifest['counts']['fit'], 'fixed_train_mean')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = {'plan_sha256': sha256(PLAN), 'data_manifest_sha256': sha256(Path(data_root)/'manifest.json'),
                'sources': sources()}
    contract_path = output/'contract.json'
    if contract_path.exists():
        shared.require(json.loads(contract_path.read_text()) == contract, 'Run source/data contract changed')
    else:
        shared.require(not list(output.iterdir()), 'Unknown existing output files')
        atomic_json(contract, contract_path)
    if (output/'summary.json').exists():
        raise ValueError('Completed summary exists; preserve it')
    results = []
    for arm in plan['arms']:
        for lr in plan['learning_rates']:
            destination = output/f'{arm}-lr{lr}'
            destination.mkdir(exist_ok=True)
            final = destination/'result.json'
            if final.exists():
                result = json.loads(final.read_text())
                shared.require(result['contract'] == contract, 'Saved result contract differs')
                results.append(result)
                continue
            state = make_state(plan, arm, lr)
            checkpoint = destination/'last.pt'
            elapsed = 0.
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                shared.require(saved['contract'] == contract, 'Recovery contract differs')
                state = restore(saved['state'], plan, arm, lr)
                elapsed = saved['training_seconds']
            units, _ = work_plan(plan, arm)
            interval = max(1, units//12)
            started = time.perf_counter()
            print(f'START {arm} lr={lr} at {state["unit"]}/{units}', flush=True)
            while state['unit'] < units:
                advance(state, data, plan, arm, lr, criterion)
                if state['unit'] % interval == 0 or state['unit'] == units:
                    atomic_save({'contract': contract, 'state': snapshot(state),
                                 'training_seconds': elapsed+time.perf_counter()-started}, checkpoint)
                    print(f'  {arm} lr={lr} work={state["work"]["backward_examples"]:,}/{plan["backward_examples_per_arm"]:,}', flush=True)
            training_seconds = elapsed+time.perf_counter()-started
            evaluation_start = time.perf_counter()
            metrics = evaluate_development(state['model'], data)
            result = {'status': 'complete', 'arm': arm, 'learning_rate': lr, 'seed': plan['seed'],
                      'units': units, 'contract': contract, 'work': state['work'],
                      'initial_model_sha256': state_digest(initial_model(plan)),
                      'final_model_sha256': state_digest(state['model']),
                      'sample_chain_sha256': state['sample_chain_sha256'],
                      'num_parameters': state['model'].num_parameters(),
                      'parameter_bytes': state['model'].parameter_bytes(),
                      'training_seconds': training_seconds,
                      'evaluation_seconds': time.perf_counter()-evaluation_start,
                      'development_metrics': metrics, 'confirmation_scored': False,
                      'original_validation_opened': False, 'test_opened': False, 'test_evaluated': False,
                      'network_measured': False}
            shared.require(result['work']['backward_examples'] == plan['backward_examples_per_arm'], 'Final work differs')
            atomic_json(result, final)
            results.append(result)
            print(f'END {arm} lr={lr} macro_f1={metrics["macro_f1"]:.6f} FAR={metrics["benign_false_alert_rate"]:.4%}', flush=True)
    shared.require(sources() == contract['sources'], 'Sources changed during run')
    selected = {arm: max((r for r in results if r['arm'] == arm),
                key=lambda r: (r['development_metrics']['macro_f1'], -r['learning_rate'])) for arm in plan['arms']}
    summary = {'status': 'complete', 'contract': contract, 'plan': plan, 'data': data.manifest,
               'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                           'numpy': np.__version__, 'platform': platform.platform(),
                           'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
               'results': results, 'selected_learning_rates': {a: r['learning_rate'] for a,r in selected.items()},
               'screen': screen(selected, plan), 'total_backward_examples': sum(r['work']['backward_examples'] for r in results),
               'confirmation_scored': False, 'original_validation_opened': False, 'test_opened': False,
               'test_evaluated': False, 'next': 'Review exploratory development endpoints before any confirmation or new method'}
    atomic_json(summary, output/'summary.json')
    print(json.dumps({'status': 'complete', 'screen': summary['screen'], 'total_backward_examples': summary['total_backward_examples']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == '__main__':
    main()
