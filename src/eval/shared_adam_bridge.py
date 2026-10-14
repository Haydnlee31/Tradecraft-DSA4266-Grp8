"""Bounded numerical audit, using only existing TRAIN rows and fresh models.

No quality metrics or held-out arrays are computed. This tests implementation
equivalence, not a claim that gradient synchronization solves classification.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.federated_class_safeguard import TrainOnlyData, state_digest
from src.federated.partition import check_exact_cover
from src.federated import shared_adam as bridge
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_losses import build_official_criterion
from src.models.research import atomic_json, atomic_save

ROOT = Path(__file__).parents[2]
PLAN = ROOT / 'reports/federated_shared_optimizer/bridge-plan.json'


def schedule(parts, step, seed, batch_size):
    """Separate sampling RNG from dropout RNG; step fully identifies restart."""
    rows, seeds = [], []
    for client, owned in enumerate(parts):
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, client, 0]))
        rows.append(rng.choice(owned, batch_size, replace=True))
        seeds.append(int(np.random.SeedSequence([seed, step, client, 1]).generate_state(1)[0]))
    return rows, seeds


def tensors(data, rows, dtype):
    x, y = data.arrays['train']
    return [(torch.from_numpy(np.array(x[r], copy=True)).to(dtype),
             torch.from_numpy(np.array(y[r], copy=True))) for r in rows]


def sampled_receipt(data, rows, seeds):
    joined = np.concatenate(rows).astype('<i8')
    return {'sample_sha256': hashlib.sha256(joined.tobytes()).hexdigest(),
            'dropout_seeds': seeds,
            'sample_class_counts': np.bincount(data.arrays['train'][1][joined], minlength=len(CLASSES)).tolist()}


def compared_step(models, optimizers, batches, counts, criterion, seeds, recipe, tolerance):
    """The server path has no raw-batch argument: it only sees messages/counts."""
    messages = [bridge.client_gradient(models['federated'], batch, criterion, seed)
                for batch, seed in zip(batches, seeds)]
    grads = {'federated': bridge.aggregate_gradients(messages, counts),
             'central': bridge.central_accumulated_gradient(models['central'], batches, counts, criterion, seeds)}
    result = {'raw_gradient': bridge.residual(grads['federated'], grads['central'], **tolerance)}
    clipped = {name: bridge.apply_gradient(model, optimizers[name], grads[name], recipe['clip_norm'])
               for name, model in models.items()}
    result['clipped_gradient'] = bridge.residual(clipped['federated'], clipped['central'], **tolerance)
    result['parameters'] = bridge.residual(tuple(models['federated'].parameters()),
                                            tuple(models['central'].parameters()), **tolerance)
    states = {name: bridge.optimizer_tensors(model, optimizers[name]) for name, model in models.items()}
    for key in ('exp_avg', 'exp_avg_sq', 'step'):
        result[key] = bridge.residual(states['federated'][key], states['central'][key], **tolerance)
    return result


def snapshot(models, optimizers):
    return {name: {'model': copy.deepcopy(model.state_dict()),
                   'optimizer': copy.deepcopy(optimizers[name].state_dict())} for name, model in models.items()}


def check_recovery(expected_models, expected_optimizers, models, optimizers):
    for name in models:
        bridge.require(state_digest(models[name]) == state_digest(expected_models[name]), 'Recovery weights differ')
        a = bridge.optimizer_tensors(models[name], optimizers[name])
        b = bridge.optimizer_tensors(expected_models[name], expected_optimizers[name])
        bridge.require(all(torch.equal(x, y) for key in a for x, y in zip(a[key], b[key])),
                       'Recovery moments/counters differ')


def run_case(data, parts, plan, case, output, receipt):
    dtype = getattr(torch, case['dtype'])
    counts = [len(p) for p in parts]
    tolerance = plan['tolerances'][case['dtype']]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan['seed'])
        initial = MLPClassifier(39, len(CLASSES), MLPConfig('shared-adam-audit', (64, 32),
                                case['dropout'], 'layer')).to(dtype).train()
    criterion = build_official_criterion('sqrt_weighted_ce', data.manifest['train_class_counts'],
                                          'fixed_train_mean').to(dtype)
    models = {name: copy.deepcopy(initial) for name in ('federated', 'central')}
    optimizers = {name: bridge.make_optimizer(model, plan['recipe']) for name, model in models.items()}
    receipt.update(case=case, initial_model_sha256=state_digest(initial), num_parameters=initial.num_parameters(),
                   parameter_bytes=initial.parameter_bytes(), tolerance=tolerance,
                   steps=[], recovery_steps=[], work={'backward_examples': 0, 'backward_batches': 0,
                   'optimizer_steps': 0, 'quality_evaluation_examples': 0})
    started = time.perf_counter()
    checkpoint = output / f"{case['dtype']}-dropout{case['dropout']}-midpoint.pt"
    def execute(step, ms, opts):
        rows, seeds = schedule(parts, step, plan['seed'], plan['batch_size'])
        batch = tensors(data, rows, dtype)
        checks = compared_step(ms, opts, batch, counts, criterion, seeds, plan['recipe'], tolerance)
        receipt['work']['backward_examples'] += 2*sum(len(r) for r in rows)
        receipt['work']['backward_batches'] += 2*len(parts)
        receipt['work']['optimizer_steps'] += 2
        return {'step': step, **sampled_receipt(data, rows, seeds), 'checks': checks}
    for step in range(1, plan['steps']+1):
        result = execute(step, models, optimizers)
        receipt['steps'].append(result)
        bridge.require(all(c['within_tolerance'] for c in result['checks'].values()),
                       f'Numerical equivalence failed at step {step}: {result["checks"]}')
        if step == plan['recovery_after']:
            atomic_save({'step': step, 'arms': snapshot(models, optimizers)}, checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    bridge.require(saved['step'] == plan['recovery_after'], 'Recovery boundary changed')
    recovered = {name: bridge.restore(initial, saved['arms'][name], plan['recipe']) for name in models}
    rms, ros = {n:r[0] for n,r in recovered.items()}, {n:r[1] for n,r in recovered.items()}
    for step in range(saved['step']+1, plan['steps']+1):
        result = execute(step, rms, ros)
        receipt['recovery_steps'].append(result)
        bridge.require(result == receipt['steps'][step-1], 'Recovered step diagnostic differs')
    check_recovery(models, optimizers, rms, ros)
    receipt.update(recovery_exact=True, final_model_sha256={n:state_digest(m) for n,m in models.items()},
                   central_federated_bit_exact=state_digest(models['federated']) == state_digest(models['central']),
                   numerical_equivalence_passed=True,
                   elapsed_seconds=time.perf_counter()-started,
                   communication={'estimated_per_path_bytes': initial.parameter_bytes()*2*len(parts)*plan['steps'],
                                  'sync_steps': plan['steps'], 'measured_network_traffic': False,
                                  'includes_recovery_audit': False, 'transport_overhead_included': False})


def run(data_root, partition_path, output, plan_path=PLAN):
    output, partition_path, plan_path = Path(output), Path(partition_path), Path(plan_path)
    bridge.require(not output.exists(), 'Output exists; preserve it')
    plan = json.loads(plan_path.read_text())
    bridge.require(plan['steps'] == 12 and plan['recovery_after'] == 6 and plan['clients'] == 20
                   and len(plan['cases']) == 4 and plan['batch_size'] == 512, 'Changed bounded scope')
    bridge.require(sha256(Path(data_root)/'manifest.json') == plan['packed_manifest_sha256']
                   and sha256(partition_path) == plan['partition_sha256'], 'Input identity mismatch')
    data = TrainOnlyData(data_root)
    with np.load(partition_path, allow_pickle=False) as z:
        bridge.require(set(z.files) == {f'client_{i}' for i in range(plan['clients'])}, 'Unexpected clients')
        parts = [z[f'client_{i}'].copy() for i in range(plan['clients'])]
    check_exact_cover(parts, len(data.arrays['train'][1]))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(bridge.__file__), ROOT/'src/models/architectures.py',
               ROOT/'src/models/official_losses.py', ROOT/'src/models/losses.py',
               ROOT/'src/eval/federated_class_safeguard.py', ROOT/'src/data/official_packed.py']
    receipt = {'status': 'running', 'plan_sha256': sha256(plan_path),
               'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
               'packed_manifest_sha256': plan['packed_manifest_sha256'],
               'partition_sha256': plan['partition_sha256'],
               'client_population_counts': [len(p) for p in parts],
               'client_class_counts': [np.bincount(data.arrays['train'][1][p], minlength=8).tolist() for p in parts],
               'class_order': CLASSES, 'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
               'torch': str(torch.__version__), 'numpy': np.__version__, 'device': 'cpu', 'threads': 2, 'interop_threads': 1},
               'scope': 'Numerical audit only; fresh short-lived models; no predictive performance inference',
               'validation_rows_opened': False, 'test_opened': False, 'test_evaluated': False,
               'quality_metrics_computed': False, 'cloud_used': False, 'historical_checkpoint_loaded': False,
               'cases': {}}
    atomic_json(receipt, output/'result.json')
    try:
        for case in plan['cases']:
            name = f"{case['dtype']}-dropout{case['dropout']}"
            receipt['cases'][name] = {}
            run_case(data, parts, plan, case, output, receipt['cases'][name])
            atomic_json(receipt, output/'result.json')
            print(f'PASS: {name}, all-step equivalence and exact serialized recovery', flush=True)
        work = {key: sum(c['work'][key] for c in receipt['cases'].values()) for key in plan['work_per_pass']}
        bridge.require(work == plan['work_per_pass'], 'Work budget mismatch')
        receipt.update(status='complete', total_work=work,
                       next='Numerical gate only passed; design separate from-scratch MIME development comparison')
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        atomic_json(receipt, output/'result.json')
        raise
    atomic_json(receipt, output/'result.json')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('data-root', 'partition-path', 'output'):
        p.add_argument('--'+arg, type=Path, required=True)
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
