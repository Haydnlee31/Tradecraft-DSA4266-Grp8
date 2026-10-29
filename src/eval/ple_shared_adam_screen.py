"""Exploratory screen: fixed PLE encoding with central and block-compressed shared Adam.

The encoding is fixed preprocessing outside the model, so the existing
gradient, codec and Adam machinery is reused unchanged. Historical modules are
imported, never edited. No confirmation, validation or test rows are opened.
"""
import argparse
import copy
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.shared_optimizer_data import ROOT, PLAN as BASE_PLAN, DevelopmentData
from src.eval import shared_optimizer_performance as training
from src.eval.federated_class_safeguard import state_digest
from src.eval.shared_optimizer_alarms import grouped_split, evaluate_scores
from src.federated import block_shared_adam as codec, compressed_shared_adam as original, shared_adam as shared
from src.federated.shared_adam import require
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_losses import build_official_criterion
from src.models.research import atomic_json

PLAN = ROOT/'reports/federated_ple/screen-plan.json'
ALARM = ROOT/'reports/federated_shared_optimizer/alarm-results.json'
BLOCK = ROOT/'reports/federated_shared_optimizer/block-compression-results.json'
BITS, BLOCK_SIZE = 4, 128


class PLE:
    """Cumulative clipped ramps over fit-quantile bins, vectorized in torch."""
    def __init__(self, x, bins):
        lo, width, column = [], [], []
        self.zero_columns = []
        for j in range(x.shape[1]):
            e = np.unique(np.quantile(np.asarray(x[:, j], dtype=np.float64), np.linspace(0, 1, bins+1)))
            if len(e) == 1:
                self.zero_columns.append(len(lo))
                lo.append(0.), width.append(1.), column.append(j)
            else:
                lo.extend(e[:-1]), width.extend(np.diff(e)), column.extend([j]*(len(e)-1))
        self.lo = torch.tensor(lo, dtype=torch.float64)
        self.width = torch.tensor(width, dtype=torch.float64)
        self.column = torch.tensor(column)
        self.dim = len(lo)

    def __call__(self, x):
        z = ((x[:, self.column].double()-self.lo)/self.width).clamp(0, 1).float()
        if self.zero_columns:
            z[:, self.zero_columns] = 0.
        return z


def initial_model(plan, dim):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan['seed'])
        a = plan['architecture']
        return MLPClassifier(dim, len(CLASSES), MLPConfig('ple-screen', tuple(a['hidden_dims']),
                                                          a['dropout'], a['normalization'])).train()


def train_central(data, base, plan, ple, criterion, settings):
    model = initial_model(plan, ple.dim)
    optimizer = shared.make_optimizer(model, {**base['adam'], 'lr': settings['learning_rate']})
    state = {'sample_chain_sha256': '0'*64}
    for unit in range(settings['steps']):
        rows = training.sampled_rows(data, base, unit, None, 0)
        training.record_rows(state, rows)
        x, y = data.batch(rows)
        shared.apply_gradient(model, optimizer, shared.client_gradient(model, (ple(x), y), criterion, base['seed']))
    return model, {'sample_chain_sha256': state['sample_chain_sha256'], 'payload_bytes': 0,
                   'backward_examples': settings['steps']*base['batch_size']}


def train_compressed(data, base, plan, ple, criterion, settings):
    c, k = base['clients'], base['mime']['local_steps']
    model = initial_model(plan, ple.dim)
    optimizer = shared.make_optimizer(model, {**base['adam'], 'lr': settings['learning_rate']})
    ps = shared.parameters(model)
    visible = copy.deepcopy(model)
    initial = original.encode(ps, 32, 0)
    with torch.no_grad():
        for v, q in zip(visible.parameters(), original.decode(initial, ps, 32, 0)):
            v.copy_(q)
    errors = [tuple(torch.zeros_like(p) for p in ps) for _ in range(c)]
    wire = {'initialization_bytes': c*len(initial), 'uplink_bytes': 0, 'downlink_bytes': 0}
    state = {'sample_chain_sha256': '0'*64}
    counts = [len(p) for p in data.parts]
    for unit in range(settings['synchronizations']):
        round_id, step = divmod(unit, k)
        messages = []
        for i in range(c):
            rows = training.sampled_rows(data, base, round_id, i, step)
            training.record_rows(state, rows)
            x, y = data.batch(rows)
            corrected = tuple(g+e for g, e in zip(shared.client_gradient(visible, (ple(x), y), criterion, base['seed']), errors[i]))
            packet = codec.encode(corrected, BITS, unit+1, BLOCK_SIZE)
            decoded = codec.decode(packet, ps, BITS, unit+1, BLOCK_SIZE)
            errors[i] = tuple(g-q for g, q in zip(corrected, decoded))
            messages.append(decoded)
            wire['uplink_bytes'] += len(packet)
        shared.apply_gradient(model, optimizer, shared.aggregate_gradients(messages, counts))
        packet = codec.encode(tuple(p.detach()-v.detach() for p, v in zip(ps, visible.parameters())),
                              BITS, unit+1, BLOCK_SIZE)
        with torch.no_grad():
            for v, q in zip(visible.parameters(), codec.decode(packet, ps, BITS, unit+1, BLOCK_SIZE)):
                v.add_(q)
        wire['downlink_bytes'] += c*len(packet)
        if (unit+1) % 360 == 0:
            print(f'  compressed step {unit+1}', flush=True)
    return visible, {'sample_chain_sha256': state['sample_chain_sha256'], 'wire': wire,
                     'payload_bytes': sum(wire.values()),
                     'backward_examples': settings['synchronizations']*c*base['batch_size']}


def logits_for(model, ple, x, batch_size):
    model.eval()
    out = np.empty((len(x), len(CLASSES)), dtype='<f4')
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out[start:start+batch_size] = model(ple(torch.from_numpy(np.array(x[start:start+batch_size], copy=True)))).numpy()
    return out


def primary(result, t):
    return next(p['audit_metrics'] for p in result['operating_points'] if p['calibration']['target_basis_points'] == t)


def run(data_root, output):
    plan, base = json.loads(PLAN.read_text()), json.loads(BASE_PLAN.read_text())
    output = Path(output)
    require(not output.exists(), 'Use a new output folder')
    torch.set_num_threads(plan['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = DevelopmentData(data_root, base)
    criterion = build_official_criterion('sqrt_weighted_ce', data.manifest['counts']['fit'], 'fixed_train_mean')
    ple = PLE(data.arrays['fit'][0], plan['bins'])
    x, y = data.arrays['development']
    y = np.asarray(y)
    calibration, _ = grouped_split(x, plan['alarm_split_domain'])
    output.mkdir(parents=True)
    np.save(output/'calibration_mask.npy', calibration, allow_pickle=False)
    require(sha256(output/'calibration_mask.npy') == plan['alarm_calibration_mask_sha256'], 'Calibration split differs')
    alarm = {r['arm']: r for r in json.loads(ALARM.read_text())['summary']['results']}
    block = json.loads(BLOCK.read_text())['summary']
    refs = {'central_adam': alarm['central_adam'], 'fedavg_adam': alarm['fedavg_adam'],
            'block128': next(r for r in block['results'] if r['arm'] == 'block128')}
    historical_chain = {'central_ple': json.loads((ROOT/'reports/federated_shared_optimizer/performance-results.json')
                                                   .read_text())['summary']['results'],
                        'compressed_ple': refs['block128']['sample_chain_sha256']}
    results = {}
    for arm, settings in plan['arms'].items():
        print(f'START {arm} (input dim {ple.dim})', flush=True)
        started = time.perf_counter()
        trainer = train_central if settings['method'] == 'central_adam' else train_compressed
        model, receipt = trainer(data, base, plan, ple, criterion, settings)
        seconds = time.perf_counter()-started
        if arm == 'central_ple':
            expected = next(r['sample_chain_sha256'] for r in historical_chain[arm]
                            if r['arm'] == 'central_adam' and r['learning_rate'] == settings['learning_rate'])
        else:
            expected = historical_chain[arm]
        require(receipt['sample_chain_sha256'] == expected, 'Sample chain differs from historical reference')
        result = {'arm': arm, 'settings': settings, 'input_dim': ple.dim, 'parameters': model.num_parameters(),
                  'training_seconds': seconds, 'final_model_sha256': state_digest(model), **receipt,
                  **evaluate_scores(logits_for(model, ple, x, plan['inference_batch_size']), y, calibration,
                                    plan['false_alert_targets_basis_points'])}
        results[arm] = result
        atomic_json(result, output/f'{arm}.json')
        m = primary(result, plan['primary_target_basis_points'])
        print(f'END {arm}: macro-F1={m["macro_f1"]:.4f} FAR={m["benign_false_alert_rate"]:.2%} '
              f'params={result["parameters"]} payload={receipt["payload_bytes"]/1e6:.1f} MB ({seconds:.0f}s)', flush=True)
    t, s = plan['primary_target_basis_points'], plan['screen']
    m, b = primary(results['compressed_ple'], t), primary(refs['block128'], t)
    q = {'macro_f1_gain_over_block128': m['macro_f1']-b['macro_f1'],
         'absolute_far_difference_from_block128': abs(m['benign_false_alert_rate']-b['benign_false_alert_rate']),
         'any_class_recall_drop_from_block128': max(b['per_class'][n]['recall']-m['per_class'][n]['recall'] for n in CLASSES),
         'central_ple_gain_over_central_adam': primary(results['central_ple'], t)['macro_f1']-primary(refs['central_adam'], t)['macro_f1'],
         'payload_ratio_to_block128': results['compressed_ple']['payload_bytes']/refs['block128']['work']['estimated_payload_bytes']}
    checks = {k: q[k.rsplit('_', 1)[0]] >= v if k.endswith('_min') else q[k.rsplit('_', 1)[0]] <= v for k, v in s.items()}
    summary = {'status': 'complete', 'plan': plan, 'plan_sha256': sha256(PLAN), 'results': results,
               'screen': {'passes_development_screen': all(checks.values()), 'checks': checks, **q},
               'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__},
               'confirmation_scored': False, 'original_validation_opened': False, 'test_opened': False}
    atomic_json(summary, output/'summary.json')
    print(json.dumps(summary['screen'], indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == '__main__':
    main()
