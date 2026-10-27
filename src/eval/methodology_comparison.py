"""Matched, retrospective thesis supplement; preserve all historical experiments.

All six requested FL families already exist. This adapter adds a common protocol
for FedProx and from-scratch SCAFFOLD, with their appropriate optimizer controls.
Training/development and confirmation scoring are deliberately separate commands.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.eval import shared_optimizer_reproduce as old
from src.eval import shared_optimizer_portable as portable
from src.federated import scaffold
from src.federated import shared_adam as shared
from src.models.research import atomic_save, atomic_json

ROOT = old.ROOT
PLAN = ROOT / 'reports/methodology_comparison/plan.json'
EXTRA = {'central_sgd', 'fedavg_sgd', 'scaffold', 'fedprox001', 'fedprox01'}


def read_plan():
    plan = json.loads(PLAN.read_text())
    old.require(plan['seeds'] == [7, 17, 27] and len(plan['arms']) == 11, 'Incomplete comparison')
    old.require(plan['total_training_backward_examples'] == len(plan['arms'])*len(plan['seeds'])*
                plan['backward_examples_per_run'], 'Plan work differs')
    return plan


def preflight(root):
    # Historical integrity checks remain mandatory, including the original runtime.
    _, _, base, alarm, performance, parent = old.confirmation.read_contract(require_runtime=True)
    plan = read_plan()
    old.require(base['backward_examples_per_arm'] == plan['backward_examples_per_run'] and
                base['mime']['local_steps'] == plan['local_steps'], 'Protocol differs')
    data = old.rep.FitData(root, base, alarm['data_manifest_sha256'])
    for name in ('fit_row_ids.npy', 'confirmation_row_ids.npy', 'scaler.json'):
        old.require(old.sha256(Path(root)/name) == data.manifest['files'][name], 'Input provenance differs')
    x, y, calibration = old.replication.development(root, alarm, performance, parent['summary']['split'])
    return plan, base, data, alarm, x, y, calibration


def contract():
    return {'plan_sha256': old.sha256(PLAN), 'runner_sha256': old.sha256(__file__),
            'scaffold_sha256': old.sha256(scaffold.__file__),
            'historical_contract': old.confirmation.contract()}


def units(base, arm):
    alias = 'shared_adam' if arm == 'block128' else 'central_adam' if arm == 'central_sgd' else \
            'fedavg_adam' if arm in EXTRA else arm
    return old.training.work_plan(base, alias)


def make_state(base, arm, lr):
    if arm == 'block128':
        return old.codec.make_state(base, 4, True, lr, 128)
    state = old.training.make_state(base, 'fedavg_adam' if arm in EXTRA else arm, lr)
    if arm == 'scaffold':
        state['controls'] = scaffold.initialize_controls(state['model'], base['clients'])
    return state


def snapshot(state, arm):
    if arm == 'block128':
        return old.codec.snapshot(state)
    saved = old.training.snapshot(state)
    if arm == 'scaffold':
        saved['controls'] = copy.deepcopy(state['controls'])
    return saved


def restore(saved, base, arm, lr):
    if arm == 'block128':
        return old.codec.restore(saved, base, lr)
    state = old.training.restore(saved, base, 'fedavg_adam' if arm in EXTRA else arm, lr)
    if arm == 'scaffold':
        state['controls'] = copy.deepcopy(saved['controls'])
    return state


def advance(state, data, base, arm, settings, criterion):
    lr = settings['lr']
    if arm == 'block128':
        old.codec.advance(state, data, base, lr, criterion)
        return
    if arm not in EXTRA:
        old.training.advance(state, data, base, arm, lr, criterion)
        return
    model, u, w = state['model'], state['unit'], state['work']
    c, b, k = base['clients'], base['batch_size'], base['mime']['local_steps']
    sizes = list(map(len, data.parts))

    def batch(client=None, step=0):
        rows = old.training.sampled_rows(data, base, u, client, step)
        old.training.record_rows(state, rows)
        return data.batch(rows)

    if arm == 'central_sgd':
        gradients = shared.client_gradient(model, batch(), criterion, base['seed'])
        with torch.no_grad():
            for p, g in zip(model.parameters(), gradients):
                p.add_(g, alpha=-lr)
        w['backward_examples'] += b
        w['backward_batches'] += 1
        w['optimizer_steps'] += 1
    else:
        starts = scaffold.parameters(model)
        states, controls, deltas = [], [], []
        for client in range(c):
            if arm.startswith('fedprox'):
                local = copy.deepcopy(model)
                opt = shared.make_optimizer(local, {**base['adam'], 'lr': lr})
                for step in range(k):
                    grads = shared.client_gradient(local, batch(client, step), criterion, base['seed'])
                    # Analytic derivative of mu/2 * ||w-w_round_start||^2.
                    # A zero coefficient takes exactly the original FedAvg path.
                    mu = settings['mu']
                    if mu:
                        grads = tuple(g + mu*(p.detach()-q.detach()) for g, p, q in
                                      zip(grads, local.parameters(), model.parameters()))
                    shared.apply_gradient(local, opt, grads)
                deltas.append(tuple(p.detach()-q.detach() for p, q in zip(local.parameters(), model.parameters())))
            else:
                def batches():
                    for step in range(k):
                        xb, yb = batch(client, step)
                        yield xb.numpy(), yb.numpy()
                control = state.get('controls')
                ci, server = (control['clients'][client], control['server']) if control else (None, None)
                local, _, _ = scaffold.local_sgd(model, batches(), criterion, lr, 0., ci, server)
                states.append(scaffold.parameters(local))
                if control:
                    controls.append(scaffold.option_two(starts, states[-1], ci, server, k, lr))
        if arm.startswith('fedprox'):
            with torch.no_grad():
                for p, d in zip(model.parameters(), shared.aggregate_gradients(deltas, sizes)):
                    p.add_(d)
        else:
            model.load_state_dict(scaffold.aggregate(states, sizes))
            if arm == 'scaffold':
                state['controls'] = {'server': scaffold.aggregate(controls, sizes), 'clients': controls}
                scaffold.verify_controls(model, state['controls'], sizes)
        w['backward_examples'] += c*k*b
        w['backward_batches'] += c*k
        w['optimizer_steps'] += c*k
        w['local_examples'] += c*k*b
        w['synchronization_rounds'] += 1
        w['logical_barriers'] += 1
        w['estimated_payload_bytes'] += (4 if arm == 'scaffold' else 2)*c*model.parameter_bytes()
    state['unit'] += 1
    old.require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite model; stop, no retuning')


def check_work(state, base, arm, data):
    u = state['unit']
    c, b, k = base['clients'], base['batch_size'], base['mime']['local_steps']
    if arm == 'block128':
        old.block.check_work(state, base)
        return
    central = arm.startswith('central_')
    anchor = u*base['fit_rows'] if arm == 'mime' else 0
    local = u*c*k*b if not central and arm != 'shared_adam' else 0
    batches = u if central else u*c if arm == 'shared_adam' else u*c*k
    if arm == 'mime':
        batches = u*(sum((len(p)+base['anchor_batch_size']-1)//base['anchor_batch_size'] for p in data.parts)+2*c*k)
    expected = {'backward_examples': u*units(base, arm)[1], 'backward_batches': batches,
                'optimizer_steps': u if central or arm == 'shared_adam' else u*c*k,
                'anchor_examples': anchor, 'local_examples': local,
                'synchronization_rounds': 0 if central else u,
                'logical_barriers': 0 if central else u*(2 if arm == 'mime' else 1),
                'estimated_payload_bytes': 0 if central else u*c*state['model'].parameter_bytes()*
                    (6 if arm == 'mime' else 4 if arm == 'scaffold' else 2)}
    old.require(state['work'] == expected, f'Work/payload mismatch: {arm}')


def train_one(data, base, arm, settings, path, identity, criterion):
    path.mkdir(parents=True, exist_ok=False)
    lr = settings['lr']
    state = make_state(base, arm, lr)
    n, _ = units(base, arm)
    started = time.perf_counter()
    for index in range(n):
        if index == n-2:
            boundary = snapshot(state, arm)
        advance(state, data, base, arm, settings, criterion)
        if (index+1) % max(1, n//12) == 0 or index+1 == n:
            check_work(state, base, arm, data)
            print(f"seed={base['seed']} {arm} step={index+1}/{n}", flush=True)
    seconds = time.perf_counter()-started
    # Replay the final two boundaries from a real serialized checkpoint.
    atomic_save({'state': boundary, 'identity': identity}, path/'recovery.pt')
    recovered = restore(torch.load(path/'recovery.pt', weights_only=True)['state'], base, arm, lr)
    for _ in range(2):
        advance(recovered, data, base, arm, settings, criterion)
    old.require(old.block.nested_equal(snapshot(state, arm), snapshot(recovered, arm)), 'Recovery differs')
    atomic_save({'state': snapshot(state, arm), 'identity': identity}, path/'last.pt')
    return state, seconds, 2*units(base, arm)[1]


def train(root, output):
    plan, base, data, alarm, x, y, calibration = preflight(root)
    output = old.require_fresh_output(output)
    identity = contract()
    atomic_json({'contract': identity, 'plan': plan, 'runtime': portable.runtime_record()}, output/'contract.json')
    atomic_json({'status': 'training', 'confirmation_scored': False}, output/'status.json')
    criterion = old.build_official_criterion('sqrt_weighted_ce', data.manifest['counts']['fit'], 'fixed_train_mean')
    results = []
    for seed in plan['seeds']:
        recipe = {**base, 'seed': seed}
        for arm, settings in plan['arms'].items():
            path = output/f'seed{seed}-{arm}'
            print(f'START seed={seed} {arm}', flush=True)
            state, seconds, replay_work = train_one(data, recipe, arm, settings, path, identity, criterion)
            model = state['visible'] if arm == 'block128' else state['model']
            logits = old.rep.infer(model, x, alarm['inference_batch_size'])
            np.save(path/'development_logits.npy', logits, allow_pickle=False)
            development = old.alarms.evaluate_scores(logits, y, calibration, [100, 500, 1000, 2000])
            old.replication.independent_points(logits, y, calibration, development['operating_points'])
            result = {'seed': seed, 'arm': arm, 'settings': settings, 'work': state['work'],
                      'initial_model_sha256': old.state_digest(old.training.initial_model(recipe)),
                      'model_sha256': old.state_digest(model), 'sample_chain_sha256': state['sample_chain_sha256'],
                      'checkpoint_sha256': old.sha256(path/'last.pt'),
                      'development_logits_sha256': old.sha256(path/'development_logits.npy'),
                      'recovery_exact': True, 'recovery_backward_examples': replay_work,
                      'training_seconds': seconds, 'development': development}
            atomic_json(result, path/'development.json')
            results.append(result)
            print(f"DONE {arm}: development audit F1={development['operating_points'][1]['audit_metrics']['macro_f1']:.6f}", flush=True)
    old.require(identity == contract(), 'Sources changed during training')
    old.require(sum(r['work']['backward_examples'] for r in results) == plan['total_training_backward_examples'], 'Total work differs')
    # Same local client rows and order for all local-Adam and local-SGD arms.
    for seed in plan['seeds']:
        rows = [r for r in results if r['seed'] == seed]
        old.require(len({r['initial_model_sha256'] for r in rows}) == 1, 'Initialization not paired')
        local = [r for r in rows if r['arm'] in EXTRA-{'central_sgd'} or r['arm'] == 'fedavg_adam']
        old.require(len({r['sample_chain_sha256'] for r in local}) == 1, 'Local row streams not paired')
    summary = {'status': 'training_complete', 'contract': identity, 'plan': plan, 'results': results,
               'runtime': portable.runtime_record(), 'confirmation_scored': False,
               'original_validation_opened': False, 'original_test_opened': False}
    atomic_json(summary, output/'training-summary.json')
    atomic_json({'status': 'training_complete', 'confirmation_scored': False}, output/'status.json')
    return summary


def evaluate(confirmation_root, output):
    output = Path(output)
    old.require(not (output/'summary.json').exists(), 'Preserve completed evaluation')
    training = json.loads((output/'training-summary.json').read_text())
    plan = read_plan()
    old.require(training['contract'] == contract() and training['plan'] == plan, 'Evaluation source/plan differs')
    old.require({(r['seed'],r['arm']) for r in training['results']} ==
                {(s,a) for s in plan['seeds'] for a in plan['arms']} and len(training['results']) == 33,
                'Incomplete training; score nothing')
    published = old.published_reference()
    receipt = Path(confirmation_root)/'manifest.json'
    old.require(old.sha256(receipt) == published['data_receipt_sha256'] and
                json.loads(receipt.read_text()) == published['data'], 'Confirmation identity differs')
    x, y = old.cohort.load_reconstructed(confirmation_root, published['data'])
    base = old.confirmation.read_contract(require_runtime=True)[2]
    results = []
    for row in training['results']:
        seed, arm = row['seed'], row['arm']
        path = output/f'seed{seed}-{arm}'
        old.require(json.loads((path/'development.json').read_text()) == row and
                    old.sha256(path/'last.pt') == row['checkpoint_sha256'] and
                    old.sha256(path/'development_logits.npy') == row['development_logits_sha256'], 'Endpoint changed')
        checkpoint = torch.load(path/'last.pt', weights_only=True)
        old.require(checkpoint['identity'] == contract(), 'Checkpoint identity differs')
        state = restore(checkpoint['state'], {**base, 'seed': seed}, arm, row['settings']['lr'])
        model = state['visible'] if arm == 'block128' else state['model']
        old.require(old.state_digest(model) == row['model_sha256'], 'Model differs')
        calibrations = [p['calibration'] for p in row['development']['operating_points']]
        logits = old.rep.infer(model, x, 8192)
        np.save(path/'confirmation_logits.npy', logits, allow_pickle=False)
        scores = old.confirmation.fixed_scores(logits, y, calibrations)
        old.confirmation.independent_metrics(logits, y, calibrations, scores)
        result = {**row, 'confirmation': scores, 'confirmation_logits_sha256': old.sha256(path/'confirmation_logits.npy')}
        atomic_json(result, path/'result.json')
        results.append(result)
    summary = {**training, 'status': 'complete', 'results': results, 'confirmation_scored': True,
               'confirmation_previously_evaluated': True, 'confirmation_thresholds_refitted': False,
               'training_summary_sha256': old.sha256(output/'training-summary.json')}
    atomic_json(summary, output/'summary.json')
    atomic_json({'status': 'complete', 'confirmation_previously_evaluated': True}, output/'status.json')
    print('PASS: all 33 frozen endpoints evaluated; no confirmation tuning', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'train', 'evaluate'))
    parser.add_argument('--data', type=Path, default=old.DATA)
    parser.add_argument('--confirmation-data', type=Path, default=old.CONFIRMATION)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.action == 'check':
        preflight(args.data)
        print('PASS: frozen sources, runtime, fitting/development data; no training or confirmation scoring')
    else:
        old.require(args.output is not None, 'A new output directory is required')
        if args.action == 'train':
            train(args.data, args.output)
        else:
            evaluate(args.confirmation_data, args.output)


if __name__ == '__main__':
    main()
