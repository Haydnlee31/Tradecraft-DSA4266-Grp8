"""A bounded CPU synchronization-interval development screen with recovery."""
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

from extensions.fedntd.data import DevelopmentData, PARENT_PLAN, ROOT, sampled_rows, sha256
from extensions.fedntd.run import initial_model, evaluate, atomic_write
from extensions.sync_intervals.core import interval_step, fedavg_step
from src.data.label_map import CLASSES
from src.federated import shared_adam as shared
from src.models.official_losses import build_official_criterion

HERE = Path(__file__).resolve().parent
PLAN = HERE/'plan.json'


def validate_plan(plan):
    for key in ('clients','fit_rows','features','batch_size','backward_examples_per_arm',
                'sample_block_steps','threads','checkpoint_every'):
        shared.require(type(plan[key]) is int and plan[key] > 0, f'Invalid {key}')
    shared.require(type(plan['seed']) is int and plan['seed'] >= 0, 'Invalid seed')
    for key in ('local_learning_rate','server_learning_rate','control_adam_learning_rate'):
        shared.require(math.isfinite(plan[key]) and plan[key] > 0, f'Invalid {key}')
    shared.require(plan['architecture'] == {'hidden_dims':[64,32], 'normalization':'layer', 'dropout':0.0},
                   'Frozen architecture required')
    adam = plan['adam']
    shared.require(len(adam['betas']) == 2 and all(math.isfinite(b) and 0 <= b < 1 for b in adam['betas'])
                   and math.isfinite(adam['eps']) and adam['eps'] > 0 and adam['weight_decay'] == 0,
                   'Valid Adam and zero decay required')
    shared.require(plan['arms'] and len({a['name'] for a in plan['arms']}) == len(plan['arms']),
                   'Unique nonempty arms required')
    shared.require(plan['arms'][0]['kind'] == 'interval' and plan['arms'][0]['interval'] == 1,
                   'First arm must be the K=1 reference')
    for arm in plan['arms']:
        name = arm['name']
        shared.require(name and all(c.isalnum() or c in '_-' for c in name), 'Unsafe arm name')
        shared.require(arm['kind'] in ('interval','fedavg_sgd','fedavg_adam'), 'Unknown updater')
        shared.require(type(arm['interval']) is int and arm['interval'] > 0, 'Invalid interval')
        units, remainder = divmod(plan['backward_examples_per_arm'],
                                  plan['clients']*plan['batch_size']*arm['interval'])
        shared.require(units > 0 and remainder == 0, 'Budget must divide full rounds')
    for key, value in plan['retention_gate'].items():
        shared.require(math.isfinite(value) and value >= 0, f'Invalid gate: {key}')


def recipe(plan):
    return {**plan['adam'], 'lr':plan['server_learning_rate']}


def make_state(plan, arm):
    model = initial_model(plan)
    return {'model':model, 'optimizer':shared.make_optimizer(model, recipe(plan))
            if arm['kind'] == 'interval' else None, 'round':0, 'work':{},
            'client_sample_chains':['0'*64 for _ in range(plan['clients'])],
            'traversal_sample_chain':'0'*64}


def advance(state, data, plan, arm, criterion):
    k = arm['interval']
    def batch(client, step):
        # Global per-client microstep, NOT the arm's synchronization-round seed.
        microstep = state['round']*k+step
        block, within = divmod(microstep, plan['sample_block_steps'])
        rows = sampled_rows(data, plan, block, client, within)
        payload = rows.astype('<i8').tobytes()
        state['client_sample_chains'][client] = hashlib.sha256(
            bytes.fromhex(state['client_sample_chains'][client])+payload).hexdigest()
        state['traversal_sample_chain'] = hashlib.sha256(
            bytes.fromhex(state['traversal_sample_chain'])+payload).hexdigest()
        return data.batch(rows)
    counts = [len(p) for p in data.parts]
    if arm['kind'] == 'interval':
        examples = interval_step(state['model'], state['optimizer'], counts, batch, criterion,
                                 k, plan['local_learning_rate'], plan['seed'])
    else:
        adam = {**plan['adam'],'lr':plan['control_adam_learning_rate']} if arm['kind'] == 'fedavg_adam' else None
        examples = fedavg_step(state['model'], counts, batch, criterion, k,
                               plan['local_learning_rate'], plan['seed'], adam)
    c = plan['clients']
    shared.require(examples == c*k*plan['batch_size'], 'Batch coverage differs')
    work = {'backward_examples':examples, 'backward_batches':c*k,
            'local_optimizer_steps':0 if arm['kind'] == 'interval' and k == 1 else c*k,
            'server_adam_steps':1 if arm['kind'] == 'interval' else 0,
            'synchronization_rounds':1, 'logical_barriers':1,
            'estimated_payload_bytes':2*c*state['model'].parameter_bytes()}
    for key, value in work.items():
        state['work'][key] = state['work'].get(key,0)+value
    state['round'] += 1


def snapshot(state):
    return {'model':copy.deepcopy(state['model'].state_dict()),
            'optimizer':copy.deepcopy(state['optimizer'].state_dict()) if state['optimizer'] else None,
            **{k:copy.deepcopy(state[k]) for k in ('round','work','client_sample_chains','traversal_sample_chain')}}


def restore(saved, plan, arm):
    state = make_state(plan, arm)
    state['model'].load_state_dict(saved['model'], strict=True)
    shared.require((saved['optimizer'] is not None) == (state['optimizer'] is not None), 'Missing optimizer state')
    if state['optimizer']:
        state['optimizer'].load_state_dict(saved['optimizer'])
    for key in ('round','work','client_sample_chains','traversal_sample_chain'):
        state[key] = copy.deepcopy(saved[key])
    completed = state['round']*plan['clients']*plan['batch_size']*arm['interval']
    shared.require(type(state['round']) is int and 0 <= completed <= plan['backward_examples_per_arm'],
                   'Invalid recovery round')
    if state['round']:
        shared.require(state['work']['backward_examples'] == completed, 'Recovery work differs')
        if state['optimizer']:
            shared.require(all(float(state['optimizer'].state[p]['step']) == state['round']
                               for p in state['model'].parameters()), 'Server Adam step counters differ')
    return state


def sources():
    paths = list((ROOT/'src').rglob('*.py'))+list((ROOT/'extensions/fedntd').glob('*.py'))+list(HERE.glob('*.py'))
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}


def comparisons(results, plan):
    base = results[0]
    gate = plan['retention_gate']
    out = []
    for result in results[1:]:
        a, b = result['development_metrics'], base['development_metrics']
        quantities = {'macro_f1_delta':a['macro_f1']-b['macro_f1'],
                      'far_delta':a['benign_false_alert_rate']-b['benign_false_alert_rate'],
                      'max_class_recall_loss':max(b['per_class'][c]['recall']-a['per_class'][c]['recall'] for c in CLASSES),
                      'barrier_reduction':base['rounds']/result['rounds']}
        checks = {'macro_f1_retained':quantities['macro_f1_delta'] >= -gate['macro_f1_loss_max'],
                  'far_retained':quantities['far_delta'] <= gate['far_increase_max'],
                  'class_recalls_retained':quantities['max_class_recall_loss'] <= gate['any_class_recall_loss_max'],
                  'barriers_reduced':quantities['barrier_reduction'] >= gate['barrier_reduction_min']}
        out.append({'arm':result['arm'], **quantities, 'checks':checks,
                    'passes_retention_gate':all(checks.values())})
    return out


def run_locked(data, output, plan, criterion, contract):
    path = output/'contract.json'
    if path.exists():
        shared.require(json.loads(path.read_text()) == contract, 'Source/data/plan/runtime contract changed')
    else:
        shared.require(set(p.name for p in output.iterdir()) <= {'RUNNING.lock'}, 'Unknown output files')
        atomic_write(path, contract)
    shared.require(not (output/'summary.json').exists(), 'Completed run exists; preserve it')
    results = []
    for arm in plan['arms']:
        dest = output/arm['name']
        dest.mkdir(exist_ok=True)
        checkpoint, final = dest/'last.pt', dest/'result.json'
        units = plan['backward_examples_per_arm']//(plan['clients']*plan['batch_size']*arm['interval'])
        if final.exists():
            result = json.loads(final.read_text())
            shared.require(result['contract'] == contract and result['arm'] == arm,
                           'Saved result contract differs')
            shared.require(sha256(checkpoint) == result['checkpoint_sha256'], 'Completed checkpoint differs')
            results.append(result)
            continue
        state, seconds = make_state(plan,arm), 0.
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            shared.require(saved['contract'] == contract and saved['arm'] == arm, 'Recovery contract differs')
            state = restore(saved['state'],plan,arm)
            seconds = saved['training_seconds_sim_host']
        print(f'START {arm["name"]} {state["round"]}/{units}',flush=True)
        while state['round'] < units:
            if state['round'] == units-1:
                atomic_write(dest/'recovery.pt',{'contract':contract,'arm':arm,'state':snapshot(state)},tensor=True)
            start = time.perf_counter()
            advance(state,data,plan,arm,criterion)
            seconds += time.perf_counter()-start
            if state['round'] % min(plan['checkpoint_every'],max(1,units//12)) == 0 or state['round'] == units:
                atomic_write(checkpoint,{'contract':contract,'arm':arm,'state':snapshot(state),
                             'training_seconds_sim_host':seconds}, tensor=True)
                print(f'  {arm["name"]} {state["round"]}/{units}',flush=True)
        metrics = evaluate(state['model'],data)
        result = {'status':'complete','arm':arm,'contract':contract,'rounds':state['round'],
                  'work':state['work'],'training_seconds_sim_host':seconds,'development_metrics':metrics,
                  'client_sample_chains':state['client_sample_chains'],
                  'traversal_sample_chain':state['traversal_sample_chain'],
                  'checkpoint_sha256':sha256(checkpoint),'recovery_sha256':sha256(dest/'recovery.pt'),
                  'num_parameters':state['model'].num_parameters(),
                  'test_opened':False,'confirmation_scored':False,'network_measured':False}
        shared.require(state['work']['backward_examples'] == plan['backward_examples_per_arm'], 'Work differs')
        atomic_write(final,result)
        results.append(result)
        print(f'END {arm["name"]}: F1={metrics["macro_f1"]:.6f} FAR={metrics["benign_false_alert_rate"]:.6f}',flush=True)
    shared.require(sources() == contract['sources'], 'Sources changed during run')
    shared.require(all(r['client_sample_chains'] == results[0]['client_sample_chains'] for r in results),
                   'Per-client samples differ across intervals')
    summary = {'status':'complete','contract':contract,'results':results,
               'comparisons_to_k1':comparisons(results,plan),'test_opened':False,'confirmation_scored':False}
    atomic_write(output/'summary.json',summary)
    return summary


def run(data_root, output, plan_path=PLAN):
    plan = json.loads(Path(plan_path).read_text())
    validate_plan(plan)
    output = Path(output).resolve()
    runs = HERE/'runs'
    shared.require(runs.resolve() in output.parents and HERE in output.parents and not runs.is_symlink(),
                   'Output must be a subdirectory of extensions/sync_intervals/runs')
    data_root = Path(data_root).resolve()
    shared.require(data_root != output and data_root not in output.parents and output not in data_root.parents,
                   'Input/output paths must be separate')
    torch.set_num_threads(plan['threads'])
    torch.use_deterministic_algorithms(True)
    data = DevelopmentData(data_root,plan)
    criterion = build_official_criterion('sqrt_weighted_ce',data.manifest['counts']['fit'],'fixed_train_mean')
    contract = {'plan':plan,'plan_sha256':sha256(plan_path),'parent_plan_sha256':sha256(PARENT_PLAN),
                'data_manifest_sha256':sha256(data.root/'manifest.json'),'files':data.file_hashes,
                'sources':sources(),'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),
                    'numpy':str(np.__version__),'platform':platform.platform()}}
    output.mkdir(parents=True,exist_ok=True)
    lock = output/'RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        return run_locked(data,output,plan,criterion,contract)
    finally:
        lock.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path,default=PLAN)
    a = p.parse_args()
    run(a.data,a.output,a.plan)


if __name__ == '__main__':
    main()
