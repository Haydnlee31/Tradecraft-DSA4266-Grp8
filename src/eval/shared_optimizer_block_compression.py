"""Frozen block128 follow-up with an exact tensor-scale compatibility control."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.shared_optimizer_data import ROOT
from src.eval import shared_optimizer_alarms as alarms
from src.eval import shared_optimizer_performance as training
from src.eval import shared_optimizer_replication as rep
from src.eval.federated_class_safeguard import state_digest
from src.federated import block_shared_adam as codec
from src.eval import shared_optimizer_compression as previous
from src.federated.shared_adam import require
from src.models.official_losses import build_official_criterion
from src.models.research import atomic_json, atomic_save

PLAN = ROOT/'reports/federated_shared_optimizer/block-compression-plan.json'
PARENT = PLAN.parent/'compression-results.json'


def sources():
    return {**previous.sources(), 'src/federated/block_shared_adam.py':sha256(codec.__file__),
            'src/eval/shared_optimizer_block_compression.py':sha256(__file__)}


def read_contract():
    plan = json.loads(PLAN.read_text())
    old,base,alarm_plan,performance,_,references = previous.read_contract()
    require(sha256(PARENT) == plan['parent_evidence_sha256'], 'Compression evidence differs')
    parent = json.loads(PARENT.read_text())
    require(parent['summary']['contract']['sources'] == previous.sources(), 'Parent sources differ')
    require(plan['seed'] == old['seed'] and plan['learning_rate'] == old['learning_rate'], 'Training settings changed')
    require(plan['arms'] == {'tensor_bridge':{'bits':4,'feedback':True,'block_size':None},
                            'block128':{'bits':4,'feedback':True,'block_size':128}}, 'Arms differ')
    for key in ('backward_examples_per_run','recovery_replay_steps','primary_target_basis_points',
                'false_alert_targets_basis_points','device','threads'):
        require(plan[key] == old[key], 'Fixed settings differ')
    screen = dict(old['screen'])
    screen['payload_ratio_to_historical_shared_max'] = screen['payload_ratio_to_historical_shared_max']['ef4']
    require(plan['screen'] == screen, 'Quality safeguards changed')
    require(plan['new_training_backward_examples'] == 2*plan['backward_examples_per_run'] and
            plan['new_recovery_backward_examples'] == 2*plan['recovery_replay_steps']*base['clients']*base['batch_size'],
            'Budgets differ')
    require(all(plan[k] is False for k in ('confirmation_scoring_permitted',
            'original_validation_scoring_permitted','test_scoring_permitted')), 'Holdout access not permitted')
    return plan,base,alarm_plan,performance,parent,references


def nested_equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype == b.dtype and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys() == b.keys() and all(nested_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a) is type(b) and len(a) == len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a == b


def check_work(state,base):
    units = state['unit']
    c,b = base['clients'],base['batch_size']
    wire = codec.expected_wire(state['model'],base,state['bits'],units,state['block_size'])
    expected = {'backward_examples':units*c*b,'backward_batches':units*c,'optimizer_steps':units,
                'anchor_examples':0,'local_examples':0,'synchronization_rounds':units,'logical_barriers':units,
                'estimated_payload_bytes':sum(wire.values())}
    require(state['wire'] == wire and state['work'] == expected, 'Work/payload accounting differs')


def train_candidate(data,base,settings,lr,destination,identity,criterion,recovery_steps):
    destination = Path(destination)
    destination.mkdir(exist_ok=True)
    checkpoint = destination/'last.pt'
    state = codec.make_state(base,settings['bits'],settings['feedback'],lr,settings['block_size'])
    elapsed = 0.
    if checkpoint.exists():
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        require(saved['identity'] == identity, 'Resume identity differs')
        state = codec.restore(saved['state'],base,lr)
        require((state['bits'],state['feedback'],state['block_size']) ==
                (settings['bits'],settings['feedback'],settings['block_size']), 'Resume codec differs')
        elapsed = saved['training_seconds']
    units,_ = training.work_plan(base,'shared_adam')
    require(0 <= state['unit'] <= units and units > recovery_steps > 0, 'Invalid recovery boundary')
    check_work(state,base)
    started = time.perf_counter()
    while state['unit'] < units:
        if state['unit'] == units-recovery_steps:
            atomic_save({'identity':identity,'state':codec.snapshot(state)},destination/'recovery.pt')
        codec.advance(state,data,base,lr,criterion)
        if state['unit'] % max(1,units//12) == 0 or state['unit'] == units:
            check_work(state,base)
            atomic_save({'identity':identity,'state':codec.snapshot(state),
                         'training_seconds':elapsed+time.perf_counter()-started},checkpoint)
            print(f'{identity["arm"]} step={state["unit"]}/{units} payload={state["work"]["estimated_payload_bytes"]:,}',flush=True)
    require((destination/'recovery.pt').exists(), 'Missing recovery boundary')
    return state,torch.load(checkpoint,map_location='cpu',weights_only=True)['training_seconds']


def primary(result,plan):
    return next(p['audit_metrics'] for p in result['operating_points']
                if p['calibration']['target_basis_points'] == plan['primary_target_basis_points'])


def assess(results,references,plan):
    require(len(results) == 2 and {r['arm'] for r in results} == set(plan['arms']), 'Missing/duplicate arms')
    shared,fed = primary(references['shared_adam'],plan),primary(references['fedavg_adam'],plan)
    screens = {}
    for r in results:
        if r['arm'] != 'block128':
            continue
        m = primary(r,plan)
        numbers = {'macro_f1_drop_from_shared':shared['macro_f1']-m['macro_f1'],
                   'absolute_far_difference_from_shared':abs(m['benign_false_alert_rate']-shared['benign_false_alert_rate']),
                   'any_class_recall_drop_from_shared':max(shared['per_class'][c]['recall']-m['per_class'][c]['recall'] for c in CLASSES),
                   'macro_f1_gain_over_fedavg':m['macro_f1']-fed['macro_f1'],
                   'payload_ratio_to_historical_shared':r['work']['estimated_payload_bytes']/references['shared_adam']['work']['estimated_payload_bytes']}
        checks = {}
        for key,value in plan['screen'].items():
            threshold = value[r['arm']] if isinstance(value,dict) else value
            metric = numbers[key.rsplit('_',1)[0]]
            checks[key] = metric >= threshold if key.endswith('_min') else metric <= threshold
        screens[r['arm']] = {'passes':all(checks.values()),'checks':checks,**numbers}
    passing = [r for r in results if r['arm'] in screens and screens[r['arm']]['passes']]
    selected = min(passing,key=lambda r:(r['work']['estimated_payload_bytes'],-primary(r,plan)['macro_f1']))['arm'] if passing else None
    return {'screens':screens,'candidate_for_replication':selected,'previous_training_screen_overridden':False}


def tensor_reference(parent):
    return next(r for r in parent['summary']['results'] if r['arm'] == 'ef4')


def bridge_check(state,reference,checkpoint):
    require(sha256(checkpoint) == reference['checkpoint_sha256'], 'Historical checkpoint differs')
    old = torch.load(checkpoint,map_location='cpu',weights_only=True)['state']
    saved = codec.snapshot(state)
    require(saved.pop('block_size') is None, 'Bridge layout differs')
    require(nested_equal(saved,old), 'Tensor bridge state differs')
    require(state_digest(state['visible']) == reference['delivered_model_sha256'], 'Delivered bridge differs')


def run(data_root,output,reference_checkpoint):
    plan,base,alarm_plan,performance,parent,references = read_contract()
    output = Path(output)
    require(not (output/'summary.json').exists(), 'Preserve completed screen')
    contract = {'plan_sha256':sha256(PLAN),'sources':sources(),'parent_evidence_sha256':sha256(PARENT)}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'contract.json').exists():
        require(json.loads((output/'contract.json').read_text()) == contract, 'Run contract differs')
    else:
        require(not list(output.iterdir()), 'Unknown output contents')
        atomic_json(contract,output/'contract.json')
    data = rep.FitData(data_root,base,alarm_plan['data_manifest_sha256'])
    x,y,_ = alarms.load_development(data_root,alarm_plan,performance)
    calibration,groups = alarms.grouped_split(x,alarm_plan['split_domain'])
    split = parent['summary']['split']
    require(all(split[k] == v for k,v in groups.items()), 'Groups differ')
    np.save(output/'calibration_mask.npy',calibration,allow_pickle=False)
    require(sha256(output/'calibration_mask.npy') == split['calibration_mask_sha256'], 'Split mask differs')
    criterion = build_official_criterion('sqrt_weighted_ce',data.manifest['counts']['fit'],'fixed_train_mean')
    results = []
    for arm,settings in plan['arms'].items():
        destination = output/arm
        identity = {'contract':contract,'arm':arm,'settings':settings,'seed':plan['seed'],'learning_rate':plan['learning_rate']}
        print(f'START {arm}',flush=True)
        state,seconds = train_candidate(data,base,settings,plan['learning_rate'],destination,identity,criterion,plan['recovery_replay_steps'])
        require(state['work']['backward_examples'] == plan['backward_examples_per_run'], 'Final budget differs')
        if arm == 'tensor_bridge':
            bridge_check(state,tensor_reference(parent),reference_checkpoint)
        logits = rep.infer(state['visible'],x,alarm_plan['inference_batch_size'])
        np.save(destination/'logits.npy',logits,allow_pickle=False)
        evaluated = alarms.evaluate_scores(logits,y,calibration,plan['false_alert_targets_basis_points'])
        if arm == 'tensor_bridge':
            require(all(evaluated[k] == tensor_reference(parent)[k] for k in evaluated), 'Historical endpoint differs')
        r = {'arm':arm,'settings':settings,'seed':plan['seed'],'learning_rate':plan['learning_rate'],
             'work':state['work'],'wire':state['wire'],'units':state['unit'],'training_seconds':seconds,
             'sample_chain_sha256':state['sample_chain_sha256'],'initial_model_sha256':state_digest(training.initial_model(base)),
             'delivered_model_sha256':state_digest(state['visible']),'server_target_sha256':state_digest(state['model']),
             'server_delivered_l2':float(torch.linalg.vector_norm(torch.cat([p.detach().reshape(-1)-v.detach().reshape(-1)
                                              for p,v in zip(state['model'].parameters(),state['visible'].parameters())]))),
             'checkpoint_sha256':sha256(destination/'last.pt'),'recovery_checkpoint_sha256':sha256(destination/'recovery.pt'),
             'logits_sha256':sha256(destination/'logits.npy'),**evaluated}
        require(r['sample_chain_sha256'] == references['shared_adam']['sample_chain_sha256'], 'Client sampling differs')
        atomic_json(r,destination/'result.json')
        results.append(r)
        m = primary(r,plan)
        print(f'END {arm}: macro-F1={m["macro_f1"]:.6f}, actual FAR={m["benign_false_alert_rate"]:.4%}',flush=True)
    require(sources() == contract['sources'] and sha256(PLAN) == contract['plan_sha256'], 'Source/plan changed')
    total = sum(r['work']['backward_examples'] for r in results)
    require(total == plan['new_training_backward_examples'], 'Total work differs')
    summary = {'status':'complete','plan':plan,'contract':contract,'results':results,'references':references,
               'split':split,'assessment':assess(results,references,plan),'new_training_backward_examples':total,
               'forward_examples':len(results)*len(y),'runtime':{'python':platform.python_version(),
                  'torch':str(torch.__version__),'numpy':np.__version__,
                  'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()},
               'tensor_bridge_exact':True,'confirmation_opened':False,'original_validation_opened':False,
               'test_opened':False,'test_evaluated':False,'network_measured':False}
    atomic_json(summary,output/'summary.json')
    return summary


def audit(data_root,output,receipt,reference_checkpoint):
    output,receipt = Path(output),Path(receipt)
    require(not receipt.exists(), 'Preserve existing audit')
    plan,base,alarm_plan,performance,parent,references = read_contract()
    s = json.loads((output/'summary.json').read_text())
    require(s['plan'] == plan and s['contract'] == {'plan_sha256':sha256(PLAN),'sources':sources(),
            'parent_evidence_sha256':sha256(PARENT)} and s['references'] == references, 'Audit contract differs')
    data = rep.FitData(data_root,base,alarm_plan['data_manifest_sha256'])
    criterion = build_official_criterion('sqrt_weighted_ce',data.manifest['counts']['fit'],'fixed_train_mean')
    x,y,_ = alarms.load_development(data_root,alarm_plan,performance)
    calibration,groups = alarms.grouped_split(x,alarm_plan['split_domain'])
    require(s['split'] == parent['summary']['split'] and all(s['split'][k] == v for k,v in groups.items()), 'Audit groups differ')
    require(sha256(output/'calibration_mask.npy') == s['split']['calibration_mask_sha256'] and
            np.array_equal(calibration,np.load(output/'calibration_mask.npy',allow_pickle=False)), 'Audit mask differs')
    checks = []
    for r in s['results']:
        path = output/r['arm']
        for name,key in [('last.pt','checkpoint_sha256'),('recovery.pt','recovery_checkpoint_sha256'),('logits.npy','logits_sha256')]:
            require(sha256(path/name) == r[key], 'Saved artifact differs')
        saved = torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        earlier = torch.load(path/'recovery.pt',map_location='cpu',weights_only=True)
        expected_identity = {'contract':s['contract'],'arm':r['arm'],'settings':plan['arms'][r['arm']],
                             'seed':plan['seed'],'learning_rate':plan['learning_rate']}
        require(saved['identity'] == earlier['identity'] == expected_identity, 'Checkpoint identity differs')
        state = codec.restore(earlier['state'],base,plan['learning_rate'])
        require(state['unit'] == r['units']-plan['recovery_replay_steps'], 'Recovery boundary differs')
        for _ in range(plan['recovery_replay_steps']):
            codec.advance(state,data,base,plan['learning_rate'],criterion)
        require(nested_equal(codec.snapshot(state),saved['state']), 'Recovery is not exact')
        check_work(state,base)
        require(state['work'] == r['work'] and state['wire'] == r['wire'] and
                state['unit'] == training.work_plan(base,'shared_adam')[0] and
                state['sample_chain_sha256'] == references['shared_adam']['sample_chain_sha256'] == r['sample_chain_sha256'],
                'Final work/samples differ')
        require(state_digest(state['visible']) == r['delivered_model_sha256'] and
                state_digest(state['model']) == r['server_target_sha256'], 'Model hashes differ')
        if r['arm'] == 'tensor_bridge':
            bridge_check(state,tensor_reference(parent),reference_checkpoint)
        logits = np.load(path/'logits.npy',allow_pickle=False)
        require(np.array_equal(logits,rep.infer(state['visible'],x,alarm_plan['inference_batch_size'])), 'Delivered inference differs')
        evaluated = alarms.evaluate_scores(logits,y,calibration,plan['false_alert_targets_basis_points'])
        require(all(r[k] == v for k,v in evaluated.items()), 'Metrics differ')
        winners = alarms.ATTACKS[logits[:,alarms.ATTACKS].argmax(axis=1)]
        top = logits[np.arange(len(y)),winners].astype(np.float64)
        benign = logits[:,alarms.BENIGN].astype(np.float64)
        ordered = np.sort((top-benign)[calibration & (y == alarms.BENIGN)])
        for point in r['operating_points']:
            c = point['calibration']; allowed = len(ordered)*c['target_basis_points']//10000
            require(c['threshold'] == float(ordered[len(ordered)-allowed-1]), 'Sorted threshold differs')
            predicted = np.where(top > benign+c['threshold'],winners,alarms.BENIGN)
            require(alarms.metrics(y[~calibration],predicted[~calibration]) == point['audit_metrics'], 'Direct-logit prediction differs')
        checks.append({'arm':r['arm'],'recovery_exact':True,'checkpoint_inference_exact':True,
                       'operating_points_exact':len(r['operating_points']),'work_and_serialized_bytes_verified':True})
    require(s['assessment'] == assess(s['results'],references,plan), 'Assessment differs')
    require(sum(r['work']['backward_examples'] for r in s['results']) == s['new_training_backward_examples'] ==
            plan['new_training_backward_examples'], 'Total training work differs')
    recovery_work = len(checks)*plan['recovery_replay_steps']*base['clients']*base['batch_size']
    require(recovery_work == plan['new_recovery_backward_examples'], 'Recovery work differs')
    result = {'status':'complete','summary_sha256':sha256(output/'summary.json'),'source_sha256':sha256(__file__),
              'checks':checks,'new_recovery_backward_examples':recovery_work,'new_forward_examples':len(checks)*len(y),
              'confirmation_opened':False,'original_validation_opened':False,'test_opened':False,'test_evaluated':False}
    atomic_json(result,receipt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','audit'))
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--reference-checkpoint',type=Path,
                        default=ROOT/'outputs/shared-optimizer-compression-v1/ef4/last.pt')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    if args.action == 'run':
        require(args.receipt is None, 'Run takes no audit receipt')
        result = run(args.data,args.output,args.reference_checkpoint)
    else:
        require(args.receipt is not None, 'Audit needs --receipt')
        result = audit(args.data,args.output,args.receipt,args.reference_checkpoint)
    print(json.dumps({'status':result['status'],'test_opened':result['test_opened'],
                      'assessment':result.get('assessment')},indent=2))


if __name__ == '__main__':
    main()
