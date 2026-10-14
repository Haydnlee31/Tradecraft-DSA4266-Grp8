"""Paired seeds for the fixed block128 candidate, without changing its updater.

All training, serialization and calibration reuse the frozen implementations.
This module only schedules the two new seeds and audits their paired endpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.shared_optimizer_data import ROOT
from src.eval import shared_optimizer_alarms as alarms
from src.eval import shared_optimizer_block_compression as block
from src.eval import shared_optimizer_performance as training
from src.eval import shared_optimizer_replication as rep
from src.eval.federated_class_safeguard import state_digest
from src.federated import block_shared_adam as codec
from src.federated.shared_adam import require
from src.models.official_losses import build_official_criterion
from src.models.research import atomic_json

PLAN = ROOT/'reports/federated_shared_optimizer/block-replication-plan.json'
PARENT = PLAN.parent/'block-compression-results.json'
REFERENCES = PLAN.parent/'replication-results.json'


def sources():
    return {**block.sources(), 'src/eval/shared_optimizer_block_replication.py':sha256(__file__)}


def runtime():
    return {'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__}


def read_contract(require_runtime=False):
    """Static contract tests stay portable; actual paired runs require the old runtime."""
    plan = json.loads(PLAN.read_text())
    old,base,alarm_plan,performance,_,_ = block.read_contract()
    require(sha256(PARENT) == plan['parent_evidence_sha256'] and
            sha256(REFERENCES) == plan['reference_evidence_sha256'], 'Parent evidence changed')
    parent,reference = [json.loads(p.read_text()) for p in (PARENT,REFERENCES)]
    for evidence,expected_sources in ((parent,block.sources()),(reference,rep.sources())):
        require(evidence['summary']['contract']['sources'] == expected_sources and
                evidence['audit']['status'] == 'complete', 'Parent not audited or sources changed')
        require(hashlib.sha256(json.dumps(evidence['summary'],indent=2).encode()).hexdigest() ==
                evidence['audit']['summary_sha256'], 'Parent audit digest differs')
        if require_runtime:
            require(all(evidence['summary']['runtime'][k] == v for k,v in runtime().items()), 'Reference runtime differs')
    require(parent['summary']['plan'] == old and reference['summary']['plan'] == rep.read_contract()[0],
            'Parent plan changed')
    require(parent['summary']['assessment']['candidate_for_replication'] == 'block128', 'Candidate not eligible')
    require(parent['summary']['split'] == reference['summary']['split'], 'Reference split differs')
    require(plan['seeds'] == [17,27] and plan['discovery_seed'] == 7 and
            plan['settings'] == old['arms']['block128'], 'Candidate/seed settings differ')
    for key in ('learning_rate','backward_examples_per_run','recovery_replay_steps','screen',
                'primary_target_basis_points','false_alert_targets_basis_points','device','threads'):
        require(plan[key] == old[key], f'Fixed setting differs: {key}')
    require(plan['new_training_backward_examples'] == 2*plan['backward_examples_per_run'] and
            plan['new_recovery_backward_examples'] == 2*plan['recovery_replay_steps']*base['clients']*base['batch_size'],
            'Budget changed')
    require(plan['require_every_seed_to_pass'] is True and all(plan[k] is False for k in
            ('confirmation_scoring_permitted','original_validation_scoring_permitted','test_scoring_permitted')),
            'Decision or data-access boundary changed')
    refs = reference['summary']['results']
    require(len(refs) == 9 and {(r['seed'],r['arm']) for r in refs} ==
            {(s,a) for s in (7,17,27) for a in ('central_adam','shared_adam','fedavg_adam')}, 'Reference endpoints differ')
    return plan,base,alarm_plan,performance,parent,refs


def contract():
    return {'plan_sha256':sha256(PLAN),'sources':sources(),
            'parent_evidence_sha256':sha256(PARENT),'reference_evidence_sha256':sha256(REFERENCES)}


def reference_for(refs,seed,arm):
    matched = [r for r in refs if r['seed'] == seed and r['arm'] == arm]
    require(len(matched) == 1, 'Missing/duplicate same-seed reference')
    return matched[0]


def check_pairing(result,refs,plan):
    ref = reference_for(refs,result['seed'],'shared_adam')
    require(result['arm'] == 'block128' and result['settings'] == plan['settings'] and
            result['learning_rate'] == plan['learning_rate'], 'Candidate settings differ')
    for key in ('initial_model_sha256','sample_chain_sha256'):
        require(result[key] == ref[key], f'Unpaired seed: {key}')
    for key,value in ref['work'].items():
        if key != 'estimated_payload_bytes':
            require(result['work'][key] == value, f'Unpaired work: {key}')


def assess(results,refs,plan):
    seeds = [plan['discovery_seed'],*plan['seeds']]
    require(len(results) == len(seeds) and {r['seed'] for r in results} == set(seeds) and
            all(r['arm'] == 'block128' for r in results), 'Missing/duplicate candidate endpoints')
    checks = []
    for r in results:
        check_pairing(r,refs,plan)
        shared = block.primary(reference_for(refs,r['seed'],'shared_adam'),plan)
        fed = block.primary(reference_for(refs,r['seed'],'fedavg_adam'),plan)
        m = block.primary(r,plan)
        drops = {c:shared['per_class'][c]['recall']-m['per_class'][c]['recall'] for c in CLASSES}
        numbers = {'macro_f1_drop_from_shared':shared['macro_f1']-m['macro_f1'],
            'absolute_far_difference_from_shared':abs(m['benign_false_alert_rate']-shared['benign_false_alert_rate']),
            'any_class_recall_drop_from_shared':max(drops.values()),
            'macro_f1_gain_over_fedavg':m['macro_f1']-fed['macro_f1'],
            'payload_ratio_to_historical_shared':r['work']['estimated_payload_bytes']/
                reference_for(refs,r['seed'],'shared_adam')['work']['estimated_payload_bytes']}
        passed = {k:(numbers[k.rsplit('_',1)[0]] >= v if k.endswith('_min') else numbers[k.rsplit('_',1)[0]] <= v)
                  for k,v in plan['screen'].items()}
        checks.append({'seed':r['seed'],'reused_discovery':r['seed'] == plan['discovery_seed'],
                       'passes':all(passed.values()),'checks':passed,'per_class_recall_drop':drops,**numbers})
    return {'per_seed':checks,'all_seeds_pass':all(c['passes'] for c in checks),
            'candidate_for_confirmation_preparation':'block128' if all(c['passes'] for c in checks) else None,
            'previous_training_screen_overridden':False,'confirmation_scoring_authorized':False}


def comparisons(results,refs,plan):
    records = []
    for r in results:
        for point in r['operating_points']:
            target,m = point['calibration']['target_basis_points'],point['audit_metrics']
            for arm in ('central_adam','shared_adam','fedavg_adam'):
                baseline = next(p['audit_metrics'] for p in reference_for(refs,r['seed'],arm)['operating_points']
                                if p['calibration']['target_basis_points'] == target)
                records.append({'seed':r['seed'],'target_basis_points':target,'reference':arm,
                    'block_minus_reference_macro_f1':m['macro_f1']-baseline['macro_f1'],
                    'block_minus_reference_far':m['benign_false_alert_rate']-baseline['benign_false_alert_rate'],
                    'per_class_recall_difference':{c:m['per_class'][c]['recall']-baseline['per_class'][c]['recall'] for c in CLASSES}})
    return records


def describe(results,refs,plan):
    return rep.describe([*refs,*results],{**plan,'arms':['central_adam','shared_adam','fedavg_adam','block128']})


def identity(plan,c,seed):
    return {'contract':c,'arm':'block128','settings':plan['settings'],'seed':seed,'learning_rate':plan['learning_rate']}


def development(data_root,alarm_plan,performance,split):
    x,y,_ = alarms.load_development(data_root,alarm_plan,performance)
    calibration,groups = alarms.grouped_split(x,alarm_plan['split_domain'])
    require(all(split[k] == v for k,v in groups.items()), 'Development groups differ')
    return x,y,calibration


def run(data_root,output):
    plan,base,alarm_plan,performance,parent,refs = read_contract(require_runtime=True)
    output = Path(output); c = contract()
    require(not (output/'summary.json').exists(), 'Preserve completed replication')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'contract.json').exists():
        require(json.loads((output/'contract.json').read_text()) == c, 'Resume contract differs')
    else:
        require(not list(output.iterdir()), 'Unknown output contents')
        atomic_json(c,output/'contract.json')
    data = rep.FitData(data_root,base,alarm_plan['data_manifest_sha256'])
    split = parent['summary']['split']
    x,y,calibration = development(data_root,alarm_plan,performance,split)
    np.save(output/'calibration_mask.npy',calibration,allow_pickle=False)
    require(sha256(output/'calibration_mask.npy') == split['calibration_mask_sha256'], 'Mask differs')
    criterion = build_official_criterion('sqrt_weighted_ce',data.manifest['counts']['fit'],'fixed_train_mean')
    discovery = next(r for r in parent['summary']['results'] if r['arm'] == 'block128')
    results = [{**discovery,'reused_discovery':True}]
    for seed in plan['seeds']:
        recipe = {**base,'seed':seed}; path = output/f'seed{seed}-block128'
        print(f'START seed={seed} block128',flush=True)
        state,seconds = block.train_candidate(data,recipe,plan['settings'],plan['learning_rate'],path,
            identity(plan,c,seed),criterion,plan['recovery_replay_steps'])
        logits = rep.infer(state['visible'],x,alarm_plan['inference_batch_size'])
        np.save(path/'logits.npy',logits,allow_pickle=False)
        r = {'arm':'block128','seed':seed,'reused_discovery':False,'settings':plan['settings'],
             'learning_rate':plan['learning_rate'],'work':state['work'],'wire':state['wire'],'units':state['unit'],
             'training_seconds':seconds,'sample_chain_sha256':state['sample_chain_sha256'],
             'initial_model_sha256':state_digest(training.initial_model(recipe)),
             'delivered_model_sha256':state_digest(state['visible']),'server_target_sha256':state_digest(state['model']),
             'checkpoint_sha256':sha256(path/'last.pt'),'recovery_checkpoint_sha256':sha256(path/'recovery.pt'),
             'logits_sha256':sha256(path/'logits.npy'),
             **alarms.evaluate_scores(logits,y,calibration,plan['false_alert_targets_basis_points'])}
        check_pairing(r,refs,plan)
        atomic_json(r,path/'result.json'); results.append(r)
        m = block.primary(r,plan)
        print(f'END seed={seed}: macro-F1={m["macro_f1"]:.6f} actual FAR={m["benign_false_alert_rate"]:.4%}',flush=True)
    require(contract() == c, 'Source/plan changed during run')
    total = sum(r['work']['backward_examples'] for r in results if not r['reused_discovery'])
    require(total == plan['new_training_backward_examples'], 'New training budget differs')
    summary = {'status':'complete','plan':plan,'contract':c,'results':results,'references':refs,'split':split,
        'assessment':assess(results,refs,plan),'comparisons':comparisons(results,refs,plan),'descriptive':describe(results,refs,plan),
        'new_training_runs':len(plan['seeds']),'new_training_backward_examples':total,'new_forward_examples':len(plan['seeds'])*len(y),
        'runtime':{**runtime(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()},
        'confirmation_opened':False,'original_validation_opened':False,'test_opened':False,'test_evaluated':False,'network_measured':False}
    atomic_json(summary,output/'summary.json')
    return summary


def independent_points(logits,y,calibration,points):
    """Rebuild decisions directly, without the main score/calibration helper."""
    winners = alarms.ATTACKS[logits[:,alarms.ATTACKS].argmax(axis=1)]
    top = logits[np.arange(len(y)),winners].astype(np.float64)
    benign = logits[:,alarms.BENIGN].astype(np.float64)
    ordered = np.sort((top-benign)[calibration & (y == alarms.BENIGN)])
    for point in points:
        c = point['calibration']; allowed = len(ordered)*c['target_basis_points']//10000
        require(c['threshold'] == float(ordered[len(ordered)-allowed-1]), 'Sorted threshold differs')
        predicted = np.where(top > benign+c['threshold'],winners,alarms.BENIGN)
        require(alarms.metrics(y[~calibration],predicted[~calibration]) == point['audit_metrics'], 'Independent metrics differ')


def audit(data_root,output,receipt):
    output,receipt = Path(output),Path(receipt)
    require(not receipt.exists(), 'Preserve existing audit')
    plan,base,alarm_plan,performance,parent,refs = read_contract(require_runtime=True)
    s = json.loads((output/'summary.json').read_text())
    require(s['plan'] == plan and s['contract'] == contract() and s['references'] == refs and
            s['split'] == parent['summary']['split'], 'Audit contract differs')
    discovery = next(r for r in parent['summary']['results'] if r['arm'] == 'block128')
    require([r for r in s['results'] if r['reused_discovery']] == [{**discovery,'reused_discovery':True}],
            'Reused discovery differs')
    new = [r for r in s['results'] if not r['reused_discovery']]
    require(len(new) == len(plan['seeds']) and {r['seed'] for r in new} == set(plan['seeds']), 'Replication endpoints differ')
    data = rep.FitData(data_root,base,alarm_plan['data_manifest_sha256'])
    criterion = build_official_criterion('sqrt_weighted_ce',data.manifest['counts']['fit'],'fixed_train_mean')
    x,y,calibration = development(data_root,alarm_plan,performance,s['split'])
    require(sha256(output/'calibration_mask.npy') == s['split']['calibration_mask_sha256'] and
            np.array_equal(calibration,np.load(output/'calibration_mask.npy',allow_pickle=False)), 'Audit mask differs')
    checks = []
    for r in new:
        recipe = {**base,'seed':r['seed']}; path = output/f'seed{r["seed"]}-block128'
        for name,key in [('last.pt','checkpoint_sha256'),('recovery.pt','recovery_checkpoint_sha256'),('logits.npy','logits_sha256')]:
            require(sha256(path/name) == r[key], 'Saved artifact differs')
        saved,earlier = [torch.load(path/name,map_location='cpu',weights_only=True) for name in ('last.pt','recovery.pt')]
        require(saved['identity'] == earlier['identity'] == identity(plan,s['contract'],r['seed']), 'Checkpoint identity differs')
        state = codec.restore(earlier['state'],recipe,plan['learning_rate'])
        require(state['unit'] == r['units']-plan['recovery_replay_steps'], 'Recovery boundary differs')
        for _ in range(plan['recovery_replay_steps']): codec.advance(state,data,recipe,plan['learning_rate'],criterion)
        require(block.nested_equal(codec.snapshot(state),saved['state']), 'Recovery is not exact')
        block.check_work(state,recipe)
        require(state['work'] == r['work'] and state['wire'] == r['wire'] and state['sample_chain_sha256'] == r['sample_chain_sha256'] and
                state['unit'] == r['units'] == training.work_plan(recipe,'shared_adam')[0], 'Work/samples differ')
        require(state_digest(state['visible']) == r['delivered_model_sha256'] and state_digest(state['model']) == r['server_target_sha256'] and
                state_digest(training.initial_model(recipe)) == r['initial_model_sha256'], 'Model hashes differ')
        check_pairing(r,refs,plan)
        logits = np.load(path/'logits.npy',allow_pickle=False)
        require(np.array_equal(logits,rep.infer(state['visible'],x,alarm_plan['inference_batch_size'])), 'Delivered inference differs')
        evaluated = alarms.evaluate_scores(logits,y,calibration,plan['false_alert_targets_basis_points'])
        require(all(r[k] == v for k,v in evaluated.items()), 'Metrics differ')
        independent_points(logits,y,calibration,r['operating_points'])
        checks.append({'seed':r['seed'],'recovery_exact':True,'checkpoint_inference_exact':True,
                       'same_seed_pairing_verified':True,'work_and_serialized_bytes_verified':True,'operating_points_exact':len(r['operating_points'])})
    require(s['assessment'] == assess(s['results'],refs,plan) and s['comparisons'] == comparisons(s['results'],refs,plan) and
            s['descriptive'] == describe(s['results'],refs,plan), 'Reported comparisons differ')
    require(sum(r['work']['backward_examples'] for r in new) == s['new_training_backward_examples'] == plan['new_training_backward_examples'],
            'Total training work differs')
    recovery_work = len(checks)*plan['recovery_replay_steps']*base['clients']*base['batch_size']
    require(recovery_work == plan['new_recovery_backward_examples'], 'Recovery work differs')
    result = {'status':'complete','summary_sha256':sha256(output/'summary.json'),'source_sha256':sha256(__file__),
              'checks':checks,'new_recovery_backward_examples':recovery_work,'new_forward_examples':len(checks)*len(y),
              'discovery_reused_not_retrained':True,'confirmation_opened':False,'original_validation_opened':False,
              'test_opened':False,'test_evaluated':False}
    atomic_json(result,receipt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','audit'))
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--receipt',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    if args.action == 'run':
        require(args.receipt is None, 'Run takes no audit receipt')
        r = run(args.data,args.output)
    else:
        require(args.receipt is not None, 'Audit requires --receipt')
        r = audit(args.data,args.output,args.receipt)
    print(json.dumps({'status':r['status'],'assessment':r.get('assessment'),'test_opened':r['test_opened']},indent=2))


if __name__ == '__main__':
    main()
