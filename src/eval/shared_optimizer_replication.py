"""Fixed-setting paired training seeds on previously examined development data.

This is training-seed robustness, not a new test set or an algorithm search.
The historical updater and benign-margin calibration modules remain unchanged.
"""
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
from src.data.shared_optimizer_data import ROOT, owner_vector
from src.eval import shared_optimizer_alarms as alarms
from src.eval import shared_optimizer_performance as training
from src.eval.federated_class_safeguard import state_digest
from src.federated.shared_adam import require
from src.models.official_losses import build_official_criterion
from src.models.research import atomic_json, atomic_save

PLAN = ROOT/'reports/federated_shared_optimizer/replication-plan.json'
EVIDENCE = PLAN.parent/'alarm-results.json'


def sources():
    return {**training.sources(), **alarms.source_hashes(),
            'src/eval/shared_optimizer_replication.py': sha256(__file__),
            'src/models/research.py': sha256(ROOT/'src/models/research.py'),
            'src/eval/federated_class_safeguard.py': sha256(ROOT/'src/eval/federated_class_safeguard.py')}


def read_contract():
    plan = json.loads(PLAN.read_text())
    require(sha256(EVIDENCE) == plan['alarm_evidence_sha256'], 'Discovery alarm evidence changed')
    old = json.loads(EVIDENCE.read_text())
    alarm_plan, performance = alarms.read_contract()
    require(old['summary']['plan'] == alarm_plan and old['summary']['sources'] == alarms.source_hashes(),
            'Alarm contract changed')
    require(plan['seeds'] == [17, 27] and plan['discovery_seed'] == 7, 'Unregistered training seeds')
    require(plan['arms'] == ['central_adam', 'shared_adam', 'fedavg_adam'], 'Unregistered arms')
    require(plan['learning_rates'] == {a:alarm_plan['selected_learning_rates'][a] for a in plan['arms']},
            'Selected learning rates changed')
    for key in ('data_manifest_sha256', 'false_alert_targets_basis_points',
                'primary_target_basis_points', 'actual_far_comparability_basis_points'):
        require(plan[key] == alarm_plan[key], f'Alarm setting differs: {key}')
    require(plan['calibration_mask_sha256'] == old['summary']['split']['calibration_mask_sha256'],
            'Calibration split changed')
    base = performance['summary']['plan']
    require(plan['backward_examples_per_run'] == base['backward_examples_per_arm'] and
            plan['new_backward_examples'] == 6*plan['backward_examples_per_run'], 'Budget differs')
    return plan, alarm_plan, performance, old


class FitData:
    """Whitelist training inputs; never even hash reserved confirmation arrays."""
    def __init__(self, root, base, manifest_sha256):
        root = Path(root)
        require(sha256(root/'manifest.json') == manifest_sha256, 'Fit manifest differs')
        self.manifest = json.loads((root/'manifest.json').read_text())
        m = self.manifest
        require(m['status'] == 'complete' and m['classes'] == CLASSES, 'Invalid fit manifest')
        for name in ('fit_x.npy', 'fit_y.npy', 'clients.npz'):
            require(sha256(root/name) == m['files'][name], f'Fit input differs: {name}')
        x, y = [np.load(root/f'fit_{suffix}.npy', mmap_mode='r', allow_pickle=False) for suffix in ('x','y')]
        require(x.shape == (base['fit_rows'], len(m['features'])) and x.dtype == np.float32 and
                y.shape == (base['fit_rows'],) and y.dtype == np.int64, 'Fit shape/dtype differs')
        require(np.bincount(y, minlength=len(CLASSES)).tolist() == [m['counts']['fit'][c] for c in CLASSES],
                'Fit counts differ')
        self.arrays = {'fit': (x,y)}
        with np.load(root/'clients.npz', allow_pickle=False) as z:
            require(set(z.files) == {f'client_{i}' for i in range(base['clients'])}, 'Client keys differ')
            self.parts = [z[f'client_{i}'] for i in range(base['clients'])]
        owner_vector(self.parts, len(y))
        require([len(p) for p in self.parts] == m['fit_client_counts'], 'Ownership counts differ')

    def batch(self, rows):
        return tuple(torch.from_numpy(np.array(a[rows], copy=True)) for a in self.arrays['fit'])


def infer(model, x, batch_size):
    model.eval()
    logits = np.empty((len(x),len(CLASSES)),dtype=np.float32)
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            logits[start:start+batch_size] = model(torch.from_numpy(np.array(x[start:start+batch_size],copy=True))).numpy()
    require(bool(np.isfinite(logits).all()), 'Nonfinite inference')
    return logits


def expected_work(base, arm, parameter_bytes):
    """Independently derive every counter from the declared update schedule."""
    units, _ = training.work_plan(base,arm)
    c,b,k = base['clients'],base['batch_size'],base['mime']['local_steps']
    batches = base['backward_examples_per_arm']//b
    federated = arm != 'central_adam'
    return {'backward_examples':base['backward_examples_per_arm'],'backward_batches':batches,
            'optimizer_steps':units if arm == 'shared_adam' else batches,'anchor_examples':0,
            'local_examples':c*k*b*units if arm == 'fedavg_adam' else 0,
            'synchronization_rounds':units if federated else 0,'logical_barriers':units if federated else 0,
            'estimated_payload_bytes':2*c*parameter_bytes*units if federated else 0}


def train_candidate(data, base, arm, lr, destination, contract, criterion):
    """Persist optimizer state at reproducible boundaries, including final state."""
    destination = Path(destination)
    destination.mkdir(exist_ok=True)
    checkpoint = destination/'last.pt'
    identity = {'contract':contract, 'seed':base['seed'], 'arm':arm, 'learning_rate':lr}
    state, elapsed = training.make_state(base,arm,lr), 0.
    if checkpoint.exists():
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        require(saved['identity'] == identity, 'Resume identity differs')
        state = training.restore(saved['state'],base,arm,lr)
        elapsed = saved['training_seconds']
    units, per_unit = training.work_plan(base,arm)
    require(0 <= state['unit'] <= units and state['work']['backward_examples'] == state['unit']*per_unit,
            'Resume work differs')
    interval = max(1, units//12)
    print(f'START seed={base["seed"]} {arm} at {state["unit"]}/{units}',flush=True)
    started = time.perf_counter()
    while state['unit'] < units:
        training.advance(state,data,base,arm,lr,criterion)
        if state['unit'] % interval == 0 or state['unit'] == units:
            atomic_save({'identity':identity, 'state':training.snapshot(state),
                         'training_seconds':elapsed+time.perf_counter()-started},checkpoint)
            print(f'  seed={base["seed"]} {arm} backward={state["work"]["backward_examples"]:,}',flush=True)
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(state['work'] == expected_work(base,arm,state['model'].parameter_bytes()), 'Final work differs')
    return state, saved['training_seconds']


def paired_comparisons(results, plan):
    output = []
    expected = {(s,a) for s in [plan['discovery_seed'], *plan['seeds']] for a in plan['arms']}
    require(len(results) == len(expected) and {(r['seed'],r['arm']) for r in results} == expected,
            'Missing/duplicate seed-arm endpoint')
    for seed in [plan['discovery_seed'], *plan['seeds']]:
        by_arm = {r['arm']:r for r in results if r['seed'] == seed}
        for target in plan['false_alert_targets_basis_points']:
            points = {a:next(p['audit_metrics'] for p in r['operating_points']
                            if p['calibration']['target_basis_points'] == target) for a,r in by_arm.items()}
            for reference in ('central_adam','fedavg_adam'):
                candidate, baseline = points['shared_adam'],points[reference]
                far_gap = candidate['benign_false_alert_rate']-baseline['benign_false_alert_rate']
                output.append({'seed':seed,'target_basis_points':target,'reference':reference,
                    'shared_minus_reference_macro_f1':candidate['macro_f1']-baseline['macro_f1'],
                    'shared_minus_reference_far':far_gap,
                    'actual_far_comparable':abs(far_gap) <= plan['actual_far_comparability_basis_points']/10000,
                    'per_class_recall_difference':{c:candidate['per_class'][c]['recall']-baseline['per_class'][c]['recall']
                                                   for c in CLASSES}})
    return output


def statistics(values):
    values = np.asarray(values,dtype=np.float64)
    require(len(values) >= 2 and bool(np.isfinite(values).all()), 'Invalid descriptive values')
    return {'mean':float(values.mean()), 'sample_sd':float(values.std(ddof=1)),
            'min':float(values.min()), 'max':float(values.max()), 'n':len(values)}


def describe(results, plan):
    records = []
    for target in plan['false_alert_targets_basis_points']:
        for arm in plan['arms']:
            points = [next(p['audit_metrics'] for p in r['operating_points']
                           if p['calibration']['target_basis_points'] == target) for r in results if r['arm'] == arm]
            records.append({'arm':arm,'target_basis_points':target,
                'macro_f1':statistics([p['macro_f1'] for p in points]),
                'benign_false_alert_rate':statistics([p['benign_false_alert_rate'] for p in points]),
                'per_class_recall':{c:statistics([p['per_class'][c]['recall'] for p in points]) for c in CLASSES}})
    return records


def run(data_root, output):
    plan, alarm_plan, performance, old = read_contract()
    output = Path(output)
    require(not (output/'summary.json').exists(), 'Completed summary exists; preserve it')
    contract = {'plan_sha256':sha256(PLAN), 'sources':sources(), 'data_manifest_sha256':plan['data_manifest_sha256']}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'contract.json').exists():
        require(json.loads((output/'contract.json').read_text()) == contract, 'Run contract differs')
    else:
        require(not list(output.iterdir()), 'Unknown output contents')
        atomic_json(contract,output/'contract.json')
    base = performance['summary']['plan']
    data = FitData(data_root,base,plan['data_manifest_sha256'])
    x,y,_ = alarms.load_development(data_root,alarm_plan,performance)
    calibration,groups = alarms.grouped_split(x,alarm_plan['split_domain'])
    require(all(old['summary']['split'][k] == v for k,v in groups.items()), 'Grouped split differs')
    np.save(output/'calibration_mask.npy',calibration,allow_pickle=False)
    require(sha256(output/'calibration_mask.npy') == plan['calibration_mask_sha256'], 'Split mask differs')
    criterion = build_official_criterion('sqrt_weighted_ce',data.manifest['counts']['fit'],'fixed_train_mean')
    results = []
    for r in old['summary']['results']:
        if r['arm'] in plan['arms']:
            previous = next(p for p in performance['summary']['results']
                            if p['arm'] == r['arm'] and p['learning_rate'] == r['learning_rate'])
            results.append({**r,'seed':plan['discovery_seed'],'reused_discovery':True,
                            **{k:previous[k] for k in ('work','initial_model_sha256','sample_chain_sha256')}})
    for seed in plan['seeds']:
        recipe = {**base,'seed':seed}
        for arm in plan['arms']:
            lr = plan['learning_rates'][arm]
            destination = output/f'seed{seed}-{arm}'
            state,seconds = train_candidate(data,recipe,arm,lr,destination,contract,criterion)
            logits = infer(state['model'],x,alarm_plan['inference_batch_size'])
            np.save(destination/'logits.npy',logits,allow_pickle=False)
            evaluated = alarms.evaluate_scores(logits,y,calibration,plan['false_alert_targets_basis_points'])
            r = {'seed':seed,'arm':arm,'learning_rate':lr,'reused_discovery':False,
                 'checkpoint_sha256':sha256(destination/'last.pt'), 'logits_sha256':sha256(destination/'logits.npy'),
                 'initial_model_sha256':state_digest(training.initial_model(recipe)),
                 'final_model_sha256':state_digest(state['model']), 'sample_chain_sha256':state['sample_chain_sha256'],
                 'work':state['work'],'units':state['unit'],'training_seconds':seconds,**evaluated}
            atomic_json(r,destination/'result.json')
            results.append(r)
            primary = next(p['audit_metrics'] for p in r['operating_points']
                           if p['calibration']['target_basis_points'] == plan['primary_target_basis_points'])
            print(f'END seed={seed} {arm}: macro-F1={primary["macro_f1"]:.6f} FAR={primary["benign_false_alert_rate"]:.4%}',flush=True)
    require(sources() == contract['sources'] and sha256(PLAN) == contract['plan_sha256'], 'Code/plan changed during run')
    total = sum(r['work']['backward_examples'] for r in results if not r['reused_discovery'])
    require(total == plan['new_backward_examples'], 'Total work differs')
    summary = {'status':'complete','contract':contract,'plan':plan,'results':results,
        'split':old['summary']['split'],'comparisons':paired_comparisons(results,plan),'descriptive':describe(results,plan),
        'new_training_runs':6,'new_backward_examples':total,'new_forward_examples':6*len(y),
        'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
                   'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()},
        'previous_screen_overridden':False,'confirmation_opened':False,'original_validation_opened':False,
        'test_opened':False,'test_evaluated':False,'network_measured':False}
    atomic_json(summary,output/'summary.json')
    return summary


def audit(data_root, output, receipt):
    """Separate-process checkpoint inference and sorted/direct-logit reference."""
    output,receipt = Path(output),Path(receipt)
    require(not receipt.exists(), 'Preserve existing audit')
    plan, alarm_plan, performance, old = read_contract()
    summary = json.loads((output/'summary.json').read_text())
    require(summary['plan'] == plan and summary['contract'] == {'plan_sha256':sha256(PLAN),'sources':sources(),
            'data_manifest_sha256':plan['data_manifest_sha256']}, 'Audit contract differs')
    x,y,_ = alarms.load_development(data_root,alarm_plan,performance)
    calibration,groups = alarms.grouped_split(x,alarm_plan['split_domain'])
    require(summary['split'] == old['summary']['split'] and
            all(summary['split'][k] == v for k,v in groups.items()), 'Audit groups differ')
    require(sha256(output/'calibration_mask.npy') == plan['calibration_mask_sha256'] and
            np.array_equal(calibration,np.load(output/'calibration_mask.npy',allow_pickle=False)), 'Audit mask differs')
    checks = []
    for r in summary['results']:
        if r['reused_discovery']:
            reference = next(p for p in old['summary']['results'] if p['arm'] == r['arm'])
            previous = next(p for p in performance['summary']['results']
                            if p['arm'] == r['arm'] and p['learning_rate'] == r['learning_rate'])
            require(r == {**reference,'seed':7,'reused_discovery':True,
                          **{k:previous[k] for k in ('work','initial_model_sha256','sample_chain_sha256')}},
                    'Reused discovery evidence differs')
            continue
        recipe = {**performance['summary']['plan'],'seed':r['seed']}
        destination = output/f'seed{r["seed"]}-{r["arm"]}'
        require(sha256(destination/'last.pt') == r['checkpoint_sha256'] and
                sha256(destination/'logits.npy') == r['logits_sha256'], 'Saved artifacts differ')
        saved = torch.load(destination/'last.pt',map_location='cpu',weights_only=True)
        require(saved['identity'] == {'contract':summary['contract'],'seed':r['seed'],'arm':r['arm'],
                                      'learning_rate':plan['learning_rates'][r['arm']]}, 'Checkpoint identity differs')
        state = training.restore(saved['state'],recipe,r['arm'],r['learning_rate'])
        require(state['unit'] == r['units'] == training.work_plan(recipe,r['arm'])[0] and
                state['work'] == r['work'] == expected_work(recipe,r['arm'],state['model'].parameter_bytes()) and
                state['sample_chain_sha256'] == r['sample_chain_sha256'], 'Endpoint budget differs')
        require(state_digest(state['model']) == r['final_model_sha256'] and
                state_digest(training.initial_model(recipe)) == r['initial_model_sha256'], 'Weight identity differs')
        logits = np.load(destination/'logits.npy',allow_pickle=False)
        require(np.array_equal(infer(state['model'],x,alarm_plan['inference_batch_size']),logits), 'Endpoint inference differs')
        evaluated = alarms.evaluate_scores(logits,y,calibration,plan['false_alert_targets_basis_points'])
        require(all(r[k] == v for k,v in evaluated.items()), 'Endpoint metrics differ')
        winners = alarms.ATTACKS[logits[:,alarms.ATTACKS].argmax(axis=1)]
        top = logits[np.arange(len(y)),winners].astype(np.float64)
        benign = logits[:,alarms.BENIGN].astype(np.float64)
        scores = np.sort((top-benign)[calibration & (y == alarms.BENIGN)])
        for point in r['operating_points']:
            c = point['calibration']
            allowed = len(scores)*c['target_basis_points']//10000
            require(c['threshold'] == float(scores[len(scores)-allowed-1]), 'Sorted threshold differs')
            pred = np.where(top > benign+c['threshold'],winners,alarms.BENIGN)
            require(alarms.metrics(y[~calibration],pred[~calibration]) == point['audit_metrics'], 'Direct-logit replay differs')
            require(int(((pred != alarms.BENIGN) & calibration & (y == alarms.BENIGN)).sum()) ==
                    c['observed_false_alerts'] <= allowed, 'Calibration cap differs')
        checks.append({'seed':r['seed'],'arm':r['arm'],'checkpoint_inference_exact':True,'operating_points_exact':4,
                       'work_verified':True})
    require(len(checks) == 6 and summary['new_backward_examples'] == plan['new_backward_examples'] ==
            sum(r['work']['backward_examples'] for r in summary['results'] if not r['reused_discovery']), 'Total audit work differs')
    require(summary['comparisons'] == paired_comparisons(summary['results'],plan) and
            summary['descriptive'] == describe(summary['results'],plan), 'Summary arithmetic differs')
    result = {'status':'complete','summary_sha256':sha256(output/'summary.json'),'source_sha256':sha256(__file__),
              'checks':checks,'reused_discovery_models_verified':3,'new_forward_examples':len(checks)*len(y),
              'new_backward_examples':0,'confirmation_opened':False,'original_validation_opened':False,
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
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.action == 'run':
        require(args.receipt is None,'Run does not accept an audit receipt')
        result = run(args.data,args.output)
    else:
        require(args.receipt is not None,'Audit needs --receipt')
        result = audit(args.data,args.output,args.receipt)
    print(json.dumps({k:result[k] for k in ('status','new_backward_examples','test_opened')},indent=2))


if __name__ == '__main__':
    main()
