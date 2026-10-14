"""Execute the authorized locked comparison once, then audit saved evidence.

No optimizer, fitting routine, calibration helper, or checkpoint selector is
called. The original project validation/test data are outside this workflow.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.shared_optimizer_data import ROOT
from src.data import shared_optimizer_confirmation as cohort
from src.eval import shared_optimizer_confirmation_prepare as prep
from src.eval import shared_optimizer_block_replication as previous
from src.eval import shared_optimizer_alarms as alarms
from src.eval import shared_optimizer_replication as rep
from src.federated.shared_adam import require
from src.models.research import atomic_json

PLAN=ROOT/'reports/federated_shared_optimizer/confirmation-execution-plan.json'
PREPARATION=PLAN.parent/'confirmation-preparation-results.json'


def sources():
    return {**prep.sources(),'src/data/shared_optimizer_confirmation.py':sha256(cohort.__file__),
            'src/eval/shared_optimizer_confirmation.py':sha256(__file__)}


def read_contract(require_runtime=False):
    execution=json.loads(PLAN.read_text())
    plan,base,alarm,performance,parent=prep.read_contract(require_runtime=require_runtime)
    require(sha256(PREPARATION)==execution['preparation_evidence_sha256'], 'Preparation receipt differs')
    evidence=json.loads(PREPARATION.read_text()); lock,checks=evidence['lock'],evidence['checks']
    require(hashlib.sha256(json.dumps(lock,indent=2).encode()).hexdigest()==execution['lock_sha256']==checks['lock_sha256'],
            'Lock digest differs')
    require(lock['plan']==plan and lock['plan_sha256']==sha256(prep.PLAN) and lock['sources']==prep.sources() and
            lock['models']==prep.registry(parent,plan), 'Locked identities or rules differ')
    require(checks['status']=='ready_for_explicit_confirmation_authorization' and len(checks['checks'])==12 and
            all(c['checkpoint_inference_exact'] and c['historical_endpoint_exact'] for c in checks['checks']),
            'Preparation incomplete')
    require(execution['confirmation_scoring_authorized'] is True and execution['authorization'] and
            all(execution[k] is False for k in ('training_permitted','threshold_refitting_permitted',
                'original_validation_scoring_permitted','test_scoring_permitted','selection_permitted')),
            'Execution authorization differs')
    require(execution['device']==plan['device']=='cpu' and execution['threads']==plan['threads']==2, 'Runtime settings differ')
    if require_runtime: require(lock['runtime']==previous.runtime(), 'Locked runtime differs')
    return execution,lock,base,alarm,performance,parent


def contract():
    return {'execution_plan_sha256':sha256(PLAN),'preparation_evidence_sha256':sha256(PREPARATION),'sources':sources()}


def fixed_scores(logits,y,calibrations):
    """Apply previously chosen thresholds; no labels enter the decisions."""
    score,winners=alarms.margins(logits)
    default=alarms.decisions(score,winners,0.)
    require(np.array_equal(default,logits.argmax(axis=1)), 'Default decision bridge differs')
    points=[]
    for c in calibrations:
        predicted=alarms.decisions(score,winners,c['threshold'])
        points.append({'calibration':copy.deepcopy(c),'confirmation_metrics':alarms.metrics(y,predicted)})
    return {'default_metrics':alarms.metrics(y,default),'operating_points':points}


def independent_metrics(logits,y,calibrations,evaluated):
    """Direct-logit decisions and an independent count-table metric check.

    Audit never calibrates on confirmation labels. Integer class counts are
    computed with boolean comparisons rather than the scoring bincount path.
    """
    winners=alarms.ATTACKS[logits[:,alarms.ATTACKS].argmax(axis=1)]
    records=[(0.,evaluated['default_metrics'])]+[(c['threshold'],p['confirmation_metrics'])
             for c,p in zip(calibrations,evaluated['operating_points'],strict=True)]
    require([p['calibration'] for p in evaluated['operating_points']]==calibrations, 'Threshold changed')
    for threshold,m in records:
        predicted=np.where(logits[np.arange(len(y)),winners].astype(np.float64)>
                           logits[:,alarms.BENIGN].astype(np.float64)+threshold,winners,alarms.BENIGN)
        cm=np.array([[int(((y==i)&(predicted==j)).sum()) for j in range(len(CLASSES))]
                     for i in range(len(CLASSES))],dtype=np.int64)
        require(cm.tolist()==m['confusion_matrix'], 'Independent confusion matrix differs')
        f1s=[]
        for i,name in enumerate(CLASSES):
            tp=int(cm[i,i]); support=int(cm[i].sum()); pp=int(cm[:,i].sum())
            p=tp/pp if pp else 0.; recall=tp/support if support else 0.
            f1=2*tp/(support+pp) if support+pp else 0.; f1s.append(f1)
            require(m['per_class'][name]=={'precision':p,'recall':recall,'f1':f1,'support':support},
                    'Independent per-class metrics differ')
        require(m['macro_f1']==float(np.mean(f1s)) and m['accuracy']==float(cm.trace()/cm.sum()) and
                m['benign_false_alert_rate']==1-cm[alarms.BENIGN,alarms.BENIGN]/cm[alarms.BENIGN].sum(),
                'Independent aggregate metrics differ')
        attack=alarms.ATTACKS
        require(m['attack_detection_recall']==float(1-cm[attack,alarms.BENIGN].sum()/cm[attack].sum()) and
                m['macro_attack_class_recall']==float(np.mean([m['per_class'][CLASSES[i]]['recall'] for i in attack])) and
                m['per_attack_detection_recall']=={CLASSES[i]:float(1-cm[i,alarms.BENIGN]/cm[i].sum()) for i in attack},
                'Independent attack-detection metrics differ')
    return len(records)


def summarize(results,lock,parent):
    require(len(results)==12 and {(r['seed'],r['arm']) for r in results}==
            {(e['seed'],e['arm']) for e in lock['models']}, 'Missing/duplicate confirmation endpoint')
    original=prep.endpoint_records(parent); views=[]
    for r in results:
        # Adapt to the frozen historical reporting API; its audit_metrics key
        # means the evaluated panel. Public results explicitly say confirmation.
        old=original[r['seed'],r['arm']]
        views.append({**old,'operating_points':[{'calibration':p['calibration'],
            'audit_metrics':p['confirmation_metrics']} for p in r['operating_points']]})
    blocks=[r for r in views if r['arm']=='block128']; refs=[r for r in views if r['arm']!='block128']
    oldplan=previous.read_contract()[0]
    a=previous.assess(blocks,refs,oldplan)
    assessment={'all_seeds_pass':a['all_seeds_pass'],'per_seed':a['per_seed'],
                'compression_safeguards_confirmed':a['all_seeds_pass'],'previous_training_screen_overridden':False,
                'next':'Final synthesis; no further tuning on these results'}
    comparisons=previous.comparisons(blocks,refs,lock['plan'])
    for r in comparisons:
        r['actual_far_comparable']=abs(r['block_minus_reference_far'])<=lock['plan']['screen']['absolute_far_difference_from_shared_max']
    return {'assessment':assessment,'comparisons':comparisons,'descriptive':rep.describe(views,lock['plan'])}


def run(data_root,parent,subsets,output,authorized=False,project_root=ROOT):
    require(authorized is True, 'Use explicit --authorize-confirmation after user approval')
    execution,lock,base,alarm,performance,old=read_contract(require_runtime=True)
    output=Path(output); c=contract(); started=time.perf_counter()
    require(not output.exists(), 'Preserve existing final comparison; no implicit repeat')
    # Validate all checkpoint identities before opening any reserved input.
    for entry in lock['models']: prep.load_model(entry,base,project_root)
    output.mkdir(parents=True)
    atomic_json(c,output/'contract.json')
    atomic_json({'status':'reconstructing','confirmation_scoring_authorized':True},output/'status.json')
    receipt=cohort.reconstruct(data_root,parent,subsets,output/'data',lock,base,performance,authorized=True)
    x,y=cohort.load_reconstructed(output/'data',receipt); results=[]
    for e in lock['models']:
        path=output/f'seed{e["seed"]}-{e["arm"]}'; path.mkdir()
        model=prep.load_model(e,base,project_root); before=time.perf_counter()
        logits=rep.infer(model,x,alarm['inference_batch_size']); seconds=time.perf_counter()-before
        np.save(path/'logits.npy',logits,allow_pickle=False)
        evaluated=fixed_scores(logits,y,e['calibrations'])
        independent_metrics(logits,y,e['calibrations'],evaluated)
        r={'seed':e['seed'],'arm':e['arm'],'checkpoint_sha256':e['checkpoint_sha256'],
           'model_sha256':e['model_sha256'],'work':e['training_work'],
           'logits_sha256':sha256(path/'logits.npy'),'inference_seconds':seconds,**evaluated}
        atomic_json(r,path/'result.json'); results.append(r)
        m=next(p['confirmation_metrics'] for p in r['operating_points'] if p['calibration']['target_basis_points']==500)
        print(f'CONFIRMATION seed={e["seed"]} {e["arm"]}: macro-F1={m["macro_f1"]:.6f} actual FAR={m["benign_false_alert_rate"]:.4%}',flush=True)
    require(contract()==c, 'Execution contract changed')
    summary={'status':'complete','execution_plan':execution,'contract':c,'lock_sha256':execution['lock_sha256'],
        'data':receipt,'data_receipt_sha256':sha256(output/'data/manifest.json'),'results':results,
        **summarize(results,lock,old),'forward_examples':len(y)*len(results),'new_training_runs':0,
        'backward_examples':0,'thresholds_refitted':False,'scaler_refitted':False,'confirmation_opened':True,
        'confirmation_scored':True,'original_validation_opened':False,'test_opened':False,'test_evaluated':False,
        'cloud_used':False,'network_measured':False,'elapsed_seconds':time.perf_counter()-started,
        'inference_seconds':sum(r['inference_seconds'] for r in results),'process_peak_rss_bytes':prep.peak_rss(),
        'runtime':{**previous.runtime(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}}
    require(summary['forward_examples']==lock['plan']['planned_confirmation_forward_examples'], 'Forward budget differs')
    atomic_json(summary,output/'summary.json'); atomic_json({'status':'complete'},output/'status.json')
    return summary


def audit(output,destination):
    _,lock,base,_,_,old=read_contract(require_runtime=True)
    output=Path(output); destination=Path(destination)
    require(not destination.exists(), 'Preserve existing audit')
    summary=json.loads((output/'summary.json').read_text())
    require(summary['status']=='complete' and summary['contract']==contract(), 'Summary contract differs')
    require(sha256(output/'data/manifest.json')==summary['data_receipt_sha256'] and
            json.loads((output/'data/manifest.json').read_text())==summary['data'], 'Data receipt differs')
    _,y=cohort.load_reconstructed(output/'data',summary['data'])
    checks=[]
    for e in lock['models']:
        matched=[r for r in summary['results'] if (r['seed'],r['arm'])==(e['seed'],e['arm'])]
        require(len(matched)==1, 'Missing/duplicate audit endpoint')
        r=matched[0]; path=output/f'seed{e["seed"]}-{e["arm"]}'
        require(r==json.loads((path/'result.json').read_text()) and r['checkpoint_sha256']==e['checkpoint_sha256'] and
                r['model_sha256']==e['model_sha256'] and r['work']==e['training_work'] and
                sha256(path/'logits.npy')==r['logits_sha256'], 'Endpoint artifact differs')
        logits=np.load(path/'logits.npy',mmap_mode='r',allow_pickle=False)
        count=independent_metrics(logits,y,e['calibrations'],r)
        checks.append({'seed':e['seed'],'arm':e['arm'],'independent_metric_endpoints':count,
                       'frozen_thresholds_exact':True,'saved_logits_verified':True})
    expected=summarize(summary['results'],lock,old)
    require(all(summary[k]==v for k,v in expected.items()), 'Assessment or comparisons differ')
    require(summary['forward_examples']==lock['plan']['planned_confirmation_forward_examples'] and
            summary['new_training_runs']==summary['backward_examples']==0 and
            all(summary[k] is False for k in ('thresholds_refitted','scaler_refitted','original_validation_opened',
                'test_opened','test_evaluated','cloud_used','network_measured')), 'Work or access boundary differs')
    result={'status':'complete','summary_sha256':sha256(output/'summary.json'),'checks':checks,
        'independent_metric_endpoints':sum(c['independent_metric_endpoints'] for c in checks),
        'additional_model_forward_examples':0,'confirmation_scored':True,'thresholds_refitted':False,
        'original_validation_opened':False,'test_opened':False,'test_evaluated':False}
    atomic_json(result,destination); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='action',required=True)
    r=sub.add_parser('run')
    for name in ('data','parent','subsets','output'): r.add_argument(f'--{name}',type=Path,required=True)
    r.add_argument('--authorize-confirmation',action='store_true')
    a=sub.add_parser('audit'); a.add_argument('--output',type=Path,required=True); a.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    if args.action=='run':
        result=run(args.data,args.parent,args.subsets,args.output,args.authorize_confirmation)
        print(json.dumps(result['assessment'],indent=2))
    else: print(json.dumps(audit(args.output,args.destination),indent=2))


if __name__=='__main__': main()
