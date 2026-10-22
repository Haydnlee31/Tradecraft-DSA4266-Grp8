"""Replay final synchronization and independently score all stored endpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from extensions.fedntd.data import DevelopmentData, ROOT, sha256
from extensions.sync_intervals import run
from src.data.label_map import CLASSES
from src.eval.federated_class_safeguard import state_digest
from src.federated.shared_adam import require
from src.models.official_losses import build_official_criterion


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return isinstance(b,(tuple,list)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def audit(data_root, run_root, output):
    run_root=Path(run_root)
    summary=json.loads((run_root/'summary.json').read_text())
    contract=summary['contract'];plan=contract['plan']
    run.validate_plan(plan)
    torch.set_num_threads(plan['threads']);torch.use_deterministic_algorithms(True)
    require(summary['status']=='complete' and contract['sources']==run.sources(),'Run/source identity differs')
    require(sha256(data_root/'manifest.json')==contract['data_manifest_sha256'],'Data manifest changed')
    data=DevelopmentData(data_root,plan)
    require(data.file_hashes==contract['files'],'Data files changed')
    criterion=build_official_criterion('sqrt_weighted_ce',data.manifest['counts']['fit'],'fixed_train_mean')
    historical=json.loads((ROOT/'reports/federated_shared_optimizer/performance-results.json').read_text())['summary']['results']
    results=[];replay_examples=0
    for result,arm in zip(summary['results'],plan['arms'],strict=True):
        require(result['arm']==arm and result['contract']==contract,'Arm contract mismatch')
        folder=run_root/arm['name']
        require(sha256(folder/'last.pt')==result['checkpoint_sha256'],'Checkpoint checksum differs')
        require(sha256(folder/'recovery.pt')==result['recovery_sha256'],'Recovery checksum differs')
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        prior=torch.load(folder/'recovery.pt',map_location='cpu',weights_only=True)
        require(last['contract']==prior['contract']==contract and last['arm']==prior['arm']==arm,'Checkpoint contracts differ')
        state=run.restore(prior['state'],plan,arm)
        require(state['round']==result['rounds']-1,'Wrong replay boundary')
        run.advance(state,data,plan,arm,criterion)
        require(equal(run.snapshot(state),last['state']),'Final update/recovery replay differs')
        replay_examples+=plan['clients']*plan['batch_size']*arm['interval']
        require(state['work']==result['work'] and state['client_sample_chains']==result['client_sample_chains'],
                'Result counters/chains differ')
        require(state['work']['backward_examples']==plan['backward_examples_per_arm'],'Budget differs')
        expected_rounds=plan['backward_examples_per_arm']//(plan['clients']*plan['batch_size']*arm['interval'])
        require(state['round']==expected_rounds==state['work']['logical_barriers'],'Barrier accounting differs')
        require(state['work']['estimated_payload_bytes']==2*plan['clients']*state['model'].parameter_bytes()*expected_rounds,
                'Payload accounting differs')
        model=state['model'].eval(); x,y=data.arrays['development'];pred=[]
        with torch.no_grad():
            for start in range(0,len(y),8192):
                logits=model(torch.from_numpy(np.array(x[start:start+8192],copy=True)))
                require(bool(torch.isfinite(logits).all()),'Nonfinite logits')
                pred.append(logits.argmax(1).numpy())
        pred=np.concatenate(pred)
        cm=confusion_matrix(y,pred,labels=np.arange(len(CLASSES)))
        p,r,f,s=precision_recall_fscore_support(y,pred,labels=np.arange(len(CLASSES)),zero_division=0)
        metrics=result['development_metrics']
        require(cm.tolist()==metrics['confusion_matrix'] and abs(float(f.mean())-metrics['macro_f1'])<1e-14,
                'Independent metrics differ')
        for i,c in enumerate(CLASSES):
            for key,value in zip(('precision','recall','f1','support'),(p[i],r[i],f[i],s[i])):
                require(abs(float(value)-metrics['per_class'][c][key])<1e-14,'Class metrics differ')
        benign=CLASSES.index('Benign')
        require(abs(float((pred[y==benign]!=benign).mean())-metrics['benign_false_alert_rate'])<1e-14,'FAR differs')
        bridge=None
        if (arm['kind']=='interval' and arm['interval']==1) or arm['kind']=='fedavg_adam':
            name='shared_adam' if arm['kind']=='interval' else 'fedavg_adam'
            expected=next(r for r in historical if r['arm']==name and r['learning_rate']==.003)
            require({k:v for k,v in metrics.items() if k!='decision_rule'}==expected['development_metrics'],
                    'Historical reference metrics differ')
            require(state_digest(model)==expected['final_model_sha256'],'Historical reference weights differ')
            require(state['traversal_sample_chain']==expected['sample_chain_sha256'],'Historical reference samples differ')
            bridge=name
        results.append({k:v for k,v in result.items() if k!='contract'} |
                       {'recovery_bit_exact':True,'independent_metrics_match':True,'historical_exact_bridge':bridge})
    require(all(r['client_sample_chains']==results[0]['client_sample_chains'] for r in results),'Sample streams differ')
    require(run.comparisons(results,plan)==summary['comparisons_to_k1'],'Comparison gates differ')
    receipt={'status':'verified','plan':plan,'runtime':contract['runtime'],
             'data_manifest_sha256':contract['data_manifest_sha256'],
             'summary_sha256':sha256(run_root/'summary.json'),'audit_source_sha256':sha256(__file__),
             'results':results,'comparisons_to_k1':summary['comparisons_to_k1'],
             'all_client_streams_identical':True,'audit_backward_examples':replay_examples,
             'audit_inference_examples':len(results)*len(y),'fit_rows':len(data.arrays['fit'][1]),'development_rows':len(y),
             'test_opened':False,'confirmation_scored':False,'scope':'Single-seed exploratory development argmax screen'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():require(json.loads(output.read_text())==receipt,'Existing audit differs; choose new path')
    else:run.atomic_write(output,receipt)
    print(json.dumps({'status':'verified','arms':[{'name':r['arm']['name'],
          'macro_f1':r['development_metrics']['macro_f1'],'far':r['development_metrics']['benign_false_alert_rate'],
          'rounds':r['rounds']} for r in results],'comparisons':receipt['comparisons_to_k1']},indent=2))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    audit(a.data,a.run,a.output)
