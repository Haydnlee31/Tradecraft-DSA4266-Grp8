"""Audit and render bounded extensions from saved results; never trains a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.label_map import CLASSES


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def audit_metric(m):
    """Independently recompute every class and macro-F1 from integer counts."""
    cm=m['confusion_matrix']
    assert len(cm)==len(CLASSES) and all(len(r)==len(CLASSES) for r in cm)
    assert all(type(v) is int and v>=0 for r in cm for v in r)
    f1=[]
    for i,c in enumerate(CLASSES):
        tp=cm[i][i]; support=sum(cm[i]); predicted=sum(row[i] for row in cm)
        p=tp/predicted if predicted else 0.; r=tp/support if support else 0.
        f=2*p*r/(p+r) if p+r else 0.
        for key,value in [('precision',p),('recall',r),('f1',f),('support',support)]:
            assert abs(m['per_class'][c][key]-value)<1e-12, (c,key)
        f1.append(f)
    assert abs(m['macro_f1']-statistics.mean(f1))<1e-12
    benign=CLASSES.index('Benign')
    assert abs(m['benign_false_alert_rate']-(1-cm[benign][benign]/sum(cm[benign])))<1e-12
    return sum(map(sum,cm))


def point(evaluation):
    return next(p['audit_metrics'] for p in evaluation['operating_points'] if p['calibration']['target_basis_points']==500)


def collect(args):
    primary=read(ROOT/'reports/methodology_comparison/results.json')['summary']
    neural=read(args.neural/'summary.json'); ple=read(args.ple/'summary.json'); trees=read(args.trees/'summary.json')
    assert all(r['status']=='complete' for r in (neural,ple,trees))
    assert neural['published_bridge_checked']==6 and neural['published_bridge_all_exact']
    rows=[]
    for r in primary['results']:
        rows.append({'family':'primary','arm':r['arm'],'seed':r['seed'],'evaluation':r['development'],
            'payload_bytes':r['work']['estimated_payload_bytes'],'synchronizations':r['work']['synchronization_rounds'],
            'parameters':5096,'work':r['work'],'evidence':'previously verified primary run'})
    for r in neural['results']:
        rows.append({'family':'neural_fedlc','arm':r['method']+('_fedlc' if r['fedlc'] else '_control'),
            'seed':r['seed'],'evaluation':r['evaluation'],'payload_bytes':sum(r['wire'].values()) if r['wire'] else r['work']['estimated_payload_bytes'],
            'synchronizations':r['units'],'parameters':r['num_parameters'],'work':r['work'],
            'sample_chain_sha256':r['sample_chain_sha256'],'initial_model_sha256':r['initial_model_sha256'],
            'model_sha256':r['model_sha256'],'published_bridge':r.get('published_bridge'),'evidence':'fresh paired run'})
    for r in ple['results']:
        rows.append({'family':'ple','arm':r['arm'],'seed':r['seed'],'evaluation':r['evaluation'],
            'payload_bytes':r['payload_bytes'],'synchronizations':0 if r['arm']=='central_ple' else 2160,
            'parameters':r['parameters'],'work':{'backward_examples':r['backward_examples']},
            'sample_chain_sha256':r['sample_chain_sha256'],'model_sha256':r['model_digest'],
            'historical_exact':r.get('historical_model_and_thresholds_exact'),'evidence':'fresh representation/control run'})
    for r in trees['results']:
        wire=r['full_model_serialized_transfer']
        rows.append({'family':'trees','arm':r['arm'],'seed':r['seed'],'evaluation':r['evaluation'],
            'payload_bytes':wire['uplink_bytes']+wire['downlink_bytes'],'synchronizations':wire['visits'],
            'cached_tree_estimate_bytes':r['historical_cached_tree_payload_estimate_bytes'],
            'model_bytes':r['model_bytes'],'work':{'boosting_rounds':r['boosting_rounds'],'fit_row_visits_per_boosting_round_sum':r['fit_row_visits_per_boosting_round_sum']},
            'historical_exact':r.get('historical_model_exact'),'evidence':'fresh tree run; full-model transfer accounting'})
    replay={family:read(folder/'confirmation-replay.json') for family,folder in [('ple',args.ple),('xgboost',args.trees)]}
    for family, recorded in replay.items():
        historical=read(ROOT/f'reports/federated_{family}/confirmation-results.json')
        for target, expected in historical['all_targets_mean'].items():
            actual=statistics.mean(r['metrics'][target]['macro_f1'] for r in recorded['results'])
            assert abs(actual-expected)<1e-14, (family,target,'historical target mean differs')
    return {'status':'complete','scope':'bounded exploratory extensions; primary selection retained', 'rows':rows,
        'neural_assessment':neural['assessment'],'confirmation_replay':replay,
        'verification':{'neural_control_bridges_exact':6,'ple_model_and_threshold_bridges_exact':sum(r.get('historical_model_and_thresholds_exact',False) for r in ple['results']),
          'tree_model_and_logits_bridges_exact':sum(r.get('historical_model_exact',False) and r.get('historical_development_logits_exact',False) for r in trees['results']),
          'fresh_runs':len(neural['results'])+len(ple['results'])+len(trees['results']),
          'confirmation_endpoints_replayed':sum(len(r['results']) for r in replay.values())},
        'source_receipts':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:sha(p) for p in
            [ROOT/'reports/methodology_comparison/results.json',args.neural/'summary.json',args.ple/'summary.json',args.trees/'summary.json',ROOT/'studies/bounded-extensions/plan.json']}}


def summarize(evidence):
    groups={}; audited=0
    for r in evidence['rows']:
        for p in r['evaluation']['operating_points']:
            assert audit_metric(p['audit_metrics'])==199530
            audited+=1
        groups.setdefault((r['family'],r['arm']),[]).append(r)
    for replay in evidence['confirmation_replay'].values():
        assert replay['status']=='complete' and not replay['thresholds_refitted'] and not replay['new_holdout']
        for r in replay['results']:
            assert r['historical_endpoint_exact']
            for m in r['metrics'].values():
                assert audit_metric(m)==398502
                audited+=1
    stats=[]
    for (family,arm),rows in groups.items():
        assert sorted(r['seed'] for r in rows)==[7,17,27]
        ms=[point(r['evaluation']) for r in rows]
        stats.append({'family':family,'arm':arm,'macro_f1':statistics.mean(m['macro_f1'] for m in ms),
            'macro_f1_sd':statistics.stdev(m['macro_f1'] for m in ms),
            'false_alert_rate':statistics.mean(m['benign_false_alert_rate'] for m in ms),
            'payload_mb':statistics.mean(r['payload_bytes']/1e6 for r in rows),
            'synchronizations':rows[0]['synchronizations'],
            'per_class_recall':{c:statistics.mean(m['per_class'][c]['recall'] for m in ms) for c in CLASSES}})
    return stats,audited


def render(evidence,out):
    out.mkdir(parents=True,exist_ok=True)
    stats,audited=summarize(evidence)
    evidence['verification']['confusion_endpoints_audited']=audited
    (out/'results.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    lines=['# Bounded extensions: development comparison','','All rows below use the same 199,530-row development audit, with thresholds fitted on the other development half. Values are three-seed means. Confirmation is shown separately. Results are exploratory, not a fresh external test.','','| Family | Method | Macro-F1 ± seed SD | Actual FAR | Payload MB | Syncs/visits |','| --- | --- | --- | --- | --- | --- |']
    for s in stats:
        lines.append(f"| {s['family']} | {s['arm']} | {s['macro_f1']:.4f} ± {s['macro_f1_sd']:.4f} | {100*s['false_alert_rate']:.2f}% | {s['payload_mb']:.2f} | {s['synchronizations']} |")
    lines+=['','Neural rows use 21.6M backward examples; larger PLE networks require more computation per example. Tree rows match 200 boosting rounds, not compute: central sees 100M row-visits versus 5M cyclic. Tree payload is full-model UBJSON uplink + downlink; historical cached-tree estimates (~54 MB for FedLC) are preserved in JSON but are not an implemented transport. No numbers here measure real network latency.','','## Class recall','','| Method | '+' | '.join(CLASSES)+' |','| --- | '+' | '.join(['---']*len(CLASSES))+' |']
    for s in stats:
        lines.append('| '+s['arm']+' | '+' | '.join(f"{100*s['per_class_recall'][c]:.1f}%" for c in CLASSES)+' |')
    pairs=[('trees','cyclic_fedlc','trees','cyclic_xgb'),
           ('ple','compressed_ple','primary','block128'),
           ('ple','compressed_ple','ple','compressed_raw_wide'),
           ('ple','central_ple','ple','compressed_ple'),
           ('trees','central_xgb','trees','cyclic_fedlc')]
    by={(r['family'],r['arm'],r['seed']):r for r in evidence['rows']}
    lines+=['','## Paired comparisons at the same seed','','Differences are treatment minus comparator. FAR differences are percentage points, not percentages of the comparator. Tree/central and PLE/raw contrasts retain the work and capacity limitations above.','','| Treatment vs comparator | Mean macro-F1 difference | Largest absolute FAR difference |','| --- | --- | --- |']
    for f,a,g,b in pairs:
        deltas=[];fars=[]
        for seed in (7,17,27):
            x,y=[point(by[(family,arm,seed)]['evaluation']) for family,arm in ((f,a),(g,b))]
            deltas.append(x['macro_f1']-y['macro_f1']);fars.append(abs(x['benign_false_alert_rate']-y['benign_false_alert_rate']))
        lines.append(f'| {a} vs {b} | {statistics.mean(deltas):+.4f} | {100*max(fars):.2f} pp |')
    lines+=['','## Existing confirmation replay','','These are replays of already exposed frozen models and thresholds, not additional model selection. New central PLE controls and neural FedLC are not evaluated on confirmation.','','| Candidate | Macro-F1 | Actual FAR | Exact historical seeds |','| --- | --- | --- | --- |']
    for family,r in evidence['confirmation_replay'].items():
        m=[s['metrics']['500'] for s in r['results']]
        lines.append(f"| {family} | {statistics.mean(s['macro_f1'] for s in m):.4f} | {100*statistics.mean(s['benign_false_alert_rate'] for s in m):.2f}% | 3/3 |")
    lines+=['','## Neural FedLC predeclared assessment','', '```json',json.dumps(evidence['neural_assessment'],indent=2),'```','','## Verification','', '```json',json.dumps(evidence['verification'],indent=2),'```']
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['family','arm','macro_f1','macro_f1_sd','false_alert_rate','payload_mb','synchronizations',*CLASSES]);writer.writeheader()
        for s in stats:
            writer.writerow({**{k:v for k,v in s.items() if k!='per_class_recall'},**s['per_class_recall']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    picked=[s for s in stats if (s['family']=='primary' and s['arm'] in ['central_adam','fedavg_adam','shared_adam','block128']) or (s['family']=='neural_fedlc' and s['arm'].endswith('_fedlc')) or s['family'] in ['ple','trees']]
    fig,ax=plt.subplots(figsize=(11,7))
    for i,s in enumerate(picked):
        ax.errorbar(s['macro_f1'],i,xerr=s['macro_f1_sd'],fmt='o',capsize=3,color={'primary':'#245b8a','neural_fedlc':'#a7631f','ple':'#35836a','trees':'#865b99'}[s['family']])
    ax.set_yticks(range(len(picked)),[s['arm'].replace('_',' ') for s in picked]);ax.invert_yaxis()
    ax.set_xlabel('Development audit macro-F1 (mean ± sample SD, 3 seeds)')
    ax.set_title('Same evaluation rows; separate model and work controls')
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(out/'development-comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(evidence['verification'],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path)
    p.add_argument('--neural',type=Path,default=ROOT/'extensions/fedlc/runs/integration-v1')
    p.add_argument('--ple',type=Path,default=ROOT/'outputs/bounded-extensions-ple-v1')
    p.add_argument('--trees',type=Path,default=ROOT/'outputs/bounded-extensions-trees-v1')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(read(a.evidence) if a.evidence else collect(a),a.output)


if __name__=='__main__':
    main()
