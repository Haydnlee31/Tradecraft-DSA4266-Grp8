"""Check complete equal-work evidence and render development-only results."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np
from src.data.label_map import CLASSES
from src.eval import mimelite_comparison as study
from scripts.methodology_report import audit_counts
from src.models.research import atomic_json


def primary(r):
    return next(p['audit_metrics'] for p in r['development']['operating_points']
                if p['calibration']['target_basis_points']==500)


def expected_work(arm,settings,budget):
    base=json.loads(study.old.training.PLAN.read_text())
    ks=study.schedule(base,arm,settings,budget)
    counts=json.loads((ROOT/'reports/federated_shared_optimizer/performance-results.json').read_text())['summary']['data']['fit_client_counts']
    c,b,n=base['clients'],base['batch_size'],base['fit_rows']
    # Dense weights/biases plus LayerNorm scales/biases in the pinned network.
    dims=[39]+base['architecture']['hidden_dims']+[len(CLASSES)]
    size=4*(sum(a*z+z for a,z in zip(dims,dims[1:]))+2*sum(dims[1:-1]))
    central=arm.startswith('central_'); anchored=arm=='mime' or study.is_lite(arm)
    units,local_steps=len(ks),sum(ks)
    anchor=units*n if anchored else 0
    local=0 if central or arm=='shared_adam' else local_steps*c*b
    batches=units if central else units*c if arm=='shared_adam' else local_steps*c*(2 if arm=='mime' else 1)
    if anchored:
        batches+=units*sum((v+base['anchor_batch_size']-1)//base['anchor_batch_size'] for v in counts)
    return {'backward_examples':budget,'backward_batches':batches,
            'optimizer_steps':units if central or arm=='shared_adam' else local_steps*c,
            'anchor_examples':anchor,'local_examples':local,
            'synchronization_rounds':0 if central else units,
            'logical_barriers':0 if central else units*(2 if arm=='mime' else 1),
            'estimated_payload_bytes':0 if central else units*c*size*(6 if arm=='mime' else 5 if study.is_lite(arm) else 2)}


def analyze(s):
    plan=study.read_plan();require=study.old.require
    require(s['status']=='complete' and s['plan']==plan and s['contract']==study.contract(),'Changed/incomplete study')
    require(not any(s[k] for k in ('confirmation_evaluated','original_validation_evaluated','test_evaluated')),'Evaluation boundary changed')
    rows=s['results']
    require(len(rows)==21 and {(r['seed'],r['arm']) for r in rows}==
            {(seed,a) for seed in plan['seeds'] for a in plan['arms']},'Missing or duplicate endpoint')
    endpoints=bridges=0
    base=json.loads(study.old.training.PLAN.read_text())
    for r in rows:
        settings=plan['arms'][r['arm']]
        require(r['job']=={'contract':s['contract'],'seed':r['seed'],'arm':r['arm'],
                          'settings':settings,'schedule':study.schedule(base,r['arm'],settings,plan['backward_examples_per_run'])},
                'Per-run settings or schedule differ')
        require(r['recovery_exact'],'Missing recovery')
        require(r['work']==expected_work(r['arm'],plan['arms'][r['arm']],plan['backward_examples_per_run']),'Work mismatch')
        if not study.is_lite(r['arm']):
            require(set(r['historical_bridge'])=={'model_exact','logits_exact','metrics_exact','work_exact','samples_exact'} and
                    all(r['historical_bridge'].values()),'Historical bridge failed')
            bridges+=1
        ms=[r['development']['full_development_default_metrics'],r['development']['audit_default_metrics']]
        require([p['calibration']['target_basis_points'] for p in r['development']['operating_points']]==[100,500,1000,2000],'Operating points differ')
        ms += [p['audit_metrics'] for p in r['development']['operating_points']]
        for m in ms:
            audit_counts(m);endpoints+=1
    require(sum(r['work']['backward_examples'] for r in rows)==485100000,'Total work differs')
    for seed in plan['seeds']:
        require(len({r['initial_model_sha256'] for r in rows if r['seed']==seed})==1,'Initialization not paired')
    means=[]
    for arm in plan['arms']:
        rs=[r for r in rows if r['arm']==arm];ms=[primary(r) for r in rs]
        means.append({'arm':arm,'label':study.LABELS[arm],
            'macro_f1_mean':float(np.mean([m['macro_f1'] for m in ms])),
            'macro_f1_sample_sd':float(np.std([m['macro_f1'] for m in ms],ddof=1)),
            'false_alert_rate_mean':float(np.mean([m['benign_false_alert_rate'] for m in ms])),
            'recall_mean':{c:float(np.mean([m['per_class'][c]['recall'] for m in ms])) for c in CLASSES},
            'work':rs[0]['work'],'training_seconds_mean':float(np.mean([r['training_seconds'] for r in rs]))})
    pairs=[]
    for seed in plan['seeds']:
        by={r['arm']:primary(r) for r in rows if r['seed']==seed}
        for arm in ('mimelite5','mimelite20','mime','shared_adam'):
            for ref in ('central_adam','fedavg_adam','central_mime_base'):
                far=by[arm]['benign_false_alert_rate']-by[ref]['benign_false_alert_rate']
                pairs.append({'seed':seed,'arm':arm,'reference':ref,
                    'macro_f1_difference':by[arm]['macro_f1']-by[ref]['macro_f1'],
                    'false_alert_difference':far,'far_within_0_5pp':abs(far)<=.005})
    return {'means':means,'paired_comparisons':pairs,'confusion_endpoints_checked':endpoints,
            'historical_bridges_exact':bridges,'recovery_checks':len(rows),
            'training_backward_examples':485100000,
            'additional_recovery_backward_examples':sum(r['recovery_backward_examples'] for r in rows)}


def figures(a,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    rows=a['means'];y=np.arange(len(rows));labels=[r['label'] for r in rows]
    fig,axs=plt.subplots(1,2,figsize=(12,5.5),gridspec_kw={'width_ratios':[1.3,1]})
    axs[0].errorbar([r['macro_f1_mean'] for r in rows],y,xerr=[r['macro_f1_sample_sd'] for r in rows],fmt='o',capsize=4,color='#087E8B')
    axs[0].set_yticks(y,labels);axs[0].set_xlabel('Development-audit macro-F1 (mean and sample SD)')
    axs[1].barh(y,[100*r['false_alert_rate_mean'] for r in rows],color='#087E8B')
    axs[1].set_yticks(y,[]);axs[1].axvline(5,ls='--',color='#C74B50');axs[1].set_xlabel('Actual benign false alerts (%)')
    for ax in axs: ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('Same 23.1M backward examples per run; all anchor work included\nThree training seeds; 5% development-calibration target; no test evaluation')
    fig.tight_layout();fig.savefig(out/'quality.png',dpi=180);fig.savefig(out/'quality.svg');plt.close(fig)
    recall=np.array([[r['recall_mean'][c]*100 for c in CLASSES] for r in rows])
    fig,ax=plt.subplots(figsize=(11.5,5.2));im=ax.imshow(recall,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    ax.set_yticks(y,labels);ax.set_xticks(np.arange(8),CLASSES)
    for i in y:
        for j in range(8):ax.text(j,i,f'{recall[i,j]:.1f}',ha='center',va='center',color='white' if recall[i,j]>65 else '#173246')
    ax.set_title('Correct-category recall on development audit (%)\nMeans of seeds 7, 17 and 27; previously examined development data')
    fig.colorbar(im,ax=ax,label='Recall (%)');fig.tight_layout();fig.savefig(out/'recall.png',dpi=180);fig.savefig(out/'recall.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.5,5))
    anchor=np.array([r['work']['anchor_examples']/1e6 for r in rows]);ordinary=23.1-anchor
    ax.barh(y,ordinary,label='Local / pooled gradient work',color='#087E8B')
    ax.barh(y,anchor,left=ordinary,label='Full-anchor gradient work',color='#91B5C9')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Backward examples (millions)')
    ax.set_title('Equal total work, different allocation\nFull MIME local work includes both correction gradients')
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.3),ncol=2,frameon=False)
    fig.tight_layout();fig.savefig(out/'work.png',dpi=180);fig.savefig(out/'work.svg');plt.close(fig)
    for f in out.glob('*.svg'):f.write_text('\n'.join(line.rstrip() for line in f.read_text().splitlines())+'\n')


def publish(s,a,out):
    lines=['# Full-anchor MimeLite: fair-work development comparison','',
      'These are new final endpoints at **23.1 million backward examples per run**, including all anchors. They are not attempts to reproduce the untraceable MimeLite scores or replacements for the historical confirmation results.','',
      'Means and sample SD use seeds 7, 17 and 27. Thresholds use the same 5% benign calibration target; actual audit false alerts are reported. All evaluation is on previously examined development data.','',
      '| Method | Macro-F1 mean +/- SD | Actual FAR | Rounds | Barriers | Payload MB |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in a['means']:
        w=r['work'];lines.append(f"| {r['label']} | {r['macro_f1_mean']:.4f} +/- {r['macro_f1_sample_sd']:.4f} | {100*r['false_alert_rate_mean']:.2f}% | {w['synchronization_rounds']:,} | {w['logical_barriers']:,} | {w['estimated_payload_bytes']/1e6:.2f} |")
    lines += ['', 'Payload is modeled FP32 application traffic, not a physical network measurement. One MB is 1,000,000 bytes. Equal backward examples do not mean equal FLOPs, elapsed time, communication or tuning effort.','',
              '![Quality and false alerts](quality.png)','', '## Work allocation','',
              '![Work including full anchors](work.png)','', '## Per-class recall','',
              '| Method | '+' | '.join(CLASSES)+' |','| --- | '+' | '.join(['---:']*8)+' |']
    for r in a['means']:lines.append('| '+r['label']+' | '+' | '.join(f"{100*r['recall_mean'][c]:.2f}%" for c in CLASSES)+' |')
    lines += ['', '![Recall](recall.png)','', '## Validation','',
              f"All {a['recovery_checks']} runs passed exact serialized recovery, including shortened final rounds. All {a['historical_bridges_exact']} historical model/logit/metric/work/sample bridges matched exactly at 21.6M before continuing.",'',
              f"Independently checked {a['confusion_endpoints_checked']} confusion-matrix endpoints. Training processed {a['training_backward_examples']:,} backward examples; recovery checks separately processed {a['additional_recovery_backward_examples']:,}.",'',
              f"Paired comparisons within 0.5 percentage points of actual FAR: {sum(p['far_within_0_5pp'] for p in a['paired_comparisons'])}/{len(a['paired_comparisons'])}. All comparisons, including any failures, are retained in results.json.",'',
              'These fixed settings were declared before training. They do not establish optimal hyperparameters, convergence, external generalization or privacy. Confirmation and original validation/test arrays were not evaluated.','',
              '[Protocol and commands](README.md) | [Full numerical evidence](results.json) | [Per-seed metrics](per-seed-metrics.csv)','']
    (out/'results.md').write_text('\n'.join(lines))
    with (out/'per-seed-metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['arm','seed','macro_f1','benign_false_alert_rate']+[c+'_recall' for c in CLASSES])
        for r in s['results']:
            m=primary(r);writer.writerow([r['arm'],r['seed'],m['macro_f1'],m['benign_false_alert_rate']]+[m['per_class'][c]['recall'] for c in CLASSES])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();source=args.run/'summary.json';s=json.loads(source.read_text());a=analyze(s)
    args.output.mkdir(parents=True,exist_ok=True)
    study.old.require(not (args.output/'results.json').exists(),'Preserve existing report')
    # Full schedules remain in raw outputs; publish their compact lengths/final
    # blocks rather than hundreds of thousands of repeated one-step entries.
    evidence=copy.deepcopy(s)
    for r in evidence['results']:
        ks=r['job'].pop('schedule');r['job']['schedule_description']={'units':len(ks),'last_local_steps':ks[-1]}
    atomic_json({'source_summary_sha256':study.old.sha256(source),'summary':evidence,'analysis':a},args.output/'results.json')
    figures(a,args.output);publish(s,a,args.output)
    print(json.dumps(a,indent=2))


if __name__=='__main__':main()
