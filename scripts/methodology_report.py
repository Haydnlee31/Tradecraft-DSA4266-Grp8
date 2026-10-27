"""Audit and summarize the matched supplement; never train or select a model."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from src.data.label_map import CLASSES
from src.eval import methodology_comparison as study
from src.eval import shared_optimizer_reproduce as old
from src.models.research import atomic_json

LABELS = {
    'central_adam': 'Central Adam', 'fedavg_adam': 'FedAvg-Adam',
    'fedprox001': 'FedProx (mu=0.01)', 'fedprox01': 'FedProx (mu=0.1)',
    'central_sgd': 'Central SGD', 'fedavg_sgd': 'FedAvg-SGD', 'scaffold': 'SCAFFOLD-SGD',
    'central_mime_base': 'Central MIME base', 'mime': 'Full MIME',
    'shared_adam': 'Shared-gradient Adam', 'block128': 'Compressed shared Adam',
}


def at_target(row, split):
    return next(p['audit_metrics' if split == 'development' else 'confirmation_metrics']
                for p in row[split]['operating_points'] if p['calibration']['target_basis_points'] == 500)


def audit_counts(m):
    cm = np.array(m['confusion_matrix'], dtype=np.int64)
    old.require(cm.shape == (8,8) and bool((cm >= 0).all()), 'Invalid confusion counts')
    fs = []
    for i, name in enumerate(CLASSES):
        tp, support, predicted = int(cm[i,i]), int(cm[i].sum()), int(cm[:,i].sum())
        values = {'precision': tp/predicted if predicted else 0.,
                  'recall': tp/support if support else 0.,
                  'f1': 2*tp/(support+predicted) if support+predicted else 0., 'support': support}
        old.require(values == m['per_class'][name], 'Class metric/count inconsistency')
        fs.append(values['f1'])
    benign = CLASSES.index('Benign')
    old.require(float(np.mean(fs)) == m['macro_f1'] and
                float(cm.trace()/cm.sum()) == m['accuracy'] and
                1-cm[benign,benign]/cm[benign].sum() == m['benign_false_alert_rate'], 'Aggregate metric differs')
    if 'attack_detection_recall' in m:
        attacks = [i for i in range(8) if i != benign]
        old.require(m['attack_detection_recall'] == float(1-cm[attacks,benign].sum()/cm[attacks].sum()) and
                    m['macro_attack_class_recall'] == float(np.mean([m['per_class'][CLASSES[i]]['recall'] for i in attacks])) and
                    m['per_attack_detection_recall'] == {CLASSES[i]:float(1-cm[i,benign]/cm[i].sum()) for i in attacks},
                    'Attack detection metric differs')


def summarize(summary):
    plan = study.read_plan()
    old.require(summary['status'] == 'complete' and summary['plan'] == plan and
                summary['contract'] == study.contract(), 'Incomplete or changed protocol')
    rows = summary['results']
    old.require(len(rows) == 33 and {(r['arm'],r['seed']) for r in rows} ==
                {(a,s) for a in plan['arms'] for s in plan['seeds']}, 'Missing/duplicate endpoint')
    old.require(sum(r['work']['backward_examples'] for r in rows) == 712800000, 'Work budget differs')
    old.require(all(r['recovery_exact'] for r in rows), 'Recovery incomplete')
    checked = 0
    for row in rows:
        for split in ('development','confirmation'):
            metrics = [p['audit_metrics' if split == 'development' else 'confirmation_metrics']
                       for p in row[split]['operating_points']]
            metrics += [row[split]['audit_default_metrics' if split == 'development' else 'default_metrics']]
            if split == 'development':
                metrics.append(row[split]['full_development_default_metrics'])
            for metric in metrics:
                audit_counts(metric)
                checked += 1
    means = []
    for arm in plan['arms']:
        group = [r for r in rows if r['arm'] == arm]
        result = {'arm': arm, 'label': LABELS[arm], 'seeds': plan['seeds']}
        for split in ('development', 'confirmation'):
            ms = [at_target(r, split) for r in group]
            result[split] = {'macro_f1_mean': float(np.mean([m['macro_f1'] for m in ms])),
                            'macro_f1_sample_sd': float(np.std([m['macro_f1'] for m in ms], ddof=1)),
                            'benign_false_alert_rate_mean': float(np.mean([m['benign_false_alert_rate'] for m in ms])),
                            'per_class_recall_mean': {c: float(np.mean([m['per_class'][c]['recall'] for m in ms])) for c in CLASSES}}
        result.update(payload_bytes=group[0]['work']['estimated_payload_bytes'],
                      synchronizations=group[0]['work']['synchronization_rounds'],
                      logical_barriers=group[0]['work']['logical_barriers'],
                      backward_examples_per_run=group[0]['work']['backward_examples'],
                      local_training_seconds_mean=float(np.mean([r['training_seconds'] for r in group])))
        means.append(result)
    # Recheck the four historical controls without assuming wrapper equivalence.
    historical = old.published_reference()['results']
    bridges = []
    for reference in historical:
        row = next(r for r in rows if r['seed'] == reference['seed'] and r['arm'] == reference['arm'])
        bridges.append({'seed': row['seed'], 'arm': row['arm'],
                        'model_exact': row['model_sha256'] == reference['model_sha256'],
                        'logits_exact': row['confirmation_logits_sha256'] == reference['logits_sha256'],
                        'metrics_exact': all(row['confirmation'][k] == reference[k] for k in ('default_metrics','operating_points'))})
    pairs = []
    # Evaluate realized FAR pairing per seed, not merely after averaging.
    references = {'fedprox001':'fedavg_adam', 'fedprox01':'fedavg_adam',
                  'scaffold':'fedavg_sgd', 'mime':'central_mime_base',
                  'shared_adam':'fedavg_adam', 'block128':'shared_adam'}
    for seed in plan['seeds']:
        by_arm = {r['arm']: r for r in rows if r['seed'] == seed}
        for arm, reference in references.items():
            for split in ('development','confirmation'):
                a, b = at_target(by_arm[arm],split), at_target(by_arm[reference],split)
                pairs.append({'seed': seed, 'split': split, 'arm': arm, 'reference': reference,
                              'macro_f1_difference': a['macro_f1']-b['macro_f1'],
                              'far_difference': a['benign_false_alert_rate']-b['benign_false_alert_rate'],
                              'far_within_0_5pp': abs(a['benign_false_alert_rate']-b['benign_false_alert_rate']) <= .005})
    return {'status': 'complete', 'means': means, 'paired_comparisons': pairs,
            'historical_bridges': bridges, 'confusion_endpoints_checked': checked,
            'total_training_backward_examples': 712800000,
            'additional_recovery_backward_examples': sum(r['recovery_backward_examples'] for r in rows),
            'interpretation': 'Retrospective matched supplement on already evaluated confirmation; not a fresh holdout'}


def figures(analysis, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(11,6))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')
    def box(x,y,text,color='#E9F2F3'):
        ax.text(x,y,text,ha='center',va='center',fontsize=11,
                bbox={'boxstyle':'round,pad=0.7','facecolor':color,'edgecolor':'#46717C'})
    def arrow(a,b):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.5,'color':'#46717C'})
    box(.5,.84,'Historical training view\n2,000,000 raw unique vectors')
    box(.17,.52,'Fitting: 500,000\nFit scaler, class weights\nand model parameters')
    box(.5,.52,'Development: 399,552\nPreviously used for\ndevelopment selection')
    box(.83,.52,'Confirmation: 398,502\nFrozen-threshold evaluation\nAlready evaluated previously')
    for target in (.17,.5,.83): arrow((.5,.75),(target,.63))
    box(.37,.19,'Calibration: 200,022\nThreshold uses 10,562\nbenign calibration rows','#F1F3F7')
    box(.64,.19,'Audit: 199,530\nReport development\noperating-point metrics','#F1F3F7')
    arrow((.5,.4),(.37,.3))
    arrow((.5,.4),(.64,.3))
    fig.suptitle('Data boundaries for the shared-optimizer comparison',fontsize=15)
    fig.text(.5,.035,'Hash-based membership; fit-only scaling; cross-cohort float32 matches removed.\n700,146 fit-eligible rows unused; 674 development and 1,126 confirmation collisions excluded.',
             ha='center',fontsize=9,color='#455968')
    fig.tight_layout(rect=(0,.08,1,.95))
    fig.savefig(output/'data-boundaries.png',dpi=180)
    fig.savefig(output/'data-boundaries.svg')
    plt.close(fig)
    rows = analysis['means']
    labels = [r['label'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13,6), gridspec_kw={'width_ratios':[1.3,1]})
    y = np.arange(len(rows))
    for split, offset, color in [('development',-.12,'#6B7C93'),('confirmation',.12,'#087E8B')]:
        axes[0].errorbar([r[split]['macro_f1_mean'] for r in rows], y+offset,
                         xerr=[r[split]['macro_f1_sample_sd'] for r in rows], fmt='o',
                         color=color, capsize=3, label=split)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Macro-F1 (mean and sample SD across three seeds)')
    axes[0].legend(loc='lower left')
    axes[0].grid(axis='x',alpha=.2)
    axes[1].barh(y, [100*r['confirmation']['benign_false_alert_rate_mean'] for r in rows], color='#087E8B')
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].axvline(5,color='#C74B50',ls='--',label='5% calibration target')
    axes[1].set_xlabel('Actual confirmation benign false alerts (%)')
    axes[1].legend(loc='lower right')
    fig.suptitle('Matched methods: fixed work, architecture and cohorts\nRetrospective supplement; calibration on development only', fontsize=13)
    fig.tight_layout()
    fig.savefig(output/'quality.png',dpi=180)
    fig.savefig(output/'quality.svg')
    plt.close(fig)
    recall=np.array([[100*r['confirmation']['per_class_recall_mean'][c] for c in CLASSES] for r in rows])
    fig,ax=plt.subplots(figsize=(12,6.5))
    im=ax.imshow(recall,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    ax.set_yticks(y,labels)
    ax.set_xticks(np.arange(len(CLASSES)),CLASSES)
    for i in y:
        for j in range(len(CLASSES)):
            ax.text(j,i,f'{recall[i,j]:.1f}',ha='center',va='center',color='white' if recall[i,j]>65 else '#152D3B')
    ax.set_title('Correct-category recall on previously evaluated confirmation (%)\nMeans across seeds 7, 17 and 27; 5% development-calibration target')
    fig.colorbar(im,ax=ax,label='Recall (%)')
    fig.tight_layout()
    fig.savefig(output/'recall.png',dpi=180)
    fig.savefig(output/'recall.svg')
    plt.close(fig)
    # Matplotlib emits trailing spaces in path coordinates; normalize only those
    # line endings so the exported vectors also pass the repository whitespace check.
    for name in ('data-boundaries.svg', 'quality.svg', 'recall.svg'):
        path = output/name
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def write_report(summary, analysis, output):
    lines = ['# Matched methodology comparison results', '',
        'This is a retrospective comparison, not a new untouched test. All 33 models were trained from scratch on CPU.', '',
        'Primary scores use the same 5% development-calibration false-alert target. Thresholds were saved before confirmation scoring.', '',
        '| Method | Development audit F1 | Confirmation F1 (mean +/- SD) | Confirmation FAR | Payload MB | Rounds (barriers) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in analysis['means']:
        d,c=r['development'],r['confirmation']
        lines.append(f"| {r['label']} | {d['macro_f1_mean']:.4f} | {c['macro_f1_mean']:.4f} +/- {c['macro_f1_sample_sd']:.4f} | {100*c['benign_false_alert_rate_mean']:.2f}% | {r['payload_bytes']/1e6:.2f} | {r['synchronizations']:,} ({r['logical_barriers']:,}) |")
    lines += ['', 'Equal backward-example budgets do not imply equal communication, FLOPs, optimizer calls or training time. FP32 payloads use raw model-vector accounting; compressed payloads include serializer metadata and initialization.', '',
              '![Quality and false alerts](quality.png)', '', '## Per-class recall', '',
              '| Method | ' + ' | '.join(CLASSES) + ' |', '| --- | ' + ' | '.join(['---:']*8) + ' |']
    for r in analysis['means']:
        lines.append('| '+r['label']+' | '+' | '.join(f"{100*r['confirmation']['per_class_recall_mean'][c]:.2f}%" for c in CLASSES)+' |')
    lines += ['', '![Per-class recall](recall.png)', '', '## Audit and limitations', '',
              f"All {analysis['confusion_endpoints_checked']} confusion-matrix endpoints independently reproduce class and aggregate metrics. Every run passed serialized recovery.", '',
              f"Historical controls reproduced exactly: {sum(all(b[k] for k in ('model_exact','logits_exact','metrics_exact')) for b in analysis['historical_bridges'])}/{len(analysis['historical_bridges'])} model/logit/metric endpoints. Paired operating-point comparisons within 0.5 percentage points of actual FAR: {sum(p['far_within_0_5pp'] for p in analysis['paired_comparisons'])}/{len(analysis['paired_comparisons'])}.", '',
              f"Training processed {analysis['total_training_backward_examples']:,} backward examples, plus {analysis['additional_recovery_backward_examples']:,} separately counted recovery examples.", '',
              'SCAFFOLD must be interpreted against FedAvg-SGD and Central SGD; MIME has its own central-base control. Learning rates were fixed from historical development/engineering choices, not exhaustively optimized for each method.', '',
              'The confirmation rows, client partition and training seeds were already used in the project. This supplement cannot establish external generalization or a new independent confirmation.', '',
              'Full results and per-seed paired false-alert checks: [results.json](results.json). Reproduction instructions and implementation inventory: [README](README.md).', '']
    (output/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    with (output/'per-seed-metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['arm','seed','split','macro_f1','benign_false_alert_rate']+[c+'_recall' for c in CLASSES])
        for r in summary['results']:
            for split in ('development','confirmation'):
                m=at_target(r,split)
                writer.writerow([r['arm'],r['seed'],split,m['macro_f1'],m['benign_false_alert_rate']]+[m['per_class'][c]['recall'] for c in CLASSES])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.run/'summary.json'
    summary=json.loads(source.read_text())
    analysis=summarize(summary)
    args.output.mkdir(parents=True,exist_ok=True)
    old.require(not (args.output/'results.json').exists(), 'Preserve existing published results')
    atomic_json({'source_summary_sha256':old.sha256(source), 'summary':summary, 'analysis':analysis},args.output/'results.json')
    figures(analysis,args.output)
    write_report(summary,analysis,args.output)
    print(json.dumps(analysis,indent=2))


if __name__ == '__main__':
    main()
