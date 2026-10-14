"""Assemble the integrated study without retraining or opening traffic arrays.

The optional history export reads checksummed result JSON, not checkpoints.
Normal rebuilds use only committed evidence and the portable history export.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from scripts import shared_optimizer_thesis_figures as shared
from src.data.label_map import CLASSES

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'reports/thesis'
LEGACY = 'reports/confirmation_2026_10_05/results/summary.json'
EXPLAIN = 'reports/explanations_2026_10_05/results/summary.json'
HISTORY = 'reports/thesis/evidence/legacy-histories.json'
HISTORY_SHA256 = '852876411642e04171cd76cdb701596eab3aa1c805d279e822837167d01c57dd'
PINS = {
    LEGACY: '5abb339532f682118c088371d06ac13f78c3de260e228a47aa98ce6237d1ce79',
    EXPLAIN: '00fdd87c53df786dc26e9a9bd81b8ffcebad4860326bd7129860351dece19628',
    **shared.SOURCES,
}
LANES = ('heavy','light','iid','dirichlet')
LABELS = ('Central heavy','Central light','IID federation','Non-IID federation')
COLORS = ('#332288','#0072B2','#56B4E9','#D55E00')
FIGURES = (
    ('study_map','Three separate studies answer different questions'),
    ('legacy_quality','Configured-model quality, alarm burden and size'),
    ('legacy_recall','Legacy correct-category recall'),
    ('legacy_curves','Saved validation trajectories, not new experiments'),
    ('official_gap','Which classes account for the official39 gap?'),
    ('explanation_quality','Stable rankings do not guarantee accurate attribution sums'),
)


def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(root=ROOT):
    result = {}
    for name, expected in PINS.items():
        if digest(Path(root)/name) != expected:
            raise ValueError(f'Source identity differs: {name}')
        result[name] = read(Path(root)/name)
    return result


def export_histories(output, root=ROOT):
    """Reduce twelve verified saved result files to 720 portable plot points."""
    output = Path(output)
    if output.exists(): raise ValueError('Preserve existing history evidence')
    s = sources(root)[LEGACY]; records = []
    for lane in LANES:
        for seed in (0,1,2):
            reference = s['lanes'][lane]['runs'][str(seed)]
            path = Path(root)/reference['path']/'result.json'
            if digest(path) != reference['result_sha256']:
                raise ValueError(f'Saved result differs: {lane}/{seed}')
            r = read(path)
            if r['test_metrics'] is not None: raise ValueError('Unexpected test evaluation')
            records.append({'lane':lane,'seed':seed,'source_result_sha256':digest(path),
                'best_step':r['best_step'],'history':[
                    {'step':h['step'],'macro_f1':h['validation_metrics']['macro_f1'],
                     'benign_false_alert_rate':h['validation_metrics']['benign_false_alert_rate']}
                    for h in r['history']]})
    result = {'legacy_summary_sha256':PINS[LEGACY], 'scope':'Validation histories only; no new inference',
              'records':records, 'model_trained':False, 'checkpoint_opened':False}
    validate_histories(result,s)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def validate_histories(history, legacy):
    if history['legacy_summary_sha256'] != PINS[LEGACY]: raise ValueError('History summary differs')
    expected = {(lane,seed) for lane in LANES for seed in (0,1,2)}
    records = history['records']
    if len(records)!=12 or {(r['lane'],r['seed']) for r in records}!=expected:
        raise ValueError('History roster differs')
    for r in records:
        ref = legacy['lanes'][r['lane']]['runs'][str(r['seed'])]
        if r['source_result_sha256'] != ref['result_sha256']: raise ValueError('History result identity differs')
        h = r['history']
        if [x['step'] for x in h] != list(range(1,61)): raise ValueError('History steps differ')
        best = max(h,key=lambda x:x['macro_f1'])
        if best['step'] != r['best_step'] or r['best_step'] != ref['best_step']:
            raise ValueError('History selection differs')
        for metric in ('macro_f1','benign_false_alert_rate'):
            if best[metric] != ref['validation_metrics'][metric]: raise ValueError('History endpoint differs')
        if not all(np.isfinite(x[k]) and 0 <= x[k] <= 1 for x in h for k in ('macro_f1','benign_false_alert_rate')):
            raise ValueError('Invalid history value')


def derive(s, history):
    legacy, explanations = s[LEGACY],s[EXPLAIN]
    validate_histories(history,legacy)
    if explanations['confirmation_sha256']!=PINS[LEGACY]: raise ValueError('Explanation endpoints differ')
    f1=[]; far=[]; recalls=[]; parameters=[]; residuals=[]; overlaps=[]
    for lane in LANES:
        rows=[legacy['lanes'][lane]['runs'][str(seed)] for seed in (0,1,2)]
        f1.append([r['validation_metrics']['macro_f1'] for r in rows])
        far.append([r['validation_metrics']['benign_false_alert_rate'] for r in rows])
        recalls.append([[r['validation_metrics']['per_class'][c]['recall'] for c in CLASSES] for r in rows])
        parameters.append(rows[0]['num_parameters'])
        e=[explanations['results'][f'{lane}-seed-{seed}'] for seed in (0,1,2)]
        if any(x['checkpoint_sha256']!=r['checkpoint_sha256'] for x,r in zip(e,rows,strict=True)):
            raise ValueError('Explanation checkpoint differs')
        residuals.append([x['relative_mean_abs_residual'] for x in e])
        overlaps.append([float(np.mean([v['mc_repeat_top5_overlap'] for v in x['classes'].values()])) for x in e])
    return {'lanes':list(LANES),'seeds':[0,1,2],'class_order':CLASSES,'macro_f1':f1,
        'false_alert_rate':far,'recall':recalls,'num_parameters':parameters,
        'relative_attribution_residual':residuals,'repeat_top5_overlap':overlaps,
        'history':history['records'], 'validation_rows':327257,
        'support':[legacy['lanes']['heavy']['runs']['0']['validation_metrics']['per_class'][c]['support'] for c in CLASSES],
        'official_gap':s['reports/federated_class_safeguards/study-synthesis-results.json']['per_class_macro_f1_gap_contributions']['light_minus_dirichlet']}


def study_map(d):
    fig,ax=plt.subplots(figsize=(10,5.3)); ax.axis('off')
    shared.framing(fig,'One research question, three distinct evaluation settings','Does compact federated learning preserve useful eight-class coverage at acceptable cost?')
    rows=[('A  Original configured models','46 features | 440,594 fitting rows\n327,257 validation rows | seeds 0/1/2',
           'Heavy vs light vs IID/non-IID\nValidation-selected checkpoints', '#EDEAF4'),
          ('B  Official39 baseline','39 features | 2M training rows\n2,040,729 primary test rows | seeds 7/17/27',
           'Same light architecture\nFixed step-20 references and masks','#E5F0F6'),
          ('C  Shared optimizer extension','39 features | 500k fitting rows\n398,502 reserved rows | seeds 7/17/27',
           'Shared updates and compression\nFrozen checkpoints and alarm thresholds','#E5F4ED')]
    for y,(title,cohort,question,color) in zip((.66,.36,.06),rows):
        ax.add_patch(FancyBboxPatch((.025,y),.95,.24,boxstyle='round,pad=.01',facecolor=color,edgecolor='#B9C7D1'))
        ax.text(.05,y+.18,title,weight='bold',fontsize=12)
        ax.text(.05,y+.075,cohort,va='center',fontsize=10)
        ax.text(.60,y+.075,question,va='center',fontsize=10)
    fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.81)
    return fig


def legacy_quality(d):
    fig,axes=plt.subplots(1,3,figsize=(12,5),gridspec_kw={'width_ratios':[1.1,1,1]})
    shared.framing(fig,'A much larger model gives a modest configured-model advantage','Legacy46 validation | open circle: screening seed 0; square/triangle: new seeds 1/2')
    for ax,values,xlabel,limits in zip(axes[:2],(d['macro_f1'],100*np.array(d['false_alert_rate'])),
            ('Selected macro-F1','Benign false alerts (%)'),((0,1),(0,30))):
        for i,row in enumerate(values):
            ax.plot([min(row),max(row)],[i,i],color=COLORS[i],alpha=.5)
            for seed,v in enumerate(row):
                ax.scatter(v,i+(seed-1)*.12,marker=('o','s','^')[seed],s=45,
                           facecolors='none' if seed==0 else COLORS[i],edgecolors=COLORS[i])
        ax.set(yticks=range(4),yticklabels=LABELS if ax==axes[0] else ['']*4,ylim=(3.5,-.5),xlim=limits,xlabel=xlabel)
        ax.grid(axis='x',alpha=.2)
    ax=axes[2]; ax.barh(range(4),d['num_parameters'],color=COLORS,height=.6)
    for i,v in enumerate(d['num_parameters']): ax.text(v*1.15,i,f'{v:,}',va='center',fontsize=9)
    ax.set(xscale='log',xlim=(1000,1500000),yticks=range(4),yticklabels=['']*4,ylim=(3.5,-.5),xlabel='Trainable parameters (log)')
    ax.grid(axis='x',alpha=.2)
    fig.subplots_adjust(left=.16,right=.98,bottom=.19,top=.80,wspace=.18)
    return fig


def legacy_recall(d):
    fig,ax=plt.subplots(figsize=(10,6.3))
    shared.framing(fig,'Low false alerts coexist with missed attack categories','Legacy46 validation | equal-weight mean recall across seeds 0/1/2 | not final-test evidence')
    values=100*np.array(d['recall']).mean(axis=1).T
    im=ax.imshow(values,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    for i in range(8):
        for j in range(4): ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',color='white' if values[i,j]>58 else '#203344')
    ax.set(xticks=range(4),xticklabels=('Central\nheavy','Central\nlight','IID','Non-IID'),yticks=range(8),
           yticklabels=[f'{c} (n={n:,})' for c,n in zip(CLASSES,d['support'])])
    ax.tick_params(length=0); fig.colorbar(im,ax=ax,pad=.03,label='Correct-category recall (%)')
    fig.subplots_adjust(left=.25,right=.97,bottom=.13,top=.83)
    return fig


def legacy_curves(d):
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    shared.framing(fig,'Saved trajectories show fixed-budget behavior, not convergence','Legacy46 | all 60 logged validation points | seed 0 informed screening | no new training')
    for ax,lane,label in zip(axes.flat,LANES,LABELS):
        for r in [r for r in d['history'] if r['lane']==lane]:
            seed=r['seed']; h=r['history']; color=('#777777','#0072B2','#D55E00')[seed]
            ax.plot([x['step'] for x in h],[x['macro_f1'] for x in h],ls='--' if seed==0 else '-',
                    color=color,lw=1.5,label=f'Seed {seed}'+(' (screen)' if seed==0 else ''))
            best=h[r['best_step']-1]; ax.scatter(best['step'],best['macro_f1'],color=color,s=25,zorder=4)
        ax.set(title=label,xlabel='Epoch' if lane in ('heavy','light') else 'Federated round',ylabel='Validation macro-F1',ylim=(0,1),xlim=(1,60))
        ax.grid(alpha=.2); ax.legend(fontsize=8,loc='lower right',frameon=False)
    fig.subplots_adjust(left=.08,right=.98,bottom=.09,top=.80,hspace=.43,wspace=.23)
    return fig


def official_gap(d):
    fig,ax=plt.subplots(figsize=(10,5.8))
    shared.framing(fig,'Spoofing contributes most to the earlier non-IID score gap','Official39 final test | centralized-light minus controlled non-IID | arithmetic, not causal effects')
    values=100*np.array([d['official_gap'][c] for c in CLASSES])
    ax.barh(range(8),values,color=['#0072B2' if x>=0 else '#D55E00' for x in values])
    for i,v in enumerate(values): ax.text(v+(.12 if v>=0 else -.12),i,f'{v:+.2f}',ha='left' if v>=0 else 'right',va='center')
    ax.axvline(0,color='#71818B',lw=.8)
    ax.set(yticks=range(8),yticklabels=CLASSES,ylim=(7.6,-.6),xlim=(-1,11),xlabel='Contribution to macro-F1 gap (percentage points)')
    ax.grid(axis='x',alpha=.2)
    ax.text(.98,.04,f'Total gap: {sum(values):.2f} points',transform=ax.transAxes,ha='right')
    fig.subplots_adjust(left=.15,right=.98,bottom=.14,top=.80)
    return fig


def explanation_quality(d):
    fig,axes=plt.subplots(1,2,figsize=(10.5,5))
    shared.framing(fig,'Stable rankings are not sufficient for precise local explanations','Legacy46 SHAP expected gradients | 12 checkpoints | 64 balanced + 60 targeted cases')
    for ax,key,xlabel,limits in zip(axes,('relative_attribution_residual','repeat_top5_overlap'),
            ('Aggregate relative reconstruction residual (%)','Mean repeated top-5 overlap (%)'),((0,12),(0,100))):
        for i,row in enumerate(100*np.array(d[key])):
            for j,v in enumerate(row): ax.scatter(v,i+(j-1)*.13,marker=('o','s','^')[j],color=COLORS[i],s=45)
        ax.set(yticks=range(4),yticklabels=LABELS if ax==axes[0] else ['']*4,ylim=(3.5,-.5),xlim=limits,xlabel=xlabel)
        ax.grid(axis='x',alpha=.2)
    fig.subplots_adjust(left=.19,right=.98,bottom=.18,top=.80,wspace=.18)
    return fig


def figures(output,root=ROOT):
    output=Path(output)
    if output.exists(): raise ValueError('Preserve existing figures')
    if digest(Path(root)/HISTORY)!=HISTORY_SHA256: raise ValueError('Portable history identity differs')
    d=derive(sources(root),read(Path(root)/HISTORY)); shared.style(); output.mkdir(parents=True)
    (output/'figure-data.json').write_text(json.dumps(d,indent=2)+'\n')
    for (name,title),plot in zip(FIGURES,(study_map,legacy_quality,legacy_recall,legacy_curves,official_gap,explanation_quality),strict=True):
        fig=plot(d)
        fig.savefig(output/f'{name}.png',dpi=220)
        svg=output/f'{name}.svg'; fig.savefig(svg,metadata={'Date':None})
        svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
        plt.close(fig)
    receipt={'status':'complete','sources':{**PINS,HISTORY:digest(Path(root)/HISTORY)},
             'generator_sha256':digest(__file__),'shared_style_sha256':digest(shared.__file__),
             'figures':[{'id':n,'title':t} for n,t in FIGURES],
             'files':{p.name:digest(p) for p in sorted(output.iterdir())},
             'model_trained':False,'new_model_inference':False,'dataset_arrays_opened':False,'cloud_used':False}
    (output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def assemble(output,root=ROOT):
    output=Path(output)
    if output.exists(): raise ValueError('Preserve existing manuscript')
    root=Path(root); base=root/'reports/thesis'
    # Reuse the reviewed optimizer sections, rather than maintaining two versions
    # of the same result. Only numbering and relative document links change.
    chapter=root/'reports/federated_shared_optimizer/thesis/chapter.md'
    text=chapter.read_text(); blocks=text.split('<!-- pagebreak -->')
    selected=blocks[1:9]  # Existing sections 1-8, excluding separate abstract/history/conclusion.
    if len(selected)!=8: raise ValueError('Optimizer chapter structure changed')
    for index,block in enumerate(selected,start=1):
        if not re.search(rf'^## {index}\. ',block,re.M): raise ValueError('Optimizer section identity changed')
    middle='\n<!-- pagebreak -->\n'.join(selected)
    middle=re.sub(r'^## (\d+)\. ',lambda m:f'## {int(m[1])+11}. ',middle,flags=re.M)
    middle=re.sub(r'Figure (\d+)\.',lambda m:f'Figure {int(m[1])+7}.',middle)
    middle=middle.replace('](figures/','](../federated_shared_optimizer/thesis/figures/')
    middle=middle.replace('](../confirmation','](../federated_shared_optimizer/confirmation')
    front=base/'parts/01_context_and_baselines.md'; end=base/'parts/02_discussion.md'
    joined=front.read_text().rstrip()+'\n\n<!-- pagebreak -->\n'+middle.strip()+'\n\n<!-- pagebreak -->\n\n'+end.read_text()
    # Every local link must resolve in the delivered repository document.
    for target in re.findall(r'\]\(([^)]+)\)',joined):
        if not target.startswith(('https://','http://')) and not (base/target.split('#')[0]).exists():
            raise ValueError(f'Broken manuscript link: {target}')
    if '\u2011' in joined or '\ufffd' in joined: raise ValueError('Unsupported prose glyph')
    # The manuscript is byte-hashed; Windows newline translation must not
    # change its identity compared with the checked-in UTF-8/LF document.
    output.parent.mkdir(parents=True,exist_ok=True); output.write_bytes(joined.encode('utf-8'))
    receipt={'sources':{str(p.relative_to(root)):digest(p) for p in (front,chapter,end)},
             'compiler_sha256':digest(__file__),'manuscript_sha256':digest(output),
             'figure_manifest_sha256':digest(base/'figures/manifest.json'),
             'shared_figure_manifest_sha256':digest(root/'reports/federated_shared_optimizer/thesis/figures/manifest.json'),
             'scope':'Integrated academic draft; institutional template and author details pending team review'}
    output.with_suffix('.manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('history','figures','assemble'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    action={'history':export_histories,'figures':figures,'assemble':assemble}[args.command]
    result=action(args.output)
    print(f'Complete: {args.command}; no training or new inference')
