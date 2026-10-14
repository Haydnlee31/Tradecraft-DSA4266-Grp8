"""Publication figures from committed aggregate evidence only.

No torch import, private arrays, checkpoint loads, model evaluation or training.
The numerical figure-data file and provenance manifest make plots reproducible.
Use a new destination on each rebuild so earlier exported evidence is preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

from src.data.label_map import CLASSES

ROOT=Path(__file__).parents[1]
SOURCES={
    'reports/federated_shared_optimizer/confirmation-results.json':'a8efb197171780decefbfe82edfade5d354924e97889cb49d3df22d471b9001b',
    'reports/federated_class_safeguards/study-synthesis-results.json':'e32b27b781ea8fa30b06cd113ce21335accdc86a078b45d3b2aa7880c06a0aa1',
    'reports/federated_shared_optimizer/performance-results.json':'5244f2655ed0176f151418861fe8df287e31760fd5639f43295d7839b420896b',
}
ARMS=('central_adam','shared_adam','block128','fedavg_adam')
LABELS=('Centralized-light','Shared Adam','Compressed shared Adam','FedAvg')
SHORT=('Centralized\nlight','Shared\nAdam','Compressed\nshared Adam','FedAvg')
COLORS=('#0072B2','#009E73','#CC79A7','#D55E00')
SEEDS=(7,17,27)
MARKERS=('o','s','^')
FIGURES=[
 ('01_design','Study design and optimizer placement','The same light network is trained under four update designs. Rows are reserved before fitting; confirmation does not refit models or thresholds. Top arrows show workflow order, not rows migrating between cohorts. The lower arrows describe simulated computation, not physical networking.'),
 ('02_primary','Primary quality and false alerts','Each small marker is one training seed; the black diamond is the mean. No confidence interval is implied. All thresholds were fixed on development at the 5% target; actual confirmation false alerts may exceed 5%.'),
 ('03_recall','Correct category recall for all eight classes','Cells are equal-weight mean recalls across seeds at the primary target, using the same confirmation cohort. Supports are per model, not tripled. The common 0-100% color scale prevents rare-class performance from appearing stronger than it is.'),
 ('04_confusion','Where each true class is predicted','Each matrix averages three row-normalized seed matrices. All methods share supports and a 0-100% color scale. Rows are true labels and columns are predictions. Blank cells are exact zero; nonzero entries below 0.5% display <0.5.'),
 ('05_rare_errors','Detecting an attack is not identifying its category','For each rare class, the three outcomes exhaust its rows: correct category, another attack category, or benign (missed attack). Percentages are averaged across three models, not independent replicated datasets.'),
 ('06_communication','Quality gains retain communication costs','Left: primary mean macro-F1 and sample SD against counted application payload on a logarithmic horizontal axis. The dashed line is centralized-light quality, not zero centralized data-transfer cost. Right: synchronization counts. Payload is not measured traffic or latency; baseline raw accounting excludes initialization and message metadata.'),
 ('07_operating_points','Four frozen operating points','Each point is the mean actual confirmation false-alert rate and mean macro-F1 at a predeclared calibration target (1%, 5%, 10%, 20%). Lines only connect those four measured points; this is not an ROC curve or threshold sweep. Large rings identify the primary 5% target.'),
 ('08_historical_context','Earlier official39 baseline in its own evaluation','These are prior final-test results for the full39 references, with 2M training rows and argmax decisions. They are not directly comparable to the later 500k-fit, alarm-calibrated confirmation results. Seed markers and means are shown separately; no new test inference was run.'),
 ('09_class_support','Class imbalance in fitting and confirmation','The horizontal axis is logarithmic; bar lengths must not be read as linear proportions. Counts are frozen cohort metadata. Confirmation supports describe one shared population, not three times as many independent observations.'),
]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence(root=ROOT):
    loaded=[]
    for relative,expected in SOURCES.items():
        path=Path(root)/relative
        if digest(path)!=expected: raise ValueError(f'Evidence identity differs: {relative}')
        loaded.append(json.loads(path.read_text()))
    final,history,performance=loaded
    s,a=final['summary'],final['audit']
    if a['status']!='complete' or hashlib.sha256(json.dumps(s,indent=2).encode()).hexdigest()!=a['summary_sha256']:
        raise ValueError('Confirmation audit differs')
    rows=s['results']; expected={(seed,arm) for seed in SEEDS for arm in ARMS}
    if len(rows)!=12 or {(r['seed'],r['arm']) for r in rows}!=expected:
        raise ValueError('Missing or duplicate seed/method')
    return s,history,performance['summary']['data']


def extract(s,history,data):
    indexed={(r['seed'],r['arm']):r for r in s['results']}
    targets=(100,500,1000,2000)
    points={}
    for target in targets:
        metrics=[[next(p['confirmation_metrics'] for p in indexed[seed,arm]['operating_points']
                      if p['calibration']['target_basis_points']==target) for seed in SEEDS] for arm in ARMS]
        points[str(target)]={k:[[m[k] for m in by_seed] for by_seed in metrics]
                            for k in ('macro_f1','benign_false_alert_rate','confusion_matrix')}
    primary=points['500']
    cm=np.asarray(primary['confusion_matrix'],dtype=np.float64)
    supports=np.array([s['data']['counts'][c] for c in CLASSES])
    if not np.array_equal(cm.sum(axis=-1),np.broadcast_to(supports,cm.shape[:-1])):
        raise ValueError('Confusion counts disagree with class supports')
    normalized=cm/supports[None,None,:,None]
    mean=normalized.mean(axis=1)
    # Select the two rarest classes from their source counts, not a new label map.
    rare=sorted(np.argsort(supports)[:2].tolist())
    rare_parts=[]
    for i in rare:
        correct=mean[:,i,i]; missed=mean[:,i,CLASSES.index('Benign')]
        rare_parts.append(np.stack([correct,1-correct-missed,missed],axis=1).tolist())
    old_lanes=('light','iid','dirichlet')
    old={k:[[next(r for r in history['records'] if r['lane']==lane and r['seed']==seed)['metrics'][k]
             for seed in SEEDS] for lane in old_lanes] for k in ('macro_f1','benign_false_alert_rate')}
    return {'arms':list(ARMS),'labels':list(LABELS),'seeds':list(SEEDS),'class_order':CLASSES,
        'points':points,'mean_confusion_row_percent':(100*mean).tolist(),
        'mean_recall_percent':(100*np.diagonal(mean,axis1=1,axis2=2).T).tolist(),
        'rare_class_indices':rare,'rare_error_percent':(100*np.asarray(rare_parts)).tolist(),
        'confirmation_support':supports.tolist(),
        'fit_support':[data['counts']['fit'][c] for c in CLASSES],
        'payload_bytes':[indexed[7,arm]['work']['estimated_payload_bytes'] for arm in ARMS],
        'synchronizations':[indexed[7,arm]['work']['synchronization_rounds'] for arm in ARMS],
        'historical':old,'historical_population':2040729,'confirmation_population':s['data']['rows'],
        'fit_population':sum(data['counts']['fit'].values()),'development_population':sum(data['counts']['development'].values())}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'figure.titlesize':16,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#9BA7B0','text.color':'#203344','axes.labelcolor':'#203344',
        'xtick.color':'#455968','ytick.color':'#455968','svg.fonttype':'none','svg.hashsalt':'tradecraft-thesis-v1',
        'figure.facecolor':'white','savefig.facecolor':'white','axes.axisbelow':True})


def framing(fig,title,subtitle):
    fig.suptitle(title,x=.04,y=.98,ha='left',fontweight='bold')
    fig.text(.04,.91,subtitle,ha='left',va='top',fontsize=10,color='#566875')


def seed_panel(ax,values,labels,colors,xlabel,limit,target=None):
    values=np.asarray(values); ys=np.arange(len(values))
    for i,row in enumerate(values):
        ax.plot([row.min(),row.max()],[i,i],color=colors[i],lw=1.8,alpha=.5)
        for j,v in enumerate(row): ax.scatter(v,i+(-.12,0,.12)[j],marker=MARKERS[j],color=colors[i],s=48,zorder=3)
        ax.scatter(row.mean(),i,marker='D',color='#203344',s=28,zorder=4)
        ax.text(limit[1]*.98,i-.24,f'{row.mean():.4f}' if limit[1]==1 else f'{row.mean():.2f}%',ha='right',fontsize=9)
    ax.set(yticks=ys,yticklabels=labels,xlabel=xlabel,xlim=limit,ylim=(len(values)-.5,-.5))
    ax.grid(axis='x',alpha=.2)
    if target is not None: ax.axvline(target,color='#71818B',ls='--',lw=1)


def primary(d):
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={'width_ratios':[1.25,1]})
    framing(fig,'Quality improves, but centralized-light still leads','Reserved confirmation | n = 398,502 | three training seeds | frozen 5% calibration target')
    seed_panel(axes[0],d['points']['500']['macro_f1'],LABELS,COLORS,'Macro-F1 (0 to 1)',(0,1))
    seed_panel(axes[1],100*np.asarray(d['points']['500']['benign_false_alert_rate']),['']*4,COLORS,'Actual benign false alerts (%)',(0,8),5)
    handles=[Line2D([],[],marker=m,ls='',color='#455968',label=f'Seed {s}') for m,s in zip(MARKERS,SEEDS)]
    handles.append(Line2D([],[],marker='D',ls='',color='#203344',label='Mean'))
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False)
    fig.subplots_adjust(left=.21,right=.97,bottom=.22,top=.82,wspace=.16)
    return fig


def recall(d):
    fig,ax=plt.subplots(figsize=(10.5,6.2))
    framing(fig,'The aggregate gain is not a rare-class solution','Mean correct-category recall (%) | primary operating point | identical population for all methods')
    v=np.asarray(d['mean_recall_percent']); im=ax.imshow(v,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    for i in range(8):
        for j in range(4): ax.text(j,i,f'{v[i,j]:.1f}',ha='center',va='center',color='white' if v[i,j]>58 else '#203344')
    ax.set(xticks=range(4),xticklabels=SHORT,yticks=range(8),
           yticklabels=[f'{c}  (n={n:,})' for c,n in zip(CLASSES,d['confirmation_support'])])
    ax.tick_params(length=0); fig.colorbar(im,ax=ax,pad=.025,label='Recall (%)')
    fig.subplots_adjust(left=.25,right=.97,bottom=.13,top=.84)
    return fig


def confusion(d):
    fig,axes=plt.subplots(2,2,figsize=(11,10.5))
    framing(fig,'Confusion matrices separate missed attacks from wrong categories','Row-normalized percentages, averaged across three seeds | identical 0-100% scales')
    abbreviated=('Benign','DDoS','DoS','Recon','Web','Brute','Spoof','Mirai')
    for a,ax in enumerate(axes.flat):
        v=np.asarray(d['mean_confusion_row_percent'][a]); im=ax.imshow(v,vmin=0,vmax=100,cmap='Blues')
        for i in range(8):
            for j in range(8):
                if v[i,j]==0: continue
                label='<0.5' if v[i,j]<.5 else f'{v[i,j]:.1f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=7,color='white' if v[i,j]>58 else '#203344')
        ax.set(xticks=range(8),xticklabels=abbreviated,yticks=range(8),yticklabels=abbreviated,
               xlabel='Predicted category',ylabel='True category',title=LABELS[a])
        ax.tick_params(axis='x',labelrotation=45,labelsize=8); ax.tick_params(axis='y',labelsize=8)
    fig.subplots_adjust(left=.08,right=.88,bottom=.09,top=.85,hspace=.40,wspace=.28)
    cax=fig.add_axes([.92,.23,.018,.48]); fig.colorbar(im,cax=cax,label='True-class rows (%)')
    return fig


def rare_errors(d):
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    framing(fig,'A wrong attack label can still raise an alert; benign can suppress it','Three outcomes exhaust each rare class | mean across the same three training seeds')
    colors=('#009E73','#E6B04A','#B75B66'); names=('Correct category','Other attack category','Predicted benign')
    for k,ax in enumerate(axes):
        values=np.asarray(d['rare_error_percent'][k]); left=np.zeros(4)
        for j in range(3):
            ax.barh(range(4),values[:,j],left=left,color=colors[j],height=.65,label=names[j])
            for i,v in enumerate(values[:,j]):
                if v>=7: ax.text(left[i]+v/2,i,f'{v:.1f}%',ha='center',va='center',fontsize=9,color='white' if j!=1 else '#203344')
            left+=values[:,j]
        idx=d['rare_class_indices'][k]
        ax.set(title=f'{CLASSES[idx]} | {d["confirmation_support"][idx]:,} rows per model',
               yticks=range(4),yticklabels=LABELS if k==0 else ['']*4,ylim=(3.6,-.6),xlim=(0,100),xlabel='True-class outcomes (%)')
        ax.grid(axis='x',alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.subplots_adjust(left=.21,right=.98,bottom=.23,top=.80,wspace=.12)
    return fig


def communication(d):
    fig,axes=plt.subplots(1,2,figsize=(11,5.2),gridspec_kw={'width_ratios':[1.3,1]})
    framing(fig,'Compression reduces bytes, not synchronization frequency','Same training backward-example budget | modeled payload, not measured network traffic')
    values=np.asarray(d['points']['500']['macro_f1']); x=np.asarray(d['payload_bytes'])/1e6
    ax=axes[0]
    ax.axhline(values[0].mean(),ls='--',color=COLORS[0],lw=1.5,label='Centralized-light mean')
    for i in (1,2,3):
        ax.errorbar(x[i],values[i].mean(),yerr=values[i].std(ddof=1),fmt='o',color=COLORS[i],capsize=4,ms=8)
        label=f'{LABELS[i]}\n{x[i]:,.2f} MB'
        offset={1:(-10,25),2:(25,-30),3:(0,-42)}[i]
        ax.annotate(label,(x[i],values[i].mean()),xytext=offset,textcoords='offset points',ha='center',fontsize=9,
                    arrowprops={'arrowstyle':'-','color':COLORS[i],'lw':.8,'shrinkA':4,'shrinkB':6})
    ax.set(xscale='log',xlim=(45,3500),ylim=(0,1),xlabel='Application payload per run (MB; log scale)',ylabel='Primary macro-F1')
    ax.grid(alpha=.2); ax.legend(loc='lower right',frameon=False,fontsize=9)
    ax=axes[1]; ix=(3,2,1)
    ax.barh(range(3),[d['synchronizations'][i] for i in ix],color=[COLORS[i] for i in ix],height=.55)
    for k,i in enumerate(ix): ax.text(d['synchronizations'][i]+45,k,f'{d["synchronizations"][i]:,}',va='center')
    ax.set(yticks=range(3),yticklabels=[SHORT[i].replace('\n',' ') for i in ix],xlim=(0,2700),ylim=(2.6,-.6),xlabel='Synchronization rounds per training run')
    ax.grid(axis='x',alpha=.2)
    fig.subplots_adjust(left=.08,right=.98,bottom=.19,top=.81,wspace=.65)
    return fig


def operating(d):
    fig,ax=plt.subplots(figsize=(9.8,5.5))
    framing(fig,'Alarm control changes the performance comparison','Only four predeclared thresholds per model | no confirmation recalibration or new sweep')
    targets=(100,500,1000,2000)
    for i in range(4):
        xs=[100*np.mean(d['points'][str(t)]['benign_false_alert_rate'][i]) for t in targets]
        ys=[np.mean(d['points'][str(t)]['macro_f1'][i]) for t in targets]
        ax.plot(xs,ys,'o-',color=COLORS[i],label=LABELS[i],lw=1.8,ms=5)
        ax.scatter(xs[1],ys[1],s=130,facecolors='none',edgecolors=COLORS[i],lw=1.6,zorder=4)
        if i==3:
            for x,y,t in zip(xs,ys,targets): ax.annotate(f'{t/100:g}% target',(x,y),xytext=(0,-17),textcoords='offset points',ha='center',fontsize=8)
    ax.set(xlim=(0,23),ylim=(0,1),xlabel='Mean actual benign false alerts (%)',ylabel='Mean macro-F1')
    ax.grid(alpha=.2); ax.legend(loc='lower right',frameon=False)
    ax.text(.02,.95,'Large rings: primary 5% calibration target',transform=ax.transAxes,fontsize=9)
    fig.subplots_adjust(left=.10,right=.98,bottom=.15,top=.82)
    return fig


def historical(d):
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    framing(fig,'Earlier official39 baseline: non-IID performance motivated the extension','Separate final-test cohort (n = 2,040,729) | 2M training rows | argmax | not an extension leaderboard')
    labels=('Centralized-light','IID federated','Controlled non-IID'); colors=('#0072B2','#56B4E9','#D55E00')
    seed_panel(axes[0],d['historical']['macro_f1'],labels,colors,'Macro-F1 (0 to 1)',(0,1))
    seed_panel(axes[1],100*np.asarray(d['historical']['benign_false_alert_rate']),['']*3,colors,'Benign false alerts (%)',(0,25))
    fig.text(.21,.06,'Low false alerts do not establish success when attack categories are missed.',fontsize=10)
    fig.subplots_adjust(left=.20,right=.98,bottom=.22,top=.79,wspace=.14)
    return fig


def support(d):
    fig,axes=plt.subplots(1,2,figsize=(10.5,6))
    framing(fig,'Rare categories remain a small fraction of each cohort','Frozen counts from the split manifest | logarithmic count axes')
    for ax,key,title,color in zip(axes,('fit_support','confirmation_support'),('Fit: 500,000 rows','Confirmation: 398,502 rows'),('#0072B2','#009E73')):
        v=d[key]; ax.barh(range(8),v,color=color,height=.65)
        for i,n in enumerate(v): ax.text(n*1.13,i,f'{n:,}',va='center',fontsize=9)
        ax.set(xscale='log',xlim=(100,1000000),yticks=range(8),yticklabels=CLASSES if key=='fit_support' else ['']*8,
               ylim=(7.6,-.6),xlabel='Rows (log scale)',title=title)
        ax.grid(axis='x',alpha=.18)
    fig.subplots_adjust(left=.14,right=.98,bottom=.13,top=.80,wspace=.12)
    return fig


def design(d):
    fig,ax=plt.subplots(figsize=(11,7.2)); ax.set_axis_off(); ax.set(xlim=(0,1),ylim=(0,1))
    framing(fig,'Change the training updates, keep the inference network fixed','All methods: 39 inputs -> 64 -> 32 -> 8 outputs | LayerNorm | 5,096 parameters')
    def box(x,y,w,h,label,color='#EAF0F5',size=10):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009',facecolor=color,edgecolor='#ADBAC4',lw=.8))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)
    def arrow(a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#617787',lw=1.2))
    box(.04,.78,.26,.11,'FIT\n500,000 rows\nScaler and model fitting',size=9)
    box(.37,.78,.26,.11,'DEVELOPMENT\n399,552 rows\nSelection and threshold calibration',size=9)
    box(.70,.78,.26,.11,'CONFIRMATION\n398,502 rows\nFrozen models and thresholds',size=9)
    arrow((.31,.835),(.355,.835)); arrow((.64,.835),(.685,.835))
    ax.text(.5,.725,'Disjoint input groups within this extension; historical source collection is not independent new traffic.',ha='center',fontsize=8.5)
    rows=[('Centralized-light','Pooled fitting rows','Persistent central Adam','43,200 optimizer steps'),
          ('FedAvg','20 clients: local Adam\n20 local steps per round','Average client models','108 synchronizations'),
          ('Shared Adam','20 clients: gradients\nat the same model state','Weighted gradients\nPersistent server Adam','2,160 synchronizations'),
          ('Compressed shared Adam','4-bit blocks of 128\nClient residual feedback','Persistent server Adam\nCompressed model delivery','2,160 synchronizations')]
    for y,(name,left,right,note),color in zip((.54,.38,.22,.06),rows,(COLORS[0],COLORS[3],COLORS[1],COLORS[2])):
        ax.text(.025,y+.065,name,ha='left',va='center',fontsize=9,color=color,fontweight='bold')
        box(.25,y,.25,.12,left,size=9); box(.59,y,.24,.12,right,size=9)
        arrow((.51,y+.06),(.58,y+.06)); ax.text(.855,y+.06,note.replace(' ', '\n',1),ha='left',va='center',fontsize=8.5)
    fig.subplots_adjust(left=.01,right=.99,bottom=.03,top=.86)
    return fig


PLOTTERS=(design,primary,recall,confusion,rare_errors,communication,operating,historical,support)


def build(output,root=ROOT):
    output=Path(output)
    if output.exists(): raise ValueError('Preserve existing figure pack; choose a new destination')
    s,h,data=evidence(root); d=extract(s,h,data); style(); output.mkdir(parents=True)
    (output/'figure-data.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    rendered=[]
    for (name,title,caption),plot in zip(FIGURES,PLOTTERS,strict=True):
        fig=plot(d)
        for suffix in ('png','svg'):
            fig.savefig(output/f'{name}.{suffix}',dpi=220,metadata={'Date':None} if suffix=='svg' else {})
            if suffix=='svg':
                # Matplotlib emits trailing spaces in path commands. Whitespace
                # normalization preserves the SVG while keeping source diffs clean.
                svg=output/f'{name}.svg'
                svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig); rendered.append({'id':name,'title':title,'caption':caption})
    receipt={'status':'complete','generator_sha256':digest(__file__),'sources':SOURCES,
        'figure_count':len(rendered),'figures':rendered,'files':{p.name:digest(p) for p in sorted(output.iterdir())},
        'model_trained':False,'new_model_inference':False,'dataset_arrays_opened':False,
        'thresholds_refitted':False,'cloud_used':False,'curve_policy':'No invented convergence traces; only saved endpoints and four frozen alarm targets'}
    (output/'manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(f'Created {len(rendered)} figures as PNG and SVG from aggregate evidence; no training/inference.')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output)
