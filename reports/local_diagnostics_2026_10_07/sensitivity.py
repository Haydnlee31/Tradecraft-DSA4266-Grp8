"""Bounded validation-only SHAP sensitivity screen on frozen seed-7 models.

Shared rows are chosen before any new attribution results. Natural backgrounds
and a class-balanced background explain different reference populations; changing
the latter must never be described as merely increasing numerical accuracy.
Private row IDs and tensors remain in ignored outputs, not the public summary.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.data.label_map import CLASSES
from src.models.dataset import to_arrays
from src.explain.research import sha, load_model, logits, confusion, attribution_summary, overlap


def backgrounds(train_y, original_rows, seed=4267):
    rng=np.random.default_rng(seed)
    # Independent natural draw may overlap by chance; do not exclude prior rows.
    natural=rng.choice(len(train_y),128,replace=False)
    balanced=np.concatenate([rng.choice(np.flatnonzero(train_y==i),16,replace=False) for i in range(len(CLASSES))])
    return {'original':np.asarray(original_rows),'independent':natural,'balanced':balanced}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    parent=Path('outputs/cloud-analysis-2026-10-06')
    sampling=json.loads((parent/'explanations/sampling.json').read_text())
    prior=json.loads((parent/'explanations/summary.json').read_text())
    torch.set_num_threads(2)
    train=pl.read_parquet('data/splits/train.parquet');val=pl.read_parquet('data/splits/val.parquet')
    for split in ('train','val'):
        assert sha(f'data/splits/{split}.parquet')==prior['split_sha256'][split], 'Split drift'
    # First two members of each previously frozen eight-row class core.
    core=np.asarray(sampling['core_rows']).reshape(len(CLASSES),8)[:,:2].ravel().tolist()
    lanes=('heavy','light','iid','dirichlet')
    extra=[]
    for lane in lanes:
        for group in ('false_alert','miss_Web-based','miss_Brute Force','miss_DoS'):
            row=sampling['groups'][f'{lane}-seed-7'][group]['selected_row']
            if row is not None and row not in core and row not in extra: extra.append(row)
    rows=np.asarray(core+extra)
    _,scaler,manifest,_=load_model(parent/'models/light-seed7')
    x,y=to_arrays(val,scaler,manifest['feature_columns']);tx,ty=to_arrays(train,scaler,manifest['feature_columns'])
    bg=backgrounds(ty,sampling['background_train_rows'])
    plan=[('original_256','original',256),('original_1024','original',1024),
          ('independent_1024','independent',1024),('balanced_1024','balanced',1024)]
    a.output.mkdir(parents=True)
    (a.output/'private_selection.json').write_text(json.dumps({'core_rows':core,'extra_rows':extra,'background_rows':{k:v.tolist() for k,v in bg.items()}},indent=2))
    result={'test_evaluated':False,'training_performed':False,'core_count':len(core),'supplement_count':len(extra),
        'source_sha256':sha(__file__),'parent_summary_sha256':sha(parent/'explanations/summary.json'),
        'split_sha256':prior['split_sha256'],'packages':{k:importlib.metadata.version(k) for k in ('torch','shap','numpy','polars')},
        'background_class_counts':{k:dict(zip(CLASSES,np.bincount(ty[v],minlength=len(CLASSES)).tolist())) for k,v in bg.items()},
        'plan':plan,'mc_seeds':[0,1],'models':{}}
    import shap
    for lane in lanes:
        model,_,m,r=load_model(parent/f'models/{lane}-seed7')
        assert m['scaler_sha256']==manifest['scaler_sha256'], 'Scaler drift'
        assert confusion(y,logits(model,x).argmax(1)).tolist()==r['validation_metrics']['confusion_matrix'], 'Inference parity'
        selected=x[rows];sy=y[rows];out=logits(model,selected)
        variants={};attributions={}
        for name,bgname,n in plan:
            reference=tx[bg[bgname]];base=logits(model,reference).mean(0)
            explainer=shap.GradientExplainer(model,torch.from_numpy(reference),batch_size=128)
            values=[]
            for seed in (0,1):
                v=np.asarray(explainer.shap_values(torch.from_numpy(selected),nsamples=n,rseed=seed))
                assert v.shape==(len(rows),len(m['feature_columns']),len(CLASSES)) and np.isfinite(v).all()
                values.append(v)
            mean=(values[0]+values[1])/2
            stats=attribution_summary(mean,values[1],out,base,sy,m['feature_columns'],len(core))
            repeats=attribution_summary(values[0],values[1],out,base,sy,m['feature_columns'],len(core))
            for c in CLASSES: stats['classes'][c]['mc_repeat_top5_overlap']=repeats['classes'][c]['mc_repeat_top5_overlap']
            stats['mc_repeat_mean_abs_attribution_difference']=repeats['mc_repeat_mean_abs_attribution_difference']
            pred=out.argmax(1);errors=np.flatnonzero(pred!=sy)
            margin=out[errors,pred[errors]]-out[errors,sy[errors]]
            residual=margin-base[pred[errors]]+base[sy[errors]]-(mean[errors,:,pred[errors]]-mean[errors,:,sy[errors]]).sum(1)
            stats['error_count']=len(errors)
            stats['error_residual_exceeds_margin']=int((np.abs(residual)>=np.abs(margin)).sum())
            stats['reference_logits']=base.tolist()
            variants[name]=stats;attributions[name]=mean
            np.savez_compressed(a.output/f'{lane}-{name}.npz',repeat0=values[0],repeat1=values[1],rows=rows)
            print(lane,name,'relative residual',round(stats['relative_mean_abs_residual'],4),flush=True)
        comparisons={}
        for first,second in [('original_256','original_1024'),('original_1024','independent_1024'),('original_1024','balanced_1024')]:
            comparisons[f'{first}_vs_{second}']={c:overlap(np.asarray(variants[first]['classes'][c]['mean_abs_attributions']),np.asarray(variants[second]['classes'][c]['mean_abs_attributions'])) for c in CLASSES}
        result['models'][lane]={'checkpoint_sha256':r['checkpoint_sha256'],'validation_metrics':r['validation_metrics'],
                                'variants':variants,'comparison_top5_overlap':comparisons}
        (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    result['complete']=True
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
