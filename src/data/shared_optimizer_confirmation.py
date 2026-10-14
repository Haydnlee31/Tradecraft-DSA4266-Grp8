"""Reconstruct the authorized reserved cohort from checksum-verified shards.

Only bounded raw batches and fingerprints of the historical 2M training view
are held in memory. Never use the unhashed staging cache. The original scaler
and split membership are replayed, not fitted or selected using model outcomes.
"""
import json
from pathlib import Path
import time

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_subsets import SubsetBatches
from src.data.shared_optimizer_data import digest_array, split_rows, remove_precision_collisions
from src.eval import shared_optimizer_confirmation_prepare as prep
from src.federated.shared_adam import require
from src.models.research import atomic_json


def load_array(root,name,manifest):
    require(Path(name).name == name and sha256(root/name) == manifest['files'][name], 'Prepared array differs')
    return np.load(root/name,mmap_mode='r',allow_pickle=False)


def reconstruct(data_root,parent,subsets,output,lock,base,performance,authorized=False):
    require(authorized is True, 'Explicit confirmation authorization required')
    output=Path(output); root=Path(data_root)
    require(not output.exists(), 'Preserve existing confirmation data')
    started=time.perf_counter()
    require(prep.metadata_preflight(root,parent,subsets,base,performance,lock['plan']) == lock['data'],
            'Locked metadata differs')
    manifest=json.loads((root/'manifest.json').read_text())
    scaler=json.loads((root/'scaler.json').read_text())
    mean,scale=np.array(scaler['mean']),np.array(scaler['scale'])
    arrays={}; ids={}
    for name in ('fit','development','confirmation'):
        ids[name]=load_array(root,f'{name}_row_ids.npy',manifest)
        r=ids[name]; n=sum(manifest['counts'][name].values())
        require(r.dtype == np.int64 and r.shape == (n,) and bool((np.diff(r)>0).all()) and
                int(r[0])>=0 and int(r[-1])<base['historical_training_rows'], 'Invalid frozen row IDs')
        if name!='confirmation':
            x,y=[load_array(root,f'{name}_{k}.npy',manifest) for k in ('x','y')]
            require(x.dtype == np.float32 and x.shape == (n,len(mean)) and y.dtype == np.int64 and
                    y.shape == (n,), 'Prepared shape differs')
            arrays[name]=(x,y)
    all_ids=np.concatenate(list(ids.values()))
    require(len(np.unique(all_ids)) == len(all_ids), 'Row ID leakage across cohorts')
    del all_ids
    data=SubsetBatches(parent,subsets,base['historical_training_rows'])
    output.mkdir(parents=True)
    atomic_json({'status':'reconstructing','confirmation_opened':True,'confirmation_scored':False},output/'status.json')
    rows=len(ids['confirmation'])
    xout=np.lib.format.open_memmap(output/'confirmation_x.npy',mode='w+',dtype='<f4',shape=(rows,len(mean)))
    yout=np.lib.format.open_memmap(output/'confirmation_y.npy',mode='w+',dtype='<i8',shape=(rows,))
    raw_hashes=np.empty(data.rows,dtype='V32'); input_hashes=np.empty(data.rows,dtype='V32')
    written={k:0 for k in ids}; counts={k:np.zeros(len(CLASSES),dtype=np.int64) for k in ids}
    offset=0
    # raw_batches verifies each shard against the pinned manifest before use.
    for raw,y in data.raw_batches():
        end=offset+len(y)
        require(end<=data.rows and raw.shape == (len(y),len(mean)) and
                bool(((y>=0)&(y<len(CLASSES))).all()), 'Invalid raw batch')
        x=((raw.astype(np.float64)-mean)/scale).astype(np.float32)
        require(bool(np.isfinite(x).all()), 'Nonfinite standardized input')
        raw_hashes[offset:end]=digest_array(raw)
        input_hashes[offset:end]=digest_array(x)
        for name,r in ids.items():
            lo,hi=np.searchsorted(r,[offset,end]); local=r[lo:hi]-offset
            if not len(local): continue
            if name=='confirmation':
                xout[lo:hi]=x[local]; yout[lo:hi]=y[local]
            else:
                old_x,old_y=arrays[name]
                require(np.array_equal(old_x[lo:hi],x[local]) and np.array_equal(old_y[lo:hi],y[local]),
                        f'Historical {name} features, labels or order differ')
            counts[name]+=np.bincount(y[local],minlength=len(CLASSES)); written[name]+=int(hi-lo)
        offset=end
    require(offset==data.rows, 'Historical row count differs')
    print(f'PASS: {offset:,} historical training rows streamed; fit/development anchors match',flush=True)
    # Replay the original label-independent membership and float32 collision
    # rule. This catches wrong ordering even for rows absent from fit/development.
    original=split_rows(raw_hashes,base['split_domain'],base['fit_rows'])
    replay,removed=remove_precision_collisions(original,input_hashes)
    require(removed == manifest['float32_collision_rows_removed'], 'Precision collision removal differs')
    for label,name in enumerate(('fit','development','confirmation')):
        require(np.array_equal(np.flatnonzero(replay==label),ids[name]), f'Frozen {name} membership differs')
        require(written[name] == len(ids[name]) and counts[name].tolist() ==
                [manifest['counts'][name][c] for c in CLASSES], f'{name} counts differ')
    groups={name:np.unique(input_hashes[r]) for name,r in ids.items()}
    overlaps={f'{a}_vs_{b}':len(np.intersect1d(groups[a],groups[b]))
              for a,b in (('fit','development'),('fit','confirmation'),('development','confirmation'))}
    require(not any(overlaps.values()), 'Standardized input-group leakage')
    xout.flush(); yout.flush()
    require(prep.metadata_preflight(root,parent,subsets,base,performance,lock['plan']) == lock['data'],
            'Data metadata changed during reconstruction')
    receipt={'status':'complete','builder_sha256':sha256(__file__),'source':lock['data'],
        'files':{p.name:sha256(p) for p in (output/'confirmation_x.npy',output/'confirmation_y.npy')},
        'rows':rows,'counts':manifest['counts']['confirmation'],'raw_rows_streamed':offset,
        'raw_vectors_unique':True,'fit_development_anchor_rows_verified':written['fit']+written['development'],
        'all_frozen_memberships_exact':True,'float32_collision_rows_removed':removed,
        'unique_input_groups':{k:len(v) for k,v in groups.items()},'cross_cohort_input_groups':overlaps,
        'elapsed_seconds':time.perf_counter()-started,'process_peak_rss_bytes':prep.peak_rss(),
        'scaler_refitted':False,'unhashed_staging_cache_used':False,'confirmation_opened':True,
        'confirmation_scored':False,'original_validation_opened':False,'test_opened':False}
    atomic_json(receipt,output/'manifest.json')
    atomic_json({'status':'complete','confirmation_opened':True,'confirmation_scored':False},output/'status.json')
    return receipt


def load_reconstructed(root,receipt):
    root=Path(root)
    require(receipt['status']=='complete' and receipt['all_frozen_memberships_exact'] and
            not any(receipt['cross_cohort_input_groups'].values()), 'Unverified confirmation data')
    x,y=[load_array(root,name,receipt) for name in ('confirmation_x.npy','confirmation_y.npy')]
    require(x.shape==(receipt['rows'],len(receipt['source']['features'])) and x.dtype==np.float32 and
            y.shape==(receipt['rows'],) and y.dtype==np.int64 and np.isfinite(x).all(), 'Confirmation shape differs')
    require(np.bincount(y,minlength=len(CLASSES)).tolist()==[receipt['counts'][c] for c in CLASSES], 'Confirmation counts differ')
    return x,y
