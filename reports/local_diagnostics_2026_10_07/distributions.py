"""Read-only train/validation provenance and feature-regime diagnostic.

Stream raw CSV record batches, never loading the whole raw dataset. Retain only
three features for the small rare-class subsets to compare their value multisets.
No test split, resampling, scaler fitting or training is performed.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.csv as csv

from src.data.label_map import CLASSES, LABEL_TO_CLASS
from src.explain.research import sha, load_model

FEATURES = ('IAT', 'Number', 'Weight')
# Descriptive bins in recorded units, not intrusion thresholds or timing claims.
IAT_EDGES = [0, 1, 1e3, 1e6, 1e8, float('inf')]
RARE = ('Web-based', 'Brute Force')


def accumulate(frame, stats, rare):
    for label in frame['label'].unique().to_list():
        c = LABEL_TO_CLASS[label]
        sub = frame.filter(pl.col('label') == label)
        values = sub.select(FEATURES).to_numpy()
        record = stats.setdefault(label, {'class': c, 'rows': 0, 'invalid': [0]*3,
                                         'iat_bins': [0]*5, 'number': Counter(), 'weight': Counter()})
        record['rows'] += len(sub)
        record['invalid'] = (np.array(record['invalid']) + (~np.isfinite(values)).sum(0)).tolist()
        record['iat_bins'] = (np.array(record['iat_bins']) + np.histogram(values[:,0], bins=IAT_EDGES)[0]).tolist()
        for i,name in ((1,'number'),(2,'weight')):
            keys,counts = np.unique(values[:,i],return_counts=True)
            record[name].update({str(k):int(n) for k,n in zip(keys,counts)})
        if c in RARE:
            rare[label].append(values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    evidence={'test_evaluated':False,'iat_bin_edges':['0','1','1000','1000000','100000000','inf'],
              'source_sha256':sha(__file__),'splits':{}}
    manifest=pl.read_csv('data/sampled/manifest.csv').filter(pl.col('split').is_in(['train','val']))
    evidence['manifest_sha256']=sha('data/sampled/manifest.csv')
    evidence['manifest_records_sampling_seed']='sampling_seed' in manifest.columns
    _,scaler,model_manifest,_=load_model('outputs/cloud-analysis-2026-10-06/models/light-seed7')
    for split,rawsplit in [('train','train'),('val','validation')]:
        path=Path(f'data/raw/CICIOT23/{rawsplit}/{rawsplit}.csv')
        sampled=Path(f'data/splits/{split}.parquet')
        assert sha(sampled)==model_manifest['split_sha256'][split], 'Sample drift'
        raw_stats={}; raw_rare=defaultdict(list)
        reader=csv.open_csv(path,read_options=csv.ReadOptions(block_size=8*1024*1024),
            convert_options=csv.ConvertOptions(include_columns=[*FEATURES,'label'],
                column_types={**{f:pa.float64() for f in FEATURES},'label':pa.string()}))
        for batch in reader:
            accumulate(pl.from_arrow(batch),raw_stats,raw_rare)
        frame=pl.read_parquet(sampled)
        sampled_stats={};sampled_rare=defaultdict(list)
        accumulate(frame.select([*FEATURES,'label']),sampled_stats,sampled_rare)
        manifest_differences={}
        for row in manifest.filter(pl.col('split')==split).iter_rows(named=True):
            assert raw_stats[row['raw_label']]['rows']==row['n_total_in_file'], 'Raw manifest count mismatch'
            difference=sampled_stats[row['raw_label']]['rows']-row['n_sampled_from_file']
            if difference: manifest_differences[row['raw_label']]=difference
        # The legacy manifest precedes cross-split duplicate removal. Reconstruct
        # that operation for rare labels only: all their rows were below the cap.
        # This is a read-only anti-join, never a replacement of the fixed split.
        reconstructed={}
        if split=='val':
            labels=list(raw_rare)
            original=pl.scan_csv(path).filter(pl.col('label').is_in(labels)).collect(engine='streaming')
            train=pl.read_parquet('data/splits/train.parquet')
            keys=[*model_manifest['feature_columns'],'label']
            retained=original.join(train.select(keys).unique(),on=keys,how='anti')
            for label in labels:
                expected=retained.filter(pl.col('label')==label).select(FEATURES).to_numpy()
                actual=frame.filter(pl.col('label')==label).select(FEATURES).to_numpy()
                expected=expected[np.lexsort(expected.T[::-1])];actual=actual[np.lexsort(actual.T[::-1])]
                reconstructed[label]={'after_purge_rows':len(expected),'sample_rows':len(actual),
                    'three_feature_multiset_matches':bool(len(expected)==len(actual) and np.allclose(expected,actual,rtol=1e-12,atol=1e-12))}
        comparisons={}
        for label,parts in raw_rare.items():
            r=np.concatenate(parts);s=np.concatenate(sampled_rare[label])
            # Sort all three columns jointly, preserving their associations.
            r=r[np.lexsort(r.T[::-1])];s=s[np.lexsort(s.T[::-1])]
            same=len(r)==len(s) and np.allclose(r,s,rtol=1e-12,atol=1e-12)
            comparisons[label]={'raw_rows':len(r),'sample_rows':len(s),
                                'feature_multisets_match_rtol_1e12':bool(same),
                                'max_abs_parser_difference':float(np.max(np.abs(r-s))) if len(r)==len(s) else None}
        scale={}
        if split=='train':
            for f in FEATURES:
                i=model_manifest['feature_columns'].index(f)
                values=frame[f].to_numpy()
                scale[f]={'sample_mean':float(np.mean(values)),'saved_scaler_mean':float(scaler.mean_[i]),
                          'mean_matches':bool(np.isclose(np.mean(values),scaler.mean_[i],rtol=1e-10)),
                          'variance_matches':bool(np.isclose(np.var(values),scaler.var_[i],rtol=1e-10))}
        evidence['splits'][split]={'raw_sha256':sha(path),'sample_sha256':sha(sampled),
            'raw':raw_stats,'sampled':sampled_stats,'rare_value_comparison':comparisons,'scaler':scale,
            'sample_count_change_since_legacy_manifest':manifest_differences,'rare_after_duplicate_removal':reconstructed}
        print(split, 'raw counts verified; legacy manifest changes',len(manifest_differences), 'rare reconstructed',reconstructed,flush=True)
    a.output.mkdir(parents=True)
    (a.output/'summary.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__': main()
