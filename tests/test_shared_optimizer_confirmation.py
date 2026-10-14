"""Final scoring cannot refit thresholds, alter cohorts or hide failed seeds."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data import shared_optimizer_data as original_data
from src.data import shared_optimizer_confirmation as cohort
from src.eval import shared_optimizer_confirmation as final
from src.eval import shared_optimizer_confirmation_prepare as prep
from src.eval import shared_optimizer_alarms as alarms
from src.models.research import atomic_json
from tests import test_shared_optimizer_confirmation_prepare as model_fixtures


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def data_fixture(self,root):
        data=root/'prepared'; parent=root/'parent'; subsets=root/'subsets'
        for p in (data,parent,subsets): p.mkdir()
        raw=np.random.default_rng(41).normal(size=(2000,39)); y=np.arange(2000,dtype=np.int64)%8
        features=[f'f{i}' for i in range(39)]; counts=dict(zip(CLASSES,[250]*8))
        protocol={'features':features,'counts_by_split_and_class':{'train':counts}}
        atomic_json(protocol,parent/'protocol.json')
        m={'status':'complete','protocol_sha256':sha256(parent/'protocol.json'),'features':features,
           'classes':CLASSES,'counts_by_split_and_class':{'train':counts},'files':[]}
        atomic_json(m,parent/'manifest.json')
        recipe={'features':features,'classes':CLASSES,'sizes':[2000],'quotas':[counts],
                'parent_manifest_sha256':sha256(parent/'manifest.json')}
        atomic_json(recipe,subsets/'recipe.json')
        table=pa.table({**{name:raw[:,i] for i,name in enumerate(features)},'_class_id':y})
        pq.write_table(table,subsets/'train.parquet')
        atomic_json({'status':'complete','recipe_sha256':sha256(subsets/'recipe.json'),
            'parent_manifest_sha256':sha256(parent/'manifest.json'),'counts_by_size':{'2000':counts},
            'shards':[{'path':'train.parquet','tier':0,'rows':2000,'sha256':sha256(subsets/'train.parquet')}]},
            subsets/'manifest.json')
        base={'fit_rows':500,'historical_training_rows':2000,'split_domain':'synthetic-confirmation:',
              'parent_manifest_sha256':sha256(parent/'manifest.json'),'subset_manifest_sha256':sha256(subsets/'manifest.json')}
        split=original_data.split_rows(original_data.digest_array(raw),base['split_domain'],500)
        scaler=original_data.fit_scaler(raw,np.flatnonzero(split==0)); x=scaler.transform(raw).astype(np.float32)
        split,removed=original_data.remove_precision_collisions(split,original_data.digest_array(x))
        counts={}
        for label,name in enumerate(('fit','development','confirmation')):
            rows=np.flatnonzero(split==label); np.save(data/f'{name}_row_ids.npy',rows,allow_pickle=False)
            counts[name]=dict(zip(CLASSES,np.bincount(y[rows],minlength=8).tolist()))
            if name!='confirmation':
                np.save(data/f'{name}_x.npy',x[rows],allow_pickle=False)
                np.save(data/f'{name}_y.npy',y[rows],allow_pickle=False)
        atomic_json({'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'n_samples_seen':500},data/'scaler.json')
        manifest={'features':features,'counts':counts,'files':{p.name:sha256(p) for p in data.iterdir()},
                  'float32_collision_rows_removed':removed}
        atomic_json(manifest,data/'manifest.json')
        plan={'data_manifest_sha256':sha256(data/'manifest.json')}
        performance={'summary':{'data':manifest}}
        lock={'plan':plan,'data':prep.metadata_preflight(data,parent,subsets,base,performance,plan)}
        return data,parent,subsets,base,performance,lock,x[split==2],y[split==2]

    def test_execution_authorized_but_old_preparation_unchanged(self):
        execution,lock,*_=final.read_contract()
        self.assertTrue(execution['confirmation_scoring_authorized'])
        self.assertFalse(lock['confirmation_scoring_authorized'])
        self.assertFalse(execution['training_permitted'])
        self.assertFalse(execution['threshold_refitting_permitted'])
        self.assertEqual(len(lock['models']),12)
        self.assertEqual(sum(len(e['calibrations']) for e in lock['models']),48)

    def test_no_authorization_stops_before_access(self):
        with patch.object(final,'read_contract',side_effect=AssertionError('Must stop first')):
            with self.assertRaisesRegex(ValueError,'authorize-confirmation'): final.run(None,None,None,None)
        with patch.object(prep,'metadata_preflight',side_effect=AssertionError('Must stop first')):
            with self.assertRaisesRegex(ValueError,'authorization'): cohort.reconstruct(None,None,None,None,None,None,None)

    def test_reconstruct_verified_parquet_and_exact_frozen_membership_without_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data,parent,subsets,base,performance,lock,x,y=self.data_fixture(root)
            # The fixture deliberately has no raw-training-staging.npy.
            with patch.object(original_data,'fit_scaler',side_effect=AssertionError('No fitting')):
                r=cohort.reconstruct(data,parent,subsets,root/'result',lock,base,performance,True)
            actual_x,actual_y=cohort.load_reconstructed(root/'result',r)
            np.testing.assert_array_equal(actual_x,x); np.testing.assert_array_equal(actual_y,y)
            self.assertTrue(r['all_frozen_memberships_exact'])
            self.assertEqual(r['raw_rows_streamed'],2000)
            self.assertFalse(any(r['cross_cohort_input_groups'].values()))
            self.assertFalse(r['unhashed_staging_cache_used'])
            with self.assertRaisesRegex(ValueError,'Preserve existing'):
                cohort.reconstruct(data,parent,subsets,root/'result',lock,base,performance,True)
            (root/'result/confirmation_y.npy').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Prepared array differs'): cohort.load_reconstructed(root/'result',r)

    def test_corrupt_source_shard_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data,parent,subsets,base,performance,lock,*_=self.data_fixture(root)
            (subsets/'train.parquet').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Shard checksum'):
                cohort.reconstruct(data,parent,subsets,root/'result',lock,base,performance,True)
            self.assertFalse((root/'result/manifest.json').exists())

    def test_wrong_membership_or_anchor_labels_cannot_reach_scoring(self):
        for mode in ('membership','labels'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp); data,parent,subsets,base,performance,lock,*_=self.data_fixture(root)
                filename='confirmation_row_ids.npy' if mode=='membership' else 'development_y.npy'
                v=np.load(data/filename)
                if mode=='membership': v[0]=v[0]+1
                else: v[0]=(v[0]+1)%8
                np.save(data/filename,v,allow_pickle=False)
                # Even a rehashed bad array must fail row/content verification.
                m=json.loads((data/'manifest.json').read_text()); m['files'][filename]=sha256(data/filename)
                atomic_json(m,data/'manifest.json'); performance['summary']['data']=m
                lock['plan']['data_manifest_sha256']=sha256(data/'manifest.json')
                lock['data']=prep.metadata_preflight(data,parent,subsets,base,performance,lock['plan'])
                with self.assertRaises(ValueError):
                    cohort.reconstruct(data,parent,subsets,root/'result',lock,base,performance,True)

    def test_frozen_scoring_independent_metrics_and_tamper(self):
        logits=np.random.default_rng(20).normal(size=(800,8)).astype(np.float32)
        y=np.arange(800,dtype=np.int64)%8
        c=[alarms.calibrate(alarms.margins(logits)[0][y==alarms.BENIGN],t) for t in (100,500,1000,2000)]
        with patch.object(alarms,'calibrate',side_effect=AssertionError('No calibration')):
            r=final.fixed_scores(logits,y,c)
            self.assertEqual(final.independent_metrics(logits,y,c,r),5)
        bad=copy.deepcopy(r); bad['operating_points'][0]['calibration']['threshold']+=1
        with self.assertRaisesRegex(ValueError,'Threshold changed'): final.independent_metrics(logits,y,c,bad)
        bad=copy.deepcopy(r); bad['default_metrics']['macro_f1']+=.1
        with self.assertRaisesRegex(ValueError,'aggregate metrics'): final.independent_metrics(logits,y,c,bad)

    def test_all_seeds_must_pass_and_every_reference_far_is_flagged(self):
        _,lock,_,_,_,parent=final.read_contract(); records=prep.endpoint_records(parent)
        results=[{'seed':e['seed'],'arm':e['arm'],'operating_points':[
            {'calibration':p['calibration'],'confirmation_metrics':copy.deepcopy(p['audit_metrics'])}
            for p in records[e['seed'],e['arm']]['operating_points']]} for e in lock['models']]
        s=final.summarize(results,lock,parent); self.assertTrue(s['assessment']['all_seeds_pass'])
        self.assertEqual(len(s['comparisons']),36); self.assertEqual(len(s['descriptive']),16)
        candidate=next(r for r in results if r['seed']==17 and r['arm']=='block128')
        candidate['operating_points'][1]['confirmation_metrics']['per_class']['Spoofing']['recall']-=.1
        self.assertFalse(final.summarize(results,lock,parent)['assessment']['all_seeds_pass'])
        central=next(r for r in results if r['seed']==7 and r['arm']=='central_adam')
        central['operating_points'][1]['confirmation_metrics']['benign_false_alert_rate']+=.02
        c=final.summarize(results,lock,parent)['comparisons']
        self.assertFalse(next(r for r in c if r['seed']==7 and r['reference']=='central_adam' and r['target_basis_points']==500)['actual_far_comparable'])

    def test_complete_synthetic_run_and_independent_audit_no_fit_or_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            plan,base,alarm,performance,parent,x,y,_=model_fixtures.ConfirmationPreparationTests().fixture(root)
            entries=prep.registry(parent,plan)
            lock={'plan':{**plan,'planned_confirmation_forward_examples':12*len(y)},'models':entries}
            execution={'lock_sha256':'synthetic'}
            receipt={'status':'complete','files':{},'rows':len(y)}
            def build(*args,**kwargs):
                output=Path(args[3]); output.mkdir(); atomic_json(receipt,output/'manifest.json'); return receipt
            # Reporting tests above exercise the real paired assessment; here
            # synthetic fixtures test checkpoint loading, IO, and audit plumbing.
            with patch.object(final,'read_contract',return_value=(execution,lock,base,alarm,performance,parent)), \
                 patch.object(cohort,'reconstruct',side_effect=build), \
                 patch.object(cohort,'load_reconstructed',return_value=(x,y)), \
                 patch.object(final,'summarize',return_value={'assessment':{'all_seeds_pass':False}}), \
                 patch.object(alarms,'calibrate',side_effect=AssertionError('No calibration')), \
                 patch.object(final.subprocess,'check_output',return_value='synthetic-commit\n'):
                s=final.run(root,root,root,root/'result',True,root)
                self.assertEqual(s['forward_examples'],9600)
                self.assertEqual(s['new_training_runs'],0); self.assertFalse(s['thresholds_refitted'])
                a=final.audit(root/'result',root/'audit.json')
                self.assertEqual(a['independent_metric_endpoints'],60)
                self.assertEqual(a['additional_model_forward_examples'],0)
                with self.assertRaisesRegex(ValueError,'Preserve existing final'): final.run(root,root,root,root/'result',True,root)
                with self.assertRaisesRegex(ValueError,'Preserve existing audit'): final.audit(root/'result',root/'audit.json')
                path=root/'result/seed7-central_adam/logits.npy'; path.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'Endpoint artifact'): final.audit(root/'result',root/'bad-audit.json')


if __name__=='__main__': unittest.main()
