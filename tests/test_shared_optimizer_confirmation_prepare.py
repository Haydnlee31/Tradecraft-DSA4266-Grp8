"""Preparation may replay development, but must not open the reserved cohort."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.data.official_inventory import sha256
from src.eval import shared_optimizer_confirmation_prepare as prep
from src.eval import shared_optimizer_alarms as alarms
from src.eval import shared_optimizer_performance as training
from src.eval import shared_optimizer_replication as rep
from src.eval.federated_class_safeguard import state_digest
from src.models.research import atomic_json, atomic_save
from tests import test_shared_optimizer_compression as fixtures


class ConfirmationPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root):
        plan,_,alarm,performance,_=prep.read_contract()
        plan=copy.deepcopy(plan)
        plan.update(development_rows=800,confirmation_rows=80,preparation_forward_examples=9600,
                    planned_confirmation_forward_examples=960)
        base,_,_=fixtures.SharedOptimizerCompressionTests().fixture()
        x=np.random.default_rng(92).normal(size=(800,39)).astype(np.float32)
        y=np.arange(800,dtype=np.int64)%8
        calibration,groups=alarms.grouped_split(x,alarm['split_domain'])
        np.save(root/'mask.npy',calibration,allow_pickle=False)
        groups['calibration_mask_sha256']=sha256(root/'mask.npy')
        refs=[]; results=[]
        for seed in plan['seeds']:
            for arm in plan['arms']:
                lr=.001 if arm=='central_adam' else .003
                recipe={**base,'seed':seed}
                model=training.initial_model(recipe); initial=state_digest(model)
                # Synthetic endpoints are not new experiments. Alter one state
                # so the hidden server model is detectably different from visible.
                with torch.no_grad():
                    next(model.parameters()).add_(.01*(1+plan['arms'].index(arm)))
                key='visible' if arm=='block128' else 'model'
                state={key:copy.deepcopy(model.state_dict()),'sample_chain_sha256':f'synthetic-seed-{seed}',
                       'work':{'backward_examples':48}}
                if key=='visible': state['model']=training.initial_model(recipe).state_dict()
                if arm=='block128':
                    folder=('outputs/shared-optimizer-block-compression-v1/block128' if seed==7 else
                            f'outputs/shared-optimizer-block-replication-v1/seed{seed}-block128')
                    logits_path=root/f'{folder}/logits.npy'
                elif seed==7:
                    folder=f'outputs/shared-optimizer-performance-v1/{arm}-lr{lr}'
                    logits_path=root/f'outputs/shared-optimizer-alarm-diagnostic-v1/{arm}-logits.npy'
                else:
                    folder=f'outputs/shared-optimizer-replication-v1/seed{seed}-{arm}'
                    logits_path=root/f'{folder}/logits.npy'
                path=root/folder; path.mkdir(parents=True)
                logits_path.parent.mkdir(parents=True,exist_ok=True)
                atomic_save({'state':state},path/'last.pt')
                logits=rep.infer(model,x,8192); np.save(logits_path,logits,allow_pickle=False)
                record={'seed':seed,'arm':arm,'learning_rate':lr,'initial_model_sha256':initial,
                        'sample_chain_sha256':state['sample_chain_sha256'],'work':state['work'],
                        'checkpoint_sha256':sha256(path/'last.pt'),'logits_sha256':sha256(logits_path),
                        ('delivered_model_sha256' if key=='visible' else 'final_model_sha256'):state_digest(model),
                        **alarms.evaluate_scores(logits,y,calibration,plan['false_alert_targets_basis_points'])}
                (results if key=='visible' else refs).append(record)
        evidence={'summary':{'references':refs,'results':results,'split':groups}}
        return plan,base,alarm,performance,evidence,x,y,calibration

    def test_frozen_contract_roster_budgets_and_boundary(self):
        plan,_,_,_,evidence=prep.read_contract()
        self.assertEqual(len(prep.endpoint_records(evidence)),12)
        self.assertEqual(plan['preparation_forward_examples'],4794624)
        self.assertEqual(plan['planned_confirmation_forward_examples'],4782024)
        self.assertEqual(plan['screen'],prep.previous.read_contract()[0]['screen'])
        self.assertTrue(plan['require_every_seed_to_pass'])
        self.assertFalse(plan['confirmation_scoring_permitted'])
        self.assertTrue(set(prep.previous.sources()).issubset(prep.sources()))

    def test_registry_has_all_original_thresholds_and_deployed_states(self):
        plan,_,_,_,evidence=prep.read_contract()
        entries=prep.registry(evidence,plan); records=prep.endpoint_records(evidence)
        self.assertEqual(len(entries),12)
        for e in entries:
            self.assertEqual(e['state_key'],'visible' if e['arm']=='block128' else 'model')
            old=records[e['seed'],e['arm']]
            self.assertEqual(e['calibrations'],[p['calibration'] for p in old['operating_points']])
            e['calibrations'][0]['threshold']+=1
            self.assertNotEqual(e['calibrations'][0],old['operating_points'][0]['calibration'])

    def test_contract_rejects_changed_primary_gate_roster_and_access(self):
        original=prep.read_contract()[0]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'plan.json'
            for key,value in (('require_every_seed_to_pass',False),('confirmation_scoring_permitted',True),
                              ('training_permitted',True),('threshold_refitting_permitted',True),
                              ('primary_target_basis_points',1000),('expected_endpoints',11),
                              ('seeds',[7,17]),('confirmation_rows',1)):
                changed=copy.deepcopy(original); changed[key]=value; atomic_json(changed,path)
                with patch.object(prep,'PLAN',path),self.assertRaises(ValueError): prep.read_contract()

    def test_missing_and_duplicate_endpoint_rejected(self):
        evidence=copy.deepcopy(prep.read_contract()[-1])
        evidence['summary']['results'][1]=evidence['summary']['results'][0]
        with self.assertRaisesRegex(ValueError,'Missing/duplicate'): prep.endpoint_records(evidence)
        evidence['summary']['results'].pop()
        with self.assertRaisesRegex(ValueError,'Missing/duplicate'): prep.endpoint_records(evidence)

    def test_registry_rejects_nonfinite_or_changed_target(self):
        plan,_,_,_,original=prep.read_contract()
        for key,value in (('threshold',float('nan')),('target_basis_points',99)):
            evidence=copy.deepcopy(original)
            evidence['summary']['results'][0]['operating_points'][0]['calibration'][key]=value
            with self.assertRaisesRegex(ValueError,'Frozen thresholds'): prep.registry(evidence,plan)

    def test_artifact_path_rejects_absolute_parent_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'project'; root.mkdir()
            (root/'escape').symlink_to(Path(temp),target_is_directory=True)
            for relative in ('../bad','/tmp/bad','escape/bad'):
                with self.assertRaises(ValueError): prep.within_root(root,relative)
            self.assertEqual(prep.within_root(root,'outputs/a.pt'),root.resolve()/'outputs/a.pt')

    def test_frozen_replay_matches_without_calling_calibrator_and_detects_tamper(self):
        logits=np.random.default_rng(55).normal(size=(800,8)).astype(np.float32)
        y=np.arange(800,dtype=np.int64)%8; mask=np.arange(800)<400
        expected=alarms.evaluate_scores(logits,y,mask,[100,500,1000,2000])
        fixed=[p['calibration'] for p in expected['operating_points']]
        with patch.object(alarms,'calibrate',side_effect=AssertionError('Refitting prohibited')):
            self.assertEqual(prep.frozen_replay(logits,y,mask,fixed),expected)
            for key in ('threshold','observed_false_alerts','boundary_ties','allowed_false_alerts'):
                bad=copy.deepcopy(fixed); bad[0][key]+=.1
                with self.assertRaises(ValueError): prep.frozen_replay(logits,y,mask,bad)

    def test_checkpoint_state_hash_work_and_visible_model_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); plan,base,_,_,evidence,*_=self.fixture(root)
            e=next(e for e in prep.registry(evidence,plan) if e['arm']=='block128')
            self.assertEqual(state_digest(prep.load_model(e,base,root)),e['model_sha256'])
            for key,value in (('checkpoint_sha256','wrong'),('initial_model_sha256','wrong'),
                              ('model_sha256','wrong'),('state_key','model'),
                              ('training_work',{}),('sample_chain_sha256','wrong')):
                with self.assertRaises(ValueError): prep.load_model({**e,key:value},base,root)

    def test_metadata_preflight_requires_only_metadata_never_reserved_or_shard_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data=root/'prepared'; parent=root/'parent'; subsets=root/'subsets'
            for p in (data,parent,subsets): p.mkdir()
            atomic_json({'n_samples_seen':40,'mean':[0.]*39,'scale':[1.]*39},data/'scaler.json')
            m={'features':[f'f{i}' for i in range(39)],'counts':{'confirmation':{'Benign':80}},
               'files':{'scaler.json':sha256(data/'scaler.json'),'confirmation_row_ids.npy':'recorded-not-opened'}}
            atomic_json(m,data/'manifest.json')
            for p in (parent/'manifest.json',parent/'protocol.json',subsets/'manifest.json',subsets/'recipe.json'):
                atomic_json({},p)
            base={'fit_rows':40,'historical_training_rows':2000000,'parent_manifest_sha256':sha256(parent/'manifest.json'),
                  'subset_manifest_sha256':sha256(subsets/'manifest.json')}
            plan={'data_manifest_sha256':sha256(data/'manifest.json')}
            view=SimpleNamespace(features=m['features'],rows=2000000,shards=[{'path':'not-present.parquet'}],
                                 raw_batches=Mock(side_effect=AssertionError('Raw data forbidden')))
            with patch.object(prep,'SubsetBatches',return_value=view), \
                 patch.object(np,'load',side_effect=AssertionError('Arrays forbidden')):
                r=prep.metadata_preflight(data,parent,subsets,base,{'summary':{'data':m}},plan)
            self.assertEqual(r['confirmation_row_ids_sha256'],'recorded-not-opened')
            self.assertFalse(r['reserved_row_ids_opened']); self.assertFalse(r['unhashed_staging_cache_used'])
            view.raw_batches.assert_not_called()
            atomic_json({'changed':True},data/'scaler.json')
            with self.assertRaisesRegex(ValueError,'Fit scaler differs'):
                prep.metadata_preflight(data,parent,subsets,base,{'summary':{'data':m}},plan)

    def test_full_synthetic_preparation_no_training_no_refitting_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); plan,base,alarm,performance,evidence,x,y,_=self.fixture(root)
            with patch.object(prep,'read_contract',return_value=(plan,base,alarm,performance,evidence)), \
                 patch.object(prep,'metadata_preflight',return_value={'confirmation_rows':80}), \
                 patch.object(alarms,'load_development',return_value=(x,y,np.arange(len(y)))), \
                 patch.object(alarms,'calibrate',side_effect=AssertionError('Refitting forbidden')), \
                 patch.object(alarms,'evaluate_scores',side_effect=AssertionError('Refitting forbidden')), \
                 patch.object(training,'advance',side_effect=AssertionError('Training forbidden')), \
                 patch.object(prep.subprocess,'check_output',return_value='synthetic-test-commit\n'):
                r=prep.prepare(root,root,root,root/'prepared',root)
                self.assertEqual(r['forward_examples'],9600)
                self.assertEqual(r['planned_confirmation_forward_examples'],960)
                self.assertEqual(len(r['checks']),12)
                self.assertEqual(sum(c['independent_frozen_operating_points'] for c in r['checks']),48)
                self.assertTrue(all(c['checkpoint_inference_exact'] and c['historical_endpoint_exact'] for c in r['checks']))
                for k in ('confirmation_opened','reserved_row_ids_opened','raw_shards_opened','test_opened',
                          'original_validation_opened','thresholds_refitted','cloud_used'):
                    self.assertFalse(r[k])
                self.assertEqual(r['new_model_fits'],0)
                before=sha256(root/'prepared/lock.json')
                with self.assertRaisesRegex(ValueError,'Preserve existing'):
                    prep.prepare(root,root,root,root/'prepared',root)
                self.assertEqual(sha256(root/'prepared/lock.json'),before)


if __name__=='__main__':
    unittest.main()
