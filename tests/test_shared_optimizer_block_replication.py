"""The replication must preserve pairing and cannot average away a failed seed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.official_inventory import sha256
from src.eval import shared_optimizer_block_replication as screen
from src.eval import shared_optimizer_block_compression as block
from src.eval import shared_optimizer_performance as training
from src.eval import shared_optimizer_alarms as alarms
from src.eval import shared_optimizer_replication as rep
from src.eval.federated_class_safeguard import state_digest
from src.federated import block_shared_adam as codec
from src.models.research import atomic_save
from tests import test_shared_optimizer_compression as fixtures


class BlockReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def passing(self):
        plan,_,_,_,_,refs=screen.read_contract()
        results=[]
        for seed in (7,17,27):
            r=copy.deepcopy(screen.reference_for(refs,seed,'shared_adam'))
            r.update(arm='block128',settings=plan['settings'],reused_discovery=seed==7)
            r['work']['estimated_payload_bytes']=240255840
            results.append(r)
        return results,refs,plan

    def test_contract_preserves_candidate_gates_sources_and_budget(self):
        plan,*_=screen.read_contract(); old=block.read_contract()[0]
        self.assertEqual(plan['settings'],old['arms']['block128'])
        self.assertEqual(plan['screen'],old['screen'])
        self.assertEqual(plan['seeds'],[17,27])
        self.assertEqual(plan['new_training_backward_examples'],43200000)
        self.assertEqual(plan['new_recovery_backward_examples'],40000)
        self.assertTrue(set(block.sources()).issubset(screen.sources()))

    def test_both_new_seeds_must_pass_not_only_the_mean(self):
        results,refs,plan=self.passing()
        self.assertTrue(screen.assess(results,refs,plan)['all_seeds_pass'])
        block.primary(results[1],plan)['per_class']['Spoofing']['recall']-=.051
        a=screen.assess(results,refs,plan)
        self.assertFalse(a['all_seeds_pass'])
        self.assertIsNone(a['candidate_for_confirmation_preparation'])
        self.assertEqual([p['passes'] for p in a['per_seed']],[True,False,True])
        self.assertLess(np.mean([p['any_class_recall_drop_from_shared'] for p in a['per_seed']]),.05)
        self.assertFalse(a['confirmation_scoring_authorized'])

    def test_each_other_gate_is_retained(self):
        for key in ('macro_f1','benign_false_alert_rate','payload','fedavg_gain'):
            results,refs,plan=self.passing()
            m=block.primary(results[2],plan)
            if key=='macro_f1': m[key]-=.011
            elif key=='benign_false_alert_rate': m[key]+=.006
            elif key=='payload': results[2]['work']['estimated_payload_bytes']=300000000
            else: block.primary(screen.reference_for(refs,27,'fedavg_adam'),plan)['macro_f1']=m['macro_f1']-.079
            self.assertFalse(screen.assess(results,refs,plan)['all_seeds_pass'],key)

    def test_duplicate_or_missing_endpoints_are_rejected(self):
        results,refs,plan=self.passing()
        for bad in (results[:2],[results[0],results[0],results[2]]):
            with self.assertRaises(ValueError): screen.assess(bad,refs,plan)
        with self.assertRaises(ValueError): screen.reference_for(refs,99,'shared_adam')
        with self.assertRaises(ValueError): screen.reference_for([*refs,refs[0]],7,'central_adam')

    def test_mispaired_initialization_samples_work_and_settings_are_rejected(self):
        for key in ('initial_model_sha256','sample_chain_sha256','work','settings'):
            results,refs,plan=self.passing(); r=results[1]
            if key=='work': r[key]['backward_examples']+=1
            elif key=='settings': r[key]={**r[key],'block_size':64}
            else: r[key]='wrong-seed'
            with self.assertRaises(ValueError): screen.check_pairing(r,refs,plan)

    def test_comparisons_cover_every_seed_class_and_operating_point(self):
        results,refs,plan=self.passing()
        comparisons=screen.comparisons(results,refs,plan)
        self.assertEqual(len(comparisons),36)
        for r in comparisons:
            self.assertEqual(len(r['per_class_recall_difference']),8)
            if r['reference']=='shared_adam':
                self.assertEqual(r['block_minus_reference_macro_f1'],0.)
                self.assertEqual(r['block_minus_reference_far'],0.)
                self.assertTrue(all(v==0 for v in r['per_class_recall_difference'].values()))

    def test_descriptive_statistics_do_not_decide_promotion(self):
        results,refs,plan=self.passing()
        d=screen.describe(results,refs,plan)
        self.assertEqual(len(d),16)
        for r in d:
            self.assertEqual(r['macro_f1']['n'],3)
        m=next(r['macro_f1'] for r in d if r['arm']=='block128' and r['target_basis_points']==500)
        values=[block.primary(r,plan)['macro_f1'] for r in results]
        self.assertEqual(m['mean'],float(np.mean(values)))
        self.assertEqual(m['sample_sd'],float(np.std(values,ddof=1)))

    def test_runtime_mismatch_is_rejected_before_training(self):
        with patch.object(screen,'runtime',return_value={'python':'unmatched','torch':'unmatched','numpy':'unmatched'}):
            screen.read_contract()  # Static contract/evidence tests must work on CI too.
            with self.assertRaisesRegex(ValueError,'Reference runtime differs'): screen.read_contract(require_runtime=True)

    def test_independent_threshold_and_metric_checks_detect_tamper(self):
        logits=np.random.default_rng(3).normal(size=(800,8)).astype(np.float32)
        y=np.arange(800,dtype=np.int64)%8; calibration=np.arange(800)<400
        points=alarms.evaluate_scores(logits,y,calibration,[100,500,1000,2000])['operating_points']
        screen.independent_points(logits,y,calibration,points)
        bad=copy.deepcopy(points); bad[0]['calibration']['threshold']+=.01
        with self.assertRaisesRegex(ValueError,'Sorted threshold'): screen.independent_points(logits,y,calibration,bad)
        bad=copy.deepcopy(points); bad[0]['audit_metrics']['macro_f1']+=.01
        with self.assertRaisesRegex(ValueError,'Independent metrics'): screen.independent_points(logits,y,calibration,bad)

    def test_toy_training_recovery_audit_preservation_and_tamper(self):
        plan,_,alarm_plan,performance,_,_=screen.read_contract()
        base,data,criterion=fixtures.SharedOptimizerCompressionTests().fixture()
        x=np.random.default_rng(35).normal(size=(800,39)).astype(np.float32); y=np.arange(800,dtype=np.int64)%8
        calibration,groups=alarms.grouped_split(x,alarm_plan['split_domain'])
        refs=[]
        for seed in (7,17,27):
            recipe={**base,'seed':seed}; state=training.make_state(recipe,'shared_adam',.003)
            for _ in range(6): training.advance(state,data,recipe,'shared_adam',.003,criterion)
            evaluated=alarms.evaluate_scores(rep.infer(state['model'],x,8192),y,calibration,plan['false_alert_targets_basis_points'])
            for arm in ('central_adam','shared_adam','fedavg_adam'):
                refs.append({'seed':seed,'arm':arm,'initial_model_sha256':state_digest(training.initial_model(recipe)),
                    'sample_chain_sha256':state['sample_chain_sha256'],'work':copy.deepcopy(state['work']),**evaluated})
        state=codec.make_state(base,4,True,.003,128)
        for _ in range(6): codec.advance(state,data,base,.003,criterion)
        discovery={'seed':7,'arm':'block128','settings':plan['settings'],'learning_rate':.003,
                   'initial_model_sha256':state_digest(training.initial_model(base)),
                   'sample_chain_sha256':state['sample_chain_sha256'],'work':state['work'],
                   **alarms.evaluate_scores(rep.infer(state['visible'],x,8192),y,calibration,plan['false_alert_targets_basis_points'])}
        plan=copy.deepcopy(plan); plan.update(backward_examples_per_run=48,new_training_backward_examples=96,new_recovery_backward_examples=32)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); np.save(root/'mask.npy',calibration,allow_pickle=False)
            groups['calibration_mask_sha256']=sha256(root/'mask.npy')
            parent={'summary':{'split':groups,'results':[discovery]}}
            with patch.object(screen,'read_contract',return_value=(plan,base,alarm_plan,performance,parent,refs)), \
                 patch.object(rep,'FitData',return_value=data), \
                 patch.object(alarms,'load_development',return_value=(x,y,np.arange(len(y)))), \
                 patch.object(screen.subprocess,'check_output',return_value='synthetic-test-commit\n'):
                s=screen.run(root,root/'run')
                self.assertEqual(s['new_training_runs'],2); self.assertEqual(s['new_training_backward_examples'],96)
                self.assertEqual([r['reused_discovery'] for r in s['results']],[True,False,False])
                a=screen.audit(root,root/'run',root/'audit.json')
                self.assertEqual(a['new_recovery_backward_examples'],32)
                self.assertTrue(all(c['recovery_exact'] and c['same_seed_pairing_verified'] for c in a['checks']))
                with self.assertRaises(ValueError): screen.run(root,root/'run')
                with self.assertRaises(ValueError): screen.audit(root,root/'run',root/'audit.json')
                path=root/'run'/'seed17-block128'/'last.pt'
                bad=torch.load(path,weights_only=True); bad['state']['uplink_errors'][0][0].zero_(); atomic_save(bad,path)
                with self.assertRaisesRegex(ValueError,'Saved artifact differs'):
                    screen.audit(root,root/'run',root/'tamper.json')

    def test_mismatched_resume_identity_is_rejected(self):
        plan,c=screen.read_contract()[0],screen.contract()
        base,data,criterion=fixtures.SharedOptimizerCompressionTests().fixture()
        recipe={**base,'seed':17}; state=codec.make_state(recipe,4,True,.003,128)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            atomic_save({'identity':screen.identity(plan,c,27),'state':codec.snapshot(state),'training_seconds':0.},path/'last.pt')
            with self.assertRaisesRegex(ValueError,'Resume identity differs'):
                block.train_candidate(data,recipe,plan['settings'],.003,path,screen.identity(plan,c,17),criterion,2)


if __name__=='__main__':
    unittest.main()
