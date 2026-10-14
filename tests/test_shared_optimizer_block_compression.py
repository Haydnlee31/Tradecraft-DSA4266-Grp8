"""Block scales must preserve the old training semantics and enforce the gates."""
import copy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.official_inventory import sha256
from src.eval import shared_optimizer_block_compression as screen
from src.eval import shared_optimizer_compression as previous
from src.eval import shared_optimizer_performance as training
from src.eval import shared_optimizer_alarms as alarms
from src.eval import shared_optimizer_replication as rep
from src.eval.federated_class_safeguard import state_digest
from src.federated import block_shared_adam as codec
from src.federated import compressed_shared_adam as original
from src.models.research import atomic_save
from tests import test_shared_optimizer_compression as fixtures


class BlockCompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        return fixtures.SharedOptimizerCompressionTests().fixture()

    def test_tensor_wire_compatibility(self):
        values=(torch.linspace(-2,2,257),torch.zeros(3))
        for bits in (4,32):
            packet=codec.encode(values,bits,9,None)
            self.assertEqual(packet,original.encode(values,bits,9))
            self.assertTrue(previous.nested_equal(codec.decode(packet,values,bits,9,None),original.decode(packet,values,bits,9)))

    def test_block_roundtrip_boundary_zero_and_short_blocks(self):
        for n in (1,127,128,129,255,256,257):
            values=(torch.linspace(-3,3,n),torch.zeros(5))
            packet=codec.encode(values,4,7,128)
            self.assertEqual(len(packet),codec.payload_size(values,4,128))
            decoded=codec.decode(packet,values,4,7,128)
            self.assertTrue(torch.equal(decoded[1],values[1]))
            for start in range(0,n,128):
                a,b=values[0][start:start+128],decoded[0][start:start+128]
                self.assertLessEqual(float((a-b).abs().max()),float(a.abs().max())/14+1e-6)

    def test_outlier_in_other_block_does_not_set_small_block_scale(self):
        a=torch.ones(256); a[0]=700.
        tensor=codec.decode(codec.encode((a,),4,1,None),(a,),4,1,None)[0]
        block=codec.decode(codec.encode((a,),4,1,128),(a,),4,1,128)[0]
        self.assertTrue(torch.equal(block[128:],a[128:]))
        self.assertGreater(float((tensor[128:]-a[128:]).abs().sum()),0.)

    def test_invalid_packets_are_rejected(self):
        refs=(torch.tensor([1.,-1.,0.]),)
        p=codec.encode(refs,4,1,128)
        bads=[p[:-1],p+b'\x00',p[:12]+struct.pack('<f',float('nan'))+p[16:],
              p[:12]+struct.pack('<f',0.)+p[16:],p[:-1]+bytes([p[-1]|240]),
              p[:16]+bytes([(p[16]&240)|15])+p[17:],p[:8]+struct.pack('<I',4)+p[12:]]
        for bad in bads:
            with self.assertRaises(ValueError): codec.decode(bad,refs,4,1,128)
        with self.assertRaises(ValueError): codec.decode(p,refs,4,2,128)
        with self.assertRaises(ValueError): codec.decode(p,refs,4,1,None)

    def test_invalid_values_and_unfrozen_layouts_are_rejected(self):
        for value in (torch.empty(0),torch.tensor([float('inf')]),torch.tensor([float('nan')]),torch.ones(3,dtype=torch.float64)):
            with self.assertRaises(ValueError): codec.encode((value,),4,1,128)
        for bits,block in ((2,128),(4,64),(4,256)):
            with self.assertRaises(ValueError): codec.encode((torch.ones(3),),bits,1,block)
        for seq in (-1,2**32,1.5):
            with self.assertRaises(ValueError): codec.encode((torch.ones(3),),4,seq,128)

    def test_payload_counts_scales_metadata_startup_and_recipients(self):
        base=json.loads(training.PLAN.read_text()); model=training.initial_model(base)
        ps=tuple(model.parameters())
        self.assertEqual(sum((p.numel()+127)//128 for p in ps),45)
        self.assertEqual(codec.payload_size(ps,4,128),2776)
        self.assertEqual(len(codec.encode(ps,4,1,128)),2776)
        self.assertEqual(sum(codec.expected_wire(model,base,4,2160,128).values()),240255840)
        self.assertEqual(sum(codec.expected_wire(model,base,4,2160,None).values()),228159840)

    def test_tensor_training_matches_old_entire_state(self):
        base,data,criterion=self.fixture()
        old=original.make_state(base,4,True,.003); new=codec.make_state(base,4,True,.003,None)
        for _ in range(6):
            original.advance(old,data,base,.003,criterion); codec.advance(new,data,base,.003,criterion)
        saved=codec.snapshot(new); self.assertIsNone(saved.pop('block_size'))
        self.assertTrue(previous.nested_equal(saved,original.snapshot(old)))
        screen.check_work(new,base)

    def test_serialized_recovery_includes_errors_both_models_and_moments(self):
        base,data,criterion=self.fixture()
        for block in (None,128):
            state=codec.make_state(base,4,True,.003,block)
            for _ in range(3): codec.advance(state,data,base,.003,criterion)
            stream=io.BytesIO(); torch.save(codec.snapshot(state),stream); stream.seek(0)
            resumed=codec.restore(torch.load(stream,weights_only=True),base,.003)
            for _ in range(3):
                codec.advance(state,data,base,.003,criterion); codec.advance(resumed,data,base,.003,criterion)
            self.assertTrue(previous.nested_equal(codec.snapshot(state),codec.snapshot(resumed)))
            self.assertGreater(sum(float(e.abs().sum()) for errors in state['uplink_errors'] for e in errors),0.)
            screen.check_work(state,base)

    def test_feedback_conservation(self):
        g=torch.linspace(-1,1,259); g[0]=7.
        residual=torch.zeros_like(g); delivered=torch.zeros_like(g)
        for step in range(30):
            corrected=g+residual
            decoded=codec.decode(codec.encode((corrected,),4,step,128),(g,),4,step,128)[0]
            residual=corrected-decoded; delivered+=decoded
        torch.testing.assert_close(delivered+residual,30*g,atol=1e-5,rtol=1e-6)

    def test_contract_and_unchanged_safeguards(self):
        p,base,_,_,_,_=screen.read_contract(); old=previous.read_contract()[0]
        self.assertEqual(p['new_training_backward_examples'],43200000)
        self.assertEqual(p['new_recovery_backward_examples'],40000)
        self.assertEqual(p['screen']['any_class_recall_drop_from_shared_max'],old['screen']['any_class_recall_drop_from_shared_max'])
        self.assertEqual(p['screen']['payload_ratio_to_historical_shared_max'],.15)

    def test_all_gates_are_required_and_bridge_not_eligible(self):
        p,_,_,_,_,refs=screen.read_contract()
        results=[]
        for arm in p['arms']:
            r=copy.deepcopy(refs['shared_adam']); r['arm']=arm; r['work']['estimated_payload_bytes']=240255840
            results.append(r)
        self.assertEqual(screen.assess(results,refs,p)['candidate_for_replication'],'block128')
        screen.primary(results[1],p)['per_class']['Spoofing']['recall']-=.051
        self.assertIsNone(screen.assess(results,refs,p)['candidate_for_replication'])

    def test_f1_alarm_payload_and_fedavg_advantage_gates(self):
        p,_,_,_,_,refs=screen.read_contract()
        for key,bad in [('macro_f1',screen.primary(refs['shared_adam'],p)['macro_f1']-.011),
                        ('benign_false_alert_rate',screen.primary(refs['shared_adam'],p)['benign_false_alert_rate']+.006),
                        ('payload',300000000),('fedavg_gain',.079)]:
            results=[]; local_refs=copy.deepcopy(refs)
            for arm in p['arms']:
                r=copy.deepcopy(refs['shared_adam']); r['arm']=arm
                r['work']['estimated_payload_bytes']=240255840; results.append(r)
            if key=='payload': results[1]['work']['estimated_payload_bytes']=bad
            elif key=='fedavg_gain':
                screen.primary(local_refs['fedavg_adam'],p)['macro_f1']=screen.primary(results[1],p)['macro_f1']-bad
            else: screen.primary(results[1],p)[key]=bad
            self.assertIsNone(screen.assess(results,local_refs,p)['candidate_for_replication'],key)

    def test_toy_run_independent_audit_preservation_and_tamper(self):
        plan,_,alarm_plan,performance,parent,refs=screen.read_contract()
        base,data,criterion=self.fixture()
        x=np.random.default_rng(35).normal(size=(800,39)).astype(np.float32); y=np.arange(800,dtype=np.int64)%8
        calibration,groups=alarms.grouped_split(x,alarm_plan['split_domain'])
        old=original.make_state(base,4,True,.003)
        for _ in range(6): original.advance(old,data,base,.003,criterion)
        evaluated=alarms.evaluate_scores(rep.infer(old['visible'],x,8192),y,calibration,plan['false_alert_targets_basis_points'])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); checkpoint=root/'old.pt'
            atomic_save({'state':original.snapshot(old)},checkpoint)
            np.save(root/'mask.npy',calibration,allow_pickle=False); groups['calibration_mask_sha256']=sha256(root/'mask.npy')
            parent=copy.deepcopy(parent); parent['summary']['split']=groups
            parent['summary']['results']=[{'arm':'ef4','checkpoint_sha256':sha256(checkpoint),
                'delivered_model_sha256':state_digest(old['visible']),**evaluated}]
            plan=copy.deepcopy(plan); plan.update(backward_examples_per_run=48,new_training_backward_examples=96,new_recovery_backward_examples=32)
            refs=copy.deepcopy(refs)
            for r in refs.values(): r['sample_chain_sha256']=old['sample_chain_sha256']
            with patch.object(screen,'read_contract',return_value=(plan,base,alarm_plan,performance,parent,refs)), \
                 patch.object(rep,'FitData',return_value=data), \
                 patch.object(alarms,'load_development',return_value=(x,y,np.arange(len(y)))), \
                 patch.object(screen.subprocess,'check_output',return_value='synthetic-test-commit\n'):
                s=screen.run(root,root/'run',checkpoint); self.assertTrue(s['tensor_bridge_exact'])
                a=screen.audit(root,root/'run',root/'audit.json',checkpoint)
                self.assertEqual(len(a['checks']),2); self.assertEqual(a['new_recovery_backward_examples'],32)
                with self.assertRaises(ValueError): screen.run(root,root/'run',checkpoint)
                with self.assertRaises(ValueError): screen.audit(root,root/'run',root/'audit.json',checkpoint)
                path=root/'run'/'block128'/'last.pt'
                bad=torch.load(path,weights_only=True); bad['state']['uplink_errors'][0][0].zero_(); atomic_save(bad,path)
                with self.assertRaisesRegex(ValueError,'Saved artifact differs'):
                    screen.audit(root,root/'run',root/'tamper.json',checkpoint)


if __name__=='__main__':
    unittest.main()
