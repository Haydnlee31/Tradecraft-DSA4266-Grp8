"""Compression must save actual bytes, preserve the identity path and recover."""
import copy
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import shared_optimizer_compression as screen
from src.eval import shared_optimizer_performance as training
from src.eval import shared_optimizer_alarms as alarms
from src.eval import shared_optimizer_replication as rep
from src.eval.federated_class_safeguard import state_digest
from src.federated import compressed_shared_adam as codec
from src.models.official_losses import build_official_criterion
from src.models.research import atomic_save


class SharedOptimizerCompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        base=json.loads(training.PLAN.read_text())
        base.update(fit_rows=40,clients=2,batch_size=4,backward_examples_per_arm=48)
        base['mime']['local_steps']=2
        x=np.random.default_rng(81).normal(size=(40,39)).astype(np.float32)
        y=np.arange(40,dtype=np.int64)%8
        data=SimpleNamespace(arrays={'fit':(x,y)},parts=[np.arange(13),np.arange(13,40)],
                             manifest={'counts':{'fit':dict(zip(CLASSES,[5]*8))}})
        data.batch=lambda rows: tuple(torch.from_numpy(a[rows].copy()) for a in (x,y))
        criterion=build_official_criterion('sqrt_weighted_ce',data.manifest['counts']['fit'],'fixed_train_mean')
        return base,data,criterion

    def test_fp32_packet_is_exact_and_length_includes_metadata(self):
        values=(torch.tensor([0.,-0.,1.,-2.]),torch.arange(15,dtype=torch.float32).reshape(3,5))
        packet=codec.encode(values,32,19)
        decoded=codec.decode(packet,values,32,19)
        self.assertEqual(len(packet),8+2*8+19*4)
        for a,b in zip(values,decoded):
            self.assertEqual(a.numpy().tobytes(),b.numpy().tobytes())

    def test_packed_roundtrip_bounds_odd_sizes_and_zero_tensors(self):
        for bits in (2,4):
            for n in (1,3,7,19):
                values=(torch.linspace(-3,3,n),torch.zeros(5))
                packet=codec.encode(values,bits,2)
                decoded=codec.decode(packet,values,bits,2)
                self.assertEqual(len(packet),8+16+(n*bits+7)//8+(5*bits+7)//8)
                qmax=2**(bits-1)-1
                self.assertLessEqual(float((values[0]-decoded[0]).abs().max()),3/qmax/2+1e-6)
                self.assertTrue(torch.equal(decoded[1],values[1]))

    def test_decoder_rejects_sequence_length_scale_reserved_and_padding(self):
        refs=(torch.tensor([1.,-1.,0.]),)
        packet=codec.encode(refs,2,1)
        for bad in (packet[:-1],packet+b'\x00'):
            with self.assertRaises(ValueError): codec.decode(bad,refs,2,1)
        with self.assertRaises(ValueError): codec.decode(packet,refs,2,2)
        # Last byte: three two-bit values and two padding bits.
        for last in ((packet[-1]&252)|3,packet[-1]|192):
            with self.assertRaises(ValueError): codec.decode(packet[:-1]+bytes([last]),refs,2,1)
        bad=packet[:12]+struct.pack('<f',float('nan'))+packet[16:]
        with self.assertRaises(ValueError): codec.decode(bad,refs,2,1)

    def test_encoder_rejects_invalid_values(self):
        for value in (torch.tensor([float('nan')]),torch.tensor([float('inf')]),torch.ones(2,dtype=torch.float64),torch.empty(0)):
            with self.assertRaises(ValueError): codec.encode((value,),4,1)
        for bits in (1,3,8):
            with self.assertRaises(ValueError): codec.encode((torch.ones(3),),bits,1)

    def test_error_feedback_retains_omitted_information(self):
        g=torch.tensor([1.,.1,.01,-.2])
        residual=torch.zeros_like(g); delivered=torch.zeros_like(g)
        # Use enough messages to cross the rounding boundary. Accumulating a
        # small residual does not promise that every coordinate is sent at once.
        for step in range(100):
            corrected=g+residual
            q=codec.decode(codec.encode((corrected,),2,step),(g,),2,step)[0]
            residual=corrected-q; delivered+=q
        torch.testing.assert_close(delivered+residual,100*g,atol=1e-5,rtol=1e-6)
        self.assertGreater(float(delivered[2]),0.)

    def test_identity_matches_original_shared_adam_state_exactly(self):
        base,data,criterion=self.fixture()
        original=training.make_state(base,'shared_adam',.003)
        state=codec.make_state(base,32,True,.003)
        for _ in range(6):
            training.advance(original,data,base,'shared_adam',.003,criterion)
            codec.advance(state,data,base,.003,criterion)
        old,new=training.snapshot(original),codec.snapshot(state)
        for k in ('model','optimizer','moments','unit','sample_chain_sha256'):
            self.assertTrue(screen.nested_equal(old[k],new[k]),k)
        self.assertEqual(state_digest(state['visible']),state_digest(original['model']))
        screen.check_work(state,base)

    def test_recovery_includes_errors_target_replica_moments_and_bytes(self):
        base,data,criterion=self.fixture()
        for bits,feedback in [(32,True),(4,True),(2,True),(4,False)]:
            state=codec.make_state(base,bits,feedback,.003)
            for _ in range(3): codec.advance(state,data,base,.003,criterion)
            stream=io.BytesIO(); torch.save(codec.snapshot(state),stream); stream.seek(0)
            resumed=codec.restore(torch.load(stream,weights_only=True),base,.003)
            for _ in range(3):
                codec.advance(state,data,base,.003,criterion)
                codec.advance(resumed,data,base,.003,criterion)
            self.assertTrue(screen.nested_equal(codec.snapshot(state),codec.snapshot(resumed)))
            screen.check_work(state,base)
            norm=sum(float(e.abs().sum()) for es in state['uplink_errors'] for e in es)
            self.assertEqual(norm>0,feedback and bits!=32)
            if bits!=32:
                self.assertNotEqual(state_digest(state['visible']),state_digest(state['model']))

    def test_wire_budget_counts_startup_and_both_directions(self):
        base=json.loads(training.PLAN.read_text()); model=training.initial_model(base)
        for bits in (2,4,32):
            wire=codec.expected_wire(model,base,bits,2160)
            self.assertEqual(wire['uplink_bytes'],wire['downlink_bytes'])
            self.assertEqual(wire['initialization_bytes'],20*codec.payload_size(tuple(model.parameters()),32))
        four=sum(codec.expected_wire(model,base,4,2160).values())
        two=sum(codec.expected_wire(model,base,2,2160).values())
        self.assertLess(four/1761177600,.15)
        self.assertLess(two/1761177600,.08)

    def test_frozen_plan_and_reference_contract(self):
        p,base,_,_,_,refs=screen.read_contract()
        self.assertEqual(p['new_training_backward_examples'],86400000)
        self.assertEqual(p['new_recovery_backward_examples'],80000)
        self.assertEqual(base['seed'],7)
        self.assertEqual(refs['shared_adam']['work']['logical_barriers'],2160)

    def test_screen_keeps_class_and_alarm_limits_and_excludes_control(self):
        p,_,_,_,_,refs=screen.read_contract()
        results=[]
        for name in p['arms']:
            r=copy.deepcopy(refs['shared_adam']); r['arm']=name
            r['work']['estimated_payload_bytes']=100000000 if name=='ef2' else 200000000
            results.append(r)
        self.assertEqual(screen.assess(results,refs,p)['candidate_for_replication'],'ef2')
        screen.primary(results[2],p)['per_class'][CLASSES[4]]['recall']=0.
        self.assertEqual(screen.assess(results,refs,p)['candidate_for_replication'],'ef4')
        screen.primary(results[1],p)['benign_false_alert_rate']+=.006
        self.assertIsNone(screen.assess(results,refs,p)['candidate_for_replication'])

    def test_complete_toy_run_recovery_audit_and_artifact_tamper(self):
        plan,_,alarm_plan,performance,parent,_=screen.read_contract()
        base,data,criterion=self.fixture()
        rng=np.random.default_rng(35)
        x=rng.normal(size=(800,39)).astype(np.float32); y=np.arange(800,dtype=np.int64)%8
        calibration,groups=alarms.grouped_split(x,alarm_plan['split_domain'])
        original=training.make_state(base,'shared_adam',.003)
        for _ in range(6): training.advance(original,data,base,'shared_adam',.003,criterion)
        evaluated=alarms.evaluate_scores(rep.infer(original['model'],x,8192),y,calibration,plan['false_alert_targets_basis_points'])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); checkpoint=root/'old.pt'
            atomic_save({'state':training.snapshot(original)},checkpoint)
            np.save(root/'mask.npy',calibration,allow_pickle=False)
            groups['calibration_mask_sha256']=sha256(root/'mask.npy')
            plan=copy.deepcopy(plan); plan.update(backward_examples_per_run=48,new_training_backward_examples=192,
                                                new_recovery_backward_examples=64)
            parent=copy.deepcopy(parent); parent['summary']['split']=groups
            refs={a:{'checkpoint_sha256':sha256(checkpoint),'final_model_sha256':state_digest(original['model']),
                     'sample_chain_sha256':original['sample_chain_sha256'],'work':original['work'],**evaluated}
                  for a in ('central_adam','shared_adam','fedavg_adam')}
            contract=(plan,base,alarm_plan,performance,parent,refs)
            with patch.object(screen,'read_contract',return_value=contract), \
                 patch.object(rep,'FitData',return_value=data), \
                 patch.object(alarms,'load_development',return_value=(x,y,np.arange(len(y)))), \
                 patch.object(screen.subprocess,'check_output',return_value='synthetic-test-commit\n'):
                s=screen.run(root,root/'run',checkpoint)
                self.assertTrue(s['fp32_bridge_exact'])
                a=screen.audit(root,root/'run',root/'audit.json',checkpoint)
                self.assertEqual(len(a['checks']),4)
                self.assertEqual(a['new_recovery_backward_examples'],64)
                with self.assertRaises(ValueError): screen.run(root,root/'run',checkpoint)
                with self.assertRaises(ValueError): screen.audit(root,root/'run',root/'audit.json',checkpoint)
                path=root/'run'/'ef4'/'last.pt'
                bad=torch.load(path,weights_only=True); bad['state']['uplink_errors'][0][0].zero_()
                atomic_save(bad,path)
                with self.assertRaisesRegex(ValueError,'Saved artifact differs'):
                    screen.audit(root,root/'run',root/'tamper.json',checkpoint)


if __name__=='__main__':
    unittest.main()
