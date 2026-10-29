"""Independent gradient/state reference, budget tests and historical bridges."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.federated import mime_lite
from src.eval import mimelite_comparison as study
from tests.test_shared_optimizer_performance import SharedOptimizerPerformanceTests


class MimeLiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_independent_local_path_and_full_anchor_moments(self):
        model = torch.nn.Linear(1,1,bias=False).double()
        with torch.no_grad(): model.weight.fill_(.4)
        counts = [2,3]
        x = [torch.tensor([[1.],[2.]],dtype=torch.float64),torch.tensor([[3.],[4.],[5.]],dtype=torch.float64)]
        y = [a*.7 for a in x]
        recipe = {'lr':.03,'beta1':.6,'beta2':.8,'eps':.1}
        state = {'m':(torch.full_like(model.weight,.2),), 'v':(torch.full_like(model.weight,.4),), 'round':2}
        before = copy.deepcopy(state)
        criterion = torch.nn.MSELoss()
        def full(i):
            for j in range(counts[i]): yield x[i][j:j+1],y[i][j:j+1],7
        def batch(i,k): return x[i][k%counts[i]:k%counts[i]+1],y[i][k%counts[i]:k%counts[i]+1],7
        # Independent scalar arithmetic: d/dw MSE = 2*x*(w*x-y).
        expected_w, anchor = 0., 0.
        for i,n in enumerate(counts):
            weight=.4
            for k in range(3):
                xx,yy,_=batch(i,k)
                gradient=float((2*xx*(weight*xx-yy)).mean())
                weight -= .03*((1-.6)*gradient+.6*.2)/(.4**.5+.1)
            expected_w += n/5*weight
            anchor += n/5*float((2*x[i]*(.4*x[i]-y[i])).mean())
        result,moments,receipt=mime_lite.lite_round(model,state,counts,full,batch,criterion,recipe,3)
        self.assertAlmostEqual(result.weight.item(),expected_w,places=12)
        self.assertAlmostEqual(moments['m'][0].item(),.6*.2+.4*anchor,places=12)
        self.assertAlmostEqual(moments['v'][0].item(),.8*.4+.2*anchor**2,places=12)
        self.assertEqual(moments['round'],3)
        self.assertTrue(study.old.block.nested_equal(state,before))
        self.assertEqual(model.weight.item(),.4)
        self.assertEqual(receipt['work']['backward_examples'],11)
        self.assertEqual(receipt['work']['backward_batches'],11)
        self.assertEqual(receipt['estimated_payload_bytes'],5*2*8)
        self.assertEqual(receipt['logical_barriers'],1)
        with self.assertRaisesRegex(ValueError,'anchor row count'):
            mime_lite.lite_round(model,state,counts,lambda i:iter([]),batch,criterion,recipe,3)

    def test_exact_real_schedules(self):
        base=json.loads(study.old.training.PLAN.read_text());p=study.read_plan()
        expected={'central_adam':(46200,1),'shared_adam':(2310,1),'mimelite5':(42,5),
                  'mimelite20':(33,20),'mime':(26,5),'fedavg_adam':(116,10),'central_mime_base':(46200,1)}
        for arm,settings in p['arms'].items():
            ks=study.schedule(base,arm,settings,p['backward_examples_per_run'])
            self.assertEqual((len(ks),ks[-1]),expected[arm])
            n,c,b=base['fit_rows'],base['clients'],base['batch_size']
            total=len(ks)*b if arm.startswith('central_') else len(ks)*c*b if arm=='shared_adam' else \
                  sum((n if arm=='mime' or study.is_lite(arm) else 0)+c*k*b*(2 if arm=='mime' else 1) for k in ks)
            self.assertEqual(total,23100000)
        with self.assertRaisesRegex(ValueError,'valid final round'):
            study.schedule(base,'mimelite5',p['arms']['mimelite5'],21600000)

    def test_existing_updaters_remain_bit_exact(self):
        fixture=SharedOptimizerPerformanceTests();base=fixture.plan();data,criterion=fixture.fixture()
        for arm in ('central_adam','shared_adam','fedavg_adam','central_mime_base','mime'):
            a=study.make_state(base,arm,.003);b=study.old.training.make_state(base,arm,.003)
            for _ in range(2):
                study.advance(a,data,base,arm,{'lr':.003},2,criterion)
                study.old.training.advance(b,data,base,arm,.003,criterion)
            self.assertTrue(study.old.block.nested_equal(study.old.training.snapshot(a),study.old.training.snapshot(b)))

    def test_serialized_recovery_including_short_final_rounds(self):
        fixture=SharedOptimizerPerformanceTests();base=fixture.plan();data,criterion=fixture.fixture()
        for arm,budget in [('fedavg_adam',24),('mime',128),('mimelite5',104),('mimelite20',104)]:
            settings={'lr':.003,'local_steps':2}
            ks=study.schedule(base,arm,settings,budget)
            self.assertEqual(ks,[2,1])
            a=study.make_state(base,arm,.003)
            study.advance(a,data,base,arm,settings,ks[0],criterion)
            stream=io.BytesIO();torch.save(study.old.training.snapshot(a),stream);stream.seek(0)
            b=study.restore(torch.load(stream,weights_only=True),base,arm,.003)
            for state in (a,b):
                study.advance(state,data,base,arm,settings,ks[1],criterion)
                study.check_work(state,base,arm,data,ks)
                self.assertEqual(state['work']['backward_examples'],budget)
            self.assertTrue(study.old.block.nested_equal(study.old.training.snapshot(a),study.old.training.snapshot(b)))
            a['work']['anchor_examples']+=1
            with self.assertRaisesRegex(ValueError,'counter mismatch'):
                study.check_work(a,base,arm,data,ks)

    def test_interrupted_job_resumes_exactly_and_rejects_changed_artifact(self):
        fixture=SharedOptimizerPerformanceTests();base=fixture.plan();data,criterion=fixture.fixture()
        settings={'lr':.003,'local_steps':2};arm='mimelite5'
        ks=study.schedule(base,arm,settings,1008)
        x,y=data.arrays['development'];mask=np.arange(len(y))<8
        evaluation=(x,y,mask,{'inference_batch_size':8})
        with tempfile.TemporaryDirectory() as tmp:
            complete=Path(tmp)/'complete';resume=Path(tmp)/'resumed'
            complete.mkdir();resume.mkdir()
            args=(base,arm,settings,ks,data,criterion,{'test':True},evaluation,[])
            expected=study.run_one(complete,*args)
            original=study.advance;calls=0
            def interrupted(*a,**kw):
                nonlocal calls
                calls+=1
                if calls==4:raise RuntimeError('simulated interruption')
                return original(*a,**kw)
            with patch.object(study,'advance',side_effect=interrupted),self.assertRaisesRegex(RuntimeError,'interruption'):
                study.run_one(resume,*args)
            actual=study.run_one(resume,*args)
            for key in ('model_sha256','logits_sha256','sample_chain_sha256','work','development','recovery_exact'):
                self.assertEqual(actual[key],expected[key],key)
            self.assertEqual(study.run_one(resume,*args),actual)
            path=resume/f"seed{base['seed']}-{arm}"/'logits.npy'
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Completed output changed'):
                study.run_one(resume,*args)


if __name__ == '__main__':
    unittest.main()
