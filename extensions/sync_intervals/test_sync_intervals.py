"""Numerical bridge, sample identity, persistent-Adam recovery and isolation."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from extensions.fedntd.test_fedntd import FedNTDTests, original_functions
from extensions.sync_intervals import core, run
from src.federated import shared_adam as shared


class IntervalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        plan = json.loads(run.PLAN.read_text())
        plan.update(clients=2,fit_rows=40,batch_size=4,backward_examples_per_arm=320,checkpoint_every=1)
        data, criterion = FedNTDTests().fixture()
        return plan,data,criterion

    def equal(self,a,b):
        if isinstance(a,torch.Tensor):
            self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.equal(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.equal(x,y)
        else:self.assertEqual(a,b)

    def test_k1_weights_moments_samples_bit_exact_original_shared_adam(self):
        plan,data,criterion = self.fixture()
        original = original_functions()
        legacy = {**plan,'mime':{'local_steps':20}}
        old = original['make_state'](legacy,'shared_adam',plan['server_learning_rate'])
        new = run.make_state(plan,plan['arms'][0])
        before = torch.get_rng_state().clone()
        for _ in range(23):
            original['advance'](old,data,legacy,'shared_adam',plan['server_learning_rate'],criterion)
            run.advance(new,data,plan,plan['arms'][0],criterion)
        self.equal(old['model'].state_dict(),new['model'].state_dict())
        self.equal(old['optimizer'].state_dict(),new['optimizer'].state_dict())
        self.assertEqual(old['sample_chain_sha256'],new['traversal_sample_chain'])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_normalized_message_equals_independent_local_sgd_displacement(self):
        # Independent closed-form quadratic trajectory, with unequal populations.
        model = torch.nn.Linear(1,1,bias=False).double()
        with torch.no_grad():model.weight.fill_(2.)
        ref = copy.deepcopy(model)
        recipe = {'lr':.003,'betas':(.9,.99),'eps':.001,'weight_decay':0.}
        optimizer, refopt = shared.make_optimizer(model,recipe),shared.make_optimizer(ref,recipe)
        counts, k, eta = [3,7],4,.1
        targets = [0.,1.]
        def batch(i,j):return torch.ones(2,1,dtype=torch.float64),torch.full((2,1),targets[i],dtype=torch.float64)
        def loss(logits,y):return ((logits-y)**2).mean()
        end = [2.,2.]
        for _ in range(k):
            end = [w-eta*2*(w-target) for w,target in zip(end,targets)]
        expected = sum(n/sum(counts)*(2-w)/(eta*k) for n,w in zip(counts,end))
        shared.apply_gradient(ref,refopt,(torch.tensor([[expected]],dtype=torch.float64),))
        core.interval_step(model,optimizer,counts,batch,loss,k,eta,7)
        torch.testing.assert_close(model.weight,ref.weight,rtol=1e-14,atol=1e-14)
        for key in ('exp_avg','exp_avg_sq'):
            torch.testing.assert_close(optimizer.state[model.weight][key],refopt.state[ref.weight][key],rtol=1e-14,atol=1e-14)

    def test_fedavg_adam_control_exact_original(self):
        plan,data,criterion = self.fixture()
        arm=plan['arms'][-1]
        original=original_functions()
        legacy={**plan,'mime':{'local_steps':20}}
        old=original['make_state'](legacy,'fedavg_adam',plan['control_adam_learning_rate'])
        new=run.make_state(plan,arm)
        for _ in range(2):
            original['advance'](old,data,legacy,'fedavg_adam',plan['control_adam_learning_rate'],criterion)
            run.advance(new,data,plan,arm,criterion)
        self.equal(old['model'].state_dict(),new['model'].state_dict())
        self.assertEqual(old['sample_chain_sha256'],new['traversal_sample_chain'])

    def test_sgd_control_matches_independent_weight_averaging(self):
        plan,data,criterion=self.fixture()
        model=run.initial_model(plan)
        reference=copy.deepcopy(model)
        local_models=[]
        batch=lambda i,j:data.batch(data.parts[i][:4])
        for i in range(2):
            local=copy.deepcopy(model)
            opt=torch.optim.SGD(local.parameters(),lr=.003)
            for _ in range(3):
                opt.zero_grad(); x,y=batch(i,0);criterion(local(x),y).backward();opt.step()
            local_models.append(local)
        with torch.no_grad():
            for p,a,b in zip(reference.parameters(),local_models[0].parameters(),local_models[1].parameters()):
                p.copy_(a*(13/40)+b*(27/40))
        core.fedavg_step(model,[13,27],batch,criterion,3,.003,7)
        for p,q in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-5,atol=3e-7)

    def test_same_per_client_stream_and_work_all_intervals(self):
        plan,data,criterion=self.fixture()
        chains=[]
        for arm in plan['arms']:
            state=run.make_state(plan,arm)
            for _ in range(40//arm['interval']):run.advance(state,data,plan,arm,criterion)
            chains.append(state['client_sample_chains'])
            self.assertEqual(state['work']['backward_examples'],320)
            self.assertEqual(state['work']['synchronization_rounds'],40//arm['interval'])
            self.assertEqual(state['work']['estimated_payload_bytes'],2*2*20384*40//arm['interval'])
        self.assertTrue(all(c==chains[0] for c in chains))

    def test_serialized_recovery_preserves_adam_moments_every_arm(self):
        plan,data,criterion=self.fixture()
        for arm in plan['arms']:
            with self.subTest(arm=arm['name']):
                state=run.make_state(plan,arm)
                run.advance(state,data,plan,arm,criterion)
                buffer=io.BytesIO();torch.save(run.snapshot(state),buffer);buffer.seek(0)
                restored=run.restore(torch.load(buffer,weights_only=True),plan,arm)
                run.advance(state,data,plan,arm,criterion)
                run.advance(restored,data,plan,arm,criterion)
                self.equal(run.snapshot(state),run.snapshot(restored))

    def test_invalid_settings_and_output_guard(self):
        plan,data,criterion=self.fixture()
        self.assertIsNone(run.validate_plan(plan))
        for key,value in [('local_learning_rate',0),('server_learning_rate',float('nan')),('backward_examples_per_arm',321)]:
            bad=copy.deepcopy(plan);bad[key]=value
            with self.assertRaises(ValueError):run.validate_plan(bad)
        with self.assertRaisesRegex(ValueError,'Output must'):
            run.run('/not-opened',run.ROOT/'reports')
        bad=copy.deepcopy(plan);bad['arms'][1]['name']='../escape'
        with self.assertRaises(ValueError):run.validate_plan(bad)

    def test_disk_runner_interruption_recovery_and_contract_refusal(self):
        plan,_,_=self.fixture()
        runs=run.HERE/'runs';runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory(dir=runs) as out:
            root=Path(tmp);FedNTDTests().write_bundle(root)
            plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan))
            full=run.run(root,Path(out)/'full',plan_path)
            original=run.advance
            calls=0
            def interrupt(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==44:raise RuntimeError('synthetic interruption')
                return original(*args,**kwargs)
            with patch.object(run,'advance',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    run.run(root,Path(out)/'resume',plan_path)
            self.assertFalse((Path(out)/'resume/RUNNING.lock').exists())
            recovered=run.run(root,Path(out)/'resume',plan_path)
            for a,b in zip(full['results'],recovered['results']):
                for key in ('development_metrics','work','client_sample_chains','traversal_sample_chain'):
                    self.equal(a[key],b[key])
                name=a['arm']['name']
                saved=[torch.load(Path(out)/d/name/'last.pt',weights_only=True)['state'] for d in ('full','resume')]
                self.equal(*saved)
            with self.assertRaisesRegex(ValueError,'Completed run'):
                run.run(root,Path(out)/'resume',plan_path)
            plan['local_learning_rate']=.004;plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'contract changed'):
                run.run(root,Path(out)/'resume',plan_path)


if __name__=='__main__':unittest.main()
