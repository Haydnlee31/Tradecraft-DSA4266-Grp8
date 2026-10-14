"""Synthetic full-MIME checks; these are not CICIoT2023 quality experiments."""
import copy
import io
import unittest

import torch
from torch import nn

from src.federated import mime_adam as mime
from src.federated.shared_adam import client_gradient, aggregate_gradients
from src.models.architectures import MLPClassifier, MLPConfig

RECIPE = {'lr': .001, 'beta1': .9, 'beta2': .99, 'eps': .001}


class MimeAdamTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(10)
        self.model = MLPClassifier(3,8,MLPConfig('mime-test',(8,4),.2,'layer')).double().train()
        self.batches = [(torch.randn(n,3,dtype=torch.float64), torch.arange(n)%8, i+11)
                        for i,n in enumerate((5,7))]
        self.counts = [5,7]
        self.criterion = nn.CrossEntropyLoss()

    def full(self, client):
        yield self.batches[client]

    def local(self, client, step):
        x,y,seed = self.batches[client]
        return x[:3],y[:3],seed+step+20

    def test_table5_moment_order_and_no_mutation(self):
        g = (torch.tensor([2., -1.], dtype=torch.float64),)
        s = {'m': (torch.tensor([.4,.2],dtype=torch.float64),),
             'v': (torch.tensor([.09,.16],dtype=torch.float64),), 'round':4}
        saved = copy.deepcopy(s)
        d = mime.directions(g,s,RECIPE)[0]
        torch.testing.assert_close(d, (.1*g[0]+.9*s['m'][0])/(s['v'][0].sqrt()+.001))
        new = mime.update_state(g,s,RECIPE)
        torch.testing.assert_close(new['m'][0], .9*s['m'][0]+.1*g[0])
        torch.testing.assert_close(new['v'][0], .99*s['v'][0]+.01*g[0].square())
        self.assertEqual(new['round'],5)
        for key in ('m','v'): self.assertTrue(torch.equal(s[key][0],saved[key][0]))
        # Updating v first would give a different direction: do not silently
        # substitute the usual Adam implementation for the published split.
        self.assertFalse(torch.allclose(d, (.1*g[0]+.9*s['m'][0])/(new['v'][0].sqrt()+.001)))

    def test_one_local_step_equals_independent_central_base_update(self):
        state = mime.initialize(self.model,initial_v=.01)
        local_gradients = [client_gradient(self.model,(x,y),self.criterion,seed) for x,y,seed in self.batches]
        common = aggregate_gradients(local_gradients,self.counts)
        expected = copy.deepcopy(self.model)
        with torch.no_grad():
            for p,g in zip(expected.parameters(),common):
                # Independent scalar-form reference with zero initial m.
                p.sub_(.001*.1*g/(.1+.001))
        result,new,receipt = mime.full_round(self.model,state,self.counts,self.full,self.local,self.criterion,RECIPE,1)
        for p,q in zip(result.parameters(),expected.parameters()):
            torch.testing.assert_close(p,q,atol=1e-14,rtol=1e-12)
        self.assertTrue(receipt['first_step_cancellation_exact'])
        self.assertEqual(new['round'],1)
        self.assertEqual(receipt['work']['backward_examples'],12+2*6)

    def test_multiple_steps_preserve_source_and_frozen_server_state(self):
        state = mime.initialize(self.model,initial_v=.01)
        saved_state, saved_model = copy.deepcopy(state),copy.deepcopy(self.model.state_dict())
        rng = torch.get_rng_state().clone()
        result,new,r = mime.full_round(self.model,state,self.counts,self.full,self.local,self.criterion,RECIPE,3)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(state['round'],0)
        for k in ('m','v'):
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(state[k],saved_state[k])))
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in saved_model.items()))
        self.assertFalse(all(torch.equal(a,b) for a,b in zip(result.parameters(),self.model.parameters())))
        common = aggregate_gradients([client_gradient(self.model,(x,y),self.criterion,s) for x,y,s in self.batches],self.counts)
        for v,g in zip(new['v'],common):
            torch.testing.assert_close(v, .99*.01+.01*g.square())
        self.assertEqual(r['work'],{'anchor_examples':12,'local_examples':18,'backward_examples':48,
                                   'backward_batches':14,'local_updates':6,'server_state_updates':1})
        self.assertEqual(r['estimated_payload_bytes'],6*self.model.parameter_bytes()*2)

    def test_streamed_anchor_accounts_for_short_last_batch(self):
        # Dropout off makes changing the full-gradient chunking comparable.
        model = copy.deepcopy(self.model).eval()
        state = mime.initialize(model,initial_v=.01)
        def chunks(client):
            x,y,s = self.batches[client]
            yield x[:3],y[:3],s
            yield x[3:],y[3:],s
        a,sa,_ = mime.full_round(model,state,self.counts,self.full,self.local,self.criterion,RECIPE,2)
        b,sb,_ = mime.full_round(model,state,self.counts,chunks,self.local,self.criterion,RECIPE,2)
        for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y,atol=1e-14,rtol=1e-12)
        for x,y in zip(sa['v'],sb['v']): torch.testing.assert_close(x,y,atol=1e-14,rtol=1e-12)

    def test_serialized_round_boundary_recovery(self):
        a,s,_ = mime.full_round(self.model,mime.initialize(self.model,.01),self.counts,self.full,self.local,self.criterion,RECIPE,2)
        buffer=io.BytesIO()
        torch.save({'model':a.state_dict(),'state':s},buffer)
        buffer.seek(0)
        restored=torch.load(buffer,weights_only=True)
        other=copy.deepcopy(self.model)
        other.load_state_dict(restored['model'])
        x,sx,rx=mime.full_round(a,s,self.counts,self.full,self.local,self.criterion,RECIPE,2)
        y,sy,ry=mime.full_round(other,restored['state'],self.counts,self.full,self.local,self.criterion,RECIPE,2)
        self.assertEqual(rx,ry)
        self.assertTrue(all(torch.equal(p,q) for p,q in zip(x.parameters(),y.parameters())))
        for key in ('m','v'): self.assertTrue(all(torch.equal(p,q) for p,q in zip(sx[key],sy[key])))

    def test_multistep_scalar_matches_analytic_frozen_state_recursion(self):
        model = nn.Linear(1,1,bias=False).double()
        with torch.no_grad(): model.weight.fill_(.5)
        populations = [([1.,3.],[0.,1.]), ([2.],[5.])]
        def full(client):
            xs,ys=populations[client]
            yield torch.tensor(xs,dtype=torch.float64)[:,None],torch.tensor(ys,dtype=torch.float64)[:,None],10
        def local(client,step):
            x,y,s=next(full(client))
            return x[:1],y[:1],s
        g = [sum(2*x*(.5*x-y) for x,y in zip(xs,ys))/len(xs) for xs,ys in populations]
        common = (2*g[0]+g[1])/3
        endpoints=[]
        for xs,ys in populations:
            w=.5
            for _ in range(3):
                corrected=2*xs[0]*(w*xs[0]-ys[0])-2*xs[0]*(.5*xs[0]-ys[0])+common
                w-=.001*(.1*corrected+.9*.2)/(.2+.001)
            endpoints.append(w)
        state=mime.initialize(model,.04)
        state['m'][0].fill_(.2)
        actual,new,_=mime.full_round(model,state,[2,1],full,local,nn.MSELoss(),RECIPE,3)
        self.assertAlmostEqual(actual.weight.item(),(2*endpoints[0]+endpoints[1])/3,places=14)
        self.assertAlmostEqual(new['m'][0].item(),.9*.2+.1*common,places=14)
        self.assertAlmostEqual(new['v'][0].item(),.99*.04+.01*common*common,places=14)

    def test_invalid_state_or_incomplete_full_anchor_rejected(self):
        state=mime.initialize(self.model)
        for overrides in ({'eps':0.},{'beta1':1.},{'lr':float('nan')}):
            with self.assertRaises(ValueError):
                mime.full_round(self.model,state,self.counts,self.full,self.local,self.criterion,dict(RECIPE,**overrides),1)
        with self.assertRaisesRegex(ValueError,'row count'):
            mime.full_round(self.model,state,[6,7],self.full,self.local,self.criterion,RECIPE,1)
        state['v'][0].fill_(-1.)
        with self.assertRaisesRegex(ValueError,'Negative'):
            mime.full_round(self.model,state,self.counts,self.full,self.local,self.criterion,RECIPE,1)


if __name__ == '__main__':
    unittest.main()
