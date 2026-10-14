"""Synthetic FIT-only sampling, historic bridge and epoch recovery fixtures."""
import io
import unittest
import numpy as np
import torch
from src.eval import hard_example_flow as h


class HardExampleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.x = np.random.default_rng(1).normal(size=(256, 9)).astype(np.float32)
        self.y = np.repeat(np.arange(8, dtype=np.int64), 32)
        self.g = np.arange(256)
        self.pop = np.arange(1, 9)*32
        self.s = dict(h.SETTINGS, epochs=3, warmup_epochs=1, per_class=4, steps_per_epoch=3)

    def test_ranking_is_within_class_and_ties_stable(self):
        losses = np.tile(np.arange(32, dtype=np.float32), 8)
        pools = h.hard_pools(self.y, losses)
        for c,p in enumerate(pools):
            np.testing.assert_array_equal(p, np.arange(c*32+24,c*32+32))
        ties = h.hard_pools(self.y, np.ones(256))
        for c,p in enumerate(ties):
            np.testing.assert_array_equal(p,np.arange(c*32,c*32+8))

    def test_mixture_distinct_balanced_and_half_forced_hard(self):
        pools = h.hard_pools(self.y,np.arange(256,dtype=float))
        a = list(h.mixed_batches(self.y,pools,7,2,5,4))
        b = list(h.mixed_batches(self.y,pools,7,2,5,4))
        for take, other in zip(a,b):
            np.testing.assert_array_equal(take,other)
            self.assertEqual(len(np.unique(take)),32)
            self.assertEqual(np.bincount(self.y[take]).tolist(),[4]*8)
            for c,p in enumerate(pools):
                self.assertGreaterEqual(np.isin(take[self.y[take]==c],p).sum(),2)

    def test_scoring_is_no_gradient_rng_and_mode_preserving(self):
        m=h.historical.FlowModel(9).train()
        rng=torch.get_rng_state().clone()
        before=h.old.state_hash(m)
        losses=h.difficulty(m,torch.from_numpy(self.x),torch.from_numpy(self.y),batch=31)
        self.assertTrue(np.isfinite(losses).all())
        self.assertTrue(m.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(before,h.old.state_hash(m))
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_control_exact_historical_weights_and_predictions(self):
        a,ar,_=h.fit(self.x,self.y,self.g,self.pop,h.ARMS[0],self.s)
        old_s=dict(h.historical.SETTINGS,epochs=3,steps_per_epoch=3,per_class=4)
        b,br=h.historical.fit(self.x,self.y,self.g,self.pop,0.,old_s)
        self.assertEqual(ar['initial_state_sha256'],br['initial_state_sha256'])
        self.assertEqual(ar['final_state_sha256'],br['final_state_sha256'])
        np.testing.assert_array_equal(h.historical.predict(a,self.x),h.historical.predict(b,self.x))
        self.assertEqual(ar['forced_hard_positions'],0)

    def test_both_arms_replay_and_tensor_only_recovery(self):
        for arm in h.ARMS:
            full,fr,_=h.fit(self.x,self.y,self.g,self.pop,arm,self.s)
            _,_,paused=h.fit(self.x,self.y,self.g,self.pop,arm,self.s,stop_after=2)
            buffer=io.BytesIO()
            torch.save(paused,buffer)
            restored=torch.load(io.BytesIO(buffer.getvalue()),weights_only=True)
            resumed,rr,_=h.fit(self.x,self.y,self.g,self.pop,arm,self.s,resume=restored)
            self.assertEqual(fr,rr)
            self.assertEqual(h.old.state_hash(full),h.old.state_hash(resumed))
            self.assertEqual(fr['examples_processed'],288)
            self.assertEqual(fr['mining_examples_scored'],768)
            self.assertEqual(fr['optimizer_steps'],9)
            self.assertEqual(fr['sampled_class_counts'],[36]*8)
            self.assertEqual(fr['forced_hard_positions'],96 if arm==h.ARMS[1] else 0)

    def test_identical_warmup_then_changed_sampler(self):
        _,a,_=h.fit(self.x,self.y,self.g,self.pop,h.ARMS[0],self.s)
        _,b,_=h.fit(self.x,self.y,self.g,self.pop,h.ARMS[1],self.s)
        self.assertEqual(a['observations'][0],b['observations'][0])
        self.assertEqual(a['mining'][:2],b['mining'][:2])
        self.assertNotEqual(a['batch_chain_sha256'],b['batch_chain_sha256'])
        self.assertNotEqual(a['final_state_sha256'],b['final_state_sha256'])

    def test_changed_recovery_inputs_rejected(self):
        _,_,state=h.fit(self.x,self.y,self.g,self.pop,h.ARMS[1],self.s,stop_after=1)
        changed=self.x.copy(); changed[0,0]+=1
        with self.assertRaisesRegex(ValueError,'Recovery inputs'):
            h.fit(changed,self.y,self.g,self.pop,h.ARMS[1],self.s,resume=state)
        with self.assertRaisesRegex(ValueError,'Recovery inputs'):
            h.fit(self.x,self.y,self.g,self.pop,h.ARMS[0],self.s,resume=state)

    def test_invalid_mining_and_recipe_rejected(self):
        with self.assertRaises(ValueError):
            h.hard_pools(self.y,np.full(256,np.nan))
        pools=h.hard_pools(self.y,np.ones(256))
        pools[0]=pools[1]
        with self.assertRaises(ValueError):
            list(h.mixed_batches(self.y,pools,7,1,2,4))
        with self.assertRaises(ValueError):
            h.fit(self.x,self.y,self.g,self.pop,h.ARMS[0],dict(self.s,per_class=3))


if __name__=='__main__':
    unittest.main()
