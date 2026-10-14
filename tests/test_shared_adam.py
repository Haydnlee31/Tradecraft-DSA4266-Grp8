"""Analytic, negative-control and recovery tests for synchronous server Adam."""
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from src.data.label_map import CLASSES
from src.federated import shared_adam as b
from src.eval.shared_adam_bridge import schedule, run_case, run, snapshot
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_losses import build_official_criterion

RECIPE = {'lr': .001, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 1e-5, 'clip_norm': 1.}


class SharedAdamTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)

    def model(self, dropout=.2):
        return MLPClassifier(3, len(CLASSES), MLPConfig('test', (8, 4), dropout, 'layer')).double().train()

    def batches(self):
        return [(torch.randn(n, 3, dtype=torch.float64), torch.arange(n) % len(CLASSES)) for n in (5, 9)]

    def criterion(self):
        return build_official_criterion('sqrt_weighted_ce', dict(zip(CLASSES, range(1, 9))), 'fixed_train_mean').double()

    def test_population_mean_has_an_independent_analytic_gradient(self):
        # MSE makes the expected derivative explicit: d(mean((wx-y)^2))/dw.
        model = nn.Linear(1, 1, bias=False).double()
        with torch.no_grad(): model.weight.fill_(2.)
        batches = [(torch.tensor([[1.], [3.]], dtype=torch.float64), torch.tensor([[0.], [1.]], dtype=torch.float64)),
                   (torch.tensor([[2.]], dtype=torch.float64), torch.tensor([[4.]], dtype=torch.float64))]
        grads = [b.client_gradient(model, batch, nn.MSELoss(), seed) for batch, seed in zip(batches, [1, 2])]
        g = b.aggregate_gradients(grads, [3, 17])[0]
        expected = .15 * (2*(2-0)*1 + 2*(6-1)*3)/2 + .85 * 2*(4-4)*2
        self.assertAlmostEqual(g.item(), expected)
        wrong = b.aggregate_gradients(grads, [1, 1])[0]
        self.assertGreater((g-wrong).abs().item(), 1.)

    def test_losses_before_backward_equal_gradients_after_backward(self):
        model, batches, criterion = self.model(), self.batches(), self.criterion()
        grads = [b.client_gradient(model, batch, criterion, seed) for batch, seed in zip(batches, [15, 16])]
        actual = b.aggregate_gradients(grads, [7, 31])
        expected = b.central_accumulated_gradient(model, batches, [7, 31], criterion, [15, 16])
        self.assertTrue(b.residual(actual, expected, 2e-12, 1e-10)['within_tolerance'])

    def test_equal_minibatches_do_not_justify_uniform_population_weights(self):
        model, criterion = self.model(0.), self.criterion()
        batches = self.batches()
        batches = [(x[:5], y[:5]) for x, y in batches]
        correct = b.central_accumulated_gradient(model, batches, [7, 31], criterion, [0, 1])
        uniform = b.central_accumulated_gradient(model, batches, [1, 1], criterion, [0, 1])
        self.assertGreater(b.residual(correct, uniform, 1e-12, 1e-12)['l2'], .01)

    def test_gradient_calls_preserve_rng_and_model(self):
        model, batches, criterion = self.model(), self.batches(), self.criterion()
        before, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
        first = b.client_gradient(model, batches[0], criterion, 12)
        second = b.client_gradient(model, batches[0], criterion, 12)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(first, second)))
        b.central_accumulated_gradient(model, batches, [3, 7], criterion, [12, 13])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_different_dropout_masks_break_the_bridge(self):
        model, batches, criterion = self.model(.5), self.batches(), self.criterion()
        a = b.client_gradient(model, batches[0], criterion, 1)
        c = b.client_gradient(model, batches[0], criterion, 2)
        self.assertGreater(b.residual(a, c, 1e-12, 1e-12)['l2'], .01)

    def test_adam_matches_hand_computed_moments_bias_correction_and_l2(self):
        model = nn.Linear(1, 1, bias=False).double()
        with torch.no_grad(): model.weight.fill_(.7)
        recipe = dict(RECIPE, weight_decay=.03)
        optimizer = b.make_optimizer(model, recipe)
        x, m, v = .7, 0., 0.
        for step, raw in enumerate((2., -.5, 3.), 1):
            # torch's norm clip uses max_norm/(norm+1e-6); Adam adds L2 later.
            clipped = raw*min(1., 1./(abs(raw)+1e-6))
            g = clipped+.03*x
            m, v = .9*m+.1*g, .999*v+.001*g*g
            x -= .001*(m/(1-.9**step))/((v/(1-.999**step))**.5+1e-8)
            b.apply_gradient(model, optimizer, (torch.tensor([[raw]], dtype=torch.float64),), 1.)
            state = b.optimizer_tensors(model, optimizer)
            self.assertAlmostEqual(model.weight.item(), x, places=14)
            self.assertAlmostEqual(state['exp_avg'][0].item(), m, places=14)
            self.assertAlmostEqual(state['exp_avg_sq'][0].item(), v, places=14)
            self.assertEqual(state['step'][0].item(), step)

    def test_resetting_moments_changes_the_update(self):
        model = nn.Linear(1, 1, bias=False).double()
        persistent, reset = copy.deepcopy(model), copy.deepcopy(model)
        optimizer = b.make_optimizer(persistent, RECIPE)
        for g in (1., -2., .25):
            gradient = (torch.tensor([[g]], dtype=torch.float64),)
            b.apply_gradient(persistent, optimizer, gradient)
            b.apply_gradient(reset, b.make_optimizer(reset, RECIPE), gradient)
        self.assertGreater(abs(persistent.weight.item()-reset.weight.item()), .0001)

    def test_serialization_restores_all_adam_state(self):
        initial = self.model()
        model, optimizer = copy.deepcopy(initial), None
        optimizer = b.make_optimizer(model, RECIPE)
        gradients = [tuple(torch.randn_like(p) for p in model.parameters()) for _ in range(4)]
        for g in gradients[:2]: b.apply_gradient(model, optimizer, g, 1.)
        buffer = io.BytesIO()
        torch.save(snapshot({'arm':model}, {'arm':optimizer}), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)['arm']
        restored, resumed = b.restore(initial, saved, RECIPE)
        for g in gradients[2:]:
            b.apply_gradient(model, optimizer, g, 1.)
            b.apply_gradient(restored, resumed, g, 1.)
        self.assertTrue(all(torch.equal(p,q) for p,q in zip(model.parameters(), restored.parameters())))
        for key, a in b.optimizer_tensors(model, optimizer).items():
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a, b.optimizer_tensors(restored,resumed)[key])))

    def test_invalid_payloads_rejected_before_step(self):
        model = self.model()
        optimizer = b.make_optimizer(model, RECIPE)
        for counts in ([], [0], [-1, 3], [1., 2]):
            with self.assertRaises(ValueError): b.population_weights(counts)
        refs = tuple(torch.zeros_like(p) for p in model.parameters())
        for bad in (refs[:-1], tuple(g.float() for g in refs), tuple(g+float('nan') for g in refs)):
            with self.assertRaises(ValueError): b.apply_gradient(model, optimizer, bad)
        with self.assertRaises(ValueError): b.aggregate_gradients([refs], [1, 2])
        with self.assertRaises(ValueError): b.apply_gradient(copy.deepcopy(model), optimizer, refs)
        with self.assertRaises(ValueError): b.parameters(nn.BatchNorm1d(3))

    def test_schedule_is_reproducible_and_ownership_preserved(self):
        parts = [np.arange(3), np.arange(3, 12)]
        a, sa = schedule(parts, 4, 10, 17)
        c, sc = schedule(parts, 4, 10, 17)
        self.assertEqual(sa, sc)
        for owned, x, y in zip(parts, a, c):
            self.assertTrue(np.array_equal(x, y))
            self.assertTrue(np.isin(x, owned).all())
        self.assertNotEqual(sa, schedule(parts, 5, 10, 17)[1])

    def test_small_end_to_end_case_and_exact_work(self):
        rng = np.random.default_rng(12)
        y = np.arange(80) % len(CLASSES)
        data = SimpleNamespace(arrays={'train': [rng.normal(size=(80,39)).astype(np.float32), y]},
                               manifest={'train_class_counts': {c:10 for c in CLASSES}})
        parts = [np.arange(11), np.arange(11, 38), np.arange(38, 80)]
        plan = {'seed': 12, 'steps': 4, 'recovery_after': 2, 'batch_size': 8,
                'recipe': RECIPE, 'tolerances': {'float64': {'atol':2e-11, 'rtol':2e-9}}}
        result = {}
        with tempfile.TemporaryDirectory() as folder:
            run_case(data, parts, plan, {'dtype':'float64', 'dropout':.2}, Path(folder), result)
        self.assertTrue(result['recovery_exact'])
        self.assertEqual(result['work'], {'backward_examples':288, 'backward_batches':36,
                                         'optimizer_steps':12, 'quality_evaluation_examples':0})
        self.assertTrue(all(v['within_tolerance'] for s in result['steps'] for v in s['checks'].values()))

    def test_existing_output_never_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'preserve'):
            run('missing', 'missing', Path(__file__).parent)


if __name__ == '__main__':
    unittest.main()
