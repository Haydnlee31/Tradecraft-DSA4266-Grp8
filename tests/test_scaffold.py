"""Synthetic mathematical, pairing and state-persistence tests for SCAFFOLD."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.eval.federated_class_safeguard import state_digest, vector
from src.eval.scaffold_trial import verify_zero_bridge, screen
from src.federated.scaffold import (
    initialize_controls, aggregate, control_digest, verify_controls, option_two,
    sampled_batches, local_sgd, federated_round, central_round, parameters,
)
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_losses import build_official_criterion
from tests import test_federated_class_safeguard as fixtures

PLAN = Path(__file__).parents[1]/'reports/federated_class_safeguards/scaffold-plan.json'


class ScaffoldTests(unittest.TestCase):
    def setUp(self):
        fixtures.SafeguardTests.setUp(self)
        self.recipe = {'seed': 7, 'batch_size': 8, 'local_steps': 3, 'lr': .01, 'weight_decay': 1e-5}

    def test_flower_uses_population_weights_not_equal_steps(self):
        result = aggregate([{'w': torch.tensor([1.])}, {'w': torch.tensor([3.])}], [1, 3])
        self.assertEqual(result['w'].item(), 2.5)
        with self.assertRaises(ValueError):
            aggregate([{'w': torch.tensor([1.])}], [0])

    def test_sampling_reproducible_training_only_and_audited(self):
        rows = np.arange(7)
        batches, meter = sampled_batches(self.data, rows, 8, 3, 17)
        actual = list(batches)
        rng, h = np.random.default_rng(17), hashlib.sha256()
        counts = np.zeros(8, dtype=np.int64)
        for x, y in actual:
            take = rows[rng.integers(0, len(rows), size=8)]
            h.update(take.astype('<i8').tobytes())
            np.testing.assert_array_equal(x, self.data.arrays['train'][0][take])
            np.testing.assert_array_equal(y, self.data.arrays['train'][1][take])
            counts += np.bincount(y, minlength=8)
        self.assertEqual(meter['sampled_rows_sha256'], h.hexdigest())
        self.assertEqual(meter['sampled_class_counts'], counts.tolist())
        self.assertEqual(meter['examples_processed'], 24)
        self.assertEqual(meter['optimizer_steps'], 3)

    def test_zero_control_first_round_exact_fedavg_bridge(self):
        controls = initialize_controls(self.model, len(self.parts))
        a, _, ar = federated_round(self.model, self.data, self.parts, self.recipe, 21)
        b, new, br = federated_round(self.model, self.data, self.parts, self.recipe, 21, controls)
        np.testing.assert_array_equal(vector(a), vector(b))
        verify_zero_bridge(ar, br)
        self.assertLess(verify_controls(b, new, list(map(len, self.parts))), 1e-6)
        self.assertNotEqual(control_digest(controls), control_digest(new))

    def test_nonzero_control_sign_and_option_two_analytic_identity(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): model.weight.fill_(.4)
        ci, c = {'weight': torch.tensor([[.7]])}, {'weight': torch.tensor([[.2]])}
        x = np.array([[1.], [2.]], dtype=np.float32)
        y = np.array([[.1], [.3]], dtype=np.float32)
        result, work, mean_raw = local_sgd(model, [(x, y)]*3, torch.nn.MSELoss(), .03, .02, ci, c)
        expected, grads = .4, []
        for _ in range(3):
            gradient = float(2*np.mean((expected*x-y)*x))+.02*expected
            grads.append(gradient)
            expected -= .03*(gradient-.7+.2)
        self.assertAlmostEqual(result.weight.item(), expected, places=6)
        updated = option_two(parameters(model), parameters(result), ci, c, 3, .03)
        self.assertAlmostEqual(updated['weight'].item(), np.mean(grads), places=6)
        torch.testing.assert_close(updated['weight'].double(), mean_raw['weight'], atol=1e-6, rtol=1e-6)
        self.assertEqual(work['optimizer_steps'], 3)
        self.assertAlmostEqual(model.weight.item(), .4, places=6)

    def test_controls_invariant_rejects_corruption_shapes_and_nonfinite(self):
        controls = initialize_controls(self.model, 3)
        key = next(iter(controls['server']))
        for mode in ('value', 'shape', 'nan'):
            bad = copy.deepcopy(controls)
            if mode == 'value': bad['server'][key].add_(1.)
            if mode == 'shape': bad['clients'][0][key] = torch.zeros(1)
            if mode == 'nan': bad['clients'][0][key].fill_(float('nan'))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                verify_controls(self.model, bad, list(map(len, self.parts)))

    def test_input_model_and_controls_are_not_mutated(self):
        controls = initialize_controls(self.model, 3)
        md, cd = state_digest(self.model), control_digest(controls)
        _, _, first = federated_round(self.model, self.data, self.parts, self.recipe, 21, controls)
        _, _, replay = federated_round(self.model, self.data, self.parts, self.recipe, 21, controls)
        self.assertEqual(first, replay)
        self.assertEqual(state_digest(self.model), md)
        self.assertEqual(control_digest(controls), cd)

    def test_serialized_boundary_recovery_preserves_all_client_variates(self):
        controls = initialize_controls(self.model, 3)
        one, c1, _ = federated_round(self.model, self.data, self.parts, self.recipe, 21, controls)
        two, c2, r2 = federated_round(one, self.data, self.parts, self.recipe, 22, c1)
        full, cf, rf = federated_round(two, self.data, self.parts, self.recipe, 23, c2)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'boundary.pt'
            torch.save({'model': one.state_dict(), 'controls': c1}, path)
            saved = torch.load(path, weights_only=True)
            resumed = copy.deepcopy(self.model)
            resumed.load_state_dict(saved['model'])
            two_again, c2_again, r2_again = federated_round(resumed, self.data, self.parts, self.recipe, 22, saved['controls'])
            final, cfinal, rfinal = federated_round(two_again, self.data, self.parts, self.recipe, 23, c2_again)
        self.assertEqual(r2, r2_again)
        self.assertEqual(rf, rfinal)
        self.assertEqual(state_digest(full), state_digest(final))
        self.assertEqual(control_digest(cf), control_digest(cfinal))

    def test_single_client_reduces_to_sgd_in_every_round(self):
        parts = [np.arange(80)]
        a, b, controls = copy.deepcopy(self.model), copy.deepcopy(self.model), initialize_controls(self.model, 1)
        for step in (21, 22, 23):
            a, _, _ = federated_round(a, self.data, parts, self.recipe, step)
            b, controls, _ = federated_round(b, self.data, parts, self.recipe, step, controls)
            self.assertEqual(state_digest(a), state_digest(b))

    def test_central_matches_work_not_federated_trajectory(self):
        central, work = central_round(self.model, self.data, 3, self.recipe, 21)
        _, _, federated = federated_round(self.model, self.data, self.parts, self.recipe, 21)
        self.assertEqual(work['examples_processed'], sum(c['examples_processed'] for c in federated['clients']))
        self.assertEqual(work['optimizer_steps'], 9)
        self.assertNotEqual(state_digest(central), federated['model_sha256'])

    def test_fixed_denominator_weighted_gradient_equals_pooled_gradient(self):
        # Class imbalance and unequal clients: uniform averaging would be WRONG.
        counts = dict(zip(CLASSES, range(1, 9)))
        y = torch.repeat_interleave(torch.arange(8), torch.arange(1, 9))
        x = torch.from_numpy(self.data.arrays['train'][0][:len(y)])
        model = copy.deepcopy(self.model).eval()
        criterion = build_official_criterion('sqrt_weighted_ce', counts, 'fixed_train_mean')
        def grad(rows):
            loss = criterion(model(x[rows]), y[rows])
            return torch.cat([g.flatten().double() for g in torch.autograd.grad(loss, tuple(model.parameters()))])
        parts = [np.arange(3), np.arange(3, 14), np.arange(14, 36)]
        average = sum(len(p)/36*grad(p) for p in parts)
        torch.testing.assert_close(average, grad(np.arange(36)), atol=3e-7, rtol=1e-5)

    def test_reject_buffered_model_and_invalid_controls(self):
        batch_model = MLPClassifier(39, 8, MLPConfig('bad', (8,), .2, 'batch'))
        with self.assertRaisesRegex(ValueError, 'buffer-free'):
            initialize_controls(batch_model, 3)
        with self.assertRaises(ValueError):
            option_two({}, {}, {}, {}, 0, .01)
        with self.assertRaises(ValueError):
            local_sgd(self.model, [], None, .01, 0, {}, None)
        with self.assertRaises(ValueError):
            sampled_batches(self.data, [], 8, 3, 7)

    def test_screen_separates_aggregate_and_rare_gains(self):
        plan = json.loads(PLAN.read_text())
        a = {'macro_f1': .4, 'benign_false_alert_rate': .01,
             'per_class': {c: {'recall': 0.} for c in ('Web-based', 'Brute Force')}}
        b = copy.deepcopy(a)
        b['macro_f1'] = .42
        row = {'arms': {k: {'evaluation': {'metrics': v}} for k, v in (('fedavg_sgd', a), ('scaffold', b))}}
        result = screen([row]*10, plan)
        self.assertTrue(result['aggregate_screen_passed'])
        self.assertFalse(result['rare_class_screen_passed'])
        b['macro_f1'] = .4
        b['per_class']['Brute Force']['recall'] = .02
        self.assertTrue(screen([row]*10, plan)['rare_class_screen_passed'])
        b['benign_false_alert_rate'] = .04
        self.assertFalse(screen([row]*10, plan)['safety_screen_passed'])
        with self.assertRaises(ValueError): screen([row]*9, plan)


if __name__ == '__main__':
    unittest.main()
