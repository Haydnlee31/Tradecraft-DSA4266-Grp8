"""Analytic and provenance tests for the small deterministic method check."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.official_inventory import sha256
from src.explain import official_convergence as mc
from src.explain.official_integration import agreement, completeness, diagnose, integrate, quadrature, run, validate_plan
from tests.test_explain_convergence import setup_fixture, synthetic_pilot

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'reports/full_data_extension/explanation-integration-plan.json'


def integration_fixture(root):
    args = setup_fixture(root)
    with patch('src.explain.official.pilot', side_effect=synthetic_pilot):
        mc.run(**args)
    plan = json.loads(PLAN.read_text())
    plan.update(convergence_plan_sha256=sha256(args['plan_path']),
                convergence_receipt_sha256=sha256(args['output']/'receipt.json'))
    path = root/'integration-plan.json'
    path.write_text(json.dumps(plan))
    return dict(preparation_root=args['preparation_root'], convergence_root=args['output'], archive_root=root,
                manifest_path=args['manifest_path'], preparation_plan_path=args['preparation_plan_path'],
                convergence_plan_path=args['plan_path'], plan_path=path, output=root/'integration')


class IntegrationTests(unittest.TestCase):
    def test_composite_quadrature_moments_and_resolution_guards(self):
        for segments in (2, 4):
            alpha, weight = quadrature(segments)
            self.assertEqual(len(alpha), 32*segments)
            for degree in range(9):
                self.assertAlmostEqual(float(np.dot(weight, alpha**degree)), 1/(degree+1), places=13)
        for bad in (1, 3, 8, 2.0, True):
            with self.assertRaises(ValueError):
                quadrature(bad)

    def test_linear_identity_batching_and_parameter_gradients_unchanged(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = torch.nn.Linear(39, 8).eval()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        rng = np.random.default_rng(7)
        x, background = rng.normal(size=(2, 39)).astype(np.float32), rng.normal(size=(3, 39)).astype(np.float32)
        expected = (x[:, None, :].astype(np.float64)-background[None].astype(np.float64)).mean(1)[:, :, None]*model.weight.detach().numpy().T
        # Input subtraction occurs in float32, so allow its expected roundoff.
        a = integrate(model, background, x, 2, batch_size=13)
        b = integrate(model, background, x, 4)
        np.testing.assert_allclose(a, expected, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(b, expected, rtol=2e-6, atol=1e-7)
        for p in model.parameters():
            self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))
        model.train()
        with self.assertRaisesRegex(ValueError, 'Evaluation mode'):
            integrate(model, background, x, 2)

    def test_quadratic_and_interaction_have_known_feature_attributions(self):
        class Polynomial(torch.nn.Module):
            def forward(self, x):
                return torch.stack([x[:, i]**2+x[:, 0]*x[:, 1] for i in range(8)], dim=1)
        model = Polynomial().eval()
        rng = np.random.default_rng(12)
        x = rng.normal(size=(2, 39)).astype(np.float32)
        bg = rng.normal(size=(5, 39)).astype(np.float32)
        expected = np.zeros((2, 39, 8))
        xd, bd = x.astype(np.float64), bg.astype(np.float64)
        for c in range(8):
            expected[:, c, c] = xd[:, c]**2-(bd[:, c]**2).mean()
            expected[:, 0, c] += ((xd[:, None, 0]-bd[None, :, 0])*(xd[:, None, 1]+bd[None, :, 1])/2).mean(1)
            expected[:, 1, c] += ((xd[:, None, 1]-bd[None, :, 1])*(xd[:, None, 0]+bd[None, :, 0])/2).mean(1)
        for segments in (2, 4):
            actual = integrate(model, bg, x, segments)
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-6)

    def test_relu_boundary_and_no_randomness(self):
        class Kink(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x[:, :8])
        model = Kink().eval()
        x, bg = np.ones((1, 39), np.float32), -np.ones((1, 39), np.float32)
        torch_state, numpy_state = torch.random.get_rng_state().clone(), np.random.get_state()
        a, b = integrate(model, bg, x, 2), integrate(model, bg, x, 4)
        expected = np.zeros((1, 39, 8))
        for c in range(8):
            expected[0, c, c] = 1
        np.testing.assert_allclose(a, expected, atol=1e-7)
        np.testing.assert_allclose(b, expected, atol=1e-7)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        current_numpy_state = np.random.get_state()
        self.assertEqual(numpy_state[0], current_numpy_state[0])
        np.testing.assert_array_equal(numpy_state[1], current_numpy_state[1])
        self.assertEqual(numpy_state[2:], current_numpy_state[2:])

    def test_cancelling_feature_errors_do_not_pass_agreement(self):
        plan = json.loads(PLAN.read_text())
        fine = np.ones((8, 39, 8))
        coarse = fine.copy()
        coarse[:, 0] += 20
        coarse[:, 1] -= 20
        self.assertTrue(completeness(coarse, fine.sum(1), plan['completeness_limits'])['completeness_screen_passed'])
        self.assertFalse(agreement(coarse, fine, plan['resolution_limits'])['resolution_screen_passed'])
        baseline = dict(logits=fine.sum(1), reference_logits=np.zeros(8), repeat_0=fine, repeat_1=fine)
        self.assertFalse(diagnose(coarse, fine, baseline, np.arange(8), plan)['method_screen_passed'])
        self.assertIsNone(agreement(coarse, fine)['resolution_screen_passed'])
        fine[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            agreement(coarse, fine)

    def test_stable_complete_attributions_pass_with_errors_or_no_errors(self):
        plan = json.loads(PLAN.read_text())
        logits = np.eye(8)*2
        logits[0, 1] = 5
        ref = np.arange(8)*.1
        values = np.broadcast_to((logits-ref)[:, None, :]/39, (8, 39, 8)).copy()
        baseline = dict(logits=logits, reference_logits=ref, repeat_0=values, repeat_1=values)
        result = diagnose(values, values, baseline, np.arange(8), plan)
        self.assertTrue(result['method_screen_passed'])
        self.assertEqual(result['error_margins']['count'], 1)
        self.assertEqual(result['error_margins']['fine']['passing_output_count'], 1)
        no_errors = diagnose(values, values, baseline, logits.argmax(1), plan)
        self.assertEqual(no_errors['error_margins']['count'], 0)
        self.assertIsNone(no_errors['error_margins']['fine'])

    def test_plan_guards_and_published_parent_identity(self):
        plan = json.loads(PLAN.read_text())
        parent_path = PLAN.parent/'explanation-convergence-plan.json'
        previous = json.loads(parent_path.read_text())
        validate_plan(plan, previous)
        self.assertEqual(plan['convergence_plan_sha256'], sha256(parent_path))
        published = json.loads((PLAN.parent/'explanation-convergence-results.json').read_text())
        self.assertEqual(plan['convergence_receipt_sha256'], published['source_receipt_sha256'])
        for key, value in [('settings', {}), ('resolution_limits', {}), ('completeness_limits', {}),
                           ('feature_interpretation_authorized', True)]:
            bad = copy.deepcopy(plan)
            bad[key] = value
            with self.assertRaises(ValueError):
                validate_plan(bad, previous)

    def test_runner_uses_frozen_inputs_only_and_does_not_hide_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            args = integration_fixture(Path(folder))
            original_load = np.load
            opened = []
            def guard(path, *a, **kw):
                opened.append(Path(path))
                self.assertEqual(Path(path).suffix, '.npz')
                return original_load(path, *a, **kw)
            def stub(model, background, x, segments):
                self.assertIn(segments, (2, 4))
                self.assertEqual(background.shape, (128, 39))
                self.assertEqual(x.shape, (8, 39))
                return np.zeros((8, 39, 8))  # Deliberately inadequate fixture.
            with patch('src.explain.official_integration.integrate', side_effect=stub) as integration, \
                 patch('src.explain.official_integration.np.load', side_effect=guard):
                result = run(**args)
                self.assertEqual(integration.call_count, 14)
            self.assertEqual(len(opened), 8)
            self.assertEqual(result['status'], 'numerical_review_required')
            self.assertFalse(result['all_method_screens_passed'])
            for flag in mc.FLAGS:
                self.assertFalse(result[flag])
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                run(**args)

    def test_last_model_or_input_tampering_stops_before_integration(self):
        with tempfile.TemporaryDirectory() as folder:
            args = integration_fixture(Path(folder))
            last = args['convergence_root']/'dirichlet-seed7-2048.npz'
            last.write_bytes(b'changed')
            with patch('src.explain.official_integration.integrate') as integration:
                with self.assertRaisesRegex(ValueError, 'Monte Carlo arrays changed'):
                    run(**args)
                integration.assert_not_called()
            self.assertFalse(args['output'].exists())
            (args['preparation_root']/'inputs.npz').write_bytes(b'changed')
            with patch('src.explain.official.load_frozen') as loader:
                with self.assertRaisesRegex(ValueError, 'Frozen pilot inputs changed'):
                    run(**args)
                loader.assert_not_called()

    def test_published_result_counts_and_scope_are_consistent(self):
        result = json.loads((PLAN.parent/'explanation-integration-results.json').read_text())
        self.assertEqual(result['plan_sha256'], sha256(PLAN))
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['all_method_screens_passed'])
        self.assertEqual(len(result['models']), 7)
        self.assertTrue(all(m['method_screen_passed'] for m in result['models'].values()))
        for nodes, summary_key in (('64', 'coarse_passing_outputs'), ('128', 'fine_passing_outputs')):
            count = sum(m['scores'][nodes]['passing_output_count'] for m in result['models'].values())
            self.assertEqual(count, result['summary'][summary_key])
            self.assertEqual(count, 448)
        self.assertEqual(sum(m['error_margins']['fine']['passing_output_count'] for m in result['models'].values()), 29)
        self.assertEqual(result['summary']['path_forward_rows'], 7*8*128*(64+128))
        self.assertEqual(result['summary']['output_gradient_rows'], 8*result['summary']['path_forward_rows'])
        for flag in mc.FLAGS:
            self.assertFalse(result[flag])


if __name__ == '__main__':
    unittest.main()
