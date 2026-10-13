"""Synthetic tests of finite-step safeguards; never open the real cohorts."""
import copy
from pathlib import Path
import json
import unittest

import numpy as np
import torch

from src.eval.federated_class_safeguard import (
    inspect_step, state_digest, vector,
)
from src.eval.federated_finite_step import (
    margin_projection, client_risk, bounded_search, inspect, continuation_gate,
)
from tests import test_federated_class_safeguard as fixtures

PLAN = Path(__file__).parents[1]/'reports/federated_class_safeguards/finite-step-plan.json'


class FiniteStepTests(unittest.TestCase):
    def setUp(self):
        fixtures.SafeguardTests.setUp(self)
        self.plan = json.loads(PLAN.read_text())

    def test_affine_projection_orthogonal_and_strict(self):
        delta = np.array([2., 3., -4.])
        result, info = margin_projection(delta, [[1., 0., 0.], [0., 1., 0.]], .01)
        margin = .01*np.linalg.norm(delta)
        np.testing.assert_allclose(result, [-margin, -margin, -4.], atol=1e-12)
        self.assertEqual(info['active_constraints'], [0, 1])
        self.assertEqual(info['status'], 'certified')

    def test_inactive_constraints_preserve_update_exactly(self):
        delta = np.array([-1., -2., 3.])
        result, info = margin_projection(delta, [[1., 0., 0.], [0., 1., 0.]], .01)
        np.testing.assert_array_equal(delta, result)
        self.assertEqual(info['active_constraints'], [])

    def test_duplicate_constraints_rescaling_and_kkt(self):
        rng = np.random.default_rng(4266)
        for _ in range(20):
            delta, g = rng.normal(size=8), rng.normal(size=(3, 8))
            result, info = margin_projection(delta, g, .01)
            changed_units, _ = margin_projection(delta, g*np.array([1e-6, 2., 1e4])[:, None], .01)
            np.testing.assert_allclose(result, changed_units, atol=1e-10)
            u = g/np.linalg.norm(g, axis=1)[:, None]
            self.assertTrue(np.all(u @ result <= -.01*np.linalg.norm(delta)+1e-10))
            active = info['active_constraints']
            a = u[active]
            lam = np.linalg.lstsq(a.T, delta-result, rcond=1e-12)[0]
            self.assertTrue(np.all(lam >= -1e-10))
            np.testing.assert_allclose(a.T @ lam, delta-result, atol=1e-10)
        result, _ = margin_projection([2., 1.], [[1., 0.], [2., 0.]], .01)
        np.testing.assert_allclose(result, [-.01*np.sqrt(5.), 1.], atol=1e-12)

    def test_opposing_and_stationary_constraints_reject_not_zero_success(self):
        for delta, g in (([2.], [[1.], [-1.]]), ([0.], [[1.]]), ([1.], [[0.]])):
            result, info = margin_projection(delta, g, .01)
            self.assertIsNone(result)
            self.assertTrue(info['status'].startswith('rejected_'))

    def test_invalid_projection_inputs(self):
        for delta, g, fraction in (([np.nan], [[1.]], .01), ([1.], [[np.inf]], .01),
                                   ([1., 2.], [[1.]], .01), ([1.], [[1.]]*4, .01),
                                   ([1.], [[1.]], 0.), ([1.], [[1.]], np.nan)):
            with self.subTest(delta=delta, g=g), self.assertRaises(ValueError):
                margin_projection(delta, g, fraction)

    def test_client_sums_pool_correctly_without_rng_or_model_mutation(self):
        digest, rng = state_digest(self.model), torch.get_rng_state().clone()
        result = client_risk(self.model, self.data, self.parts, 8)
        pooled = client_risk(self.model, self.data, [np.arange(80)], 8)
        np.testing.assert_allclose(result['mean_class_ce'], pooled['mean_class_ce'], atol=3e-7)
        counts, sums = np.array(result['client_counts']), np.array(result['client_loss_sums'])
        np.testing.assert_array_equal(counts.sum(0), [10]*3)
        np.testing.assert_array_equal(sums.sum(0)/counts.sum(0), result['mean_class_ce'])
        self.assertEqual(result['forward_examples'], 30)
        self.assertEqual(state_digest(self.model), digest)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_empty_client_class_returns_zero_count_not_fake_mean(self):
        y = self.data.arrays['train'][1]
        parts = [np.where(y == 0)[0], np.where(y != 0)[0]]
        result = client_risk(self.model, self.data, parts)
        self.assertEqual(result['client_counts'], [[10, 0, 0], [0, 10, 10]])
        self.assertEqual(result['client_loss_sums'][0][1:], [0., 0.])
        self.assertEqual(result['client_loss_sums'][1][0], 0.)

    def scalar_search(self, displacement, loss, plan=None, gradient=1.):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        calls = []
        def risk(m):
            calls.append(float(vector(m)[0]))
            return {'mean_class_ce': [loss(calls[-1])]*3}
        result, info = bounded_search(model, np.array([displacement]), np.full((3, 1), gradient),
                                     {'mean_class_ce': [loss(1.)]*3}, risk, plan or self.plan)
        self.assertEqual(float(vector(model)[0]), 1.)
        return result, info, calls

    def test_backtracking_stops_at_first_passing_scale(self):
        result, info, calls = self.scalar_search(-3., lambda x: .5*x*x)
        self.assertIsNotNone(result)
        self.assertEqual([t['scale'] for t in info['trials']], [1., .5])
        self.assertEqual(info['accepted_scale'], .5)
        self.assertEqual(calls, [-2., -.5])
        self.assertEqual(info['trials'][-1]['retained_norm_fraction'], .5)

    def test_all_four_fail_without_fallback_or_extra_attempt(self):
        result, info, calls = self.scalar_search(-1., lambda x: 1.)
        self.assertIsNone(result)
        self.assertEqual(len(calls), 4)
        self.assertEqual(info['reason'], 'rejected_all_four_scales')
        self.assertFalse(any(t['accepted'] for t in info['trials']))

    def test_armijo_rejects_tiny_improvement_despite_negative_change(self):
        result, info, _ = self.scalar_search(-1., lambda x: 1.+.001*x)
        self.assertIsNone(result)
        self.assertTrue(all(all(v < 0 for v in t['actual_ce_changes']) for t in info['trials']))

    def test_rounding_and_minimum_progress_are_not_silently_ignored(self):
        # This ideal descent rounds to NO CHANGE in real FP32 parameters.
        result, info, _ = self.scalar_search(-1e-12, lambda x: .5*x*x)
        self.assertIsNone(result)
        self.assertTrue(all(t['retained_norm_fraction'] == 0. for t in info['trials']))
        # The projection reverses a one-dimensional ascent to a tiny move. It
        # descends but fails the retained-norm floor, and must not be accepted.
        result, info, _ = self.scalar_search(1., lambda x: .5*x*x)
        self.assertIsNone(result)
        self.assertTrue(all(t['retained_norm_fraction'] < .1 for t in info['trials']))

    def test_excessive_expansion_rejected_before_client_forwards(self):
        model = torch.nn.Linear(2, 1, bias=False)
        gradient = np.array([[1., 0.], [-1., .001], [1., 0.]])
        def should_not_run(_):
            self.fail('Rejected direction must not issue candidate checks')
        result, info = bounded_search(model, np.array([0., -1.]), gradient,
                                     {'mean_class_ce': [1.]*3}, should_not_run, self.plan)
        self.assertIsNone(result)
        self.assertEqual(info['reason'], 'rejected_direction_expansion')
        self.assertEqual(info['trials'], [])

    def test_bad_search_recipe_rejected(self):
        for key, value in (('scales', [1., .01]), ('armijo_fraction', .7),
                           ('loss_tolerance', 0.), ('min_retained_norm_fraction', 0.)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.scalar_search(-1., lambda x: .5*x*x, dict(self.plan, **{key: value}))

    def test_exact_replay_baseline_bridge_and_budget_on_synthetic_train_only_data(self):
        prior = inspect_step(self.model, self.data, self.parts, self.settings, 2)
        a = inspect(self.model, self.data, self.parts, self.settings, 2, prior, self.plan)
        b = inspect(self.model, self.data, self.parts, self.settings, 2, prior, self.plan)
        self.assertEqual(a, b)
        self.assertTrue(a['baseline_bridge_exact'])
        checks = len(a['search']['trials'])
        self.assertLessEqual(checks, 4)
        self.assertEqual(a['protected_risk_forward_examples'], 30*(1+checks))
        self.assertEqual(a['training_forward_evaluation_examples'], 80*(2+a['search']['accepted']))
        self.assertEqual(a['communication']['risk_reply_bytes_all_clients'], 48*3*(1+checks))
        altered = copy.deepcopy(prior)
        altered['fedavg_model_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'bridge differs'):
            inspect(self.model, self.data, self.parts, self.settings, 2, altered, self.plan)

    def test_gate_requires_all_safe_and_two_useful_states(self):
        c = {'search': {'accepted': True, 'trials': [{'accepted': True, 'actual_ce_changes': [-.01]*3}]},
             'fedavg_predicted_protected_ce_changes': [.1]*3,
             'fedavg_actual_protected_ce_changes': [.1]*3}
        decision = 'eligible_for_separately_planned_short_trajectory'
        self.assertEqual(continuation_gate([c]*3, self.plan)['decision'], decision)
        bad = copy.deepcopy(c)
        bad['search']['accepted'] = False
        self.assertEqual(continuation_gate([c, c, bad], self.plan)['decision'], 'stop_no_full_training')
        no_harm = dict(c, fedavg_actual_protected_ce_changes=[-.1]*3)
        self.assertEqual(continuation_gate([c, no_harm, no_harm], self.plan)['decision'], 'stop_no_full_training')


if __name__ == '__main__':
    unittest.main()
