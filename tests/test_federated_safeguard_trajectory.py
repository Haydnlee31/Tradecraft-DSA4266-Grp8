"""Synthetic paired-trajectory tests; neither real data nor held-out arrays."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.eval.federated_class_safeguard import inspect_step, state_digest, PROTECTED
from src.eval.federated_finite_step import inspect
from src.eval.federated_safeguard_trajectory import paired_round, verify_first_step, development_screen
from tests import test_federated_class_safeguard as fixtures

REPORTS = Path(__file__).parents[1]/'reports/federated_class_safeguards'


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        fixtures.SafeguardTests.setUp(self)
        self.recipe = json.loads((REPORTS/'finite-step-plan.json').read_text())
        self.plan = json.loads((REPORTS/'trajectory-plan.json').read_text())

    def test_paired_local_rounds_start_identically_and_replay_exactly(self):
        models = {a: copy.deepcopy(self.model) for a in ('fedavg', 'corrected')}
        a, result = paired_round(models, self.data, self.parts, self.settings, 3, self.recipe)
        b, replay = paired_round(models, self.data, self.parts, self.settings, 3, self.recipe)
        self.assertEqual(result, replay)
        self.assertEqual(result['fedavg_model_sha256'], result['proposal_model_sha256'])
        self.assertEqual(result['client_work']['fedavg'], result['client_work']['corrected'])
        self.assertEqual(result['work']['client_training_examples'], 160)
        self.assertEqual(result['work']['gradient_examples'], 80)
        for arm in models:
            self.assertEqual(state_digest(models[arm]), state_digest(self.model))
            self.assertEqual(state_digest(a[arm]), state_digest(b[arm]))

    def test_first_step_bridge_checks_complete_prior_evidence(self):
        old = inspect_step(self.model, self.data, self.parts, self.settings, 2)
        prior = inspect(self.model, self.data, self.parts, self.settings, 2, old, self.recipe)
        models = dict.fromkeys(('fedavg', 'corrected'), self.model)
        _, result = paired_round(models, self.data, self.parts, self.settings, 3, self.recipe)
        # This seeded fixture passes the finite-step check. Bridge compares
        # class sums, all client work and metrics, not merely similar F1.
        self.assertTrue(result['search']['accepted'])
        verify_first_step(result, prior)
        changed = copy.deepcopy(result)
        changed['proposal_model_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'bridge differs'):
            verify_first_step(changed, prior)

    def test_rejected_round_preserves_previous_corrected_weights(self):
        models = dict.fromkeys(('fedavg', 'corrected'), self.model)
        with patch('src.eval.federated_safeguard_trajectory.bounded_search',
                   return_value=(None, {'accepted': False, 'trials': []})):
            out, result = paired_round(models, self.data, self.parts, self.settings, 3, self.recipe)
        self.assertIs(out['corrected'], self.model)
        self.assertIsNone(result['corrected'])
        self.assertIsNone(result['corrected_model_sha256'])
        self.assertEqual(result['work']['client_training_examples'], 160)
        self.assertEqual(result['work']['training_forward_evaluation_examples'], 80)
        self.assertEqual(development_screen([result], self.plan)['decision'], 'stop_rejected_update')

    def fake_result(self, rare_gain=.02, macro_change=0., alert_change=0.):
        a = {'macro_f1': .4, 'benign_false_alert_rate': .01,
             'per_class': {name: {'recall': 0.} for name in PROTECTED[1:]}}
        b = copy.deepcopy(a)
        b['macro_f1'] += macro_change
        b['benign_false_alert_rate'] += alert_change
        b['per_class'][PROTECTED[1]]['recall'] = rare_gain
        return {'search': {'accepted': True}, 'fedavg': {'metrics': a}, 'corrected': {'metrics': b}}

    def test_endpoint_screen_requires_detection_not_just_safety(self):
        good = self.fake_result()
        self.assertEqual(development_screen([good]*5, self.plan)['decision'],
                         'eligible_for_separately_planned_development_validation')
        for r in (self.fake_result(rare_gain=0), self.fake_result(macro_change=-.01),
                  self.fake_result(alert_change=.002)):
            self.assertEqual(development_screen([r]*5, self.plan)['decision'],
                             'stop_no_detection_gain_within_budget')

    def test_no_best_intermediate_selection_or_early_success(self):
        good, bad = self.fake_result(), self.fake_result(rare_gain=0)
        self.assertEqual(development_screen([good]*4, self.plan)['decision'], 'stop_rejected_update')
        self.assertEqual(development_screen([good]*4+[bad], self.plan)['decision'],
                         'stop_no_detection_gain_within_budget')
        rejected = copy.deepcopy(good)
        rejected['search']['accepted'] = False
        self.assertEqual(development_screen([rejected]+[good]*4, self.plan)['decision'], 'stop_rejected_update')


if __name__ == '__main__':
    unittest.main()
