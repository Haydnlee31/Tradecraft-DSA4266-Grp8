"""Public evidence audits for the paired five-round stopping decision."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.data.official_inventory import sha256
from src.data.label_map import CLASSES
from src.eval.federated_class_safeguard import PROTECTED
from src.eval.federated_safeguard_trajectory import development_screen, verify_first_step
from src.models.official_streaming import confusion_metrics

ROOT = Path(__file__).parents[1]
REPORTS = ROOT/'reports/federated_class_safeguards'


class TrajectoryReceiptTests(unittest.TestCase):
    def setUp(self):
        self.e = json.loads((REPORTS/'trajectory-results.json').read_text())
        self.r = self.e['primary']
        self.plan = json.loads((REPORTS/'trajectory-plan.json').read_text())
        self.recipe = json.loads((REPORTS/'finite-step-plan.json').read_text())

    def test_frozen_sources_and_replay_and_first_round_bridge(self):
        for field, path in (
            ('script_sha256', ROOT/'src/eval/federated_safeguard_trajectory.py'),
            ('plan_sha256', REPORTS/'trajectory-plan.json'),
            ('recipe_sha256', REPORTS/'finite-step-plan.json'),
            ('finite_step_runner_sha256', ROOT/'src/eval/federated_finite_step.py'),
            ('finite_step_evidence_sha256', REPORTS/'finite-step-results.json'),
        ):
            self.assertEqual(self.r[field], sha256(path))
        stable = copy.deepcopy(self.r)
        for r in stable['rounds']:
            r.pop('elapsed_seconds')
        encoded = json.dumps(stable, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), self.e['replay']['non_timing_sha256'])
        self.assertTrue(self.e['replay']['all_non_timing_fields_exact'])
        prior = json.loads((REPORTS/'finite-step-results.json').read_text())['primary']['cases'][-1]
        verify_first_step(self.r['rounds'][0], prior)
        self.assertEqual(self.r['source_provenance'], prior['provenance'])

    def test_chained_models_equal_local_work_and_no_held_out_evaluation(self):
        self.assertEqual(self.r['status'], 'complete')
        self.assertEqual([r['round'] for r in self.r['rounds']], [21, 22, 23, 24, 25])
        for key in ('validation_rows_opened', 'test_opened', 'test_evaluated', 'cloud_used'):
            self.assertFalse(self.r[key])
        for previous, current in zip(self.r['rounds'], self.r['rounds'][1:]):
            for arm in ('fedavg', 'corrected'):
                self.assertEqual(current['starting_model_sha256'][arm], previous[arm+'_model_sha256'])
        work = {key: sum(r['work'][key] for r in self.r['rounds']) for key in self.r['work']}
        self.assertEqual(work, self.r['work'])
        self.assertEqual(work, self.e['replay']['work'])
        for key, value in work.items():
            self.assertLessEqual(value, self.plan['max_work'][key])
        self.assertEqual(work['client_training_examples'], 20000000)
        self.assertEqual(work['client_optimizer_steps'], 39170)
        for r in self.r['rounds']:
            for arm in ('fedavg', 'corrected'):
                self.assertEqual(sum(w['examples_processed'] for w in r['client_work'][arm]), 2000000)
                self.assertEqual(sum(w['optimizer_steps'] for w in r['client_work'][arm]), 3917)

    def test_metrics_and_accepted_loss_changes_recompute(self):
        for r in self.r['rounds']:
            for arm in ('fedavg', 'corrected'):
                metrics = r[arm]['metrics']
                cm = np.array(metrics['confusion_matrix'])
                self.assertEqual(metrics, confusion_metrics(cm))
                self.assertEqual(int(cm.sum()), 2000000)
                self.assertEqual(set(metrics['per_class']), set(CLASSES))
            self.assertTrue(r['search']['accepted'])
            self.assertEqual(len(r['search']['trials']), 1)
            trial = r['search']['trials'][0]
            self.assertEqual(trial['scale'], 1.)
            baseline = r['baseline_client_risk']
            for risk in (baseline, trial['risk']):
                counts, sums = np.array(risk['client_counts']), np.array(risk['client_loss_sums'])
                np.testing.assert_array_equal(counts.sum(0), [106148, 2391, 1269])
                np.testing.assert_array_equal(sums.sum(0)/counts.sum(0), risk['mean_class_ce'])
            changes = np.array(trial['risk']['mean_class_ce'])-baseline['mean_class_ce']
            np.testing.assert_array_equal(changes, trial['actual_ce_changes'])
            upper = self.recipe['armijo_fraction']*np.array(trial['predicted_ce_changes'])+self.recipe['loss_tolerance']
            self.assertTrue(np.all(changes < -self.recipe['loss_tolerance']))
            self.assertTrue(np.all(changes <= upper))
            self.assertGreaterEqual(trial['retained_norm_fraction'], self.recipe['min_retained_norm_fraction'])

    def test_failed_screen_tiny_detection_gain_and_macro_tradeoff(self):
        self.assertEqual(self.r['screen'], development_screen(self.r['rounds'], self.plan))
        self.assertEqual(self.r['screen']['decision'], 'stop_no_detection_gain_within_budget')
        self.assertFalse(self.r['screen']['full_training_launched'])
        r = self.r['rounds'][-1]
        bf, web = [CLASSES.index(c) for c in (PROTECTED[2], PROTECTED[1])]
        self.assertEqual(r['fedavg']['metrics']['confusion_matrix'][bf][bf], 0)
        self.assertEqual(r['corrected']['metrics']['confusion_matrix'][bf][bf], 2)
        self.assertEqual(r['corrected']['metrics']['confusion_matrix'][web][web], 0)
        self.assertLess(self.r['screen']['rare_recall_gains'][PROTECTED[2]], .01)
        self.assertLess(self.r['screen']['macro_f1_change'], -.005)
        self.assertLess(self.r['screen']['false_alert_change'], 0.)


if __name__ == '__main__':
    unittest.main()
