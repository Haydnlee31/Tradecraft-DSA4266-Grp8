"""Recompute the finite-step evidence without private data or checkpoints."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.federated_class_safeguard import PROTECTED
from src.eval.federated_finite_step import continuation_gate
from src.models.official_streaming import confusion_metrics

ROOT = Path(__file__).parents[1]
REPORTS = ROOT/'reports/federated_class_safeguards'


class FiniteStepReceiptTests(unittest.TestCase):
    def setUp(self):
        self.e = json.loads((REPORTS/'finite-step-results.json').read_text())
        self.r = self.e['primary']
        self.p = json.loads((REPORTS/'finite-step-plan.json').read_text())
        self.old = json.loads((REPORTS/'diagnostic-results.json').read_text())['primary']

    def test_source_and_plan_pins_and_exact_replay_digest(self):
        self.assertEqual(self.r['script_sha256'], sha256(ROOT/'src/eval/federated_finite_step.py'))
        self.assertEqual(self.r['plan_sha256'], sha256(REPORTS/'finite-step-plan.json'))
        self.assertEqual(self.r['previous_runner_sha256'], sha256(ROOT/'src/eval/federated_class_safeguard.py'))
        self.assertEqual(self.p['previous_evidence_sha256'], sha256(REPORTS/'diagnostic-results.json'))
        stable = copy.deepcopy(self.r)
        for case in stable['cases']:
            case.pop('elapsed_seconds')
        encoded = json.dumps(stable, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), self.e['replay']['non_timing_sha256'])
        self.assertTrue(self.e['replay']['all_non_timing_fields_exact'])

    def test_original_baseline_and_failed_first_attempt_are_preserved(self):
        self.assertEqual(self.old['gate']['decision'], 'stop_no_full_training')
        self.assertEqual([c['source_step'] for c in self.r['cases']], [2, 5, 20])
        for new, old in zip(self.r['cases'], self.old['cases']):
            for key in ('source_model_sha256', 'fedavg_model_sha256', 'provenance',
                        'gradient_work', 'client_work', 'before', 'fedavg'):
                self.assertEqual(new[key], old[key])
            self.assertTrue(new['baseline_bridge_exact'])

    def test_work_and_evaluation_boundaries(self):
        self.assertEqual(self.r['status'], 'complete')
        for key in ('validation_rows_opened', 'test_opened', 'test_evaluated'):
            self.assertFalse(self.r[key])
        self.assertEqual(self.r['fixed_work'], self.p['fixed_work'])
        for key, value in self.r['bounded_work'].items():
            self.assertLessEqual(value, self.p['max_work'][key])
        for key in ('fixed_work', 'bounded_work'):
            self.assertEqual(self.r[key], self.e['replay'][key])
        self.assertEqual(sum(w['examples_processed'] for c in self.r['cases'] for w in c['client_work']), 6000000)
        self.assertEqual(sum(w['optimizer_steps'] for c in self.r['cases'] for w in c['client_work']), 11751)
        checks, risk_rows = 0, 0
        for c in self.r['cases']:
            trials = c['search']['trials']
            checks += len(trials)
            risks = [c['baseline_client_risk']]+[t['risk'] for t in trials]
            risk_rows += sum(r['forward_examples'] for r in risks)
            self.assertEqual(c['protected_risk_forward_examples'], sum(r['forward_examples'] for r in risks))
            self.assertEqual(c['communication']['risk_reply_bytes_all_clients'], (1+len(trials))*20*48)
            self.assertFalse(c['communication']['measured_network_traffic'])
        self.assertEqual(checks, self.r['bounded_work']['candidate_checks'])
        self.assertEqual(risk_rows, self.r['bounded_work']['protected_risk_forward_examples'])

    def test_client_sum_pooling_and_first_passing_acceptance_recompute(self):
        for c in self.r['cases']:
            baseline = c['baseline_client_risk']
            trials = c['search']['trials']
            self.assertEqual([t['scale'] for t in trials], self.p['scales'][:len(trials)])
            for risk in [baseline]+[t['risk'] for t in trials]:
                counts, sums = np.array(risk['client_counts']), np.array(risk['client_loss_sums'])
                self.assertEqual(counts.tolist(), baseline['client_counts'])
                np.testing.assert_array_equal(counts.sum(0), [106148, 2391, 1269])
                np.testing.assert_array_equal(sums.sum(0)/counts.sum(0), risk['mean_class_ce'])
                self.assertEqual(int(counts.sum()), risk['forward_examples'])
            for i, t in enumerate(trials):
                changes = np.array(t['risk']['mean_class_ce'])-baseline['mean_class_ce']
                predicted = np.array(t['predicted_ce_changes'])
                upper = self.p['armijo_fraction']*predicted+self.p['loss_tolerance']
                np.testing.assert_array_equal(changes, t['actual_ce_changes'])
                np.testing.assert_array_equal(upper, t['armijo_upper_bounds'])
                accepted = bool(np.all(predicted < 0) and np.all(changes < -self.p['loss_tolerance'])
                                and np.all(changes <= upper)
                                and t['retained_norm_fraction'] >= self.p['min_retained_norm_fraction'])
                self.assertEqual(t['accepted'], accepted)
                if i != len(trials)-1:
                    self.assertFalse(accepted)
            self.assertTrue(trials[-1]['accepted'])
            self.assertEqual(c['search']['accepted_scale'], trials[-1]['scale'])
        self.assertEqual(self.r['gate'], continuation_gate(self.r['cases'], self.p))
        self.assertEqual(self.r['gate']['safe_checkpoint_flags'], [True]*3)
        self.assertFalse(self.r['gate']['full_training_launched'])

    def test_all_eight_metrics_and_unchanged_zero_rare_recall(self):
        for c in self.r['cases']:
            for arm in ('before', 'fedavg', 'accepted_evaluation'):
                m = c[arm]['metrics']
                cm = np.array(m['confusion_matrix'])
                self.assertEqual(m, confusion_metrics(cm))
                self.assertEqual(cm.sum(), 2000000)
                self.assertEqual(set(m['per_class']), set(CLASSES))
                for name in PROTECTED[1:]:
                    self.assertEqual(m['per_class'][name]['recall'], 0.)
        # Loss acceptance must not be misreported as an aggregate performance win.
        last = self.r['cases'][-1]
        self.assertLess(last['accepted_evaluation']['metrics']['macro_f1'], last['fedavg']['metrics']['macro_f1'])


if __name__ == '__main__':
    unittest.main()
