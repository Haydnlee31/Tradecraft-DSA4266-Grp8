"""Audit the public SGD/SCAFFOLD comparison without opening private arrays."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.scaffold_trial import screen, verify_zero_bridge, ARMS
from src.models.official_streaming import confusion_metrics

ROOT = Path(__file__).parents[1]
REPORTS = ROOT/'reports/federated_class_safeguards'


class ScaffoldReceiptTests(unittest.TestCase):
    def setUp(self):
        self.e = json.loads((REPORTS/'scaffold-results.json').read_text())
        self.r = self.e['primary']
        self.p = json.loads((REPORTS/'scaffold-plan.json').read_text())

    def test_frozen_sources_plan_and_non_timing_replay(self):
        for field, path in (
            ('script_sha256', ROOT/'src/eval/scaffold_trial.py'),
            ('scaffold_source_sha256', ROOT/'src/federated/scaffold.py'),
            ('historical_helper_sha256', ROOT/'src/eval/federated_class_safeguard.py'),
            ('plan_sha256', REPORTS/'scaffold-plan.json'),
        ):
            self.assertEqual(self.r[field], sha256(path))
        stable = copy.deepcopy(self.r)
        for row in stable['rounds']: row.pop('elapsed_seconds')
        encoded = json.dumps(stable, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), self.e['replay']['non_timing_sha256'])
        self.assertTrue(self.e['replay']['all_non_timing_fields_exact'])
        self.assertTrue(self.e['replay']['saved_models_and_controls_verified_from_tensor_bytes'])

    def test_zero_bridge_all_round_pairing_and_model_continuity(self):
        rounds = self.r['rounds']
        self.assertTrue(self.r['zero_control_first_round_exact'])
        verify_zero_bridge(rounds[0]['arms']['fedavg_sgd'], rounds[0]['arms']['scaffold'])
        self.assertEqual(rounds[0]['starting_model_sha256'], dict.fromkeys(ARMS, self.r['source_model_sha256']))
        for prev, row in zip(rounds, rounds[1:]):
            for arm in ARMS:
                self.assertEqual(row['starting_model_sha256'][arm], prev['arms'][arm]['model_sha256'])
        for row in rounds:
            a, b = [row['arms'][k]['clients'] for k in ('fedavg_sgd', 'scaffold')]
            self.assertEqual(len(a), 20)
            self.assertEqual(len(b), 20)
            self.assertEqual(sum(c['assigned_examples'] for c in a), 2000000)
            self.assertGreater(len(set(c['assigned_examples'] for c in a)), 1)
            for x, y in zip(a, b):
                for key in ('client', 'assigned_examples', 'seed', 'sampled_rows_sha256', 'sampled_class_counts',
                            'examples_processed', 'optimizer_steps'):
                    self.assertEqual(x[key], y[key])
                self.assertLess(y['option_two_identity_residual_l2'], 1e-4)
            self.assertLess(row['arms']['scaffold']['server_invariant_residual_linf'], 1e-6)

    def test_exact_work_and_closed_evaluation(self):
        self.assertEqual(self.r['status'], 'complete')
        self.assertEqual([r['step'] for r in self.r['rounds']], list(range(21, 31)))
        for key in ('validation_rows_opened', 'test_opened', 'test_evaluated', 'cloud_used'):
            self.assertFalse(self.r[key])
        self.assertEqual(self.r['total_work'], self.p['total_work'])
        self.assertEqual(self.r['total_work'], self.e['replay']['total_work'])
        for arm in ARMS:
            examples, updates = 0, 0
            for row in self.r['rounds']:
                result = row['arms'][arm]
                work = result['clients'] if 'clients' in result else [result]
                for w in work:
                    examples += w['examples_processed']
                    updates += w['optimizer_steps']
                    self.assertEqual(sum(w['sampled_class_counts']), w['examples_processed'])
                    self.assertEqual(len(w['sampled_rows_sha256']), 64)
            self.assertEqual({'client_training_examples': examples, 'optimizer_steps': updates}, self.p['work_per_arm'])
            self.assertEqual(self.r['work_per_arm'][arm], self.p['work_per_arm'])
        communication = self.r['communication']
        self.assertEqual(communication['parameter_bytes'], 5096*4)
        self.assertEqual(communication['extra_control_upload_bytes_all_rounds'], 5096*4*20*10)
        self.assertFalse(communication['measured_network_traffic'])

    def test_every_class_metric_recomputes_from_confusion(self):
        expected = [106148, 1178561, 362423, 66277, 2391, 1269, 44095, 238836]
        evaluations = [self.r['initial_evaluation']]+[r['arms'][a]['evaluation'] for r in self.r['rounds'] for a in ARMS]
        for e in evaluations:
            m = e['metrics']
            cm = np.array(m['confusion_matrix'])
            self.assertEqual(m, confusion_metrics(cm))
            self.assertEqual(set(m['per_class']), set(CLASSES))
            np.testing.assert_array_equal(cm.sum(1), expected)

    def test_final_screen_records_gain_but_fails_false_alert_safety(self):
        s = self.r['screen']
        self.assertEqual(s, screen(self.r['rounds'], self.p))
        self.assertEqual(s['decision'], 'stop_no_convincing_gain_at_frozen_recipe')
        self.assertGreater(s['macro_f1_gain'], .02)
        self.assertGreater(s['false_alert_change'], .09)
        self.assertFalse(s['safety_screen_passed'])
        self.assertFalse(s['full_training_launched'])
        self.assertLess(s['rare_recall_gains']['Brute Force'], .01)
        end = self.r['rounds'][-1]['arms']
        bf, web = CLASSES.index('Brute Force'), CLASSES.index('Web-based')
        self.assertEqual(end['scaffold']['evaluation']['metrics']['confusion_matrix'][bf][bf], 286)
        self.assertEqual(end['fedavg_sgd']['evaluation']['metrics']['confusion_matrix'][bf][bf], 281)
        for arm in ARMS:
            self.assertEqual(end[arm]['evaluation']['metrics']['confusion_matrix'][web][web], 0)


if __name__ == '__main__':
    unittest.main()
