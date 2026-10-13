"""Public operating-point audit; private training arrays are never opened."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.scaffold_operating_points import ARMS, primary_diagnostic
from src.models.official_streaming import confusion_metrics

ROOT = Path(__file__).parents[1]
REPORTS = ROOT/'reports/federated_class_safeguards'


class OperatingPointReceiptTests(unittest.TestCase):
    def setUp(self):
        self.e = json.loads((REPORTS/'operating-points-results.json').read_text())
        self.r = self.e['primary']
        self.p = json.loads((REPORTS/'operating-points-plan.json').read_text())
        self.old = json.loads((REPORTS/'scaffold-results.json').read_text())['primary']

    def test_sources_plan_and_exact_replay(self):
        for field, path in (
            ('script_sha256', ROOT/'src/eval/scaffold_operating_points.py'),
            ('helper_sha256', ROOT/'src/eval/federated_class_safeguard.py'),
            ('architecture_sha256', ROOT/'src/models/architectures.py'),
            ('metrics_source_sha256', ROOT/'src/models/official_streaming.py'),
            ('plan_sha256', REPORTS/'operating-points-plan.json'),
            ('evidence_sha256', REPORTS/'scaffold-results.json'),
        ):
            self.assertEqual(self.r[field], sha256(path))
        stable = copy.deepcopy(self.r)
        stable.pop('elapsed_seconds')
        digest = hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        self.assertEqual(digest.hexdigest(), self.e['replay']['non_timing_sha256'])
        self.assertTrue(self.e['replay']['all_non_timing_fields_exact'])
        self.assertEqual(self.r['endpoint_sha256'], self.p['endpoint_sha256'])

    def test_training_only_work_and_partition_coverage(self):
        self.assertEqual(self.r['status'], 'complete')
        for key in ('validation_rows_opened', 'test_opened', 'test_evaluated', 'cloud_used'):
            self.assertFalse(self.r[key])
        for key, value in self.p['work_per_pass'].items():
            self.assertEqual(self.r[key], value)
        self.assertTrue(self.r['frozen_inputs_unchanged'])
        self.assertEqual(self.r['forward_examples'], self.e['replay']['forward_examples'])
        self.assertEqual(self.r['class_order'], CLASSES)
        counts = self.r['partition']['counts']
        expected = [106148, 1178561, 362423, 66277, 2391, 1269, 44095, 238836]
        np.testing.assert_array_equal(np.array(counts['calibration'])+counts['audit'], expected)
        self.assertTrue(min(counts['calibration']) > 0 and min(counts['audit']) > 0)
        self.assertEqual(sum(counts['calibration']), 1000223)
        self.assertEqual(sum(counts['audit']), 999777)
        self.assertTrue(self.r['partition']['all_rows_previously_available_for_model_fitting'])

    def test_argmax_bridges_and_every_metric_recompute(self):
        self.assertEqual(set(self.r['arms']), set(ARMS))
        for arm, r in self.r['arms'].items():
            self.assertTrue(r['argmax_historical_bridge_exact'])
            self.assertTrue(r['zero_threshold_argmax_bridge_exact'])
            self.assertEqual(r['model_sha256'], self.p['model_sha256'][arm])
            self.assertEqual(r['argmax_all_training'], self.old['rounds'][-1]['arms'][arm]['evaluation']['metrics'])
            for m in [r['argmax_all_training'], r['argmax_audit']]+[v['audit'] for v in r['targets'].values()]:
                self.assertEqual(m, confusion_metrics(np.array(m['confusion_matrix'])))
                self.assertEqual(set(m['per_class']), set(CLASSES))
            for v in r['targets'].values():
                cm = np.array(v['audit']['confusion_matrix'])
                np.testing.assert_array_equal(cm.sum(1), self.r['partition']['counts']['audit'])
                for c, name in enumerate(CLASSES[1:], 1):
                    self.assertEqual(v['audit_attack_alarm_recall'][name], 1-cm[c, 0]/cm[c].sum())
                    ceiling = r['attack_rank_recall_ceiling'][name]
                    self.assertEqual(ceiling['support'], int(cm[c].sum()))
                    self.assertEqual(ceiling['recall'], ceiling['correct_attack_rank']/ceiling['support'])
                    self.assertLessEqual(cm[c, c], ceiling['correct_attack_rank'])

    def test_all_targets_and_empirical_calibration_limits(self):
        for arm, r in self.r['arms'].items():
            self.assertEqual(set(r['targets']), {'100', '500', '1000'})
            thresholds, alerts = [], []
            for target in self.p['targets_basis_points']:
                v = r['targets'][str(target)]
                cal = v['calibration']
                self.assertEqual(v['target_false_alert_rate'], target/10000)
                self.assertEqual(cal['benign_rows'], self.r['partition']['counts']['calibration'][0])
                self.assertEqual(cal['allowed_false_alerts'], cal['benign_rows']*target//10000)
                self.assertLessEqual(cal['false_alerts'], cal['allowed_false_alerts'])
                self.assertEqual(cal['achieved_false_alert_rate'], cal['false_alerts']/cal['benign_rows'])
                self.assertGreaterEqual(cal['boundary_ties'], 1)
                thresholds.append(cal['threshold'])
                alerts.append(v['audit']['benign_false_alert_rate'])
            self.assertEqual(thresholds, sorted(thresholds, reverse=True))
            self.assertEqual(alerts, sorted(alerts))

    def test_frozen_primary_screen_and_rare_category_failure(self):
        d = self.r['diagnostic']
        self.assertEqual(d, primary_diagnostic(self.r['arms'], self.p))
        self.assertTrue(d['audit_operating_points_comparable_under_plan'])
        self.assertFalse(d['eligible_for_separately_planned_confirmation'])
        self.assertFalse(d['prior_training_screen_overturned'])
        self.assertFalse(d['generalization_established'])
        self.assertFalse(d['new_training_launched'])
        self.assertLess(d['macro_f1_gain'], .01)
        self.assertLess(d['rare_recall_gains']['Brute Force'], 0)
        self.assertFalse(self.old['screen']['safety_screen_passed'])
        for arm in ARMS:
            self.assertEqual(self.r['arms'][arm]['attack_rank_recall_ceiling']['Web-based']['correct_attack_rank'], 0)
        bf = CLASSES.index('Brute Force')
        a, b = [self.r['arms'][arm]['targets']['500']['audit']['confusion_matrix'][bf][bf]
                for arm in ('fedavg_sgd', 'scaffold')]
        self.assertEqual((a, b), (116, 111))


if __name__ == '__main__':
    unittest.main()
