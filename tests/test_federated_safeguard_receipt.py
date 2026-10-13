"""Audit the public diagnostic evidence without opening private data or weights."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.federated_class_safeguard import gate, PROTECTED
from src.models.official_streaming import confusion_metrics

ROOT = Path(__file__).parents[1]
REPORTS = ROOT/'reports/federated_class_safeguards'


class SafeguardReceiptTests(unittest.TestCase):
    def setUp(self):
        self.evidence = json.loads((REPORTS/'diagnostic-results.json').read_text())
        self.r = self.evidence['primary']
        self.plan = json.loads((REPORTS/'diagnostic-plan.json').read_text())

    def test_frozen_plan_source_and_replay_digest(self):
        self.assertEqual(self.r['plan_sha256'], sha256(REPORTS/'diagnostic-plan.json'))
        self.assertEqual(self.r['script_sha256'], sha256(ROOT/'src/eval/federated_class_safeguard.py'))
        stable = copy.deepcopy(self.r)
        for case in stable['cases']:
            case.pop('elapsed_seconds')
        encoded = json.dumps(stable, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), self.evidence['replay']['non_timing_sha256'])
        self.assertTrue(self.evidence['replay']['all_non_timing_fields_exact'])

    def test_closed_evaluation_and_fixed_work_budget(self):
        self.assertEqual(self.r['status'], 'complete')
        for key in ('validation_rows_opened', 'test_opened', 'test_evaluated'):
            self.assertFalse(self.r[key])
        self.assertEqual([c['source_step'] for c in self.r['cases']], [2, 5, 20])
        self.assertEqual(self.r['work'], self.plan['work'])
        self.assertEqual(self.r['work'], self.evidence['replay']['work'])
        self.assertEqual(sum(w['optimizer_steps'] for c in self.r['cases'] for w in c['client_work']), 11751)
        self.assertEqual(sum(c['gradient_work']['gradient_examples'] for c in self.r['cases']), 6000000)

    def test_every_metric_recomputes_from_confusion(self):
        for c in self.r['cases']:
            counts = np.array(c['gradient_work']['client_counts']).sum(axis=0)
            for arm in ('before', 'fedavg', 'corrected'):
                metrics = c[arm]['metrics']
                cm = np.array(metrics['confusion_matrix'])
                np.testing.assert_array_equal(cm.sum(axis=1), counts)
                self.assertEqual(metrics, confusion_metrics(cm))
                self.assertEqual(set(metrics['per_class']), set(CLASSES))

    def test_gate_is_not_relaxed_and_losses_match_reported_changes(self):
        self.assertEqual(self.r['gate'], gate(self.r['cases'], self.plan))
        self.assertEqual(self.r['gate']['decision'], 'stop_no_full_training')
        self.assertEqual(self.r['gate']['safe_checkpoint_flags'], [True, False, False])
        for c in self.r['cases']:
            for i, name in enumerate(CLASSES):
                values = c['per_class'][name]
                self.assertEqual(values['fedavg_actual_ce_change'], c['fedavg']['mean_class_ce'][i]-c['before']['mean_class_ce'][i])
                self.assertEqual(values['projected_actual_ce_change'], c['corrected']['mean_class_ce'][i]-c['before']['mean_class_ce'][i])
                # Flower averages in FP32; weighted deltas are audited in FP64.
                bound = values['gradient_norm']*c['flower_rounding_delta_norm']+1e-10
                self.assertLessEqual(abs(sum(values['weighted_client_directional_contributions'])-
                                         values['fedavg_predicted_ce_change']), bound)
            self.assertGreater(c['per_class']['Brute Force']['fedavg_actual_ce_change'], 0)
            self.assertLess(c['per_class']['Brute Force']['projected_actual_ce_change'], 0)
            for name in PROTECTED[1:]:
                self.assertEqual(c['corrected']['metrics']['per_class'][name]['recall'], 0.)


if __name__ == '__main__':
    unittest.main()
