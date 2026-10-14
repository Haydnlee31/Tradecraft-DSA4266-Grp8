"""Check published alarm diagnostics without opening private rows or weights."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.shared_optimizer_alarms import PLAN, EVIDENCE, ATTACKS, BENIGN, comparisons
from src.models.official_streaming import confusion_metrics

ROOT = Path(__file__).parents[1]


class SharedOptimizerAlarmEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads((ROOT/'reports/federated_shared_optimizer/alarm-results.json').read_text())
        cls.summary = cls.evidence['summary']
        cls.parent = json.loads(EVIDENCE.read_text())

    def test_frozen_source_and_receipt_hashes(self):
        s = self.summary
        self.assertEqual(s['plan'],json.loads(PLAN.read_text()))
        self.assertEqual(s['plan_sha256'],sha256(PLAN))
        self.assertEqual(s['plan']['performance_evidence_sha256'],sha256(EVIDENCE))
        for name,digest in s['sources'].items():
            self.assertEqual(sha256(ROOT/name),digest)
        self.assertEqual(hashlib.sha256(json.dumps(s,indent=2).encode()).hexdigest(),
                         self.evidence['replay']['summary_sha256'])
        self.assertEqual(self.evidence['replay']['source_sha256'],sha256(ROOT/'src/eval/shared_optimizer_alarms.py'))

    def test_same_selected_models_and_historical_bridges(self):
        s = self.summary
        self.assertEqual(len(s['results']),5)
        self.assertEqual({r['arm']:r['learning_rate'] for r in s['results']},
                         self.parent['summary']['selected_learning_rates'])
        for r in s['results']:
            old = next(p for p in self.parent['summary']['results']
                        if p['arm']==r['arm'] and p['learning_rate']==r['learning_rate'])
            check = next(p for p in self.parent['audit']['checks']
                         if p['arm']==r['arm'] and p['learning_rate']==r['learning_rate'])
            self.assertEqual(r['checkpoint_sha256'],check['checkpoint_sha256'])
            self.assertEqual(r['final_model_sha256'],old['final_model_sha256'])
            self.assertEqual(r['full_development_default_metrics'],old['development_metrics'])
            self.assertTrue(r['zero_margin_argmax_exact'])
            self.assertTrue(r['historical_default_endpoint_exact'])

    def test_all_confusion_matrices_and_detection_decomposition(self):
        expected_support = [self.summary['split']['counts']['audit'][n] for n in CLASSES]
        for r in self.summary['results']:
            for m in [r['audit_default_metrics'],*(p['audit_metrics'] for p in r['operating_points'])]:
                cm = np.array(m['confusion_matrix'])
                self.assertEqual(cm.sum(axis=1).tolist(),expected_support)
                for k,v in confusion_metrics(cm).items():
                    self.assertEqual(m[k],v)
                self.assertEqual(m['attack_detection_recall'],float(1-cm[ATTACKS,BENIGN].sum()/cm[ATTACKS].sum()))
                self.assertEqual(m['macro_attack_class_recall'],
                                 float(np.mean([m['per_class'][CLASSES[i]]['recall'] for i in ATTACKS])))
                for i in ATTACKS:
                    self.assertEqual(m['per_attack_detection_recall'][CLASSES[i]],float(1-cm[i,BENIGN]/cm[i].sum()))

    def test_targets_and_exact_calibration_caps(self):
        n = self.summary['split']['counts']['calibration'][CLASSES[BENIGN]]
        self.assertEqual(n,10562)
        self.assertEqual(self.summary['plan']['primary_target_basis_points'],500)
        for r in self.summary['results']:
            self.assertEqual([p['calibration']['target_basis_points'] for p in r['operating_points']],
                             [100,500,1000,2000])
            for p in r['operating_points']:
                c = p['calibration']
                allowed = n*c['target_basis_points']//10000
                self.assertEqual(c['calibration_benign_rows'],n)
                self.assertEqual(c['allowed_false_alerts'],allowed)
                self.assertEqual(c['ascending_order_statistic_index'],n-allowed-1)
                self.assertLessEqual(c['observed_false_alerts'],allowed)
                self.assertEqual(c['calibration_false_alert_rate'],c['observed_false_alerts']/n)
                self.assertGreaterEqual(c['boundary_ties'],1)
                self.assertTrue(np.isfinite(c['threshold']))

    def test_comparisons_use_actual_rates_and_preserve_failed_training_gate(self):
        s = self.summary
        self.assertEqual(s['comparisons'],comparisons(s['results'],s['plan']))
        self.assertEqual(len(s['comparisons']),16)
        self.assertFalse(s['previous_screen_overridden'])
        self.assertFalse(self.parent['summary']['screen']['shared_adam']['passes_development_screen'])
        self.assertFalse(self.parent['summary']['screen']['mime']['passes_development_screen'])
        # Comparability is not silently asserted when observed FARs differ too far.
        flags = [(c['target_basis_points'],c['candidate'],c['reference']) for c in s['comparisons']
                 if not c['audit_far_within_comparability_tolerance']]
        self.assertEqual(flags,[(1000,'shared_adam','central_adam')])

    def test_grouped_boundaries_replay_and_zero_training(self):
        s, r = self.summary,self.evidence['replay']
        self.assertEqual({k:sum(v.values()) for k,v in s['split']['counts'].items()},
                         {'calibration':200022,'audit':199530})
        self.assertEqual(s['split']['cross_split_input_groups'],0)
        self.assertEqual(s['forward_examples'],1997760)
        self.assertEqual(s['new_model_fits'],0)
        self.assertEqual(s['backward_examples'],0)
        self.assertTrue(r['replay_exact'])
        self.assertEqual(len(r['checks']),5)
        self.assertEqual(sum(c['operating_points_exact'] for c in r['checks']),20)
        self.assertEqual(r['new_forward_examples'],0)
        self.assertEqual(r['backward_examples'],0)
        for receipt in (s,r):
            for key in ('confirmation_opened','confirmation_scored','original_validation_opened','test_opened','test_evaluated'):
                self.assertFalse(receipt[key])


if __name__=='__main__':
    unittest.main()
