"""Verify aggregate replication receipts without opening private model/data files."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import shared_optimizer_replication as rep
from src.models.official_streaming import confusion_metrics

ROOT = Path(__file__).parents[1]


class SharedOptimizerReplicationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence=json.loads((rep.PLAN.parent/'replication-results.json').read_text())
        cls.s=cls.evidence['summary']
        cls.parent=json.loads(rep.EVIDENCE.read_text())
        cls.performance=json.loads((rep.PLAN.parent/'performance-results.json').read_text())

    def test_hashes_and_frozen_settings(self):
        s=self.s
        self.assertEqual(s['plan'],json.loads(rep.PLAN.read_text()))
        self.assertEqual(s['contract']['plan_sha256'],sha256(rep.PLAN))
        self.assertEqual(s['plan']['alarm_evidence_sha256'],sha256(rep.EVIDENCE))
        for name,digest in s['contract']['sources'].items():
            self.assertEqual(sha256(ROOT/name),digest)
        self.assertEqual(hashlib.sha256(json.dumps(s,indent=2).encode()).hexdigest(),
                         self.evidence['audit']['summary_sha256'])
        self.assertEqual(self.evidence['audit']['source_sha256'],sha256(ROOT/'src/eval/shared_optimizer_replication.py'))
        self.assertEqual(s['plan']['learning_rates'],{'central_adam':.001,'shared_adam':.003,'fedavg_adam':.003})

    def test_nine_endpoints_but_only_six_new_runs(self):
        s=self.s
        self.assertEqual(len(s['results']),9)
        self.assertEqual({(r['seed'],r['arm']) for r in s['results']},
                         {(seed,arm) for seed in (7,17,27) for arm in s['plan']['arms']})
        self.assertEqual(sum(not r['reused_discovery'] for r in s['results']),s['new_training_runs'])
        self.assertEqual(s['new_training_runs'],6)
        self.assertEqual(s['new_backward_examples'],129600000)
        self.assertEqual(s['new_forward_examples'],6*399552)
        self.assertEqual(sum(r['work']['backward_examples'] for r in s['results'] if not r['reused_discovery']),129600000)
        base=self.performance['summary']['plan']
        for r in s['results']:
            self.assertEqual(r['work'],rep.expected_work(base,r['arm'],20384))
            self.assertEqual(r['learning_rate'],s['plan']['learning_rates'][r['arm']])
            if r['seed']==7:
                old=next(p for p in self.parent['summary']['results'] if p['arm']==r['arm'])
                for k,v in old.items():
                    self.assertEqual(r[k],v)
                self.assertTrue(r['reused_discovery'])
            else:
                self.assertFalse(r['reused_discovery'])

    def test_paired_initialization_and_separate_seeds(self):
        hashes=[]
        for seed in (7,17,27):
            h={r['initial_model_sha256'] for r in self.s['results'] if r['seed']==seed}
            self.assertEqual(len(h),1)
            hashes.extend(h)
        self.assertEqual(len(set(hashes)),3)

    def test_confusion_matrices_and_class_support(self):
        support=[self.s['split']['counts']['audit'][c] for c in CLASSES]
        for r in self.s['results']:
            for m in [r['audit_default_metrics'],*(p['audit_metrics'] for p in r['operating_points'])]:
                cm=np.asarray(m['confusion_matrix'])
                self.assertEqual(cm.sum(axis=1).tolist(),support)
                for k,v in confusion_metrics(cm).items():
                    self.assertEqual(m[k],v)
            cm=np.asarray(r['full_development_default_metrics']['confusion_matrix'])
            self.assertEqual(int(cm.sum()),399552)
            self.assertEqual(confusion_metrics(cm),r['full_development_default_metrics'])

    def test_same_calibration_split_and_count_caps(self):
        self.assertEqual(self.s['split'],self.parent['summary']['split'])
        self.assertEqual(self.s['split']['cross_split_input_groups'],0)
        n=self.s['split']['counts']['calibration']['Benign']
        for r in self.s['results']:
            self.assertEqual([p['calibration']['target_basis_points'] for p in r['operating_points']],
                             [100,500,1000,2000])
            for p in r['operating_points']:
                c=p['calibration']; allowed=n*c['target_basis_points']//10000
                self.assertEqual(c['calibration_benign_rows'],n)
                self.assertEqual(c['allowed_false_alerts'],allowed)
                self.assertLessEqual(c['observed_false_alerts'],allowed)
                self.assertEqual(c['calibration_false_alert_rate'],c['observed_false_alerts']/n)
                self.assertEqual(c['ascending_order_statistic_index'],n-allowed-1)
                self.assertTrue(np.isfinite(c['threshold']))

    def test_paired_arithmetic_and_descriptive_statistics(self):
        self.assertEqual(self.s['comparisons'],rep.paired_comparisons(self.s['results'],self.s['plan']))
        self.assertEqual(self.s['descriptive'],rep.describe(self.s['results'],self.s['plan']))
        for c in self.s['comparisons']:
            self.assertEqual(c['actual_far_comparable'],abs(c['shared_minus_reference_far'])<=.005)
        self.assertEqual(len(self.s['comparisons']),24)
        for d in self.s['descriptive']:
            self.assertEqual(d['macro_f1']['n'],3)

    def test_audit_and_closed_holdouts(self):
        audit=self.evidence['audit']
        self.assertEqual(len(audit['checks']),6)
        self.assertEqual(sum(c['operating_points_exact'] for c in audit['checks']),24)
        self.assertTrue(all(c['checkpoint_inference_exact'] and c['work_verified'] for c in audit['checks']))
        self.assertEqual(audit['new_backward_examples'],0)
        self.assertEqual(audit['new_forward_examples'],6*399552)
        self.assertEqual(audit['reused_discovery_models_verified'],3)
        for r in (self.s,audit):
            for k in ('confirmation_opened','original_validation_opened','test_opened','test_evaluated'):
                self.assertFalse(r[k])
        self.assertFalse(self.s['network_measured'])
        self.assertFalse(self.s['previous_screen_overridden'])
        self.assertFalse(self.performance['summary']['screen']['shared_adam']['passes_development_screen'])


if __name__=='__main__':
    unittest.main()
