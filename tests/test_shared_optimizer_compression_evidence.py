"""Validate published aggregate compression evidence without private artifacts."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import shared_optimizer_compression as screen
from src.eval import shared_optimizer_performance as training
from src.federated import compressed_shared_adam as codec
from src.models.official_streaming import confusion_metrics

ROOT=Path(__file__).parents[1]


class SharedOptimizerCompressionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=json.loads((screen.PLAN.parent/'compression-results.json').read_text())
        cls.s=cls.e['summary']
        cls.parent=json.loads(screen.PARENT.read_text())
        cls.base=json.loads(training.PLAN.read_text())

    def test_frozen_contract_and_audit_digest(self):
        s=self.s
        self.assertEqual(s['plan'],json.loads(screen.PLAN.read_text()))
        self.assertEqual(s['contract']['plan_sha256'],sha256(screen.PLAN))
        self.assertEqual(s['contract']['parent_evidence_sha256'],sha256(screen.PARENT))
        self.assertEqual(s['contract']['sources'],screen.sources())
        self.assertEqual(self.e['audit']['source_sha256'],sha256(ROOT/'src/eval/shared_optimizer_compression.py'))
        self.assertEqual(hashlib.sha256(json.dumps(s,indent=2).encode()).hexdigest(),self.e['audit']['summary_sha256'])

    def test_identity_bridge_and_fixed_client_samples(self):
        refs={r['arm']:r for r in self.parent['summary']['results'] if r['seed']==7}
        self.assertEqual(self.s['references'],refs)
        bridge=next(r for r in self.s['results'] if r['arm']=='fp32_bridge')
        old=refs['shared_adam']
        self.assertEqual(bridge['delivered_model_sha256'],old['final_model_sha256'])
        self.assertEqual(bridge['server_target_sha256'],old['final_model_sha256'])
        for k in ('full_development_default_metrics','audit_default_metrics','operating_points'):
            self.assertEqual(bridge[k],old[k])
        for r in self.s['results']:
            self.assertEqual(r['sample_chain_sha256'],old['sample_chain_sha256'])
            self.assertEqual(r['initial_model_sha256'],old['initial_model_sha256'])
            self.assertEqual(r['learning_rate'],.003)
            self.assertEqual(r['seed'],7)

    def test_work_and_serialized_payload_include_both_directions(self):
        model=training.initial_model(self.base)
        self.assertEqual(len(self.s['results']),4)
        self.assertEqual({r['arm'] for r in self.s['results']},set(self.s['plan']['arms']))
        for r in self.s['results']:
            self.assertEqual(r['settings'],self.s['plan']['arms'][r['arm']])
            wire=codec.expected_wire(model,self.base,r['settings']['bits'],2160)
            self.assertEqual(r['wire'],wire)
            self.assertEqual(r['work']['estimated_payload_bytes'],sum(wire.values()))
            self.assertEqual(r['units'],2160)
            self.assertEqual(r['work']['backward_examples'],21600000)
            self.assertEqual(r['work']['backward_batches'],43200)
            self.assertEqual(r['work']['optimizer_steps'],2160)
            self.assertEqual(r['work']['logical_barriers'],2160)
            self.assertEqual(r['work']['local_examples'],0)
            self.assertEqual(r['work']['anchor_examples'],0)
        self.assertEqual(self.s['new_training_backward_examples'],86400000)
        self.assertEqual(sum(r['work']['backward_examples'] for r in self.s['results']),86400000)

    def test_all_confusion_matrices_recalculate(self):
        support=[self.s['split']['counts']['audit'][c] for c in CLASSES]
        for r in self.s['results']:
            for m in [r['audit_default_metrics'],*(p['audit_metrics'] for p in r['operating_points'])]:
                cm=np.asarray(m['confusion_matrix'])
                self.assertEqual(cm.sum(axis=1).tolist(),support)
                for k,v in confusion_metrics(cm).items():
                    self.assertEqual(m[k],v)
            m=r['full_development_default_metrics']; cm=np.asarray(m['confusion_matrix'])
            self.assertEqual(int(cm.sum()),399552)
            self.assertEqual(m,confusion_metrics(cm))

    def test_calibration_split_targets_and_caps(self):
        self.assertEqual(self.s['split'],self.parent['summary']['split'])
        self.assertEqual(self.s['split']['cross_split_input_groups'],0)
        n=self.s['split']['counts']['calibration']['Benign']
        for r in self.s['results']:
            self.assertTrue(r['zero_margin_argmax_exact'])
            self.assertEqual([p['calibration']['target_basis_points'] for p in r['operating_points']],[100,500,1000,2000])
            for p in r['operating_points']:
                c=p['calibration']; allowed=n*c['target_basis_points']//10000
                self.assertEqual(c['calibration_benign_rows'],n)
                self.assertEqual(c['allowed_false_alerts'],allowed)
                self.assertLessEqual(c['observed_false_alerts'],allowed)
                self.assertEqual(c['ascending_order_statistic_index'],n-allowed-1)
                self.assertEqual(c['calibration_false_alert_rate'],c['observed_false_alerts']/n)

    def test_frozen_screen_and_control_not_selected(self):
        self.assertEqual(self.s['assessment'],screen.assess(self.s['results'],self.s['references'],self.s['plan']))
        self.assertIn(self.s['assessment']['candidate_for_replication'],(None,'ef4','ef2'))
        self.assertFalse(self.s['assessment']['previous_training_screen_overridden'])
        self.assertEqual(self.s['plan']['screen']['macro_f1_drop_from_shared_max'],.01)
        self.assertEqual(self.s['plan']['screen']['any_class_recall_drop_from_shared_max'],.05)
        self.assertEqual(self.s['plan']['screen']['absolute_far_difference_from_shared_max'],.005)

    def test_recovery_work_and_closed_holdouts(self):
        audit=self.e['audit']
        self.assertTrue(self.s['fp32_bridge_exact'])
        self.assertEqual(len(audit['checks']),4)
        self.assertEqual(sum(c['operating_points_exact'] for c in audit['checks']),16)
        self.assertTrue(all(c['recovery_exact'] and c['checkpoint_inference_exact'] and c['work_and_serialized_bytes_verified']
                            for c in audit['checks']))
        self.assertEqual(audit['new_recovery_backward_examples'],80000)
        self.assertEqual(audit['new_forward_examples'],4*399552)
        self.assertEqual(self.s['forward_examples'],4*399552)
        for r in (self.s,audit):
            for key in ('confirmation_opened','original_validation_opened','test_opened','test_evaluated'):
                self.assertFalse(r[key])
        self.assertFalse(self.s['network_measured'])


if __name__=='__main__':
    unittest.main()
