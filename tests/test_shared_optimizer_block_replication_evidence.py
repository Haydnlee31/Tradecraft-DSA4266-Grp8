"""Check the aggregate seed replication evidence without private training files."""
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import shared_optimizer_block_replication as screen
from src.eval import shared_optimizer_block_compression as block
from src.eval import shared_optimizer_performance as training
from src.federated import block_shared_adam as codec
from src.models.official_streaming import confusion_metrics


class BlockReplicationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=json.loads((screen.PLAN.parent/'block-replication-results.json').read_text())
        cls.s=cls.e['summary']; cls.plan=cls.s['plan']
        cls.parent=json.loads(screen.PARENT.read_text())
        cls.refs=json.loads(screen.REFERENCES.read_text())['summary']['results']
        cls.base=json.loads(training.PLAN.read_text())

    def test_pinned_contract_and_independent_audit_digest(self):
        self.assertEqual(self.plan,json.loads(screen.PLAN.read_text()))
        self.assertEqual(self.s['contract'],screen.contract())
        self.assertEqual(self.e['audit']['summary_sha256'],hashlib.sha256(json.dumps(self.s,indent=2).encode()).hexdigest())
        self.assertEqual(self.e['audit']['source_sha256'],sha256(screen.__file__))
        self.assertEqual(self.s['references'],self.refs)
        self.assertEqual(self.s['split'],self.parent['summary']['split'])

    def test_discovery_is_preserved_and_only_two_seeds_are_new(self):
        discovery=next(r for r in self.parent['summary']['results'] if r['arm']=='block128')
        self.assertEqual([r for r in self.s['results'] if r['reused_discovery']],[{**discovery,'reused_discovery':True}])
        new=[r for r in self.s['results'] if not r['reused_discovery']]
        self.assertEqual(len(new),2); self.assertEqual({r['seed'] for r in new},{17,27})
        self.assertEqual(self.s['new_training_runs'],2)
        for r in self.s['results']:
            screen.check_pairing(r,self.refs,self.plan)
            self.assertEqual(r['settings'],{'bits':4,'feedback':True,'block_size':128})
            self.assertEqual(r['learning_rate'],.003)

    def test_budget_and_serialization_counts_both_directions(self):
        model=training.initial_model(self.base)
        wire=codec.expected_wire(model,self.base,4,2160,128)
        self.assertEqual(sum(wire.values()),240255840)
        for r in self.s['results']:
            self.assertEqual(r['wire'],wire)
            self.assertEqual(r['work']['estimated_payload_bytes'],240255840)
            self.assertEqual(r['work']['backward_examples'],21600000)
            self.assertEqual(r['work']['backward_batches'],43200)
            self.assertEqual(r['units'],2160)
            self.assertEqual(r['work']['optimizer_steps'],2160)
            self.assertEqual(r['work']['logical_barriers'],2160)
            self.assertEqual(r['work']['local_examples'],0)
        self.assertEqual(sum(r['work']['backward_examples'] for r in self.s['results'] if not r['reused_discovery']),43200000)
        self.assertEqual(self.s['new_training_backward_examples'],43200000)
        self.assertEqual(self.e['audit']['new_recovery_backward_examples'],40000)

    def test_every_confusion_matrix_recalculates_all_class_metrics(self):
        support=[self.s['split']['counts']['audit'][c] for c in CLASSES]
        for r in self.s['results']:
            for m in [r['audit_default_metrics'],*(p['audit_metrics'] for p in r['operating_points'])]:
                cm=np.asarray(m['confusion_matrix'])
                self.assertEqual(cm.sum(axis=1).tolist(),support)
                for k,v in confusion_metrics(cm).items(): self.assertEqual(m[k],v)
            m=r['full_development_default_metrics']
            self.assertEqual(int(np.asarray(m['confusion_matrix']).sum()),399552)
            self.assertEqual(m,confusion_metrics(np.asarray(m['confusion_matrix'])))

    def test_calibration_targets_caps_and_group_separation(self):
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

    def test_each_seed_assessment_is_recomputed_not_averaged(self):
        expected=screen.assess(self.s['results'],self.refs,self.plan)
        self.assertEqual(self.s['assessment'],expected)
        self.assertEqual(expected['all_seeds_pass'],all(r['passes'] for r in expected['per_seed']))
        self.assertEqual(self.plan['screen'],self.parent['summary']['plan']['screen'])
        self.assertFalse(expected['confirmation_scoring_authorized'])
        self.assertFalse(expected['previous_training_screen_overridden'])

    def test_paired_comparisons_and_descriptive_statistics(self):
        self.assertEqual(self.s['comparisons'],screen.comparisons(self.s['results'],self.refs,self.plan))
        self.assertEqual(len(self.s['comparisons']),36)
        self.assertEqual(self.s['descriptive'],screen.describe(self.s['results'],self.refs,self.plan))
        for record in self.s['descriptive']:
            points=[next(p['audit_metrics'] for p in r['operating_points'] if p['calibration']['target_basis_points']==record['target_basis_points'])
                    for r in [*self.refs,*self.s['results']] if r['arm']==record['arm']]
            for key in ('macro_f1','benign_false_alert_rate'):
                values=[p[key] for p in points]; mean=math.fsum(values)/3
                self.assertEqual(record[key]['n'],3)
                self.assertAlmostEqual(record[key]['mean'],mean,places=14)
                self.assertAlmostEqual(record[key]['sample_sd'],math.sqrt(math.fsum((v-mean)**2 for v in values)/2),places=14)

    def test_recovery_audits_and_closed_holdouts(self):
        audit=self.e['audit']
        self.assertEqual(audit['status'],'complete')
        self.assertEqual({c['seed'] for c in audit['checks']},{17,27})
        self.assertEqual(sum(c['operating_points_exact'] for c in audit['checks']),8)
        self.assertTrue(all(c['recovery_exact'] and c['checkpoint_inference_exact'] and c['same_seed_pairing_verified'] and
                            c['work_and_serialized_bytes_verified'] for c in audit['checks']))
        self.assertTrue(audit['discovery_reused_not_retrained'])
        self.assertEqual(self.s['new_forward_examples'],2*399552)
        self.assertEqual(audit['new_forward_examples'],2*399552)
        for r in (self.s,audit):
            for key in ('confirmation_opened','original_validation_opened','test_opened','test_evaluated'):
                self.assertFalse(r[key])
        self.assertFalse(self.s['network_measured'])


if __name__=='__main__':
    unittest.main()
