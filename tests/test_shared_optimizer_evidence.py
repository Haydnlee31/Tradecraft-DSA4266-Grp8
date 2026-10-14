"""Public numerical evidence is auditable without private arrays or checkpoints."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.shared_optimizer_performance import work_plan, screen
from src.models.official_streaming import confusion_metrics

ROOT = Path(__file__).parents[1]


class SharedOptimizerEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads((ROOT/'reports/federated_shared_optimizer/performance-results.json').read_text())
        cls.summary = cls.evidence['summary']
        cls.plan = cls.summary['plan']

    def test_receipt_and_implementation_identity(self):
        raw = json.dumps(self.summary,indent=2).encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),self.evidence['audit']['summary_sha256'])
        self.assertEqual(self.summary['contract']['plan_sha256'],
                         sha256(ROOT/'reports/federated_shared_optimizer/performance-plan.json'))
        self.assertEqual(self.plan,json.loads((ROOT/'reports/federated_shared_optimizer/performance-plan.json').read_text()))
        for name,digest in self.summary['contract']['sources'].items():
            self.assertEqual(sha256(ROOT/name),digest)
        self.assertEqual(sha256(ROOT/'src/eval/shared_optimizer_audit.py'),self.evidence['audit']['auditor_sha256'])
        self.assertEqual(hashlib.sha256(json.dumps(self.summary['data'],indent=2).encode()).hexdigest(),
                         self.summary['contract']['data_manifest_sha256'])

    def test_all_ten_endpoints_metrics_and_equal_work(self):
        r = self.summary['results']
        self.assertEqual(len(r),10)
        self.assertEqual({(x['arm'],x['learning_rate']) for x in r},
                         {(a,lr) for a in self.plan['arms'] for lr in self.plan['learning_rates']})
        self.assertEqual(len({x['initial_model_sha256'] for x in r}),1)
        for x in r:
            cm = np.array(x['development_metrics']['confusion_matrix'])
            self.assertEqual(x['development_metrics'],confusion_metrics(cm))
            self.assertEqual(set(x['development_metrics']['per_class']),set(CLASSES))
            self.assertEqual(int(cm.sum()),399552)
            self.assertEqual(x['work']['backward_examples'],21600000)
            self.assertEqual(x['units'],work_plan(self.plan,x['arm'])[0])
            self.assertEqual(x['parameter_bytes'],x['num_parameters']*4)
            self.assertEqual(x['contract'],self.summary['contract'])
            self.assertGreater(x['training_seconds'],0)
        self.assertEqual(sum(x['work']['backward_examples'] for x in r),216000000)
        self.assertEqual(self.summary['total_backward_examples'],216000000)

    def test_selection_and_screen_not_edited_after_scores(self):
        chosen = {a:max((r for r in self.summary['results'] if r['arm']==a),
                       key=lambda r:(r['development_metrics']['macro_f1'],-r['learning_rate'])) for a in self.plan['arms']}
        self.assertEqual(self.summary['selected_learning_rates'],{a:r['learning_rate'] for a,r in chosen.items()})
        self.assertEqual(self.summary['screen'],screen(chosen,self.plan))

    def test_full_anchor_work_and_modeled_payload(self):
        for r in self.summary['results']:
            work, arm = r['work'],r['arm']
            payload = {'central_adam':0,'central_mime_base':0,'shared_adam':1761177600,
                       'fedavg_adam':88058880,'mime':58705920}[arm]
            self.assertEqual(work['estimated_payload_bytes'],payload)
            if arm=='mime':
                self.assertEqual(work['anchor_examples'],12000000)
                self.assertEqual(work['local_examples'],4800000)
                self.assertEqual(work['backward_examples'],work['anchor_examples']+2*work['local_examples'])
                self.assertEqual(work['logical_barriers'],48)
            self.assertFalse(r['network_measured'])

    def test_boundaries_and_checkpoint_replays(self):
        self.assertEqual({k:sum(v.values()) for k,v in self.summary['data']['counts'].items()},
                         {'fit':500000,'development':399552,'confirmation':398502})
        self.assertEqual(self.summary['data']['float32_collision_rows_removed'],{'development':674,'confirmation':1126})
        self.assertEqual(sum(self.summary['data']['fit_client_counts']),500000)
        self.assertEqual(len(self.summary['data']['fit_client_counts']),20)
        for r in [self.summary,self.evidence['audit'],*self.summary['results']]:
            for k in ('confirmation_scored','original_validation_opened','test_opened','test_evaluated'):
                self.assertFalse(r[k])
        audit = self.evidence['audit']
        self.assertEqual(len(audit['checks']),10)
        self.assertTrue(all(c['development_replay_exact'] for c in audit['checks']))
        self.assertEqual({(c['arm'],c['learning_rate']) for c in audit['checks']},
                         {(r['arm'],r['learning_rate']) for r in self.summary['results']})
        self.assertEqual(audit['additional_backward_examples'],0)
        self.assertEqual(audit['development_evaluation_examples'],3995520)


if __name__=='__main__':
    unittest.main()
