"""Check the public numerical-audit receipt without opening private arrays."""
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
REPORTS = ROOT/'reports/federated_shared_optimizer'


class SharedAdamEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads((REPORTS/'bridge-results.json').read_text())
        cls.r = cls.receipt['primary']
        cls.plan = json.loads((REPORTS/'bridge-plan.json').read_text())

    def test_plan_and_source_bytes_match(self):
        self.assertEqual(hashlib.sha256((REPORTS/'bridge-plan.json').read_bytes()).hexdigest(),self.r['plan_sha256'])
        for name,expected in self.r['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected)
        for field in ('packed_manifest_sha256','partition_sha256'):
            self.assertEqual(self.r[field],self.plan[field])

    def test_both_original_receipts_reconstruct_from_shared_fields(self):
        self.assertTrue(self.receipt['independent_replay']['non_timing_exact'])
        for suffix in ('v1','replay-v1'):
            r=copy.deepcopy(self.r)
            if suffix=='replay-v1':
                for name,row in r['cases'].items():
                    row['elapsed_seconds']=self.receipt['independent_replay']['elapsed_seconds'][name]
            encoded=json.dumps(r,indent=2,allow_nan=False).encode()
            expected=self.receipt['source_files_sha256'][f'outputs/shared-adam-bridge-{suffix}/result.json']
            self.assertEqual(hashlib.sha256(encoded).hexdigest(),expected)

    def test_every_precision_dropout_case_and_every_step_passes(self):
        self.assertEqual(len(self.r['cases']),4)
        expected={f"{c['dtype']}-dropout{c['dropout']}" for c in self.plan['cases']}
        self.assertEqual(set(self.r['cases']),expected)
        for row in self.r['cases'].values():
            self.assertTrue(row['numerical_equivalence_passed'])
            self.assertTrue(row['recovery_exact'])
            self.assertEqual(row['num_parameters'],5096)
            self.assertEqual([r['step'] for r in row['steps']],list(range(1,13)))
            self.assertEqual(row['recovery_steps'],row['steps'][6:])
            for step in row['steps']:
                self.assertEqual(set(step['checks']),{'raw_gradient','clipped_gradient','parameters','exp_avg','exp_avg_sq','step'})
                for check in step['checks'].values():
                    self.assertTrue(check['within_tolerance'])
                    self.assertTrue(math.isfinite(check['max_abs']) and math.isfinite(check['l2']))
                    self.assertLessEqual(check['max_abs'],row['tolerance']['atol'])
                    self.assertGreaterEqual(check['l2'],check['max_abs'])
                self.assertTrue(step['checks']['step']['bit_exact'])

    def test_samples_are_paired_with_population_weights_not_uniform_counts(self):
        counts=self.r['client_population_counts']
        self.assertEqual(len(counts),20)
        self.assertEqual(sum(counts),2000000)
        self.assertGreater(max(counts),min(counts))
        for n,classes in zip(counts,self.r['client_class_counts']):
            self.assertEqual(sum(classes),n)
        schedules=[]
        for row in self.r['cases'].values():
            schedules.append([{k:s[k] for k in ('sample_sha256','dropout_seeds','sample_class_counts')} for s in row['steps']])
            for step in row['steps']:
                self.assertEqual(sum(step['sample_class_counts']),20*512)
                self.assertEqual(len(step['dropout_seeds']),20)
        self.assertTrue(all(s==schedules[0] for s in schedules))

    def test_work_and_estimated_payload(self):
        total={key:0 for key in self.plan['work_per_pass']}
        for row in self.r['cases'].values():
            self.assertEqual(row['work'],{'backward_examples':368640,'backward_batches':720,
                                          'optimizer_steps':36,'quality_evaluation_examples':0})
            for key,value in row['work'].items(): total[key]+=value
            size=5096*(8 if row['case']['dtype']=='float64' else 4)
            self.assertEqual(row['parameter_bytes'],size)
            self.assertEqual(row['communication']['estimated_per_path_bytes'],size*2*20*12)
            self.assertFalse(row['communication']['measured_network_traffic'])
            self.assertFalse(row['communication']['includes_recovery_audit'])
        self.assertEqual(total,self.plan['work_per_pass'])
        self.assertEqual(total,self.r['total_work'])

    def test_correctness_is_not_performance_evidence(self):
        self.assertEqual(self.r['status'],'complete')
        for key in ('validation_rows_opened','test_opened','test_evaluated','quality_metrics_computed',
                    'cloud_used','historical_checkpoint_loaded'):
            self.assertFalse(self.r[key])
        self.assertFalse(self.plan['performance_training_authorized'])
        # Central versus federated rounding is NOT bit-identical; recovery is.
        self.assertTrue(all(not r['central_federated_bit_exact'] for r in self.r['cases'].values()))


if __name__=='__main__':
    unittest.main()
