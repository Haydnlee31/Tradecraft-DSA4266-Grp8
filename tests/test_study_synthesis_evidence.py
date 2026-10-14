"""Portable audits of the A/B findings; no private files, training or timing gate.

Recorded timings are recomputed from their samples, never re-benchmarked in CI.
The original final evidence supplies quality; the new benchmark supplies cost.
"""
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import statistics
import unittest

from scripts import official39_study_synthesis as synthesis
from scripts import official39_final_report as final

ROOT = Path(__file__).parents[1]
REPORTS = ROOT / 'reports/federated_class_safeguards'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PublishedStudyEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.synthesis = read(REPORTS / 'study-synthesis-results.json')
        cls.cost = read(REPORTS / 'cost-benchmark-results.json')
        cls.plan = read(REPORTS / 'cost-benchmark-plan.json')

    def test_synthesis_and_tables_rebuild_from_public_counts(self):
        rebuilt = synthesis.build()
        # The frozen producer records native relative path separators. Compare
        # portable path identities, preserving every source hash and metric.
        # Do not edit the producer or rewrite the published historical receipt.
        rebuilt['source_sha256'] = {PureWindowsPath(name).as_posix(): digest
                                   for name, digest in rebuilt['source_sha256'].items()}
        self.assertEqual(rebuilt, self.synthesis)
        self.assertEqual(synthesis.tables(self.synthesis),
                         (REPORTS / 'study-synthesis-tables.md').read_text(encoding='utf-8'))

    def test_class_gap_decomposition_preserves_the_total(self):
        for pair, classes in self.synthesis['per_class_macro_f1_gap_contributions'].items():
            self.assertEqual(set(classes), set(self.synthesis['class_order']))
            self.assertAlmostEqual(sum(classes.values()), self.synthesis['paired_macro_f1_gaps'][pair]['mean'])
            a, b = pair.split('_minus_')
            for name, value in classes.items():
                self.assertAlmostEqual(value, (self.synthesis['groups'][a]['per_class'][name]['f1']['mean']
                                              - self.synthesis['groups'][b]['per_class'][name]['f1']['mean']) / 8)

    def test_recorded_benchmark_source_and_input_plan_are_pinned(self):
        paths = {
            'plan_sha256': REPORTS / 'cost-benchmark-plan.json',
            'script_sha256': ROOT / 'src/eval/official39_cost_benchmark.py',
            'final_evidence_sha256': final.EVIDENCE,
            'final_plan_sha256': ROOT / final.PLAN,
            'loader_sha256': ROOT / 'scripts/official39_final_evaluation.py',
            'architecture_sha256': ROOT / 'src/models/architectures.py',
        }
        for field, path in paths.items():
            self.assertEqual(self.cost['primary'][field], digest(path))
        p = read(ROOT / final.PLAN)
        self.assertEqual(self.cost['primary']['packed_manifest_sha256'],
                         p['data_identity']['packed_manifest_sha256'])

    def test_replay_and_original_receipt_bytes(self):
        self.assertTrue(self.cost['non_timing_replay_exact'])
        a, b = (self.cost[k] for k in ('primary', 'replay'))
        self.assertEqual({k:v for k,v in a.items() if k != 'measurements'},
                         {k:v for k,v in b.items() if k != 'measurements'})
        for name, path in [('primary', 'outputs/official39-cost-benchmark-v1/result.json'),
                           ('replay', 'outputs/official39-cost-benchmark-replay-v1/result.json')]:
            # atomic_json used indent=2 with no trailing newline. Reconstruct
            # the original bytes, not a JS or sorted-key reserialization.
            raw = json.dumps(self.cost[name], indent=2, allow_nan=False).encode()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), self.cost['source_files_sha256'][path])

    def test_all_nine_models_match_final_weight_identity(self):
        evidence = read(final.EVIDENCE)
        refs = {r['candidate']['id']: r for name, r in evidence['documents'].items()
                if name.startswith(final.FINAL + 'case-') and r['candidate']['arm'] == 'full39'}
        receipt = self.cost['primary']
        self.assertEqual(set(receipt['models']), set(refs))
        self.assertEqual(len(refs), 9)
        self.assertEqual(set(receipt['prediction_fingerprints']), set(refs))
        for name, model in receipt['models'].items():
            self.assertEqual(model['model_state_sha256'], refs[name]['model_state_sha256'])
            self.assertEqual(model['artifact'], refs[name]['candidate']['artifact'])
            self.assertEqual(model['num_parameters'], 5096)
            self.assertEqual(model['parameter_bytes'], 20384)

    def test_all_timing_samples_and_summaries(self):
        for run in ('primary', 'replay'):
            r = self.cost[run]
            self.assertEqual(set(r['measurements']), set(r['models']))
            for row in r['measurements'].values():
                self.assertEqual(set(row), {'1', '512'})
                for size, summary in row.items():
                    values = summary['repeat_mean_call_ms']
                    self.assertEqual(len(values), self.plan['repeats'])
                    self.assertTrue(all(math.isfinite(x) and x > 0 for x in values))
                    median = statistics.median(values)
                    self.assertEqual(summary['median_mean_call_ms'], median)
                    self.assertEqual(summary['min_mean_call_ms'], min(values))
                    self.assertEqual(summary['max_mean_call_ms'], max(values))
                    self.assertEqual(summary['amortized_ms_per_flow'], median / int(size))
                    self.assertEqual(summary['flows_per_second'], 1000 * int(size) / median)
                    self.assertFalse(summary['tail_request_latency_measured'])

    def test_scope_and_exact_forward_work(self):
        p = self.plan
        for run in ('primary', 'replay'):
            r = self.cost[run]
            self.assertEqual(r['runtime']['device'], 'cpu')
            self.assertEqual(r['runtime']['threads'], p['threads'])
            self.assertEqual(r['runtime']['interop_threads'], p['interop_threads'])
            for flag in ('validation_rows_opened', 'test_rows_opened', 'test_scored',
                         'training_labels_opened', 'quality_scoring_performed',
                         'cloud_used', 'edge_hardware_measured'):
                self.assertFalse(r[flag])
            self.assertTrue(r['weights_unchanged'])
            n = len(r['models'])
            self.assertEqual(r['work'], {
                'warmup_forward_examples': n*p['warmup_calls_per_case']*sum(p['batch_sizes']),
                'timed_forward_examples': n*p['repeats']*sum(s*p['calls_per_repeat'][str(s)] for s in p['batch_sizes']),
                'timed_calls': n*p['repeats']*sum(p['calls_per_repeat'].values()),
                'fingerprint_forward_examples': 2*512*n,
                'optimizer_steps': 0,
                'backward_batches': 0,
            })


if __name__ == '__main__':
    unittest.main()
