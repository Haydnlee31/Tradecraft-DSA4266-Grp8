"""Offline closeout checks: no training, dataset arrays or checkpoints required."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import official39_closeout as closeout
from src.data.label_map import BENIGN, WEB_BASED, BRUTE_FORCE


class OfficialCloseoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = closeout.read(closeout.EVIDENCE)
        cls.plan = closeout.read(closeout.FINAL_PLAN)

    def test_complete_published_evidence(self):
        result = closeout.audit_evidence(self.evidence)
        self.assertEqual(result['endpoints'], 27)
        self.assertEqual(result['prospective_runs'], 18)
        self.assertFalse(result['test_evaluated'])
        new = [r for r in self.evidence['records'] if r['seed'] != 7]
        self.assertEqual(sum(r['examples_processed'] for r in new), 720000000)
        self.assertEqual(sum(r['optimizer_steps'] for r in new), 1409280)

    def test_all_recall_classes_and_distinct_error_categories(self):
        for record in self.evidence['records']:
            self.assertEqual(set(record['metrics']['per_class']), set(closeout.CLASSES))
            for name, counts in record['errors'].items():
                total = (counts['correct'] + counts['false_alerts'] if name == BENIGN else
                         counts['correct_category'] + counts['predicted_benign'] + counts['wrong_attack_category'])
                self.assertEqual(total, counts['support'])

    def test_prospective_results_not_hidden_by_seed7(self):
        pair = self.evidence['paired_analysis']['dirichlet/number_total_masked-minus-full39']
        self.assertGreater(pair['macro_f1']['values'][0], 0)
        self.assertLess(pair['prospective_seeds_17_27']['macro_f1']['mean'], 0)
        self.assertEqual(pair['prospective_seeds_17_27']['macro_f1']['values'], pair['macro_f1']['values'][1:])
        for record in self.evidence['records']:
            if record['group'] == 'dirichlet':
                for name in (WEB_BASED, BRUTE_FORCE):
                    self.assertEqual(record['metrics']['per_class'][name]['recall'], 0.)

    def test_missing_duplicate_or_substituted_endpoint_rejected(self):
        for change in ('missing', 'duplicate', 'best', 'budget', 'artifact', 'seed_role'):
            with self.subTest(change=change):
                data = copy.deepcopy(self.evidence)
                if change == 'missing':
                    data['records'].pop()
                elif change == 'duplicate':
                    data['records'][-1] = copy.deepcopy(data['records'][0])
                elif change == 'best':
                    data['records'][0]['endpoint'] = 16
                elif change == 'budget':
                    data['records'][0]['examples_processed'] += 1
                elif change == 'artifact':
                    data['records'][0]['artifact']['checkpoint_member'] = '../best.pt'
                else:
                    data['records'][0]['seed_role'] = 'prospective_confirmation'
                with self.assertRaises(ValueError):
                    closeout.audit_evidence(data)

    def test_counts_metrics_and_paired_statistics_checked(self):
        for change in ('counts', 'macro_f1', 'recall', 'errors', 'summary', 'paired', 'fresh_seeds'):
            with self.subTest(change=change):
                data = copy.deepcopy(self.evidence)
                record = data['records'][0]
                if change == 'counts':
                    record['metrics']['confusion_matrix'][0][0] += 1
                elif change == 'macro_f1':
                    record['metrics']['macro_f1'] += .01
                elif change == 'recall':
                    record['metrics']['per_class'][WEB_BASED]['recall'] = 1.
                elif change == 'errors':
                    record['errors'][BRUTE_FORCE]['predicted_benign'] += 1
                elif change == 'summary':
                    data['summary']['groups']['light']['full39']['macro_f1']['mean'] += .01
                elif change == 'paired':
                    data['paired_analysis']['light/number_masked-minus-full39']['macro_f1']['mean'] += .01
                else:
                    data['paired_analysis']['light/number_masked-minus-full39']['prospective_seeds_17_27']['macro_f1']['mean'] = 1.
                with self.assertRaises(ValueError):
                    closeout.audit_evidence(data)

    def test_scope_and_provenance_cannot_silently_change(self):
        for flag in ('dataset_opened', 'checkpoint_opened', 'model_trained', 'test_evaluated',
                     'deployment_authorized', 'model_promoted'):
            data = copy.deepcopy(self.evidence)
            data[flag] = True
            with self.assertRaises(ValueError):
                closeout.audit_evidence(data)
        data = copy.deepcopy(self.evidence)
        data['source_evidence']['confirmation']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            closeout.audit_evidence(data)

    def test_final_protocol_is_closed_and_complete(self):
        result = closeout.audit_final_plan(self.plan)
        self.assertEqual(result['primary_references'], 9)
        self.assertEqual(result['secondary_sensitivity'], 18)
        self.assertFalse(result['test_access_authorized'])
        self.assertEqual(self.plan['data_identity']['test_rows_before_overlap_filter'], 2060864)
        self.assertIsNone(self.plan['data_identity']['shared_test_panel_rows'])

    def test_final_protocol_rejects_cherry_picking_and_threshold_tuning(self):
        for change in ('candidate', 'checkpoint', 'seed', 'threshold', 'population', 'gate', 'action'):
            with self.subTest(change=change):
                plan = copy.deepcopy(self.plan)
                if change == 'candidate':
                    plan['candidates'].pop()
                elif change == 'checkpoint':
                    plan['candidates'][0]['artifact']['checkpoint_member'] = 'best.pt'
                elif change == 'seed':
                    plan['training_seeds'] = [7]
                elif change == 'threshold':
                    plan['policy']['threshold_tuning_allowed'] = True
                elif change == 'population':
                    plan['policy']['primary_population'] = 'best_scoring_subset'
                elif change == 'gate':
                    plan['approval_gates'].pop()
                else:
                    plan['test_access_authorized'] = True
                with self.assertRaises(ValueError):
                    closeout.audit_final_plan(plan)

    def test_final_protocol_rejects_changed_data_or_fabricated_panel(self):
        for change in ('scaler_sha256', 'parent_shards_manifest_sha256', 'features',
                       'test_class_counts_before_overlap_filter', 'shared_test_panel_rows'):
            plan = copy.deepcopy(self.plan)
            data = plan['data_identity']
            if change.endswith('sha256'):
                data[change] = '0' * 64
            elif change == 'features':
                data[change].reverse()
            elif change == 'shared_test_panel_rows':
                data[change] = 1
            else:
                data[change][BENIGN] += 1
            with self.assertRaises(ValueError):
                closeout.audit_final_plan(plan)

    def test_operational_rules_cannot_be_invented_during_research_closeout(self):
        for key, value in [('automatic_allow_or_block', True),
                           ('numerical_deployment_thresholds', {'false_alert_rate': .01}),
                           ('existing_legacy46_decision_defaults_changed', True)]:
            plan = copy.deepcopy(self.plan)
            plan['operating_requirements'][key] = value
            with self.assertRaises(ValueError):
                closeout.audit_final_plan(plan)

    def test_final_plan_pins_the_published_evidence_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.json'
            # Even whitespace changes the checksum. Publication is immutable;
            # an intentional revision needs a new reviewed protocol identity.
            path.write_bytes(closeout.EVIDENCE.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'candidate evidence changed'):
                closeout.audit_final_plan(self.plan, path)

    def test_offline_cli_does_not_import_training_dependencies(self):
        code = (
            'import sys; from scripts import official39_closeout as c; '
            'c.audit_evidence(c.read(c.EVIDENCE)); c.audit_final_plan(c.read(c.FINAL_PLAN)); '
            'assert not ({"torch", "numpy", "polars", "pyarrow"} & set(sys.modules))'
        )
        subprocess.run([sys.executable, '-c', code], cwd=closeout.ROOT, check=True,
                       capture_output=True, text=True)

    def test_cli_has_no_evaluation_action(self):
        result = subprocess.run([sys.executable, str(closeout.ROOT / 'scripts/official39_closeout.py'), 'evaluate'],
                                cwd=closeout.ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid choice', result.stderr)

    def test_build_refuses_to_overwrite_evidence(self):
        result = subprocess.run([sys.executable, str(closeout.ROOT / 'scripts/official39_closeout.py'), 'build',
                                 '--archive-dir', '.', '--output', str(closeout.EVIDENCE)],
                                cwd=closeout.ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Preserve existing evidence', result.stderr)

    def test_build_writes_portable_lf_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rebuilt.json'
            args = ['closeout', 'build', '--archive-dir', directory, '--output', str(path)]
            # Mock only archive acquisition. The real audit and writer still
            # run, so Windows CI checks byte stability without private backups.
            with patch.object(closeout, 'export', return_value=copy.deepcopy(self.evidence)), \
                    patch.object(sys, 'argv', args), redirect_stdout(io.StringIO()):
                closeout.main()
            expected = (json.dumps(self.evidence, indent=2, allow_nan=False) + '\n').encode('utf-8')
            self.assertEqual(path.read_bytes(), expected)
            self.assertNotIn(b'\r\n', path.read_bytes())


if __name__ == '__main__':
    unittest.main()
