"""Final-test closeout stays JSON-only, complete and separate from legacy46.

The published evidence is deliberately small enough for CI. These tests do not
need the private backup, original traffic, saved weights, NumPy or a GPU.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import official39_final_report as report
from scripts import official39_closeout as c
from src.eval.official_evidence import close_tree, read_json


class FinalReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = c.read(report.EVIDENCE)

    def test_published_complete_archive_audit(self):
        result = report.audit(self.evidence)
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['candidates'], 27)
        self.assertEqual(result['metric_views_recomputed'], 108)
        self.assertEqual((result['primary_rows'], result['original_rows']), (2040729, 2060864))
        self.assertTrue(result['source_test_evaluated'])
        self.assertFalse(result['inference_performed'])
        self.assertFalse(result['deployment_authorized'])

    def test_cli_without_site_packages_or_private_data(self):
        # -S disables site packages: accidentally importing Torch/NumPy breaks
        # this test, rather than making teammates install training packages.
        result = subprocess.run([sys.executable, '-S', 'scripts/official39_final_report.py', 'audit'],
                                cwd=report.ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'verified')

    def test_missing_extra_or_wrong_archive_documents_rejected(self):
        for mutation in ('missing', 'extra', 'archive'):
            with self.subTest(mutation=mutation):
                evidence = copy.deepcopy(self.evidence)
                if mutation == 'missing':
                    del evidence['documents'][report.FINAL + 'case-00.json']
                elif mutation == 'extra':
                    evidence['documents']['../../untrusted.py'] = {}
                else:
                    evidence['archive']['sha256'] = '0' * 64
                with self.assertRaises(ValueError):
                    report.audit(evidence)

    def test_changed_case_or_self_consistent_rewritten_receipt_rejected(self):
        for rewrite_receipt in (False, True):
            with self.subTest(rewrite_receipt=rewrite_receipt):
                evidence = copy.deepcopy(self.evidence)
                d = evidence['documents']
                name = 'case-00.json'
                d[report.FINAL + name]['panel']['metrics']['macro_f1'] += .1
                if rewrite_receipt:
                    # Even replacing its case hash cannot replace the reviewed
                    # top-level receipt whose hash is anchored outside the file.
                    d[report.FINAL + 'checks.json']['cases'][name] = report.json_digest(d[report.FINAL + name])
                with self.assertRaises(ValueError):
                    report.audit(evidence)

    def test_changed_gate_session_journal_or_preparation_rejected(self):
        changes = [(report.FINAL + 'session.json', 'explicit_final_test_opt_in', False),
                   (report.FINAL + 'session.json', 'runtime', {}),
                   (report.FINAL + 'progress.json', 'cases', {}),
                   (report.REPLAY + 'checks.json', 'test_scoring_ready', False),
                   (report.PREP, 'rows', 1),
                   (report.PLAN, 'test_access_authorized', True)]
        for name, key, value in changes:
            with self.subTest(name=name, key=key):
                evidence = copy.deepcopy(self.evidence)
                evidence['documents'][name][key] = value
                with self.assertRaises(ValueError):
                    report.audit(evidence)

    def validate(self, record):
        prep = self.evidence['documents'][report.PREP]
        original = self.evidence['documents'][report.FINAL + 'case-00.json']
        supports = dict(all=prep['class_counts'], panel=prep['panel']['retained_class_counts'])
        report.validate_record(record, original['candidate'], original['session_sha256'], supports, True)

    def test_confusion_metric_error_and_forward_guards(self):
        original = self.evidence['documents'][report.FINAL + 'case-00.json']
        self.validate(original)
        for mutation in ('metric', 'error', 'counts', 'forward', 'weights', 'scope'):
            with self.subTest(mutation=mutation):
                r = copy.deepcopy(original)
                if mutation == 'metric':
                    r['panel']['metrics']['macro_f1'] = float('nan')
                elif mutation == 'error':
                    r['panel']['errors']['Web-based']['predicted_benign'] += 1
                elif mutation == 'counts':
                    r['panel']['metrics']['confusion_matrix'][0][0] -= 1
                elif mutation == 'forward':
                    r['forward_batches'] -= 1
                elif mutation == 'weights':
                    r['weights_unchanged'] = False
                else:
                    r['test_evaluated'] = False
                with self.assertRaises(ValueError):
                    self.validate(r)

    def test_summary_keeps_all_seeds_and_both_populations(self):
        d = self.evidence['documents']
        records = [d[report.FINAL + f'case-{i:02d}.json'] for i in range(27)]
        close_tree(report.summarize(records), d[report.FINAL + 'checks.json']['summary'])
        for altered in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError):
                report.summarize(altered)

    def test_decision_has_no_operational_action_or_new_gate(self):
        decision = report.research_decision(self.evidence)
        close_tree(decision, c.read(c.REPORTS / 'final-test-decision.json'))
        for name in ('automatic_allow_or_block', 'deployment_authorized', 'model_promoted',
                     'legacy46_defaults_changed', 'post_test_retuning_allowed'):
            self.assertIs(decision[name], False)
        self.assertIsNone(decision['numerical_deployment_thresholds'])
        self.assertEqual(set(decision['research_references']), set(c.LANES))
        for lane, reference in decision['research_references'].items():
            self.assertEqual(set(reference['metrics']['per_class']), set(c.CLASSES))
            self.assertEqual(set(reference['errors_by_seed']), {'7', '17', '27'})
        self.assertEqual(decision['research_references']['dirichlet']['zero_category_recall_in_all_seeds'],
                         ['Web-based', 'Brute Force'])
        self.assertIn('not final-test explanations', decision['explanation_context']['population'])
        self.assertIn('not exact conditional SHAP', decision['explanation_context']['method'])

    def test_generated_tables_and_all_class_supports_are_current(self):
        text = report.tables(self.evidence)
        self.assertEqual(text, (c.REPORTS / 'final-test-tables.md').read_text(encoding='utf-8'))
        self.assertIn('2,040,729', text)
        self.assertIn('2,060,864', text)
        self.assertIn('Secondary original test population', text)
        self.assertEqual(text.count('### Class recall for'), 6)
        for name in c.CLASSES:
            self.assertIn('| ' + name + ' |', text)

    def test_wrong_archive_rejected_before_tar_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'not-an-archive.tgz'
            path.write_bytes(b'not a reviewed archive')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                report.build(path)

    def test_write_is_exclusive_and_utf8_lf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nested' / 'receipt.json'
            value = dict(scope='test', count=3)
            report.write_new(path, report.encode(value))
            self.assertEqual(path.read_bytes(), report.encode(value))
            with self.assertRaises(FileExistsError):
                report.write_new(path, b'replacement')
            self.assertEqual(read_json(path.read_bytes()), value)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self):
        for raw in (b'{"a": 1, "a": 2}', b'{"a": NaN}', b'{"a": Infinity}'):
            with self.assertRaises(ValueError):
                read_json(raw)


if __name__ == '__main__':
    unittest.main()
