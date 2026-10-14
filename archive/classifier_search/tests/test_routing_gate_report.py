"""Published aggregate checks need no private flows, checkpoints or GPU."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import official39_routing_gate_report as report

EVIDENCE = report.ROOT/'reports/risk_aware_routing/router-results.json'


class RoutingGateReportTests(unittest.TestCase):
    def setUp(self):
        self.record = report.frozen.read(EVIDENCE)

    def test_published_receipt_and_stop_rule_audit(self):
        result = report.audit(self.record)
        self.assertEqual(result['gates'], 6)
        self.assertEqual(result['audit_rows'], 408027)
        self.assertFalse(result['test_opened'])
        self.assertTrue(all(not s['all_seeds_pass'] for lane in self.record['summary'].values()
                            for s in lane['continuation_screen']))

    def test_scope_and_model_pairs_cannot_change(self):
        for field in report.runner.FALSE_FLAGS:
            with self.assertRaisesRegex(ValueError, 'scope'):
                report.audit(dict(self.record, **{field: True}))
        bad = copy.deepcopy(self.record)
        bad['prepared_pairs'].pop()
        with self.assertRaisesRegex(ValueError, 'pairs'):
            report.audit(bad)

    def test_audit_cannot_change_frozen_threshold(self):
        self.record['pairs'][0]['policies'][0]['threshold'] = .123
        with self.assertRaisesRegex(ValueError, 'after calibration'):
            report.audit(self.record)

    def test_false_alert_or_budget_breach_cannot_be_hidden(self):
        for field in ('budget_respected', 'benign_guard_respected'):
            bad = copy.deepcopy(self.record)
            r = bad['pairs'][0]['policies'][0]
            r[field] = not r[field]
            with self.assertRaisesRegex(ValueError, 'constraint'):
                report.audit(bad)

    def test_counts_and_recall_cannot_be_changed(self):
        bad = copy.deepcopy(self.record)
        bad['pairs'][0]['policies'][0]['corrected_rows'] += 1
        with self.assertRaisesRegex(ValueError, 'totals'):
            report.audit(bad)
        bad = copy.deepcopy(self.record)
        bad['pairs'][0]['policies'][0]['metrics']['per_class']['Web-based']['recall'] = 1.
        with self.assertRaisesRegex(ValueError, 'metrics'):
            report.audit(bad)

    def test_summary_cannot_claim_success(self):
        self.record['summary']['iid']['continuation_screen'][0]['all_seeds_pass'] = True
        with self.assertRaisesRegex(ValueError, 'Summary'):
            report.audit(self.record)

    def test_publish_never_overwrites_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'existing.json'
            p.write_text('keep')
            with patch.object(report.frozen, 'read', side_effect=AssertionError('Must not read')):
                with self.assertRaisesRegex(ValueError, 'Preserve'):
                    report.publish('absent', p)
            self.assertEqual(p.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
