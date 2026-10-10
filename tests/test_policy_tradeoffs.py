"""Frozen-policy sensitivity must not turn referrals into detections."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import official39_policy_tradeoffs as runner
from tests.test_routing_gate import fixture

gate, prior = runner.gate, runner.prior


class PolicyTradeoffTests(unittest.TestCase):
    def test_zero_slack_and_quota_exactly_match_historical_grid(self):
        z, y, base, expert = fixture()
        for scores in (z[:, 0], np.round(z[:, 0], 1), np.ones(len(y))):
            grid = runner.candidate_grid(scores, y, base, expert)
            old = gate.calibrate(scores, y, base, expert)
            for policy in old['policies']:
                b = policy['budget_percent']
                self.assertEqual(runner.choose(grid, b, 0), policy['guarded'])
                self.assertEqual(runner.review_choice(grid, b), {k: policy['quota'][k] for k in ('calls', 'threshold')})

    def test_allowance_uses_benign_denominator_and_integer_floor(self):
        # 399 benign rows permit one extra false alert at +50/10,000,
        # not 50 alerts based on the entire 10,000-flow cohort.
        grid = dict(rows=10000, benign_rows=399, base_false_alerts=2, candidates=[
            dict(calls=0, threshold=None, macro_f1=.2, false_alerts=2),
            dict(calls=10, threshold=.8, macro_f1=.3, false_alerts=3),
            dict(calls=20, threshold=.7, macro_f1=.4, false_alerts=4)])
        self.assertEqual(runner.choose(grid, 1, 0)['calls'], 0)
        self.assertEqual(runner.choose(grid, 1, 50)['calls'], 10)
        self.assertEqual(runner.choose(grid, 1, 100)['calls'], 20)

    def test_constraints_and_nested_calibration_objectives(self):
        z, y, base, expert = fixture()
        scores = z[:, 0]
        grid = runner.candidate_grid(scores, y, base, expert)
        for b in (1, 5, 10):
            values = []
            for slack in (0, 50, 100, 200):
                choice = runner.choose(grid, b, slack)
                r = gate.evaluate_policy(y, base, expert, scores, choice['threshold'], b)
                self.assertEqual(choice['calls'], r['calls'])
                self.assertEqual(choice['macro_f1'], r['macro_f1'])
                self.assertTrue(r['budget_respected'])
                self.assertLessEqual(r['false_alerts'], grid['base_false_alerts']+grid['benign_rows']*slack//10000)
                values.append(choice['macro_f1'])
            self.assertEqual(values, sorted(values))

    def test_equal_f1_prefers_fewer_calls_and_all_harm_chooses_zero(self):
        _, y, _, _ = fixture()
        grid = runner.candidate_grid(np.arange(len(y), dtype=float), y, y, y)
        self.assertEqual(runner.choose(grid, 10, 200)['calls'], 0)
        grid = runner.candidate_grid(np.arange(len(y), dtype=float), y, y, (y+1) % 8)
        self.assertEqual(runner.choose(grid, 10, 200)['calls'], 0)

    def test_review_threshold_selection_is_label_blind(self):
        z, y, base, expert = fixture()
        first = runner.candidate_grid(z[:, 0], y, base, expert)
        second = runner.candidate_grid(z[:, 0], (y+1) % 8, (base+2) % 8, (expert+3) % 8)
        for b in (1, 5, 10):
            self.assertEqual(runner.review_choice(first, b), runner.review_choice(second, b))

    def test_candidate_grid_json_round_trip_is_exact(self):
        z, y, base, expert = fixture()
        grid = runner.candidate_grid(z[:, 0], y, base, expert)
        reread = json.loads(json.dumps(grid, allow_nan=False))
        self.assertEqual(grid, reread)
        self.assertEqual(runner.review_choice(grid, 10), runner.review_choice(reread, 10))

    def test_review_does_not_fix_labels_or_assume_reviewer_success(self):
        y = np.tile(np.arange(8), 100)
        base = np.zeros_like(y)
        original = base.copy()
        r = runner.review_counts(y, base, np.ones(len(y)), .5, 10)
        np.testing.assert_array_equal(base, original)
        self.assertEqual(r['calls'], 800)
        self.assertFalse(r['budget_respected'])  # No audit-time clipping.
        self.assertEqual(r['base_errors_referred'], 700)
        self.assertEqual(r['benign_referrals'], 100)
        self.assertEqual(r['reviews_per_10000_flows'], 10000)
        self.assertFalse(r['classification_labels_changed'])
        self.assertFalse(r['human_review_performed'])
        self.assertIsNone(r['successful_reviews'])
        for name in gate.CLASSES:
            c = r['per_class'][name]
            self.assertEqual(c['referred'], 100)
            self.assertEqual(c['review_coverage'], 1)
            self.assertEqual(c['attacks_predicted_benign_referred'], 0 if name == gate.BENIGN else 100)

    def test_no_review_retains_missed_attack_workload(self):
        y = np.tile(np.arange(8), 100)
        r = runner.review_counts(y, np.zeros_like(y), np.ones(len(y))*1e10, None, 1)
        self.assertEqual(r['calls'], 0)
        self.assertEqual(r['per_class']['Web-based']['attacks_predicted_benign_not_referred'], 100)
        self.assertEqual(r['per_class']['Web-based']['base_errors_referred'], 0)

    def test_whole_review_score_ties_are_not_split(self):
        _, y, base, expert = fixture()
        grid = runner.candidate_grid(np.ones(len(y)), y, base, expert)
        self.assertEqual(runner.review_choice(grid, 10), dict(calls=0, threshold=None))

    def test_invalid_scores_budgets_and_missing_benign_rejected(self):
        z, y, base, expert = fixture()
        with self.assertRaises(ValueError):
            runner.candidate_grid(z[:, 0]*np.nan, y, base, expert)
        with self.assertRaisesRegex(ValueError, 'benign'):
            runner.candidate_grid(z[:, 0], np.ones_like(y), base, expert)
        with self.assertRaises(ValueError):
            runner.review_counts(y, base, z[:, 0], np.nan, 10)
        grid = runner.candidate_grid(z[:, 0], y, base, expert)
        for b, e in [(50, 0), (10, 300)]:
            with self.assertRaisesRegex(ValueError, 'budget'):
                runner.choose(grid, b, e)

    def test_plan_pins_scope_and_previous_evidence(self):
        p = runner.frozen.read(runner.PLAN)
        runner.validate_plan(p)
        for key, value in [('device', 'cuda'), ('seeds', [7]), ('extra_false_alert_basis_points', [0, 500]),
                           *[(k, True) for k in runner.FLAGS]]:
            with self.assertRaisesRegex(ValueError, 'scope'):
                runner.validate_plan(dict(p, **{key: value}))
        with self.assertRaisesRegex(ValueError, 'evidence'):
            runner.validate_plan(dict(p, router_results_sha256='wrong'))

    def test_existing_run_rejected_before_reads(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.frozen, 'read', side_effect=AssertionError('No read')):
            with self.assertRaisesRegex(ValueError, 'Preserve'):
                runner.run('absent', 'absent', 'absent', tmp)

    def test_synthetic_frozen_pipeline_bridges_without_fitting(self):
        z, y, base, expert = fixture()
        split = np.repeat(np.array([0, 1, 2], np.uint8), [4800, 1600, 1600])
        rows = np.arange(len(y))
        p = prior.frozen.read(prior.PLAN)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root/'seed7-iid-probabilities.npy', z[:, 39:])
            for lane, pred in [('iid', base), ('balanced_panel', expert), ('light', base)]:
                np.save(root/f'seed7-{lane}-predictions.npy', pred)
            old = prior.prepare_pair(7, 'iid', z[:, :39], y, rows, split, np.arange(4800), root, p)
            old_audit = prior.audit_pair(old, z[:, :39], y, rows, split, root, p)
            before = (root/old['gate']).read_bytes()
            with patch.object(gate, 'fit_gate', side_effect=AssertionError('No refit')):
                prepared = runner.prepare_pair(old, z[:, :39], y, rows, split, root, p)
                # Altering audit labels/features must not affect calibration.
                changed_y, changed_x = y.copy(), z[:, :39].copy()
                changed_y[split == 2] = (changed_y[split == 2]+1) % 8
                changed_x[split == 2] *= -10
                self.assertEqual(prepared, runner.prepare_pair(old, changed_x, changed_y, rows, split, root, p))
                result = runner.score_pair(prepared, old, z[:, :39], y, rows, split, root, p, old_audit)
            self.assertEqual(len(result['automatic']), 60)
            self.assertEqual(len(result['reviews']), 9)
            self.assertTrue(result['zero_slack_audit_exact'])
            self.assertEqual(before, (root/old['gate']).read_bytes())

    def test_summary_rejects_missing_or_duplicate_seed_pairs(self):
        with self.assertRaisesRegex(ValueError, 'paired'):
            runner.summarize([])


class PublishedTradeoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = runner.ROOT/'reports/risk_aware_routing/tradeoff-results.json'
        if not path.exists():
            raise unittest.SkipTest('Published experiment not generated yet')
        cls.record = runner.frozen.read(path)

    def test_aggregate_audit_requires_no_private_arrays(self):
        with patch.object(np, 'load', side_effect=AssertionError('No flow data')):
            self.assertEqual(runner.audit(self.record)['automatic_scenarios'], 360)

    def test_tampered_metrics_and_scope_rejected(self):
        for flag in (*runner.FLAGS, 'operating_point_selected'):
            with self.assertRaises(ValueError):
                runner.audit(dict(self.record, **{flag: True}))
        for section, field, value in [('automatic', 'extra_false_alerts', -90000),
                                       ('reviews', 'successful_reviews', 100),
                                       ('reviews', 'base_errors_referred', 1000000)]:
            bad = copy.deepcopy(self.record)
            bad['pairs'][0][section][0][field] = value
            with self.assertRaises(ValueError):
                runner.audit(bad)

    def test_missing_scenario_or_changed_threshold_rejected(self):
        bad = copy.deepcopy(self.record)
        bad['pairs'][0]['automatic'].pop()
        with self.assertRaises(ValueError):
            runner.audit(bad)
        bad = copy.deepcopy(self.record)
        bad['pairs'][0]['reviews'][0]['threshold'] = 999
        with self.assertRaises(ValueError):
            runner.audit(bad)


if __name__ == '__main__':
    unittest.main()
