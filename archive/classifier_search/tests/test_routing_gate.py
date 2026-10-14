"""Portable fixtures exercise routing without real flows or model artifacts."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.eval import routing_gate as gate
from scripts import official39_routing_gate as runner


def fixture(n=8000):
    rng = np.random.default_rng(70)
    x = rng.normal(size=(n, 39)).astype(np.float32)
    y = (np.arange(n) % 8).astype(np.int64)
    base, expert = y.copy(), y.copy()
    base[x[:, 0] > .4] = (base[x[:, 0] > .4]+1) % 8
    expert[x[:, 0] < -.4] = (expert[x[:, 0] < -.4]+1) % 8
    p = np.full((n, 8), .1, dtype=np.float32)
    p[np.arange(n), base] = .3
    return gate.features(x, p), y, base, expert


class RoutingGateTests(unittest.TestCase):
    def test_grouping_is_label_blind_and_preserves_duplicates_and_signed_zero(self):
        x = np.zeros((12, 39), np.float32)
        x[1::2, 0] = -0.
        keys = gate.group_keys(x)
        self.assertEqual(len(np.unique(keys)), 1)
        self.assertEqual(len(np.unique(gate.partitions(keys))), 1)
        self.assertTrue(np.array_equal(keys, np.concatenate([gate.group_keys(x[:3]), gate.group_keys(x[3:])])) )
        self.assertFalse(np.array_equal(keys, gate.group_keys(x, seed=1)))

    def test_grouping_rejects_nonfinite_and_wrong_schema(self):
        for x in (np.ones((2, 38)), np.full((2, 39), np.nan), np.full((2, 39), np.inf)):
            with self.assertRaises(ValueError):
                gate.group_keys(x)

    def test_partition_buckets_and_coverage(self):
        keys = np.arange(100, dtype=np.uint64)
        split = gate.partitions(keys)
        self.assertEqual(np.bincount(split).tolist(), [60, 20, 20])
        with self.assertRaises(ValueError):
            gate.partitions(keys.astype(float))

    def test_fit_sampler_capped_disjoint_and_unchanged_by_audit_labels(self):
        y = np.tile(np.arange(8, dtype=np.int64), 1000)
        split = np.tile(np.array([0, 1, 2], np.uint8), 2667)[:len(y)]
        chosen = gate.fit_rows(y, split, cap=13)
        self.assertEqual(len(chosen), 104)
        self.assertTrue(np.all(split[chosen] == 0))
        self.assertTrue(np.all(chosen[1:] > chosen[:-1]))
        changed = y.copy()
        changed[split != 0] = (changed[split != 0]+1) % 8
        self.assertTrue(np.array_equal(chosen, gate.fit_rows(changed, split, cap=13)))
        with self.assertRaisesRegex(ValueError, 'Missing fit'):
            gate.fit_rows(np.zeros(8, dtype=np.int64), np.zeros(8, np.uint8))

    def test_inputs_only_flow_and_base_probability(self):
        x, p = np.zeros((3, 39)), np.full((3, 8), .125)
        self.assertEqual(gate.features(x, p).shape, (3, 47))
        self.assertEqual(gate.features(x, p).dtype, np.float32)
        for bad in (p*0, p[:, :7], p*np.nan, p*9):
            with self.assertRaises(ValueError):
                gate.features(x, bad)

    def test_small_tree_json_exact_and_deterministic(self):
        z, y, base, expert = fixture()
        tree = gate.fit_gate(z, y, base, expert)
        self.assertEqual(tree, gate.fit_gate(z, y, base, expert))
        self.assertLessEqual(tree['nodes'], 63)
        self.assertEqual(tree['fit_rows'], len(y))
        self.assertEqual(sum(tree['target_counts']), len(y))
        score = gate.gate_score(tree, z)
        self.assertTrue(np.all((score >= -1) & (score <= 1)))
        # These regions have deliberately pure repair/harm outcomes.
        self.assertGreater(score[z[:, 0] > .8].mean(), .9)
        self.assertLess(score[z[:, 0] < -.8].mean(), -.9)

    def test_tree_missing_outcome_classes_supported(self):
        z, y, _, _ = fixture()
        tree = gate.fit_gate(z, y, y, y)
        self.assertEqual(tree['classes'], [0])
        self.assertTrue(np.all(gate.gate_score(tree, z) == 0))

    def test_serialized_leaf_probabilities_are_not_renormalized(self):
        z, y, _, _ = fixture()
        tree = gate.fit_gate(z, y, y, y)
        # A tiny sum discrepancy is legal floating-point output, not a reason
        # to silently alter a saved routing score during inference.
        tree['value'][0][0] = np.nextafter(1., 0.)
        p = gate.tree_probabilities(tree, z)
        self.assertEqual(p[0, 0], np.nextafter(1., 0.))

    def test_tree_input_and_edges_fail_closed(self):
        z, y, base, expert = fixture()
        tree = gate.fit_gate(z, y, base, expert)
        for key, val in [('inputs', 48), ('nodes', 100), ('depth', 8)]:
            bad = dict(tree, **{key: val})
            with self.assertRaises(ValueError):
                gate.gate_score(bad, z)
        bad = copy.deepcopy(tree)
        bad['children_left'][0] = 0
        with self.assertRaisesRegex(ValueError, 'edges'):
            gate.gate_score(bad, z)
        with self.assertRaises(ValueError):
            gate.gate_score(tree, z[:, :-1])

    def test_calibration_guard_and_all_budgets(self):
        z, y, base, expert = fixture()
        scores = z[:, 0].astype(float)
        result = gate.calibrate(scores, y, base, expert)
        for policy in result['policies']:
            for kind in ('guarded', 'quota'):
                chosen = policy[kind]
                out = gate.evaluate_policy(y, base, expert, scores, chosen['threshold'], policy['budget_percent'])
                self.assertEqual(out['calls'], chosen['calls'])
                self.assertEqual(out['false_alerts'], chosen['false_alerts'])
                self.assertEqual(out['macro_f1'], chosen['macro_f1'])
                self.assertTrue(out['budget_respected'])
            self.assertTrue(policy['guarded']['false_alerts'] <= result['base_false_alerts'])
            self.assertGreaterEqual(policy['guarded']['macro_f1'], result['base_macro_f1'])

    def test_all_harm_falls_back_to_zero_calls(self):
        y = np.tile(np.arange(8, dtype=np.int64), 100)
        base, expert = y.copy(), (y+1) % 8
        result = gate.calibrate(np.arange(len(y), dtype=float), y, base, expert)
        self.assertTrue(all(p['guarded']['threshold'] is None and p['guarded']['calls'] == 0 for p in result['policies']))

    def test_whole_ties_never_split_by_row_id(self):
        y = np.tile(np.arange(8, dtype=np.int64), 100)
        base, expert = (y+1) % 8, y.copy()
        r = gate.calibrate(np.ones(len(y)), y, base, expert)
        self.assertTrue(all(p['quota']['calls'] == 0 for p in r['policies']))
        scores = np.zeros(len(y))
        scores[:40] = 1
        r = gate.calibrate(scores, y, base, expert)
        self.assertEqual([p['quota']['calls'] for p in r['policies']], [0, 40, 40])

    def test_no_calls_remains_no_calls_under_distribution_shift(self):
        self.assertFalse(gate.decisions(np.array([1e50, 1e60]), None).any())

    def test_audit_reports_breaches_instead_of_reclipping_or_retuning(self):
        y = np.tile(np.arange(8, dtype=np.int64), 100)
        out = gate.evaluate_policy(y, y.copy(), (y+1) % 8, np.ones(len(y)), .5, 1)
        self.assertEqual(out['calls'], len(y))
        self.assertFalse(out['budget_respected'])
        self.assertFalse(out['benign_guard_respected'])
        self.assertEqual(out['damaged_rows'], len(y))
        self.assertEqual(out['corrected_rows'], 0)

    def test_audit_selection_independent_of_true_labels(self):
        z, y, base, expert = fixture()
        one = gate.evaluate_policy(y, base, expert, z[:, 0], .8, 10)
        two = gate.evaluate_policy((y+1) % 8, base, expert, z[:, 0], .8, 10)
        self.assertEqual(one['calls'], two['calls'])
        self.assertEqual(one['escalation_rate'], two['escalation_rate'])

    def test_invalid_scores_and_labels_rejected(self):
        z, y, base, expert = fixture()
        with self.assertRaises(ValueError):
            gate.calibrate(np.full(len(y), np.nan), y, base, expert)
        with self.assertRaises(ValueError):
            gate.calibrate(z[:, 0], y, base, expert, budgets=[50])
        with self.assertRaises(ValueError):
            gate.evaluate_policy(y, base, expert, z[:, 0], float('nan'), 10)
        with self.assertRaises(ValueError):
            gate.confusion(y, np.full(len(y), 8))

    def test_plan_freezes_scope_model_and_sources(self):
        plan = runner.frozen.read(runner.PLAN)
        runner.validate_plan(plan)
        for key, value in [('device', 'cuda'), ('seeds', [7]), ('fit_cap_per_class', 100000), ('budgets_percent', [50]),
                           *[(flag, True) for flag in runner.FALSE_FLAGS]]:
            with self.assertRaisesRegex(ValueError, 'scope'):
                runner.validate_plan(dict(plan, **{key: value}))
        with self.assertRaisesRegex(ValueError, 'evidence'):
            runner.validate_plan(dict(plan, specialist_results_sha256='wrong'))

    def test_existing_run_prevents_reads_and_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner.frozen, 'read', side_effect=AssertionError('Must not read')):
                with self.assertRaisesRegex(ValueError, 'Preserve'):
                    runner.run('absent', 'absent', 'absent', 'absent', tmp)

    def test_batched_features_and_scores_match_direct(self):
        z, y, base, expert = fixture(9000)
        tree = gate.fit_gate(z, y, base, expert)
        rows = np.arange(len(y))[::-1]
        self.assertTrue(np.array_equal(runner.make_features(z[:, :39], z[:, 39:], rows), z[rows]))
        self.assertTrue(np.array_equal(runner.score_rows(tree, z[:, :39], z[:, 39:], rows), gate.gate_score(tree, z[rows])))

    def test_fit_and_calibration_do_not_use_audit_labels_or_features(self):
        z, y, base, expert = fixture()
        split = np.repeat(np.array([0, 1, 2], np.uint8), [4800, 1600, 1600])
        rows, chosen = np.arange(len(y)), np.arange(4800)
        p = runner.frozen.read(runner.PLAN)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            np.save(out/'seed7-iid-probabilities.npy', z[:, 39:])
            for lane, pred in [('iid', base), ('balanced_panel', expert), ('light', base)]:
                np.save(out/f'seed7-{lane}-predictions.npy', pred)
            original = runner.prepare_pair(7, 'iid', z[:, :39], y, rows, split, chosen, out, p)
            first = (out/original['gate']).read_bytes()
            changed_y, changed_x = y.copy(), z[:, :39].copy()
            changed_y[split == 2] = (changed_y[split == 2]+1) % 8
            changed_x[split == 2] *= -10
            again = runner.prepare_pair(7, 'iid', changed_x, changed_y, rows, split, chosen, out, p)
            self.assertEqual(original, again)
            self.assertEqual(first, (out/original['gate']).read_bytes())
            with patch.object(gate, 'fit_gate', side_effect=AssertionError('Audit must not fit')):
                audit = runner.audit_pair(original, z[:, :39], y, rows, split, out, p)
            self.assertEqual(len(audit['policies']), 21)
            self.assertEqual(first, (out/original['gate']).read_bytes())

    def test_calibration_matches_independent_threshold_search(self):
        z, y, base, expert = fixture()
        scores = np.round(z[:, 0], 1)
        actual = gate.calibrate(scores, y, base, expert)
        sorted_scores = sorted(scores, reverse=True)
        options = {None}
        for basis_points in range(10, 1001, 10):
            limit = len(y)*basis_points//10000
            allowed = sorted({s for s in sorted_scores[:limit]}, reverse=True)
            options.update(s for s in allowed if np.count_nonzero(scores >= s) <= limit)
        for choice in actual['policies']:
            candidates = []
            for threshold in options:
                result = gate.evaluate_policy(y, base, expert, scores, threshold, choice['budget_percent'])
                if result['budget_respected'] and result['benign_guard_respected']:
                    candidates.append((result['macro_f1'], -result['calls']))
            self.assertEqual((choice['guarded']['macro_f1'], -choice['guarded']['calls']), max(candidates))

    def test_split_panel_has_no_cross_partition_duplicate_groups(self):
        z, y, _, _ = fixture(8000)
        x = z[:, :39].copy()
        x[800:] = x[:800].repeat(9, axis=0)
        # Exact membership is determined by features, even with conflicting
        # synthetic labels on equal vectors. No label-based split repair.
        with tempfile.TemporaryDirectory() as tmp:
            split, chosen, info = runner.split_panel(x, y, np.arange(len(y)), Path(tmp), runner.frozen.read(runner.PLAN))
            keys = gate.group_keys(x)
            groups = [set(keys[split == i]) for i in range(3)]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            self.assertTrue(np.all(split[chosen] == 0))
            self.assertEqual(sum(r['rows'] for r in info['partitions'].values()), len(y))


if __name__ == '__main__':
    unittest.main()
