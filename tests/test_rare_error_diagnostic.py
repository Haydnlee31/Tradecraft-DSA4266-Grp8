"""Synthetic fixtures only: no real flows, training or original test access."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from src.data.label_map import CLASSES, WEB_BASED, BRUTE_FORCE, BENIGN, RECON
from src.eval import rare_error_diagnostic as r
from scripts import official39_rare_error_diagnostic as runner


class RareErrorTests(unittest.TestCase):
    def test_overlap_and_oracle_are_paired_not_sum_of_hits(self):
        y = np.array([4, 4, 4, 4, 0, 3, 5])
        p = dict(tree=np.array([4, 0, 3, 0, 4, 3, 5]),
                 other=np.array([0, 4, 3, 0, 0, 4, 5]))
        s = r.overlap(y, p)['classes'][WEB_BASED]
        self.assertEqual(s['correct_model_count_histogram'], [2, 2, 0])
        self.assertEqual(s['any_model_correct'], 2)
        self.assertEqual(s['oracle_recall'], .5)
        self.assertEqual(s['unanimous_wrong_destinations'][BENIGN], 1)
        self.assertEqual(s['unanimous_wrong_destinations'][RECON], 1)
        arm = s['arms']['other']
        self.assertEqual(arm['recovered_tree_misses'], 1)
        self.assertEqual(arm['lost_tree_hits'], 1)
        self.assertEqual(arm['false_positive_sources'][RECON], 1)
        self.assertEqual(s['arms']['tree']['false_positive_sources'][BENIGN], 1)
        self.assertIsNone(r.overlap(y, p)['classes'][CLASSES[1]]['oracle_recall'])

    def test_exact_conflicting_groups_and_signed_zero(self):
        x = np.array([[0, 2], [-0., 2], [0, 2], [1, 2], [1, 2]], np.float32)
        y = np.array([4, 4, 0, 5, 5])
        s = r.exact_conflicts(x, y)
        self.assertEqual(s['unique_vectors'], 2)
        self.assertEqual(s['mixed_label_groups'], 1)
        self.assertEqual(s['minimum_total_fitting_errors_from_exact_conflicts'], 1)
        self.assertEqual(s['classes'][WEB_BASED]['rows_in_mixed_label_groups'], 2)
        self.assertEqual(s['classes'][WEB_BASED]['conflicting_groups_by_other_class'][BENIGN], 1)
        self.assertEqual(s['classes'][BRUTE_FORCE]['rows_in_mixed_label_groups'], 0)

    def test_nearest_matches_brute_force_and_duplicate_invariance(self):
        rng = np.random.default_rng(20)
        x = rng.normal(size=(32, 3)).astype(np.float32)
        y = np.tile(np.arange(8), 4)
        q = rng.normal(size=(5, 3))
        actual = r.nearest_by_class(q, x, y)
        expected = np.stack([np.sqrt(((q[:, None]-x[y == k][None])**2).sum(2)).min(1)
                             for k in range(8)], axis=1)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(actual, r.nearest_by_class(
            q, np.concatenate([x, x[:4]]), np.concatenate([y, y[:4]])))

    def test_missing_class_rejected(self):
        with self.assertRaises(ValueError):
            r.nearest_by_class(np.ones((1, 2)), np.zeros((4, 2)), np.arange(4))

    def test_proximity_ties_strata_and_empty_stratum(self):
        y = np.array([4, 4, 4, 5])
        p = dict(tree=np.array([4, 0, 0, 0]), other=np.array([0, 4, 0, 0]))
        d = np.full((4, 8), 10.)
        d[np.arange(4), y] = 2
        d[0, 0], d[1, 3], d[2, 0], d[3, 4] = 3, 1, 2, 0
        s = r.proximity(d, y, p)
        web = s[WEB_BASED]
        self.assertEqual([web[g]['rows'] for g in ['all', 'tree_correct',
                         'tree_missed_other_model_correct', 'all_models_missed']], [3, 1, 1, 1])
        self.assertEqual(web['all']['other_class_strictly_closer'], 1)
        self.assertEqual(web['all']['nearest_other_equal_distance'], 1)
        self.assertEqual(web['all']['comparisons'][BENIGN],
                         dict(other_closer=0, equal_distance=1, own_closer=2))
        self.assertEqual(s[BRUTE_FORCE]['tree_correct'], dict(rows=0))
        self.assertEqual(s[BRUTE_FORCE]['all']['zero_any_class_distance'], 1)

    def test_bad_inputs_fail_closed(self):
        for y in [np.array([-.1]), np.array([8]), np.array([], dtype=int), np.array([True])]:
            with self.assertRaises(ValueError):
                r.labels(y)
        with self.assertRaises(ValueError):
            r.exact_conflicts(np.array([[np.nan]], np.float32), np.array([4]))
        with self.assertRaises(ValueError):
            r.overlap(np.array([4, 5]), dict(tree=np.array([4]), other=np.array([5])))
        with self.assertRaises(ValueError):
            r.proximity(np.full((1, 8), -1.), np.array([4]), dict(tree=np.array([4])))

    def test_misaligned_rows_rejected_before_scoring(self):
        d = SimpleNamespace(fit=np.array([2, 3]), calibration=np.array([4, 5]))
        with patch.object(runner.np, 'load', return_value=np.array([3, 2])):
            with self.assertRaisesRegex(ValueError, 'Row alignment'):
                runner.aligned_predictions(d, {}, ['tree'])

    def test_new_output_and_independent_replay_required(self):
        from pathlib import Path
        with self.assertRaisesRegex(ValueError, 'Preserve existing'):
            runner.run(Path(__file__))
        with self.assertRaisesRegex(ValueError, 'Fresh independent'):
            runner.publish(Path('same'), Path('same'), Path('unused'))


if __name__ == '__main__':
    unittest.main()
