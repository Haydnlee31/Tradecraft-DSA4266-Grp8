"""Diagnostic ranks and float32 collision counts are not accuracy claims."""
import unittest
import numpy as np
from src.data.label_map import CLASSES
from src.eval.official_rare_diagnostic import rank_summary, select_rows, rare_packed_overlap


class RareDiagnosticTests(unittest.TestCase):
    def test_rank_and_margin(self):
        x = np.zeros((3, len(CLASSES)))
        x[:, 4] = [3, 2, 1]
        x[:, 0] = [1, 3, 3]
        x[:, 1] = [0, 0, 2]
        r = rank_summary(x, 4)
        self.assertEqual(r['recall'], 1/3)
        self.assertEqual(r['top2_recall'], 2/3)
        self.assertEqual(r['top3_recall'], 1)
        self.assertEqual(r['median_true_vs_best_other_logit'], -1)
        with self.assertRaises(ValueError):
            rank_summary(np.full((2, 8), np.nan), 0)

    def test_all_rare_and_bounded_other_rows(self):
        y = np.repeat(np.arange(8), 10)
        a, b = select_rows(y, 3), select_rows(y, 3)
        for name in CLASSES:
            np.testing.assert_array_equal(a[name], b[name])
            self.assertEqual(len(a[name]), 10 if name in ('Web-based', 'Brute Force') else 3)

    def test_collision_counts_and_signed_zero(self):
        web = CLASSES.index('Web-based')
        a = np.array([[0., 1.], [-0., 1.], [2., 3.]], dtype=np.float32)
        b = np.array([[0., 1.]], dtype=np.float32)
        r = rare_packed_overlap({'train': (a, np.array([web, 0, web])),
                                 'val': (b, np.array([web]))})
        self.assertEqual(r['train/Web-based']['rows_matching_other_class_in_train'], 1)
        self.assertEqual(r['val/Web-based']['rows_matching_any_train_vector'], 1)
        self.assertEqual(r['val/Web-based']['rows_matching_other_class_in_train'], 1)


if __name__ == '__main__':
    unittest.main()
