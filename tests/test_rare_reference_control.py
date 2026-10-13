"""Count-control fixtures; no real data or frozen models are loaded."""
import unittest
from pathlib import Path
import numpy as np
from scripts import official39_rare_reference_control as c


class RareReferenceTests(unittest.TestCase):
    def test_within_class_unique_preserves_conflicting_labels(self):
        x = np.zeros((16, 2), np.float32)
        y = np.tile(np.arange(8), 2)
        groups = c.unique_candidates(x, y)
        self.assertEqual([a.tolist() for a in groups], [[k] for k in range(8)])

    def test_balanced_deterministic_no_replacement(self):
        groups = [np.arange(k*30, (k+1)*30) for k in range(8)]
        a = c.balanced_indices(groups, 12, 7)
        np.testing.assert_array_equal(a, c.balanced_indices(groups, 12, 7))
        self.assertFalse(np.array_equal(a, c.balanced_indices(groups, 12, 17)))
        self.assertEqual(np.bincount(a//30).tolist(), [12]*8)
        self.assertEqual(len(np.unique(a)), 96)

    def test_rejects_missing_or_overlapping_candidates(self):
        for groups in [[np.arange(3)]*8, [np.arange(3)]*7]:
            with self.assertRaises(ValueError):
                c.balanced_indices(groups, 2, 7)
        with self.assertRaises(ValueError):
            c.balanced_indices([np.arange(k*2, k*2+2) for k in range(8)], 3, 7)

    def test_no_overwrite_or_self_replay(self):
        with self.assertRaises(ValueError):
            c.run(Path(__file__))
        with self.assertRaises(ValueError):
            c.publish(Path('same'), Path('same'), Path('unused'))


if __name__ == '__main__':
    unittest.main()
