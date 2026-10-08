"""Old rows and class support must survive controlled cohort expansion exactly."""
import unittest
import numpy as np

from src.federated.official_nested_partition import nested_offsets, scaled_counts, expand
from src.federated.partition import check_exact_cover


class NestedPartitionTests(unittest.TestCase):
    def test_interleaved_shard_mapping(self):
        shards = [{'path': str(i), 'tier': tier, 'rows': n}
                  for i, (tier, n) in enumerate([(0, 2), (1, 3), (2, 9), (0, 1), (1, 2)])]
        mapping, size = nested_offsets(shards)
        np.testing.assert_array_equal(mapping, [0, 1, 5])
        self.assertEqual(size, 8)
        with self.assertRaises(ValueError):
            nested_offsets(shards + [shards[0]])

    def test_rounding_and_zero_support(self):
        anchor = np.array([[0, 2], [1, 3], [2, 0]])
        target = scaled_counts(anchor, np.array([10, 21]))
        np.testing.assert_array_equal(target.sum(axis=0), [10, 21])
        self.assertTrue(np.all(target[anchor == 0] == 0))
        self.assertTrue(np.all(target >= anchor))
        self.assertLess(np.abs(target-anchor/anchor.sum(axis=0)*[10, 21]).max(), 1)
        with self.assertRaises(ValueError):
            scaled_counts(anchor, np.array([2, 21]))
        with self.assertRaises(ValueError):
            scaled_counts(anchor.astype(float), np.array([10, 21]))

    def test_exact_cover_owner_retention_and_reproducibility(self):
        small = np.tile(np.arange(8), 3)
        large = np.tile(np.arange(8), 11)
        mapping = np.arange(len(small))
        original = [np.flatnonzero(small < 4), np.flatnonzero(small >= 4)]
        parts, anchor, target = expand(small, large, original, mapping)
        repeated, _, _ = expand(small, large, original, mapping)
        check_exact_cover(parts, len(large))
        for old, new, again in zip(original, parts, repeated):
            self.assertTrue(np.isin(mapping[old], new).all())
            np.testing.assert_array_equal(new, again)
        self.assertTrue(np.all(target[anchor == 0] == 0))
        wrong = mapping.copy()
        wrong[0] = wrong[1]
        with self.assertRaises(ValueError):
            expand(small, large, original, wrong)
        with self.assertRaises(ValueError):
            expand(small, large, original, mapping+1)

    def test_many_integer_targets_stay_nested(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            anchor = rng.integers(0, 100, (20, 8))
            totals = anchor.sum(axis=0)+rng.integers(0, 1000, 8)
            target = scaled_counts(anchor, totals)
            np.testing.assert_array_equal(target.sum(axis=0), totals)
            self.assertTrue(np.all(target >= anchor))
            self.assertLess(np.abs(target-anchor/anchor.sum(axis=0)*totals).max(), 1)


if __name__ == '__main__':
    unittest.main()
