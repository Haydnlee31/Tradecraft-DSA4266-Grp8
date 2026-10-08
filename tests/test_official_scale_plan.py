"""Training-work accounting must follow the actual singleton-tail policy."""
import unittest
from src.eval.official_scale_plan import batches_per_pass, budgets


class ScalePlanTests(unittest.TestCase):
    def test_singleton_tail_and_known_counts(self):
        self.assertEqual(batches_per_pass(513), 1)
        self.assertEqual(batches_per_pass(1025), 2)
        self.assertEqual(batches_per_pass(500000), 977)
        self.assertEqual(batches_per_pass(2000000), 3907)
        with self.assertRaises(ValueError):
            batches_per_pass(1)

    def test_equal_examples_do_not_imply_equal_updates(self):
        small, large = budgets([500000], 20), budgets([2000000], 5)
        self.assertEqual(small['examples_processed'], large['examples_processed'])
        self.assertEqual(small['optimizer_steps'], 19540)
        self.assertEqual(large['optimizer_steps'], 19535)
        self.assertEqual(budgets([25000]*20, 20)['optimizer_steps'],
                         budgets([100000]*20, 5)['optimizer_steps'])


if __name__ == '__main__':
    unittest.main()
