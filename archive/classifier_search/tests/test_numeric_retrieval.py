"""Small synthetic fixtures; no project data or original test is opened."""
import unittest
import numpy as np
import torch
from src.eval import numeric_retrieval as r
from src.eval import matched_neural as old


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.x = np.random.default_rng(20).normal(size=(96, 6)).astype(np.float32)
        self.y = np.tile(np.arange(8), 12)

    def test_rank_coordinates_clipped_and_constant(self):
        np.testing.assert_array_equal(r.coordinates(np.array([[-1, 8], [2, -2], [9, 1]]),
                                                   [[0, 2, 4], [1]]), [[0, 0], [.5, 0], [1, 0]])
        with self.assertRaises(ValueError):
            r.coordinates(np.ones((3, 2)), [[1, 0], [1]])

    def test_bank_unique_deterministic_and_class_cap(self):
        x = np.concatenate([self.x, self.x[:8]])
        y = np.concatenate([self.y, np.roll(self.y[:8], 1)])
        bank, groups = r.bank(x, y, cap=4)
        again, other = r.bank(x, y, cap=4)
        np.testing.assert_array_equal(bank, again)
        np.testing.assert_array_equal(groups, other)
        self.assertTrue((bank < len(self.x)).all())
        self.assertEqual(len(np.unique(groups[bank])), len(bank))
        self.assertEqual(np.bincount(y[bank]).tolist(), [4]*8)

    def test_excludes_self_and_duplicate_vectors(self):
        x = np.concatenate([self.x, self.x[:8]])
        y = np.concatenate([self.y, self.y[:8]])
        bank, groups = r.bank(x, y)
        context = r.neighbours(x, x[bank], k=3, query_groups=groups, reference_groups=groups[bank])
        self.assertFalse(np.any(groups[:, None] == groups[bank][context]))
        distance = ((x[:, None]-x[bank][None])**2).sum(-1)
        distance[groups[:, None] == groups[bank]] = np.inf
        np.testing.assert_array_equal(context, np.argsort(distance, axis=1)[:, :3])
        with self.assertRaises(ValueError):
            r.neighbours(x, x[bank], query_groups=groups)

    def test_voting(self):
        np.testing.assert_array_equal(r.vote(np.array([[0, 1, 2], [2, 3, 4]]),
                                             np.array([1, 1, 2, 3, 3])), [1, 3])
        with self.assertRaises(ValueError):
            r.vote(np.array([[-1, 0]]), self.y)

    def test_replay_and_no_reference_control(self):
        bank, groups = r.bank(self.x, self.y)
        context = r.neighbours(self.x, self.x[bank], k=3, query_groups=groups,
                               reference_groups=groups[bank])
        records = []
        for enabled in [False, True]:
            args = (self.x, self.y, np.full(8, 12, dtype=np.int64), self.x[bank], self.y[bank], context, enabled)
            a, ar = r.fit(*args, epochs=2, batch=31)
            b, br = r.fit(*args, epochs=2, batch=31)
            self.assertEqual(ar, br)
            self.assertEqual(old.state_hash(a), old.state_hash(b))
            self.assertEqual(ar['examples_processed'], 192)
            self.assertEqual(ar['optimizer_steps'], 8)
            self.assertEqual(ar['reference_uses'], 576 if enabled else 0)
            pred = r.predict(a, self.x, self.x[bank], self.y[bank], context)
            np.testing.assert_array_equal(pred, r.predict(b, self.x, self.x[bank], self.y[bank], context))
            # Changing reference labels cannot affect the no-reference control.
            if not enabled:
                np.testing.assert_array_equal(pred, r.predict(a, self.x, self.x[bank],
                                                              (self.y[bank]+1)%8, context))
            records.append(ar)
        self.assertEqual(records[1]['parameters']-records[0]['parameters'], 256)

    def test_reference_labels_can_change_logits(self):
        torch.manual_seed(7)
        model = r.RetrievalModel(6).eval()
        x = torch.from_numpy(self.x[:4])
        context = torch.tensor([[1, 2]]*4)
        reference = torch.from_numpy(self.x)
        a = model(x, reference, torch.zeros(96, dtype=torch.long), context)
        b = model(x, reference, torch.ones(96, dtype=torch.long), context)
        self.assertFalse(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
