"""Loss mathematics, sample weighting and deterministic synthetic fits."""
import unittest
import numpy as np
import torch
from src.eval import contrastive_flow as c
from src.eval import matched_neural as old


class ContrastiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_importance_correction_preserves_old_expected_objective(self):
        y = np.repeat(np.arange(8), np.arange(2, 10))
        pop = np.arange(2, 10)*100
        losses = np.random.default_rng(1).uniform(size=len(y))
        original = np.mean(old.fit_weights(y, pop)*losses)
        balanced = np.mean(c.class_weights(pop)*[losses[y == i].mean() for i in range(8)])
        self.assertAlmostEqual(original, balanced, places=14)
        self.assertAlmostEqual(c.class_weights(pop).mean(), 1.)

    def test_same_balanced_batches_without_within_batch_replacement(self):
        y = np.repeat(np.arange(8), 12)
        a = list(c.batches(y, 7, 1, steps=3, per_class=4))
        b = list(c.batches(y, 7, 1, steps=3, per_class=4))
        for x, z in zip(a, b):
            np.testing.assert_array_equal(x, z)
            self.assertEqual(len(np.unique(x)), 32)
            self.assertEqual(np.bincount(y[x]).tolist(), [4]*8)
        with self.assertRaises(ValueError):
            list(c.batches(y, 7, 1, per_class=13))

    def test_loss_matches_explicit_formula_with_duplicate_exclusion(self):
        x = torch.tensor([[1., 0.], [.8, .2], [0., 1.], [.2, .8], [1., 0.]], dtype=torch.float64)
        labels = torch.tensor([0, 0, 1, 1, 1])
        groups = torch.tensor([0, 1, 2, 3, 0])  # last row conflicts with first
        loss, used = c.supcon(x, labels, groups)
        z = x.numpy()/np.linalg.norm(x.numpy(), axis=1, keepdims=True)
        values = []
        for i in range(len(z)):
            allowed = groups.numpy() != groups[i].item()
            positives = allowed & (labels.numpy() == labels[i].item())
            if positives.any():
                logits = z[i] @ z.T/.1
                values.append(np.log(np.exp(logits[allowed]).sum())-logits[positives].mean())
        self.assertEqual(used, len(values))
        self.assertAlmostEqual(float(loss), np.mean(values), places=12)

    def test_separation_lowers_loss_and_gradients_are_finite(self):
        y, group = torch.tensor([0, 0, 1, 1]), torch.arange(4)
        good = torch.tensor([[1., 0.], [1., .1], [-1., 0.], [-1., .1]], requires_grad=True)
        bad = good.detach()[[0, 2, 1, 3]].clone().requires_grad_()
        a, _ = c.supcon(good, y, group)
        b, _ = c.supcon(bad, y, group)
        self.assertLess(float(a.detach()), float(b.detach()))
        b.backward()
        self.assertTrue(torch.isfinite(bad.grad).all())

    def test_no_positives_or_all_duplicate_groups_is_connected_zero(self):
        for y, group in [(torch.arange(4), torch.arange(4)), (torch.zeros(4), torch.zeros(4))]:
            x = torch.zeros((4, 3), requires_grad=True)
            loss, used = c.supcon(x, y, group)
            self.assertEqual(used, 0)
            self.assertEqual(float(loss.detach()), 0.)
            loss.backward()
            self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))

    def test_identical_initialization_sampling_and_replay(self):
        x = np.random.default_rng(1).normal(size=(96, 9)).astype(np.float32)
        y = np.repeat(np.arange(8), 12)
        s = dict(c.SETTINGS, epochs=2, steps_per_epoch=2, per_class=4)
        records = []
        for coefficient in [0., .1]:
            a, ar = c.fit(x, y, np.arange(len(y)), np.full(8, 12), coefficient, s)
            b, br = c.fit(x, y, np.arange(len(y)), np.full(8, 12), coefficient, s)
            self.assertEqual(ar, br)
            self.assertEqual(old.state_hash(a), old.state_hash(b))
            np.testing.assert_array_equal(c.predict(a, x), c.predict(b, x))
            self.assertEqual(ar['examples_processed'], 128)
            self.assertEqual(ar['optimizer_steps'], 4)
            self.assertEqual(ar['sampled_class_counts'], [16]*8)
            self.assertEqual(ar['eligible_contrastive_anchors'], 128)
            records.append(ar)
        for field in ['initial_state_sha256', 'batch_order_sha256', 'class_weights', 'parameters']:
            self.assertEqual(records[0][field], records[1][field])
        self.assertNotEqual(records[0]['final_state_sha256'], records[1]['final_state_sha256'])

    def test_projector_not_used_at_inference(self):
        torch.manual_seed(7)
        model = c.FlowModel(3)
        x = np.random.default_rng(1).normal(size=(12, 3)).astype(np.float32)
        first = c.predict(model, x)
        with torch.no_grad():
            for p in model.projector.parameters():
                p.fill_(999.)
        np.testing.assert_array_equal(first, c.predict(model, x))


if __name__ == '__main__':
    unittest.main()
