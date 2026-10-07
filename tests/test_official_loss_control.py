"""Mathematical loss checks and label-only exposure replay parity."""
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from src.data.label_map import CLASSES
from src.data.official_packed import PackedData
from src.eval.official_exposure_audit import selected_rows, summarize
from src.models.losses import build_criterion
from src.models.official_losses import build_official_criterion, FixedTrainingDenominatorCE


class OfficialLossControlTests(unittest.TestCase):
    def setUp(self):
        self.counts = dict(zip(CLASSES, range(1, 9)))
        self.labels = torch.repeat_interleave(torch.arange(8), torch.arange(1, 9))
        torch.manual_seed(7)
        self.logits = torch.randn(36, 8, requires_grad=True)

    def test_default_and_complete_population_match_historical_loss(self):
        old = build_criterion('sqrt_weighted_ce', self.counts)
        default = build_official_criterion('sqrt_weighted_ce', self.counts)
        fixed = build_official_criterion('sqrt_weighted_ce', self.counts, 'fixed_train_mean')
        self.assertTrue(torch.equal(old(self.logits, self.labels), default(self.logits, self.labels)))
        torch.testing.assert_close(old(self.logits, self.labels), fixed(self.logits, self.labels))
        torch.testing.assert_close(torch.autograd.grad(old(self.logits, self.labels), self.logits)[0],
                                   torch.autograd.grad(fixed(self.logits, self.labels), self.logits)[0])

    def test_sample_weighted_client_gradients_equal_pooled_fixed_objective(self):
        fixed = build_official_criterion('sqrt_weighted_ce', self.counts, 'fixed_train_mean')
        pooled = fixed(self.logits, self.labels)
        # Strongly skewed clients evaluated at IDENTICAL logits/parameters. This
        # identity does not extend to different models after local Adam updates.
        parts = [slice(0, 3), slice(3, 15), slice(15, 36)]
        combined = sum(len(self.labels[p])/36 * fixed(self.logits[p], self.labels[p]) for p in parts)
        torch.testing.assert_close(pooled, combined)
        torch.testing.assert_close(torch.autograd.grad(pooled, self.logits)[0],
                                   torch.autograd.grad(combined, self.logits)[0])
        old = build_criterion('sqrt_weighted_ce', self.counts)
        old_split = sum(len(self.labels[p])/36 * old(self.logits[p], self.labels[p]) for p in parts)
        self.assertFalse(torch.isclose(old(self.logits, self.labels), old_split).item())

    def test_single_class_effect_and_weight_scale_invariance(self):
        fixed = build_official_criterion('sqrt_weighted_ce', self.counts, 'fixed_train_mean')
        labels = torch.zeros(36, dtype=torch.long)
        ratio = fixed.weight[0]/fixed.training_mean_weight
        torch.testing.assert_close(fixed(self.logits, labels), F.cross_entropy(self.logits, labels)*ratio)
        scaled = FixedTrainingDenominatorCE(fixed.weight*100, self.counts)
        torch.testing.assert_close(fixed(self.logits, self.labels), scaled(self.logits, self.labels))
        self.assertIn('training_mean_weight', fixed.state_dict())
        with self.assertRaisesRegex(ValueError, 'requires weighted'):
            build_official_criterion('ce', self.counts, 'fixed_train_mean')

    def test_exposure_prefix_matches_real_batching_including_singleton(self):
        data = object.__new__(PackedData)
        for n in (2, 8, 9, 16, 17, 31):
            x = np.zeros((n, 39), dtype=np.float32)
            x[:, 0] = np.arange(n)
            data.arrays = {'train': (x, np.arange(n, dtype=np.int64)%8)}
            for cap in (0, 1, 2, 100):
                batches = list(data.batches('train', 8, seed=17))
                if cap:
                    batches = batches[:cap]
                expected = np.concatenate([b[0][:, 0] for b in batches]).astype('int64')
                np.testing.assert_array_equal(selected_rows(np.arange(n), 8, 17, cap), expected)
        counts = summarize(np.arange(8), np.array([0, 1, 2, 3, 4, 5, 6, 7]))
        self.assertEqual(counts[CLASSES[0]]['unseen_unique'], 1)
        self.assertEqual(sum(v['processed_examples'] for v in counts.values()), 28)


if __name__ == '__main__':
    unittest.main()
