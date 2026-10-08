"""Bounded fixtures check probe sampling, weighting, and deterministic replay."""
import copy
import unittest

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_packed import PackedData
from src.eval.official_aggregation_probe import (
    common_probe, conditional_summary, weighted_probabilities, one_round, TARGETS,
)
from src.models.architectures import MLPClassifier, MLPConfig
from src.utils.seed import set_seed


class AggregationProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(7)
        labels = np.arange(96, dtype=np.int64) % len(CLASSES)
        self.data = object.__new__(PackedData)
        self.data.arrays = {s: (rng.normal(size=(96, 39)).astype('float32'), labels.copy())
                            for s in ('train', 'val')}
        self.data.manifest = {'features': [f'f{i}' for i in range(39)], 'train_rows': 96,
                             'train_class_counts': dict.fromkeys(CLASSES, 12)}
        self.settings = {'seed': 7, 'batch_size': 8, 'loss': 'sqrt_weighted_ce',
                         'loss_reduction': 'batch_weight_sum', 'lr': .001, 'weight_decay': 1e-5}
        set_seed(7)
        self.model = MLPClassifier(39, len(CLASSES), MLPConfig('fixture', (8,), normalization='layer'))

    def test_probe_includes_all_target_rows_and_is_repeatable(self):
        labels = self.data.arrays['val'][1]
        a, b = common_probe(labels, 3), common_probe(labels, 3)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), len(np.unique(a)))
        for c, name in enumerate(CLASSES):
            self.assertEqual(int((labels[a] == c).sum()), 12 if name in TARGETS else 3)

    def test_probability_weights_and_guards(self):
        a, b = np.zeros((1, 8)), np.zeros((1, 8))
        a[0, :2], b[0, :2] = [.8, .2], [.1, .9]
        np.testing.assert_allclose(weighted_probabilities([a, b], [3, 1])[0, :2], [.625, .375])
        with self.assertRaises(ValueError):
            weighted_probabilities([a, b], [1, 0])
        with self.assertRaises(ValueError):
            weighted_probabilities([a*2, b], [1, 1])

    def test_probability_shapes_and_extreme_weights(self):
        p = np.full((2, len(CLASSES)), 1/len(CLASSES))
        for matrices, weights in (([p], [[1]]), ([p[:, 0]], [1]),
                                  ([p[:, :2]], [1]), ([p], [float('nan')]),
                                  ([p], [float('inf')]), ([p], [-1]), ([], [])):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    weighted_probabilities(matrices, weights)
        np.testing.assert_array_equal(weighted_probabilities([p, p], [1e308, 1e308]), p)

    def test_absent_class_is_not_zero_recall(self):
        r = conditional_summary(np.array([0, 1]), np.array([0, 0]))
        self.assertEqual(r['per_class'][CLASSES[0]]['recall'], .5)
        self.assertIsNone(r['per_class'][CLASSES[1]]['recall'])
        self.assertNotIn('precision', r['per_class'][CLASSES[0]])

    def test_one_client_has_identical_aggregation_rules(self):
        parts = [np.arange(96)]
        result = one_round(self.data, self.model, parts, self.settings, np.arange(96), 21)
        self.assertEqual(result['full_validation_parameter_average'], result['full_validation_probability_average'])
        self.assertEqual(result['examples_processed'], 96)
        self.assertEqual(result['optimizer_steps'], 12)

    def test_replay_preserves_global_weights_and_full_exposure(self):
        original = copy.deepcopy(self.model.state_dict())
        parts = [np.arange(0, 32), np.arange(32, 96)]
        a = one_round(self.data, self.model, parts, self.settings, np.arange(96), 21)
        b = one_round(self.data, self.model, parts, self.settings, np.arange(96), 21)
        self.assertEqual(a, b)
        self.assertEqual(a['aggregation_weights'], [32, 64])
        self.assertEqual(a['examples_processed'], 96)
        self.assertEqual(a['optimizer_steps'], 12)
        for name, tensor in original.items():
            self.assertTrue(torch.equal(tensor, self.model.state_dict()[name]))


if __name__ == '__main__':
    unittest.main()
