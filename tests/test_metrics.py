from __future__ import annotations

import unittest

import numpy as np

from src.data.label_map import CLASSES
from src.eval.metrics import compute_metrics, summarize


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions_include_every_class(self) -> None:
        labels = np.arange(len(CLASSES))
        metrics = compute_metrics(labels, labels)

        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(set(metrics["per_class_recall"]), set(CLASSES))
        self.assertTrue(
            all(value == 1.0 for value in metrics["per_class_recall"].values())
        )

    def test_missing_classes_are_not_silently_dropped(self) -> None:
        metrics = summarize(np.array([0, 0]), np.array([0, 0]))

        self.assertEqual(metrics["recall/Benign"], 1.0)
        for class_name in CLASSES[1:]:
            self.assertIn(f"recall/{class_name}", metrics)
            self.assertEqual(metrics[f"recall/{class_name}"], 0.0)

    def test_mismatched_lengths_fail(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics(np.array([0, 1]), np.array([0]))


if __name__ == "__main__":
    unittest.main()
