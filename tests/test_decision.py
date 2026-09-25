from __future__ import annotations

import unittest

from src.data.label_map import CLASSES
from src.eval.decision import Criteria, build_decision


def make_report(
    lane: str,
    seed: int,
    validation_f1: float,
    test_f1: float,
    parameters: int,
    loss: str = "sqrt_weighted_ce",
) -> dict:
    federated = lane == "federated-light"
    variant = "heavy" if lane == "centralized-heavy" else "light"
    report = {
        "variant": variant,
        "lane": "federated" if federated else "centralized",
        "config": {
            "hidden_dims": [512, 256, 128, 64] if variant == "heavy" else [64, 32],
            "dropout": 0.3 if variant == "heavy" else 0.2,
        },
        "loss": loss,
        "num_parameters": parameters,
        "history": [{"val_macro_f1": validation_f1}],
        "test_metrics": {
            "accuracy": test_f1,
            "macro_f1": test_f1,
            "per_class_recall": {class_name: test_f1 for class_name in CLASSES},
        },
        "args": {"seed": seed},
        "_source": f"{lane}_seed{seed}.json",
    }
    if federated:
        report.update(
            {
                "partitioner": "dirichlet",
                "alpha": 0.5,
                "num_clients": 20,
                "local_epochs": 1,
                "fraction_train": 1.0,
                "class_weights": "global",
                "strategy": "FedAvg",
            }
        )
    return report


class DecisionTests(unittest.TestCase):
    def test_complete_decision_can_prefer_equal_size_federated_model(self) -> None:
        reports = []
        for seed in range(3):
            reports.extend(
                [
                    make_report("centralized-heavy", seed, 0.90, 0.90, 200_000),
                    make_report("centralized-light", seed, 0.89, 0.89, 5_500),
                    make_report("federated-light", seed, 0.885, 0.885, 5_500),
                ]
            )

        decision = build_decision(
            reports,
            Criteria(
                min_macro_f1=0.8,
                min_class_recall=0.8,
                max_macro_f1_drop=0.02,
                prefer_federated=True,
            ),
        )

        self.assertEqual(decision["status"], "ready")
        self.assertEqual(decision["recommended_lane"], "federated-light")

    def test_missing_lane_blocks_final_recommendation(self) -> None:
        reports = [
            make_report("centralized-light", seed, 0.8, 0.8, 5_500) for seed in range(3)
        ]
        decision = build_decision(reports, Criteria())

        self.assertEqual(decision["status"], "incomplete")
        self.assertIsNone(decision["recommended_lane"])
        self.assertTrue(
            any(
                "missing federated-light" in issue
                for issue in decision["blocking_issues"]
            )
        )

    def test_configuration_selection_uses_validation_not_test(self) -> None:
        reports = []
        for seed in range(3):
            # Candidate A has better validation but worse test. It must be selected.
            reports.append(
                make_report("centralized-light", seed, 0.80, 0.70, 5_500, loss="ce")
            )
            reports.append(
                make_report("centralized-light", seed, 0.75, 0.95, 5_500, loss="focal")
            )
        decision = build_decision(reports, Criteria(loss=None, min_seeds=3))

        selected = decision["selected_candidates"]["centralized-light"]
        self.assertEqual(selected["loss"], "ce")
        self.assertAlmostEqual(selected["test_macro_f1_mean"], 0.70)


if __name__ == "__main__":
    unittest.main()
