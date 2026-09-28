"""Regression checks for the Joel/centralized-heavy branch integration."""

import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAdagrad, FedAvg

from src.data.label_map import CLASSES
from src.federated import task
from src.federated.client_app import train_partition
from src.federated.custom_strategy import CustomFedAvg, CustomFedAdagrad
from src.federated.server_app import build_strategy, report_filename
from src.eval.decision import aggregate_candidates
from src.eval.study import name as study_name
from tests.test_decision import make_report

ROOT = Path(__file__).resolve().parents[1]


class MergeTests(unittest.TestCase):
    def config(self):
        return tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]

    def test_baseline_stays_fedavg_and_alternative_is_real_adagrad(self):
        rc = self.config()
        baseline, _ = build_strategy(rc)
        self.assertIsInstance(baseline, FedAvg)
        self.assertNotIsInstance(baseline, FedAdagrad)
        self.assertEqual(rc["lr-decay-every"], 0)
        self.assertFalse(rc["use-wandb"])
        baseline_name = report_filename(rc, 20)
        rc["strategy"] = "fedadagrad"
        adaptive, settings = build_strategy(rc)
        self.assertIsInstance(adaptive, CustomFedAdagrad)
        self.assertIsInstance(adaptive, FedAdagrad)
        self.assertEqual(settings["eta"], rc["fedadagrad-eta"])
        self.assertNotEqual(baseline_name, report_filename(rc, 20))

    def test_invalid_strategy_does_not_silently_fall_back(self):
        rc = self.config()
        rc["strategy"] = "typo"
        with self.assertRaisesRegex(ValueError, "unknown strategy"):
            build_strategy(rc)

    def test_client_f1_spread_not_weighted_global_f1(self):
        strategy = CustomFedAvg()
        replies = []
        for pid, score, size in ((0, .2, 1), (1, .8, 9)):
            record = MetricRecord({"client_macro_f1": score, "partition-id": pid,
                                   "num-examples": size, "eval_loss": 1., "eval_acc": score})
            # Message construction outside a Flower task needs runtime IDs;
            # mock only the envelope here. The Ray smoke test covers transport.
            message = MagicMock(spec=Message)
            message.has_error.return_value = False
            message.content = RecordDict({"metrics": record})
            replies.append(message)
        metrics = strategy.aggregate_evaluate(1, replies)
        self.assertAlmostEqual(metrics["client_macro_f1_median"], .5)
        self.assertAlmostEqual(metrics["client_eval_acc"], .74)
        self.assertEqual(strategy._client_eval[1], {0: .2, 1: .8})
        self.assertNotIn("eval_macro_f1", metrics)

    def test_failed_run_keeps_history_and_closes_optional_logger(self):
        strategy = CustomFedAvg(fraction_evaluate=0)
        strategy.configure_train = MagicMock(return_value=[MagicMock()])
        grid = MagicMock()
        grid.send_and_receive.return_value = []
        logger = MagicMock()
        strategy._wandb = logger  # Mock only: never contact the W&B service.
        with tempfile.TemporaryDirectory() as folder:
            strategy.set_output(Path(folder), {})
            with self.assertRaises(RuntimeError):
                strategy.start(grid, ArrayRecord({"weight": torch.ones(1)}), num_rounds=1,
                               evaluate_fn=lambda rnd, arrays: MetricRecord({"val_macro_f1": .2}))
            history = json.loads((Path(folder) / "history.json").read_text())
            self.assertEqual(history[0]["server_eval"]["val_macro_f1"], .2)
            logger.finish.assert_called_once()

    def test_schedule_changes_only_when_explicitly_enabled(self):
        with patch.object(FedAvg, "configure_train", return_value=[]):
            for interval, expected in ((0, 1.), (5, .5)):
                strategy = CustomFedAvg(lr_decay_every=interval)
                config = ConfigRecord({"lr": 1.})
                for rnd in range(1, 7):
                    strategy.configure_train(rnd, ArrayRecord({"weight": torch.ones(1)}), config, MagicMock())
                self.assertEqual(config["lr"], expected)

    def test_adagrad_hyperparameters_are_not_grouped_together(self):
        reports = []
        for eta in (.1, .2):
            report = make_report("federated-light", 0, .7, .7, 5544)
            report["strategy"] = "FedAdagrad"
            report["args"].update({"fedadagrad-eta": eta, "fedadagrad-tau": .001})
            reports.append(report)
        self.assertEqual(len(aggregate_candidates(reports)), 2)

    def test_study_labels_distinguish_optimizers(self):
        candidate = {"lane": "federated-light", "partitioner": "iid", "strategy": "fedavg"}
        baseline = study_name(candidate)
        candidate["strategy"] = "fedadagrad"
        self.assertNotEqual(baseline, study_name(candidate))

    def test_client_cache_reuses_tensors_but_not_shuffle_state(self):
        with tempfile.TemporaryDirectory() as folder:
            splits = Path(folder)
            frame = pl.DataFrame({"x": np.arange(64, dtype=float),
                                  "class": [CLASSES[i % 8] for i in range(64)]})
            for split in ("train", "val", "test"):
                frame.write_parquet(splits / f"{split}.parquet")
            config = {"splits-dir": folder, "partitioner": "iid"}
            task.configure(config)
            with patch.object(task, "_load_modulo_slice", wraps=task._load_modulo_slice) as reader:
                first = task.load_client_train(0, 2, 8, seed=1)
                second = task.load_client_train(0, 2, 8, seed=2)
                reader.assert_called_once()
                self.assertIs(first.dataset, second.dataset)
                self.assertIsNot(first.generator, second.generator)
            # Same pathname, changed source: no stale tensor/scaler reuse.
            frame.head(32).write_parquet(splits / "train.parquet")
            task.configure(config)
            third = task.load_client_train(0, 2, 8, seed=1)
            self.assertEqual(len(third.dataset), 16)
            self.assertIsNot(first.dataset, third.dataset)
            config.update({"seed": 0, "batch-size": 8, "local-epochs": 1,
                           "loss": "sqrt_weighted_ce", "class-weights": "global"})
            torch.set_num_threads(1)
            model = task.load_model()
            cached, _ = train_partition(model.state_dict(), 0, 2, config, 1, .001)
            task._tensor_cache.clear()
            uncached, _ = train_partition(model.state_dict(), 0, 2, config, 1, .001)
            for name in cached:
                torch.testing.assert_close(cached[name], uncached[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
