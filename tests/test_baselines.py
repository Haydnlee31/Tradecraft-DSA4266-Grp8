"""Small real-training checks, never using the research data or test results."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
import torch
from torch.utils.data import TensorDataset

from src.baselines.run import LogisticRegression, loader, main, train_pass
from src.data.label_map import CLASSES


def synthetic_splits(path, include_test=False):
    path.mkdir()
    rng = np.random.default_rng(9)
    for name, rows in [("train", 129), ("val", 32)] + ([("test", 32)] if include_test else []):
        frame = {f"x{i}": rng.normal(size=rows) for i in range(46)}
        frame["class"] = [CLASSES[i % len(CLASSES)] for i in range(rows)]
        pl.DataFrame(frame).write_parquet(path / f"{name}.parquet")


class BaselineTests(unittest.TestCase):
    def test_linear_model_and_identical_update_paths(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        central = LogisticRegression(46)
        local = copy.deepcopy(central)
        self.assertEqual(sum(p.numel() for p in central.parameters()), 376)
        data = TensorDataset(torch.randn(33, 46), torch.arange(33) % len(CLASSES))
        criterion = torch.nn.CrossEntropyLoss()
        for model in (central, local):
            opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
            # The final singleton must be included (there is no BatchNorm).
            self.assertEqual(train_pass(model, loader(data, 16, 7), criterion, opt), (33, 3))
        for key in central.state_dict():
            torch.testing.assert_close(central.state_dict()[key], local.state_dict()[key], rtol=0, atol=0)

    def test_all_logistic_lanes_validation_only_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits"
            synthetic_splits(splits)  # Deliberately no test.parquet: never touch it.
            common = ["--splits", str(splits), "--seeds", "0", "--clients", "2",
                      "--epochs", "2", "--batch-size", "32", "--threads", "1",
                      "--lanes", "logistic", "logistic-iid", "logistic-dirichlet"]
            main(common + ["--output", str(root / "first")])
            main(common + ["--output", str(root / "second")])
            manifest = json.loads((root / "first/environment.json").read_text())
            self.assertEqual(set(manifest["split_sha256"]), {"train", "val"})
            self.assertTrue(manifest["source_sha256"])
            self.assertEqual(manifest["settings"]["partition_seed"], 0)
            for lane in ("logistic", "logistic-iid", "logistic-dirichlet"):
                report = json.loads((root / "first" / f"{lane}_seed0.json").read_text())
                other = json.loads((root / "second" / f"{lane}_seed0.json").read_text())
                self.assertIsNone(report["test_metrics"])
                self.assertEqual(report["num_parameters"], 376)
                self.assertEqual(report["examples_processed"], 258)
                self.assertEqual(report["history"], other["history"])
                self.assertEqual(set(report["validation_metrics"]["per_class"]), set(CLASSES))
                saved = torch.load(root / "first" / report["checkpoint"], weights_only=True)
                self.assertEqual(saved["classes"], CLASSES)
            with self.assertRaises(FileExistsError):
                main(common + ["--output", str(root / "first")])

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "optional XGBoost not installed")
    def test_xgboost_checkpoint_and_explicit_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits"
            synthetic_splits(splits, include_test=True)
            output = root / "run"
            main(["--splits", str(splits), "--output", str(output), "--lanes", "xgboost",
                  "--seeds", "0", "--boost-rounds", "3", "--max-depth", "2",
                  "--threads", "1", "--patience", "0", "--evaluate-test"])
            report = json.loads((output / "xgboost_seed0.json").read_text())
            self.assertEqual(report["saved_boost_rounds"], report["best_step"])
            self.assertTrue((output / report["checkpoint"]).exists())
            self.assertEqual(set(report["test_metrics"]["per_class"]), set(CLASSES))
            self.assertEqual(sum(map(sum, report["test_metrics"]["confusion_matrix"])), 32)

    def test_rejects_bad_inputs_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaises(ValueError):
                main(["--output", str(output), "--lanes", "logistic", "--epochs", "0"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
