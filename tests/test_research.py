"""Recovery equivalence and fail-closed checks on synthetic train/val data."""

import json
import importlib.metadata
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from tests.test_baselines import synthetic_splits
from src.data.label_map import CLASSES
from src.eval.training_parity import compare
from src.models.research import main, resolve_device


class ResearchTests(unittest.TestCase):
    def test_failed_atomic_save_preserves_last_checkpoint(self):
        from src.models.research import atomic_save
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            atomic_save({"step": 1}, path)
            with patch("src.models.research.torch.save", side_effect=OSError("interrupted write")):
                with self.assertRaises(OSError):
                    atomic_save({"step": 2}, path)
            self.assertEqual(torch.load(path, weights_only=True)["step"], 1)

    def test_plan_is_bounded_and_all_commands_parse(self):
        from src.eval.tuning_plan import commands
        from src.models.research import parse
        plan = json.loads((Path(__file__).resolve().parents[1] / "configs/local_tuning_plan.json").read_text())
        generated = commands(plan, Path("outputs/test-plan"))
        self.assertEqual(len(generated), 12)
        for command in generated:
            args = parse(command[3:])
            self.assertEqual(args.epochs, 60)
            self.assertEqual(args.partition_seed, 0)

    def test_legacy_manifest_does_not_require_ray(self):
        from src.experiment import package_versions
        def version(package):
            if package == "ray":
                raise importlib.metadata.PackageNotFoundError(package)
            return "test-version"
        with patch("src.experiment.importlib.metadata.version", side_effect=version):
            self.assertIsNone(package_versions()["ray"])

    def test_real_central_and_client_loop_parity(self):
        torch.manual_seed(0)
        data = TensorDataset(torch.randn(64, 46), torch.arange(64) % 8)
        checks = compare(data, {c:8 for c in CLASSES}, batch_size=16)
        self.assertTrue(all(c["passed"] for c in checks))

    def test_recovery_exact_all_lanes_and_test_never_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_splits(root / "splits")  # No test.parquet exists.
            for lane in ("heavy", "light", "iid", "dirichlet"):
                with self.subTest(lane=lane):
                    common = ["--splits", str(root / "splits"), "--lane", lane,
                              "--epochs", "3", "--clients", "2", "--threads", "1", "--batch-size", "32"]
                    full, resumed = root / f"{lane}_full", root / f"{lane}_resumed"
                    main(common + ["--output", str(full)])
                    main(common + ["--output", str(resumed), "--stop-after", "1"])
                    self.assertFalse((resumed / "result.json").exists())
                    main(common + ["--output", str(resumed), "--resume"])
                    # Compare last training state too, not merely an early best.
                    a = torch.load(full / "last.pt", weights_only=True)
                    b = torch.load(resumed / "last.pt", weights_only=True)
                    for key, value in a["model"].items():
                        torch.testing.assert_close(value, b["model"][key], rtol=0, atol=0)
                    ar = json.loads((full / "result.json").read_text())
                    br = json.loads((resumed / "result.json").read_text())
                    self.assertEqual(ar["validation_metrics"], br["validation_metrics"])
                    self.assertEqual(ar["best_step"], br["best_step"])
                    self.assertIsNone(br["test_metrics"])
                    for ah, bh in zip(ar["history"], br["history"]):
                        self.assertEqual(ah["validation_metrics"], bh["validation_metrics"])
                        self.assertEqual(ah["val_loss"], bh["val_loss"])
                    with self.assertRaises(ValueError):
                        main(common + ["--output", str(resumed), "--resume", "--lr", ".0003"])
                    with self.assertRaises(FileExistsError):
                        main(common + ["--output", str(full)])

    def test_layernorm_and_reject_unavailable_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")
            self.assertEqual(resolve_device("auto"), "cpu")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_splits(root / "splits")
            main(["--splits", str(root / "splits"), "--output", str(root / "out"),
                  "--lane", "light", "--normalization", "layer", "--epochs", "1",
                  "--batch-size", "32", "--threads", "1"])
            checkpoint = torch.load(root / "out/best.pt", weights_only=True)
            self.assertEqual(checkpoint["config"]["normalization"], "layer")
            result = json.loads((root / "out/result.json").read_text())
            self.assertEqual(result["examples_processed"], 129)


if __name__ == "__main__":
    unittest.main()
