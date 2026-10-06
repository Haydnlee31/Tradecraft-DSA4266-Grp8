"""FedProx math, isolation, FedAvg equivalence and exact CPU recovery."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.research import main, parse
from src.models.train import Trainer
from tests.test_baselines import synthetic_splits


class FedProxTests(unittest.TestCase):
    def test_pilot_preflight_does_not_create_output_on_failure(self):
        from src.eval.fedprox_pilot import main as pilot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                pilot(["--splits", str(root / "missing"), "--output", str(root / "out")])
            self.assertFalse((root / "out").exists())
            with patch.dict("os.environ", {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), \
                    patch("torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA requested"):
                    pilot(["--device", "cuda", "--output", str(root / "out")])
            self.assertFalse((root / "out").exists())

    def test_bounded_plan_commands_parse(self):
        from src.eval.tuning_plan import commands
        plan = json.loads(Path("configs/cloud_fedprox_plan.json").read_text())
        jobs = commands(plan, Path("outputs/test-fedprox-plan"))
        self.assertEqual(len(jobs), 3)
        parsed = [parse(job[3:]) for job in jobs]
        self.assertEqual([p.proximal_mu for p in parsed], [0, .01, .1])
        self.assertEqual([p.federated_method for p in parsed], ["fedavg", "fedprox", "fedprox"])
        self.assertTrue(all(p.seed == 7 and p.partition_seed == 0 and p.epochs == 60 for p in parsed))

    def test_penalty_value_gradient_and_detached_anchor(self):
        model = nn.Linear(2, 1)
        reference = {k: torch.zeros_like(p, requires_grad=True)
                     for k, p in model.named_parameters()}
        trainer = Trainer(model, nn.MSELoss(), device="cpu", proximal_mu=0.2,
                          proximal_reference=reference)
        with torch.no_grad():
            model.weight.fill_(2)
            model.bias.fill_(3)
        penalty = trainer._proximal_penalty()
        self.assertAlmostEqual(penalty.item(), 1.7, places=6)
        penalty.backward()
        torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 0.4))
        torch.testing.assert_close(model.bias.grad, torch.full_like(model.bias, 0.6))
        for value in reference.values():
            self.assertIsNone(value.grad)
            with torch.no_grad():
                value.fill_(100)
        self.assertAlmostEqual(trainer._proximal_penalty().item(), 1.7, places=6)

    def test_positive_penalty_changes_updates_but_not_validation_loss(self):
        torch.manual_seed(23)
        model = nn.Linear(3, 8)
        reference = {k: p.detach().clone() for k, p in model.named_parameters()}
        plain = Trainer(copy.deepcopy(model), nn.CrossEntropyLoss(), device="cpu", lr=.02)
        prox = Trainer(copy.deepcopy(model), nn.CrossEntropyLoss(), device="cpu", lr=.02,
                       proximal_mu=10, proximal_reference=reference)
        loader = DataLoader(TensorDataset(torch.randn(32, 3), torch.arange(32) % 8), batch_size=4)
        plain._run_epoch(loader, True)
        prox._run_epoch(loader, True)
        self.assertGreater(prox.last_epoch_proximal_penalty, 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(plain.model.parameters(), prox.model.parameters())))
        # Compare evaluation on the SAME weights, with/without a proximal target.
        plain.model.load_state_dict(prox.model.state_dict())
        self.assertEqual(plain._run_epoch(loader, False), prox._run_epoch(loader, False))
        self.assertEqual(prox.last_epoch_proximal_penalty, 0)

    def test_invalid_configuration_fails_closed(self):
        common = ["--lane", "dirichlet", "--output", "unused"]
        for value in ("-1", "nan", "inf"):
            with self.assertRaises(ValueError):
                parse(common + ["--federated-method", "fedprox", "--proximal-mu", value])
        with self.assertRaises(ValueError):
            parse(common + ["--proximal-mu", ".1"])
        with self.assertRaises(ValueError):
            parse(["--lane", "light", "--output", "unused", "--federated-method", "fedprox"])
        with self.assertRaises(ValueError):
            Trainer(nn.Linear(2, 8), nn.CrossEntropyLoss(), device="cpu", proximal_mu=.1)

    def test_zero_mu_exact_fedavg_and_positive_mu_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_splits(root / "splits")  # Deliberately has no test split.
            for lane in ("iid", "dirichlet"):
                common = ["--splits", str(root / "splits"), "--lane", lane,
                          "--epochs", "3", "--clients", "2", "--threads", "1",
                          "--batch-size", "16", "--normalization", "layer", "--device", "cpu"]
                dirs = {k: root / (lane + k) for k in ("avg", "zero", "prox", "resume")}
                main(common + ["--output", str(dirs["avg"])])
                main(common + ["--output", str(dirs["zero"]), "--federated-method", "fedprox"])
                positive = ["--federated-method", "fedprox", "--proximal-mu", ".1"]
                main(common + positive + ["--output", str(dirs["prox"])])
                main(common + positive + ["--output", str(dirs["resume"]), "--stop-after", "1"])
                for override in (["--proximal-mu", ".01"],
                                 ["--federated-method", "fedavg", "--proximal-mu", "0"]):
                    with self.assertRaisesRegex(ValueError, "settings changed"):
                        main(common + positive + ["--output", str(dirs["resume"]), "--resume"] + override)
                main(common + positive + ["--output", str(dirs["resume"]), "--resume"])
                for left, right in (("avg", "zero"), ("prox", "resume")):
                    a = torch.load(dirs[left] / "last.pt", weights_only=True)
                    b = torch.load(dirs[right] / "last.pt", weights_only=True)
                    for key in a["model"]:
                        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
                    for ah, bh in zip(a["history"], b["history"]):
                        for key in ("validation_metrics", "val_loss", "train_batch_mean_task_loss",
                                    "train_batch_mean_proximal_penalty"):
                            self.assertEqual(ah[key], bh[key])
                r = json.loads((dirs["prox"] / "result.json").read_text())
                self.assertEqual(r["federated_method"], "fedprox")
                self.assertEqual(r["proximal_mu"], .1)
                self.assertIsNone(r["test_metrics"])
                self.assertGreater(r["history"][0]["train_batch_mean_proximal_penalty"], 0)


if __name__ == "__main__":
    unittest.main()
