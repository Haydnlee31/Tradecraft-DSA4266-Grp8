"""Bounded two-round correctness pilot; never provisions cloud resources.

Checks default FedAvg against FedProx(mu=0), then positive-mu uninterrupted vs
paused/resumed training on the supplied train/validation splits. Exact comparison
is intentionally strict: investigate a mismatch, do not relax tolerances blindly.
"""

import argparse
import json
import os
from pathlib import Path

import torch

from src.models.research import main as train, resolve_device


def compare(left, right):
    a = torch.load(left / "last.pt", map_location="cpu", weights_only=True)
    b = torch.load(right / "last.pt", map_location="cpu", weights_only=True)
    if a["step"] != 2 or b["step"] != 2 or a["model"].keys() != b["model"].keys():
        raise ValueError("Incomplete or incompatible pilot states")
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    # Compare both rounds, not just a possibly early selected checkpoint. Timing
    # and process memory are deliberately excluded from deterministic comparison.
    for ah, bh in zip(a["history"], b["history"]):
        for key in ("validation_metrics", "val_loss", "train_batch_mean_task_loss",
                    "train_batch_mean_proximal_penalty", "examples_processed", "optimizer_steps"):
            if ah[key] != bh[key]:
                raise ValueError(f"Pilot history mismatch: {key}")
    ar = json.loads((left / "result.json").read_text())
    br = json.loads((right / "result.json").read_text())
    if ar["validation_metrics"] != br["validation_metrics"]:
        raise ValueError("Pilot metric mismatch")
    if ar["test_metrics"] is not None or br["test_metrics"] is not None:
        raise ValueError("Unexpected test evaluation")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", type=Path, default=Path("data/splits"))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args(argv)
    if args.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching this pilot")
    # Fail before creating output or starting any paid compute work when inputs
    # or the explicitly requested GPU are unavailable.
    resolve_device(args.device)
    for split in ("train", "val"):
        if not (args.splits / f"{split}.parquet").is_file():
            raise FileNotFoundError(args.splits / f"{split}.parquet")
    args.output.mkdir(parents=True, exist_ok=False)
    common = ["--lane", "dirichlet", "--normalization", "layer", "--dropout", ".2",
              "--epochs", "2", "--seed", "7", "--partition-seed", "0", "--clients", "20",
              "--alpha", ".5", "--patience", "0", "--batch-size", "512", "--lr", ".001",
              "--weight-decay", ".00001", "--loss", "sqrt_weighted_ce", "--threads", "2",
              "--device", args.device, "--splits", str(args.splits)]
    avg, zero, full, resumed = (args.output / k for k in ("fedavg", "mu0", "mu01-full", "mu01-resumed"))
    train(common + ["--output", str(avg)])
    train(common + ["--output", str(zero), "--federated-method", "fedprox"])
    compare(avg, zero)  # Stop immediately on regression before positive-mu jobs.
    positive = ["--federated-method", "fedprox", "--proximal-mu", ".1"]
    train(common + positive + ["--output", str(full)])
    train(common + positive + ["--output", str(resumed), "--stop-after", "1"])
    if json.loads((resumed / "status.json").read_text())["status"] != "paused":
        raise ValueError("Expected a paused recovery state")
    train(common + positive + ["--output", str(resumed), "--resume"])
    compare(full, resumed)
    report = json.loads((full / "result.json").read_text())
    if not any(h["train_batch_mean_proximal_penalty"] > 0 for h in report["history"]):
        raise ValueError("Positive-mu pilot did not exercise a nonzero penalty")
    summary = {"device": args.device, "fedavg_mu0_exact": True, "positive_mu_resume_exact": True,
               "nonzero_penalty_observed": True, "test_evaluated": False,
               "next": "Review pilot before same-code FedAvg bridge; no sweep launched"}
    (args.output / "checks.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
