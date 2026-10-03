"""Reproduce the full three-lane study on the existing sampled splits.

All runs use 30 epochs/rounds at most, patience 5, batch 512, Adam 1e-3,
weight decay 1e-5 and sqrt-weighted CE. The IID federated control separates
the effects of federation from the effects of class skew. Nothing resamples
the official upstream train/validation/test boundaries.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from src.data.label_map import CLASSES
from src.eval.metrics import compute_metrics
from src.federated import task
from src.federated.client_app import train_partition
from src.federated.partition import build_partition
from src.models.architectures import LIGHT_CONFIG
from src.utils.seed import set_seed


def package_versions():
    """Record optional Ray as absent instead of breaking core-only installs."""
    versions = {}
    for package in ("torch", "flwr", "ray", "polars", "numpy", "scikit-learn", "shap"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if package != "ray":
                raise
            versions[package] = None
    return versions


def evaluate(model, loader):
    loss, truth, prediction = task.predict(model, loader, "cpu")
    return loss, compute_metrics(truth, prediction)


def federated(args, seed, partitioner):
    """Full-participation sequential Flower aggregation, with val selection.

    This execution mode measures training behavior on one host. Network time,
    parallel scaling, privacy, and edge-device performance are not measured.
    """
    rc = {
        "seed": seed,
        "splits-dir": str(args.splits.resolve()),
        "partitioner": partitioner,
        "dirichlet-alpha": 0.5,
        "batch-size": 512,
        "local-epochs": 1,
        "learning-rate": 0.001,
        "loss": "sqrt_weighted_ce",
        "class-weights": "global",
        "weight-decay": 1e-5,
        "num-server-rounds": args.epochs,
        "patience": 5,
    }
    if partitioner == "dirichlet":
        rc["partition-file"] = str(
            build_partition(
                args.splits / "train.parquet",
                args.output / "partitions",
                0.5,
                args.clients,
                seed,
            ).resolve()
        )
    task.configure(rc)
    set_seed(seed)
    model = task.load_model()
    best_score, best_state, best_round, stale = -1.0, None, 0, 0
    history = []
    started = time.perf_counter()
    for rnd in range(1, args.epochs + 1):
        replies = []
        for client in range(args.clients):
            state, metrics = train_partition(
                model.state_dict(), client, args.clients, rc, rnd, 0.001
            )
            replies.append(
                RecordDict(
                    {"arrays": ArrayRecord(state), "metrics": MetricRecord(metrics)}
                )
            )
        state = aggregate_arrayrecords(replies, "num-examples").to_torch_state_dict()
        model.load_state_dict(state)
        val_loss, val = evaluate(model, task.load_server_val())
        history.append(
            {
                "round": rnd,
                "val_loss": val_loss,
                "val_macro_f1": val["macro_f1"],
                "val_per_class_recall": val["per_class_recall"],
            }
        )
        print(
            f"federated {partitioner} seed={seed} round={rnd} val_f1={val['macro_f1']:.4f}",
            flush=True,
        )
        if val["macro_f1"] > best_score:
            best_score, best_state, best_round, stale = (
                val["macro_f1"],
                copy.deepcopy(model.state_dict()),
                rnd,
                0,
            )
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(best_state)
    _, validation_metrics = evaluate(model, task.load_server_val())
    _, test_metrics = evaluate(model, task.load_test())
    tag = "a0.5" if partitioner == "dirichlet" else "iid"
    name = f"federated_light_sqrt_weighted_ce_{tag}_n{args.clients}_seed{seed}"
    checkpoint = args.output / "checkpoints" / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "feature_columns": task._feature_names(),
            "classes": CLASSES,
            "config": {
                "hidden_dims": list(LIGHT_CONFIG.hidden_dims),
                "dropout": LIGHT_CONFIG.dropout,
            },
            "num_features": len(task._feature_names()),
            "num_classes": len(CLASSES),
        },
        checkpoint,
    )
    report = {
        "lane": "federated",
        "variant": "light",
        "loss": rc["loss"],
        "config": {
            "hidden_dims": list(LIGHT_CONFIG.hidden_dims),
            "dropout": LIGHT_CONFIG.dropout,
        },
        "num_parameters": model.num_parameters(),
        "parameter_bytes": model.parameter_bytes(),
        "history": history,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "args": rc,
        "partitioner": partitioner,
        "alpha": 0.5 if partitioner == "dirichlet" else None,
        "num_clients": args.clients,
        "local_epochs": 1,
        "fraction_train": 1.0,
        "class_weights": "global",
        "strategy": "FedAvg",
        "backend": "sequential-flower-aggregation",
        "best_round": best_round,
        "rounds": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
        "edge_hardware_measurements": False,
    }
    (args.output / f"{name}.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/full_run"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--lanes", nargs="+", default=["heavy", "light", "iid", "dirichlet"]
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.clients < 1 or args.threads < 1 or not args.seeds:
        raise SystemExit("epochs, clients, threads and seed count must be positive")
    if set(args.lanes) - {"heavy", "light", "iid", "dirichlet"}:
        raise SystemExit("lanes must be heavy, light, iid or dirichlet")
    torch.set_num_threads(args.threads)
    # Small tabular networks are often slower with eight BLAS threads per
    # minibatch. Keep the same CPU budget in each subprocess and FL client.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("train", "val", "test"):
        with (args.splits / f"{name}.parquet").open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "split_sha256": hashes,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "packages": package_versions(),
    }
    manifest_path = args.output / "environment.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        keys = ("epochs", "clients", "threads", "splits")
        if (previous["split_sha256"] != hashes
                or previous["packages"] != manifest["packages"]
                or any(previous["settings"].get(k) != manifest["settings"].get(k) for k in keys)):
            raise SystemExit("Existing output belongs to a different experiment; choose a new --output folder")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    for lane in args.lanes:
        for seed in args.seeds:
            # Completed reports are restart boundaries; interrupted runs never
            # masquerade as completed results. Use a new output folder to rerun.
            tag = "a0.5" if lane == "dirichlet" else "iid"
            filename = (
                f"centralized_{lane}_sqrt_weighted_ce_seed{seed}.json"
                if lane in ("heavy", "light")
                else f"federated_light_sqrt_weighted_ce_{tag}_n{args.clients}_seed{seed}.json"
            )
            if (args.output / filename).exists():
                print(f"Already complete: {filename}", flush=True)
                continue
            if lane in ("heavy", "light"):
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "src.models.train_centralized",
                        "--variant",
                        lane,
                        "--loss",
                        "sqrt_weighted_ce",
                        "--seed",
                        str(seed),
                        "--epochs",
                        str(args.epochs),
                        "--splits-dir",
                        str(args.splits),
                        "--reports-dir",
                        str(args.output),
                        "--checkpoint-dir",
                        str(args.output / "checkpoints"),
                    ],
                    check=True,
                )
            else:
                federated(args, seed, lane)


if __name__ == "__main__":
    main()
