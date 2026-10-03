"""Validation-only MLP research runner with step-boundary crash recovery.

Use this entry point for new controlled experiments; src.experiment preserves
the historical study interface. This runner never loads or evaluates test data.
One central epoch and one full-participation FL round expose roughly the same
rows, but are not equal optimizer trajectories or computational budgets.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import joblib
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.baselines.run import metrics, sha256
from src.data.label_map import CLASSES
from src.federated.partition import build_partition, check_exact_cover
from src.federated.task import predict
from src.models.architectures import MLPClassifier, config_for_variant
from src.models.dataset import class_counts, load_or_fit_scaler, to_arrays
from src.models.losses import build_criterion
from src.models.train import Trainer
from src.utils.seed import client_seed, set_seed

ROOT = Path(__file__).resolve().parents[2]


def resolve_device(request):
    """Never silently fall back when CUDA was explicitly requested."""
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or install a matching CUDA build")
    return request


def atomic_save(value, path):
    """A killed write cannot replace the previous valid recovery checkpoint."""
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def atomic_json(value, path):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temp, path)


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def memory_metrics(device):
    # Unix reports a process lifetime high-water mark, NOT inference model RAM.
    # Windows has no resource module; leave unknown instead of fabricating it.
    peak = None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        pass
    return {"process_peak_rss_bytes": peak,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}


def make_loader(dataset, batch_size, seed=None, batch_norm=True):
    if batch_norm and seed is not None and len(dataset) < 2:
        raise ValueError("BatchNorm needs at least two training rows per client")
    return DataLoader(dataset, batch_size=batch_size, shuffle=seed is not None,
                      generator=torch.Generator().manual_seed(seed) if seed is not None else None,
                      drop_last=seed is not None and batch_norm and len(dataset) % batch_size == 1,
                      num_workers=0)


def provenance(args, device):
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(args).items()
                if k not in {"output", "resume", "stop_after"}}
    settings["resolved_device"] = device
    packages = {p: importlib.metadata.version(p) for p in
                ("torch", "flwr", "numpy", "polars", "scikit-learn")}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = status = None
    return {"settings": settings, "packages": packages,
            "git_commit": revision, "git_status": status,
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))},
            "split_sha256": {s: sha256(args.splits / f"{s}.parquet") for s in ("train", "val")},
            "platform": platform.platform(), "python": sys.version,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
            "edge_hardware_measurements": False}


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane", choices=["heavy", "light", "iid", "dirichlet"], required=True)
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--partition-seed", type=int, default=0)
    p.add_argument("--clients", type=int, default=20)
    p.add_argument("--alpha", type=float, default=.5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--loss", choices=["ce", "sqrt_weighted_ce", "weighted_ce"], default="sqrt_weighted_ce")
    p.add_argument("--normalization", choices=["batch", "layer"], default="batch")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int, default=0, help="Controlled pause after this absolute epoch/round; 0 runs normally")
    args = p.parse_args(argv)
    if min(args.epochs, args.clients, args.threads) < 1 or args.batch_size < 2:
        raise ValueError("epochs/clients/threads must be positive and batch size >=2")
    if min(args.seed, args.partition_seed, args.patience, args.stop_after) < 0:
        raise ValueError("seeds, patience and stop-after cannot be negative")
    if any(not np.isfinite(x) or x <= 0 for x in (args.lr, args.alpha)):
        raise ValueError("learning rate and alpha must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight decay must be finite and nonnegative")
    if args.dropout is not None and not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    return args


def main(argv=None):
    args = parse(argv)
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    manifest = provenance(args, device)
    if args.resume:
        previous = json.loads((args.output / "environment.json").read_text())
        for key in ("settings", "packages", "source_sha256", "split_sha256"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Refusing resume: {key} changed")
        manifest = previous
    else:
        args.output.mkdir(parents=True, exist_ok=False)

    scaler, features = load_or_fit_scaler(args.splits)
    if args.resume:
        if sha256(args.output / "scaler.joblib") != manifest["scaler_sha256"]:
            raise ValueError("Saved scaler changed")
        if features != manifest["feature_columns"]:
            raise ValueError("Feature order changed")
        # Resume uses the exact saved preprocessing artifact, not a potentially
        # replaced cache beside the shared split files.
        scaler = joblib.load(args.output / "scaler.joblib")
    datasets = {}
    counts = None
    for split in ("train", "val"):
        frame = pl.read_parquet(args.splits / f"{split}.parquet")
        if not frame.height:
            raise ValueError(f"Empty {split} split")
        datasets[split] = TensorDataset(*map(torch.from_numpy, to_arrays(frame, scaler, features)))
        if split == "train":
            counts = class_counts(frame)
    if not args.resume:
        joblib.dump(scaler, args.output / "scaler.joblib")

    variant = "heavy" if args.lane == "heavy" else "light"
    config = replace(config_for_variant(variant), normalization=args.normalization)
    if args.dropout is not None:
        config = replace(config, dropout=args.dropout)
    set_seed(args.seed)
    model = MLPClassifier(len(features), len(CLASSES), config)
    criterion = build_criterion(args.loss, counts)
    trainer = Trainer(model, criterion, lr=args.lr, weight_decay=args.weight_decay,
                      patience=args.patience, device=device)
    train_loader = make_loader(datasets["train"], args.batch_size, args.seed, args.normalization == "batch")
    val_loader = make_loader(datasets["val"], args.batch_size)
    parts = None
    if args.lane in {"iid", "dirichlet"}:
        n = len(datasets["train"])
        if args.clients > n // (2 if args.normalization == "batch" else 1):
            raise ValueError("Too many clients for training rows")
        partition_file = args.output / "assignments.npz"
        if args.resume:
            if sha256(partition_file) != manifest["partition_sha256"]:
                raise ValueError("Partition assignments changed")
            with np.load(partition_file) as saved:
                parts = [saved[f"client_{i}"] for i in range(args.clients)]
        else:
            if args.lane == "iid":
                parts = [np.arange(i, n, args.clients) for i in range(args.clients)]
            else:
                mapping = build_partition(args.splits / "train.parquet", args.output / "partitions",
                                          args.alpha, args.clients, args.partition_seed)
                frame = pl.read_parquet(mapping).sort("_row")
                parts = [frame.filter(pl.col("partition_id") == i)["_row"].to_numpy() for i in range(args.clients)]
            np.savez_compressed(partition_file, **{f"client_{i}": rows for i,rows in enumerate(parts)})
        check_exact_cover(parts, n)

    manifest.update({"classes": CLASSES, "feature_columns": features, "model_config": asdict(config),
                     "train_class_counts": counts, "scaler_sha256": sha256(args.output / "scaler.joblib"),
                     "partition_sha256": sha256(args.output / "assignments.npz") if parts is not None else None,
                     "row_counts": {k: len(v) for k,v in datasets.items()},
                     "checkpoint_selection": "strict maximum validation macro-F1; candidates start at step 1",
                     "validation_loss_definition": "unweighted sample-mean cross entropy",
                     "optimizer_state": "reset per client-round" if parts is not None else "persistent across epochs",
                     "local_epochs": 1 if parts is not None else None,
                     "fraction_train": 1.0 if parts is not None else None})
    if not args.resume:
        atomic_json(manifest, args.output / "environment.json")

    history, best_state, best_score, best_step, stale, start = [], None, -1., 0, 0, 1
    checkpoint_path = args.output / "last.pt"
    if args.resume:
        recovery = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(recovery["model"])
        trainer.optimizer.load_state_dict(recovery["optimizer"])
        train_loader.generator.set_state(recovery["loader_rng"])
        restore_rng(recovery["rng"])
        history, best_state = recovery["history"], recovery["best_state"]
        best_score, best_step, stale = recovery["best_score"], recovery["best_step"], recovery["stale"]
        start = recovery["step"] + 1

    def stopped():
        return args.patience > 0 and stale >= args.patience

    for step in range(start, args.epochs + 1):
        if stopped():
            break
        begun = time.perf_counter()
        if parts is None:
            train_loss, _ = trainer._run_epoch(train_loader, train=True)
            examples = len(train_loader.dataset) - (1 if train_loader.drop_last else 0)
            updates = len(train_loader)
        else:
            from flwr.app import ArrayRecord, MetricRecord, RecordDict
            from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords
            replies, examples, updates, loss_sum = [], 0, 0, 0.
            for client, rows in enumerate(parts):
                local_seed = client_seed(args.seed, client, step)
                set_seed(local_seed)
                local_model = MLPClassifier(len(features), len(CLASSES), config)
                local_model.load_state_dict(model.state_dict())
                local = Trainer(local_model, build_criterion(args.loss, counts), lr=args.lr,
                                weight_decay=args.weight_decay, device=device)
                local_loader = make_loader(Subset(datasets["train"], rows.tolist()), args.batch_size,
                                           local_seed, args.normalization == "batch")
                loss, _ = local._run_epoch(local_loader, train=True)
                processed = len(rows) - (1 if local_loader.drop_last else 0)
                examples += processed
                updates += len(local_loader)
                loss_sum += loss * processed
                replies.append(RecordDict({"arrays": ArrayRecord(local_model.cpu().state_dict()),
                                           "metrics": MetricRecord({"num-examples": len(rows)})}))
            # Dataset-count weighting preserves the historical FedAvg policy;
            # actual processed examples are reported separately for BN tails.
            model.load_state_dict(aggregate_arrayrecords(replies, "num-examples").to_torch_state_dict())
            train_loss = loss_sum / examples
        val_loss, truth, prediction = predict(model, val_loader, device)
        val = metrics(truth, prediction)
        if device == "cuda":
            torch.cuda.synchronize()
        history.append({"step": step, "val_loss": val_loss, "validation_metrics": val,
                        "train_batch_mean_task_loss": train_loss, "examples_processed": examples,
                        "optimizer_steps": updates, "elapsed_seconds": time.perf_counter()-begun,
                        **memory_metrics(device)})
        if val["macro_f1"] > best_score:
            best_score, best_step, stale = val["macro_f1"], step, 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        # Recovery state is saved every completed step, not just on improvement.
        # A mid-round failure repeats that round from the last completed one.
        atomic_save({"step": step, "model": model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                     "loader_rng": train_loader.generator.get_state(), "rng": rng_state(),
                     "history": history, "best_state": best_state, "best_score": best_score,
                     "best_step": best_step, "stale": stale}, checkpoint_path)
        atomic_json(history, args.output / "history.json")
        print(f"{args.lane} step={step} val_macro_f1={val['macro_f1']:.5f} device={device}", flush=True)
        if args.stop_after and step >= args.stop_after and step < args.epochs and not stopped():
            atomic_json({"status": "paused", "step": step}, args.output / "status.json")
            return

    # Recovering after a crash between last.pt and the final outputs is safe too.
    model.load_state_dict(best_state)
    val_loss, truth, prediction = predict(model, val_loader, device)
    best_path = args.output / "best.pt"
    atomic_save({"model_state_dict": best_state, "config": asdict(config), "num_features": len(features),
                 "num_classes": len(CLASSES), "feature_columns": features, "classes": CLASSES,
                 "seed": args.seed, "loss": args.loss, "best_step": best_step}, best_path)
    report = {"lane": args.lane, "seed": args.seed, "best_step": best_step, "steps": len(history),
              "validation_metrics": metrics(truth, prediction), "val_loss": val_loss, "test_metrics": None,
              "history": history, "checkpoint": "best.pt", "checkpoint_sha256": sha256(best_path),
              "manifest": "environment.json", "num_parameters": model.num_parameters(),
              "parameter_bytes": model.parameter_bytes(), "edge_hardware_measurements": False,
              "step_seconds_sum": sum(h["elapsed_seconds"] for h in history),
              "timing_scope": "training plus full validation; excludes setup/checkpoint IO",
              "examples_processed": sum(h["examples_processed"] for h in history),
              "optimizer_steps": sum(h["optimizer_steps"] for h in history), **memory_metrics(device)}
    atomic_json(report, args.output / "result.json")
    atomic_json({"status": "complete", "step": len(history)}, args.output / "status.json")


if __name__ == "__main__":
    main()
