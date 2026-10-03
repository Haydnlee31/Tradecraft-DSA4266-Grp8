"""Run classical baselines on the existing sampled parquet splits.

Logistic regression is a single linear layer, not a deep neural network even
though PyTorch trains it. Its three lanes share the exact training helper.
Federation is sequential, full-participation Flower FedAvg, not real devices.
XGBoost is centralized only. Test evaluation is explicitly opt-in.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset, Subset

from src.data.label_map import CLASSES
from src.eval.metrics import compute_metrics
from src.federated.partition import build_partition, check_exact_cover
from src.models.dataset import load_or_fit_scaler, to_arrays, class_counts
from src.models.losses import build_criterion
from src.utils.seed import set_seed, client_seed

ROOT = Path(__file__).resolve().parents[2]
LANES = ("logistic", "logistic-iid", "logistic-dirichlet", "xgboost")


class LogisticRegression(torch.nn.Linear):
    """Eight-class softmax regression: 46*8 + 8 = 376 parameters.

CrossEntropyLoss applies log-softmax internally, so forward returns logits.
No BatchNorm, hidden layers, activation or dropout; singleton batches are safe.
"""

    def __init__(self, num_features):
        super().__init__(num_features, len(CLASSES))


def metrics(truth, prediction):
    result = compute_metrics(truth, prediction)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=range(len(CLASSES)), zero_division=0
    )
    result["per_class"] = {
        name: dict(precision=float(p), recall=float(r), f1=float(f), support=int(n))
        for name, p, r, f, n in zip(CLASSES, precision, recall, f1, support)
    }
    benign = CLASSES.index("Benign")
    mask = np.asarray(truth) == benign
    result["benign_false_alert_rate"] = (
        float(np.mean(np.asarray(prediction)[mask] != benign)) if mask.any() else None
    )
    return result


def train_pass(model, loader, criterion, optimizer):
    """One pass shared by centralized and federated logistic regression."""
    model.train()
    examples = steps = 0
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        criterion(model(x), y).backward()
        optimizer.step()
        examples += len(y)
        steps += 1
    return examples, steps


def loader(dataset, batch_size, seed=None):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=seed is not None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        drop_last=False, num_workers=0,
    )


def evaluate(model, arrays, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in loader(TensorDataset(*map(torch.from_numpy, arrays)), batch_size):
            predictions.append(model(x).argmax(1).numpy())
    return metrics(arrays[1], np.concatenate(predictions))


def fit_logistic(args, seed, lane, arrays, counts, parts):
    # All lanes begin with the same weights for a given model seed.
    set_seed(seed)
    model = LogisticRegression(arrays["train"][0].shape[1])
    criterion = build_criterion(args.loss, counts)
    dataset = TensorDataset(*map(torch.from_numpy, arrays["train"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    central_loader = loader(dataset, args.batch_size, seed)
    history, best_state = [], None
    best, stale, best_step = -1., 0, 0
    total_examples = total_updates = 0
    for step in range(1, args.epochs + 1):
        if lane == "logistic":
            examples, updates = train_pass(model, central_loader, criterion, optimizer)
        else:
            # Import only on FL paths: centralized runs do not require Ray.
            from flwr.app import ArrayRecord, MetricRecord, RecordDict
            from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords
            replies, examples, updates = [], 0, 0
            for client, rows in enumerate(parts):
                local = copy.deepcopy(model)
                local_optimizer = torch.optim.Adam(
                    local.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )
                local_loader = loader(Subset(dataset, rows.tolist()), args.batch_size,
                                      client_seed(seed, client, step))
                n, u = train_pass(local, local_loader, criterion, local_optimizer)
                examples += n
                updates += u
                replies.append(RecordDict({
                    "arrays": ArrayRecord(local.state_dict()),
                    "metrics": MetricRecord({"num-examples": n}),
                }))
            # No failure is swallowed: every client must return before aggregation.
            model.load_state_dict(aggregate_arrayrecords(replies, "num-examples").to_torch_state_dict())
        total_examples += examples
        total_updates += updates
        val = evaluate(model, arrays["val"], args.batch_size)
        history.append({"step": step, "validation_metrics": val,
                        "examples_processed": examples, "optimizer_steps": updates})
        print(f"{lane} seed={seed} step={step} val_macro_f1={val['macro_f1']:.4f}", flush=True)
        if val["macro_f1"] > best:
            best, best_step, stale = val["macro_f1"], step, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if args.patience and stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, {"history": history, "best_step": best_step,
                   "examples_processed": total_examples, "optimizer_steps": total_updates,
                   "num_parameters": sum(p.numel() for p in model.parameters()),
                   "parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
                   "optimizer": "Adam", "optimizer_state": "persistent" if lane == "logistic" else "reset_each_round",
                   "strategy": None if lane == "logistic" else "FedAvg",
                   "backend": "cpu" if lane == "logistic" else "sequential-flower-aggregation"}


def fit_xgboost(args, seed, arrays, counts, checkpoint):
    """Use a fresh process to isolate Torch/XGBoost native OpenMP runtimes."""
    criterion = build_criterion(args.loss, counts)
    weights = criterion.weight.numpy() if criterion.weight is not None else np.ones(len(CLASSES))
    params = dict(objective="multi:softprob", num_class=len(CLASSES), tree_method="hist",
                  max_depth=args.max_depth, eta=args.xgb_lr, reg_lambda=args.xgb_lambda,
                  subsample=1., colsample_bytree=1., seed=seed, nthread=args.threads,
                  disable_default_eval_metric=1)
    with tempfile.TemporaryDirectory(prefix="tradecraft-xgb-") as directory:
        directory = Path(directory)
        payload = {f"{name}_{axis}": value for name, pair in arrays.items()
                   for axis, value in zip(("x", "y"), pair)}
        np.savez(directory / "arrays.npz", weights=weights, **payload)
        request = {"arrays": str(directory / "arrays.npz"), "params": params,
                   "rounds": args.boost_rounds, "patience": args.patience,
                   "checkpoint": str(checkpoint.resolve()), "result": str(directory / "result.json")}
        (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
        subprocess.run([sys.executable, "-m", "src.baselines.xgboost_worker",
                        str(directory / "request.json")], cwd=ROOT, check=True)
        return json.loads((directory / "result.json").read_text())


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", type=Path, default=ROOT / "data/splits")
    p.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
    p.add_argument("--lanes", nargs="+", choices=LANES, default=list(LANES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--partition-seed", type=int, default=0)
    p.add_argument("--clients", type=int, default=20)
    p.add_argument("--alpha", type=float, default=.5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5, help="0 disables early stopping")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--loss", choices=["ce", "weighted_ce", "sqrt_weighted_ce"], default="sqrt_weighted_ce")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--boost-rounds", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--xgb-lr", type=float, default=.05)
    p.add_argument("--xgb-lambda", type=float, default=1.)
    p.add_argument("--evaluate-test", action="store_true", help="Opt in only for a frozen final recipe")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("clients", "epochs", "batch_size", "threads", "boost_rounds", "max_depth"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if (args.patience < 0 or min(args.seeds + [args.partition_seed]) < 0
            or any(not np.isfinite(x) or x <= 0 for x in (args.lr, args.xgb_lr, args.alpha))
            or any(not np.isfinite(x) or x < 0 for x in (args.weight_decay, args.xgb_lambda))):
        raise ValueError("Invalid seed, patience, learning rate or regularization")
    if len(set(args.lanes)) != len(args.lanes) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate lanes/seeds would overwrite results")
    if "xgboost" in args.lanes:
        if importlib.util.find_spec("xgboost") is None:
            raise SystemExit('Install XGBoost with: python -m pip install -e ".[baselines]"')
    torch.set_num_threads(args.threads)
    # Refuse existing folders rather than silently resuming stale code/configs.
    args.output.mkdir(parents=True, exist_ok=False)
    names = ["train", "val"] + (["test"] if args.evaluate_test else [])
    hashes = {name: sha256(args.splits / f"{name}.parquet") for name in names}
    scaler, features = load_or_fit_scaler(args.splits)
    joblib.dump(scaler, args.output / "scaler.joblib")
    arrays, counts = {}, None
    for name in names:
        # Only sampled splits are materialized, never the full raw release.
        frame = pl.read_parquet(args.splits / f"{name}.parquet")
        if frame.height == 0:
            raise ValueError(f"Empty {name} split")
        arrays[name] = to_arrays(frame, scaler, features)
        if name == "train":
            counts = class_counts(frame)
    parts = {}
    n = len(arrays["train"][1])
    if any(lane.startswith("logistic-") for lane in args.lanes) and args.clients > n:
        raise ValueError("clients exceeds training rows")
    if "logistic-iid" in args.lanes:
        parts["logistic-iid"] = [np.arange(i, n, args.clients) for i in range(args.clients)]
    if "logistic-dirichlet" in args.lanes:
        mapping = build_partition(args.splits / "train.parquet", args.output / "partitions",
                                  args.alpha, args.clients, args.partition_seed)
        frame = pl.read_parquet(mapping).sort("_row")
        parts["logistic-dirichlet"] = [frame.filter(pl.col("partition_id") == i)["_row"].to_numpy()
                                       for i in range(args.clients)]
    for lane, assignments in parts.items():
        check_exact_cover(assignments, n)
        # Save exact positional row assignments for both IID and non-IID.
        np.savez_compressed(args.output / f"{lane}_partitions.npz",
                            **{f"client_{i}": rows for i, rows in enumerate(assignments)})
    def git(*arguments):
        try:
            return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    packages = {}
    for package in ("torch", "flwr", "polars", "numpy", "scikit-learn", "xgboost"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {"settings": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))},
                "split_sha256": hashes, "packages": packages, "platform": platform.platform(),
                "feature_columns": features, "classes": CLASSES, "train_class_counts": counts,
                "scaler_sha256": sha256(args.output / "scaler.joblib"),
                "partition_sha256": {p.name: sha256(p) for p in args.output.glob("*_partitions.npz")},
                "row_counts": {name: len(a[1]) for name,a in arrays.items()},
                "edge_hardware_measurements": False}
    (args.output / "environment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for lane in args.lanes:
        for seed in args.seeds:
            started = time.perf_counter()
            if lane == "xgboost":
                checkpoint = args.output / f"{lane}_seed{seed}.ubj"
                report = fit_xgboost(args, seed, arrays, counts, checkpoint)
            else:
                model, report = fit_logistic(args, seed, lane, arrays, counts, parts.get(lane))
                def score(a):
                    return evaluate(model, a, args.batch_size)
                checkpoint = args.output / f"{lane}_seed{seed}.pt"
                torch.save({"model_state_dict": model.state_dict(), "num_features": len(features),
                            "classes": CLASSES, "feature_columns": features, "seed": seed,
                            "loss": args.loss, "model_family": "logistic_regression"}, checkpoint)
                report["validation_metrics"] = score(arrays["val"])
                report["test_metrics"] = score(arrays["test"]) if args.evaluate_test else None
            report.update({"lane": lane, "seed": seed, "loss": args.loss,
                           "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
                           "checkpoint_file_bytes": checkpoint.stat().st_size,
                           "elapsed_seconds": time.perf_counter()-started,
                           "timing_scope": "fit, checkpoint and final evaluation; excludes data setup",
                           "manifest": "environment.json", "edge_hardware_measurements": False})
            (args.output / f"{lane}_seed{seed}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
