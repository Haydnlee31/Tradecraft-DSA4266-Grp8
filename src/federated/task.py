"""Shared model, data, loss, and train/evaluate helpers for Flower clients.

The federated lane deliberately imports the centralized-light architecture,
loss factory, label order, and scaler contract instead of maintaining copies.
That is the core standardization guarantee: changing the light model or
preprocessing in one place changes both centralized-light and federated-light.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.label_map import CLASSES
from src.federated.partition import partition_filename
from src.models.architectures import LIGHT_CONFIG, MLPClassifier
from src.models.dataset import (
    SPLITS_DIR as DEFAULT_SPLITS_DIR,
    class_counts,
    feature_columns_from_names,
    load_or_fit_scaler,
    split_path,
    to_arrays,
)
from src.models.losses import build_criterion

ROOT = Path(__file__).resolve().parents[2]

# Defaults for importing/running outside Flower. Under ``flwr run`` the app is
# packaged separately, so configure() points these paths back to the repo data.
SPLITS_DIR = DEFAULT_SPLITS_DIR
TRAIN_PATH = SPLITS_DIR / "train.parquet"
VAL_PATH = SPLITS_DIR / "val.parquet"
TEST_PATH = SPLITS_DIR / "test.parquet"

# Training partition settings, overwritten from Flower's run config.
PARTITIONER = "iid"
DIRICHLET_ALPHA = 0.5
PARTITION_SEED = 0
PARTITION_FILE = ""

CLASS_TO_IDX = {class_name: index for index, class_name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

_scaler: StandardScaler | None = None
_feat_cols: list[str] | None = None
_server_val: TensorDataset | None = None
_test_data: TensorDataset | None = None
_class_counts: dict[str, int] | None = None


def configure(run_config: Mapping[str, Any]) -> None:
    """Apply data/partition settings and invalidate path-dependent caches."""
    global SPLITS_DIR, TRAIN_PATH, VAL_PATH, TEST_PATH
    global PARTITIONER, DIRICHLET_ALPHA, PARTITION_SEED, PARTITION_FILE
    global _scaler, _feat_cols, _server_val, _test_data, _class_counts

    partitioner = str(run_config.get("partitioner", "iid"))
    if partitioner not in {"iid", "dirichlet"}:
        raise ValueError(f"unknown partitioner: {partitioner}")
    PARTITIONER = partitioner
    DIRICHLET_ALPHA = float(run_config.get("dirichlet-alpha", DIRICHLET_ALPHA))
    PARTITION_SEED = int(run_config.get("seed", PARTITION_SEED))
    PARTITION_FILE = str(run_config.get("partition-file", "") or "")

    configured_dir = str(run_config.get("splits-dir", "") or "")
    new_splits_dir = Path(configured_dir) if configured_dir else SPLITS_DIR
    if new_splits_dir == SPLITS_DIR:
        return

    SPLITS_DIR = new_splits_dir
    TRAIN_PATH = SPLITS_DIR / "train.parquet"
    VAL_PATH = SPLITS_DIR / "val.parquet"
    TEST_PATH = SPLITS_DIR / "test.parquet"
    _scaler = None
    _feat_cols = None
    _server_val = None
    _test_data = None
    _class_counts = None


def _feature_names() -> list[str]:
    """Read the stable model-input order from the train parquet schema."""
    global _feat_cols
    if _feat_cols is None:
        path = split_path("train", SPLITS_DIR)
        names = pl.scan_parquet(path).collect_schema().names()
        _feat_cols = feature_columns_from_names(names)
    return _feat_cols


def load_model() -> MLPClassifier:
    """Return the exact canonical centralized-light MLP (64-32)."""
    return MLPClassifier(
        in_features=len(_feature_names()),
        num_classes=NUM_CLASSES,
        config=LIGHT_CONFIG,
    )


def _init_scaler() -> None:
    """Load or stream-fit the train-only scaler shared by every lane."""
    global _scaler, _feat_cols
    if _scaler is None:
        _scaler, _feat_cols = load_or_fit_scaler(SPLITS_DIR, _feature_names())


def _to_tensors(frame: pl.DataFrame) -> TensorDataset:
    _init_scaler()
    if _scaler is None or _feat_cols is None:  # Narrows types for static tools.
        raise RuntimeError("scaler initialization failed")
    features, labels = to_arrays(frame, _scaler, _feat_cols)
    return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))


def _load_modulo_slice(
    path: Path, partition_id: int, num_partitions: int
) -> TensorDataset:
    """Load every Nth row as a deterministic approximately-IID client slice.

    Sampled parquets are grouped by label, so taking every Nth row distributes
    each contiguous class block across all clients without loading the full
    file into each worker.
    """
    if not 0 <= partition_id < num_partitions:
        raise ValueError(f"partition_id {partition_id} outside [0, {num_partitions})")
    _init_scaler()
    frame = (
        pl.scan_parquet(path)
        .with_row_index("_row")
        .filter(pl.col("_row") % num_partitions == partition_id)
        .select([*_feature_names(), "class"])
        .collect(engine="streaming")
    )
    return _to_tensors(frame)


def partition_path(num_partitions: int) -> Path:
    """Resolve the explicit or convention-based Dirichlet mapping path."""
    if PARTITION_FILE:
        return Path(PARTITION_FILE)
    return (
        SPLITS_DIR.parent
        / "partitions"
        / partition_filename(DIRICHLET_ALPHA, num_partitions, PARTITION_SEED)
    )


def validate_partition_file(num_partitions: int) -> Path:
    """Fail early when a mapping does not belong to this train split/run."""
    path = partition_path(num_partitions)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing partition file {path}. Build it first: "
            "python -m src.federated.partition "
            f"--alpha {DIRICHLET_ALPHA} --num-partitions {num_partitions} "
            f"--seed {PARTITION_SEED} --splits-dir {SPLITS_DIR}"
        )

    summary = (
        pl.scan_parquet(path)
        .select(
            pl.len().alias("rows"),
            pl.col("partition_id").min().alias("min_pid"),
            pl.col("partition_id").max().alias("max_pid"),
            pl.col("partition_id").n_unique().alias("unique_pids"),
        )
        .collect()
        .row(0, named=True)
    )
    train_rows = pl.scan_parquet(TRAIN_PATH).select(pl.len()).collect().item()
    expected = {
        "rows": train_rows,
        "min_pid": 0,
        "max_pid": num_partitions - 1,
        "unique_pids": num_partitions,
    }
    if any(summary[key] != value for key, value in expected.items()):
        raise RuntimeError(
            f"{path.name} describes {summary}, but this run expects {expected}. "
            "Rebuild the partition mapping."
        )

    metadata_path = path.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stat = TRAIN_PATH.stat()
        if metadata.get("source_size_bytes") not in (
            None,
            stat.st_size,
        ) or metadata.get("source_mtime_ns") not in (None, stat.st_mtime_ns):
            raise RuntimeError(
                f"{path.name} was built for a different train.parquet. "
                "Rebuild the partition mapping."
            )
    return path


def _load_dirichlet_partition(partition_id: int, num_partitions: int) -> TensorDataset:
    """Load one train shard using a lazy semi-join on the persisted mapping."""
    if not 0 <= partition_id < num_partitions:
        raise ValueError(f"partition_id {partition_id} outside [0, {num_partitions})")
    _init_scaler()
    mapping_path = validate_partition_file(num_partitions)
    rows = (
        pl.scan_parquet(mapping_path)
        .filter(pl.col("partition_id") == partition_id)
        .select("_row")
    )
    frame = (
        pl.scan_parquet(TRAIN_PATH)
        .with_row_index("_row")
        .join(rows, on="_row", how="semi")
        .sort("_row")
        .select([*_feature_names(), "class"])
        .collect(engine="streaming")
    )
    if frame.height < 2:
        raise RuntimeError(
            f"client {partition_id} has {frame.height} training rows; "
            "BatchNorm training requires at least two"
        )
    return _to_tensors(frame)


def _train_loader(dataset: TensorDataset, batch_size: int, seed: int) -> DataLoader:
    if len(dataset) < 2:
        raise ValueError("a training client needs at least two rows")
    # BatchNorm cannot compute variance from a final batch of one. Only that
    # one-row remainder is dropped; other partial batches remain available.
    drop_last = len(dataset) % batch_size == 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(seed),
    )


def load_client_train(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    seed: int = 0,
) -> DataLoader:
    """Load one client's IID or persisted non-IID training shard."""
    _init_scaler()
    if PARTITIONER == "dirichlet":
        dataset = _load_dirichlet_partition(partition_id, num_partitions)
    else:
        dataset = _load_modulo_slice(TRAIN_PATH, partition_id, num_partitions)
    return _train_loader(dataset, batch_size, seed)


def load_client_val(
    partition_id: int, num_partitions: int, batch_size: int
) -> DataLoader:
    """Load a deterministic validation slice for optional site-level metrics."""
    _init_scaler()
    dataset = _load_modulo_slice(VAL_PATH, partition_id, num_partitions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Compatibility helper returning one client's train and val loaders."""
    return (
        load_client_train(partition_id, num_partitions, batch_size, seed=seed),
        load_client_val(partition_id, num_partitions, batch_size),
    )


def load_server_val(batch_size: int = 512) -> DataLoader:
    """Load/caches the full validation split for server-side model selection."""
    global _server_val
    if _server_val is None:
        _init_scaler()
        frame = pl.read_parquet(VAL_PATH, columns=[*_feature_names(), "class"])
        _server_val = _to_tensors(frame)
    return DataLoader(_server_val, batch_size=batch_size, shuffle=False)


def load_test(batch_size: int = 512) -> DataLoader:
    """Load/caches test only for the one final val-selected evaluation."""
    global _test_data
    if _test_data is None:
        _init_scaler()
        frame = pl.read_parquet(TEST_PATH, columns=[*_feature_names(), "class"])
        _test_data = _to_tensors(frame)
    return DataLoader(_test_data, batch_size=batch_size, shuffle=False)


def train_class_counts() -> dict[str, int]:
    """Return streamed global train counts for centralized-matched loss weights."""
    global _class_counts
    if _class_counts is None:
        frame = pl.read_parquet(TRAIN_PATH, columns=["class"])
        _class_counts = class_counts(frame)
    return _class_counts


def loss_class_counts(mode: str, trainloader: DataLoader) -> dict[str, int]:
    """Choose global or client-local class frequencies for weighted loss.

    Global counts make the loss exactly comparable to centralized-light. Local
    counts are closer to a strict FL setting but floor missing classes at one
    to avoid infinite weights.
    """
    if mode == "global":
        return train_class_counts()
    if mode == "local":
        labels = trainloader.dataset.tensors[1]
        counts = torch.bincount(labels, minlength=NUM_CLASSES).tolist()
        return {
            class_name: max(int(count), 1) for class_name, count in zip(CLASSES, counts)
        }
    raise ValueError(f"unknown class-weights mode: {mode}")


def train(
    model,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device,
    loss_name: str = "ce",
    class_weights: str = "global",
) -> float:
    """Train one client using the shared loss family and Adam optimizer."""
    if epochs < 1:
        raise ValueError("local epochs must be >= 1")
    model.to(device)
    criterion = build_criterion(
        loss_name, loss_class_counts(class_weights, trainloader)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss_sum = 0.0
    examples = 0
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * features.size(0)
            examples += features.size(0)
    if examples == 0:
        raise RuntimeError("client DataLoader produced no training batches")
    return loss_sum / examples


def predict(model, loader: DataLoader, device):
    """Return sample-mean CE loss, true labels, and predicted labels."""
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss_sum += float(criterion(logits, labels).item())
            predictions.append(logits.argmax(dim=1).cpu())
            targets.append(labels.cpu())
    if not targets:
        raise RuntimeError("evaluation DataLoader produced no batches")
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(predictions).numpy()
    return loss_sum / len(y_true), y_true, y_pred


def test(model, testloader: DataLoader, device) -> tuple[float, float, float]:
    """Compatibility helper returning loss, accuracy, and macro-F1."""
    from sklearn.metrics import f1_score

    loss, targets, predictions = predict(model, testloader, device)
    accuracy = float((predictions == targets).mean())
    macro_f1 = float(
        f1_score(
            targets,
            predictions,
            average="macro",
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )
    return loss, accuracy, macro_f1
