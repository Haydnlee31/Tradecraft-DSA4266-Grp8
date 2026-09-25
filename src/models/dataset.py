"""Shared data loading and preprocessing for all three training lanes.

Centralized-heavy, centralized-light, and federated-light must see the same
feature order, label indices, and train-fitted standardizer. This module owns
that contract. The fitted ``StandardScaler`` and a small provenance sidecar are
stored beside the parquet splits, then reused by every lane.

The scaler is fit on the *training* split only. Validation and test data are
never used to estimate preprocessing statistics because that would leak their
distribution into training. Fitting uses parquet record batches so a client
that starts before a centralized run does not need to load the full training
file just to create the shared scaler.

Why z-score scaling? CICIoT2023 features span very different scales (for
example, header lengths versus near-binary flag counts). Standardization makes
optimization much more stable. This is a simulation simplification: all
clients share global train statistics; a genuinely privacy-preserving system
would need to aggregate those statistics without centralizing them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.label_map import CLASSES

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits"
SCALER_FILENAME = "feature_scaler.joblib"
SCALER_METADATA_FILENAME = "feature_scaler.json"

META_COLS = frozenset(("label", "class", "source_file"))
CLASS_TO_IDX = {class_name: index for index, class_name in enumerate(CLASSES)}


def split_path(name: str, splits_dir: Path = SPLITS_DIR) -> Path:
    """Return a validated train/val/test parquet path."""
    if name not in {"train", "val", "test"}:
        raise ValueError(f"unknown split {name!r}; expected train, val, or test")
    path = Path(splits_dir) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the data pipeline first (see README.md)."
        )
    return path


def feature_columns_from_names(names: list[str]) -> list[str]:
    """Select model-input columns in their stable parquet order."""
    columns = [name for name in names if name not in META_COLS]
    if not columns:
        raise ValueError("no feature columns remain after removing metadata columns")
    return columns


def feature_columns(frame: pl.DataFrame) -> list[str]:
    """Select numeric feature columns and reject unexpected text features."""
    columns = feature_columns_from_names(frame.columns)
    non_numeric = [name for name in columns if not frame.schema[name].is_numeric()]
    if non_numeric:
        raise ValueError(f"non-numeric model feature columns: {non_numeric}")
    return columns


def class_counts(frame: pl.DataFrame) -> dict[str, int]:
    """Return counts in canonical class order, including absent classes as 0."""
    observed = {
        str(class_name): int(count)
        for class_name, count in frame.group_by("class").len().iter_rows()
    }
    unknown = set(observed) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown mapped classes: {sorted(unknown)}")
    return {class_name: observed.get(class_name, 0) for class_name in CLASSES}


def _source_signature(train_path: Path, feat_cols: list[str]) -> dict:
    stat = train_path.stat()
    return {
        "source_file": str(train_path.resolve()),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "feature_columns": feat_cols,
    }


def _metadata_matches(path: Path, expected: dict) -> bool:
    if not path.exists():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def load_or_fit_scaler(
    splits_dir: Path = SPLITS_DIR,
    feat_cols: list[str] | None = None,
    batch_rows: int = 100_000,
) -> tuple[StandardScaler, list[str]]:
    """Load the current train scaler or fit it incrementally and atomically.

    The metadata sidecar prevents a stale scaler from being reused after the
    sampled training parquet or feature order changes. Temporary files include
    the process ID so concurrent Flower workers never read a half-written file.
    """
    splits_dir = Path(splits_dir)
    train_path = split_path("train", splits_dir)
    schema = pl.scan_parquet(train_path).collect_schema()
    schema_names = schema.names()
    discovered = feature_columns_from_names(schema_names)
    bad = [name for name in discovered if not schema[name].is_numeric()]
    if bad:
        raise ValueError(f"non-numeric model feature columns: {bad}")
    if feat_cols is not None and feat_cols != discovered:
        raise ValueError(
            "requested feature order differs from train.parquet; refusing to "
            "apply a scaler to mismatched columns"
        )
    feat_cols = discovered

    scaler_path = splits_dir / SCALER_FILENAME
    metadata_path = splits_dir / SCALER_METADATA_FILENAME
    signature = _source_signature(train_path, feat_cols)
    if scaler_path.exists() and _metadata_matches(metadata_path, signature):
        scaler = joblib.load(scaler_path)
        if int(scaler.n_features_in_) != len(feat_cols):
            raise ValueError(
                f"{scaler_path} expects {scaler.n_features_in_} features, "
                f"but the parquet has {len(feat_cols)}"
            )
        return scaler, feat_cols

    scaler = StandardScaler()
    parquet = pq.ParquetFile(train_path)
    for record_batch in parquet.iter_batches(
        batch_size=batch_rows, columns=feat_cols, use_threads=True
    ):
        # A batch bounds peak memory while partial_fit produces the same global
        # mean/variance contract used by centralized and federated training.
        scaler.partial_fit(record_batch.to_pandas().to_numpy(copy=False))

    splits_dir.mkdir(parents=True, exist_ok=True)
    scaler_tmp = scaler_path.with_name(f"{scaler_path.name}.tmp{os.getpid()}")
    metadata_tmp = metadata_path.with_name(f"{metadata_path.name}.tmp{os.getpid()}")
    joblib.dump(scaler, scaler_tmp)
    samples_seen = np.asarray(scaler.n_samples_seen_)
    serialized_samples_seen: int | list[int]
    if samples_seen.ndim == 0:
        serialized_samples_seen = int(samples_seen)
    else:
        serialized_samples_seen = [int(value) for value in samples_seen]
    metadata_tmp.write_text(
        json.dumps(
            {
                **signature,
                "n_samples_seen": serialized_samples_seen,
                "fit_scope": "train_only",
                "method": "sklearn.StandardScaler.partial_fit",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(scaler_tmp, scaler_path)
    os.replace(metadata_tmp, metadata_path)
    return scaler, feat_cols


def to_arrays(
    frame: pl.DataFrame,
    scaler: StandardScaler,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one frame using the shared feature order and label mapping."""
    missing = [name for name in [*feat_cols, "class"] if name not in frame.columns]
    if missing:
        raise ValueError(f"split is missing required columns: {missing}")
    unknown = set(frame["class"].unique()) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown mapped classes: {sorted(unknown)}")

    features = scaler.transform(frame.select(feat_cols).to_numpy()).astype(np.float32)
    # Future raw samples may contain extreme values. Failing closed to finite
    # tensors is safer than letting one inf/nan poison an entire training run.
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.fromiter(
        (CLASS_TO_IDX[name] for name in frame["class"].to_list()),
        dtype=np.int64,
        count=frame.height,
    )
    return features, labels


def build_dataloaders(
    batch_size: int = 512,
    num_workers: int = 0,
    seed: int = 0,
    splits_dir: Path = SPLITS_DIR,
) -> dict:
    """Build centralized loaders plus their shared preprocessing metadata."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    splits_dir = Path(splits_dir)
    scaler, feat_cols = load_or_fit_scaler(splits_dir)

    train_frame = pl.read_parquet(split_path("train", splits_dir))
    counts = class_counts(train_frame)
    loaders: dict[str, DataLoader] = {}
    for split_name, frame, shuffle in (
        ("train", train_frame, True),
        ("val", pl.read_parquet(split_path("val", splits_dir)), False),
        ("test", pl.read_parquet(split_path("test", splits_dir)), False),
    ):
        features, labels = to_arrays(frame, scaler, feat_cols)
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        loaders[split_name] = DataLoader(
            TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)),
            batch_size=batch_size,
            shuffle=shuffle,
            # BatchNorm cannot estimate variance from a one-row final batch.
            # Drop only in that exact case; keep all other partial batches.
            drop_last=shuffle and frame.height % batch_size == 1,
            num_workers=num_workers,
            generator=generator,
        )

    return {
        "loaders": loaders,
        "scaler": scaler,
        "scaler_path": splits_dir / SCALER_FILENAME,
        "class_counts": counts,
        "feature_columns": feat_cols,
        "num_features": len(feat_cols),
        "num_classes": len(CLASSES),
    }
