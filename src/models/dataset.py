"""Shared data loading for the centralized-heavy and centralized-light lanes.

Both lanes train on the same pooled, standardized `data/splits/{train,val,test}.parquet`
splits produced by `src/data/sample_dataset.py` + `src/data/check_leakage.py` -- this
module is the one place that turns those parquet files into standardized tensors, so
Heavy/Light stay a controlled "same data, different model" comparison per CLAUDE.md.

The StandardScaler is fit on the *train* split only (never val/test, to avoid leaking
their distribution into preprocessing) and persisted to
`data/splits/feature_scaler.joblib` so the same fitted scaler can be reused across
Heavy/Light runs and reported on consistently.

Normalization choice: Lecture 2's "Input Normalization" slide lists L2-norm,
min-max, and L1-norm scaling as typical options, with a caution about losing the
meaning of absolute scale. This project uses z-score standardization (mean 0,
unit variance per feature) instead -- not one of the three listed, but the
standard choice for exactly the reason the slide is getting at: several features
here are extremely heavy-tailed (e.g. `Header_Length` ranges from 0 to ~9.9M), so
min-max scaling would let one outlier flow compress the entire rest of that
feature's range toward 0, and both min-max and L1/L2-norm scaling are sensitive to
those same outliers in a way z-score (centered on the mean, scaled by std) is not.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.label_map import CLASSES

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits"
SCALER_PATH = SPLITS_DIR / "feature_scaler.joblib"

META_COLS = ("label", "class", "source_file")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def _load_split(name: str) -> pl.DataFrame:
    path = SPLITS_DIR / f"{name}.parquet"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the data pipeline first (see README.md).")
    return pl.read_parquet(path)


def class_counts(df: pl.DataFrame) -> dict[str, int]:
    counts = df.group_by("class").len()
    return {row[0]: row[1] for row in counts.iter_rows()}


def _to_arrays(df: pl.DataFrame, scaler: StandardScaler, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = scaler.transform(df.select(feat_cols).to_numpy()).astype(np.float32)
    y = np.array([CLASS_TO_IDX[c] for c in df["class"].to_list()], dtype=np.int64)
    return X, y


def build_dataloaders(batch_size: int = 512, num_workers: int = 0) -> dict:
    """Loads all three splits, fits the scaler on train, and returns everything the
    training script needs: DataLoaders, the fitted scaler, train class counts (for
    loss weighting), the feature column order, and dimensionality."""
    train_df = _load_split("train")
    feat_cols = feature_columns(train_df)
    counts = class_counts(train_df)

    scaler = StandardScaler()
    scaler.fit(train_df.select(feat_cols).to_numpy())
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    loaders = {}
    for split, df, shuffle in (
        ("train", train_df, True),
        ("val", _load_split("val"), False),
        ("test", _load_split("test"), False),
    ):
        X, y = _to_arrays(df, scaler, feat_cols)
        loaders[split] = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    return {
        "loaders": loaders,
        "scaler": scaler,
        "class_counts": counts,
        "feature_columns": feat_cols,
        "num_features": len(feat_cols),
        "num_classes": len(CLASSES),
    }
