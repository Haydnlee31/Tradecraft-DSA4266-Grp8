"""fed_ciciot: CICIoT2023 data loading, centralized-light MLP, and train/test for Flower."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.label_map import CLASSES
from src.models.architectures import LIGHT_CONFIG, MLPClassifier

ROOT = Path(__file__).resolve().parents[2]
# Default for running outside Flower; under `flwr run` the packaged app copy has no data/,
# so configure() overrides these from the "splits-dir" run config.
SPLITS_DIR = ROOT / "data" / "splits"
TRAIN_PATH = SPLITS_DIR / "train.parquet"
VAL_PATH = SPLITS_DIR / "val.parquet"
TEST_PATH = SPLITS_DIR / "test.parquet"
SCALER_PATH = SPLITS_DIR / "feature_scaler.joblib"

META_COLS = frozenset(("label", "class", "source_file"))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_FEATURES = 46
NUM_CLASSES = len(CLASSES)

_scaler: StandardScaler | None = None  # Cache per process
_feat_cols: list[str] | None = None


def configure(run_config) -> None:
    """Point the data paths at run_config["splits-dir"] (empty = keep the default)."""
    global SPLITS_DIR, TRAIN_PATH, VAL_PATH, TEST_PATH, SCALER_PATH, _scaler, _feat_cols
    splits_dir = str(run_config.get("splits-dir", "") or "")
    if not splits_dir or Path(splits_dir) == SPLITS_DIR:
        return
    SPLITS_DIR = Path(splits_dir)
    TRAIN_PATH = SPLITS_DIR / "train.parquet"
    VAL_PATH = SPLITS_DIR / "val.parquet"
    TEST_PATH = SPLITS_DIR / "test.parquet"
    SCALER_PATH = SPLITS_DIR / "feature_scaler.joblib"
    _scaler = _feat_cols = None


def load_model() -> MLPClassifier:
    """Centralized-light MLP (64-32) so federated results compare directly to that lane."""
    return MLPClassifier(in_features=NUM_FEATURES, num_classes=NUM_CLASSES, config=LIGHT_CONFIG)


def _init_scaler() -> None:
    """Load the train-fitted scaler, computing it from streamed train stats if absent.

    Never loads the full train split, so many concurrent clients stay light.
    """
    global _scaler, _feat_cols
    if _scaler is not None:
        return

    names = pl.scan_parquet(TRAIN_PATH).collect_schema().names()
    _feat_cols = [c for c in names if c not in META_COLS]

    if not SCALER_PATH.exists():
        lf = pl.scan_parquet(TRAIN_PATH)
        mean = np.array(lf.select(_feat_cols).mean().collect(engine="streaming").row(0))
        std = np.array(lf.select(_feat_cols).std(ddof=0).collect(engine="streaming").row(0))
        n_rows = lf.select(pl.len()).collect(engine="streaming").item()
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = np.where(std == 0, 1.0, std)
        scaler.var_ = std**2
        scaler.n_features_in_ = len(_feat_cols)
        scaler.n_samples_seen_ = n_rows
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SCALER_PATH.with_suffix(f".tmp{os.getpid()}")
        joblib.dump(scaler, tmp)
        os.replace(tmp, SCALER_PATH)  # Atomic: parallel clients never read a half-written file

    _scaler = joblib.load(SCALER_PATH)


def _to_tensors(df: pl.DataFrame) -> TensorDataset:
    X = _scaler.transform(df.select(_feat_cols).to_numpy()).astype(np.float32)
    y = np.array([CLASS_TO_IDX[c] for c in df["class"].to_list()], dtype=np.int64)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


def _load_slice(path: Path, partition_id: int, num_partitions: int) -> TensorDataset:
    """Load every num_partitions-th row of a parquet split.

    The parquets are ordered by label, so this gives each client a balanced,
    deterministic sample of every class without loading the whole file.
    """
    df = (
        pl.scan_parquet(path)
        .with_row_index("_row")
        .filter(pl.col("_row") % num_partitions == partition_id)
        .select(_feat_cols + ["class"])
        .collect(engine="streaming")
    )
    return _to_tensors(df)


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load this client's slice of train.parquet (train) and val.parquet (val)."""
    _init_scaler()
    trainloader = DataLoader(
        _load_slice(TRAIN_PATH, partition_id, num_partitions), batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(
        _load_slice(VAL_PATH, partition_id, num_partitions), batch_size=batch_size
    )
    return trainloader, valloader


def load_centralized_dataset(batch_size: int = 512):
    """Load the full test split for server-side global evaluation."""
    _init_scaler()
    df = pl.read_parquet(TEST_PATH, columns=_feat_cols + ["class"])
    return DataLoader(_to_tensors(df), batch_size=batch_size)


def train(model, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    running_loss = 0.0
    for _ in range(epochs):
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / (epochs * len(trainloader))


def test(model, testloader, device):
    """Validate the model on the test set."""
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss += criterion(logits, y).item()
            correct += (logits.argmax(dim=1) == y).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
