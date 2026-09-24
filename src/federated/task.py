"""fed_ciciot: CICIoT2023 data loading, centralized-light MLP, and train/test for Flower."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.label_map import CLASSES
from src.models.architectures import LIGHT_CONFIG, MLPClassifier
from src.federated.partition import partition_filename
from src.models.losses import build_criterion

ROOT = Path(__file__).resolve().parents[2]
# Default for running outside Flower; under `flwr run` the packaged app copy has no data/,
# so configure() overrides these from the "splits-dir" run config.
SPLITS_DIR = ROOT / "data" / "splits"
TRAIN_PATH = SPLITS_DIR / "train.parquet"
VAL_PATH = SPLITS_DIR / "val.parquet"
TEST_PATH = SPLITS_DIR / "test.parquet"
SCALER_PATH = SPLITS_DIR / "feature_scaler.joblib"

# Train partitioning across clients, set from the run config by configure():
# "iid" = every num_partitions-th row; "dirichlet" = mapping file from partition.py
PARTITIONER = "iid"
DIRICHLET_ALPHA = 0.5
PARTITION_SEED = 0
PARTITION_FILE = ""  # Explicit mapping path; empty = derive from alpha/N/seed

META_COLS = frozenset(("label", "class", "source_file"))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_FEATURES = 46
NUM_CLASSES = len(CLASSES)

_scaler: StandardScaler | None = None  # Cache per process
_feat_cols: list[str] | None = None
_server_val: TensorDataset | None = None  # Full val split, loaded once for server eval
_class_counts: dict[str, int] | None = None  # Global train class counts, for loss weights


def configure(run_config) -> None:
    """Point the data paths at run_config["splits-dir"] (empty = keep the default) and
    read the partitioning settings."""
    global SPLITS_DIR, TRAIN_PATH, VAL_PATH, TEST_PATH, SCALER_PATH, _scaler, _feat_cols, _server_val, _class_counts
    global PARTITIONER, DIRICHLET_ALPHA, PARTITION_SEED, PARTITION_FILE
    PARTITIONER = str(run_config.get("partitioner", "iid"))
    if PARTITIONER not in ("iid", "dirichlet"):
        raise ValueError(f"unknown partitioner: {PARTITIONER}")
    DIRICHLET_ALPHA = float(run_config.get("dirichlet-alpha", DIRICHLET_ALPHA))
    PARTITION_SEED = int(run_config.get("seed", PARTITION_SEED))
    PARTITION_FILE = str(run_config.get("partition-file", "") or "")

    splits_dir = str(run_config.get("splits-dir", "") or "")
    if not splits_dir or Path(splits_dir) == SPLITS_DIR:
        return
    SPLITS_DIR = Path(splits_dir)
    TRAIN_PATH = SPLITS_DIR / "train.parquet"
    VAL_PATH = SPLITS_DIR / "val.parquet"
    TEST_PATH = SPLITS_DIR / "test.parquet"
    SCALER_PATH = SPLITS_DIR / "feature_scaler.joblib"
    _scaler = _feat_cols = _server_val = _class_counts = None


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


def partition_path(num_partitions: int) -> Path:
    """The Dirichlet mapping file for this run: run_config["partition-file"] if set,
    else <splits-dir>/../partitions/dirichlet_a{alpha}_n{N}_s{seed}.parquet -- resolved
    from splits-dir because the packaged Flower app has no copy of data/."""
    if PARTITION_FILE:
        return Path(PARTITION_FILE)
    return SPLITS_DIR.parent / "partitions" / partition_filename(
        DIRICHLET_ALPHA, num_partitions, PARTITION_SEED
    )


def _load_dirichlet_partition(partition_id: int, num_partitions: int) -> TensorDataset:
    """Load this client's train rows via a lazy semi-join on the partition mapping."""
    path = partition_path(num_partitions)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing partition file {path}. Build it first: python -m src.federated.partition "
            f"--alpha {DIRICHLET_ALPHA} --num-partitions {num_partitions} --seed {PARTITION_SEED}"
        )
    # Guard against a mapping built for a different train split or client count
    n_map, max_pid = pl.scan_parquet(path).select(pl.len(), pl.col("partition_id").max()).collect().row(0)
    n_train = pl.scan_parquet(TRAIN_PATH).select(pl.len()).collect().item()
    if n_map != n_train or max_pid != num_partitions - 1:
        raise RuntimeError(
            f"{path.name} maps {n_map} rows to {max_pid + 1} partitions, but train has "
            f"{n_train} rows and this run has {num_partitions} clients. Rebuild the partition."
        )

    rows = pl.scan_parquet(path).filter(pl.col("partition_id") == partition_id).select("_row")
    df = (
        pl.scan_parquet(TRAIN_PATH)
        .with_row_index("_row")
        .join(rows, on="_row", how="semi")
        .sort("_row")  # Fixed row order, so seeded shuffling is reproducible
        .select(_feat_cols + ["class"])
        .collect(engine="streaming")
    )
    return _to_tensors(df)


def load_data(partition_id: int, num_partitions: int, batch_size: int, seed: int = 0):
    """Load this client's train partition and val slice.

    Train follows PARTITIONER ("iid" modulo slice or "dirichlet" mapping). Val is
    always the modulo slice; the val result that counts is the server's full-val eval.
    `seed` drives the train loader's shuffle order via its own generator.
    """
    _init_scaler()
    if PARTITIONER == "dirichlet":
        train_ds = _load_dirichlet_partition(partition_id, num_partitions)
    else:
        train_ds = _load_slice(TRAIN_PATH, partition_id, num_partitions)
    trainloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    valloader = DataLoader(
        _load_slice(VAL_PATH, partition_id, num_partitions), batch_size=batch_size
    )
    return trainloader, valloader


def load_server_val(batch_size: int = 512):
    """Load the full val split for server-side global evaluation.

    Read from disk once per process and cached, so each round only rebuilds the
    (cheap) DataLoader. The test split stays untouched until final evaluation.
    """
    global _server_val
    if _server_val is None:
        _init_scaler()
        df = pl.read_parquet(VAL_PATH, columns=_feat_cols + ["class"])
        _server_val = _to_tensors(df)
    return DataLoader(_server_val, batch_size=batch_size)


def load_test(batch_size: int = 512):
    """Load the full test split. Only for the single final evaluation of the chosen
    model -- never for training or model selection."""
    _init_scaler()
    df = pl.read_parquet(TEST_PATH, columns=_feat_cols + ["class"])
    return DataLoader(_to_tensors(df), batch_size=batch_size)


def train_class_counts() -> dict[str, int]:
    """Per-class row counts of the full train split, streamed and cached per process.

    Global (not per-client) counts, so the loss weights match the centralized lane's
    build_criterion(loss, class_counts) exactly, and a class missing from one
    client's slice can't produce an infinite weight.
    """
    global _class_counts
    if _class_counts is None:
        counts = pl.scan_parquet(TRAIN_PATH).group_by("class").len().collect(engine="streaming")
        _class_counts = {row[0]: row[1] for row in counts.iter_rows()}
    return _class_counts


def loss_class_counts(mode: str, trainloader: DataLoader) -> dict[str, int]:
    """Class counts the loss weights are built from.

    "global": the whole train split (train_class_counts). Matches centralized-light
      exactly; a real deployment would need clients to share their class counts.
    "local": this client's own slice. Realistic for FL, but a client can have none of
      some class, so each count is floored at 1 to avoid a divide-by-zero weight.
    """
    if mode == "global":
        return train_class_counts()
    if mode == "local":
        labels = trainloader.dataset.tensors[1]
        counts = torch.bincount(labels, minlength=NUM_CLASSES).tolist()
        return {c: max(int(n), 1) for c, n in zip(CLASSES, counts)}
    raise ValueError(f"unknown class-weights mode: {mode}")


def train(model, trainloader, epochs, lr, device, loss_name: str = "ce", class_weights: str = "global"):
    """Train the model on the training set with the loss named by `loss_name`,
    weighted by `class_weights` counts ("global" or "local")."""
    model.to(device)
    criterion = build_criterion(loss_name, loss_class_counts(class_weights, trainloader)).to(device)
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


def predict(model, loader, device):
    """Run the model over a loader; returns (mean batch loss, y_true, y_pred)."""
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss += criterion(logits, y).item()
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(y.cpu())
    return loss / len(loader), torch.cat(targets).numpy(), torch.cat(preds).numpy()


def test(model, testloader, device):
    """Evaluate the model; returns (loss, accuracy, macro-F1)."""
    loss, targets, preds = predict(model, testloader, device)
    accuracy = float((preds == targets).mean())
    macro_f1 = float(
        f1_score(targets, preds, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    )
    return loss, accuracy, macro_f1
