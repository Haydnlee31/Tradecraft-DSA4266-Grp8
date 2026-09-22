"""Load the sampled splits and carve them into non-IID virtual clients.

Replaces the notebook's teaching partition (three hardcoded class groups) with
Dirichlet partitioning, the standard non-IID scheme in the FL literature: for
each class, the rows are split across clients in proportions drawn from
Dir(alpha). Low alpha => each class concentrates on few clients (strongly
non-IID); high alpha => every client sees a near-identical class mix (close to
IID). One knob, reproducible from a seed, and it sweeps the whole spectrum,
which is what the federated-vs-centralized comparison needs.

Standardization note: feature means/stds are computed on the *training split as
a whole*, then shared with every client. A strict federated setup would have to
negotiate those statistics without centralizing them. This is a stated
simplification for the simulation, in the same spirit as CLAUDE.md's "federated
learning is simulated, not physically distributed" — do not describe the
resulting numbers as fully privacy-preserving.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.data.label_map import CLASSES

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits"

# Columns that are not model inputs: the raw label, the mapped 8-class label,
# and the provenance column written by sample_dataset.py.
NON_FEATURE_COLUMNS = {"label", "class", "source_file"}

CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class Standardizer:
    """Per-feature mean/std fitted on the training split."""

    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        # Two CICIoT2023 columns are constant across the sample, so an
        # unclamped divide would emit inf/nan for them.
        return (x - self.mean) / np.clip(self.std, 1e-6, None)


@dataclass
class SplitData:
    """A materialized split: standardized features, integer labels, feature names."""

    x: torch.Tensor
    y: torch.Tensor
    feature_names: list[str]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def class_counts(self) -> dict[str, int]:
        counts = torch.bincount(self.y, minlength=len(CLASSES)).tolist()
        return {name: int(count) for name, count in zip(CLASSES, counts)}


def feature_columns(frame: pl.DataFrame) -> list[str]:
    """Numeric model-input columns, in file order (46 for this dataset)."""
    return [
        name
        for name, dtype in zip(frame.columns, frame.dtypes)
        if name not in NON_FEATURE_COLUMNS and dtype.is_numeric()
    ]


def _read_split(split: str, max_rows: int | None, seed: int, splits_dir: Path) -> pl.DataFrame:
    path = splits_dir / f"{split}.parquet"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run `python -m src.data.sample_dataset` first "
            "(see README 'Getting the data')."
        )
    frame = pl.read_parquet(path)
    if max_rows is not None and frame.height > max_rows:
        # Stratified downsample: keep the class proportions of the full split
        # rather than taking a head(), which would be ordered by label.
        frame = frame.sample(n=max_rows, seed=seed, shuffle=True)
    return frame


def load_split(
    split: str,
    max_rows: int | None = None,
    seed: int = 0,
    standardizer: Standardizer | None = None,
    splits_dir: Path = SPLITS_DIR,
) -> tuple[SplitData, Standardizer]:
    """Load one split as tensors, fitting or reusing a Standardizer.

    Pass the train split's Standardizer when loading val/test — refitting on
    the evaluation split would leak its distribution into the scaling.
    """
    frame = _read_split(split, max_rows, seed, splits_dir)
    names = feature_columns(frame)
    x = frame.select(names).to_numpy().astype(np.float32)

    unknown = set(frame["class"].unique()) - set(CLASS_TO_ID)
    if unknown:
        raise SystemExit(
            f"Unmapped class values in {split}.parquet: {sorted(unknown)}. "
            "Expected the 8 categories in src/data/label_map.CLASSES."
        )
    y = np.asarray([CLASS_TO_ID[name] for name in frame["class"].to_list()], dtype=np.int64)

    if standardizer is None:
        standardizer = Standardizer(mean=x.mean(axis=0), std=x.std(axis=0))
    x = standardizer.apply(x)
    # Guard against any non-finite value surviving preprocessing; the raw rate
    # columns are large enough that a future sample could overflow float32.
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    data = SplitData(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        feature_names=names,
    )
    return data, standardizer


def dirichlet_partition(
    y: torch.Tensor, num_clients: int, alpha: float, seed: int
) -> list[np.ndarray]:
    """Split row indices across clients, non-IID by class mix.

    Returns one index array per client. Every client is guaranteed at least one
    row: with low alpha a client can otherwise draw an empty shard, which would
    crash local training rather than teach anything about federation.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    rng = np.random.default_rng(seed)
    labels = y.numpy()
    shards: list[list[np.ndarray]] = [[] for _ in range(num_clients)]

    for class_id in range(len(CLASSES)):
        index = np.where(labels == class_id)[0]
        if index.size == 0:
            continue
        rng.shuffle(index)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # cumulative cut points, dropping the final one (it is just len(index))
        cuts = (np.cumsum(proportions) * index.size).astype(int)[:-1]
        for client_id, piece in enumerate(np.split(index, cuts)):
            shards[client_id].append(piece)

    partitions = [
        np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64) for pieces in shards
    ]

    empty = [i for i, part in enumerate(partitions) if part.size == 0]
    if empty:
        donor = int(np.argmax([part.size for part in partitions]))
        if partitions[donor].size <= len(empty):
            raise SystemExit(
                f"Cannot give every client a row: {len(y)} rows across {num_clients} clients."
            )
        for client_id in empty:
            partitions[client_id] = partitions[donor][-1:]
            partitions[donor] = partitions[donor][:-1]

    for part in partitions:
        rng.shuffle(part)
    return partitions


@lru_cache(maxsize=2)
def _all_shards(
    num_partitions: int,
    max_rows: int | None,
    alpha: float,
    seed: int,
    splits_dir_str: str,
) -> tuple[SplitData, ...]:
    """Every client's shard, built from a single read of the training split.

    Cached at the whole-partition level rather than per client: the parquet is
    read and standardized once per worker process no matter how many clients
    that process serves. Caching per client would re-read the full split once
    per client on the first round.

    Arguments are plain scalars (not a config object) so they stay hashable for
    lru_cache and serializable across the simulation backend.
    """
    splits_dir = Path(splits_dir_str)
    train, _ = load_split("train", max_rows=max_rows, seed=seed, splits_dir=splits_dir)
    partitions = dirichlet_partition(train.y, num_partitions, alpha, seed)
    return tuple(
        SplitData(
            x=train.x[torch.from_numpy(index)],
            y=train.y[torch.from_numpy(index)],
            feature_names=train.feature_names,
        )
        for index in partitions
    )


def client_shard(
    partition_id: int,
    num_partitions: int,
    max_rows: int | None,
    alpha: float,
    seed: int,
    splits_dir_str: str,
) -> SplitData:
    """The training shard belonging to one simulated client."""
    shards = _all_shards(num_partitions, max_rows, alpha, seed, splits_dir_str)
    return shards[partition_id]


def describe_partitions(y: torch.Tensor, partitions: list[np.ndarray]) -> str:
    """Human-readable table of each client's class mix, for run logs."""
    lines = [f"{'client':>7}  {'rows':>8}  class counts ({', '.join(CLASSES)})"]
    for client_id, index in enumerate(partitions):
        counts = torch.bincount(y[torch.from_numpy(index)], minlength=len(CLASSES)).tolist()
        lines.append(f"{client_id:>7}  {index.size:>8}  {counts}")
    return "\n".join(lines)
