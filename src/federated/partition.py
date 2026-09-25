"""Build reproducible non-IID partitions for the federated-light lane.

Run this once per ``(alpha, number of clients, seed)`` before training so the
virtual clients do not independently recompute different assignments::

    python -m src.federated.partition --alpha 0.5 --num-partitions 20 --seed 0

Rows are partitioned by the mapped eight-class ``class`` column, not by the 34
raw attack labels. For each class, client shares are drawn from a Dirichlet
distribution. Small alpha values produce strongly skewed/non-IID clients;
large values approach the pooled class mix.

This follows the idea used by Flower Datasets' ``DirichletPartitioner``:
clients at or above the average shard size stop receiving later classes and a
draw is retried until every client has enough rows. The implementation lives
here to keep the assignment explicit, reviewable, and independent of a second
dataset framework.

The clients are arbitrary simulated shards of pooled flows. They are *not* the
105 physical CICIoT2023 devices because the released flow CSVs contain no
device identity. The output mapping and metadata make that distinction and the
exact rows used by a run auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.data.label_map import CLASSES

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "splits"
PARTITIONS_DIR = ROOT / "data" / "partitions"
MAX_ATTEMPTS = 10  # Same retry budget as Flower's teaching partitioner.


def partition_filename(alpha: float, num_partitions: int, seed: int) -> str:
    """Return the stable mapping filename used by both builder and clients."""
    return f"dirichlet_a{float(alpha)}_n{int(num_partitions)}_s{int(seed)}.parquet"


def dirichlet_split(
    classes: np.ndarray,
    num_partitions: int,
    alpha: float,
    seed: int,
    min_partition_size: int,
    self_balancing: bool = True,
) -> tuple[list[np.ndarray], int]:
    """Split row indices using per-class ``Dirichlet(alpha)`` proportions.

    Returns ``(row_indices_per_partition, attempts_used)``. It raises instead
    of emitting empty/undersized clients because those would either crash a
    local DataLoader or make comparisons between runs misleading.
    """
    labels = np.asarray(classes)
    if labels.ndim != 1:
        raise ValueError("classes must be a one-dimensional array")
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if min_partition_size < 1:
        raise ValueError("min_partition_size must be >= 1")
    if len(labels) < num_partitions * min_partition_size:
        raise ValueError(
            f"{len(labels)} rows cannot give {num_partitions} clients at least "
            f"{min_partition_size} rows each"
        )

    unknown = set(np.unique(labels)) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown mapped classes: {sorted(unknown)}")

    rng = np.random.default_rng(seed)
    n_rows = len(labels)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        parts: list[list[np.ndarray]] = [[] for _ in range(num_partitions)]
        sizes = np.zeros(num_partitions, dtype=np.int64)

        for class_name in CLASSES:
            indices = np.flatnonzero(labels == class_name)
            if indices.size == 0:
                # The sampler normally preserves every class, but keeping this
                # branch makes the partitioner safe for small smoke datasets.
                continue
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(num_partitions, alpha))
            if self_balancing:
                proportions *= sizes < n_rows / num_partitions
                proportions /= proportions.sum()
            cuts = (np.cumsum(proportions) * len(indices)).astype(np.int64)[:-1]
            for partition_id, chunk in enumerate(np.split(indices, cuts)):
                parts[partition_id].append(chunk)
                sizes[partition_id] += len(chunk)

        if sizes.min() >= min_partition_size:
            merged = [
                np.concatenate(chunks).astype(np.int64, copy=False) for chunks in parts
            ]
            # Fixed row order makes later seeded DataLoader shuffling fully
            # reproducible even when Polars changes join execution details.
            for rows in merged:
                rows.sort()
            return merged, attempt

    raise RuntimeError(
        f"No partition gave every client >= {min_partition_size} rows after "
        f"{MAX_ATTEMPTS} attempts (alpha={alpha}, N={num_partitions}). Lower "
        "--min-partition-size/--num-partitions or raise --alpha."
    )


def check_exact_cover(parts: list[np.ndarray], n_rows: int) -> None:
    """Fail unless the shards are disjoint and cover every train row once."""
    if not parts:
        raise RuntimeError("partition list is empty")
    all_rows = np.concatenate(parts)
    if len(all_rows) != n_rows or not np.array_equal(
        np.sort(all_rows), np.arange(n_rows)
    ):
        raise RuntimeError(
            f"Partition is not an exact cover of train: {len(all_rows)} assigned "
            f"rows, {len(np.unique(all_rows))} unique, {n_rows} train rows."
        )


def build_partition(
    train_path: Path,
    out_dir: Path,
    alpha: float,
    num_partitions: int,
    seed: int,
    min_partition_size: int = 10,
) -> Path:
    """Build and persist one mapping plus its counts and provenance metadata."""
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path}. Run the data pipeline first (see README.md)."
        )

    # Only the class column is materialized here: about one million short
    # strings, rather than all 46 feature columns.
    classes = pl.read_parquet(train_path, columns=["class"])["class"].to_numpy()
    parts, attempts = dirichlet_split(
        classes=classes,
        num_partitions=num_partitions,
        alpha=alpha,
        seed=seed,
        min_partition_size=min_partition_size,
    )
    check_exact_cover(parts, len(classes))

    partition_ids = np.empty(len(classes), dtype=np.uint32)
    for partition_id, rows in enumerate(parts):
        partition_ids[rows] = partition_id

    mapping = pl.DataFrame(
        {
            "_row": np.arange(len(classes), dtype=np.uint32),
            "partition_id": partition_ids,
        }
    )

    # First count each (client, class) pair, then pivot. Using the class column
    # as both `on` and `values` is rejected by newer Polars versions.
    counts = (
        pl.DataFrame({"partition_id": partition_ids, "class": classes})
        .group_by(["partition_id", "class"])
        .len()
        .pivot(on="class", index="partition_id", values="len")
        .fill_null(0)
    )
    for class_name in CLASSES:
        if class_name not in counts.columns:
            counts = counts.with_columns(pl.lit(0).alias(class_name))
    counts = (
        counts.select(["partition_id", *CLASSES])
        .with_columns(pl.sum_horizontal(CLASSES).alias("total"))
        .sort("partition_id")
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / partition_filename(alpha, num_partitions, seed)
    mapping.write_parquet(out_path)
    counts.write_csv(out_path.with_name(out_path.stem + "_counts.csv"))

    source_stat = train_path.stat()
    metadata = {
        "alpha": alpha,
        "num_partitions": num_partitions,
        "seed": seed,
        "min_partition_size": min_partition_size,
        "self_balancing": True,
        "attempts": attempts,
        "partition_by": "class",
        "source_file": str(train_path.resolve()),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "num_rows": len(classes),
        "partition_sizes": {
            "min": int(counts["total"].min()),
            "max": int(counts["total"].max()),
        },
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--num-partitions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-partition-size",
        type=int,
        default=10,
        help="Retry until every client has at least this many train rows.",
    )
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--out-dir", type=Path, default=PARTITIONS_DIR)
    args = parser.parse_args()

    out_path = build_partition(
        train_path=args.splits_dir / "train.parquet",
        out_dir=args.out_dir,
        alpha=args.alpha,
        num_partitions=args.num_partitions,
        seed=args.seed,
        min_partition_size=args.min_partition_size,
    )
    metadata = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    counts = pl.read_csv(out_path.with_name(out_path.stem + "_counts.csv"))

    print(
        f"Saved {out_path} ({metadata['num_rows']} rows, "
        f"{metadata['num_partitions']} partitions, {metadata['attempts']} attempt(s))"
    )
    print(
        f"Partition sizes: min={metadata['partition_sizes']['min']}  "
        f"max={metadata['partition_sizes']['max']}"
    )
    for class_name in CLASSES:
        holders = int((counts[class_name] > 0).sum())
        print(f"  {class_name:<12} held by {holders}/{args.num_partitions} clients")


if __name__ == "__main__":
    main()
