"""Build a non-IID Dirichlet partition of train.parquet across N simulated clients.

Run once per (alpha, N, seed), before training, so the 105 (or N) clients don't each
recompute it:

    python -m src.federated.partition --alpha 0.5 --num-partitions 20 --seed 0

Partitions by the 8-class `class` column (CLAUDE.md: by attack category, not the 34
raw labels). For each class, client shares are drawn from Dirichlet(alpha) and that
class's rows are split accordingly -- small alpha gives each client a few dominant
classes, large alpha approaches the IID mix.

Same algorithm as flwr_datasets' DirichletPartitioner (self_balancing: a client that
already holds >= N_total/N rows gets no share of later classes; resample until every
partition has >= min_partition_size rows), written out here because it's ~15 lines
and avoids pulling in flwr_datasets + HF datasets for one function.

Clients are arbitrary shards of the pooled data, *not* the 105 real CICIoT2023
devices -- device identity isn't in the flow CSVs.

Outputs, in data/partitions/ (or --out-dir):
    dirichlet_a{alpha}_n{N}_s{seed}.parquet       _row -> partition_id, one row per train row
    dirichlet_a{alpha}_n{N}_s{seed}_counts.csv    client x class row counts (report evidence)
    dirichlet_a{alpha}_n{N}_s{seed}.json          parameters + source file provenance
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
MAX_ATTEMPTS = 10  # Same as DirichletPartitioner's default retry budget


def partition_filename(alpha: float, num_partitions: int, seed: int) -> str:
    """Mapping file name; task.py uses this to find the file a run needs."""
    return f"dirichlet_a{float(alpha)}_n{int(num_partitions)}_s{int(seed)}.parquet"


def dirichlet_split(
    classes: np.ndarray,
    num_partitions: int,
    alpha: float,
    seed: int,
    min_partition_size: int,
    self_balancing: bool = True,
) -> tuple[list[np.ndarray], int]:
    """Split row indices by per-class Dirichlet(alpha) shares.

    Returns (row indices per partition, attempts used). Raises if no attempt gives
    every partition at least min_partition_size rows.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(classes)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        parts: list[list[np.ndarray]] = [[] for _ in range(num_partitions)]
        sizes = np.zeros(num_partitions, dtype=np.int64)
        for c in CLASSES:
            idx = np.flatnonzero(classes == c)
            rng.shuffle(idx)
            p = rng.dirichlet(np.full(num_partitions, alpha))
            if self_balancing:
                p = p * (sizes < n_rows / num_partitions)
                p = p / p.sum()
            cuts = (np.cumsum(p) * len(idx)).astype(np.int64)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].append(chunk)
                sizes[k] += len(chunk)
        if sizes.min() >= min_partition_size:
            return [np.concatenate(p) for p in parts], attempt
    raise RuntimeError(
        f"No partition with every client >= {min_partition_size} rows after {MAX_ATTEMPTS} "
        f"attempts (alpha={alpha}, N={num_partitions}). Lower --min-partition-size or "
        "--num-partitions, or raise --alpha."
    )


def check_exact_cover(parts: list[np.ndarray], n_rows: int) -> None:
    """Fail unless the partitions are disjoint and cover every train row exactly once."""
    all_rows = np.concatenate(parts)
    if len(all_rows) != n_rows or not np.array_equal(np.sort(all_rows), np.arange(n_rows)):
        raise RuntimeError(
            f"Partition is not an exact cover of train: {len(all_rows)} assigned rows, "
            f"{len(np.unique(all_rows))} unique, {n_rows} train rows."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--num-partitions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-partition-size", type=int, default=10,
        help="Resample until every client has at least this many train rows "
        "(DirichletPartitioner's default is 10).",
    )
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--out-dir", type=Path, default=PARTITIONS_DIR)
    args = parser.parse_args()

    train_path = args.splits_dir / "train.parquet"
    # Only the class column: ~1M short strings, not the 46 feature columns
    classes = pl.read_parquet(train_path, columns=["class"])["class"].to_numpy()
    n_rows = len(classes)

    parts, attempts = dirichlet_split(
        classes, args.num_partitions, args.alpha, args.seed, args.min_partition_size
    )
    check_exact_cover(parts, n_rows)

    partition_id = np.empty(n_rows, dtype=np.uint32)
    for k, rows in enumerate(parts):
        partition_id[rows] = k
    mapping = pl.DataFrame(
        {"_row": np.arange(n_rows, dtype=np.uint32), "partition_id": partition_id}
    )

    counts = (
        pl.DataFrame({"partition_id": partition_id, "class": classes})
        .pivot(on="class", index="partition_id", values="class", aggregate_function="len")
        .fill_null(0)
    )
    for c in CLASSES:  # A class no client got would otherwise be missing as a column
        if c not in counts.columns:
            counts = counts.with_columns(pl.lit(0, dtype=pl.UInt32).alias(c))
    counts = (
        counts.select(["partition_id", *CLASSES])
        .with_columns(pl.sum_horizontal(CLASSES).alias("total"))
        .sort("partition_id")
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / partition_filename(args.alpha, args.num_partitions, args.seed)
    mapping.write_parquet(out_path)
    counts.write_csv(out_path.with_name(out_path.stem + "_counts.csv"))
    out_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "alpha": args.alpha,
                "num_partitions": args.num_partitions,
                "seed": args.seed,
                "min_partition_size": args.min_partition_size,
                "self_balancing": True,
                "attempts": attempts,
                "partition_by": "class",
                "source_file": str(train_path),
                "num_rows": n_rows,
                "partition_sizes": {"min": int(counts["total"].min()), "max": int(counts["total"].max())},
            },
            indent=2,
        )
    )

    print(f"Saved {out_path} ({n_rows} rows, {args.num_partitions} partitions, {attempts} attempt(s))")
    print(f"Partition sizes: min={counts['total'].min()}  max={counts['total'].max()}")
    for c in CLASSES:
        holders = int((counts[c] > 0).sum())
        print(f"  {c:<12} held by {holders}/{args.num_partitions} clients")


if __name__ == "__main__":
    main()
