"""Seeded, stratified train/val/test splits from data/sampled/sample.parquet.

Exact-duplicate rows are dropped before splitting so the same flow can't land
in two different splits. Splitting is stratified by the 8-class label so
each split preserves the sampled class balance.

Usage:
    python -m src.data.make_splits --train 0.7 --val 0.15 --test 0.15 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SAMPLED_DIR = ROOT / "data" / "sampled"
SPLITS_DIR = ROOT / "data" / "splits"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-path", type=Path, default=SAMPLED_DIR / "sample.parquet")
    parser.add_argument("--out-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit("--train + --val + --test must sum to 1.0")

    df = pl.read_parquet(args.sample_path)
    n_before = df.height
    df = df.unique(subset=[c for c in df.columns if c != "source_file"], keep="first")
    n_deduped = n_before - df.height
    if n_deduped:
        print(f"Dropped {n_deduped} exact-duplicate rows before splitting.")

    df = df.with_columns(pl.int_range(pl.len()).shuffle(seed=args.seed).over("class").alias("_rank"),
                          pl.len().over("class").alias("_class_n"))

    train_cut = (pl.col("_rank") < (pl.col("_class_n") * args.train).floor())
    val_cut = (pl.col("_rank") < (pl.col("_class_n") * (args.train + args.val)).floor())

    train_df = df.filter(train_cut).drop(["_rank", "_class_n"])
    val_df = df.filter(~train_cut & val_cut).drop(["_rank", "_class_n"])
    test_df = df.filter(~val_cut).drop(["_rank", "_class_n"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = args.out_dir / f"{name}.parquet"
        split_df.write_parquet(out_path)
        print(f"{name}: {split_df.height} rows -> {out_path}")
        print(split_df.group_by("class").len().sort("class"))


if __name__ == "__main__":
    main()
