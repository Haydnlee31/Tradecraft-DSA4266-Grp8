"""Check (and optionally fix) exact-duplicate flow rows across
data/splits/{train,val,test}.parquet.

The CICIoT2023 Kaggle mirror ships its own fixed train/validation/test split
(see sample_dataset.py), so this project doesn't split the data itself — but
an identical flow row appearing in more than one split would still be
leakage worth catching before training. In practice a small overlap shows up
here, most likely identical flow-statistic vectors from repetitive flood
traffic (e.g. DDoS) rather than a pipeline bug, but it's dropped either way
per CLAUDE.md's "no flow leakage across splits" convention.

With --fix, train is kept as-is (authoritative) and overlapping rows are
dropped from val, then from test (against the now-deduped val).

Usage:
    python -m src.data.check_leakage [--fix]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"


def _load(name: str) -> pl.DataFrame:
    path = SPLITS_DIR / f"{name}.parquet"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `python -m src.data.sample_dataset` first.")
    return pl.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true",
                         help="Drop overlapping rows from val/test (train is authoritative) and overwrite the parquet files.")
    args = parser.parse_args()

    frames = {name: _load(name) for name in ("train", "val", "test")}
    feature_cols = [c for c in frames["train"].columns if c != "source_file"]

    def overlap_count(a: str, b: str) -> int:
        return frames[a].join(frames[b], on=feature_cols, how="inner").height

    print("Before:")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(f"  {a} <-> {b}: {overlap_count(a, b)} duplicate rows")

    if not args.fix:
        return

    frames["val"] = frames["val"].join(frames["train"], on=feature_cols, how="anti")
    frames["test"] = frames["test"].join(frames["train"], on=feature_cols, how="anti").join(
        frames["val"], on=feature_cols, how="anti"
    )

    print("\nAfter fix:")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(f"  {a} <-> {b}: {overlap_count(a, b)} duplicate rows")

    for name in ("val", "test"):
        path = SPLITS_DIR / f"{name}.parquet"
        frames[name].write_parquet(path)
        print(f"\n{name}: {frames[name].height} rows -> {path}")
        print(frames[name].group_by("class").len().sort("class"))


if __name__ == "__main__":
    main()
