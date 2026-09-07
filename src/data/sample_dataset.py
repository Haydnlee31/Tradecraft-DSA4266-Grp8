"""Stratified, chunked sampling of raw CICIoT2023 CSVs into data/sampled/.

Never loads the full dataset into memory (per CLAUDE.md). Two passes over
data/raw/*.csv using polars' lazy/streaming engine:

  1. Count rows per (source file, raw label) without materializing rows.
  2. For each raw label, sample up to --per-class-cap rows total, split
     proportionally across the files that contain it, with a seeded random
     draw so results are reproducible.

Writes data/sampled/sample.parquet (with a source_file column for
provenance) and data/sampled/manifest.csv (source_file, raw_label, class,
n_total_in_file, n_sampled_from_file).

Usage:
    python -m src.data.sample_dataset --per-class-cap 50000 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.data.label_map import LABEL_TO_CLASS, to_class

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SAMPLED_DIR = ROOT / "data" / "sampled"


def _find_label_column(csv_path: Path) -> str:
    header = pl.scan_csv(csv_path, n_rows=0).collect_schema().names()
    for name in header:
        if name.strip().lower() == "label":
            return name
    raise ValueError(f"No label column found in {csv_path}: {header}")


def _count_labels_per_file(csv_files: list[Path], label_col: str) -> pl.DataFrame:
    """Pass 1: per-file, per-raw-label row counts, via lazy streaming."""
    frames = []
    for f in csv_files:
        counts = (
            pl.scan_csv(f, schema_overrides={label_col: pl.Utf8})
            .select(label_col)
            .group_by(label_col)
            .len()
            .with_columns(pl.lit(str(f)).alias("source_file"))
            .collect(engine="streaming")
        )
        frames.append(counts)
    return pl.concat(frames).rename({label_col: "raw_label", "len": "n_total_in_file"})


def _plan_sample_fractions(counts: pl.DataFrame, per_class_cap: int) -> pl.DataFrame:
    """Decide, per raw label, what fraction of each file's rows to keep so
    the total sampled for that label is min(total_available, per_class_cap).
    """
    totals = counts.group_by("raw_label").agg(
        pl.col("n_total_in_file").sum().alias("n_total_label")
    )
    plan = counts.join(totals, on="raw_label").with_columns(
        (pl.min_horizontal(pl.lit(per_class_cap), pl.col("n_total_label")) / pl.col("n_total_label")).alias(
            "keep_fraction"
        )
    )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class-cap", type=int, default=50_000,
                         help="Max sampled rows per raw label (34 labels, not 8 classes).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=SAMPLED_DIR)
    args = parser.parse_args()

    csv_files = sorted(args.raw_dir.rglob("*.csv"))
    if not csv_files:
        raise SystemExit(
            f"No CSV files found under {args.raw_dir}. "
            "Run `python -m src.data.download_ciciot` first."
        )

    label_col = _find_label_column(csv_files[0])

    print(f"Found {len(csv_files)} CSV files. Label column: {label_col!r}.")
    print("Pass 1/2: counting rows per (file, label) ...")
    counts = _count_labels_per_file(csv_files, label_col)

    bad_labels = set(counts["raw_label"].unique()) - set(LABEL_TO_CLASS)
    if bad_labels:
        raise SystemExit(
            f"Unmapped raw labels found: {sorted(bad_labels)}. "
            "Update src/data/label_map.py before sampling."
        )

    plan = _plan_sample_fractions(counts, args.per_class_cap)
    print(plan.group_by("raw_label").agg(pl.col("n_total_in_file").sum().alias("n_total")).sort("raw_label"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    sample_frames = []

    print("Pass 2/2: sampling rows per file ...")
    for f in csv_files:
        file_plan = plan.filter(pl.col("source_file") == str(f))
        if file_plan.is_empty():
            continue
        lf = pl.scan_csv(f, schema_overrides={label_col: pl.Utf8})
        per_label_frames = []
        for row in file_plan.iter_rows(named=True):
            raw_label = row["raw_label"]
            frac = min(row["keep_fraction"], 1.0)
            sub = (
                lf.filter(pl.col(label_col) == raw_label)
                .collect(engine="streaming")
            )
            n_target = min(row["n_total_in_file"], round(row["n_total_in_file"] * frac))
            sampled = sub.sample(n=n_target, seed=args.seed, shuffle=True) if n_target > 0 else sub.clear()
            per_label_frames.append(sampled)
            manifest_rows.append(
                {
                    "source_file": str(f),
                    "raw_label": raw_label,
                    "class": to_class(raw_label),
                    "n_total_in_file": row["n_total_in_file"],
                    "n_sampled_from_file": sampled.height,
                }
            )
        if per_label_frames:
            file_sample = pl.concat(per_label_frames).with_columns(
                pl.lit(str(f)).alias("source_file")
            )
            sample_frames.append(file_sample)
        print(f"  {f.name}: sampled {sum(fr.height for fr in per_label_frames)} rows")

    final = pl.concat(sample_frames, how="vertical_relaxed")
    final = final.with_columns(pl.col(label_col).replace(LABEL_TO_CLASS).alias("class"))

    out_parquet = args.out_dir / "sample.parquet"
    final.write_parquet(out_parquet)

    manifest = pl.DataFrame(manifest_rows)
    manifest.write_csv(args.out_dir / "manifest.csv")

    print(f"\nWrote {final.height} rows to {out_parquet}")
    print(f"Manifest: {args.out_dir / 'manifest.csv'}")
    print(final.group_by("class").len().sort("class"))


if __name__ == "__main__":
    main()
