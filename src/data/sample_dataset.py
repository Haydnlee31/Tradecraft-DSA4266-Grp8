"""Stratified, chunked sampling of the CICIoT2023 Kaggle mirror.

The `himadri07/ciciot2023` mirror ships pre-split into train/validation/test
(one merged CSV per split, ~7.84M rows total — a downsampled/merged version
of the official ~46M-row release, per CLAUDE.md's caveat about this mirror).
Confirmed via inspection: all 34 raw labels (33 attacks + BenignTraffic) are
present in every split and match src/data/label_map.py exactly.

Because the split boundary is already fixed upstream, this script respects
it rather than re-splitting: for each split file, it samples independently
and stratified by raw label (capped at --per-class-cap rows per raw label
per split), using polars' lazy/streaming engine so the full ~5.5M-row
train.csv is never loaded into memory at once. Output goes straight to
data/splits/{train,val,test}.parquet with a manifest for provenance.

Usage:
    python -m src.data.sample_dataset --per-class-cap 50000 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.data.label_map import LABEL_TO_CLASS, to_class

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "CICIOT23"
SPLITS_DIR = ROOT / "data" / "splits"
MANIFEST_DIR = ROOT / "data" / "sampled"

# mirror folder/file name -> output split name
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def _find_label_column(csv_path: Path) -> str:
    header = pl.scan_csv(csv_path, n_rows=0).collect_schema().names()
    for name in header:
        if name.strip().lower() == "label":
            return name
    raise ValueError(f"No label column found in {csv_path}: {header}")


def _count_labels(csv_path: Path, label_col: str) -> pl.DataFrame:
    return (
        pl.scan_csv(csv_path, schema_overrides={label_col: pl.Utf8})
        .select(label_col)
        .group_by(label_col)
        .len()
        .collect(engine="streaming")
        .rename({label_col: "raw_label", "len": "n_total_in_file"})
        # Group-by output order is not guaranteed. Sorting makes the output row
        # order and manifest stable across Polars versions/thread schedules.
        .sort("raw_label")
    )


def _sample_split(
    csv_path: Path, label_col: str, per_class_cap: int, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    counts = _count_labels(csv_path, label_col)

    bad_labels = set(counts["raw_label"]) - set(LABEL_TO_CLASS)
    if bad_labels:
        raise SystemExit(
            f"Unmapped raw labels found in {csv_path}: {sorted(bad_labels)}. "
            "Update src/data/label_map.py before sampling."
        )

    lf = pl.scan_csv(csv_path, schema_overrides={label_col: pl.Utf8})
    sampled_frames = []
    manifest_rows = []
    for row in counts.iter_rows(named=True):
        raw_label, n_total = row["raw_label"], row["n_total_in_file"]
        n_target = min(n_total, per_class_cap)
        sub = lf.filter(pl.col(label_col) == raw_label).collect(engine="streaming")
        sampled = (
            sub.sample(n=n_target, seed=seed, shuffle=True)
            if n_target > 0
            else sub.clear()
        )
        sampled_frames.append(sampled)
        manifest_rows.append(
            {
                "source_file": str(csv_path),
                "raw_label": raw_label,
                "class": to_class(raw_label),
                "n_total_in_file": n_total,
                "n_sampled_from_file": sampled.height,
                "per_class_cap": per_class_cap,
                "sampling_seed": seed,
            }
        )

    sampled_df = pl.concat(sampled_frames, how="vertical_relaxed").with_columns(
        pl.col(label_col).replace(LABEL_TO_CLASS).alias("class"),
        pl.lit(str(csv_path)).alias("source_file"),
    )
    return sampled_df, pl.DataFrame(manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-class-cap",
        type=int,
        default=50_000,
        help="Max sampled rows per raw label (34 labels), per split.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=MANIFEST_DIR,
        help="Directory for the sampling provenance manifest.",
    )
    args = parser.parse_args()

    if args.per_class_cap < 1:
        raise SystemExit("--per-class-cap must be >= 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)

    all_manifests = []
    for mirror_name, out_name in SPLITS.items():
        csv_path = args.raw_dir / mirror_name / f"{mirror_name}.csv"
        if not csv_path.exists():
            raise SystemExit(
                f"Expected {csv_path} not found. Run `python -m src.data.download_ciciot` first."
            )
        label_col = _find_label_column(csv_path)
        print(f"\n[{out_name}] sampling {csv_path} (label column: {label_col!r}) ...")
        sampled_df, manifest = _sample_split(
            csv_path, label_col, args.per_class_cap, args.seed
        )
        manifest = manifest.with_columns(pl.lit(out_name).alias("split"))
        all_manifests.append(manifest)

        out_path = args.out_dir / f"{out_name}.parquet"
        sampled_df.write_parquet(out_path)
        print(f"[{out_name}] wrote {sampled_df.height} rows -> {out_path}")
        print(sampled_df.group_by("class").len().sort("class"))

    manifest_path = args.manifest_dir / "manifest.csv"
    pl.concat(all_manifests).write_csv(manifest_path)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
