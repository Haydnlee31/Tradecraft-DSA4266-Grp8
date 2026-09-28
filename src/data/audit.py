"""Read-only dataset audit with content hashes and exact feature overlap.

Provenance and labels are excluded from duplicate keys: identical features
with different labels still expose the same input to train and evaluation.
Hash joins avoid materializing a potentially enormous many-to-many join.
"""

from pathlib import Path
import argparse
import hashlib
import json
import polars as pl
from src.data.label_map import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/full_run/data_audit.json")
    )
    args = parser.parse_args()
    result, keys = {}, {}
    for name in ("train", "val", "test"):
        path = args.splits / f"{name}.parquet"
        frame = pl.read_parquet(path)
        features = [
            c for c in frame.columns if c not in ("class", "label", "source_file")
        ]
        counts = dict(frame.group_by("class").len().iter_rows())
        assert set(counts) == set(CLASSES), (name, counts)
        assert len(features) == 46, features
        invalid = frame.select(
            [
                ((~pl.col(c).is_finite()) | pl.col(c).is_null()).sum().alias(c)
                for c in features
            ]
        ).row(0, named=True)
        assert sum(invalid.values()) == 0, (name, invalid)
        # Two independently seeded 64-bit hashes make accidental matches
        # negligible. These counts are an overlap screen, not device identity.
        keys[name] = frame.select(
            pl.struct(features).hash(seed=0).alias("h0"),
            pl.struct(features).hash(seed=1).alias("h1"),
        ).unique()
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        result[name] = {
            "rows": frame.height,
            "class_counts": counts,
            "features": features,
            "invalid_values": invalid,
            "unique_feature_vectors": keys[name].height,
            "sha256": digest,
        }
    result["cross_split_feature_overlap"] = {
        f"{a}_{b}": keys[a].join(keys[b], on=["h0", "h1"], how="semi").height
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                s: {
                    k: v
                    for k, v in d.items()
                    if k not in ("features", "invalid_values")
                }
                if isinstance(d, dict)
                else d
                for s, d in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
