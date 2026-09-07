"""Download the CICIoT2023 Kaggle mirror (himadri07/ciciot2023) into data/raw/.

Requires Kaggle credentials in one of the forms the `kaggle` CLI accepts:
  - ~/.kaggle/access_token (new token-based auth, kaggle.com/settings/api), or
  - ~/.kaggle/kaggle.json (legacy username+key), or
  - the KAGGLE_API_TOKEN env var.

Usage:
    python -m src.data.download_ciciot
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DATASET = "himadri07/ciciot2023"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _check_credentials() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    access_token = kaggle_dir / "access_token"
    kaggle_json = kaggle_dir / "kaggle.json"

    if access_token.exists():
        if oct(access_token.stat().st_mode)[-3:] != "600":
            access_token.chmod(0o600)
        return
    if kaggle_json.exists():
        if oct(kaggle_json.stat().st_mode)[-3:] != "600":
            kaggle_json.chmod(0o600)
        return
    if os.environ.get("KAGGLE_API_TOKEN"):
        return

    sys.exit(
        "No Kaggle credentials found. Generate a token at "
        "kaggle.com/settings/api and either save it to "
        f"{access_token} or set the KAGGLE_API_TOKEN env var."
    )


def main() -> None:
    _check_credentials()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} into {RAW_DIR} ...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(RAW_DIR),
            "--unzip",
        ],
        check=True,
    )
    csv_files = sorted(RAW_DIR.rglob("*.csv"))
    print(f"Done. {len(csv_files)} CSV files in {RAW_DIR}.")


if __name__ == "__main__":
    main()
