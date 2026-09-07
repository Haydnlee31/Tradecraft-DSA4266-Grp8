"""Download the CICIoT2023 Kaggle mirror (himadri07/ciciot2023) into data/raw/.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (Account -> Create New
API Token on kaggle.com). Usage:

    python -m src.data.download_ciciot
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATASET = "himadri07/ciciot2023"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def main() -> None:
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    if not token_path.exists():
        sys.exit(
            f"Missing Kaggle API token at {token_path}. "
            "Create one at kaggle.com -> Account -> Create New API Token, "
            "then place the downloaded kaggle.json there (chmod 600)."
        )
    if oct(token_path.stat().st_mode)[-3:] != "600":
        token_path.chmod(0o600)

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
