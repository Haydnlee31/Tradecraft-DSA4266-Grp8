# Edge Sentinel

IoT network intrusion classification on CICIoT2023, comparing centralized-heavy,
centralized-light, and federated-light training/deployment settings, with SHAP-based
explainability translated into security-policy recommendations.

See [CLAUDE.md](CLAUDE.md) for full project context, locked scope decisions, and the
link to the complete narrative plan.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Getting the data

1. Create a Kaggle API token at kaggle.com/settings/api and save it to
   `~/.kaggle/access_token` (new token-based auth) or `~/.kaggle/kaggle.json`
   (legacy username+key) — either is accepted.
2. Download the raw CSVs:
   ```bash
   python -m src.data.download_ciciot
   ```
   This pulls the `himadri07/ciciot2023` Kaggle mirror, which — confirmed by
   inspection — ships pre-split into `train.csv`/`validation.csv`/`test.csv`
   (~7.84M rows total, a downsampled/merged version of the official
   ~46M-row release, not the original 169-file layout). All 34 raw labels
   (33 attacks + BenignTraffic) are present in every split and match
   `src/data/label_map.py` exactly.
3. Build a stratified sample per split (capped per raw label, seeded):
   ```bash
   python -m src.data.sample_dataset --per-class-cap 50000 --seed 0
   ```
   Since the mirror's train/val/test boundary is already fixed upstream,
   this samples independently within each split rather than re-splitting,
   and writes straight to `data/splits/{train,val,test}.parquet` plus
   `data/sampled/manifest.csv` for provenance.
4. Sanity-check for leakage (identical flow rows appearing in more than one split):
   ```bash
   python -m src.data.check_leakage
   ```

Raw and derived data are gitignored — `data/raw/`, `data/sampled/`, and `data/splits/`
are populated locally by the commands above, not committed.
