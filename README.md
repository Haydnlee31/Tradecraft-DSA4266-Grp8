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

1. Create a Kaggle API token: kaggle.com -> Account -> Create New API Token.
   Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`.
2. Download the raw CSVs:
   ```bash
   python -m src.data.download_ciciot
   ```
3. Build a stratified sample with per-file provenance:
   ```bash
   python -m src.data.sample_dataset --per-class-cap 50000 --seed 0
   ```
4. Build seeded train/val/test splits:
   ```bash
   python -m src.data.make_splits --seed 0
   ```

Raw and derived data are gitignored — `data/raw/`, `data/sampled/`, and `data/splits/`
are populated locally by the commands above, not committed.
