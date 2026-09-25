# Tradecraft

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
4. Check and remove exact duplicate rows across splits:
   ```bash
   python -m src.data.check_leakage --fix
   ```

Raw and derived data are gitignored — `data/raw/`, `data/sampled/`, and `data/splits/`
are populated locally by the commands above, not committed.

## Running the centralized lanes

Both variants use the same split files, train-only fitted scaler, label order,
training loop, loss factory, and report schema. The only intended difference is model
capacity.

```bash
python -m src.models.train_centralized --variant heavy --loss sqrt_weighted_ce --seed 0
python -m src.models.train_centralized --variant light --loss sqrt_weighted_ce --seed 0
```

Repeat the final chosen configuration for seeds 0, 1, and 2. Reports are written to
`reports/centralized_{heavy|light}_<loss>_seed<N>.json`; checkpoints go to
`models/checkpoints/` and are ignored by Git.

## Running the federated lane

Federated-light uses the exact same light architecture, scaler, loss, class order,
validation split, and JSON metric schema as centralized-light. The clients are
simulated data shards—not the 105 real devices, whose identities are absent from the
public CSVs.

Build a reproducible non-IID assignment, then run Flower:

```bash
python -m src.federated.partition --alpha 0.5 --num-partitions 20 --seed 0
NUM_NODES=20 ./run_fed.sh "dirichlet-alpha=0.5 seed=0"
```

`alpha=0.1` is strongly non-IID, `0.5` is moderately skewed, and larger values
approach IID. `./run_sweep.sh` runs alpha `{0.1, 0.5, 1.0}` plus an IID baseline for
three seeds. Full operating notes are in [src/federated/README.md](src/federated/README.md).

If Ray cannot run on the machine, this slower sequential smoke path exercises the
same client code and Flower aggregation helpers:

```bash
python -m src.federated.run_simulation --rounds 2 --clients 3 --alpha 0.5
```

The in-process runner is for debugging, not the final benchmark. Wall-clock time from
either simulation path is not measured edge-hardware latency, memory, or power.

## Building the decision summary

Once reports exist, the decision layer groups repeated seeds, selects configurations
using validation macro-F1 (not test performance), checks rare-class recall and model
size constraints, then compares the three lanes:

```bash
python -m src.eval.decision --reports-dir reports
```

It writes `reports/decision_summary.json` and `.md`. It deliberately returns an
`incomplete` decision when a lane or the required seed count is missing instead of
silently recommending from a partial comparison.

## Metrics and interpretation

The headline is macro-F1 together with all eight per-class recalls. Accuracy remains
secondary because a high-accuracy model can still collapse on Brute Force and
Web-based traffic. The server's full shared validation split is used for model
selection; test is read once for the validation-selected checkpoint. Client-shard
metrics describe local behavior and are not cross-lane benchmarks.
