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
4. Sanity-check for leakage (identical flow rows appearing in more than one split):
   ```bash
   python -m src.data.check_leakage
   ```

Raw and derived data are gitignored — `data/raw/`, `data/sampled/`, and `data/splits/`
are populated locally by the commands above, not committed.

## Running the federated lane

Simulated federated training of the light MLP over virtual clients, with Flower
(`flwr run`). Requires the splits above. Settings live in `[tool.flwr.app.config]` in
`pyproject.toml` (strategy, rounds, loss, partitioner, alpha, seed, ...).

```bash
# 1. Build the non-IID partition for (alpha, clients, seed) -- once, before training
python -m src.federated.partition --alpha 0.5 --num-partitions 20 --seed 0
# 2. Train; NUM_NODES must equal --num-partitions
NUM_NODES=20 SEED=0 ALPHA=0.5 ./run_fed.sh
# IID baseline / FedAvg baseline / a one-round smoke test
NUM_NODES=20 PARTITIONER=iid ./run_fed.sh
NUM_NODES=20 ./run_fed.sh "strategy='fedavg'"
NUM_NODES=20 ./run_fed.sh "num-server-rounds=1"
# Full sweep: alpha in {0.1, 0.5, 1.0} + IID, seeds 0-2
NUM_NODES=20 ./run_sweep.sh
```

The simulated clients are arbitrary shards of the pooled data, not the 105 real
CICIoT2023 devices (device identity is not in the flow CSVs).

Each run writes `outputs/federated/<date>/<time>_seed<N>/` (run_info.json,
history.json, best_model.pt, test_metrics.json) and a summary JSON in `reports/` with
the same fields as the centralized results. W&B logging is off unless `use-wandb = true`.

### Windows

`flwr run` uses Flower's Ray simulation runtime. **This does not work on Windows with
Smart App Control enabled**: Ray launches native helpers (`raylet.exe`,
`gcs_server.exe`) as subprocesses and Windows blocks them with
`OSError: [WinError 4551] An Application Control policy has blocked this file`.
Note also that `pip install "flwr[simulation]"` installs *nothing* on Windows for
Python >=3.13 -- the extra's `ray` dependency is gated on `sys_platform != 'win32'` --
so the failure looks like a successful install.

Run it under WSL2 instead (Flower's own recommendation for Windows):

```bash
wsl --install                      # once, from PowerShell, then reboot
# inside WSL, from the repo directory:
python3 -m venv .venv-wsl && source .venv-wsl/bin/activate
pip install -r requirements.txt "flwr[simulation]"
NUM_NODES=20 ./run_fed.sh
```

Timing from a simulation is measured on the simulation host, *not* an edge-hardware
number (see CLAUDE.md).

### Metrics

Headline numbers come from the server's pass over the full validation split -- the
same split all three lanes use, so it is the only comparable one. It reports macro-F1
and per-class recall every round; the best round by val macro-F1 is then scored once
on test. Client-side eval (off by default, `fraction-evaluate`) is summarized as the
min/median/max of per-client macro-F1 -- how evenly clients are served -- never as a
substitute for the global macro-F1.
