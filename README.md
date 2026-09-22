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

Simulated FedAvg over non-IID virtual clients (Flower). Requires the splits above.

```bash
python -m src.federated.run_simulation --rounds 15 --clients 5 --alpha 0.5
```

Every run prints its configuration and each client's class mix before training, so a
result can be traced to the partition that produced it. Useful flags:

| flag | meaning |
| --- | --- |
| `--alpha` | Dirichlet concentration. `0.1` = strongly non-IID, `100` ≈ IID. |
| `--clients` | Number of simulated sites. |
| `--local-epochs` | Local passes per round before aggregation. |
| `--max-rows` | Training rows sampled before partitioning (`0` = all ~936k). |
| `--history` | Write per-round metrics to JSON for the write-up. |
| `--backend` | `local` (default) or `ray`. |

### Backends

`--backend local` (default) runs the clients sequentially in one process and
aggregates with Flower's own `aggregate_arrayrecords`, so the FedAvg arithmetic is
Flower's. It needs no Ray and runs anywhere.

`--backend ray` uses Flower's full simulation runtime (process isolation, concurrent
clients, real message serialization). **This does not work on Windows with Smart App
Control enabled**: Ray launches native helpers (`raylet.exe`, `gcs_server.exe`) as
subprocesses and Windows blocks them with
`OSError: [WinError 4551] An Application Control policy has blocked this file`.
Note also that `pip install "flwr[simulation]"` installs *nothing* on Windows for
Python ≥3.13 — the extra's `ray` dependency is gated on `sys_platform != 'win32'` —
so the failure looks like a successful install.

To use the Ray backend, run it under WSL2 (Flower's own recommendation for Windows):

```bash
wsl --install                      # once, from PowerShell, then reboot
# inside WSL, from the repo directory:
python3 -m venv .venv-wsl && source .venv-wsl/bin/activate
pip install -r requirements.txt "flwr[simulation]"
python -m src.federated.run_simulation --backend ray --rounds 15 --clients 5
```

Both backends run the same client code and the same aggregation, so the learned
parameters agree; only isolation, concurrency and timing differ. Timing from either
is *not* a measured edge-hardware number (see CLAUDE.md).

### Metrics

Headline numbers come from the server's centralized pass over the shared validation
split — the same split all three lanes use, so it is the only comparable one. The
`federated (per-shard)` line is each client scoring the global model on its own
non-IID shard, which describes site-level behaviour, not project benchmarks.
Per-class recall is printed every round because macro-F1 alone hides rare-class
collapse: Brute Force and Web-based are surfaced explicitly for that reason.
