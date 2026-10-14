# Original teammate comments — Joel be0ea8e

Archived verbatim comment/docstring excerpts from incoming files whose implementations overlap the integrated pipeline. These historical notes are preserved for learning, NOT current API/default documentation. See MERGE_NOTES.md for the resolutions. Executable files retain their applicable explanations.

## README.md

~~~text
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

~~~

## pyproject.toml

~~~text
# Only needed with use-wandb = true
# 1.0 = every client trains every round, so 1 round x 1 local epoch ~ one pass over
# train -- keeps the round budget comparable to centralized-light's epochs
# 0 = no client-side eval; the server's full-val eval is the val result that counts
# "fedadagrad" (adaptive server optimizer) or "fedavg" (plain baseline)
# FedAdagrad only: server learning rate and adaptivity term (Flower's defaults, made explicit)
# Also log to W&B (falls back to offline mode if not logged in); history.json is always written
# Train split across clients: "dirichlet" (non-IID by class, mapping from
# src/federated/partition.py) or "iid" (every N-th row; the IID baseline)
# Explicit mapping path; empty = <splits-dir>/../partitions/dirichlet_a{alpha}_n{N}_s{seed}.parquet
# Loss class weights from "global" (whole train split; matches centralized) or
# "local" (each client's own counts, floored at 1)
~~~

## run_fed.sh

~~~text
#!/usr/bin/env bash
# Usage: [NUM_NODES=20] [SEED=0] [PARTITIONER=dirichlet|iid] [ALPHA=0.5] ./run_fed.sh [extra run-config pairs]
#   e.g. SEED=1 ALPHA=0.1 ./run_fed.sh "strategy='fedavg' num-server-rounds=1"
# NUM_NODES: number of simulated clients (Flower supernodes), default 105. Each client
#   gets one partition, so with partitioner=dirichlet it must equal the N the partition
#   file was built with (python -m src.federated.partition --num-partitions N). Clients
#   are arbitrary shards of the data, not the 105 real CICIoT2023 devices.
# SEED / PARTITIONER / ALPHA: passed through as seed / partitioner / dirichlet-alpha;
#   unset = pyproject.toml's value. Pairs given as arguments override them.
# Concurrency = INIT_CPUS / CLIENT_CPUS (default 4 / 2 = 2 clients at a time).
# Needs Flower's Ray runtime: on Windows with Smart App Control, run under WSL2 (see README).
~~~

## src/eval/metrics.py

~~~text
"""Evaluation metrics for the CICIoT2023 classifiers.

CLAUDE.md is explicit that bare accuracy is not the headline metric here -- with
DDoS/Mirai dominating and Brute Force/Web-based as thin slices (~130x imbalance),
a model can hit 95%+ accuracy while never learning the rare classes. macro-F1
(unweighted mean across classes) and per-class recall are what this module reports;
accuracy is kept only as a secondary reference number.
"""
"""Recall for each of the 8 classes, keyed by class name.

    Classes absent from y_true still appear, with recall 0.0 — a client shard
    under non-IID partitioning may legitimately hold none of a rare class, and
    silently dropping it from the report would hide exactly the failure mode
    the metric exists to catch.
    """
"""Flat metric dict: macro-F1, accuracy, and one recall entry per class.

    Flat (rather than nested) because Flower's MetricRecord only carries
    scalars, so this is what crosses the wire from client to server.
    """
"""Full sklearn text report, for logs and the write-up."""
"""One-line round summary: headline macro-F1 plus the rare-class recalls.

    Brute Force and Web-based are the classes most likely to collapse under
    federated averaging with skewed shards, so they are surfaced every round
    rather than only in the final report.
    """
~~~

## src/federated/__init__.py

~~~text
"""Federated-light lane: Flower client/server apps, strategy, and data partitioning."""
~~~

## src/federated/client_app.py

~~~text
"""fed_ciciot: Flower ClientApp for CICIoT2023 federated learning simulation."""
# Flower ClientApp
"""Train the model on local data."""
# Seed from (run seed, partition, round): distinct per client, independent of
# which worker runs it or in what order
# Load the model and initialize it with the received weights
# Load the data
# Call the training function
# Measured (simulation host): wall-clock seconds on the machine running the
# simulation. Not an edge-device figure -- edge latency/memory/power stay "projected".
# Construct and return reply Message
"""Evaluate the global model on this client's val slice.

    Reports this client's own macro-F1 (with its partition id) so the strategy can show
    how evenly clients are served. Not a substitute for the server's full-val macro-F1:
    a weighted average of per-client macro-F1 is a different number.
    """
# Load the model and initialize it with the received weights
# Load the data (val only; train isn't needed here)
# Call the evaluation function
# Construct and return reply Message
~~~

## src/federated/custom_strategy.py

~~~text
"""Server-side strategies for the federated-light lane.

`CustomFedAvg` (plain FedAvg baseline) and `CustomFedAdagrad` (adaptive server
optimizer) share one training loop via `_TradecraftStrategy`:

- model selection and early stopping on the server's full-val macro-F1, the same rule
  as the centralized Trainer.fit; the best weights go to best_model.pt/.json;
- a halving of the client learning rate every 5 rounds;
- per-round metrics always written to history.json, and to W&B only if enabled;
- client-side eval summarized as the spread (min/median/max) of per-client macro-F1,
  never as a weighted average -- that average is not the global macro-F1.
"""
"""Mixin holding the shared loop; combine with a Flower strategy class."""
"""`patience`: stop after this many rounds without a val macro-F1 improvement,
        mirroring Trainer.fit; 0 disables early stopping. `use_wandb`: also log to W&B."""
"""Configure the next round of federated training and maybe do LR decay."""
# Decrease learning rate by a factor of 0.5 every 5 rounds
# Clients derive their per-round seed from this; set it ourselves rather than
# rely on the parent strategy adding it
# Pass the updated config and the rest of arguments to the parent class
"""Summarize client-side eval as the spread of per-client macro-F1.

        Keeps each client's value (by partition id) for history.json; the returned
        record has weighted eval loss/accuracy plus min/median/max client macro-F1.
        """
"""Where checkpoints and history.json go, and the config recorded with W&B."""
# Optional dependency: only needed when use-wandb = true
"""Record a round's metrics in history (always) and W&B (if enabled)."""
"""Save a checkpoint if this round has the best val macro-F1 so far.
        Returns whether it improved.

        Same selection rule as the centralized Trainer.fit, so the lanes compare
        like for like. Overwrites best_model.pt / best_model.json each time.
        """
"""Execute the federated learning strategy, saving results to disk (and W&B
        if enabled)."""
# Keep track of best val macro-F1
# Initialize if None
# Evaluate starting global parameters
# -----------------------------------------------------------------
# --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
# -----------------------------------------------------------------
# Call strategy to configure training round
# Send messages and wait for replies
# Aggregate train
# Log training metrics and append to history
# -----------------------------------------------------------------
# --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
# -----------------------------------------------------------------
# Call strategy to configure evaluation round
# Send messages and wait for replies
# Aggregate evaluate
# Log evaluation metrics and append to history
# -----------------------------------------------------------------
# --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
# -----------------------------------------------------------------
# Centralized evaluation
# Maybe save to disk if new best is found
# Rewrite every round, so a crashed run still leaves its history
"""Plain FedAvg baseline."""
"""FedAdagrad (adaptive server optimizer over the averaged client update)."""
~~~

## src/federated/partition.py

~~~text
"""Build a non-IID Dirichlet partition of train.parquet across N simulated clients.

Run once per (alpha, N, seed), before training, so the 105 (or N) clients don't each
recompute it:

    python -m src.federated.partition --alpha 0.5 --num-partitions 20 --seed 0

Partitions by the 8-class `class` column (CLAUDE.md: by attack category, not the 34
raw labels). For each class, client shares are drawn from Dirichlet(alpha) and that
class's rows are split accordingly -- small alpha gives each client a few dominant
classes, large alpha approaches the IID mix.

Same algorithm as flwr_datasets' DirichletPartitioner (self_balancing: a client that
already holds >= N_total/N rows gets no share of later classes; resample until every
partition has >= min_partition_size rows), written out here because it's ~15 lines
and avoids pulling in flwr_datasets + HF datasets for one function.

Clients are arbitrary shards of the pooled data, *not* the 105 real CICIoT2023
devices -- device identity isn't in the flow CSVs.

Outputs, in data/partitions/ (or --out-dir):
    dirichlet_a{alpha}_n{N}_s{seed}.parquet       _row -> partition_id, one row per train row
    dirichlet_a{alpha}_n{N}_s{seed}_counts.csv    client x class row counts (report evidence)
    dirichlet_a{alpha}_n{N}_s{seed}.json          parameters + source file provenance
"""
# Same as DirichletPartitioner's default retry budget
"""Mapping file name; task.py uses this to find the file a run needs."""
"""Split row indices by per-class Dirichlet(alpha) shares.

    Returns (row indices per partition, attempts used). Raises if no attempt gives
    every partition at least min_partition_size rows.
    """
"""Fail unless the partitions are disjoint and cover every train row exactly once."""
# Only the class column: ~1M short strings, not the 46 feature columns
# A class no client got would otherwise be missing as a column
~~~

## src/federated/server_app.py

~~~text
"""fed_ciciot: Flower ServerApp for CICIoT2023 federated learning simulation."""
# Create ServerApp
"""The strategy named by run_config["strategy"], with its settings made explicit.

    Returns (strategy, settings dict for run_info.json).
    """
# Server-side learning rate
# Adaptivity / numerical-stability term
# Flower only records eta_l; clients actually train with "learning-rate"
# (halved every 5 rounds), so record that value rather than a dummy
"""Main entry point for the ServerApp."""
# Read run config
# Seed before building the model: fixes the initial weights and Flower's
# per-round client sampling
# Load global model
# FedAvg or FedAdagrad, per run_config["strategy"]
# outputs/federated/<date>/<time>_seed{N}/ at the repo root. The seed is in the
# name so runs with different seeds started in the same second don't collide.
# Log what this run used, per the repo's reproducibility convention
# Where checkpoints and history.json are saved
# Run the strategy for up to `num_rounds` (fewer if early stopping triggers)
# Save final model to disk
"""Write reports/federated_light_{strategy}_{loss}_{a<alpha>|iid}_n{clients}_seed{N}.json.

    Same fields as the centralized JSONs (variant, config, loss, num_parameters,
    history, test_metrics, args) so one script can tabulate all three lanes, plus
    FL-specific ones. `rounds` is the number actually run (after early stopping);
    with fraction-train = 1 and 1 local epoch, each round is ~one pass over train,
    so it is the counterpart of the centralized history's epoch count. The run
    folder's history.json has the full per-round record.
    """
# Measured (simulation host), not an edge-device figure
# FL-specific
# Strategy and partition setting in the name, so sweeps don't overwrite each other
"""Score the val-selected best_model.pt on the test split, once, after training.

    This is the only place test.parquet is read; training and model selection only
    ever see val. Writes the full compute_metrics output (per-class recall, confusion
    matrix, classification report), matching the centralized result files.
    """
"""Evaluate the global model on the full val split -- the numbers that count.

    Returns a flat MetricRecord (values must be numbers, not dicts): val_loss,
    val_accuracy, val_macro_f1 and one val_recall_<Class> per class in CLASSES.
    """
# Load the model and initialize it with the received weights
# Full val split (cached after the first round)
~~~

## src/federated/task.py

~~~text
"""fed_ciciot: CICIoT2023 data loading, centralized-light MLP, and train/test for Flower."""
# Default for running outside Flower; under `flwr run` the packaged app copy has no data/,
# so configure() overrides these from the "splits-dir" run config.
# Train partitioning across clients, set from the run config by configure():
# "iid" = every num_partitions-th row; "dirichlet" = mapping file from partition.py
# Explicit mapping path; empty = derive from alpha/N/seed
# Cache per process
# Loaded tensors keyed by (what, path, partition...). Simulation workers are reused
# across rounds, so each client slice / server split is read from disk once per worker
# Global train class counts, for loss weights
"""Point the data paths at run_config["splits-dir"] (empty = keep the default) and
    read the partitioning settings."""
# Scaled with the old scaler
"""Centralized-light MLP (64-32) so federated results compare directly to that lane."""
"""Load the train-fitted scaler, computing it from streamed train stats if absent.

    Never loads the full train split, so many concurrent clients stay light.
    """
# Atomic: parallel clients never read a half-written file
"""Return _tensor_cache[key], calling load() to fill it on a miss."""
"""Load every num_partitions-th row of a parquet split.

    The parquets are ordered by label, so this gives each client a balanced,
    deterministic sample of every class without loading the whole file.
    """
"""The Dirichlet mapping file for this run: run_config["partition-file"] if set,
    else <splits-dir>/../partitions/dirichlet_a{alpha}_n{N}_s{seed}.parquet -- resolved
    from splits-dir because the packaged Flower app has no copy of data/."""
"""Load this client's train rows via a lazy semi-join on the partition mapping."""
# Guard against a mapping built for a different train split or client count
# Fixed row order, so seeded shuffling is reproducible
"""This client's train rows, cached in memory after the first load.

    Follows PARTITIONER ("iid" modulo slice or "dirichlet" mapping). `seed` drives the
    loader's shuffle order via its own generator.
    """
"""This client's val rows (always the modulo slice), cached in memory.

    For client-side eval only; the val result that counts is the server's full-val eval.
    """
"""The full val split for server-side evaluation, cached in memory. The test split
    stays untouched until final evaluation."""
"""The full test split, cached in memory. Only for the single final evaluation of
    the chosen model -- never for training or model selection."""
"""Per-class row counts of the full train split, streamed and cached per process.

    Global (not per-client) counts, so the loss weights match the centralized lane's
    build_criterion(loss, class_counts) exactly, and a class missing from one
    client's slice can't produce an infinite weight.
    """
"""Class counts the loss weights are built from.

    "global": the whole train split (train_class_counts). Matches centralized-light
      exactly; a real deployment would need clients to share their class counts.
    "local": this client's own slice. Realistic for FL, but a client can have none of
      some class, so each count is floored at 1 to avoid a divide-by-zero weight.
    """
"""Train the model on the training set with the loss named by `loss_name`,
    weighted by `class_weights` counts ("global" or "local")."""
"""Run the model over a loader; returns (mean batch loss, y_true, y_pred).

    Scoring is left to src/eval/metrics.compute_metrics, so every lane computes its
    metrics the same way.
    """
~~~

## src/models/mlp.py

~~~text
"""Light MLP classifier for the 8-class CICIoT2023 task.

This is the "light" architecture used by both the centralized-light and
federated-light lanes (CLAUDE.md's three-lane comparison). Keeping it in one
place matters for the federated lane specifically: FedAvg averages parameter
tensors elementwise, so every client must instantiate a structurally
identical model or aggregation is meaningless.

Deliberately small — the point of the comparison is whether a model this size,
trained federated, is good enough to justify skipping a heavy centralized one.
Parameter counts here feed the efficiency/trade-off analysis; any latency or
power figure derived from them is *projected*, not measured (see CLAUDE.md).
"""
"""Fully-connected classifier over the 46 CICIoT2023 flow statistics.

    Args:
        input_features: number of numeric flow features (46 for this dataset).
        hidden_dims: widths of the hidden layers.
        num_classes: output classes (8 — see src/data/label_map.CLASSES).
        dropout: applied after each hidden activation; 0.0 disables it.
    """
"""Build a LightMLP with optionally seeded initialization.

    Every federated run is seeded (CLAUDE.md convention). The server seeds the
    initial global model once; clients receive those weights over the wire and
    never re-initialize, so client-side seeding only affects batch shuffling.
    """
"""Total parameter count — input to the model-size side of the trade-off table."""
"""Size of the parameter tensors in bytes.

    This is also the per-client upload volume of one FedAvg round, since a
    client returns exactly one full set of parameters per round.
    """
~~~

## requirements.txt

~~~text
# The federated code targets Flower's message API (flwr.serverapp /
# flwr.clientapp), introduced well after 1.8.
# Optional: only needed for `--backend ray`, Flower's full simulation runtime.
# `flwr[simulation]` pins ray==2.55.1 and excludes Windows on Python >=3.13, so
# it installs nothing there; ray is listed separately and left unpinned-by-extra
# for that reason. The default `--backend local` needs none of this.
# ray>=2.55
~~~

## run_sweep.sh

~~~text
#!/usr/bin/env bash
# Non-IID sweep: Dirichlet alpha in {0.1, 0.5, 1.0} plus the IID baseline, 3 seeds each.
# Builds each partition file first (skipped if it exists), then runs ./run_fed.sh.
# Usage: NUM_NODES=20 ./run_sweep.sh [extra run-config pairs, e.g. "strategy='fedavg'"]
~~~
