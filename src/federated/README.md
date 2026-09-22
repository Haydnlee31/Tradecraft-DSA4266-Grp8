# Federated lane — usage reference

Simulated FedAvg over non-IID virtual clients, for the **federated-light** lane of the
three-lane comparison in `CLAUDE.md`. This file is the working reference: how to run it,
what the output means, where the traps are.

Prerequisite: `data/splits/{train,val,test}.parquet` must exist. If not, see the
"Getting the data" section of the top-level [README](../../README.md) — the pipeline is
`download_ciciot` → `sample_dataset` → `check_leakage`.

---

## Quick start

```bash
# from the repo root, with .venv active
python -m src.federated.run_simulation --rounds 15 --clients 5 --alpha 0.5
```

A short smoke run (finishes in well under a minute):

```bash
python -m src.federated.run_simulation --rounds 3 --clients 3 --max-rows 8000 --eval-max-rows 4000
```

Save per-round metrics for the write-up:

```bash
python -m src.federated.run_simulation --rounds 20 --clients 5 \
    --history reports/fed_alpha05.json
```

---

## What the pieces are

| file | role |
| --- | --- |
| `partition.py` | Loads splits, standardizes features, carves non-IID client shards |
| `client_app.py` | Local training/evaluation for one client — the actual learning |
| `server_app.py` | Initial model, per-round config, FedAvg, centralized val scoring |
| `local_backend.py` | Sequential in-process execution driver (the default) |
| `run_simulation.py` | CLI entrypoint; picks a backend and wires everything together |

Supporting modules outside this folder: `src/models/mlp.py` (the light MLP),
`src/eval/metrics.py` (macro-F1 + per-class recall), `src/data/label_map.py`
(the 8-class mapping — never hardcode category strings anywhere else).

The split that matters architecturally: `local_train` / `local_evaluate` in
`client_app.py` take plain tensors and a config dict and know nothing about Flower.
The `@app.train()` / `@app.evaluate()` handlers are thin adapters that unwrap a Flower
`Message` and call them. Both backends therefore run byte-identical client code.

---

## Flags

| flag | default | meaning |
| --- | --- | --- |
| `--rounds` | 5 | Federated rounds (broadcast → local train → aggregate). |
| `--clients` | 3 | Number of simulated sites. |
| `--alpha` | 0.5 | Dirichlet concentration. `0.1` = strongly non-IID, `100` ≈ IID. |
| `--local-epochs` | 1 | Local passes over a client's shard per round. |
| `--batch-size` | 64 | Local SGD batch size. |
| `--lr` | 0.05 | Local SGD learning rate. |
| `--momentum` | 0.9 | Local SGD momentum. |
| `--seed` | 7 | Seeds partitioning, init, and batch shuffling. |
| `--max-rows` | 50000 | Training rows sampled before partitioning (`0` = all ~936k). |
| `--eval-max-rows` | 50000 | Rows sampled from val for centralized scoring (`0` = all). |
| `--hidden-dims` | `64,32` | Hidden layer widths of the light MLP. |
| `--history` | — | Path to write per-round metrics as JSON. |
| `--backend` | `local` | `local` or `ray` (see below). |
| `--splits-dir` | `data/splits` | Where the parquet splits live. |
| `--num-cpus` | 1.0 | CPU share per client — Ray backend only. |

### The `--alpha` knob

This is the main experimental dial. Dirichlet partitioning splits each class across
clients in proportions drawn from `Dir(alpha)`:

- `--alpha 0.1` — each class concentrates on one or two clients. Clients disagree
  strongly; expect client drift and worse rare-class recall.
- `--alpha 0.5` — moderate skew. Reasonable default.
- `--alpha 100` — every client sees roughly the dataset's class mix. Close to IID, and
  close to a centralized run split across machines.

Sweeping alpha at fixed everything-else is the cleanest evidence for "how much does
non-IID data cost us", which is the question the federated lane exists to answer.

---

## Backends

### `local` (default)

Clients run sequentially in one process. Aggregation calls Flower's own
`aggregate_arrayrecords`, so the FedAvg arithmetic is Flower's, not a second
implementation that could silently drift.

It does **not** reproduce process isolation, concurrent clients, message
serialization, or Grid-based client sampling. Those affect throughput and
communication realism — not the learned parameters.

### `ray`

Flower's full simulation runtime. **It does not run on this machine.** Ray launches
native helpers (`raylet.exe`, `gcs_server.exe`) as subprocesses, and Windows Smart App
Control blocks them:

```
OSError: [WinError 4551] An Application Control policy has blocked this file
```

Smart App Control is enabled here (`VerifiedAndReputablePolicyState = 1` under
`HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy`). **Turning it off is irreversible
without reinstalling Windows** — not worth it for this project.

Two related traps worth remembering:

1. `pip install "flwr[simulation]"` installs **nothing** on Windows with Python ≥3.13.
   The extra gates its `ray` dependency on `sys_platform != 'win32'`, so pip exits 0
   and you still have no Ray. A "successful" install proves nothing; check with
   `pip show ray`.
2. `flwr[simulation]` pins `ray==2.55.1`, which has no wheel for Python 3.14. The
   version installed here is `ray==2.58.0`, installed directly.

To use the Ray backend, run under WSL2 (Flower's own recommendation for Windows):

```bash
wsl --install                     # once, from PowerShell, then reboot
# then, inside WSL, from the repo directory:
python3 -m venv .venv-wsl && source .venv-wsl/bin/activate
pip install -r requirements.txt "flwr[simulation]"
python -m src.federated.run_simulation --backend ray --rounds 15 --clients 5
```

The Ray path has **not** been run successfully in this repo yet — the code is written
against Flower's current message API but is unverified end-to-end. Verify it before
citing any result from it.

---

## Reading the output

A run prints its full configuration and each client's class mix *before* training, so
any result can be traced back to the partition that produced it:

```
 client      rows  class counts (Benign, DDoS, DoS, Recon, Web-based, Brute Force, Spoofing, Mirai)
      0      1814  [34, 56, 375, 323, 18, 7, 331, 670]
      1      1487  [146, 1051, 119, 9, 3, 1, 118, 40]
      2      4699  [252, 3008, 843, 23, 4, 1, 4, 564]
```

Then per round, two different measurements:

```
[local]  round 3  federated (per-shard) macro_f1=0.3913 ...
[server] round 3  centralized          macro_f1=0.3944 ...
```

- **`[server] centralized`** — the aggregated global model on the shared validation
  split. **These are the numbers that go in the report.** All three lanes use this same
  split, so it is the only comparable measurement.
- **`[local] federated (per-shard)`** — each client scoring the global model on its own
  non-IID shard, averaged by shard size. Describes site-level behaviour; not a
  cross-lane benchmark. Under skewed partitioning it can look better than the
  centralized number simply because a client's shard is an easier, narrower problem.

Never quote accuracy alone. A model that ignores Brute Force and Web-based entirely
still scores ~78% accuracy on this data.

---

## Baseline results (measured in this repo, 2026-09-21)

Config: `--rounds 15 --clients 5 --alpha 0.5 --max-rows 100000 --eval-max-rows 40000
--local-epochs 2 --seed 7`. Light MLP: 46 → (64, 32) → 8, 5,352 params, 20.9 KiB
uploaded per client per round.

```
              precision    recall  f1-score   support
      Benign      0.791     0.938     0.858      1650
        DDoS      0.743     0.994     0.850     23449
         DoS      0.921     0.161     0.274      9604
       Recon      0.810     0.551     0.656       557
   Web-based      0.000     0.000     0.000        35
 Brute Force      1.000     0.105     0.190        19
    Spoofing      0.747     0.671     0.707       696
       Mirai      0.999     0.993     0.996      3990

    accuracy                          0.779     40000
   macro avg      0.751     0.552     0.567     40000
```

**This is a weak baseline, not a result to defend.** What it shows:

- **Web-based collapses to zero recall.** 35 validation rows; under Dirichlet
  partitioning most clients hold almost none.
- **DoS: 0.92 precision at 0.16 recall.** The model is confident but nearly silent —
  it is losing DoS to DDoS, which is the most interesting failure here given how
  similar those flow signatures are.
- **Mirai and DDoS are near-solved** (F1 0.996 / 0.850), unsurprising given they
  dominate the sample.

Things worth trying, roughly in order of expected payoff:

1. Class-weighted `CrossEntropyLoss` (weights ∝ inverse class frequency) in
   `local_train` — most direct attack on the rare-class collapse.
2. Balanced sampling: cap per-class rows when building the partition rather than
   sampling `--max-rows` uniformly, so rare classes are not diluted away.
3. More rounds (`--rounds 50+`) — round 14 hit 0.5999 and round 15 fell to 0.5665, so
   it is still oscillating, not converged.
4. Lower `--lr` or more `--local-epochs`, and compare: more local work is cheaper in
   communication but increases client drift.
5. A centralized-light run on the pooled data with the identical model and split — the
   comparison the lane exists for. Without it the federated number has no reference.

---

## Reproducibility

Runs are deterministic: a fixed `--seed` reproduces the partition, the model
initialization, and the batch order. Verified — repeated runs at the same seed produce
bit-identical metrics.

Every run logs its config and per-client class mix. When recording a result, keep:
seed, `--alpha`, `--clients`, `--rounds`, `--local-epochs`, `--batch-size`, `--lr`,
`--max-rows`, and the resulting macro-F1 **plus per-class recall**.

---

## Scope caveats

Straight from `CLAUDE.md`, and they belong in the write-up too:

- Federated learning here is **simulated**, not physically distributed. It is a stated
  scope choice, not a limitation to work around.
- The 20.9 KiB/client/round figure is a real parameter count. Any **latency, memory, or
  power** number is **projected** against the reference paper's Jetson Orin Nano
  benchmarks — never present it as measured.
- Wall-clock timing from either backend is not an edge-hardware measurement.
- Feature standardization uses statistics fitted on the whole training split and shared
  with every client. A strict federated setup would have to negotiate those without
  centralizing them, so don't describe this pipeline as fully privacy-preserving.
- Explainability output stays at the protocol/feature level. The dataset has no device
  identity, so no per-device vulnerability claims.

---

## Troubleshooting

| symptom | cause |
| --- | --- |
| `ModuleNotFoundError: No module named 'tensorflow'` | Stale `__pycache__` from the old TFF placeholder that used to live in `__init__.py`. Delete `src/federated/__pycache__/`. |
| `WinError 4551 ... Application Control policy` | Smart App Control blocking Ray. Use `--backend local`, or WSL2. |
| `Missing data/splits/train.parquet` | Run `python -m src.data.sample_dataset` first. |
| `pip show ray` → not found, after installing `flwr[simulation]` | The extra excludes Windows on Python ≥3.13. Install `ray` directly. |
| A client gets 0 rows of some class | Expected at low `--alpha`. Only a bug if a client gets *zero rows total* — the partitioner guards against that. |
| Rare-class recall is 0.000 | Expected for this baseline. See the improvement list above. |
