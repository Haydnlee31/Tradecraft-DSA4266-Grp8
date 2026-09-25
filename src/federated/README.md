# Federated-light lane

This lane simulates sample-count-weighted FedAvg over non-IID virtual clients. It
shares `MLPClassifier(LIGHT_CONFIG)`, the train-fitted scaler, loss implementations,
label order, metrics, validation split, and report schema with centralized-light.
That shared contract is what makes the centralized-versus-federated result
interpretable.

The virtual clients are arbitrary flow shards. They are not the 105 physical devices
in CICIoT2023 because the released flow CSVs contain no device identity.

## Quick start

Prerequisite: build `data/splits/{train,val,test}.parquet` and run leakage removal as
described in the root README.

```bash
# Build one persistent mapping and its class-count/provenance files.
python -m src.federated.partition --alpha 0.5 --num-partitions 20 --seed 0

# Run the report-producing Flower simulation.
NUM_NODES=20 ./run_fed.sh "dirichlet-alpha=0.5 seed=0"
```

`run_sweep.sh` builds/runs alpha `{0.1, 0.5, 1.0}` plus IID for seeds 0–2:

```bash
NUM_NODES=20 ./run_sweep.sh
```

Generated partition artifacts are ignored by Git. Each JSON sidecar records the
source train file, size, modification time, alpha, seed, and client count; a run
refuses a stale or incompatible mapping.

## Module map

| file | role |
| --- | --- |
| `partition.py` | Persist and validate Dirichlet row-to-client assignments. |
| `task.py` | Shared light model, scaler, loss, loaders, and train/predict helpers. |
| `client_app.py` | Thin Flower message adapters around pure client helpers. |
| `custom_strategy.py` | FedAvg plus best-val checkpointing and early stopping. |
| `server_app.py` | Run provenance, full-val scoring, final test, unified report. |
| `local_backend.py` | Sequential compatibility runner using the same client code. |
| `run_simulation.py` | CLI for a small no-Ray smoke run. |

The active experiment strategy is FedAvg. A pre-integration branch accidentally
subclassed FedAdagrad while naming its output FedAvg; results from that implementation
must not be mixed with the integrated reports.

## Important run configuration

Defaults live in `pyproject.toml` and can be overridden through `run_fed.sh`.

| key | default | meaning |
| --- | --- | --- |
| `num-server-rounds` | 25 | Maximum federated rounds. |
| `fraction-train` | 1.0 | Fraction of virtual clients training each round. |
| `fraction-evaluate` | 0.0 | Client-side evaluation fraction; full server val always runs. |
| `local-epochs` | 1 | Local passes before one aggregation. |
| `learning-rate` | 0.001 | Client Adam learning rate. |
| `loss` | `sqrt_weighted_ce` | Shared loss from `src/models/losses.py`. |
| `class-weights` | `global` | `global` matches centralized; `local` is an FL ablation. |
| `partitioner` | `dirichlet` | `dirichlet` mapping or modulo-row `iid`. |
| `dirichlet-alpha` | 0.5 | Lower means more class skew. |
| `patience` | 5 | Stop after this many non-improving val macro-F1 rounds. |
| `lr-decay-every` | 0 | Optional schedule; 0 keeps the centralized-matched constant LR. |

With `fraction-train=1` and `local-epochs=1`, every round processes approximately one
pooled pass over train. That gives the fairest available comparison with a centralized
epoch, though federated optimization is still not mathematically identical.

## Metrics and checkpoints

Every round is scored on the entire shared validation split. The best global model is
selected by validation macro-F1 and only that checkpoint is evaluated on test. Reports
contain accuracy, macro-F1, all per-class recalls, a confusion matrix, and the sklearn
classification report. Accuracy alone is never sufficient for this imbalanced task.

Client validation, when enabled, describes how the global model behaves on individual
site mixes. It is not comparable across clients or with the three headline lanes.

## Backends and Windows

`flwr run` uses Flower's simulation runtime and Ray workers. Flower recommends WSL2
when native Windows/Ray process launching is unreliable. The compatibility command
below runs the same pure client code sequentially and aggregates with Flower's own
helpers:

```bash
python -m src.federated.run_simulation --rounds 2 --clients 3 --alpha 0.5
```

It is a smoke/debug path, not the final report producer. It does not reproduce process
isolation, concurrency, message serialization, or client sampling.

## Scope caveats

- Federated learning is simulated, not physically distributed.
- The globally fitted/shared scaler is a declared simulation simplification; it is not
  a claim of end-to-end privacy.
- Parameter counts and tensor bytes are exact derived quantities. Simulation wall time
  is measured locally. Latency, RAM, and power projected from Jetson reference numbers
  must remain labelled **projected**, never measured.
- Explainability and policy recommendations stay at feature/protocol level; no
  per-device vulnerability claims are supported by these CSVs.
