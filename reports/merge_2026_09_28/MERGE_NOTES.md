# Teammate branch integration — 2026-09-28

Target: `integration/combined-pipeline`, following baseline commit `166ead3`.

## Branches incorporated

- `origin/centralized-heavy` at `b252d84`: tuning commit `d1a5f98` followed by its
  exact revert. The net source difference is empty. Merge `56ec83e` retains both
  commits without resurrecting reverted hyperparameters.
- `origin/Joel` at `be0ea8e`: Flower cleanup, selectable server strategies,
  per-client evaluation summaries, caching, local history, optional W&B and
  improved shell launchers.

## Conflict resolutions

| Area | Integrated behavior |
| --- | --- |
| Default optimizer | FedAvg remains the baseline; FedAdagrad is a real, explicit opt-in, not an alias |
| Learning-rate schedule | Off by default; optional halving after each five completed rounds |
| Model and preprocessing | Keep the single canonical MLP, train-fitted scaler, shared loss/metrics and weight decay |
| Sequential support | Keep the tested full-study/local runners and their documentation; FedAvg-only |
| Core dependencies | Keep cross-platform requirements and optional Ray; add W&B as an optional package extra only |
| Data cache | Integrate tensor caching with LRU bounds and source/partition identity keys; fresh seeded loaders each round |
| Evaluation API | Keep `predict -> (loss, y_true, y_pred)` and existing `test -> (loss, accuracy, macro-F1)` compatibility; internal reporting uses `predict` |
| Client evaluation | Retain per-partition macro-F1 and min/median/max spread, plus weighted loss/accuracy; not global macro-F1 |
| Checkpoints and selection | Keep validation-only gates, round-zero fallback, best checkpoint, complete validation metrics and failed-client guards |
| Logging | Write local `history.json` during rounds and on failure; W&B disabled unless requested |
| Report identity | New worker filenames include strategy; decision grouping also distinguishes FedAdagrad eta/tau |
| Launcher | Forward SEED/PARTITIONER/ALPHA; keep 20 clients, strict shell checks, explicit argument overrides and LF line endings |
| Output folder | Move new worker artifacts to `outputs/federated/`, retaining unique microsecond timestamps |

Applicable beginner-facing explanations remain in the code. Verbatim incoming
comments/docstrings from overlapping implementations are also retained in
[JOEL_COMMENT_REFERENCE.md](JOEL_COMMENT_REFERENCE.md), marked historical so old
defaults/API descriptions are not mistaken for the current instructions. Original
source history is retained by real merge commits, not squash replacement.

## Verification

- 20 automated tests pass, including cache reuse/invalidation, identical cached
  versus uncached client updates, real optimizer identity, opt-in LR scheduling,
  client-F1 spread, report identity, and failure-history/logger cleanup.
- Shell syntax checks pass for `run_fed.sh` and `run_sweep.sh`.
- Real Ray/API smoke runs passed for **both FedAvg and FedAdagrad**, each with
  two clients, one training round, client evaluation, checkpointing and reporting.
  Both received two successful training and two successful evaluation replies.
- Smoke artifacts are synthetic and ignored under `outputs/runtime_smoke/`;
  they are not additions to the research results.
- No full-data retraining was performed for this merge. The previous 12-run
  study remains historical baseline evidence, not FedAdagrad performance evidence.
- W&B network logging was not exercised; cleanup was checked with a mock logger.
  No new online logging service was enabled or installed.

## Issues to keep visible

1. Client validation slices remain approximately IID modulo slices, even when
   training uses Dirichlet shards. Their F1 spread is not a measurement on matched
   non-IID client evaluation populations.
2. Flower's simulation extra omits Ray on native Windows/Python 3.13+ in the
   installed 1.38 metadata. Use the documented Python 3.12/WSL2 worker path or
   the sequential backend; do not disable security controls.
3. The optional adaptive optimizer changes the experiment. Never relabel existing
   FedAvg results, pool different eta/tau settings across seeds, or treat the tiny
   synthetic smoke scores as evidence that FedAdagrad improves detection.
4. W&B can upload configuration/metrics when explicitly enabled and authenticated.
   Default local history requires neither credentials nor network access.
