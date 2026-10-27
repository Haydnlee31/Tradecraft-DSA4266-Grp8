# Matched comparison for the methodology chapter

The six requested method families are implemented in the repository. What was
missing was one matched comparison spanning all of them. Historical FedProx used
the older feature/split protocol; historical SCAFFOLD was a continuation of a
trained model. Those scores must not be merged into the final Shared Adam table.

## Implementation inventory

| Method | Existing implementation | Role in the new comparison |
| --- | --- | --- |
| FedAvg | `src/eval/shared_optimizer_performance.py` and Flower simulation entry points | Frozen reset-client-Adam baseline plus a matched SGD control |
| FedProx | `src/models/train.py` and `src/models/research.py` | Common-protocol adapter using the same quadratic penalty gradient; both historical coefficients retained |
| SCAFFOLD | `src/federated/scaffold.py` | Existing plain-SGD/Option-II kernels, now from fresh initialization on the shared fit cohort |
| MIME | `src/federated/mime_adam.py` and `src/eval/shared_optimizer_performance.py` | Unchanged full-MIME runner plus its centralized base updater |
| Shared-gradient Adam | `src/federated/shared_adam.py` and `src/eval/shared_optimizer_performance.py` | Unchanged persistent server-Adam runner |
| Compressed shared-gradient Adam | `src/federated/block_shared_adam.py` | Unchanged four-bit block128 codec, residual state and delivered-model evaluation |

The new adapter is [methodology_comparison.py](../../src/eval/methodology_comparison.py).
It preserves all historical source files and comments. The [plan](plan.json) was
committed as `a62b4fd` before the new results were observed. It contains eleven
configurations, including optimizer controls and both FedProx coefficients, with
three seeds each. This is a fixed-setting retrospective supplement, not a new
hyperparameter search or a replacement for the closed prospective confirmation.

## Data and evaluation boundary

Use the existing checksum-pinned 500,000 fitting rows, 399,552 development rows
and 398,502 confirmation rows. Calibration uses the existing development-only
mask. All training endpoints and thresholds must be saved before the separate
confirmation command runs. The latter cohort was already evaluated in the study;
it is not a fresh holdout. The original official validation and test arrays are
not opened by these commands. Here, validation performance means the development
audit and subsequent confirmation scores, with their distinct roles retained.

Every method receives 21.6 million backward examples. The 33 runs total 712.8
million; additional recovery replay work is separately counted. FedAvg, FedProx
and SCAFFOLD share their sampled client row streams within each seed. MIME additionally streams
full anchors and therefore completes fewer local blocks under the fixed budget.
Equal backward work is not equal FLOPs, optimizer calls, communication or time.

## Reproduce the current local comparison

The recorded execution uses Python 3.13.9, PyTorch 2.14.0, NumPy 2.5.3 and Flower
1.38.0 in the existing project `.venv`, on a macOS ARM CPU with two threads. This
new runner retains the historical strict runtime guard; unlike the separate
compressed-model portable release, it has not been validated on WSL or CUDA.
No RONIN instance is required. Do not describe these runs as GPU measurements.

The v1 reproduction release supplies the prepared inputs. Its minimal
requirements do not include Flower; this comparison additionally needs Flower
for the existing weighted SCAFFOLD aggregation helper. Preserve an existing
working environment rather than reinstalling it blindly.

For a separate environment, `requirements-methodology.txt` pins the recorded
training and plotting packages. Use Python 3.13.9 and run `python -m pip check`
after installation. This dependency list does not remove the cross-platform
numerical limitations or the runner's strict runtime checks.

From the repository root:

```bash
.venv/bin/python -m unittest tests.test_methodology_comparison tests.test_scaffold tests.test_mime_adam -q
.venv/bin/python -m src.eval.methodology_comparison check
.venv/bin/python -u -m src.eval.methodology_comparison train --output outputs/my-methodology-comparison
.venv/bin/python -m src.eval.methodology_comparison evaluate --output outputs/my-methodology-comparison
.venv/bin/python scripts/methodology_report.py --run outputs/my-methodology-comparison --output outputs/my-methodology-report
```

Training requires a **new directory** and does not silently resume, overwrite,
reuse a historical model or select a favorable seed. Serialized final-two-step
recovery is checked for every run; this is an implementation test, not a user-
facing interrupted-job resume feature. If interrupted, preserve the partial
outputs and use a new directory for a full rerun. Evaluation refuses an
incomplete set of runs and will not overwrite a completed summary.

Results include default argmax metrics, the full development endpoint, held-out
development-audit metrics, all four fixed threshold targets, all per-class
metrics and confusion matrices, work counters, source identities, model and
sample-stream hashes, and actual runtime information. The report generator
checks confusion-derived metrics independently and compares the four historical
control arms against their recorded confirmation results.

## Writing deliverables

- [Set-up and methodology draft](methodology-draft.md): data processing,
  algorithm settings, matched controls, budget and truthful hardware description.
- [Results](results.md), [quality plot](quality.png) and [recall plot](recall.png):
  completed development and confirmation results for all 33 runs.
- [Interpretation and revised method summaries](findings.md): what the results
  support, what changed from the earlier narrative, and what remains unresolved.
- [results.json](results.json): aggregate evidence, all 33 endpoints and independent checks.
- [per-seed-metrics.csv](per-seed-metrics.csv): numerical table for teammates, not private input rows.

All 33 serialized-recovery checks passed. The four original controls across
three seeds reproduced all 12 model, confirmation-logit and metric endpoints
exactly. The report independently recalculated 363 confusion-matrix endpoints.
All 36 predeclared paired development/confirmation comparisons have realized
false-alert differences within 0.5 percentage points.

The existing thesis manuscript and its frozen manifest are not overwritten.
Incorporate this supplement deliberately and retain the distinction between
historical prospective evidence and the new retrospective comparison.
