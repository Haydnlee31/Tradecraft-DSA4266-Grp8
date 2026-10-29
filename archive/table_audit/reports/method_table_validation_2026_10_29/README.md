# Validation of the supplied six-row method table

Validation date: 29 October 2026. Source checkout: `3d90a8d`, branch
`thesis/matched-methodology-comparison`. This audit freshly trained the four
traceable methods using the existing frozen kernels. It did not implement a
replacement for the missing MimeLite experiments or tune models to the supplied
scores.

## Outcome

| Supplied method | Fresh macro-F1 | Fresh benign false alerts | Rounds | Logical barriers | Modeled payload, decimal MB | Validation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Central Adam | 0.645634829 | 5.333459% | 0 | 0 | 0 | Exact reproduction |
| Shared Adam | 0.624637202 | 5.427523% | 2,160 | 2,160 | 1,761.177600 | Exact reproduction |
| MimeLite, 5 local steps | Unverified | Unverified | Unverified | Unverified | Unverified | No executable pipeline or matching result located |
| MimeLite, 20 local steps | Unverified | Unverified | Unverified | Unverified | Unverified | No executable pipeline or matching result located |
| Full MIME | 0.515306824 | 5.239394% | 24 | 48 | 58.705920 | Exact reproduction |
| FedAvg-Adam | 0.513690314 | 5.700310% | 108 | 108 | 88.058880 | Exact reproduction |

Every traceable supplied value matches after its displayed rounding. Fresh model
parameters, saved development logits, sampled-row stream hashes, work counters,
all development metrics and thresholds match the recorded seed-7 endpoints
exactly. Each run also passed a serialized final-two-step recovery replay.
The run processed 86,400,000 training backward examples, with recovery work
counted separately in [results.json](results.json). Thirty-three targeted tests
for training, MIME, threshold calibration and common comparison adapters passed.
Per-class recall, precision, F1, supports and confusion matrices are preserved
in that JSON; these checks do not rely only on the aggregate score.

## What these numbers measure

This is the **single-seed development-audit table**, using seed 7 and thresholds
calibrated to a 5% benign false-alert target. The fitting cohort has 500,000 rows.
Development has 399,552 rows, split into 200,022 calibration rows and 199,530
audit rows. Calibration and audit belong to previously examined development
data. These are not the three-seed means on the 398,502-row confirmation cohort.

The apparent difference from the thesis table is therefore expected: for
example, centralized Adam's 0.6456 here and its 0.6535 three-seed confirmation
mean refer to different evaluations. Do not mix those values in one comparison.
No confirmation or original validation/test inference was performed in this
audit. The reruns used the recorded strict macOS ARM CPU environment, two
threads, and the historical fit-only scaler, fixed client ownership and loss.

## Pipeline inventory

- `src/eval/shared_optimizer_performance.py` contains the existing centralized
  Adam, shared-gradient Adam, reset-client-Adam FedAvg and full MIME training
  paths. `src/federated/mime_adam.py` implements **full MIME**, not MimeLite.
- `src/eval/shared_optimizer_alarms.py` provides development-only calibration
  and audit evaluation. The matched comparison wrapper preserves these kernels.
- `scripts/validate_seed7_method_table.py` is the new validation entry point. It
  reruns the four existing endpoints, independently checks calibrated metrics,
  compares against saved evidence, and reports the other two rows as unverified.
- Searches of current code, saved outputs and advertised repository branch tips
  found MimeLite in proposal/literature notes only. The synchronization-interval
  extension on `extension_approach` uses normalized local-SGD messages with server
  Adam; it is not evidence for the supplied MimeLite results. See
  [branch-search.json](branch-search.json) for the inspected branch objects.

The absence finding is limited to the searched repository and available outputs.
An unpushed teammate script or external run may exist. Its exact code, learning
rate, state-update rule, seed, data hashes and result file are needed before
0.5843 and 0.5477 can be called reproducible project results.

## Two communication and fairness corrections

**Use distinct columns for rounds and barriers.** Full MIME performs an anchor
aggregation and a model aggregation per round in the current implementation.
It has 24 rounds but 48 logical barriers. Calling all entries simply "syncs"
conceals this distinction. Payload is modeled FP32 application data, with one
MB equal to 1,000,000 bytes; it is not measured network traffic.

**MimeLite's extra gradient work must be accounted for.** Algorithm 1 in the
[original MIME/MimeLite paper](https://arxiv.org/html/2008.03606) uses full local
gradients at the round-start model to update the server optimizer state, even
though MimeLite omits MIME's local gradient correction.

If the supplied MimeLite rows used that full-anchor rule with this study's
500,000 fitting rows, 20 participating clients and batch size 500, their implied
work would be:

| Claimed configuration | Local backward examples | Full-anchor backward examples | Total backward examples |
| --- | ---: | ---: | ---: |
| 5 local steps, 432 rounds | 21,600,000 | 216,000,000 | 237,600,000 |
| 20 local steps, 108 rounds | 21,600,000 | 54,000,000 | 75,600,000 |

The other four rows each use **21,600,000 total backward examples**. Thus these
MimeLite counts would not be compute-matched under that assumption. This is a
conditional accounting check, not evidence of what an unavailable script did.
A stochastic-anchor adaptation needs its own explicit description and work
counters.

The displayed 881 MB and 220 MB are arithmetically consistent with five FP32
model-sized vectors per client per round: 880.588800 MB and 220.147200 MB for
the 5,096-parameter model. That arithmetic cannot verify either training run,
its payload implementation, or its classification score.

## Reproduce this audit

Use the already prepared reproduction inputs and the strict runtime documented
in the [methodology guide](../methodology_comparison/README.md). From the repo
root, choose a new output directory:

```bash
.venv/bin/python -m unittest tests.test_shared_optimizer_performance tests.test_shared_optimizer_alarms tests.test_mime_adam tests.test_methodology_comparison -q
.venv/bin/python -u scripts/validate_seed7_method_table.py --output outputs/my-seed7-table-validation
```

The final status is deliberately `partial_validation_complete`: four reproduced
rows, two unverified rows. Do not describe it as verification of all six methods.
No existing training kernels, historical results or model-selection decisions
were modified.
