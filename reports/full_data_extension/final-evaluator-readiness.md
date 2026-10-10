# Frozen evaluator readiness

The evaluator passes the historical replay gate on the original CUDA runtime:
all 27 saved candidates reproduce both validation populations exactly. The
six small differences on the Mac remain documented below, but no longer block
use of the validated CUDA environment. The subsequent approved
[final evaluation](final-closeout.md) is now complete. No training or test
evaluation occurred during the validation checks described here.

## What is now implemented

`scripts/official39_final_evaluation.py` reads all 27 hash-pinned final
checkpoints directly from the two reviewed backup archives, without extracting
or executing archive code. It checks lane, seed, final step, architecture,
feature order, scaler/pack identity, input mask and the saved environment.
It uses restricted Torch weight loading and the unchanged light architecture.

One forward pass supplies both population views. Scores retain all eight
classes, confusion counts and attack-error categories. Exact ties choose the
first canonical class. There is no optimizer, threshold, ensemble, best-seed
selection or model promotion. Interrupted jobs can resume only at completed
model boundaries with the same runtime, inputs, code and checksummed results.
An orphan output or stale process lock requires inspection, not deletion by the
runner. Completed cases are never silently adopted or overwritten.

`scripts/official39_test_panel.py` implements future, separately authorized
packing and disk-backed exact-input screening. It reconstructs the saved
train-only scaler, replays packed rows for identity, and excludes matches to
training **or validation** under any input mask. Every candidate will use the
same retained rows. Membership ignores labels; reporting retains class supports
and projected duplicate counts. Empty retained classes block scoring. See the
[test preparation handoff](test-preparation.md) for the subsequent authorized
data-access stage; it is separate from this validation-only replay.

## Checks actually run

- 21 new synthetic tests passed, including complete 27-case fake campaigns,
  resume/tampering, masks, singleton-tail batching, ties, nonfinite values,
  exact-key overlap union, signed zero, label permutation and saved scaling.
- All 309 tests passed in a clean HEAD-plus-this-change snapshot. Unrelated
  local work was excluded from that verification and remains untouched.
- `pip check` reports no broken requirements; no dependency was added.
- The frozen final-plan audit still reports `prepared_test_closed`.
- All 27 actual saved models were replayed locally on 2,059,284 validation rows
  and the existing shared 2,047,805-row panel. The new evaluator exactly matches
  the unchanged historical evaluator **on this runtime** for every confusion
  matrix in both populations.
- A second invocation with `--resume` preserved the completed results and
  reproduced the same receipt without new inference.

The measured sum of inference/replay sections was 36.91 seconds. This includes
two validation forwards per candidate (new and unchanged evaluators), not
archive/input verification or total wall time. It is not an edge benchmark.

## CPU replay differences

The local runtime is macOS arm64, Python 3.13.9, Torch 2.14.0, NumPy 2.5.3,
two CPU threads. The archived models were evaluated on CUDA with Torch
2.7.0+cu128. Twenty-one endpoints reproduce the archived counts exactly.
The following six have confusion-matrix L1 difference **2** in both views:

| Candidate | Shared-panel macro-F1 difference, CPU minus archive |
| --- | ---: |
| light / full39 / seed 7 | -0.000000155532 |
| light / Number masked / seed 17 | -0.000000156881 |
| IID / full39 / seed 7 | -0.000000258555 |
| IID / full39 / seed 27 | +0.000000158067 |
| IID / Number masked / seed 17 | +0.000000249524 |
| non-IID / Number + Tot sum masked / seed 7 | -0.000000486272 |

L1=2 means a net count moved between two cells; aggregate matrices alone do
not identify which rows changed or exclude canceling row-level differences.
The largest macro-F1 difference across either population is about 0.000049
percentage points. These observed score differences do not change the prior
research interpretation. Nevertheless, the strict count-exact gate was not
relaxed after observing them.

Matching the unchanged evaluator locally makes a new counting/masking bug less
likely. The differences are consistent with runtime/device numerical variation,
but this comparison does not isolate the cause. The subsequent replay in the
original CUDA environment reproduces all historical counts exactly. Final
inference must use that validated environment, not silently substitute the Mac.

The local receipt is `outputs/official39-final-validation-replay-v1/checks.json`,
with SHA-256 `741ae1ff88786d85290a3c7097ab5eeafdbfc81cf1a2fe8425198cc211b2601b`.
It binds all 27 case-file hashes, runtime, runner sources and frozen protocol
`48f105e89229830556f4252952876f09a2984044fb1b4fb5ecef0914551ae247`.
Each case preserves both populations' full metrics, per-class recalls and errors.
Private checkpoints, arrays and individual replay outputs remain outside Git.

## Completed CUDA replay

The [CUDA receipt](final-validation-cuda-checks.json) records 27 same-runtime
matches and 27 exact historical matches on the Tesla T4, Torch 2.7.0+cu128,
Python 3.12.10 and NumPy 2.5.3. The local archive review checked every case hash,
the session and recovery journal, all frozen checkpoint state hashes, both
confusion matrices and all eight classes' metrics against the original records.
It did not perform new inference. Both validation views pass for all 27 models.

The cloud inference/replay sections sum to 253.44 seconds, excluding setup,
archive checks and file I/O outside those sections. This is not billing time
or an edge benchmark. All three CI jobs for evaluator commit `e4c3215` passed.

The backup archive SHA-256 is
`d4627714a5b19e73c084c2821c2d0458c548db5d5dad087e8c7eee0c350a57e3`;
the unchanged exported receipt has SHA-256
`1b0ea88403fe1b591021f77b4a028e18f4d948ebd6fd182bc51efb7c809d2643`.
The [replay instructions](final-validation-cloud.md) are retained for recovery,
not a request to repeat the successfully completed job.

## Subsequent completed stages

The authorized [test preparation](test-preparation.md) is complete, with all
eight classes retained. The separately approved final evaluation, research
decision view and extension report are now complete in the [final handoff](final-closeout.md).
They report every candidate and both populations. A final test result cannot
itself authorize operational deployment or cure rare-class failures. This
extension still has no matched official39 heavy model.

The immutable protocol and historical training code are unchanged. The original
46-feature study and its decision defaults remain separate.
