# Frozen evaluator readiness — 10 October 2026

The evaluator is implemented and locally checked. Final test scoring is still
blocked by the historical replay gate. No official test rows were opened in
this work, no model was trained, and no cloud machine was started.

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
and projected duplicate counts. Empty retained classes block scoring. This
production preparation has **not** been executed.

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

## Why the final-test gate is still blocked

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
but this comparison does not isolate the cause. Replaying in the original CUDA
environment is the next diagnostic. This requires inference only, not training,
more data, a larger model or a more expensive GPU.

The local receipt is `outputs/official39-final-validation-replay-v1/checks.json`,
with SHA-256 `741ae1ff88786d85290a3c7097ab5eeafdbfc81cf1a2fe8425198cc211b2601b`.
It binds all 27 case-file hashes, runtime, runner sources and frozen protocol
`48f105e89229830556f4252952876f09a2984044fb1b4fb5ecef0914551ae247`.
Each case preserves both populations' full metrics, per-class recalls and errors.
Private checkpoints, arrays and individual replay outputs remain outside Git.

## Remaining gates

1. [Validation-only CUDA replay](final-validation-cloud.md), with no tolerance
   change and no test access. If any endpoint still differs, preserve the output
   and diagnose before advancing.
2. Explicit approval to prepare the label-blind test overlap panel; inspect its
   receipt and retained supports without model scoring.
3. Separate approval for one complete frozen final evaluation on the validated
   runtime. Report all candidates and both populations, including poor results.
4. Complete the research write-up and decision-layer warnings. A final test
   result cannot itself authorize operational deployment or cure rare-class
   failures. This extension still has no matched official39 heavy model.

The immutable protocol and historical training code are unchanged. The original
46-feature study and its decision defaults remain separate.
