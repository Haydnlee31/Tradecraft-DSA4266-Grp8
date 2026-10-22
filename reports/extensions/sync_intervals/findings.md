# Synchronization intervals: real-data findings

22 October 2026. **Increasing the synchronization interval did not improve macro-F1 over the original shared-gradient Adam reference in this fixed-recipe, seed-7 screen.** Both interval variants beat the reset-client-Adam FedAvg control on macro-F1, but neither preserved shared Adam quality within the predeclared tolerances.

## Measured outcomes

- **Shared Adam K=1: macro-F1 0.652142**, benign false-alert rate **19.554%**, 2,160 synchronizations; modeled payload 1,761.178 MB; measured CPU training loop 52.96 s.
- **Normalized FedOpt K=4: macro-F1 0.611025**, benign false-alert rate **15.425%**, 540 synchronizations; modeled payload 440.294 MB; measured CPU training loop 58.64 s.
- **Normalized FedOpt K=20: macro-F1 0.574382**, benign false-alert rate **19.799%**, 108 synchronizations; modeled payload 88.059 MB; measured CPU training loop 53.44 s.
- **FedAvg with local SGD K=20: macro-F1 0.493947**, benign false-alert rate **29.198%**, 108 synchronizations; modeled payload 88.059 MB; measured CPU training loop 52.79 s.
- **FedAvg with reset client Adam K=20: macro-F1 0.483683**, benign false-alert rate **1.425%**, 108 synchronizations; modeled payload 88.059 MB; measured CPU training loop 59.96 s.

K=4 reduces synchronization count and uncompressed tensor payload by **75%**, but loses **4.112 macro-F1 percentage points** versus K=1. K=20 reduces both by **95%**, but loses **7.776 points**. Their macro-F1 gains over reset-Adam FedAvg are **12.734** and **9.070 points**, respectively. These are not alarm-matched gains: the false-alert rates differ substantially.

All eight class recalls (%) are listed in order **K=1 / K=4 / K=20 / SGD FedAvg / Adam FedAvg**:

- **Benign**: 80.446 / 84.575 / 80.201 / 70.802 / 98.575; support 21,193.
- **DDoS**: 83.910 / 95.384 / 87.952 / 83.721 / 99.418; support 235,894.
- **DoS**: 72.434 / 42.029 / 54.278 / 52.317 / 18.362; support 72,090.
- **Recon**: 68.904 / 68.164 / 67.930 / 68.640 / 26.793; support 13,246.
- **Web-based**: 12.323 / 3.434 / 0.000 / 0.000 / 0.000; support 495.
- **Brute Force**: 28.000 / 29.600 / 34.800 / 0.000 / 19.600; support 250.
- **Spoofing**: 76.702 / 59.131 / 45.435 / 25.368 / 15.467; support 8,696.
- **Mirai**: 99.495 / 99.392 / 98.956 / 98.329 / 98.960; support 47,688.

All precision, F1, support and confusion-matrix values are retained in [audit.json](audit.json).

## What happened

K=4 loses substantial DoS recall (72.43% → 42.03%) and Spoofing recall (76.70% → 59.13%). K=20 lowers Spoofing recall further to 45.44%. Web-based recall falls from 12.32% at K=1 to 3.43% at K=4 and zero at K=20. Brute Force recall improves (28.0% → 29.6% → 34.8%), but that does not compensate for the broader losses.

The method changes the optimization trajectory as well as communication. At K>1 clients evaluate gradients along separate local SGD paths, and the server applies fewer Adam updates: 540 or 108 instead of 2,160. Local drift and the smaller server-update count are plausible explanations, but this screen does not identify their separate causal effects. Learning rates were held fixed, not tuned separately for each interval.

No CPU speedup was demonstrated in this single-machine simulation: all arms took roughly 53–60 seconds of training-loop time. Communication savings are tensor-byte and synchronization counts, not measured network latency. The extra local work and simulation overhead remain even when synchronization count falls. A real network could change timing, but that was not measured.

## Predeclared decision

The plan required at least fourfold fewer barriers, with no more than .01 macro-F1 loss, .005 FAR increase, or .05 recall loss in any class relative to K=1. Both candidate intervals pass the barrier and FAR checks but **fail macro-F1 and per-class recall retention**. The largest recall losses are 30.405 points at K=4 and 31.267 points at K=20. Neither candidate should replace K=1 under this criterion.

The SGD and Adam FedAvg controls help characterize the recipe; they are not exhaustive optimizer benchmarks. In particular, fixed local SGD rate .003 was not optimized in a separate search. This is a normalized FedOpt-inspired adaptation, not a reproduction of every detail of published FedAdam.

## Frozen experimental setup

Five endpoints used the same 500,000 fit rows, 399,552 development rows, twenty client populations and canonical 39→64→32→8 LayerNorm MLP (5,096 parameters, dropout zero). Each arm consumed 21.6 million backward examples: **108 million** across the screen. Temperature/distillation was not used. Local and server rates were .003, and server Adam retained betas (.9,.99), epsilon .001 and zero weight decay. Loss was the existing fixed-training-mean square-root-weighted CE.

For interval arms, each client takes K plain SGD steps and sends the mean of its path gradients. This equals the normalized SGD displacement mathematically; directly accumulating gradients avoids float32 subtraction cancellation. The server population-weights the messages and updates persistent Adam. K=1 uses the exact original gradient path. Compression is disabled, so the savings are relative to **uncompressed** shared Adam, not the existing four-bit method.

Each client receives the same ordered minibatch stream across K through the historical 20-step sampling blocks. All per-client sample hashes match. The inter-client traversal order changes with K and is deliberately recorded separately.

These scores are final **development argmax** results. The historical Shared Adam confirmation score of 0.6325 uses a different cohort and calibrated alarm rule; it is not the comparator here. No new confirmation/test scoring, threshold fitting, additional seeds, or post-result parameter search was performed.

## Audit and isolation

- **19 extension-module tests passed:** eight new synchronization tests plus eleven reused FedNTD tests. New checks include exact K=1 weights/moments/samples, independent normalized-message algebra, exact Adam FedAvg, independent SGD aggregation, per-client sample identity, accounting, persistent-Adam recovery and disk interruption handling.
- **19 existing report/archive regression tests passed.**
- All five actual checkpoints were reloaded. Their final synchronization was rerun from a saved prior state, reproducing weights, persistent Adam moments, counters and sample hashes **bit for bit**.
- Metrics were independently recomputed using scikit-learn on every development row for all five checkpoints.
- The real K=1 shared Adam and reset-Adam FedAvg endpoints exactly reproduce the historical weights, complete metrics and traversal sample hashes.
- The final-update audits cost an explicitly separate **650,000 backward examples**; metric replay scored **1,997,760 development examples**. These are verification costs, not part of the original 108-million training budget.
- All **662 previously tracked files** retained their pre-implementation SHA-256 hashes. The original pipeline and completed FedNTD files remain unchanged. No dependencies were added, and no code was pushed.

This is a self-audit supported by executable checks, not an external review. The [test/isolation receipt](isolation-and-tests.json) records the verification scope. The full repository test suite was not run.

## Files and reproduction

- [Implementation and run instructions](../../../extensions/sync_intervals/README.md).
- [Frozen plan](../../../extensions/sync_intervals/plan.json).
- [Audited results](audit.json).
- Checkpoints and original summary: `extensions/sync_intervals/runs/seed7-screen-v1/` (ignored by Git).
- Training log: `outputs/sync-intervals-seed7-screen-v1.log`.
- Existing runtime: `outputs/Tradecraft-reproduce/.venv-fedntd/bin/python` (Python 3.13.9).

A rerun of the audit requires the checkpoint directory and original prepared bundle. The historical-reference audit is for this frozen seed-7 protocol; a new seed or changed recipe requires an appropriate reference rather than bypassing its equality checks.

The results support retaining K=1 for this quality criterion. They do not prove that every synchronization schedule fails: this is one seed, one partition, fixed learning rates and an already used development cohort. A further schedule or rate study would need a new predeclared comparison.
