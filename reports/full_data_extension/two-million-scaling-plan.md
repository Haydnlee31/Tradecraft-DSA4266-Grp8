# Staged 2M-row scaling experiment

Prepared 2026-10-08. Status: local preparation only; no 2M model trained.

## Question and motivation

Does increasing training diversity improve rare-class detection, or does it mainly
benefit already strong classes? The fixed-loss-denominator experiment produced
almost unchanged results, so further denominator tuning is not the next priority.
Keep LayerNorm, square-root class weighting, the original batch-weight-sum loss
reduction, and full local epochs as the provisional reference configuration.

This is an exploratory, single-seed comparison within the official 39-feature
collection. It is not evidence of generalization to unseen capture sessions and
is not directly comparable to the earlier 46-feature study.

## Prepared data

The nested 2M training cohort and 2,059,284 validation rows were packed and checked
against a second replay of their source data. Array hashes, finite values, class
counts, and partition coverage passed. No test data was opened by this preparation.
The validation population is unchanged. Each cohort has its own scaler fitted only
on its training rows; consequently its standardized validation features differ.

| Class | 2M training rows |
|---|---:|
| Benign | 106,148 |
| DDoS | 1,178,561 |
| DoS | 362,423 |
| Recon | 66,277 |
| Web-based | 2,391 |
| Brute Force | 1,269 |
| Spoofing | 44,095 |
| Mirai | 238,836 |

More total data does not eliminate imbalance: Brute Force still has only 1,269
training examples. Report precision and recall for every class, macro-F1, and
benign false-alert rate rather than headline accuracy alone.

## Frozen first stage: centralized-light and IID federated-light

Use seed 7, partition seed 7, 20 clients, batch size 512, learning rate 0.001,
weight decay 0.00001, LayerNorm, `sqrt_weighted_ce`, `batch_weight_sum`, two CPU
threads, and CUDA. Disable local batch caps and total-update-budget overrides.

| Endpoint | Training examples processed | Light optimizer updates | IID optimizer updates |
|---|---:|---:|---:|
| Existing 500k, 20 passes | 10,000,000 | 19,540 | 19,600 |
| New 2M, 5 passes | 10,000,000 | 19,535 | 19,600 |
| New 2M, 20 passes | 40,000,000 | 78,140 | 78,400 |

The primary comparison matches **processed training examples**, not rounds,
communication, wall time, or exactly all optimizer updates. The secondary
comparison matches 20 passes and therefore allows four times as many examples
to be processed. Federated rounds contain one full local epoch per client.

Create one 20-pass run per lane and pause at step 5. Preserve `last.pt`, history,
environment, status, and partition metadata before resuming that same run to 20.
Check the observed work counters against the table. Do not repeat the first five
passes in a separate run. Use final-at-budget validation metrics for comparisons;
best-checkpoint scores are descriptive because checkpoint-selection opportunities
differ. Stop for nonfinite losses, incomplete client participation, provenance
mismatches, or incorrect work accounting. Do not extend beyond the frozen budget
because a curve looks promising.

## Why non-IID scaling is deferred

The same Dirichlet recipe and seed do not preserve client mixtures when row count
changes. Client sizes shift from 5,200–49,561 at 500k to 12,142–274,037 at 2M.
Allocation profiles also change substantially. Thus an apparent improvement or
decline would mix a data-size effect with a different heterogeneity problem.
Numbered client IDs are arbitrary simulated clients, not physical devices.

Hold the Dirichlet run until a profile-controlled comparison is designed and
tested. IID repartitioning also changes membership, but every client retains all
eight classes and equal size. Even the staged results include cohort-specific
preprocessing and class-weight changes: do not describe them as an isolated
causal effect of sample count. Do not launch 5M or a heavy-model sweep yet.

## Review and next decision

Compare the fixed-budget endpoints first, then the 20-pass endpoints. Examine
whether rare-class gains come with additional false alerts or poor precision.
An improvement seen only at 20 passes cannot establish a benefit at equal work.
Before claiming a robust advantage, freeze and run a multi-seed confirmation;
do not use the test split to choose the next experiment. A larger GPU alone is
not evidence of a better model.

## Reproducibility and handoff

Local receipts (ignored outputs, not committed datasets):

- `outputs/official39-packed-2m-check.json`
- `outputs/official39-2m-scale-plan-v2.json` — authoritative staged plan; supersedes the initial three-lane receipt.
- `outputs/cloud-handoff-2m-v1/official39-packed-2m-v1.tgz`

Packed manifest SHA256:
`dfd1caf01cbe233b52ce55083941001f390ec251dd89b37543630a2bc53935be`

Scaler SHA256:
`6731ce8c113fa568006ef551bb40b00e027169598d16f9b96e02f491ca681b9e`

Transfer archive SHA256:
`60ac3e212c0648c531db73cc5d5df5b3706d8a6b37aab40dffe69e5b49f19fc5`

The planning utility is `src/eval/official_scale_plan.py`; it verifies cohort
provenance, derives budgets from actual partition sizes and batch-tail handling,
and records class-allocation profiles without training. The local workspace test
suite passed 90 tests. This does not replace checking the transferred archive,
cloud environment, and 2M packed-data receipt before training.

Next: make a scoped commit of the planner, its tests, and this plan, then prepare
the exact cloud handoff. Keep unrelated work out of that commit. No commit, push,
cloud launch, or test-set evaluation was performed during local preparation.
