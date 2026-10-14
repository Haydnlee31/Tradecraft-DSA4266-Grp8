# Class gradient conflict is real but the first safeguard is incomplete

The training-only diagnostic found the proposed mechanism, but **the first
correction does not pass its predeclared continuation gate**. Ordinary FedAvg
increased Brute Force training loss at all three inspected states. Projecting
the update reversed those increases without shrinking the update toward zero.
However, actual benign loss increased at two states despite satisfying the
first-order benign constraint. Do not start a full corrected training run or
claim that rare-attack detection has improved.

This refocuses the project on centralized versus federated optimization. The
standalone classifier-search extensions are discontinued research directions;
their code, comments and historical results remain intact. No original runner
was changed and no model was added to the inference pipeline.

## What was tested

Three frozen seed-7, controlled non-IID checkpoints after rounds 2, 5 and 20
were selected before execution. They share the same two-million-row training
cohort, 20 clients, assignments and 5,096-parameter LayerNorm model. Round 5 is
an available intermediate snapshot, not a newly trained midpoint. Each source
model independently produces one ordinary local-training round; corrected
weights are never fed into another round.

Clients retain the historical full-epoch reset-Adam recipe and original
batch-weight loss denominator. Their updates are aggregated by Flower using
assigned row counts. At the common starting weights, full-network gradients
of each class's mean cross-entropy are computed over all of its training
examples. Class gradient sums are pooled by their class counts, not by a
uniform average of client means. All eight logits participate.

The counterfactual chooses the closest parameter displacement to FedAvg with
nonpositive gradient dot products for Benign, Web-based and Brute Force. These
are deterministic **evaluation-mode training losses**, with dropout disabled
for the gradient probe. They are not the stochastic local Adam objective or
a validation objective. All eight class gradients are computed for diagnosis;
only three would be required by the proposed correction.

The dedicated loader opens only training arrays. No validation or test rows
are opened. These are fitting diagnostics, not generalization results and not
an additional evaluation of the completed final test.

## Brute Force learning is pushed backwards by the combined update

Positive loss change means worse fit; negative means better fit. Values below
are actual finite-update changes in mean training cross-entropy, not percentages.

| Source round | Ordinary Brute Force change | Corrected Brute Force change | Corrected benign change | Retained update norm |
| --- | ---: | ---: | ---: | ---: |
| 2 | +0.656296 | −0.135379 | −0.001913 | 99.6725% |
| 5 | +0.033229 | −0.030514 | +0.001951 | 99.9972% |
| 20 | +0.049922 | −0.002446 | +0.000263 | 99.9923% |

The normal update's first-order prediction also worsened Brute Force loss in
all three cases. At round 20, nine clients hold Brute Force examples and eleven
do not. Their respective sums of row-weighted directional contributions are
approximately **−1.7301** and **+1.7808**, leaving a net **+0.0507**. Thus helpful
directions from holders are slightly outweighed by the other group at this
shared model. This is a directional decomposition using the pooled Brute Force
gradient, not evidence that every holder is helpful or every other client is
harmful. Removing all negative evidence from nonholders would be unjustified.

Web-based loss improves under the correction at all three states. Ordinary
FedAvg already improves it at rounds 5 and 20, so the evidence does not support
calling every rare-class failure an aggregation reversal.

## The safeguard does not yet protect actual benign loss

At rounds 5 and 20, the projection places the benign directional derivative
approximately on zero. Nevertheless, its actual finite step raises benign loss
by 0.001951 and 0.000263. The first-order approximation misses curvature or
other finite-step nonlinear effects. The predeclared tolerance was 0.000001;
it was not relaxed after the result. Only round 2 passes the full useful-step
criterion, whereas the plan requires at least two useful states and safety at
all three. The recorded decision is `stop_no_full_training`.

Benign loss is also not interchangeable with false-alert rate. For example,
after the round-20 counterfactual, corrected training false alerts are 0.4305%
versus ordinary FedAvg's 0.4324%, even though corrected benign loss increased
relative to the starting model. The gate concerns the proposed loss safeguard,
not a demonstrated deterioration on every possible metric.

Neither one-step update correctly classifies any Web-based or Brute Force
training examples at these checkpoints. Better loss can precede a change in
the winning class, but **no such future improvement has been demonstrated**.

All eight training recalls after the round-20 source, in percent:

| Class | Starting model | Ordinary next round | Corrected counterfactual |
| --- | ---: | ---: | ---: |
| Benign | 99.5930 | 99.5676 | 99.5695 |
| DDoS | 99.0514 | 99.0274 | 99.0269 |
| DoS | 17.5116 | 18.0709 | 18.0753 |
| Recon | 21.3634 | 21.6727 | 21.6651 |
| Web-based | 0.0000 | 0.0000 | 0.0000 |
| Brute Force | 0.0000 | 0.0000 | 0.0000 |
| Spoofing | 0.2109 | 0.8618 | 0.9298 |
| Mirai | 98.7824 | 98.8607 | 98.8611 |

Corresponding macro-F1 is 0.399746, 0.403070 and 0.403243. The corrected
counterfactual's small difference from ordinary FedAvg is not evidence of
substantial improvement or a substitute for a multi-round independent study.

## A bounded idea suggested by the result

A potential next design is **finite-step-aware class safeguarding**: reserve
strict descent room rather than landing exactly on a zero-slope boundary, then
ask clients to check the actual protected training losses before committing
the server step. Raw rows would remain local; clients would return aggregated
loss sums and counts. Failed proposals would be rejected rather than silently
declared safe by their tangent approximation.

This is a hypothesis, not an implemented second method. Shrinking a step alone
may fail when the first-order derivative is zero, and an acceptance check can
reject every useful step. Any follow-up needs a bounded acceptance rule and
explicit limits on retries, communication, extra loss evaluations and stalling.
It also needs matched centralized and established federated controls before
claiming a federated-specific advantage. No new loss margin, backtracking
sweep or full run has been launched to rescue this failed gate.

The immediate contribution is a directly measured mechanism and an identified
failure of the first safeguard, not a novel high-performance classifier.
The gradient-projection prior art and novelty limitations remain as described
in the [archived proposal](../../archive/proposals/reports/Federated%20gradient%20correction%20proposal.md).

## Verification and cost

The [plan](diagnostic-plan.json) was committed before any successful real-data
diagnostic. The first attempt stopped at a source guard before training: a
later optional ablation path had changed the runner's whole-file checksum.
Inspection of commits `1d767e2` and `5cc8206` confirmed unchanged default local
training and exact ASTs for the imported training and metric functions. A
narrow, hash-pinned compatibility bridge was committed before rerunning;
unrecognized changes still fail. No archived code is executed.

The successful three-checkpoint run and independent replay match on **every
non-timing result**, including model hashes, per-class gradients' directional
effects, losses, confusion counts and work counters. This is CPU replay
reproducibility, not bitwise CPU versus historical CUDA training equivalence.

Each pass processes 6,000,000 client-training examples in 11,751 Adam updates,
6,000,000 gradient-probe examples and 18,000,000 forward-evaluation examples.
Including replay doubles those quantities. Extra gradient and evaluation work
is not counted as extra optimizer steps. The first pass took approximately
16 seconds, overlapping repository tests; this is not a clean timing benchmark.
The proposed three summaries add 61,152 FP32 payload bytes per client per round;
the diagnostic's eight summaries correspond to 163,072 bytes. These are
calculated payload sizes, not measured network traffic. Inference is unchanged.

All 588 tests passed in an isolated committed snapshot, including 13 new
numerical, source-bridge, loader-isolation and deterministic-replay tests and
four public-evidence checks. The final full-suite run took 53.622 seconds.
The aggregate [results](diagnostic-results.json) contain no traffic rows or
model weights. Historical checkpoints and the original archive are unchanged.
Unrelated working-tree changes were preserved, and no cloud computation or
push was performed.

Reproduce with the existing private artifacts and a fresh output directory:

```bash
.venv/bin/python -m src.eval.federated_class_safeguard \
  --data-root outputs/official39-packed-2m-v1 \
  --archive /Users/haydn/Downloads/tradecraft-official39-controlled-noniid-backup.tgz \
  --plan-path reports/federated_class_safeguards/diagnostic-plan.json \
  --output outputs/federated-class-safeguard-fresh-run
```
