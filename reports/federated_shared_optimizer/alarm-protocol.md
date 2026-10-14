# Compare saved models at common false alert targets

The shared-gradient model nearly matched centralized macro-F1, but raised far more benign alerts than FedAvg. This diagnostic asks whether its recognition advantage remains when both methods are calibrated to the same alarm target. It uses saved weights, no retraining, no new classifier and no human-review layer. The [original failed continuation criteria](performance-findings.md) remain unchanged.

## Fixed models and boundaries

Use the five previously selected methods and learning rates, identified by the existing performance receipt and checkpoint hashes. Do not select another learning rate after inspecting alarm-matched results. Split the existing development inputs approximately in half using a fixed hash of each standardized feature vector. Identical float32 vectors stay together. The hash does not depend on labels or model predictions.

Use benign rows from the calibration half to choose thresholds; use the other half to measure outcomes. Both halves have already contributed to the earlier learning-rate selection, so this is exploratory development analysis, not a fresh holdout test. Do not open fitting arrays, reserved confirmation arrays or the original validation/test arrays for this diagnostic.

## One automatic decision adjustment

For each flow, subtract the benign logit from the largest attack logit. If this margin exceeds a threshold, predict that highest-scoring attack category; otherwise predict Benign. At threshold zero, this reproduces the original argmax decision, with ties going to Benign. Changing the threshold is equivalent to adding one constant offset to the benign logit. It cannot repair confusion between two attack categories, because their ordering never changes.

Freeze targets of 1%, 5%, 10% and 20% calibration benign false alerts, with **5% as the primary comparison**. For a target expressed in basis points, let `k = floor(n * target / 10000)` for the `n` calibration benign scores. Choose the score at ascending index `n-k-1`; predict an attack only for a strictly greater score. This permits at most `k` calibration false alerts, including when scores tie. No interpolation or random tie breaking is used.

Thresholds depend only on calibration benign scores and the specified budget, not on attack labels, macro-F1 or audit labels. Freeze them before computing audit metrics. This empirical calibration cap does not guarantee a population or deployment false-alert cap. Report the actual audit FAR; never tune again on the audit half to force exact matching. Flag method pairs whose actual audit FAR differs by more than 0.5 percentage points.

## Outputs and interpretation

Report all eight classes' precision, recall and F1, confusion matrices, macro-F1, benign false alerts and thresholds. Also distinguish identifying an attack's correct category from merely flagging it as some attack. Both are useful diagnostics, but only the former satisfies the eight-class target. Keep zero-recall classes visible.

Report the original zero-threshold outcomes on the same audit rows as a reference, and verify that full-development zero-threshold predictions reproduce the saved performance receipt exactly. Preserve inference logits privately for deterministic audit; do not commit raw rows, predictions or model weights. Publish aggregate results, split fingerprints, code/data/checkpoint hashes and an independent replay of the saved logits using the frozen thresholds.

The primary interpretation compares each federated method with the selected centralized-light and FedAvg controls at the 5% calibration target, showing the other targets as sensitivity checks. It is not a new method-selection gate, a claim of statistical significance or permission to score confirmation. Original training work and modeled communication costs do not disappear when changing a decision threshold. Freeze this protocol and implementation before producing the comparison.
