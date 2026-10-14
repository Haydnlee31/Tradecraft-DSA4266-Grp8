# Assessment: working research pipeline, not a deployment-ready detector

All 12 planned training runs completed on the existing sampled splits: three seeds
each for centralized-heavy, centralized-light, federated IID and federated
Dirichlet alpha=0.5. See [RESULTS.md](RESULTS.md) for the complete measured tables,
[decision_summary.md](decision_summary.md) for validation-based gates and
[recall_comparison.png](recall_comparison.png) for the class-level comparison.

## Are these useful results?

Yes—as evidence about trade-offs and failure modes, not as evidence of a successful
federated security deployment.

| Configuration | Test macro-F1, mean ± seed SD | Main finding |
| --- | ---: | --- |
| Centralized-heavy | 0.6580 ± 0.0126 | Strongest configured baseline, still weak on rare attacks |
| Centralized-light | 0.6466 ± 0.0036 | 35.9× fewer parameters for a 0.0114 mean macro-F1 reduction |
| FL IID control | 0.5927 ± 0.0066 | Federation/optimization introduces a gap even without intended label skew |
| FL non-IID | 0.4412 ± 0.0304 | A further 0.1515 reduction relative to IID; severe class collapse |

These are descriptive differences across three seeds, not statistical proof of
equivalence or superiority across independent datasets. Heavy/light also differ in
depth and dropout, not only parameter count. The IID assignment uses every Nth row;
it is an approximately IID control, not a device-based federation.

The most important operational problem is **false alarms**: both centralized
models classify about 26% of benign flows as attacks. Heavy's Web-based recall is
only 25.4%, and Brute Force recall is 30.9%. Its 99.4% pooled binary attack-detection
recall hides wrong attack categories and is dominated by common attacks. Neither
high binary recall nor the research decision's `ready` status justifies deployment.

Non-IID FL averages only 8.0% DoS recall, 0.6% Web-based recall and 6.3% Brute Force
recall. About 91.6% of DoS flows become DDoS predictions. Around 68.1% of Web-based
flows become Benign predictions. This is more than a modest accuracy-size trade-off.

## Should the architecture change?

Keep the overall three-lane design, shared preprocessing, eight-class target and
validation-only selection. They now produce an informative comparison. Do not
replace everything with a bigger network: the light model is already quite close
to heavy, and increasing capacity has not solved the rare-class/false-alarm problem.

Revise the training and decision design in controlled steps:

1. **Specify the decision's actual cost constraints.** Agree on acceptable benign
   false-alert rate and minimum recall for each attack category before the next
   sweep. The current 0.60 macro-F1 / 0.20 class-recall gates are illustrative.
   Develop any confidence threshold or abstention/manual-review policy on validation
   only. Evaluate calibration rather than treating softmax confidence as trustworthy.
2. **Check FL convergence before changing network size.** IID seeds 0 and 1 and
   non-IID seed 1 selected round 30, the maximum. Run a validation-only longer-budget
   experiment with a predeclared patience and compare learning curves. Equal
   row passes do not imply equal optimizer updates: FL averages independently
   optimized client models and resets Adam's moments each round.
3. **Run rare-class loss/sampling ablations on the same splits.** Compare current
   sqrt-weighted CE with CE, weighted CE and focal loss already supported by the
   repo. Change one factor at a time, use the same seeds, and evaluate benign false
   alarms alongside rare-class recall. Do not simply maximize rare-class recall at
   any false-positive cost.
4. **Then test a normalization/FL-optimization ablation.** For example, compare
   BatchNorm with LayerNorm in both light lanes while keeping widths and dropout
   fixed. Local normalization methods such as FedBN are a separate experimental
   setting, not a drop-in global-model fix. The [FedBN paper](https://arxiv.org/abs/2102.07623)
   motivates studying normalization under feature shift; it does not prove it will
   fix this label-skew experiment.
5. **Validate explanation stability before producing security rules.** Repeat
   feature rankings across backgrounds, training seeds and correctly/incorrectly
   classified validation cases. Keep any resulting policy at the flow/protocol
   level; there are no device identifiers here.

### What the normalization diagnostic actually showed

The validation-only [normalization audit](normalization_audit.json) recalibrated
BatchNorm running statistics on pooled training features without changing learned
weights. Non-IID seed 0 changed from **0.4706 to 0.4754** validation macro-F1; IID
seed 0 changed from **0.5856 to 0.5862**. Centralized scores slightly worsened.
This does not rescue FL, so blaming stale running statistics alone is unsupported.
It does not rule out training-time normalization effects. This uses server access
to pooled data solely as a diagnostic, not as a privacy-compatible federated fix.
No original checkpoint was overwritten and no test data entered this diagnostic.

## Explainability: completed, but exploratory

Four seed-0 SHAP audits are in [explanations/](explanations/). Each uses 128 natural-
prevalence training background rows and 16 validation examples per true class,
including mistakes. They explain class logits using
[SHAP expected gradients](https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html).
This is approximate attribution, not an exact or causal explanation.

For heavy seed 0, SSH is among the leading Brute Force features, while Number,
IAT and Weight lead the Web-based ranking. That supports hypotheses about what
telemetry to inspect; it does **not** establish a direction, a deny threshold or
that those features truly identify an attack. Several correlated aggregate
statistics dominate, and rare-class predictions are unreliable. Mean absolute
additivity residuals range from 0.084 to 0.624 logits across the four audits, so
numerical approximation error is explicitly recorded rather than hidden.

Policy guidance for now: use independent application/authentication telemetry to
investigate rare attacks, and treat model alerts as analyst-review candidates.
Do not automate blocking from these models or their feature rankings.

## Verification and environment

- Local `.venv` installed; `pip check` reports no broken requirements.
- macOS Apple Silicon, Python 3.13.9; exact tested packages in
  [requirements-macos-py313.lock](../../requirements-macos-py313.lock).
- **10 automated tests pass**, including real training, scaler equality, Flower
  aggregation, test-independent decision gates and failed-client handling.
- Actual **Ray worker smoke test passes** through the Python simulation API:
  two client updates, zero failures, server aggregation, checkpoint and report.
  The API is deprecated upstream; the pinned installed version still supports it.
- The Flower 1.38 **CLI path was not successfully validated**: it creates a second
  runtime dependency environment and stalled in `uv sync` for the 240-second
  smoke-test budget. The test run was stopped. This is not evidence of a model or
  client-training failure. The tested full-study entry point does not require it.
- Training used all **440,594 train / 327,257 validation / 325,954 test** sampled
  rows, subject only to a one-row BatchNorm tail being dropped where necessary.
  There are 46 finite numeric features. The hash-based exact-feature overlap
  screen found zero cross-split matches. This does not establish session/device
  independence or audit the upstream split procedure.
- Parameter sizes are measured from tensors. No physical edge latency, power,
  runtime RAM or network traffic was measured. Sequential FL training timings
  must not be described as distributed-system performance.

The decision layer currently recommends heavy under the illustrative validation
gates. Light misses the Brute Force validation-recall gate; FL misses macro-F1 and
multiple class-recall gates. This is a research comparison outcome, not deployment
approval. Preserve these results as baseline evidence; further tuning should use
validation and a fresh independent final evaluation if confirmatory claims are needed.
