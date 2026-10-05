# Failure-focused SHAP audit: explanations do not remove the detection gaps

Completed on 2026-10-05 for all **12 frozen checkpoints** (four configurations × three seeds). Every checkpoint reproduced its original full-validation confusion matrix before explanation. The real test split was not read; no retraining, cloud resources, automatic blocking rules, commits or pushes were used.

## Main findings

**1. Shared feature rankings can be stable even when detection is weak.** `IAT`, `Number` and `Weight` repeatedly appear among leading true-class-logit features for Web and Brute Force across models. Heavy Web's top-five set agrees across all three seeds on the shared core, despite its large Web-recall variation. This is evidence of similar model reliance on these sampled rows, not evidence that the model detects those attacks reliably.

`Weight`, `Number` and `Magnitue` are dataset feature names; do not infer their exact extractor semantics or turn them into network-control thresholds from their names alone. The explanation audit does not establish that these are causal attack signatures, or that higher/lower raw values imply malicious traffic. Correlated flow statistics and artificial interpolation paths can affect attribution.

**2. Non-IID failures include genuinely missed malicious traffic.** Full-validation counts—not estimates from the SHAP sample—show:

| True attack | Non-IID seed 1 predicted benign | Non-IID seed 2 predicted benign |
|---|---:|---:|
| Web | 358 / 628 (57.01%) | 443 / 628 (70.54%) |
| Brute Force | 183 / 277 (66.06%) | 188 / 277 (67.87%) |
| DoS | 4 / 61,572 (0.01%) | 4 / 61,572 (0.01%) |

Zero correct Web classifications does **not** mean every Web attack was called benign: many were assigned Recon/Spoofing instead. Nevertheless, the benign assignments above are serious misses. Low benign false-alert rates do not offset them.

**3. DoS/DDoS confusion is a different kind of failure.** Non-IID seeds 1/2 label 52,107 and 52,738 of 61,572 DoS rows as DDoS. These are attack-category errors, not benign acceptance. Keep the locked 8-class training/evaluation target, but distinguish category errors from attack-to-benign misses in downstream decision reporting. A secondary attack-versus-benign summary would supplement—not replace—macro-F1 and every class's recall.

**4. Recon explanations are less stable under non-IID.** Mean pairwise top-five overlap across training seeds is only 33.3% for non-IID Recon, with no feature appearing in all three seed top-five sets. A single Recon feature narrative would overstate this evidence. In contrast, several class/model combinations reach 80–100% overlap on the same small core.

**5. Individual explanations need numerical caution.** Across models, mean absolute additivity residual divided by mean absolute output difference is 5.4–8.8%. Mean top-five overlap between two Monte Carlo repeats is 85–97.5% across classes. These support exploratory feature-group discussion, not exact local explanations: some error-case residuals are comparable to or exceed the observed predicted-versus-true logit margin. For example, IID seed 1's sampled missed Web row has margin 0.10 logits but residual −0.79 logits. Do not interpret that row's attribution as a precise explanation of the boundary crossing.

## Policy hypotheses supported by this evidence

These are proposals for investigation, not validated deployment rules.

| Evidence | Proposed analyst/policy action | Required safeguard |
|---|---|---|
| Web/Brute Force are often mapped to benign by non-IID | Retain independent application/web and authentication monitoring; do not let this model's benign output suppress those alerts | Evaluate missed-attack and false-alert costs with appropriate logs before any automation |
| Benign false alerts mainly become Recon/Spoofing | Route those predictions to contextual review rather than automatic blocking | Establish service-specific normal behavior and assess alert burden |
| DoS is frequently called DDoS | Preserve an attack-alert signal while treating the subtype as uncertain | Keep the original 8-class result visible; do not claim subtype correctness |
| Recurrent flow-statistic attributions | Prioritize recording/inspecting these existing flow features alongside protocol counters and application context | Verify extractor definitions, assess correlation and data-processing artifacts; no SHAP-derived thresholds |
| Non-IID Recon rankings vary by seed | Avoid a single-feature Recon rule | Require explanation sensitivity checks across backgrounds/samples before stronger claims |

Nothing here supports device-specific vulnerability claims: the data contain no device identity. SHAP explains this model's behavior relative to the chosen background; it does not establish a causal security policy.

## Design and coverage

- **Background:** the same 128 seeded uniform rows from the sampled training split for every model. This draw has no Web or Brute Force rows and only four benign rows. It is not a benign-only baseline or a reference covering every class.
- **Core:** the same eight seeded validation rows per true class, 64 total. Only this balanced core is used for class feature rankings and cross-seed overlap. It is not population-weighted.
- **Supplement:** one seeded correct prediction, false alert and missed Web/Brute Force/DoS example per checkpoint, combined into a shared 60-row supplement. Every checkpoint explains the same total 124 rows. Empty groups are explicitly reported rather than inventing successful detections.
- **Method:** SHAP expected gradients on all eight class logits; two Monte Carlo seeds, 256 integration samples each, with averaged attributions and both repeats retained. For mistakes, signed predicted-minus-true logit attributions are also saved. A negative contribution opposes that wrong preference relative to the background, even if the final prediction is wrong.
- **Stability:** training-seed top-five overlap and numerical-repeat overlap are separate diagnostics. Neither is causal validity or statistical significance. Independent background/sample sensitivity was not run.
- **Identity:** sample row indices, source-file names, raw labels, selection reasons, checkpoint hashes, split hashes, package versions and explanation-code hash are saved.

Method reference: [SHAP GradientExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html). The method approximates output differences relative to a background; finite-sample residuals are recorded, not hidden.

## Engineering work and verification

Added [the research explanation runner](../../src/explain/research.py) without changing the legacy explainer or training/model code. It loads each verified saved scaler, enforces checkpoint schema and normalization, checks inference-source/data hashes, and rejects predictions that do not reproduce the original validation confusion matrix.

Added [tests](../../tests/test_explain_research.py) for both normalization types, checkpoint corruption rejection, deterministic shared sampling and the expected-gradients linear-model contract. **All 34 repository tests passed**, as did the dependency check. Unit tests use synthetic temporary data; they do not evaluate the project's real test split.

One separate integration issue was identified: the legacy decision-report loader in `src/eval/decision.py` expects old-style fields and populated test metrics, so it is not directly compatible with these validation-only research results. It was not invoked or changed. Adapt its input contract before using it for a new decision report; do not access test data merely to satisfy that legacy schema.

## Next stage

The bounded explanation stage is complete, with useful failure insights but not deployment-ready policy evidence. The next engineering step is a **validation-only decision/report integration**: ingest the frozen confirmation outputs, preserve per-class metrics and seed caveats, distinguish attack-to-benign misses from category errors, and carry these policy hypotheses as human-review recommendations. Existing gates must not be weakened to force a model to pass.

Before stronger local SHAP claims, use a separately specified background/sample-sensitivity audit and improve numerical integration for low-margin cases. If improving non-IID rare-attack recall is required, open a new predeclared experiment stage focused on the observed failures; do not enlarge the heavy model or overwrite the completed study by default.

## Artifacts and reproduction

- [Numerical diagnostics, all feature rankings, failure destinations and examples](results/README.md).
- [Complete explanation summary](results/summary.json) and [sampling identities](results/sampling.json).
- Raw attribution arrays, both Monte Carlo repeats and standardized sampled inputs: `outputs/explain-confirmation-2026-10-05` (Git-ignored).
- [Report compiler](summarize.py).

Run from the repository root, using a **new** output directory:

```sh
.venv/bin/python -m src.explain.research --output outputs/explain-confirmation-rerun
.venv/bin/python reports/explanations_2026_10_05/summarize.py --input outputs/explain-confirmation-rerun --output /tmp/tradecraft-explanations-regenerated
```

Do not load untrusted checkpoints/scalers. The original frozen model results, prior reports and comments are preserved. This interpretation is separate from the generated numerical report.
