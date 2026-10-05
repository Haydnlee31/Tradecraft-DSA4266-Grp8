# Validation-only decision integration completed

The decision layer now accepts the frozen configured-model study without requiring or inventing test scores. The legacy entry point remains available, including descriptive test metrics for older evaluated runs. All existing illustrative research thresholds are unchanged.

## Result under the existing gates

| Candidate | Mean validation macro-F1 | Lowest mean class recall | Research eligibility |
|---|---:|---:|---|
| Central heavy | 0.6747 | 25.51% (Brute Force) | Pass |
| Central light | 0.6583 | 16.49% (Brute Force) | Fail: Brute Force recall |
| Federated non-IID | 0.5557 | 1.43% (Web) | Fail: macro-F1 and DoS/Web/Brute Force recall |

**Centralized-heavy is the gate-based research choice, not an operational deployment approval.** IID remains a diagnostic control, not a replacement for the non-IID target. When federated training is required, the decision correctly returns **no recommendation**.

The inherited gates are: square-root-weighted CE, mean validation macro-F1 ≥0.60, every class's mean validation recall ≥0.20, saved parameter tensors ≤1 MiB, at least three seeds, and preference for smaller eligible models within 0.02 macro-F1 of the best eligible model. These are absolute decision gates; they are different from the earlier screening promotion gates A/B.

The light model's smaller size and near-heavy aggregate score do not override its failed recall gate. No threshold was relaxed to force a compact or federated model to pass.

## Important qualifications

- Seed 0 was reused from screening. The all-seed view satisfies the existing count of three training seeds, but does not represent three independent confirmation runs. The separate seed-1/2 view preserves the same three-seed requirement, reports `incomplete`, and gives heavy only as a provisional research choice.
- The gate uses **mean** recall. Heavy seed 2's Web recall is 19.75%, below the 20% threshold even though the three-seed mean passes. Per-seed warnings now expose this instead of silently treating mean eligibility as robustness.
- The inherited gates contain **no false-alert ceiling**. Heavy's mean benign false-alert rate is approximately 21.97%, so eligibility must not be read as operational acceptability. A future operational gate would need an explicitly chosen cost/alert-budget criterion; none was invented here.
- This is the approved configured-model comparison. Normalization differs centrally versus federally and dropout differs heavy versus light; it does not isolate pure federation or capacity effects.
- Parameter memory (0.7591 MiB heavy, 0.0211 MiB light) is not runtime RAM, latency, power or physical edge feasibility.

## Error types and policy handoff

Every attack-class confusion row is now explicitly partitioned into:

1. Correct attack category.
2. Attack predicted as benign.
3. Attack predicted as another attack category.

The categories sum to the full class support. Benign false alerts are reported separately for every seed. This supplements—not replaces—the locked 8-class metrics and gates. It prevents treating DoS→DDoS mistakes as equivalent to Web→Benign misses.

The report carries the [SHAP audit's human-review hypotheses](../explanations_2026_10_05/README.md): retain independent web/authentication monitoring, review Recon/Spoofing alerts in context, distinguish uncertain attack subtypes, and never translate feature rankings directly into blocking thresholds or device-vulnerability claims. SHAP was not added to a numerical eligibility score; no automatic security policy was activated.

## Engineering changes

- [Existing decision engine](../../src/eval/decision.py): accepts genuinely absent test metrics; displays `N/A`; does not aggregate a partial test-seed subset as if complete. Its misleading “worst recall” column now explicitly shows validation recall, matching the gate. Existing comments and legacy tests are retained.
- [Frozen-study adapter](../../src/eval/research_decision.py): verifies the plan/launch receipt, raw result and checkpoint hashes, manifests, selected steps, full histories, saved scaler/partition hashes and per-seed configuration consistency. It recomputes macro-F1, accuracy, recall and false alerts from confusion counts before decision-making. It reads only saved artifacts, not datasets or model objects.
- [New regression tests](../../tests/test_research_decision.py): exercise null/mixed test metrics, unchanged criteria, IID exclusion, missing seeds, configuration/partition drift, nonfinite/inconsistent metrics, error-count partitioning and mandatory-federated failure.

**All 40 repository tests passed.** Dependency and whitespace checks passed. The default strict CLI returned exit 0 (research gates met), while the mandatory-federated strict CLI returned the expected exit 2 (no eligible federated choice). Synthetic unit-test data are separate from the project's real test split.

No new model training, real test evaluation, cloud resources, commits or pushes were performed. Earlier reports remain intact; the initially generated report from this implementation stage was regenerated after adding the per-seed warnings.

## Outputs and reproduction

- [Default research decision and full per-class/error tables](results/README.md), [JSON](results/decision.json).
- [Mandatory-federated scenario](federated_required/README.md), [JSON](federated_required/decision.json).

Run from the repository root into new output directories:

```sh
.venv/bin/python -m src.eval.research_decision --output outputs/decision-default --strict
.venv/bin/python -m src.eval.research_decision --output outputs/decision-federated --require-federated --strict
```

The second command deliberately exits 2 with the current frozen results. Neither command evaluates test data. The original `python -m src.eval.decision` workflow remains supported for legacy reports.

## Next step

The local study now connects screening, frozen-seed confirmation, failure-focused explanations and an auditable decision. The next deliverable is a consolidated project write-up with the configured-model trade-off, limitations and reproducible commands. Improving rare-attack detection or changing operational gates should be a separately agreed experiment/design stage, not an unrecorded continuation of this completed study.
