# Screening complete: final dropout comparison and confirmation handoff

All **12 of 12** predeclared local screening runs are complete. The final heavy-dropout run finished on 2026-10-05. No extra learning-rate screens, confirmation runs, test evaluations or cloud jobs were launched. Training code, comments, data and the frozen screening plan are unchanged.

## Final experiment: heavy dropout 0.3 versus 0.2

Both models used BatchNorm, square-root-weighted cross-entropy, model seed 0, the same data and 60 epochs. Each checkpoint was selected by its own best validation macro-F1.

| Metric | Dropout 0.3 | Dropout 0.2 |
|---|---:|---:|
| Best epoch | 50 | 53 |
| Macro-F1 | 0.6769 | 0.6758 |
| Benign false alerts | 20.35% | 20.41% |
| Web recall | 33.44% | 42.68% |
| Brute Force recall | 25.27% | 36.46% |
| Spoofing recall | 80.94% | 75.51% |

Dropout 0.2 is not uniformly worse: it improves Web recall by 9.24 percentage points, Brute Force by 11.19, DoS by 3.37 and Recon by 4.85. But Spoofing recall drops by 5.43 points, and DDoS by 1.97. Neither predeclared promotion criterion passes. The 0.11-point macro-F1 difference alone is not evidence of meaningful superiority for either configuration.

There is also a precision cost: Web precision changes from 23.05% to 20.47%, and Brute Force precision from 30.30% to 21.17%. Higher rare-class recall is useful evidence, not proof that the model has become a better operational detector. Retain dropout 0.3 under the existing gates; preserve dropout 0.2 as a documented trade-off rather than changing the gates after observing results.

## Findings across all screens

1. **Training duration matters.** The IID control improved beyond 30 rounds; more rounds alone did not rescue BatchNorm non-IID. Convergence is not established for all lanes.
2. **Keep weighted loss.** None of the four unweighted-CE replacements passed promotion. Several severely reduced Web detection and increased false alerts.
3. **LayerNorm is a federated candidate.** IID and non-IID passed criterion B (false-alert-focused), not criterion A (all-class recall protection). Non-IID macro-F1 improved from 0.4706 to 0.5705, but Web recall remains 4.30%, Brute Force 14.80% and DoS 22.51%.
4. **Central changes have trade-offs.** Neither central-light LayerNorm nor heavy dropout 0.2 passed promotion. Do not choose solely by macro-F1 or assume separate changes combine additively.
5. **These are insights, not a deployment result.** Rare-class detection remains weak and false alerts remain substantial in several models. Single-seed validation selection cannot establish robustness or significance.

## Proposed next stage: freeze the comparison before confirmation

The gate-consistent shortlist is:

| Lane | Loss | Normalization | Dropout |
|---|---|---|---:|
| Central heavy | Square-root-weighted CE | BatchNorm | 0.3 |
| Central light | Square-root-weighted CE | BatchNorm | 0.2 |
| Federated IID | Square-root-weighted CE | LayerNorm | 0.2 |
| Federated non-IID | Square-root-weighted CE | LayerNorm | 0.2 |

This is a **proposal, not an enacted confirmation configuration**. It supports a practical comparison of selected configurations, but normalization differs between central and federated light, and dropout differs between heavy and light. Consequently it cannot isolate a pure federation effect or pure capacity effect. The existing matched BatchNorm and LayerNorm seed-0 screens should accompany it as exploratory controls, not be mislabeled multi-seed confirmation.

Before confirmation, choose the scientific claim: confirm the gate-consistent configured-model shortlist, or explicitly amend the plan for matched normalization/dropout across models. A matched design is defensible, but is a different selection rationale; do not silently promote a failed candidate or add an untested heavy-LayerNorm combination. The current 12-run screening allowance is exhausted.

After that choice, record a frozen confirmation plan before launch: model seeds 0/1/2, fixed partition seed 0, unchanged train/validation boundary, checkpoint rule and training budgets, at most 12 runs. Report mean and sample SD, every class recall and false-alert rate. Seed 0 already influenced selection and is not independent confirmation; highlight the new seed-1/2 results separately. Fixed partition seed means this measures training-seed variation, not variation across client partitions.

No test access is needed for that stage. Historical test scores have already been inspected and cannot be turned into fresh confirmation. Final SHAP/policy synthesis should follow recipe freezing and stability checks. Do not spend RONIN credits merely to enlarge model capacity at this point.

## Artifacts and verification

- [All 12 scores, final dropout recalls and curves](results/README.md).
- [Full machine-readable metrics and provenance](results/summary.json), including every class precision/recall/F1 and confusion matrices for all 12 runs.
- [Analysis script](analyze.py): checks every completed run against the frozen plan, verifies checkpoint/scaler/partition hashes, matching source/data/packages, selected metrics, and declared-only paired configuration differences.
- Final run: `outputs/mlp-tuning/heavy_dropout/heavy`; earlier checkpoints and histories remain in their original stage directories (Git-ignored).

The dependency check, artifact audit, script compilation and whitespace check passed. No training-source changes were needed. CPU timing is descriptive, not a causal speed comparison or physical edge measurement. Existing uncommitted reports are preserved; nothing was committed or pushed in this stage.

Regenerate generated tables into a new directory, from the repository root:

```sh
.venv/bin/python reports/screening_2026_10_05/analyze.py --output /tmp/tradecraft-screening-regenerated
```

The script reads saved artifacts only, not datasets or model objects. This manually authored interpretation remains separate from generated tables. The training command is the heavy-dropout command printed by `python -m src.eval.tuning_plan`; use a new output directory or the research runner's documented matching-checkpoint resume workflow.
