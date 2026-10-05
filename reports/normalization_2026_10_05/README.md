# Normalization screen: promising for federated learning, with important recall trade-offs

Completed the three planned LayerNorm runs locally on 2026-10-05. The paired controls use BatchNorm. Weighted loss, model seed 0, partition seed 0, data, optimizer settings and 60-step budgets were held fixed. No test evaluation or cloud resources were used; training code and comments are unchanged.

## Results

Scores are best validation macro-F1 within 60 epochs/rounds, not multi-seed estimates or test results.

| Model | BatchNorm | LayerNorm | Gain (percentage points) | Promotion criterion |
|---|---:|---:|---:|---|
| Central light | 0.6576 | 0.6761 | +1.85 | Neither |
| Federated IID | 0.6140 | 0.6339 | +1.99 | B only |
| Federated non-IID | 0.4706 | 0.5705 | +9.98 | B only |

Criterion A protects every class's recall (no drop above 2 percentage points) alongside aggregate improvement. Criterion B prioritizes a false-alert reduction of at least 5 points while protecting macro-F1 and Web/Brute Force recall. Passing B does **not** mean every class improved, or that the detector is deployment-ready.

- **Central light:** false alerts fall from 22.66% to 19.52%, but DoS recall falls from 77.16% to 69.23%. DDoS recall improves from 80.57% to 92.85%. The macro-F1 increase does not satisfy either predefined gate.
- **Federated IID:** false alerts fall from 31.50% to 23.88%; Web recall rises slightly from 11.62% to 12.26%, and Brute Force recall stays at 14.80%. DoS recall falls from 83.06% to 75.86%. This passes B, but fails A.
- **Federated non-IID:** false alerts fall from 12.74% to 4.78%. DoS recall recovers from 0.13% to 22.51%, Recon from 34.83% to 53.71%, and Web from zero to 4.30%. DDoS recall falls from 99.93% to 97.20%, causing A to fail; B passes. Web, Brute Force (14.80%) and DoS detection remain weak.

## What this tells us

Normalization choice materially changes this simulated non-IID result. The centralized-light minus non-IID macro-F1 gap narrows from **18.70 to 10.56 percentage points** when both use LayerNorm. The IID gap changes only slightly, from 4.36 to 4.22 points: its aggregate improvement largely tracks the centralized improvement.

This supports shortlisting LayerNorm for federated confirmation, not declaring federated training equivalent to centralized training. The experiment changes normalization as a whole; it does not isolate whether BatchNorm running statistics, batch-dependent training behavior, or other consequences cause the difference. It is not a FedBN experiment.

All models still have 5,544 trainable parameters, and processed examples and optimizer-step counts match within each pair. LayerNorm does not add capacity here. Best LayerNorm checkpoints occur late (light/non-IID 58, IID 59), so convergence is not established by these 60-step screens.

The single fixed client partition and model seed limit generalization. Web and Brute Force have only 628 and 277 validation examples. Multi-seed confirmation and full per-class reporting remain necessary; a gate pass is a screening preference, not statistical significance.

## Next stage

Finish the **one remaining predeclared screen**: heavy-model dropout 0.3 versus 0.2, retaining BatchNorm and weighted loss to isolate dropout against the saved heavy control. Do not silently combine this with LayerNorm or add learning-rate experiments beyond the 12-run screening cap.

Then review all screens and freeze the confirmation recipe. LayerNorm is shortlisted for federated models under criterion B, with the DoS trade-off explicitly retained. Central-light LayerNorm has not passed the gates and must not be silently promoted to create a matched comparison. If the final heavy/light recipes differ in normalization or dropout, describe them as configured-model comparisons, not a pure capacity effect. A new matched-recipe study would need an explicit plan before running it.

Eleven of twelve screening runs are complete; no confirmation runs or heavy-dropout run were launched in this stage. Keep final SHAP/policy conclusions until the model recipe is frozen and confirmed. These results do not warrant claiming rare-attack detection is solved or increasing cloud spending.

## Artifacts and reproducibility

- [Detailed metrics and curves](results/README.md): all class recalls, false alerts, federation gaps and measured compute.
- [Machine-readable results](results/summary.json): precision/recall/F1, confusion matrices, checkpoint hashes and paired manifests.
- [Analysis script](analyze.py): checks full 60-step completion, selected metrics, checkpoint/scaler/partition hashes, and normalization-only paired configuration changes.
- Raw histories/checkpoints: `outputs/mlp-tuning/normalization/{light,iid,dirichlet}`; controls remain under `outputs/mlp-tuning/convergence` (both Git-ignored).

All artifact checks, `pip check`, analysis-script compilation and `git diff --check` passed. Existing reports and model comments are preserved. Training source was not edited. CPU timings are descriptive and were collected separately under uncontrolled host load; they are neither causal speed comparisons nor edge-hardware measurements.

Use the three normalization commands printed by `python -m src.eval.tuning_plan` to reproduce training in new output directories, or follow the research runner's matching-checkpoint resume instructions. Regenerate tables into a new directory with:

```sh
.venv/bin/python reports/normalization_2026_10_05/analyze.py --output /tmp/tradecraft-normalization-regenerated
```

The analysis reads saved artifacts only, not datasets. This interpretation is separate from the generated tables.
