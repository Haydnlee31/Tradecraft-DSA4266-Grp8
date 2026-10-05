# Loss ablation: retain weighted loss for the next stage

Completed all four planned unweighted cross-entropy (CE) screens locally on 2026-10-05. No test evaluation or cloud resources were used. Training code, comments, the frozen plan and previous outputs are unchanged.

## Results

Each entry is the best validation macro-F1 within 60 epochs/rounds, using model seed 0 and partition seed 0. The paired controls use square-root class-weighted CE.

| Model | Weighted CE | Unweighted CE | Change (percentage points) |
|---|---:|---:|---:|
| Central heavy | 0.6769 | 0.6656 | −1.14 |
| Central light | 0.6576 | 0.6324 | −2.53 |
| Federated IID | 0.6140 | 0.5793 | −3.47 |
| Federated non-IID | 0.4706 | 0.4669 | −0.37 |

None passes either predefined promotion criterion. The criteria are project preferences, not statistical significance tests.

The main insight is that removing weighting does not solve the observed detection failures in this configuration:

- **Heavy:** Web recall falls from 33.44% to 10.83%, Brute Force recall from 25.27% to 16.97%; false alerts rise from 20.35% to 24.45%. Some classes improve, including Recon, so this is a class trade-off rather than uniformly worse predictions.
- **Light:** Web recall falls from 30.41% to 2.71%; false alerts rise from 22.66% to 29.27%.
- **IID:** Web recall falls to zero, DoS recall falls from 83.06% to 54.59%, and false alerts rise from 31.50% to 45.17%. DDoS recall improves, illustrating why aggregate accuracy is insufficient.
- **Non-IID:** False alerts improve from 12.74% to 9.23%, but both Web and Brute Force recall are zero, and DoS recall is only 0.88%. This is not an acceptable detector merely because it raises fewer alerts.

These are single-seed, validation-selected comparisons, not final generalization claims. Only 628 Web and 277 Brute Force validation examples are available. IID CE peaks at round 60; this budget does not establish convergence. CE is not proven universally inferior, but it should not replace the weighted control based on this screen.

## Next planned stage

Keep square-root-weighted CE and test **LayerNorm versus BatchNorm** in centralized-light, federated-IID and federated-non-IID, holding the remaining settings fixed. This tests a normalization hypothesis; these loss results do not prove BatchNorm causes the federated gap. Keep the weighted BatchNorm runs as the controls, rather than mixing in the rejected loss change.

Eight of twelve screening runs are complete. The three normalization runs and one heavy-dropout run remain. No learning-rate substitutions were made, and no normalization or confirmation runs were launched in this stage. Freeze the eventual recipe before multi-seed confirmation and final SHAP/policy conclusions. There is no evidence here to justify increasing heavy-model capacity or renting cloud compute yet.

## Artifacts and checks

- [Detailed tables and curves](results/README.md): every class's recall, false alerts and measured compute.
- [Machine-readable results](results/summary.json): full metrics, confusion matrices, deltas and paired manifests.
- [Analysis script](analyze.py): verifies completion, best-step selection, checkpoint/scaler/partition hashes, and that loss is the only paired configuration difference.
- Raw histories and checkpoints: `outputs/mlp-tuning/loss/{heavy,light,iid,dirichlet}` (Git-ignored); original controls remain under `outputs/mlp-tuning/convergence`.

All verification checks passed, as did `pip check` and Python compilation of the analysis script. No training-source edits required a new training test suite run. CPU timing is descriptive: controls were measured on a different day, and host load is uncontrolled, so differences must not be presented as a causal speed benefit of CE. These are not edge-hardware measurements.

To reproduce, use the four `loss` commands printed by `python -m src.eval.tuning_plan`, substituting the project virtual environment's Python. Output directories must be new, or use the research runner's documented resume workflow for matching existing runs. Regenerate the tables into a new directory with:

```sh
.venv/bin/python reports/loss_2026_10_05/analyze.py --output /tmp/tradecraft-loss-regenerated
```

The analysis reads only saved experiment artifacts, not the test dataset. This interpretation is separate from generated tables and is preserved when regenerating them.
