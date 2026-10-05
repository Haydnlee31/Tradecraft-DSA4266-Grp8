# Option 1 completed: configured-model training-seed stability

On 2026-10-05 the user approved option 1: confirm the shortlisted configurations, not a matched architecture/federation experiment. The [confirmation plan](../../configs/local_confirmation_plan.json) was frozen before launching eight new seed-1/2 runs. Four verified seed-0 screening runs were reused in place. All twelve configurations/seeds have complete 60-step histories; no test evaluation or cloud resources were used.

## Frozen configurations

All use square-root-weighted cross-entropy, Adam learning rate 0.001, weight decay 0.00001, batch size 512 and no early stopping. The federated simulations use 20 clients, partition seed 0, one local epoch per round and full participation; non-IID alpha is 0.5.

| Lane | Normalization | Dropout |
|---|---|---:|
| Central heavy | BatchNorm | 0.3 |
| Central light | BatchNorm | 0.2 |
| Federated IID | LayerNorm | 0.2 |
| Federated non-IID | LayerNorm | 0.2 |

## Results

Mean ± sample standard deviation of validation macro-F1. Each run selects its first-best checkpoint within 60 epochs/rounds. No best seed is selected.

| Model | Seeds 0/1/2 | New seeds 1/2 only |
|---|---:|---:|
| Heavy | 0.6747 ± 0.0031 | 0.6735 ± 0.0034 |
| Light | 0.6583 ± 0.0029 | 0.6587 ± 0.0040 |
| Federated IID | 0.6328 ± 0.0024 | 0.6322 ± 0.0031 |
| Federated non-IID | 0.5557 ± 0.0128 | 0.5483 ± 0.0016 |

Seed 0 influenced screening and is not independent confirmation. Even seeds 1/2 reuse the same data and validation-based checkpoint selection: this is a training-seed stability study, not an independent-data generalization test. Sample SD is not a confidence interval, and two new seeds provide limited evidence.

## What the results support

- **The aggregate ranking is consistent across all three seeds:** heavy > light > IID > non-IID. The three-seed heavy/light gap is only 1.63 percentage points of macro-F1, despite about 36 times as many trainable parameters (198,984 versus 5,544). This is a useful configured-model trade-off, not proof of a pure capacity effect or measured edge feasibility.
- **Central light remains a useful compact reference.** New-seed mean macro-F1 is 0.6587 versus IID's 0.6322. Normalization differs, so the gap cannot be attributed solely to federation.
- **Low macro-F1 variation hides class-level instability.** Heavy Web recall is 39.97% and 19.75% on the two new seeds. IID Web recall is 18.31% and 7.80%. Report these rather than declaring the models broadly robust from aggregate SD.
- **Non-IID rare-attack detection remains a major failure.** Both new seeds have zero Web recall. Their mean DoS recall is only 14.75%, Brute Force 14.80%, and Spoofing 48.19%. Mean false alerts of 4.06% do not make this a good detector; many attacks are missed. The seed-0 Web recall improvement to 4.30% did not persist in the new seeds.
- **False alerts remain substantial centrally and under IID.** New-seed means are 22.78% (heavy), 24.02% (light), and 24.40% (IID), all measured on this fixed validation set. Do not project these to a live network without evidence.
- **Several runs still peak at the budget boundary.** Light seed 2, IID seed 1 and non-IID seed 1 select step 60; the other federated checkpoints are at 58–59. This does not establish convergence. The budget was not extended after seeing results.

LayerNorm's earlier seed-0 advantage remains screening evidence; this stage did not rerun new-seed BatchNorm controls and therefore cannot independently confirm a causal normalization benefit. Fixed partition seed also leaves partition-to-partition robustness untested.

## Next stage

Keep these recipes frozen as the completed option-1 study. Proceed to **failure-focused explanation and policy analysis**, not an automatic larger-model/cloud sweep:

1. Verify the explanation pipeline can load the research checkpoints, especially LayerNorm, with the exact saved feature order and scaler. Do not assume legacy checkpoint loading is interchangeable.
2. Use the same seeded train-only SHAP background and validation explanation sample across models. Include correct detections, false alerts and missed Web/Brute Force/DoS cases. Report how any class-stratified explanation sample differs from the natural validation distribution.
3. Check whether feature rankings and failure explanations persist across seeds; do not select whichever seed gives the most attractive story. Keep explanations at protocol/flow-feature level, never device vulnerability claims or causal proof.
4. Treat resulting policy recommendations as hypotheses requiring validation, not deployable blocking rules. Explicitly state when the detector cannot reliably support a recommendation.

The results are sufficient to motivate and analyze model trade-offs, but not to claim that simulated non-IID learning is a satisfactory replacement for centralized detection. If rare-attack detection is a required success criterion, a separate, predeclared revision stage is warranted after examining failures—for example, investigate where missed classes are mapped and whether client class scarcity is associated with those errors. Do not change the completed study's recipe or retroactively soften its criteria.

No SHAP, further training, test scoring, cloud launch, commit or push was performed in this stage.

## Artifacts and verification

- [Detailed tables and seed trajectories](results/README.md): every seed, every class recall, all-seed and new-seed summaries.
- [Full machine-readable results](results/summary.json): per-class precision/recall/F1, confusion matrices, mean/sample SD and manifests.
- [Frozen plan](../../configs/local_confirmation_plan.json).
- [Launcher](run.py): previews commands by default; `--execute` runs only eight new jobs, at most two concurrently, and requires a new output root.
- [Analysis](analyze.py): checks the plan against the pre-run receipt, selected checkpoints, complete histories, source/data/package consistency, effective model settings, scaler hashes and exact fixed client partitions.

New checkpoints remain at `outputs/mlp-confirmation/{lane}/seed-{1,2}`; reused seed-0 paths are listed in the plan. `outputs/mlp-confirmation/launch.json` records the plan and its SHA-256 before new training. The generated summary also records the plan hash. None of the original screening artifacts was overwritten.

All artifact checks and `pip check` passed. Both scripts compiled, and sample-SD calculations passed known-value checks. Training source was unchanged. New runs shared the CPU in pairs; their timing is not a controlled comparison with earlier runs or a physical edge measurement.

Regenerate the report into a new directory from the repository root:

```sh
.venv/bin/python reports/confirmation_2026_10_05/analyze.py --output /tmp/tradecraft-confirmation-regenerated
```

The analysis reads saved artifacts only, not datasets or model objects. For interruption recovery, resume individual research runs with their exact original command plus `--resume`; the launcher intentionally refuses to reuse an existing output root.
