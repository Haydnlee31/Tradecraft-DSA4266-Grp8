# Local readiness results — 3 October 2026

All experiments used the existing sampled train and validation splits. No test
scores were computed, no cloud resources were launched, and previous reports
were not overwritten. Machine-readable metrics and provenance are in
[summary.json](summary.json). Full checkpoints, logs and manifests remain under
`outputs/readiness-20261003/` (ignored by Git; back these up separately).

## Completed local work

- All 31 local tests pass, dependency checks pass, and Git whitespace checks pass.
  Cross-platform CI now includes optional XGBoost; those remote CI runs are not
  represented as completed local checks.
- Baseline implementation committed at `20d17cf`; research runner at `5316499`.
- Baseline screen: four configurations, seed 0, 30 logistic epochs/rounds or
  200 XGBoost rounds maximum, patience 5, validation macro-F1 selection.
- Train-only MLP parity diagnostic: 4096 fixed, class-stratified rows; shared
  global class weights, batch 512; one pass, two passes with matched resets,
  and two passes with matched persistent Adam all passed. Maximum state
  difference was exactly 0.0, with gradients also within 1e-6/1e-5 tolerances.
- Synthetic resume tests pass for all four MLP lanes. A real-data centralized-light
  pause/resume in fresh Python processes exactly matched uninterrupted weights.
- New MLP pilots reproduced the first two historical seed-0 validation macro-F1
  values exactly in every lane, confirming unchanged default training behavior.
- New runner standardizes validation CE, step-1 checkpoint eligibility, data/code
  provenance, device selection and atomic step-boundary recovery. Legacy worker
  entry points remain explicitly labeled, not silently treated as equivalent.
- Optional-Ray manifest defect fixed; no Ray is required by the new CPU workflow.
- LayerNorm/dropout/loss/LR controls and a 12-screen/12-confirmation maximum plan
  are available. The full tuning sweep was NOT launched in this readiness pass.

## Classical baseline screen (validation, one seed only)

| Model | Macro-F1 | Benign false alerts | Web recall | Brute Force recall | Best step | Host elapsed |
|---|---:|---:|---:|---:|---:|---:|
| Central logistic | 0.5763 | 28.72% | 5.89% | 16.25% | 17 | 39.9 s |
| IID logistic | 0.5519 | 43.56% | 3.18% | 14.80% | 30 | 54.6 s |
| Non-IID logistic | 0.4919 | 29.73% | 0.16% | 14.80% | 14 | 33.9 s |
| Central XGBoost | 0.8528 | 12.32% | 43.15% | 55.23% | 29 | 7.2 s |

These are measured host times including fit, checkpoint and final evaluation,
excluding common data setup. XGBoost includes subprocess/array-transfer overhead.
Different stopping points and model families mean these are not equal-budget
speed benchmarks. IID logistic selected its cap, so convergence is unresolved.

XGBoost is a promising reference, substantially stronger than the historical MLP
validation scores (~0.65), but this is not a tuned multi-seed/test comparison.
Its Web recall is still only 43%, and 12% benign false alerts is not evidence of
deployment readiness. Do not infer that deep learning is necessary, that more
MLP parameters would close the gap, or that federated XGBoost must be added.

The logistic IID gap (~0.0244 macro-F1) is smaller than the historical MLP gap,
but this is only exploratory evidence: budget, convergence and seed counts are
not matched. No hidden layers or BatchNorm does not eliminate federation effects.

## CPU pilots on the actual sampled dataset

Two epochs/rounds per lane, seed 0, partition seed 0, batch 512, two Torch CPU
threads. Each step processed 440,594 training examples. Timings include training
and full validation; exclude setup/checkpoint IO. Normal host background activity
can affect them. They are not measurements of Jetson, network or GPU performance.

| Lane | Step 1 / 2 seconds | Updates per step | Process peak RSS |
|---|---:|---:|---:|
| Heavy | 5.387 / 5.377 | 861 | 1139 MiB |
| Light | 2.330 / 2.310 | 861 | 1195 MiB |
| IID | 2.775 / 2.401 | 880 across clients | 1078 MiB |
| Non-IID | 2.604 / 2.388 | 870 across clients | 988 MiB |

RSS is process-lifetime high-water memory including data and libraries, NOT
inference model RAM. Do not interpret the small RSS differences as architecture
memory rankings. CUDA was unavailable; its explicit failure path is unit-tested,
but CUDA training/recovery/performance remain unverified on real hardware.

These times suggest the small-data tuning plan is feasible locally; first complete
controlled CPU comparisons rather than renting a large GPU solely for capacity.

## Next steps and safeguards

1. Use [the research runner](../../src/models/RESEARCH.md), not legacy entry points,
   for new tuning. Print the frozen plan with `python -m src.eval.tuning_plan`.
2. Begin with four 60-step convergence controls. Keep test data untouched. Later
   ablations change one factor; preserve original baselines and use equal search
   budgets for best-tuned comparisons.
3. Confirm selected recipes with three training seeds on fixed partitions; assess
   partition variability separately. Seed SD is not a confidence interval.
4. Perform final SHAP stability/comparison after freezing models. Existing SHAP
   output is exploratory and should not dictate unvalidated blocking policies.
5. Before cloud use, transfer exact committed code, hashed sampled data and whole
   run directories; install dependencies, run tests, verify the actual device,
   exercise recovery, then time a short job on the candidate machine.
6. Set RONIN budget/shutdown controls and durable output storage before launch.
   None was configured here, and no RONIN credits were consumed.

Checkpoint recovery is implemented for the new MLP runner. Classical screen runs
still restart from initialization after interruption; keep those short local
jobs or add separate recovery before turning them into long cloud workloads.
