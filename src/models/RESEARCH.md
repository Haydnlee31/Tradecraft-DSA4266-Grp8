# Local readiness and controlled MLP experiments

Use `python -m src.models.research` for new tuning runs. The old
`src.experiment`, centralized CLI and Flower app remain available for historical
reproduction, but are not interchangeable research entry points: the worker app
can select round 0, and legacy validation losses differ. The new runner applies
one validation/checkpoint protocol to heavy, light, IID and non-IID lanes.

## Guarantees and limits

- Train and validation only. There is deliberately no test flag in this runner.
- Same canonical MLP initialization, labels, train-only scaler and global loss
  weights. No raw-data resampling. Scaler and source/data hashes are recorded.
- Validation loss is unweighted sample-mean CE across all four lanes; weighted
  training batch-mean loss is separately named. Selection strictly maximizes
  validation macro-F1 from epoch/round 1, with first-best tie handling.
- Model seed and partition seed are separate. Non-IID assignments use the existing
  class-ordered, self-balancing Dirichlet builder; exact assignments are saved.
- Central Adam persists across epochs; client Adam resets each round. One local
  epoch and full client participation are fixed. All state, including BN buffers,
  is aggregated with Flower using dataset counts, preserving historical policy.
- Actual processed examples and optimizer updates are logged (BN can drop a
  singleton tail). These counts do not make FL rounds equivalent to central epochs.
- Default CPU execution is portable; `--device cuda` fails if CUDA is unavailable.
  `--device auto` chooses CUDA if available, otherwise CPU. MPS is not claimed
  supported/tested. An actual CUDA pilot is still needed on future GPU hardware.
- Host step time includes training and full validation, excludes setup/checkpoint
  IO. Unix peak RSS is process-lifetime high-water memory, not model inference RAM;
  Windows reports this field as unknown. GPU allocation is not total GPU memory.
- Pooled scaler/class counts and full server validation are research simulation
  assumptions. No device topology, physical edge, privacy or network claim follows.

## Run and recover

```sh
python -m src.models.research --lane light --output outputs/mlp/light-s0 --seed 0 --epochs 60 --patience 0 --device cpu
```

Every new output directory must be absent. To resume a stopped/interrupted run,
repeat **the same command** with `--resume`. Keep the original total epoch budget;
do not change it to the remaining epochs. Source/data/settings/package changes
are rejected instead of silently mixing experiments. Use a new directory for a
new experiment. Completed runs can be resumed safely to regenerate final outputs.

`last.pt` is an atomic epoch/round-boundary recovery checkpoint containing current
weights, central Adam state, RNG states, shuffle generator, history, early-stopping
counter and best weights. Client Adam state is absent by design: clients reset it.
A mid-step interruption repeats that entire step. `--stop-after 1` simulates a
clean pause after step 1; omit it on resume. This also tests recovery in CI.
`best.pt` is the final validation-selected inference artifact, not the resume file.
Only load trusted project checkpoints/scalers; keep the complete output directory
and the original sampled data when transferring environments. Exact bitwise
reproducibility is tested on CPU, not guaranteed across hardware/PyTorch versions.

```sh
python -m src.eval.training_parity --output outputs/parity-check.json
python -m unittest discover -v
python -m src.eval.tuning_plan
```

The plan printer emits, but never executes, the bounded commands in
`configs/local_tuning_plan.json`: four convergence controls, four loss comparisons,
three light-model normalization comparisons and one heavy dropout comparison.
Run stages in that order and stop if the next stage no longer answers a useful
question. Maximum 12 screens plus 12 frozen-recipe confirmation runs. LR changes
may replace later screens, never silently increase the budget. Individual screen
effects are exploratory; combined improvements are not assumed additive.

Confirm shortlisted recipes using model seeds 0/1/2 with fixed partition seed 0.
For a pure capacity claim, match heavy/light dropout and normalization; otherwise
describe a configured-model comparison. Report every class and false alerts.
Seed SD is not a confidence interval. Test scores already inspected historically
cannot become fresh confirmation by reshuffling rows. Final SHAP should follow
model freezing; existing validation explanations are sufficient for initial checks.

## Before renting a machine

1. Run `pip check`, the tests, and a two-step job after transferring this exact
   committed code and hashed sampled data.
2. Confirm the recorded device is the one requested. Benchmark the same settings
   on the candidate machine; local CPU timings do not predict GPU speedups.
3. Exercise pause/resume, verify output storage persists, and copy checkpoints/logs
   somewhere recoverable before stopping or deleting a machine.
4. Calculate actual hourly price times measured runtime, allowing setup, storage
   and retries. Configure budget alerts/auto-shutdown separately in RONIN.

This repository does not launch cloud resources or configure RONIN billing.
