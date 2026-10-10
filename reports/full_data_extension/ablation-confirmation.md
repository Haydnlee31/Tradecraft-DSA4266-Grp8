# Feature masking results and seed confirmation

The nine seed-7 models passed independent artifact and metric verification.
Masking Number and Tot sum nearly preserved centralized macro-F1, slightly
lowered IID federated macro-F1, and slightly raised controlled non-IID macro-F1.
None of those changes solved the rare-class or false-alert problems. Keep the
full-feature models as references; no masked model is promoted.

The next experiment adds training seeds 17 and 27 without changing the data,
model, loss, client assignments, training budget or validation panel. The
[confirmation plan](ablation-confirmation-plan.json) specifies 18 new runs.
They are prepared, not executed. No cloud machine was accessed during this
local preparation.

## What the first seed showed

All scores below use the final step-20 checkpoint on the same 2,047,805-row
validation panel. The original all-validation view remains separate. False
alerts mean benign flows classified as any attack, not a percentage of all flows.

| Training setting | Mask | Macro-F1 | Benign false alerts | Web recall | Brute Force recall |
|---|---|---:|---:|---:|---:|
| Centralized-light | None | 0.65456 | 18.00% | 6.55% | 29.44% |
| Centralized-light | Number | 0.64548 | 19.62% | 6.38% | 29.21% |
| Centralized-light | Number and Tot sum | 0.65428 | 19.80% | 4.73% | 28.89% |
| Federated IID | None | 0.61557 | 19.17% | 0.00% | 26.64% |
| Federated IID | Number | 0.61261 | 19.91% | 0.00% | 26.40% |
| Federated IID | Number and Tot sum | 0.60919 | 19.52% | 0.00% | 26.56% |
| Federated non-IID | None | 0.40123 | 0.41% | 0.00% | 0.00% |
| Federated non-IID | Number | 0.40230 | 0.15% | 0.00% | 0.00% |
| Federated non-IID | Number and Tot sum | 0.40858 | 0.45% | 0.00% | 0.00% |

The [portable seed-7 evidence](ablation-seed7-results.json) retains every class's
precision, recall, F1 and support, both final confusion matrices, error categories,
trajectories, work counters, timing and artifact hashes. It records independent
verification of nine CUDA recovery pairs, three historical 20-step bridges,
86 training-source files, and 360 full/panel confusion-matrix metric views.
No model was trained or scored during that archive review.

The joint-mask non-IID model classified 97.78% of Web-based attacks and 96.26% of
Brute Force attacks as benign. Low false alerts therefore do not make this model
a reliable detector. The centralized joint-mask model also illustrates why
macro-F1 alone is insufficient: aggregate performance barely changes while false
alerts increase and Web-based recall declines.

SHAP importance and retraining dependence answer different questions. A trained
model can rely strongly on Number while another model learns substitutes after
its removal. The joint mask breaks the demonstrated Tot sum ratio routes, but
other window-size proxies may remain; it also removes legitimate size information.
These runs do not prove either shortcut-free learning or that the original scores
were wholly artificial.

## Fixed confirmation schedule

Reuse the nine seed-7 results and add all nine combinations for each new seed:

| Training seed | Lanes | Feature arms | New runs | Processed training examples |
|---|---|---|---:|---:|
| 17 | Light, IID, controlled non-IID | Full39, Number masked, Number and Tot sum masked | 9 | 360,000,000 |
| 27 | Light, IID, controlled non-IID | Full39, Number masked, Number and Tot sum masked | 9 | 360,000,000 |

Each model receives 20 epochs or rounds and 40M processed examples from the same
2M-row training cohort. That is 720M repeated training exposures, not 720M distinct
records. Optimizer updates per model remain 78,140 for light, 78,400 for IID and
78,340 for controlled non-IID: 1,409,280 updates across the 18 new runs. Exposure
is paired within a lane; the lanes do not have identical optimizer trajectories.

Training seeds change initialization, dropout and training order. Partition seed
stays 7 and both federated assignment hashes remain fixed. Thus this stage measures
training-seed sensitivity, not sensitivity to different client mixtures. The
model keeps 39 input slots and 5,096 parameters, even when columns are masked.

Run seed 17 first, then seed 27. The boundary is an integrity check, not a decision
based on scores. Complete every planned arm unless a runtime or integrity problem
requires review. Do not extend a promising run, drop an unfavorable seed, use the
best checkpoint as the primary endpoint, or add hyperparameters.

## What will count as evidence

For each lane and seed, subtract the full-feature result from each masked result.
Report all paired values, their mean, sample standard deviation and range, including
macro-F1, false alerts, all eight class precision/recall/F1 values, and attacks
missed as benign versus assigned the wrong attack category. Keep work and timing
alongside the metrics. Seed 7 is exploratory; show both prospective seeds separately
as well as in the three-seed summary.

Consistent effects across the new seeds support repeatability under this protocol.
Mixed signs or sizeable variation mean the effect is uncertain. Three seeds do
not establish statistical significance or equivalence. There is no automatic
promotion threshold: a small aggregate gain cannot hide persistent rare-class
failure or increased false alerts.

The validation panel has been reused extensively and remains within-collection
evidence. Final-test evaluation, unseen-session claims, additional data, bigger
models and new algorithms are separate decisions after this review.

## Execution and recovery safeguards

The orchestration entry point is
`scripts/official39_ablation_confirmation.py`. It lives outside `src/`, so the
training-source inventory from commit `5cc8206` remains byte-for-byte unchanged.
No dependencies or model code are updated for confirmation.

`check` reads the hash-bound seed-7 archive and verifies every training-source
file, runtime, packed array, scaler, validation panel and federated assignment.
It never trains a model. `check --metadata-only` is a local data/provenance check;
it explicitly does not establish CUDA or runtime readiness and cannot be used
as a training mode.

`run --seed 17` schedules exactly nine models. `run --seed 27` first verifies
all nine completed seed-17 cases, then schedules the remaining nine. The wrapper
checks initialization, masks, full participation, work, checkpoint identity,
both metric views and final-panel selection. Results are saved after each case.

Existing output requires explicit `--resume`. Completed runs are reverified
before being skipped; partial runs resume only from a completed-step checkpoint.
A changed plan, wrapper, source, runtime or saved artifact stops recovery rather
than silently overwriting evidence. Do not change packages or code during a batch.

An exclusive per-seed lock prevents duplicate launches. SSH disconnection does
not kill a job inside tmux: reconnect and attach, rather than launching again.
A hard process kill can leave `seed17.lock` or `seed27.lock`. Inspect its recorded
host/PID and confirm no job is active before requesting recovery; do not delete
checkpoints or clear locks blindly.

`summarize` requires all 27 final endpoints, including the retained seed-7 evidence.
It rechecks confirmation artifacts, produces the paired statistics and saves all
final class metrics and work records. It does not open a dataset or evaluate a
model. Existing summaries are not overwritten.

## Cloud commands after the code handoff

Use the existing RONIN T4 and environment. From the Mac, reconnect with:

```bash
ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o IdentitiesOnly=yes -i "/Users/haydn/Downloads/trade_ssh_01.pem" ubuntu@haydn-tradecraft-02.nus.cloud
```

After transferring and verifying the new code bundle using the chat handoff,
run the following on RONIN. If a tmux session is already active, keep using it.

```bash
if [ -z "$TMUX" ]; then
  tmux new-session -A -s tradecraft-confirmation-masks
fi
cd /home/ubuntu/Tradecraft-DSA4266-Grp8
source /home/ubuntu/.venvs/tradecraft-official39/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

First, check readiness. Stop if it reports any mismatch; do not reinstall packages
to suppress the mismatch or rewrite a hash-bound plan.

```bash
python scripts/official39_ablation_confirmation.py check \
  --seed7-archive /home/ubuntu/tradecraft-official39-ablation-backup.tgz \
  --data outputs/official39-packed-2m-v1 \
  --panel outputs/official39-ablation-inputs-v1/official39-ablation-panel-v1 \
  --partition-root outputs/official39-ablation-inputs-v1/official39-nested-partition-v1
```

After readiness passes, this block runs both explicitly scheduled batches in
order, stopping on an error. It does not stop or continue based on model scores.
All 18 cases use fresh output directories. Do not run another GPU training job
concurrently.

```bash
(
set -euo pipefail
confirmation_args=(
  --seed7-archive /home/ubuntu/tradecraft-official39-ablation-backup.tgz
  --data outputs/official39-packed-2m-v1
  --panel outputs/official39-ablation-inputs-v1/official39-ablation-panel-v1
  --partition-root outputs/official39-ablation-inputs-v1/official39-nested-partition-v1
  --output outputs/official39-ablation-confirmation-v1
)
python -u scripts/official39_ablation_confirmation.py run --seed 17 "${confirmation_args[@]}" \
  2>&1 | tee -a outputs/official39-ablation-confirmation-seed17.log
python -u scripts/official39_ablation_confirmation.py run --seed 27 "${confirmation_args[@]}" \
  2>&1 | tee -a outputs/official39-ablation-confirmation-seed27.log
python scripts/official39_ablation_confirmation.py summarize \
  --output outputs/official39-ablation-confirmation-v1
)
```

On an actual process interruption, preserve all outputs and inspect the error
before rerunning the affected seed with `--resume`. Do not add `--resume` merely
because an SSH window disconnected. The normal tmux job may still be running.

Return the new confirmation directory and logs for review; do not include raw
training arrays in the backup. Do not select a model or evaluate the final test
from the console's individual run scores.
