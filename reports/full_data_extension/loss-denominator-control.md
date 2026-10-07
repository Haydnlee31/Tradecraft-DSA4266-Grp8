# Rare class exposure and loss denominator control

The bounded cloud comparison does not support adopting the five-batch cap as a
general improvement. At matched aggregate updates, IID macro-F1 fell from 0.58912
to 0.55444, while non-IID rose from 0.36683 to 0.38229. Web-based and Brute Force
recall remained zero under the cap in both settings; non-IID Spoofing recall also
remained zero. The next controlled question is whether batch-dependent loss
normalization contributes to this behavior, not whether to buy a stronger GPU.

## Exposure audit

The audit replays the exact recorded client assignments, round seeds, NumPy
shuffles, batch caps and singleton-tail rule against the verified training labels.
Recorded processed-example counts match the reconstruction. Batching/seeding
source hashes and NumPy versions are checked. No feature matrix, validation split
or test split is read by the exposure audit.

Every one of the 500,000 unique training rows was processed at least once in
both completed capped runs. In particular, all 598 Web-based, 317 Brute Force and
11,024 Spoofing examples were seen. Their total processed exposures were:

| Class | Full epoch reference in either partition | IID capped | Non IID capped |
|---|---:|---:|---:|
| Benign | 530,740 | 533,056 | 955,813 |
| DDoS | 5,892,800 | 5,913,677 | 5,136,221 |
| DoS | 1,812,120 | 1,820,609 | 1,920,093 |
| Recon | 331,380 | 331,123 | 482,241 |
| Web-based | 11,960 | 11,952 | 13,260 |
| Brute Force | 6,340 | 6,420 | 10,784 |
| Spoofing | 220,480 | 221,881 | 301,546 |
| Mirai | 1,194,180 | 1,196,482 | 1,256,202 |

This rules out complete omission of rare training rows in these runs. It does not
establish sufficient diversity or effective learning. The cap redistributes
exposure: individual non-IID Brute Force rows appear 12–88 times, versus exactly
20 each in the reference. Example exposure is not an effective gradient weight;
loss reduction, model state, optimizer state and aggregation weights also matter.

The local receipt is `outputs/official39-exposure-audit-v2.json`. It contains all
eight class totals, per-client class exposure, unique coverage, repeat ranges and
provenance hashes. Its input archive SHA256 is
`295220ebc795aa2012e5ed21a5eb76840327dba0962767811532ab3b654c0b60`.

## Loss control

The historical weighted loss divides each batch's weighted loss sum by that
batch's sum of target-class weights. For a single-class batch, the class weight
cancels. This is standard weighted cross-entropy behavior, not a library bug.

The new opt-in `--loss-reduction fixed_train_mean` instead uses:

```text
training_mean_weight = sum(training_class_count[c] * weight[c]) / training_rows
loss = mean(weight[target] * unweighted_cross_entropy) / training_mean_weight
```

All clients use the same training-derived constant. No validation or test counts
enter the calculation. Square-root class weights themselves remain unchanged;
this is a denominator control, not a stronger-weighting experiment.

At identical parameters, weighting client losses/gradients by client sample count
now reproduces the pooled fixed-denominator loss/gradient. On the full training
population, the loss also matches historical weighted cross-entropy. These
identities are covered by tests, along with invariance to uniformly rescaling
all class weights and retention of class-weight effects in single-class batches.

The identities do not make multi-step local Adam plus FedAvg equivalent to
centralized training. Adam can dampen the effect of scalar gradient rescaling;
changing loss scale can also interact with its epsilon, moments and weight decay.
This control is a hypothesis test, not a promised fix.

## Implementation and recovery

The new factory lives in `src/models/official_losses.py`; historical loss code is
unchanged. The official runner defaults to `batch_weight_sum`, which delegates to
the historical factory. The reduction setting is part of resume provenance;
switching it when resuming is rejected. Models must be trained afresh for the
paired comparison, not continued from the selected cloud checkpoints.

The local two-step recovery pilot on the real 500k cohort passed exactly for
centralized-light, IID and non-IID with LayerNorm and the fixed denominator.
Its receipt is `outputs/official39-fixed-loss-cpu-recovery-v1/checks.json`.
This establishes CPU recovery, not CUDA compatibility or model improvement.

## Frozen next comparison

The [plan](loss-denominator-plan.json) compares both reductions across centralized
light, IID and non-IID. Keep 500k training rows, LayerNorm, square-root weighting,
learning rate 0.001, batch size 512, seed 7 and partition seed 7 unchanged.
Use 20 full epochs/rounds, no cap, no early stopping and no automatic extensions.
Compare final-step full-validation macro-F1, all eight recalls, precision and
benign false alerts. Best-checkpoint metrics are secondary. No test evaluation.

Before the cloud comparison: commit a reviewed snapshot, transfer it, pass its
CUDA recovery pilot and verify same-code baseline bridges against prior results.
The existing recovery entry point now accepts the explicit control:

```bash
python -m src.eval.official_streaming_recovery \
  --data outputs/official39-packed-500k-v1 \
  --output outputs/official39-fixed-loss-gpu-recovery-v1 \
  --device cuda --normalization layer \
  --loss-reduction fixed_train_mean --lanes light iid dirichlet
```

This is not an instruction to restart the machine before the transfer is ready.
Do not relax the baseline checks or sweep additional loss powers or learning
rates. If the planned comparison gives no useful recovery of missing classes,
retain that negative finding and proceed to a separately budgeted data-size
comparison rather than extending the tuning indefinitely. Any promising change
still requires additional predefined seeds before promotion.
