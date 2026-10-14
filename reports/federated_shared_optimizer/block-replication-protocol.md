# Paired seed replication of blockwise compression

14 October 2026. The block128 candidate nearly matches uncompressed shared-gradient Adam on seed 7 while using 86.4% less payload. Its largest class-recall loss is 4.70 percentage points, only slightly inside the five-point allowance. This experiment asks whether the fixed setting remains acceptable when initialization and minibatch sampling change.

## Two new runs with existing references

Run seeds 17 and 27 with the unchanged four-bit, 128-coordinate-block codec and error feedback in both directions. Reuse the frozen training loop, client populations, fitting rows, loss, learning rate, architecture, calibration/audit groups and 2,160-step budget. Match each result to the already audited centralized-light, uncompressed shared-Adam and FedAvg results for that same seed and runtime. Verify matching initialization and shared-client sample-chain hashes.

Both seeds run to completion even if the first misses a safeguard. Each processes 21.6 million backward examples, for 43.2 million new training examples total. Recovery replays two steps per seed, adding 40,000 backward examples. The seed 7 candidate and all nine reference results are reused, not retrained or counted as new runs. Delivered-model inference uses the existing development panel only.

## Each seed must pass

At the primary 5% calibration false-alert target, each seed must satisfy all five unchanged limits relative to its own same-seed references:

- Macro-F1 loss from uncompressed shared Adam no greater than 0.01.
- Absolute actual false-alert difference from shared Adam no greater than 0.005.
- Recall loss for every class from shared Adam no greater than 0.05.
- Macro-F1 gain over FedAvg at least 0.08.
- Serialized payload at most 15% of historical uncompressed shared Adam.

A passing mean cannot rescue an individual seed that fails. Nor can a better secondary threshold, earlier checkpoint, changed block size or relaxed class allowance. The [machine-readable plan](block-replication-plan.json) is committed with the runner and synthetic tests before new model scores are calculated. The earlier uncalibrated training gate remains unchanged.

Report all three seeds separately, all class recalls, paired differences and all four existing alarm targets. Three-seed arithmetic means, sample standard deviations and ranges are descriptive; seed 7 remains discovery evidence. This tests sensitivity to training randomness on one fixed partition, not performance on new captures or a new partition.

## Independent checks and the next decision

In a separate process, reload each saved recovery state and reproduce the final delivered weights, server target, Adam moments, compression residuals, sample chain and byte counters exactly. Recompute all reported metrics from the delivered checkpoint and independently verify the calibration threshold and decisions. Private rows, logits and checkpoints remain outside Git; publish only aggregate evidence and hashes.

If every seed passes, prepare a locked comparison for reserved confirmation. This stage does not authorize opening that data automatically. If any seed fails, retain the method as a measured quality-versus-payload trade-off rather than a validated replacement, and summarize the central-versus-federated evidence without launching another automatic parameter search. Reserved confirmation and original validation/test arrays remain closed. No cloud compute is needed.
