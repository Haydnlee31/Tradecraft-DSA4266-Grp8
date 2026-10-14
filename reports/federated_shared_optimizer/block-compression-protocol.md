# Blockwise compression with unchanged quality safeguards

14 October 2026. Test whether smaller compression blocks preserve the previous shared-gradient model's class recall while retaining most of the communication savings. The earlier four-bit method lost only 0.69 macro-F1 points but lowered Spoofing recall by 7.82 points, failing the five-point class safeguard. This follow-up changes only the scale granularity; it does not lower that safeguard.

## Two paired runs

The tensor-scale control must reproduce the previous four-bit error-feedback model exactly, including optimizer moments, residuals, delivered weights, sample chain, byte counters and all four alarm operating points. The sole candidate uses a separate scale for each consecutive block of at most 128 coordinates within each tensor. Both runs start from the same original seed 7 initialization, not the previous trained checkpoint.

The hypothesis is that smaller blocks can protect small updates from being rounded too coarsely when another coordinate is large. This is not an established explanation for the previous class regression. Block quantization is established prior art, and this is a bounded engineering follow-up rather than a new-algorithm claim.

Both runs keep the same 500,000 fitting rows, twenty non-IID client populations, sample sequence, 39-feature LayerNorm network, square-root weighted loss with fixed fitting denominator, learning rate 0.003 and persistent server Adam. Clients compute gradients at the delivered model. Uplink residuals preserve omitted gradients; downstream messages encode the difference between the server target and the last delivered model. Adam moments stay FP32. No client-local optimizer, classifier extension, new learning-rate search or cloud computation is introduced.

## Work and message accounting

Each run processes 21.6 million backward examples in 2,160 synchronization steps, using twenty client batches of 500 per step. Together they process 43.2 million training examples. A separate recovery audit replays the final two steps of each run, adding 40,000 backward examples. Inference and recovery work are reported separately.

Block messages use a versioned 8-byte header and, for each tensor, a 4-byte element count, an FP32 scale per block and packed four-bit symbols. The 5,096-parameter model needs 45 blocks across ten tensors. This gives 2,776 bytes per message and 240,255,840 bytes for a run, including the initial FP32 broadcast and twenty recipients in each direction. These are design counts to verify against actual serialization, not measured network traffic. No round-trip reduction is claimed.

## Frozen acceptance rules

The primary endpoint remains the 5% calibration false-alert target on the existing, duplicate-group-separated development audit panel. Relative to the original uncompressed seed 7 shared-gradient reference, the candidate must satisfy all of these:

- Macro-F1 loss no greater than 0.01.
- Absolute actual false-alert-rate difference no greater than 0.005.
- Recall loss for every class no greater than 0.05.
- Macro-F1 advantage over the seed 7 FedAvg reference at least 0.08.
- Serialized payload no greater than 15% of the historical shared-gradient raw payload.

All class recalls and all four existing alarm targets are reported. The tensor control is not eligible for selection. Passing only warrants paired-seed replication; failure stops this screen without changing block size or relaxing the rules. The earlier failed uncalibrated training gate remains unchanged.

## Data and verification boundaries

The source, [plan](block-compression-plan.json) and synthetic tests are committed before real-data scores are computed. Private outputs retain resumable checkpoints, logits and immutable run identities. A separate process must reproduce the saved endpoint from two steps earlier and recalculate scores from the delivered checkpoint. Malformed packets, sequence mismatches, reserved symbols, padding and invalid scales are rejected.

Use only the existing fitting and development arrays. Reserved confirmation and original validation/test arrays remain closed. This is exploratory evidence on previously inspected development data and a fixed client partition, not a fresh generalization test. Compression does not provide a privacy guarantee or establish real-network speed.
