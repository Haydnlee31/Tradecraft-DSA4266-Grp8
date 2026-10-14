# Blockwise compression preserves aggregate quality with smaller messages

14 October 2026. **Four-bit compression with a scale per 128 coordinates passes all five predeclared seed 7 safeguards.** It scores 0.6241 macro-F1 against 0.6246 for uncompressed shared-gradient Adam, while using 86.4% less modeled application payload. The earlier tensor-scale method scored 0.6177 and failed the class-recall safeguard. This is the first passing candidate in these two compression screens, and warrants replication—not a claim that the research problem is solved.

The remaining weakness matters: Spoofing recall is still 4.70 percentage points below uncompressed shared Adam, just inside the five-point allowance. Rare-class recognition also remains weak. We should test the unchanged candidate on the two existing additional seeds before opening reserved confirmation data or promoting it as the final federated method.

## What changed

The previous method used one rounding scale for an entire weight tensor. The new method gives each consecutive block of at most 128 coordinates its own scale. Think of a ruler: a very large value can force a coarse ruler for all other values in a tensor; smaller blocks let different groups use finer rulers. The experiment supports this change as a useful engineering choice in this setting, but does not prove that an outlier caused the earlier Spoofing regression.

Both versions retain information lost through rounding and add it to later messages. Clients compute gradients using the model they actually received. The server combines decoded gradients using client population weights and keeps one persistent, full-precision Adam optimizer. The final scores use the delivered model, not the server's hidden full-precision target. No raw rows are passed through the simulated aggregation interface, and no local client Adam steps are introduced.

The [protocol](block-compression-protocol.md), [plan](block-compression-plan.json), implementation and thirteen preliminary tests were frozen in commit `82c7ec3` before real-data scores were calculated. Both fresh runs use the same initialization, fitting rows, non-IID ownership, client sample sequence, learning rate, architecture, loss and 21.6-million-example backward budget. The tensor-scale control exactly reproduces its historical endpoint, including its entire saved training state and all four alarm operating points.

Block quantization, error feedback and shared server optimization are established ideas. The contribution here is their controlled application and measured trade-off in this IoT classification study, not a claim to have invented those mechanisms. The frozen historical sources were not modified.

## Primary results

These are paired seed 7 results on the existing development audit panel, using the same 5% calibration false-alert target. The centralized reference is **centralized-light with the same architecture**, not the separate heavy model. MB means decimal megabytes.

| Method | Macro-F1 | Actual false alerts | Payload MB | Status |
| --- | ---: | ---: | ---: | --- |
| Centralized-light Adam | 0.6456 | 5.33% | Not modeled | Reused quality reference |
| FedAvg with client Adam | 0.5137 | 5.70% | 88.06 | Reused raw-payload reference |
| Uncompressed shared Adam | 0.6246 | 5.43% | 1,761.18 | Reused raw-payload reference |
| Four-bit tensor scales | 0.6177 | 5.34% | 228.16 | Exact replay; fails class safeguard |
| Four-bit block scales | 0.6241 | 5.39% | 240.26 | Passes this seed's five checks |

The blockwise model is **11.04 macro-F1 points above FedAvg and 2.15 points below centralized-light** at this operating point. It closes about 83.7% of that seed's FedAvg-to-centralized gap. Its difference from uncompressed shared Adam is only 0.054 macro-F1 points. These are single-seed differences, not statistical-significance claims or the earlier three-seed averages.

Compared with tensor scales, blocks improve macro-F1 by 0.64 points and Spoofing recall by 3.12 points for 5.30% more payload. All five rules remain exactly as declared before this follow-up:

| Check against the frozen references | Observed | Limit | Result |
| --- | ---: | ---: | --- |
| Macro-F1 loss from shared Adam | 0.054 points | At most 1.00 point | Pass |
| Absolute actual false-alert difference | 0.038 points | At most 0.50 points | Pass |
| Largest class recall loss | 4.697 points, Spoofing | At most 5.00 points | Pass, narrowly |
| Macro-F1 gain over FedAvg | 11.041 points | At least 8.00 points | Pass |
| Payload relative to historical shared Adam | 13.642% | At most 15.00% | Pass |

The 5% target is applied to calibration rows. The observed audit false-alert rate is 5.39%, so this is not a guaranteed deployment limit. The earlier failed uncalibrated training gate remains failed; this separate communication screen does not override it.

## Every class still matters

Correct eight-class recalls at the primary operating point are:

| Class | Uncompressed shared Adam | Tensor scales | Block scales |
| --- | ---: | ---: | ---: |
| Benign | 94.57% | 94.66% | 94.61% |
| DDoS | 83.80% | 83.21% | 84.41% |
| DoS | 72.45% | 72.68% | 71.07% |
| Recon | 43.84% | 45.59% | 46.15% |
| Web-based | 9.63% | 8.89% | 8.52% |
| Brute Force | 18.46% | 20.00% | 19.23% |
| Spoofing | 63.54% | 55.71% | 58.84% |
| Mirai | 99.47% | 99.46% | 99.46% |

The blockwise model correctly identifies 23 of 270 Web-based examples and 25 of 130 Brute Force examples. It does not solve rare-class detection. It also loses some DoS recall while improving Recon and DDoS recall. Aggregate quality can be nearly unchanged even while individual decision boundaries move; that is why the class safeguard must remain in the replication stage.

Across the four predeclared alarm targets, aggregate performance stays close to the uncompressed reference:

| Calibration target | Block macro-F1 | Block actual false alerts | Macro-F1 difference from shared Adam |
| --- | ---: | ---: | ---: |
| 1% | 0.5831 | 1.18% | −0.073 points |
| 5% | 0.6241 | 5.39% | −0.054 points |
| 10% | 0.6362 | 10.36% | −0.154 points |
| 20% | 0.6428 | 20.79% | +0.211 points |

These secondary operating points are descriptive; none replaces the primary five-percent rule or cancels a class-specific weakness. The [aggregate evidence](block-compression-results.json) includes every confusion matrix, precision, recall, calibration threshold and support count.

## Smaller payload does not mean fewer synchronization steps

The model has 5,096 parameters in ten tensors. Blockwise encoding uses 45 scales, an 8-byte message header, tensor counts and packed four-bit values. The actual serialized message length is 2,776 bytes, versus 2,636 bytes for tensor scales. Initialization remains an exact FP32 broadcast.

The recorded 240,255,840-byte run total charges twenty uploads and twenty delivered updates per synchronization step, plus startup. It is 13.64% of the historical shared-gradient raw payload and **2.73 times the historical FedAvg raw payload**. The compressed counts include metadata and startup, while the older raw references omit those extras; the savings comparison is conservative in this respect.

However, the candidate still uses **2,160 synchronization steps versus FedAvg's 108 rounds**. It reduces bytes, not synchronization frequency. Real high-latency networks could therefore remain unfavorable even with much smaller messages. Transport, secure aggregation, retransmissions and real distributed latency are not measured, and compression supplies no privacy guarantee.

The tensor replay took 56.54 seconds and the block candidate 57.96 seconds of recorded local CPU training-loop time. These exclude evaluation and are not network or edge-device benchmarks. Two runs processed 43.2 million backward examples; independent recovery added another 40,000. No cloud session was used.

## Verification and next decision

A separate process restored both runs from two steps before completion and reproduced their entire final states exactly, including server weights, delivered models, Adam moments, client compression residuals, sample chains and payload counters. Reloaded delivered checkpoints reproduced the saved logits; independent sorted-threshold and direct-logit checks reproduced all eight operating points. The tensor control also matched the historical tensor-scale checkpoint exactly.

Reserved confirmation and original validation/test arrays remained closed. The fitting and development arrays were the existing approved arrays; no fresh holdout was used to choose this setting. This development panel and fixed client partition have already informed previous choices, so a successful screen is not yet independent evidence of generalization.

**Next, replicate only this fixed block128 candidate on seeds 17 and 27 locally**, pairing it with the already recorded same-seed central, shared-Adam and FedAvg references. Keep the learning rate, budget, partition, loss, alarm rule and every safeguard unchanged. Freeze the replication decision rule before running; report per-seed class losses as well as the mean. Do not search additional block sizes or raise the five-point class allowance if a seed fails.

If the candidate survives replication, freeze it for one reserved-confirmation evaluation and a final quality-versus-communication summary. If it does not, keep uncompressed shared Adam as the quality result and compressed variants as an honest trade-off result. Either outcome supports the centralized-versus-federated thesis more directly than another classifier search. Cloud compute is unnecessary for these small local runs.

The [codec](../../src/federated/block_shared_adam.py), [runner](../../src/eval/shared_optimizer_block_compression.py), [preliminary tests](../../tests/test_shared_optimizer_block_compression.py) and [evidence tests](../../tests/test_shared_optimizer_block_compression_evidence.py) are versioned. Private checkpoints, row-level outputs and raw data are not committed. Full regression verification is recorded below when complete.
