# Error feedback preserves much of the quality with smaller messages

14 October 2026. **Four-bit error feedback reduces modeled payload by 87.0% while losing only 0.69 macro-F1 percentage points relative to the uncompressed shared-gradient reference.** At the same payload, removing error feedback lowers macro-F1 from 0.6177 to 0.5435. This is useful evidence that retaining omitted updates matters in this training setup.

However, **neither compressed candidate passes every predeclared safeguard**. Four-bit error feedback lowers Spoofing recall by 7.82 points, exceeding the five-point limit. Two-bit compression loses 3.04 macro-F1 points and causes larger class regressions. Keep the uncompressed shared-gradient model as the quality reference; no compressed candidate is promoted to confirmation or declared a completed solution.

## What was tested

The [protocol](compression-protocol.md), [plan](compression-plan.json), codec, runner and eleven synthetic/contract tests were frozen in commit `ee77fce` before real-data scores were calculated. Four fresh seed 7 runs used identical initialization, client samples, learning rate, architecture, loss and backward-example budgets:

- A full-precision message path to reproduce the historical shared-gradient result.
- Four-bit compression with error feedback in both directions.
- Two-bit compression with error feedback in both directions.
- Four-bit compression without error feedback as a diagnostic control.

Error feedback means keeping the numerical information omitted by rounding and including it in later messages. The server retains full-precision Adam moments and target weights; clients differentiate the model they actually received. Scores use that delivered model, not the hidden server target. Clients do not run local Adam or transmit raw training rows through the simulated aggregation interface.

The motivation comes from established [error-feedback work](https://proceedings.mlr.press/v97/karimireddy19a.html) and [quantized distributed Adam](https://arxiv.org/abs/2004.14180). This custom implementation is not a new-method claim. In particular, [1-bit Adam](https://proceedings.mlr.press/v139/tang21a.html) uses a different warmup and variance-preconditioning scheme; its guarantees do not apply to this experiment merely because both use compression and Adam.

## Primary results

These are seed 7 results, not the previous three-seed averages. All models use the fixed 5% calibration false-alert target and the same development audit rows. MB below means decimal megabytes, not MiB.

| Model | Audit macro-F1 | Actual false alerts | Payload MB | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Centralized-light Adam, reused | 0.6456 | 5.33% | Not modeled | Pooled training reference |
| FedAvg, reused | 0.5137 | 5.70% | 88.06 | Historical raw tensor estimate |
| Shared Adam, reproduced FP32 path | 0.6246 | 5.43% | 1,769.19 | Includes new packet metadata and startup |
| Four-bit error feedback | 0.6177 | 5.34% | 228.16 | Fails Spoofing recall safeguard |
| Two-bit error feedback | 0.5942 | 5.49% | 118.09 | Fails aggregate and class safeguards |
| Four-bit without feedback | 0.5435 | 5.61% | 228.16 | Diagnostic control, not eligible for selection |

Four-bit error feedback remains 10.40 macro-F1 points above this FedAvg configuration and 2.79 points below centralized-light. Its payload is approximately 2.59 times FedAvg's historical estimate, rather than the uncompressed reference's twenty-fold ratio. Two-bit uses 1.34 times that FedAvg estimate but loses more classification quality.

The 87.0% reduction uses the older shared-gradient raw payload, 1,761,177,600 bytes, as the conservative denominator. The new FP32 identity path costs 1,769,190,240 bytes because it additionally charges packet metadata and a startup broadcast. Compressed totals include these extra charges too; no direction or startup cost is omitted to improve the comparison.

Every primary compressed-versus-FP32 comparison has an actual false-alert difference below the fixed 0.5-percentage-point tolerance. All calibration rules remain unchanged. The actual audit false-alert rates exceed 5%, so this is not a guaranteed deployment alarm budget.

## The class safeguard catches a real trade-off

Correct eight-class recalls at the primary target are:

| Class | FP32 shared | Four-bit feedback | Two-bit feedback | Four-bit without feedback |
| --- | ---: | ---: | ---: | ---: |
| Benign | 94.57% | 94.66% | 94.51% | 94.39% |
| DDoS | 83.80% | 83.21% | 77.43% | 83.98% |
| DoS | 72.45% | 72.68% | 79.01% | 65.87% |
| Recon | 43.84% | 45.59% | 49.77% | 40.41% |
| Web-based | 9.63% | 8.89% | 1.48% | 0.00% |
| Brute Force | 18.46% | 20.00% | 18.46% | 24.62% |
| Spoofing | 63.54% | 55.71% | 45.44% | 32.83% |
| Mirai | 99.47% | 99.46% | 99.42% | 99.11% |

Four-bit feedback passes the aggregate F1, false-alert, FedAvg advantage and payload checks, but fails the maximum class-recall-drop check. Two-bit additionally fails the one-point aggregate F1 allowance; its largest recall drop is 18.09 points. The [machine-readable assessment](compression-results.json) therefore selects no candidate for replication. We did not relax these limits after seeing the outcomes or revise the older uncalibrated training gate.

Rare-class recognition is still weak. Four-bit feedback correctly classifies 24 of 270 Web-based rows and 26 of 130 Brute Force rows. The no-feedback control does better on Brute Force despite worse overall macro-F1, so feedback does not improve every class. A good aggregate result must not hide either kind of regression.

## What the control diagnoses

At identical 228,159,840-byte payload, error feedback improves macro-F1 by 7.42 points over the no-feedback control. Both use the same data, seed, quantizer and training budget. This supports retaining omitted information in this recipe, rather than treating quantization as harmless rounding.

The control removes upstream residual retention and downstream catch-up together. It does not isolate which direction contributes most. It is also only one paired seed, so the improvement needs replication before a general claim. The final distance between server target and delivered weights is 0.0051 with four-bit feedback, versus 0.8909 without it; this is a numerical consistency diagnostic, not a security or accuracy guarantee.

The four-bit aggregate result is not limited to the primary threshold. It remains within 0.44–0.69 macro-F1 points of FP32 at all four predeclared alarm targets; their actual false-alert differences remain within tolerance. That strengthens the aggregate observation, but does not cancel the class-specific failure.

## Communication and runtime limits

The implementation actually packs and decodes symbols. A message contains an 8-byte version/precision/tensor-count/sequence header, ten tensor element counts and scales, and packed values with checked padding. Message sizes are 20,472 bytes for FP32, 2,636 for four-bit and 1,362 for two-bit. Both directions and all twenty virtual recipients are charged, including the final delivered update and initial FP32 broadcast.

All arms still require **2,160 synchronization steps**, plus startup. FedAvg uses 108 rounds in the historical comparison. This experiment reduces bytes, not round-trip latency. Application payload is counted from serialized byte lengths, but transport, security, retransmission and schema-negotiation overhead are not measured. Compression provides no privacy guarantee.

The four runs total 86.4 million backward examples and approximately 204 seconds of recorded local training-loop time. FP32 takes 43.01 seconds, four-bit feedback 55.45, two-bit feedback 54.14 and no-feedback four-bit 51.28. Thus byte savings did not make this CPU simulation faster: codec work costs time, while the simulation has no actual network transfer to save. No cloud or physical edge hardware was used.

## Recommended next experiment

Keep four-bit error feedback as a promising but failed candidate, retire this two-bit setting, and make one small change aimed at rounding damage: **use a separate scale for each block of 128 coordinates instead of one scale for an entire tensor**. This could stop a few large values from forcing coarse rounding across many smaller coordinates. It is a hypothesis, not an established explanation for the Spoofing regression, and block quantization itself is not a novelty claim.

[Dettmers et al.](https://arxiv.org/abs/2110.02861) use independently quantized blocks to help preserve precision in 8-bit optimizer states. That is prior art for the general mechanism, not evidence that our proposed 4-bit network messages will preserve class recall. Our Adam states would remain FP32.

Under a proposed fixed block layout, the same model needs 45 scales. A packet would contain 2,776 bytes and the modeled run payload would be 240,255,840 bytes—about 5.3% more than current four-bit, while remaining below the existing 15% payload allowance. These are calculated design sizes, not a measured new training result. Keep the same quality and class safeguards rather than lowering the bar. Compare against the frozen tensor-scale and FP32 references before considering additional seeds.

This is a narrower next step than reopening model selection: retain the training method and budget, and test whether a finer compression scale preserves the class boundary. If that also fails, keep the quality-versus-payload trade-off as a study result instead of claiming that every trade-off has been removed. No cloud scaling or holdout scoring is justified yet.

## Verification and reproducibility

The full FP32 run exactly reproduces the historical model weights, Adam state, sample chain and all four calibrated endpoints. In a separate process, all four runs resume exactly from two steps before completion, including client residuals, delivered models, server targets, optimizer moments and byte counters. That audit uses 80,000 additional backward examples, reported separately from training. Reloaded delivered-model inference and independent sorted-threshold/direct-logit checks reproduce all sixteen new operating points.

The [aggregate receipt](compression-results.json), [codec](../../src/federated/compressed_shared_adam.py), [runner](../../src/eval/shared_optimizer_compression.py), [synthetic tests](../../tests/test_shared_optimizer_compression.py) and [evidence tests](../../tests/test_shared_optimizer_compression_evidence.py) retain the complete accounting and all class metrics. Checkpoints, messages, row-level logits and raw data are not committed. Reserved confirmation and original validation/test arrays stayed closed. This is exploratory development evidence on one fixed client partition; it does not establish performance on new capture sessions.

The full committed regression suite passed **756 tests in 57.633 seconds** in an isolated source archive of `cc3705d`, excluding unrelated uncommitted work. All 25 focused compression and previous-evidence checks passed, along with local report-link and Git whitespace checks. These are local checks, not a new GitHub CI result. All frozen training and evaluation sources remain unchanged. The work is committed locally on `experiment/federated-shared-optimizer`; it has not been pushed.

The subsequent [blockwise compression follow-up](block-compression-findings.md) passes the same five safeguards on seed 7. It leaves this tensor-scale failure unchanged and recommends paired-seed replication before promotion.
