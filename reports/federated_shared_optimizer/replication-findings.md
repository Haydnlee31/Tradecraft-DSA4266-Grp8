# Shared gradient improvement repeats across training seeds

14 October 2026. **The improvement over FedAvg repeats in both new training seeds without changing the settings.** At the primary 5% calibration false-alert target, shared-gradient Adam gains 10.76 and 11.89 macro-F1 percentage points over FedAvg in seeds 17 and 27. It remains 1.46 and 1.77 points below centralized-light Adam, respectively. Actual benign false-alert rates are comparable under the fixed tolerance.

This gives us a credible development reference for the next question: can we preserve most of that quality while communicating less? It is not yet a rare-class solution or a new federated algorithm. Shared-gradient Adam still exchanges twenty times the modeled training payload of this FedAvg configuration.

## What changed and what stayed fixed

The [protocol](replication-protocol.md), [plan](replication-plan.json), runner and eight synthetic/contract tests were committed as `9fc25f3` before the six full runs. Only initialization and sampling seeds changed. The 500,000 fitting rows, scaler, loss denominator, 20-client non-IID partition, architecture, learning rates and alarm-calibration rule stayed fixed. All models have the same 39 inputs and 5,096 parameters.

Centralized Adam uses learning rate 0.001; shared-gradient Adam and reset-client-Adam FedAvg use 0.003. Each run receives 21.6 million backward examples. We did not choose a best epoch, search another learning rate or stop an unsuccessful run early. Seed 7 is reused discovery evidence: its results helped choose the settings, so it is not a third new replication.

All thresholds use the same 200,022 calibration rows, with evaluation on the same 199,530 audit rows. Identical standardized input vectors remain in one part only. The two new seeds test sensitivity to training randomness, not to new client partitions or capture sessions. Both parts are previously examined development data drawn from historical training rows; reserved confirmation and the original validation and test arrays stayed closed.

## Results at the primary alarm target

Each model cell shows **audit macro-F1 / actual benign false-alert rate**. Differences are macro-F1 percentage points, not relative percentages.

| Training seed | Centralized Adam | Shared-gradient Adam | FedAvg with reset Adam | Shared gain over FedAvg | Central minus shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7, reused discovery | 0.6456 / 5.33% | 0.6246 / 5.43% | 0.5137 / 5.70% | +11.09 points | 2.10 points |
| 17, new replication | 0.6354 / 5.41% | 0.6209 / 5.24% | 0.5133 / 5.34% | +10.76 points | 1.46 points |
| 27, new replication | 0.6430 / 5.15% | 0.6253 / 5.44% | 0.5065 / 5.16% | +11.89 points | 1.77 points |

All six primary paired FAR differences satisfy the predeclared 0.5-percentage-point tolerance. This is a comparison at similar observed alarm burdens, not exactly equal rates. Every model produces 528 false alerts among 10,562 calibration benign rows, or 4.999%; all actual audit rates exceed 5%. Calibration therefore remains an empirical operating rule, not a deployment guarantee.

The three-seed descriptive summary is:

| Method | Mean macro-F1 ± sample SD | Macro-F1 range | Mean actual false alerts |
| --- | ---: | ---: | ---: |
| Centralized-light Adam | 0.6414 ± 0.0053 | 0.6354–0.6456 | 5.30% |
| Shared-gradient Adam | 0.6236 ± 0.0024 | 0.6209–0.6253 | 5.37% |
| FedAvg with reset Adam | 0.5112 ± 0.0041 | 0.5065–0.5137 | 5.40% |

Shared Adam's mean advantage over FedAvg is 11.25 points, with a 1.77-point mean gap to centralized Adam. Using these three means, it closes 86.4% of the FedAvg-to-central gap. This ratio describes this development comparison only. Three seeds are not enough to establish statistical significance or equivalence, and averaging does not erase seed 7's 2.10-point central gap. The original uncalibrated continuation gate remains failed; this separate alarm-matched study does not rewrite it.

## Which classes benefit

Correct eight-class recall averaged over seeds 7, 17 and 27 at the primary target is:

| Class | Central Adam | Shared Adam | FedAvg |
| --- | ---: | ---: | ---: |
| Benign | 94.70% | 94.63% | 94.60% |
| DDoS | 86.06% | 82.78% | 99.09% |
| DoS | 67.82% | 73.80% | 20.46% |
| Recon | 49.26% | 46.07% | 42.42% |
| Web-based | 13.83% | 10.00% | 0.37% |
| Brute Force | 22.82% | 20.51% | 19.74% |
| Spoofing | 66.82% | 59.52% | 21.01% |
| Mirai | 99.60% | 99.52% | 99.04% |

The main shared-versus-FedAvg improvements are DoS and Spoofing. They come with a DDoS recall reduction, so this is not an improvement in every category. Shared Adam's slight mean Brute Force advantage is also not consistent across all seeds: it loses to FedAvg in seed 7.

Rare classes remain weak. Shared Adam correctly identifies 26, 32 and 23 of the same 270 audit Web-based rows in seeds 7, 17 and 27; Brute Force counts are 24, 28 and 28 of 130. These are repeated predictions of the same rows, not three independent datasets. Centralized Adam is also weak on these classes. The results support improving the federated training trade-off, but not declaring rare-attack recognition solved.

## Sensitivity to the alarm target

The primary 5% target was fixed before these runs. Mean audit macro-F1 at the other predeclared targets is included to show sensitivity, not to select a better headline result.

| Calibration false-alert target | Central Adam | Shared Adam | FedAvg |
| --- | ---: | ---: | ---: |
| 1% | 0.6043 | 0.5808 | 0.4782 |
| 5%, primary | 0.6414 | 0.6236 | 0.5112 |
| 10% | 0.6559 | 0.6363 | 0.5268 |
| 20% | 0.6548 | 0.6409 | 0.5446 |

Shared Adam exceeds FedAvg numerically at every target in every seed. Four individual comparisons exceed the FAR comparability tolerance and must not be called tightly alarm-matched: shared versus central at 10% in seeds 7 and 27; shared versus FedAvg at 20% in seed 17 and at 10% in seed 27. The primary comparisons have no such flags. The [complete receipt](replication-results.json) retains actual rates, thresholds, all class metrics, confusion matrices and default-argmax outcomes.

## Cost and interpretation

The six new runs processed exactly 129.6 million backward examples. Recorded training-loop time totals approximately 323 seconds on the local CPU with two threads, excluding preparation and final inference. No cloud compute was used. This local simulation timing does not include actual network delays or measure physical edge performance.

| Method per run | Optimizer updates | Communication barriers | Modeled FP32 payload |
| --- | ---: | ---: | ---: |
| Centralized Adam | 43,200 | Not modeled | Not modeled |
| Shared-gradient Adam | 2,160 server updates | 2,160 | 1,761,177,600 bytes |
| FedAvg | 43,200 local updates | 108 | 88,058,880 bytes |

Payload counts include model broadcasts and gradient/model uploads to all 20 virtual clients, but exclude protocol and security overhead. Central data collection is not modeled and should not be interpreted as free. Equal backward examples do not mean equal optimizer updates, effective batch size, FLOPs or communication.

The supported insight is that this federated gradient calculation can approach centralized-light quality without moving raw training rows to the server. That does not identify one causal explanation: optimizer state, synchronization frequency and update granularity differ together. Shared-gradient Adam is an established reference, not our novelty claim, and gradients provide no privacy guarantee by themselves.

## Next experiment

Proceed to one bounded **communication-reduction experiment**, keeping the shared-gradient result as the quality reference and FedAvg as the lower-communication reference. Define the allowable quality loss and communication budget before scoring candidates. Any extension should retain globally consistent optimizer information while testing how much synchronization or transmitted payload can be removed. Report both bytes and barriers; fewer bytes alone do not resolve round-trip latency.

Do not return to a broad classifier search, expand model size or reopen MIME tuning at this stage. The immediate contribution to pursue is a better measured quality-versus-communication trade-off, not an unsupported claim that a new method already exists. Keep this next experiment local and development-only. Once a method and its settings are fixed, the reserved confirmation can support a separately authorized evaluation.

## Verification

All six final checkpoints were reloaded in a separate process and reproduced their stored logits exactly. Independent sorted-threshold and direct-logit checks reproduced all 24 new operating points. All training counters were checked against the declared schedules; the three reused discovery endpoints match their earlier evidence. Synthetic interrupted training resumes exactly for all three methods, and mismatched seed checkpoints are rejected.

The [runner](../../src/eval/shared_optimizer_replication.py), [recovery tests](../../tests/test_shared_optimizer_replication.py) and [aggregate evidence tests](../../tests/test_shared_optimizer_replication_evidence.py) preserve the historical training and calibration implementations. Row-level logits, masks, data and checkpoints stay in ignored local outputs. Only code, protocols and aggregate receipts are committed; unrelated user changes remain untouched.

The complete committed regression suite passed **738 tests in 57.464 seconds** in an isolated source archive of `e20432b`, excluding unrelated uncommitted files. The first archive run exposed a synthetic test's dependency on Git metadata; the test now mocks that provenance lookup. No frozen training/evaluation source or result changed. The 26 focused replication and earlier-evidence tests, report-link checks and Git whitespace checks also passed. These are local checks, not a GitHub CI result. This branch remains locally committed and has not been pushed.
