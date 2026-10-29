# PLE lifts compressed shared Adam to the old centralized level

**A fixed piecewise-linear numerical encoding (PLE) passes the development
screen.** Compressed shared Adam with PLE reaches 0.6465 macro-F1 at the 5%
alarm target, against 0.6241 without it and 0.6456 for the historical
centralized-light model. That gain costs 3.1 times the payload: 752 MB instead
of 240 MB.

This is exploratory evidence from one training seed on the historical client
partition. The development rows were examined in earlier studies.
Confirmation, original validation and test arrays were not opened.

## Design

The [plan](screen-plan.json) fixed every setting before training. Each
standardized feature is divided into up to 16 quantile bins fitted on the
500,000 fit rows, without labels. Each value becomes cumulative clipped ramps
over its feature's bins; the encoding matches the archived pilot's encoder.
All 39 features expand to 213 inputs. The 64–32 LayerNorm network therefore
has 16,232 parameters instead of 5,096. The bins are fixed preprocessing, like
the shared scaler, and are fitted on pooled data as a simulation convenience.

Both arms reuse the historical learning rates without a new search: 0.001
centrally and 0.003 for the compressed method. Both also reuse the historical
initialization seed, sample chains (verified by hash), loss, codec, the 21.6M
backward-example budget and the calibration/audit split.

## Results (audit half of development, seed 7)

| Method | Macro-F1 @5% | Actual FAR | Parameters | Payload |
| --- | --- | --- | --- | --- |
| Central Adam with PLE | 0.6534 | 5.10% | 16,232 | – |
| Central Adam (historical) | 0.6456 | 5.33% | 5,096 | – |
| **Compressed shared Adam with PLE** | **0.6465** | 5.63% | 16,232 | 752 MB |
| Compressed shared Adam (historical) | 0.6241 | 5.39% | 5,096 | 240 MB |
| FedAvg (historical) | 0.5137 | 5.70% | 5,096 | 88 MB |

Screen checks against historical compressed shared Adam all pass: +2.24
macro-F1 points (needs ≥ 1.0), 0.24-point FAR difference (≤ 0.5) and largest
class recall loss 1.5 points, for Web-based (≤ 5). The gain holds at every
alarm target:

| Target | Compressed | Compressed + PLE | Central | Central + PLE |
| --- | --- | --- | --- | --- |
| 1% | 0.5831 | 0.6079 | 0.6125 | 0.6207 |
| 5% | 0.6241 | 0.6465 | 0.6456 | 0.6534 |
| 10% | 0.6362 | 0.6589 | 0.6606 | 0.6643 |
| 20% | 0.6428 | 0.6597 | 0.6582 | 0.6614 |

## Class recall at 5% (%)

| Class | FedAvg | Central | Compressed | Central + PLE | Compressed + PLE |
| --- | --- | --- | --- | --- | --- |
| Benign | 94.3 | 94.7 | 94.6 | 94.9 | 94.4 |
| DDoS | 99.4 | 87.7 | 84.4 | 86.9 | 84.9 |
| DoS | 18.0 | 64.7 | 71.1 | 69.4 | 72.4 |
| Recon | 44.3 | 50.5 | 46.1 | 53.9 | 53.1 |
| Web-based | 0.7 | 11.9 | 8.5 | 11.1 | 7.0 |
| Brute Force | 19.2 | 24.6 | 19.2 | 21.5 | 20.8 |
| Spoofing | 23.2 | 68.6 | 58.8 | 72.1 | 72.2 |
| Mirai | 99.0 | 99.5 | 99.5 | 99.5 | 99.5 |

Most of the gain is Spoofing (+13.4 points) and Recon (+7.0). Spoofing was the
largest contributor to the earlier non-IID gap. Rare classes are unchanged
or slightly worse: Web-based recall is 7.0%, so this is not a rare-class fix.

## Follow-up: the gain replicates and is not explained by capacity

The [follow-up plan](followup-plan.json) fixed two rules before training. PLE
must beat block128 at the same seed at every seed, under the screen's three
checks. Its three-seed mean must also beat a raw-input network of matched
capacity by at least one point. That control is 39 → 214 → 32 LayerNorm with
16,196 parameters, trained by the same compressed shared Adam. Seeds set
initialization and client sampling. Every sample chain matched the historical
block128 run at its seed, and seed 7 reproduced the screen result exactly.

| Seed | Block128 (historical) | Compressed + PLE | Raw, matched capacity |
| --- | --- | --- | --- |
| 7 | 0.6241 | 0.6465 | 0.6177 |
| 17 | 0.6163 | 0.6536 | 0.6229 |
| 27 | 0.6253 | 0.6484 | 0.6190 |
| **Mean** | **0.6219** | **0.6495** | **0.6198** |

Both rules pass. PLE gains 2.24, 3.72 and 2.31 points at the three seeds. FAR
differences are at most 0.34 points, and no class loses more than 3.0 recall
points. The capacity margin is 2.96 points. The wider raw network scores no
better than the small one, so extra parameters alone do not explain the gain.

Mean recall at 5% (%), with mean actual FAR:

| | Benign | DDoS | DoS | Recon | Web | Brute | Spoof | Mirai | FAR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Block128 | 94.7 | 83.7 | 72.2 | 47.1 | 9.5 | 19.2 | 56.3 | 99.5 | 5.33% |
| + PLE | 94.8 | 84.4 | 73.5 | 53.3 | 8.4 | 21.3 | 70.6 | 99.6 | 5.24% |
| Raw wide | 94.6 | 87.0 | 65.8 | 47.2 | 10.0 | 19.0 | 55.9 | 99.6 | 5.44% |

The Spoofing (+14.3) and Recon (+6.2) gains hold across seeds. Web-based
recall stays below 10%.

## What this still does not establish

- **Generalization.** All results reuse previously examined development rows
  and one client partition. A claim beyond exploration needs a newly reserved
  evaluation population; the opened confirmation set must not be reused.
- **A matched central PLE comparison across seeds.** Central + PLE was run at
  seed 7 only (0.6534).
- **Communication efficiency.** Payload rises 3.1 times; synchronizations stay
  at 2,160. Most added parameters sit in the first layer (213 × 64).

Reproduce with
`python -m src.eval.ple_shared_adam_screen --data outputs/shared-optimizer-development-data-v1 --output <new folder>`
and `python -m src.eval.ple_shared_adam_followup` with the same arguments.
