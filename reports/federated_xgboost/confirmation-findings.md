# FedLC cyclic XGBoost on the reserved confirmation cohort

**The frozen FedLC cyclic XGBoost passes on the same 398,502 confirmation rows
that produced compressed shared Adam's 0.6325.** Its three-seed mean macro-F1
at the 5% alarm target is **0.6456**, against 0.6325 for block128 compressed
shared Adam, 0.5187 for FedAvg and 0.6535 for centralized-light. Modeled
payload is about 54 MB, against 240 MB for block128 and 88 MB for FedAvg.

## Procedure

The procedure followed the shared-optimizer confirmation. The
[plan](confirmation-plan.json) and [freeze receipt](confirmation-freeze.json)
were committed in `4f9f75e` before any confirmation inference.

- Each seed's model was retrained with the committed code and reproduced its
  stored development margins byte for byte.
- Thresholds are the development-calibrated ones; none were refitted.
- The confirmation data receipt hash equals the one in the committed
  block128 confirmation results.
- The cohort was scored once.

## Results (5% development-calibrated target)

| Seed | FedLC cyclic XGBoost | Block128 | Difference | FedLC actual FAR | Block128 FAR | Central-light | FedAvg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 0.6427 | 0.6303 | +1.25 pt | 5.37% | 5.45% | 0.6582 | 0.5213 |
| 17 | 0.6474 | 0.6304 | +1.70 pt | 5.46% | 5.33% | 0.6448 | 0.5197 |
| 27 | 0.6466 | 0.6369 | +0.97 pt | 5.42% | 5.29% | 0.6575 | 0.5149 |
| **Mean** | **0.6456** | **0.6325** | **+1.31 pt** | | | **0.6535** | **0.5187** |

The predeclared rule passes at every seed: macro-F1 is at least block128's
minus one point, actual FAR is within 0.13 points (limit 0.5), and payload is at
most 54.5 MB (limit 88 MB). The mean also exceeds block128's, so the plan's
descriptive superiority condition holds. That is not a significance test.

FedLC closes about 94% of the FedAvg-to-centralized gap, against 84% for
block128. At seed 17 it scores above centralized-light.

Mean macro-F1 by target:

| Target | FedLC | Block128 |
| --- | --- | --- |
| 1% | 0.5864 | 0.5876 |
| 5% | 0.6456 | 0.6325 |
| 10% | 0.6616 | 0.6471 |
| 20% | 0.6640 | 0.6507 |

Mean recall at 5% (%):

| | Benign | DDoS | DoS | Recon | Web | Brute | Spoof | Mirai |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FedLC cyclic XGBoost | 94.6 | 87.1 | 65.1 | 53.1 | 10.5 | 18.8 | 68.2 | 99.2 |
| Block128 shared Adam | 94.6 | 83.9 | 72.2 | 47.5 | 9.9 | 26.3 | 55.2 | 99.5 |

The gain comes mainly from Spoofing (+13.0 points) and Recon (+5.6). DoS
(−7.1) and Brute Force (−7.5, about 20 of 260 rows) are lower. At 1% FAR the
two methods are effectively tied.

## Limits

- The cohort was opened before this study, and block128/FedAvg/central scores
  on it were known while choosing which methods to explore. FedLC itself was
  never scored on it before this single run. This is weaker evidence than the
  original prospective confirmation.
- The rows come from the same collection as training; one client partition;
  three training seeds on the same rows.
- Payload is modeled application bytes, not measured traffic. Cyclic visits are
  sequential, so latency scales with client count. The model is about 2.7 MB
  versus 20 KB for the neural network.
- The cohort must not be used to tune or select anything further.

Reproduce: `python -m src.eval.federated_xgboost_confirmation freeze|infer|score`.
Results: [confirmation-results.json](confirmation-results.json).
