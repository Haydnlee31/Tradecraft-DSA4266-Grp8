# FedLC calibration recovers DoS in cyclic federated XGBoost

**Porting FedLC's label-skew calibration to cyclic XGBoost passes the
development rule.** The three-seed mean macro-F1 at the 5% alarm target rises
from 0.6240 for the committed cyclic scheme to **0.6334**, against 0.6219 for
compressed shared Adam (block128). Payload is about 54 MB. DoS recall rises
from 46.6% to 64.9%, recovering most of the trees' main weakness.

This is exploratory evidence. The development rows had already scored three
cyclic XGBoost checks, so the study is development-adaptive in motivation,
although selection used only the component study's fit-row holdout.
Confirmation, original validation and test rows were not opened.

## Which mechanisms transfer to trees

The [plan](mechanism-plan.json) mapped neural federated mechanisms onto
cyclic boosting:

| Mechanism | Tree version | Outcome |
| --- | --- | --- |
| FedLC (Zhang et al., ICML 2022) | Each client trains against the shared model's scores shifted by −τ·n_ic^(−1/4) (τ = 1), so its own class counts are not baked into the shared trees | **Selected** |
| Local-prior correction | Shift = log π_local − log π_global | Rejected: rare-class recall collapsed |
| FedAvg → FedXgbBagging | All clients boost from the same model; leaves averaged by population weight | Dropped: averaged eta-1.0 steps diverged to DDoS-only predictions; stable small-eta bagging needs ~20× the payload |
| FedNTD | Not-true distillation during local steps | Not applicable: with one round per visit the local model equals the global one, so the term is zero |
| FedPGP | Prototype projection | Not applicable: trees have no learned embedding |
| FedProx / SCAFFOLD | Proximal term / gradient control variates | No shared parameter vector; FedLC's per-class score shifts are the nearest tree analog |

Offsets are passed through XGBoost's `base_margin`. From the second client
visit, each client trains with base_margin = the booster's fitted intercept +
its offset. The first visit trains exactly as the committed scheme. Prediction
uses the booster alone. With a zero offset this path is bit-identical to plain
training.

One design error was caught before any development scoring. The first tuning
run used a fixed log-prior intercept for every arm, which lowered the paired
reference from 0.6397 to 0.6269 on the holdout and would have overstated the
mechanisms. Amendment 1 in the plan records the correction. After it, the
reference's holdout margins are byte-identical to the committed stage-2 run.

## Tuning holdout (seeds 7 and 17)

| Arm | Macro-F1 | DoS | DDoS | Spoofing | Web | Brute |
| --- | --- | --- | --- | --- | --- | --- |
| FedLC | **0.6422** (0.6418, 0.6425) | 0.655 | 0.869 | 0.625 | 0.102 | 0.212 |
| Reference (committed scheme) | 0.6397 (0.6457, 0.6336) | 0.482 | 0.937 | 0.635 | 0.093 | 0.212 |
| Local-prior correction | 0.6189 | 0.593 | 0.891 | 0.766 | 0.024 | 0.061 |
| FedLC + power-0.75 class weights (stage b) | 0.6342 | ~0.80 | – | – | 0.130 | 0.190 |

The stage-b weight variant raised DoS further but lowered macro-F1, so the
plan kept sqrt weights.

## Development result

| Seed | FedLC | Reference (reproduces commit d3093d8) | Block128 | Payload |
| --- | --- | --- | --- | --- |
| 7 | 0.6337 | 0.6264 | 0.6241 | 54.5 MB |
| 17 | 0.6370 | 0.6260 | 0.6164 | 53.6 MB |
| 27 | 0.6295 | 0.6197 | 0.6252 | 53.9 MB |
| **Mean** | **0.6334** | **0.6240** | **0.6219** | |

Rule checks all pass:

- FedLC exceeds the paired reference by 0.94 points (needs 0.5).
- Every seed is above its block128 score (needs no more than one point below).
- Actual FAR is within 0.20 points of block128 (needs 0.5).
- Payload is at most 54.5 MB (needs 88 MB).

Mean recall at 5% (%):

| | Benign | DDoS | DoS | Recon | Web | Brute | Spoof | Mirai |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FedLC cyclic XGBoost | 94.6 | 87.1 | 64.9 | 52.5 | 6.5 | 15.6 | 68.2 | 99.2 |
| Reference cyclic XGBoost | 94.6 | 94.1 | 46.6 | 51.9 | 6.0 | 14.4 | 66.6 | 99.1 |
| Block128 shared Adam | 94.7 | 83.7 | 72.2 | 47.1 | 9.5 | 19.2 | 56.3 | 99.5 |

| Target | FedLC | Reference | Block128 |
| --- | --- | --- | --- |
| 1% | 0.5723 | 0.5585 | 0.5781 |
| 5% | 0.6334 | 0.6240 | 0.6219 |
| 10% | 0.6515 | 0.6449 | 0.6367 |
| 20% | 0.6563 | 0.6513 | 0.6404 |

FedLC improves on the reference at every alarm target and leads block128 from
5% upward. At 1% FAR block128 remains slightly ahead.

## Limits

- DoS is still 7 points below block128, and Web-based/Brute Force recall is
  below block128 (6.5% / 15.6% vs 9.5% / 19.2%).
- The model is about 2.7 MB versus 20 KB for the neural network.
- τ = 1 is FedLC's calibration constant here with no search; the global class
  counts used for weights are a simulation convenience.
- One client partition and development-adaptive history; a confirmation claim
  needs a newly reserved population.

Reproduce: `python -m src.eval.federated_xgboost_mechanisms tune|final|score-final`
(`final --final-arms fedlc:sqrt reference:sqrt`); holdout scoring uses
`python -m src.eval.federated_xgboost_components score`.
