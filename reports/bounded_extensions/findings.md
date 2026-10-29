# What the bounded extensions add

**Retain compressed Shared Adam as the primary method, and present the extensions
as evidence about mechanisms and costs.** It is a compact compromise, not the
highest-scoring method among every representation and model family. All numbers
in the first table are three-seed means on the same development audit population;
they must not be substituted for the primary confirmation scores.

| Comparison | Development macro-F1 | Interpretation |
| --- | --- | --- |
| FedAvg → FedAvg + neural FedLC | 0.5112 → 0.5223 | A modest, consistent improvement with unchanged payload |
| Compressed Shared Adam → + neural FedLC | 0.6219 → 0.6202 | No aggregate improvement; useful class trade-offs |
| Compressed Shared Adam → + PLE | 0.6219 → 0.6495 | Strong representation gain, with 3.13× training payload |
| Raw wider compressed model → PLE compressed model | 0.6198 → 0.6495 | Extra parameter count alone does not explain the gain |
| Cyclic XGBoost → + FedLC | 0.6240 → 0.6334 | Label calibration transfers usefully to the tree mechanism |

## Neural FedLC helps FedAvg, not every optimizer

The paired runs share initialization, sampled rows, learning rate, model, client
partition and 21.6M backward examples. All six unchanged controls reproduced their
published model, sampling, work and evaluation endpoints exactly.

FedAvg gains at all three seeds and passes the predeclared rule. Mean recall
across Web-based, Brute Force and Spoofing rises by 2.42 percentage points, with
mean F1 across those classes rising by 1.64 points. Mean actual false alerts rise
from 5.40% to 5.53%; these are not identical operating points.

Compressed Shared Adam fails that same rule. Mean target-class recall rises by
1.70 points, below the 2-point requirement. At seed 7, DoS recall gains 10.10 points
but DDoS loses 6.91 points, exceeding the allowed 5-point loss for a non-target
class. The mean macro-F1 falls slightly. Calibration redistributes errors; it does
not create universally better recognition. These are fixed tau=1 adaptations
stacked on the existing weighted loss, not a comprehensive FedLC tuning study.

## PLE narrows the gap with a matching central representation

The missing centralized PLE controls were run at all three seeds. Central PLE
scores **0.6554**, versus **0.6495** for compressed PLE: a 0.0059 gap. Comparing
compressed PLE only against raw-input Central Adam (0.6414) would conflate a
representation improvement with an optimizer improvement.

Both use the same development calibration rule, not exactly the same achieved
false-alert rate. Their largest paired FAR difference is 0.54 percentage points
at seed 7, slightly above the 0.5-point safeguard used elsewhere; thresholds were
not adjusted afterward to hide it. Mean FAR is 5.29% central and 5.24% compressed.

PLE has 16,232 parameters; the raw wider control has 16,196 and scores 0.6198.
The chosen small model has 5,096. PLE's gain is therefore not explained by adding
similar capacity alone. Spoofing and Recon improve most; Web-based recall remains
only 8.4% on development, and Brute Force 21.3%. This does not solve rare classes.

The cost is **752.29 MB** rather than **240.26 MB**, with the same 2,160 frequent
synchronizations. Quantile bins are fitted on pooled fitting rows only; that
preprocessing is a simulation convenience, not an implemented private distributed
quantile protocol.

## Cyclic trees improve, but the communication assumption matters

FedLC improves cyclic XGBoost by 0.0094 macro-F1, with paired actual FAR differing
by at most 0.03 percentage points. DoS recall rises from 46.6% to 64.9%; DDoS falls
from 94.1% to 87.1%. Web-based/Brute Force remain weak (6.5% / 15.6%).

The new centralized tree control scores **0.6697**, compared with **0.6334** for
FedLC cyclic trees. It matches the maximum depth and 200 boosting rounds, but
uses 100M fitting-row visits versus 5M cyclic. It is a capacity/schedule control,
not an equal-compute experiment. It is deterministic across these seeds; its zero
seed SD does not imply zero uncertainty on future data.

The historical ~54 MB figure assumes cached incremental trees distributed to
clients. The simulation did not implement that transport. Serializing and sending
the full model to and from each client visit totals **529.44 MB on average** for
FedLC, versus 542.95 MB for plain cyclic trees. Both figures are application-byte
models, not measured network traffic. The tree model itself is about 2.7 MB,
versus about 20 KB of parameters in the small neural model. There are 200 sequential
client visits, which are not equivalent to 200 full-participation neural rounds.

## Confirmation is reproduced, not newly discovered

All three frozen candidate models in each family reproduce their historical
confirmation results with unchanged development thresholds:

| Frozen candidate | Confirmation macro-F1 | Actual FAR |
| --- | --- | --- |
| Chosen compressed Shared Adam (primary reference) | 0.6325 | 5.36% |
| PLE + compressed Shared Adam | 0.6603 | 5.31% |
| FedLC cyclic XGBoost | 0.6456 | 5.42% |

These are already exposed, within-collection confirmation rows. The new neural
FedLC and centralized PLE/tree controls were **not** scored on confirmation.
Do not claim PLE beats a matched centralized PLE model on confirmation: that
comparison was not performed. No original validation/test arrays were opened.

## Recommended thesis presentation

Keep the original centralized/FedAvg/shared/compressed comparison as the main
experiment. Use these extensions to make three supporting points: loss correction
can help a weaker optimizer without helping an already stronger one; numerical
representation matters beyond parameter count; and communication rankings depend
on the implemented transport and synchronization pattern. Keep the adverse class
results and the stronger centralized tree control visible. The study remains a
trade-off analysis, not a claim that the selected model is universally best.

The [full table](results.md), [machine-readable evidence](results.json),
[plot](development-comparison.png) and [reproduction guide](../../studies/bounded-extensions/README.md)
provide the values and limitations behind these conclusions.
