# SCAFFOLD retains a small gain at matched alarm targets

**SCAFFOLD has a modest advantage over FedAvg at similar false-alert rates,
but it does not solve rare-class classification.** At the predeclared 5%
alarm target, macro-F1 is 0.551147 versus 0.541743, a gain of 0.9403 percentage
point. Brute Force recall falls from 18.68% to 17.87%; Web-based recall remains
zero. The fixed diagnostic screen fails. Stop this warm-start continuation;
do not spend cloud credits extending it or relabel the earlier failed screen
as a success.

These are **training-only diagnostic results**, not unseen-data performance.
All three frozen models previously had access to every row used here during
fitting. The audit partition was excluded only from threshold fitting. No
model was retrained, and the original validation and test arrays stayed closed.

## The same automatic decision rule applies to every model

The models are the final centralized-SGD, FedAvg-SGD and SCAFFOLD endpoints
from the [ten-round trial](scaffold-findings.md). Their weights, architecture,
loss history and attack-category rankings are unchanged. This compares the
same small model, not centralized-heavy against federated-light.

For each flow, subtract the benign logit from the largest attack logit. If
that margin exceeds a threshold, predict the highest-scoring attack category;
otherwise predict Benign. This changes how readily the model raises an alarm,
not which attack category it prefers. No human review, probability-calibration
model or additional classifier is involved.

The [plan](operating-points-plan.json) was committed as `839ff82` before scores
were computed. It fixed targets of 1%, 5% and 10%, with **5% primary**. A hash
of each full39 model-input vector assigns approximately half of the training
records to calibration and half to audit. Identical float32 vectors, including
signed-zero equivalents or conflicting labels, always share a partition.
Membership does not use labels or model scores. This is not a new capture-session
split or a guarantee against near-duplicates.

Calibration contains 1,000,223 rows, including 53,216 benign examples. Audit
contains 999,777 rows, including 52,932 benign, 1,174 Web-based and 621 Brute
Force examples. Each model's threshold uses **only benign calibration scores**.
An order statistic and strict greater-than rule cap empirical calibration
false alerts at the requested target, rounding down and excluding boundary
ties. This does not guarantee the same rate in another population.

## The gain persists but is smaller at controlled alarm rates

All requested targets are shown below. Rates are achieved on the shared
training audit partition, not forced equal by fitting to audit labels.

| Target false alerts | Model | Audit false alerts | Macro-F1 | Brute Force recall | Web-based recall |
| --- | --- | ---: | ---: | ---: | ---: |
| 1% | Centralized-SGD | 0.9578% | 0.532728 | 17.71% | 0.00% |
| 1% | FedAvg-SGD | 0.9994% | 0.459983 | 0.32% | 0.00% |
| 1% | SCAFFOLD | 1.0107% | 0.474996 | 0.16% | 0.00% |
| 5% primary | Centralized-SGD | 4.9913% | 0.579812 | 24.96% | 0.00% |
| 5% primary | FedAvg-SGD | 4.9989% | 0.541743 | 18.68% | 0.00% |
| 5% primary | SCAFFOLD | 4.9724% | 0.551147 | 17.87% | 0.00% |
| 10% | Centralized-SGD | 10.1035% | 0.599611 | 27.38% | 0.00% |
| 10% | FedAvg-SGD | 9.9618% | 0.564456 | 21.90% | 0.00% |
| 10% | SCAFFOLD | 9.9467% | 0.573300 | 21.58% | 0.00% |

SCAFFOLD gains 1.5013, 0.9403 and 0.8844 macro-F1 percentage points over
FedAvg at the respective targets. It loses Brute Force recall at all three.
The gain is therefore not entirely explained by its more aggressive default
alarm rate, but it is neither large nor a rare-class rescue in this diagnostic.
Centralized-SGD remains ahead at every target.

The primary screen required at least one percentage point of macro-F1 gain,
neither rare recall declining, and sufficiently similar achieved audit alarm
rates. The alarm-rate checks pass: FedAvg and SCAFFOLD differ by only 0.0264
percentage point. The macro-F1 and Brute Force requirements fail. Brute Force
falls from **116 to 111 correct predictions out of 621**. Do not switch the
primary target to 1% after observing its larger aggregate gain.

## Gains and losses differ across attack categories

All eight audit recalls at the primary 5% target, in percent:

| Class | Centralized-SGD | FedAvg-SGD | SCAFFOLD |
| --- | ---: | ---: | ---: |
| Benign | 95.0087 | 95.0011 | 95.0276 |
| DDoS | 82.8360 | 91.1816 | 83.1162 |
| DoS | 69.4332 | 48.6458 | 67.5717 |
| Recon | 42.7712 | 44.8690 | 44.3944 |
| Web-based | 0.0000 | 0.0000 | 0.0000 |
| Brute Force | 24.9597 | 18.6795 | 17.8744 |
| Spoofing | 36.5904 | 18.5136 | 22.1727 |
| Mirai | 99.3455 | 99.3096 | 99.3188 |

Compared with FedAvg, SCAFFOLD helps DoS and Spoofing at this alarm budget,
but sacrifices DDoS and slightly reduces Recon and Brute Force recall. The
full evidence includes every class's precision, recall, F1 and confusion
matrix rather than hiding these differences behind macro-F1.

## Alarm calibration cannot repair the Web-based ranking

For every Web-based audit example, another attack category scores above
Web-based in **all three models**. Even if benign rejection were removed
entirely, none of the 1,174 Web-based rows would be correctly classified by
this fixed attack-ranking rule. The Web-based recall ceiling is therefore
zero for any single benign-versus-attack threshold on these frozen models.
This is a property of these endpoints, not proof that the dataset or neural
networks cannot learn the category.

At the primary threshold, SCAFFOLD raises some kind of attack alarm on
302 / 1,174 Web-based rows (25.72%), but assigns them to Recon (250) or
Spoofing (52); 872 are labeled Benign. Calling that 25.72% Web-based recall
would be incorrect for the project's eight-class task. Its correct Web-based
recall is zero.

Brute Force's best possible recall under unchanged attack ranking is also
limited: 185 / 621 (29.79%) for centralized-SGD, 147 / 621 (23.67%) for
FedAvg and 149 / 621 (23.99%) for SCAFFOLD. These are per-class ceilings
with benign rejection removed, not deployable operating points: removing
benign rejection also alarms on every benign flow.

## Stop this recipe and narrow the next research question

This closes the specific question about SCAFFOLD's alarm trade-off: correcting
local drift has some aggregate benefit, but changing the decision threshold
does not recover the rare categories. More rounds, a stronger GPU or more
thresholds are not justified by these results. No confirmation training is
authorized by the failed diagnostic screen.

The next method, if pursued as a separate development experiment, must change
how the model learns to distinguish one attack category from another—not
only how its benign decision boundary is placed. A useful design must address
why **centralized-SGD also misses Web-based traffic**, and separate that
shared learning failure from the additional cost of federated optimization.
Any class-aware objective would need the same centralized and FedAvg controls
before an improvement could be attributed to a federated update rule.

Do not claim a novel method from SCAFFOLD plus threshold adjustment. Preserve
this negative result in the centralized-versus-federated thesis, keep the
completed final test closed, and design any genuinely different training
method separately before launching it. The present evidence does not select
such a method or promise that it will solve the problem.

## Verification and scope

The [aggregate receipt](operating-points-results.json) records model and input
hashes, shared split counts, calibration thresholds and every result above.
Each model exactly reproduces its historical full-training argmax confusion
matrix. A zero-margin threshold also exactly reproduces ordinary argmax.
Independent execution reproduces every non-timing field, including score
hashes, thresholds and class counts. Saved model and source bytes remain
unchanged.

Each pass uses 6,000,000 forward evaluations, zero optimizer steps and zero
backward batches; replay doubles the forward count. This is local CPU work,
not cloud training. No dependency, historical runner, source checkpoint or
existing comment was changed. Pooled benign-score calibration is an offline
simulator diagnostic, not an implemented privacy-preserving federated
calibration protocol or a physical-device benchmark.
