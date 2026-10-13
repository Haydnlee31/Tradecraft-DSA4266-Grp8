# SCAFFOLD improves fitting but raises false alerts

**SCAFFOLD improves macro-F1 over matched FedAvg-SGD, but fails the
predeclared false-alert limit.** At the fixed endpoint, macro-F1 rises from
0.571436 to 0.593971 while benign false alerts rise from 15.10% to 24.45%.
Brute Force gains only five correct training predictions; Web-based detection
remains zero. Do not extend this recipe or present it as a deployable win.

These are **training-only results** on the official39 two-million-row cohort,
from one seed and one warm-started non-IID model. Both runs used the local CPU;
no cloud, validation arrays or test arrays were used. The completed final test
has not been reopened. The results support an optimization diagnosis, not
generalization, global novelty or a claim that rare attacks are solved.

## A matched comparison isolates the correction

All three arms start from the same archived round-20 LayerNorm model and use
the same class weights. The new shared recipe is plain SGD, learning rate
0.01, weight decay 0.00001, batch size 512 and a fixed training-mean loss
denominator. Each of 20 federated clients draws 50 uniform minibatches with
replacement per round. Both federated arms use identical rows, row order and
dropout seeds. Every client participates in all ten rounds.

The centralized control takes 1,000 sequential SGD steps per comparison
round, sampling uniformly from the pooled training population. It matches
processed examples and total optimizer calls, not the realized rows or the
federated update trajectory. Centralized sequential updates and parallel
local updates followed by averaging are different optimization processes.

Equal local step counts prevent unequal client sizes from also creating
unequal local optimizer-step counts. Aggregation still weights by each
client's assigned population size, not its equal number of sampled examples.
Small clients may repeat rows more often; this is not a full-epoch protocol
or a claim of equal unique-row coverage. The fixed loss denominator makes
population-weighted client gradients target the same expected objective as
the centralized reference.

SCAFFOLD uses persistent client and server correction vectors to compensate
for local drift. The implementation uses Option II, plain SGD and full
participation, with a data-size-weighted server control. This is an adaptation
of the published method, not a new algorithm. ([SCAFFOLD, Algorithm 1 and
Section 4](https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf))

This trial changes optimizer, loss denominator and local sampling relative
to historical reset-Adam runs. Therefore, **the SCAFFOLD-specific comparison
is against the new FedAvg-SGD arm**, not the older Adam result. Zero-control
SCAFFOLD and FedAvg-SGD reproduce each other exactly in their first round,
including model weights and client work. Both reach macro-F1 0.497545 before
SCAFFOLD has a distinct effect. The large shared recovery cannot be credited
to the control-vector correction alone.

## The final endpoint improves some attacks and worsens benign traffic

The endpoint is the tenth scheduled continuation round, never a best-round
selection. All three models processed 5,120,000 sampled training examples
and made 10,000 optimizer calls.

| Training metric | Centralized-SGD | FedAvg-SGD | SCAFFOLD |
| --- | ---: | ---: | ---: |
| Macro-F1 | 0.606799 | 0.571436 | 0.593971 |
| Benign false alerts | 18.3828% | 15.1034% | 24.4470% |
| Web-based correct | 0 / 2,391 | 0 / 2,391 | 0 / 2,391 |
| Brute Force correct | 346 / 1,269 | 281 / 1,269 | 286 / 1,269 |

Relative to matched FedAvg-SGD, SCAFFOLD gains **2.2534 percentage points** of
macro-F1 but adds **9.3436 percentage points** of benign false alerts. That
is 9,918 additional false alerts among 106,148 benign training examples.
Brute Force recall gains only 0.3940 percentage point. The fixed screen allowed
at most a 0.5-percentage-point increase in false alerts, so the overall gain
does not qualify for automatic continuation. The separate one-percentage-point
rare-recall threshold is also unmet. None of these settings was retuned after
seeing the result.

All eight recalls, in percent:

| Class | Centralized-SGD | FedAvg-SGD | SCAFFOLD |
| --- | ---: | ---: | ---: |
| Benign | 81.6172 | 84.8966 | 75.5530 |
| DDoS | 82.8079 | 91.1386 | 83.0617 |
| DoS | 69.2864 | 48.5587 | 67.4919 |
| Recon | 68.6709 | 68.6256 | 75.4862 |
| Web-based | 0.0000 | 0.0000 | 0.0000 |
| Brute Force | 27.2656 | 22.1434 | 22.5374 |
| Spoofing | 54.8339 | 25.9236 | 43.4562 |
| Mirai | 99.3326 | 99.2886 | 99.3004 |

The gain is concentrated in DoS, Recon and Spoofing, not the rare classes.
SCAFFOLD closes about 64% of the measured macro-F1 gap between FedAvg-SGD
and centralized-SGD in this short fitting experiment, but this is not a
64% improvement in accuracy or a reduction in every kind of error.

## Client updates become more consistent

At the endpoint, the population-weighted RMS spread of client parameter
displacements falls from 0.26271 for FedAvg-SGD to 0.07699 for SCAFFOLD,
about 71% lower. This measures dispersion around each arm's own weighted mean;
it is not distance to a known true global update, and the arms now start from
different models. It is consistent with synchronization, not independent proof
that all harmful drift has disappeared.

The weighted server-control invariant remains within 1.75e-8 in maximum
absolute error. Option II's update agrees with the independently accumulated
mean local gradient within a maximum L2 residual of 1.06e-5, including FP32
update rounding. The initial exact FedAvg bridge, synthetic sign tests,
weighted-objective checks and serialized-control recovery tests guard against
mistaking an implementation bug for a scientific finding.

## Costs move to training rather than inference

SCAFFOLD adds one 5,096-element FP32 control vector per client: 20,384 bytes
of persistent client state and one extra model-sized payload in each direction
per round. Across 20 clients and ten rounds, this is an estimated 4,076,800
extra upload bytes and the same extra broadcast volume, excluding protocol
overhead. The single-machine simulator stores 428,064 bytes for the server
and all client controls. These are tensor/payload counts, not measured edge
RAM, network latency or power.

The classifier and inference parameter count are unchanged. Raw records need
not leave their assigned clients in the algorithm, but sharing model/control
updates is not a privacy guarantee. No differential privacy or secure
aggregation was added.

One pass processed 15,360,000 training examples across all three arms,
30,000 optimizer calls and 62,000,000 training-evaluation forwards. The latter
includes the shared initial evaluation and every round's full training
metrics. Independent replay doubles those audit costs. No hyperparameter
sweep or additional gradient-probe backpropagation was launched.

## Compare the frozen models at matched false-alert rates next

Keep the current result as a useful but unsuccessful trade-off under the
predeclared screen. Do not add rounds or lower the safety standard to call it
a win. The correction has a measurable effect, so another classifier search
would not answer the remaining question.

The next small diagnostic should ask whether SCAFFOLD separates benign from
attacks better at the **same false-alert rate**, or simply operates at a more
aggressive decision point. Freeze these models. Predeclare a training-only
calibration/audit partition and operating points before selecting any
threshold; evaluate the same automatic rule for all three models. Both
partitions were already seen during model fitting, so that diagnostic would
still not estimate unseen-data performance. No human-review layer or new
classifier is needed.

Only if a useful advantage survives that matched comparison should a separate
from-scratch, paired-seed development experiment be planned. The original
test set must remain closed. No further training or calibration was performed
as part of this trial.

## Reproducibility

The [plan](scaffold-plan.json) and implementation were committed as `30220fc`
before execution. [Full aggregate evidence](scaffold-results.json) includes
every round, class confusion matrix, client sample hash, work counter and
control check. The primary run and independent replay match in every
non-timing field. Actual tensors in both saved endpoint files match the
recorded hashes for all three models and all control states. Private arrays
and checkpoint tensors remain outside Git. Original code and comments, earlier
failed methods and unrelated user changes are preserved.
