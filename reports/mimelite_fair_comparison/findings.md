# What the MimeLite comparison can establish

## Observed results

All 21 runs completed. At the common 5% development-calibration target,
three-seed mean macro-F1 was **0.6403 for Central Adam, 0.6241 for Shared Adam,
0.5186 for FedAvg-Adam, 0.5034 for full MIME, 0.4978 for MimeLite-5 and 0.4630
for MimeLite-20**. The centralized MIME-base control scored 0.6468. The
[complete table](results.md) includes sample SD, actual false alerts, payload,
rounds, barriers and all eight class recalls.

Shared Adam stays close to Central Adam here, with a 0.0162 macro-F1 gap, but
requires 2,310 synchronization rounds and 1,883.48 MB of modeled payload.
FedAvg uses 116 rounds and 94.58 MB but loses considerable classification
quality. Neither MimeLite setting improves on FedAvg's mean macro-F1 at this
budget. Their payloads are 85.61 MB and 67.27 MB, so communication savings do
not translate into an overall quality improvement in these runs.

The centralized MIME-base control performs well. This is evidence against
blaming MIME's lower score entirely on its underlying optimizer formula. The
federated schedule, non-IID local trajectories and budget allocation remain
important differences; this comparison does not isolate their individual
causal contributions.

Rare-class recognition is still unresolved. Both MimeLite variants have zero
Web-based recall at the primary operating point. Shared Adam achieves 8.15%,
compared with Central Adam's 14.57%. MimeLite-20 has higher Brute Force recall
(25.38%) than Central Adam (20.77%), but loses heavily on other categories,
including Spoofing (8.55% versus 66.97%). A gain on one rare class is not a
general solution.

Thirty-five of the 36 predeclared paired comparisons have actual false-alert
rates within 0.5 percentage points. The exception is seed 7, MimeLite-5 versus
FedAvg: 5.0795% versus 5.6251%, a 0.5456-point difference. Their macro-F1 gap is
only 0.00019. Do not characterize that single-seed result as a controlled win;
the same calibration target does not guarantee identical realized false alerts.

For the thesis, retain the centralized-versus-federated quality/communication
trade-off. Present MimeLite as a tested comparison with a negative result under
this fixed budget, rather than use the earlier untraceable scores. Further
MimeLite work would need a separately declared tuning or anchor-approximation
study, with all extra work counted. It is not necessary to launch more cloud
training to establish the results reported here.

## Why the schedules differ from the proposed table

The earlier MimeLite scores (0.5843 and 0.5477) had no executable or recorded
source in the inspected repository. They should not be cited as reproduced
results. This study implements full-anchor MimeLite, retains all requested
methods, and runs a new comparison with an explicit computation budget.

Under this implementation, every MimeLite round computes a full gradient over
the 500,000 fitting examples, as well as its local updates. With 20 clients and
500-example local minibatches, the two schedules from the proposed table would
cost:

| Proposed schedule | Local backward examples | Full-anchor backward examples | Total backward examples |
| --- | ---: | ---: | ---: |
| 432 rounds, 5 local steps | 21,600,000 | 216,000,000 | 237,600,000 |
| 108 rounds, 20 local steps | 21,600,000 | 54,000,000 | 75,600,000 |

These are accounting calculations, **not measured runs**. They assume the
full-anchor algorithm implemented here; the unknown source of the original
table might have used another anchor approximation. Comparing those schedules
with the existing 21.6M-backward-example baselines would give MimeLite 11 times
and 3.5 times as much gradient-example work, respectively.

The new comparison instead gives every arm 23.1M backward examples. This allows
complete rounds for both MimeLite variants, with valid shorter final local
blocks for FedAvg and full MIME. The schedule was committed before training.
There are no unused gradients or anchor-only padding to make the totals match.

## What equal work does and does not mean

Every arm uses the same network, fit rows, development split, loss, client
ownership and participation, with seed-paired initial weights. The threshold
procedure uses the same benign-only calibration subset and 5% primary target.
Actual false-alert rates and all eight class recalls are retained.

Counting backward examples exposes an important cost that a table of only
local steps or synchronization rounds can conceal. It does not equalize FLOPs,
wall time, optimizer calls, unique-example coverage or tuning effort. Different
batch aggregation and anchor allocation remain properties of the algorithms.
The centralized MIME-base control also separates the lagged-moment updater
from ordinary bias-corrected Adam.

These are fixed settings, including a MimeLite learning rate inherited from
MIME. They do not establish the best possible MimeLite configuration. The
three seeds vary training randomness, not the client partition or dataset.
All new scores are from previously examined development data; confirmation
and the original validation/test arrays were not evaluated. All runs used the
local macOS ARM CPU, not RONIN or a GPU.

## Reproduction evidence

The working checkout passed 659 tests. All 21 real runs passed exact serialized
recovery of their final two units, and all 15 applicable historical checkpoints
matched in model, logits, metrics, sampled rows and counters. The report
independently recomputed 126 confusion-matrix endpoints.

The completed-job `--resume` command reused all 21 runs without training and
left the aggregate summary SHA-256 unchanged. A separate automated interruption
test verifies partial-job checkpoint resume on a small synthetic dataset; we
did not deliberately interrupt the real-data training jobs. See the
[verification record](verification.json) and [run guide](README.md).
