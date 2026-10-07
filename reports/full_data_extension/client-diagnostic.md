# Official data federated client diagnostic

The local audit separates two problems in the 500k-row extension: some client
models detect a class that the aggregated model misses, while other classes are
missed by every client on the diagnostic probe. Stronger class weights alone did
not fix the cloud results. Keep LayerNorm with square-root weighting as the
provisional reference, and keep the cloud machine stopped while preparing a
controlled local-update experiment.

## Cloud findings

All scores below are final-step validation macro-F1 after 20 epochs or rounds,
using training seed 7 and partition seed 7. These are exploratory, single-seed
comparisons within the new 39-feature study, not comparisons with the legacy
46-feature study or claims about unseen capture sessions.

| Configuration | Centralized light | IID FedAvg | Non IID FedAvg |
|---|---:|---:|---:|
| BatchNorm and square-root weighting | 0.63152 | 0.58398 | 0.20812 |
| LayerNorm and square-root weighting | 0.62615 | 0.58912 | 0.36683 |
| LayerNorm and full inverse-frequency weighting | 0.57807 | 0.54736 | 0.35230 |

Replacing BatchNorm recovered non-IID Mirai recall, but did not recover Web-based,
Brute Force or Spoofing recall. Full weighting raised centralized Web-based recall
to 50.67%, but precision was only 3.57% and benign false alerts rose to 26.98%.
Those additional detections therefore come with many incorrect classifications.
The full-weighted non-IID model still had zero recall for all three missing classes.

## What the implementation actually does

Each client starts from the same global parameters, trains for one complete local
epoch with a freshly initialized Adam optimizer, and contributes to Flower FedAvg
with a weight proportional to its number of training examples. In the non-IID
partition, clients receive between 11 and 97 optimizer updates per round. IID
clients each receive 49. A round therefore does not give each client an equal
number of updates, and it is not equivalent to a centralized epoch.

The loss uses PyTorch weighted cross-entropy with mean reduction. This divides
the weighted loss sum by the sum of target weights in that batch. For a batch
containing only one class, that class weight cancels in both loss and gradient.
Thus stronger global class weights need not have the same effect on skewed
clients as they do on a mixed centralized batch. This is an objective/optimization
distinction, not evidence of a faulty PyTorch implementation.

The diagnostic computes a client-composition denominator proxy. For this
square-root-weighted non-IID partition, its relative scale ranges from 0.498 to
1.657 compared with the pooled denominator. Actual minibatch composition and Adam
also affect updates; these values are not measured gradient magnitudes.

## Local experiment

The diagnostic reads the trusted cloud archive without extracting its contents,
verifies checkpoint/environment/partition hashes and the packed-data manifest,
and checks exact client coverage and class counts against local training labels.
It uses the LayerNorm square-root-weighted final checkpoint from round 20.

Two CPU branches start independently from those same saved parameters. One trains
a full local epoch; the other takes only the first five batches of the same
seeded client shuffle. Both retain the original partition-size aggregation weights.
Changing weights to the capped number of examples would introduce another
intervention, making clients nearly equally weighted.

Each local model is evaluated before aggregation on the same seeded validation
probe: 200 examples per class, 1,600 in total. Stratification changes prevalence,
so probe precision and F1 are diagnostic, not population estimates. The global
model is also evaluated on the entire 2,059,284-row validation set before and
after each branch. No test data is opened. No saved cloud model is overwritten,
and these CPU branches are not checkpoint resumes or new selected models.

| Starting checkpoint | Before | After full local epoch | After five batches per client |
|---|---:|---:|---:|
| IID round 20 | 0.58912 | 0.58999 | 0.59033 |
| Non IID round 20 | 0.36683 | 0.36834 | 0.37677 |

These are full-validation macro-F1 scores. The full-epoch branches process 500,000
examples and take 980 IID or 984 non-IID optimizer steps. Each capped branch
processes 51,200 examples and takes 100 steps. This is deliberately a one-round
mechanism probe with unequal exposure, not a matched-compute learning curve.

In the non-IID branch, the largest local parameter displacement falls from 1.5324
to 0.3281 under the cap. Aggregate displacement falls from 0.2975 to 0.0718. Both
are Euclidean distances in parameter space, not accuracy or proof of client drift
as the sole failure mechanism.

After a full local epoch, the strongest local Spoofing recall on the shared probe
is 100%, whereas aggregate Spoofing recall is zero. A specialist may also
overpredict its preferred class, so local recall alone does not establish a good
local classifier; the receipts retain precision and all-class metrics too.
For Web-based and Brute Force, every local model has zero recall on this probe.
Aggregation can therefore help explain some missing detections, but not all.
The capped aggregate still has zero full-validation recall for these three classes.

## Reproduce and inspect

Run from the repository root using the existing local environment. Use a fresh
output filename; the diagnostic refuses to overwrite evidence.

```bash
.venv/bin/python -m src.eval.official_client_diagnostic \
  --data-root outputs/official39-packed-500k-v1 \
  --archive /Users/haydn/Downloads/tradecraft-official39-layernorm-backup.tgz \
  --member-root outputs/official39-layernorm-500k-v1/dirichlet \
  --output outputs/official39-local-client-diagnostic-recheck/dirichlet.json
```

Use the `iid` member root for the IID control. Detailed receipts are kept locally
in `outputs/official39-local-client-diagnostic-v1/dirichlet.json` and `iid.json`.
They include archive/checkpoint/source hashes, probe row indices, every client's
metrics and displacement, actual examples and updates, and full-validation scores.

Tests check single-class loss/gradient cancellation, deterministic probe selection,
repeatable capped branches, preservation of the starting model and aggregation
weights, budget accounting, and single-client equivalence to direct training.
The existing training runner and historical comments are unchanged.

## Next experiment gate

Do not launch another ad hoc sweep or the 2M/5M learning curve yet. The
[bounded-update extension](bounded-updates.md) now supplies an opt-in recoverable
policy and exact aggregate-update accounting, with a CPU-tested recovery gate
that must also pass on CUDA before the planned comparison.
Keep normalization, loss, client assignments, initialization and learning rate
fixed. Include IID as a control and record actual samples, optimizer steps and
aggregation rounds. More frequent aggregation also means more Adam resets and
communication; a capped-round intervention must be described as that combined
policy, not as an isolated pure effect of parameter drift.

Compare both a fixed-round view and an explicitly matched-update view with a
predefined final-budget evaluation. The present one-round probe implements neither
a multi-round training policy nor an exact matched-update budget. Confirm recovery
on CPU and GPU before any long run, then use additional predefined seeds before
promoting a configuration. Leave the new test split sealed.
