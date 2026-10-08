# Scaling findings and the controlled non IID comparison

The 2M cohort improves overall validation macro-F1 consistently when given 20
passes, but does not consistently beat 500k at equal processed examples. Rare-class
failures persist even on training examples. Keep these as two separate questions:
whether scaling helps federated training under controlled client mixtures, and
whether the model learns useful boundaries for rare attacks. Changing both the
partition and learning objective in one run would obscure the answer.

## Confirmed scaling results

These are final-at-budget validation scores across training seeds 7, 17, and 27.
Seed 7 motivated the study; 17 and 27 are the prospective confirmation seeds.
Values after the plus/minus sign are sample standard deviations, not confidence
intervals. False-alert rates are three-seed means.

| Model | Cohort and passes | Macro-F1 mean and SD | Benign false alerts |
|---|---|---:|---:|
| Centralized-light | 500k, 20 | 0.63223 ± 0.00568 | 19.93% |
| Centralized-light | 2M, 5 | 0.63515 ± 0.00216 | 19.68% |
| Centralized-light | 2M, 20 | 0.65572 ± 0.00102 | 17.55% |
| IID federated-light | 500k, 20 | 0.58526 ± 0.00450 | 22.31% |
| IID federated-light | 2M, 5 | 0.58257 ± 0.00560 | 21.90% |
| IID federated-light | 2M, 20 | 0.61395 ± 0.00413 | 19.07% |

At equal work (10M processed examples), centralized macro-F1 changes by +1.03,
−0.48, and +0.32 percentage points in seed order; IID changes by −0.20, −0.20,
and −0.40. At 20 passes, centralized gains are +2.85, +1.86, and +2.35 points,
and IID gains are +2.65, +3.06, and +2.89. Thus the larger training setup benefits
from the larger budget, but there is no consistent equal-work benefit. IID false
alerts decline at 20 passes in every seed; centralized false alerts do not.

[Machine-readable results](scaling-confirmation-results.json) include every seed,
all eight classes' precision/recall/F1, confusion matrices, paired differences,
means, sample standard deviations and ranges. Source settings, manifest hashes,
reported work counters and confusion-derived metrics are checked by
`src/eval/official_scaling_review.py`. This is a within-collection comparison, not
an unseen-session benchmark or a statistical significance claim.

## Rare class learning diagnosis

A read-only CPU pass evaluated each of the six final 2M models. It used all 2,391
Web-based and 1,269 Brute Force training rows, and all 2,475 and 1,284 corresponding
validation rows. Other classes have fixed probes of at most 5,000 rows. No model
was retrained and the test set remained sealed. CPU rare-class recall was checked
against the saved GPU endpoint, allowing at most one prediction difference.

| Model and class | Training recall range | Validation recall range |
|---|---:|---:|
| Centralized Web-based | 6.48–8.95% | 6.02–8.04% |
| IID Web-based | 0–2.55% | 0–2.55% |
| Centralized Brute Force | 26.95–30.50% | 26.32–29.44% |
| IID Brute Force | 27.66–28.61% | 26.64–27.34% |

Low recall on training and validation points toward failure to learn the desired
rare-class decision regions under this setup, rather than the simple story of
excellent training fit followed by a large generalization gap. This does NOT
identify capacity, optimization, preprocessing, or the objective as the sole cause.
Training recall is measured in evaluation mode, with dropout off, just like
validation. Brute Force and Web-based often become Benign or Recon predictions.

The correct Web-based class ranks in the top two on only 17.0–23.8% of centralized
validation examples and 3.35–7.56% under IID. Brute Force top-two recall is
36.4–39.0% centrally and 28.9–31.2% under IID. Most missed rare examples are not
merely a second-choice tie. These ranks are diagnostics, not an alternative
deployment metric. A threshold adjustment might trade false alerts for recall,
but cannot be presented as a demonstrated solution from these checks.

### Feature overlap and packing checks

The diagnostic compared each of the 39 individual training features for each
rare class against Benign, Recon, and Spoofing. It uses all rare training rows
and a fixed sample of up to 5,000 rows from each competitor. Symmetric univariate
AUC is max(AUC, 1−AUC), so either large or small feature values can separate a pair.
An AUC near 0.5 indicates weak one-feature ordering; this is descriptive training
separation, NOT model importance, validation performance, or a feature-selection
procedure.

Against Recon, the strongest individual feature has AUC about 0.652 for Web-based
(AVG/Tot size) and 0.658 for Brute Force (SSH). Against Benign, HTTPS reaches about
0.752 for both. These results show some signal and substantial overlap in these
one-feature comparisons. They do not prove that combinations of features cannot
separate the classes, nor that any protocol causes an attack.

An exact-vector scan of all 2M training and 2,059,284 validation inputs found no
float32 collisions involving the rare rows, no contradictory-class exact matches
for them, and no rare-vector train/validation matches. All rare rows remain unique
within their split after packing. This rules out that specific precision-collapse
explanation; it does not rule out near-duplicates, session dependence, or missing
information in the 39-feature representation.

Full diagnostic receipt: `outputs/official39-rare-diagnostic-v3.json`.
Reproducible implementation: `src/eval/official_rare_diagnostic.py`. Stratified
probe precision is deliberately not reported because sampling changes prevalence.

## Controlled non IID partition design

The previous recipe redraws client class allocations after a size-dependent
shuffle. The same seed therefore produced very different mixtures at 500k and
2M. Changing both training data size and client heterogeneity would make a
scaling conclusion ambiguous.

The replacement control anchors the existing seed-7 500k Dirichlet assignments:

1. Recover shared row positions from the verified nested subset shard order,
   not approximate comparisons of differently scaled floating-point features.
2. Keep every existing row on its original client.
3. Scale each class's client counts to its new total using largest-remainder
   allocation. Each cell differs from its ideal scaled count by less than one row.
4. Assign only the added rows with a fixed, separate per-class random stream.
   Keep originally absent client classes absent.
5. Verify complete non-overlapping coverage, exact counts, ownership retention,
   unchanged class support, and source/assignment hashes before saving artifacts.

The real-data build passed: all 500,000 old rows retained their clients. Client
sizes grow from 5,200–49,561 to 20,800–198,245. The largest absolute change in a
client's share of any class is 0.000612, or **0.0612 percentage points**. This is a
share of the class allocated to a client, not a claim of exactly identical
within-client prevalence. Global class proportions have small integer rounding
differences too. The support pattern is preserved exactly, including clients
with no rare-class examples.

| Comparison endpoint | Processed examples | Optimizer updates |
|---|---:|---:|
| Existing 500k, 20 rounds | 10,000,000 | 19,680 |
| Controlled 2M, 5 rounds | 10,000,000 | 19,585 |
| Controlled 2M, 20 rounds | 40,000,000 | 78,340 |

The primary control matches examples, not optimizer steps, aggregation rounds,
wall time, or communication. Bigger full local epochs still change local update
length and possible client drift; that is part of this scaling regime, not a
factor this partition control removes. Cohort-specific scalers and class weights
also remain. This is one anchored simulated partition, not a new general family
of heterogeneous device populations.

Prepared artifacts are in `outputs/official39-nested-partition-v1/`, with the
builder in `src/federated/official_nested_partition.py`. The runner now accepts
them through the opt-in `--partition-root` argument. The default partition path
is unchanged. See [runner readiness](frozen-partition-runner-readiness.md) for the
completed CPU gates and the remaining CUDA gates. The partition artifacts are
not a completed non-IID scaling result.

## Next actions and stopping rules

The frozen-assignment input now verifies exact coverage and class counts, and
binds the assignment and plan hashes into checkpoint provenance. CPU recovery
and the same-code 500k anchor bridge pass. The next gates are a CUDA recovery
pilot and the same-code GPU anchor bridge, followed by a full-budget 500k GPU
bridge against the archived reference before any 2M scaling conclusion.

The first controlled non-IID run should use seed 7, unchanged LayerNorm and
square-root weighted CE, all 20 clients, full local epochs, and the same 5/20
endpoints. This is a mechanism experiment, not a chosen winner. Review it before
authorizing more seeds; do not tune alpha, redraw clients, or expand to 5M based
on a favourable checkpoint.

For rare-class improvement, the next separate local diagnostic is a bounded
training-fit check and a modest nonlinear baseline on the SAME 39 features and
split. That can test whether the current model/objective is leaving available
signal unused before proposing a capacity or preprocessing ablation. Do not
combine that change with the partition control. Neither a larger GPU, more data,
nor SHAP can by itself repair a poorly learned decision boundary. SHAP later
explains model behavior; it is not proof of causal vulnerability or model quality.

Keep the cloud machine stopped until the scoped commit and handoff are ready.
No cloud training or test-set evaluation was performed in this stage.
