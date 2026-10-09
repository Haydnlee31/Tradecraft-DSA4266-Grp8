# Feature explanation findings for the official 39 feature study

The full **219-case explanation study passes its numerical checks across all
seven frozen checkpoints**. We can now describe which inputs influence these
models on the selected cases. The central finding is not a new detection rule:
the models share some influential features, but their rankings and error
explanations also vary across training seeds. Numerically reliable explanations
do not make an unreliable detector safe.

Keep RONIN stopped. The next useful check is sensitivity to the training
background and to related feature groups, before translating the strongest
patterns into monitoring hypotheses. No additional training, final-test
evaluation or deployment change was made.

## What was explained

All seven networks use the official 39-feature extension, two million training
rows and final-at-budget step-20 checkpoints: centralized light and IID light
at seeds 7, 17 and 27, plus controlled non-IID at seed 7. The original 46-feature
study remains separate. There is no matched official-39 heavy model in this
comparison.

The shared 219 cases contain a **64-case balanced core**, eight examples of each
true class, and a **155-case targeted supplement** frozen during preparation.
The supplement covers model-specific errors and correct detections; it is not
155 mistakes, and it does not reflect population prevalence. Every model
explains the same selected rows, but each can classify them differently.
Core and supplement results are never pooled into the feature rankings below.

The [frozen protocol](explanation-study-plan.json) retains the previous
[integration method](explanation-integration.md): background-averaged integrated
gradients, using all 128 original training-background rows and both 64 and 128
points per interpolation path. These are logit contributions relative to that
background, not probability changes, exact conditional Shapley values or causal
effects. The fine-grid values supply the descriptive summaries.

## Numerical reliability

Both resolutions pass **12,264 of 12,264 class-score reconstruction checks**
and **835 of 835 misclassification-margin checks**. The score and margin
featurewise resolution checks also pass, as do the global checks separately
for every model's core and supplement. No case, seed or output was excluded.
All eight pilot-case attribution arrays reproduce exactly inside the full run.

| Model | Core errors out of 64 | Supplement errors out of 155 | Fine-grid relative score residual on core | On supplement |
|---|---:|---:|---:|---:|
| Centralized seed 7 | 20 | 88 | 0.208% | 0.181% |
| Centralized seed 17 | 22 | 84 | 0.123% | 0.120% |
| Centralized seed 27 | 20 | 90 | 0.193% | 0.150% |
| IID seed 7 | 24 | 98 | 0.076% | 0.081% |
| IID seed 17 | 23 | 97 | 0.073% | 0.068% |
| IID seed 27 | 25 | 93 | 0.068% | 0.068% |
| Controlled non-IID seed 7 | 39 | 112 | 0.094% | 0.101% |

The error counts describe these selected cases, **not validation error rates**.
The percentage columns measure numerical reconstruction error, not prediction
accuracy. For actual model quality and all eight class recalls, use the
[full-validation comparison](decision-summary.md#detection-failures-that-the-averages-hide).

Global featurewise differences between resolutions range from 0.243% to 0.376%
across the 14 model/stratum groups. The largest individual class-output L1
difference is 3.071%, below the predeclared 25% limit. For error margins the
largest is 2.153%; the largest group relative fine-grid reconstruction residual
is 0.297%. No tolerances were loosened and no residual was redistributed.

The run evaluated 37,675,008 forward path rows and 301,400,064 output-gradient
rows, plus endpoint checks. These are explanation calculations, not training
examples. A separate audit verified every saved array hash and recomputed all
diagnostics and descriptive rankings. It did not rerun integration. This stage
was not independently replayed in full; its pilot subset was replayed exactly.

## Which features recur across models

For each core case, the summary uses the contribution to its **true class's
score**, whether or not the model predicted that class. Mean absolute
contribution measures magnitude; it does not say that a feature helps correct
classification. Each model's feature vector is normalized to sum to one before
the three centralized or IID seeds are equally averaged. This prevents a model
with larger logits from automatically dominating the comparison.

| Setting | Five largest features in the balanced core summary | Share assigned to Number |
|---|---|---:|
| Centralized light, three seeds | Number, Protocol Type, LLC, Header_Length, ack_flag_number | 19.12% |
| IID light, three seeds | Number, Protocol Type, ack_flag_number, Tot sum, psh_flag_number | 23.43% |
| Controlled non-IID, one seed | Number, ack_flag_number, Protocol Type, AVG, Header_Length | 20.31% |

`Number` ranks first in every individual model's balanced-core summary.
However, only `Number` and `Protocol Type` appear in all three centralized
top-five lists; IID additionally shares `ack_flag_number` across all three.
Average pairwise top-five overlap, measured as intersection divided by union,
is 0.448 for centralized and 0.587 for IID. These are descriptive overlaps on
the same 64 cases, not confidence intervals or evidence that IID explanations
generalize better.

Some class-specific patterns recur more clearly. The following features appear
in the top five for **all three centralized seeds and all three IID seeds**
for the indicated true-class score:

| True class | Shared features across all six models |
|---|---|
| Benign | Number |
| DDoS | Number |
| DoS | Number, psh_flag_number, ack_flag_number |
| Recon | Number |
| Web-based | Number, LLC |
| Brute Force | Number, SSH |
| Spoofing | Number, Max, Tot size |
| Mirai | Protocol Type, Tot sum |

This is a shortlist for investigation, not a rulebook. For example, SSH has a
repeatable contribution to Brute Force scores, but the networks still miss
many Brute Force flows. Its presence in a ranking neither proves an SSH flow is
malicious nor supplies an alert threshold. Likewise, `Number` is retained as
the release's exact column name; its prominence is not grounds to invent a
semantic definition or operational threshold without checking the extractor.

## What the missed attacks reveal

For a wrong prediction, the study explains **predicted-class score minus
true-class score**, relative to the same background margin. A positive feature
contribution pushes that contrast toward the wrong label; a negative one
opposes it. The background margin also contributes to the final decision, so
the highest-ranked feature is not necessarily what caused the mistake.

The controlled non-IID model calls all eight core Web-based cases, seven core
Brute Force cases and all eight core Spoofing cases Benign. In each of those
three error groups, `Number` and `ack_flag_number` are the two largest mean
absolute contributors to the Benign-minus-attack contrast, and their signed
means are positive. The same two features lead the corresponding supplement
groups, containing 15, 18 and 19 missed attacks respectively. This is a
consistent **single-model diagnostic** across the two selected groups, not an
independent confirmation of a causal mechanism or a measured population rate.

There is no equally simple feature rule across the centralized seeds. For
example, in selected supplemental Web-based-to-Benign errors, the mean signed
`Number` contribution is positive for centralized seeds 7 and 27, but negative
for seed 17. Those groups contain six, three and five cases respectively, so
both model and group membership differ. This is not a matched-case sign-change
test. It does show why pooling all these explanations into one blocking rule
would overstate the evidence.

Large protocol-indicator contributions sometimes oppose one another. On the
single core Spoofing-to-Benign error for centralized seed 7, for example, LLC
contributes about -27.865 logit units while ARP contributes +23.340. A small
reconstruction residual can coexist with large cancelling feature terms.
Numerical completeness alone therefore cannot establish that an individual
protocol feature has a robust, independent real-world meaning.

## Limits and next decision

These results support the existing research conclusion: the small models can
use recurring flow/protocol cues while still failing rare-attack detection.
They do not explain every training failure, establish that non-IID learning
must fail, or identify vulnerable devices. Non-IID has only one seed; the
balanced core has only eight cases per class; validation has been reused; and
the split remains within-collection rather than unseen capture sessions.

The 128-row background contains no Web-based or Brute Force examples. Reliable
integration against this one reference does not establish background
robustness. Related indicators and discrete features also make straight-line
interpolations potentially unrealistic. Feature magnitude is not a causal
effect, and these values are not ready-made firewall or anomaly thresholds.

Next, design a **small local background and feature-redundancy audit**, retaining
the same checkpoints and matched cases. First verify important feature
definitions and relationships. Then compare another predeclared training-only
background without changing the cases, method, models or QA criteria. If a
class-balanced background is added, label it a different reference question,
not simply a more accurate replacement for a prevalence-based background.
Use the resulting stability evidence to separate supported monitoring
hypotheses from background-sensitive stories. No new GPU training is justified
by this explanation stage alone.

## Evidence and reproduction

The [compact audited results](explanation-study-results.json) retain numerical
totals, global errors, any failed row/output identities, conditional top-five
summaries and seed comparisons. Raw arrays and the full receipt remain in
ignored `outputs/official39-explanation-study-v1/`. Its receipt SHA256 is
`142e159af6bbd0fe334cc4934b4006651ac781c1abf15908e6b1311bea120ea5`.

```bash
.venv/bin/python -m src.explain.official_study \
  --preparation-root outputs/official39-explanation-preparation-v2 \
  --convergence-root outputs/official39-explanation-convergence-v1 \
  --integration-root outputs/official39-explanation-integration-v1 \
  --archive-root /Users/haydn/Downloads \
  --manifest-path outputs/official39-packed-2m-v1/manifest.json \
  --preparation-plan-path reports/full_data_extension/explanation-cpu-plan.json \
  --convergence-plan-path reports/full_data_extension/explanation-convergence-plan.json \
  --integration-plan-path reports/full_data_extension/explanation-integration-plan.json \
  --plan-path reports/full_data_extension/explanation-study-plan.json \
  --output outputs/official39-explanation-study-recheck
```

The publisher `python -m src.explain.official_study_report --help` exposes the
independent saved-array audit. Neither command opens the test split. Tests cover
scope/budget guards, signed error contrasts, global and individual failures,
whole-group withholding, seed normalization, chunk limits, pilot replay,
overwrite protection, tampering and published-result integrity. Existing
training defaults, dependencies, comments and user-owned changes are preserved.
