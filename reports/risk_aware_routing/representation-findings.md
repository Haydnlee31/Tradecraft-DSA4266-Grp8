# Flow representations and rare attack detection

Stop the off-the-shelf Laya approach for this task. Neither tested text format
produces useful eight-class flow classification. The numerical fallback is
more useful: it improves a similarly sized neural network and nearly matches
the tree's overall macro-F1. It still does not solve rare-class detection or
meet the rule for replacing the tree. Keep it as a stronger neural comparator,
not a deployment-ready winner.

Neither representation adds packet payloads, device identity or new
observations. The eight-class target remains unchanged.

## Why these approaches are worth distinguishing

Turning a number into text makes it readable by a language model, but does not
teach that model which values distinguish this dataset's attacks. Laya's
[pinned model card](https://huggingface.co/convaiinnovations/laya-typed-decisions/blob/e929ae5cf69bc34259cd2f95c9e91145b818b1f0/README.md)
describes a 421-million-parameter model specialized for four synthetic text
workflows, including security incidents. Those are not benchmarks of numeric
CICIoT flow classification. The card also warns that its confidence is
uncalibrated; this experiment uses its chosen class, not a confidence-based
security policy.

Numerical embeddings have a more direct research basis for this input type.
[Gorishniy, Rubachev and Babenko, NeurIPS 2022](https://arxiv.org/abs/2203.05556)
study representing each numeric feature as a vector before feeding a neural
network. Their results motivate this experiment, not a promise of improvement
on CICIoT or its rare classes. The [authors' implementation](https://github.com/yandex-research/rtdl-num-embeddings)
provides the primary technical reference.

## Bounded Laya screen

The fixed screen uses 128 examples from each class: 1,024 rows from the
previously screened development calibration partition. Both input formats
omit `Number` and `Tot sum`. The first lists the remaining 37 feature names
and standardized values; the second adds each value's bin among fitting-data
quantiles. Neither contains the target label, source filename, row identifier
or an invented attack narrative. Nine significant digits preserve float32
values. All options and complete inputs must fit the checkpoint's context.

This is off-the-shelf inference on the Mac GPU, with no domain fine-tuning,
external decision API, upload or cloud run. A fixed, class-balanced 128-row
subset per format is replayed in a separate process. This resource-bound
amendment was declared before inspecting the first inference scores, after
the full screen proved slow and memory-intensive. It does not change the
1,024-row classification screen or its acceptance criteria, and is not full
prediction replay. The comparison tree is the existing seed-7 jointly masked,
square-root-weighted tree on exactly the same sampled rows.

Equal class sampling is useful for seeing rare attacks, but raw precision on
that balanced sample would be misleading for normal traffic prevalence.
The continuation screen therefore reweights each class's observed error
proportions to the full calibration class distribution. These are small-sample
estimates, not full-population measurements or deployment guarantees.
An arm must improve macro-F1 and both rare-class F1 scores without increasing
benign false alerts. Neither a larger prompt search nor automatic fine-tuning
is part of the plan.

### Laya results

The following F1 values are reweighted estimates at the calibration class
distribution, from the same balanced 1,024-row screen. False alerts are the
observed proportion of its 128 benign examples classified as attacks.
They must not be compared directly with the later full-audit neural scores.

| Method on this screen | Macro F1 | Benign false alerts | Web F1 | Brute Force F1 |
| --- | ---: | ---: | ---: | ---: |
| Existing tree | 0.73854 | 10.94% | 0.38939 | 0.45279 |
| Laya with numeric values | 0.00739 | 87.50% | 0.00212 | 0.00000 |
| Laya with values and fitting bins | 0.01288 | 0.00% | 0.00000 | 0.00000 |

Plain values led to 802 Web-based predictions, 220 benign predictions and
two Recon predictions. The bin-augmented format predicted benign for 1,021
rows and Web-based for three. Neither format predicted Brute Force at all.
The plain format's 71.09% Web recall is not useful detection: estimated Web
precision at the calibration prevalence is only 0.106%. The bin format's
zero benign false alerts is likewise not success; it effectively ignores
the attacks. Both fail the fixed continuation rule by a wide margin.

The fixed repeat subset reproduced every chosen class and returned rounded
probability vector exactly in both formats. The
[Laya result receipt](laya-screen-results.json) retains both the balanced-sample
and prevalence-reweighted metrics, all eight class recalls, confusion matrices,
artifact identities and the explicitly limited replay scope. The
[resource amendment](laya-repeatability-plan.json) preserves why verification
used a subset rather than repeating the entire slow GPU screen.

All features fit the context: maximum sequence lengths were 492 and 660
tokens, below the 1,024-token ceiling. This was not a truncation failure.
The two inference passes took 494 and 825 seconds, respectively, on the Mac
GPU. These include tokenization and batched inference in an unoptimized eager
runtime; the machine experienced memory pressure, so these are not clean
production latency benchmarks or comparisons with optimized Laya backends.

This result rejects these two off-the-shelf serializers and this checkpoint,
not every possible domain-fine-tuned text classifier. It does not test Jev or
a hosted Decision API. This negative screen triggered the planned numerical
alternative instead of further language-model tuning.

## Numerical embedding comparison

The numerical model divides each retained feature's fitting values into at
most 16 quantile intervals. A value becomes a vector describing its progress
through those intervals. For example, with boundaries 0, 1 and 2, the value
1.5 becomes `[1, 0.5]`: it has crossed the first interval and is halfway
through the second. This gives the network explicit local changes to learn
from instead of requiring it to construct all boundaries from a scalar.

Repeated boundaries collapse; constant features receive a zero column. This
implementation explicitly clips values outside the fitting range. It is a
fixed clipped-tail adaptation, not a reproduction of the paper's complete
architecture and hyperparameter search. Bin fitting uses only the existing
270,326 fitting rows and no labels. The resulting representation has 225
columns and feeds a 64–32-unit LayerNorm network with 17,000 parameters.

A raw-input 225–32-unit network has 17,010 parameters and serves as the
capacity control. Both use the same fitting rows, square-root class objective,
Adam settings, dropout, 60 epochs and seeds 7, 17 and 27. Each fit processes
16,219,560 examples in 31,680 optimizer steps. All six final-epoch models are
frozen before scoring the unchanged 408,027-row development audit. The
historical tree predictions must match exactly, and all six new fits are
repeated independently. No best epoch or best seed is selected.

### Numerical results

These are means across seeds 7, 17 and 27 on the same 408,027 audit rows,
including 472 Web-based and 253 Brute Force examples. F1 uses the 0–1 scale.
The near-equal parameter counts are a capacity control, not an otherwise
identical architecture: the raw network has a wider first hidden layer.

| Classifier | Macro F1 | Benign false alerts | Web F1 | Brute Force F1 |
| --- | ---: | ---: | ---: | ---: |
| Existing tree | 0.69876 | 14.50% | 0.23249 | 0.34164 |
| Raw capacity control | 0.67284 | 17.08% | 0.17998 | 0.34632 |
| Numerical embeddings | 0.69849 | 14.98% | 0.22401 | 0.36739 |

Embeddings improve macro-F1 by 0.02565 and lower benign false alerts by
2.09 percentage points versus the capacity control. Both improvements occur
at every seed, as does higher Web F1. Brute Force F1 improves at two seeds
but regresses at seed 17. A larger plain network alone does not explain the
observed representation benefit.

The tree comparison is less favourable. The embedding model has slightly
lower mean macro-F1 and Web F1, higher mean Brute Force F1, and more benign
false alerts at every seed. No seed passes all four tree-replacement criteria.
Its lower between-seed macro-F1 SD, 0.00192 versus the tree's 0.00373, measures
only model-seed variation on these same rows; it is not independent evidence
of better generalization.

| Class | Tree recall | Raw control recall | Embedding recall |
| --- | ---: | ---: | ---: |
| Benign | 85.50% | 82.92% | 85.02% |
| DDoS | 84.39% | 84.40% | 85.37% |
| DoS | 73.98% | 71.11% | 71.65% |
| Recon | 76.46% | 74.29% | 75.87% |
| Web-based | 21.82% | 13.91% | 19.21% |
| Brute Force | 34.52% | 29.64% | 33.60% |
| Spoofing | 84.51% | 76.04% | 83.89% |
| Mirai | 99.80% | 99.66% | 99.68% |

The embedding model's higher Brute Force F1 comes from better precision
(41.24% versus 33.87% for the tree), not higher recall. It still misses about
two thirds of Brute Force flows and four fifths of Web-based flows. An almost
tied macro-F1 must not hide that operational weakness.

A paired-error check also shows why simply choosing between the tree and
embedding model is not enough. Averaged across seeds, at least one is correct
on only 25.85% of Web-based examples and 39.92% of Brute Force examples. Those
are label-informed upper bounds for selecting between their two existing
class outputs, not achieved classifier performance. They do not constrain a
learned model that can produce a different prediction. Most rare examples
are currently wrong for both models.

## Decision and next experiment

Keep the numerical-embedding implementation as a credible neural extension,
but do not expand the bin grid or buy a larger GPU merely to chase this audit.
The fixed pilot has answered its question: numerical representation helps
more than extra parameters alone, but does not by itself fix the rare classes.

For a subsequent bounded local experiment, favour a retrieval-assisted numeric
classifier inspired by [TabR](https://arxiv.org/abs/2307.14338). TabR combines a
feed-forward network with learned retrieval of labelled training examples and
reports strong tabular benchmark results. The hypothesis here is that nearby
examples may preserve rare local patterns that the compact classifier's global
decision boundary misses; the paper does not establish that this will solve
CICIoT's rare classes.

Use only fitting rows as retrieval references, keep query and audit labels out
of inference, and retain both the tree and no-retrieval neural controls. This
would be automatic classification, not a human-review queue or an LLM API.
It has not been implemented or evaluated in this stage. Confirm any eventual
gain on independent data before making deployment or unseen-capture claims.

## Verification and scope

All six numerical fits replayed exactly, including fitted tensor states,
training observations and audit predictions. Each pass processed 97,317,360
examples in 190,080 optimizer steps across its six models. The historical
tree audit predictions also matched exactly. The
[numerical result receipt](numerical-embedding-results.json) preserves the
full per-seed metrics, confusion matrices, work counters and source hashes.

The clean committed snapshot passed all 519 regression tests. Both the base
project environment and isolated Laya environment passed dependency checks.
The [optional Laya environment record](laya-environment.txt) captures the
specific Mac runtime; it does not replace the universal project requirements.
The first regression-suite attempt was deliberately interrupted during Laya
inference to relieve memory pressure, then rerun successfully after Laya ended.

These are centralized classifier experiments on reused development data,
not new federated-learning results or fresh independent confirmation. The
original final test stays closed. Repeated model seeds do not create more
independent rare examples. Mac timings are local measurements, not physical
edge-device or Jetson measurements. No human-in-the-loop decision is needed.

The [fixed plan](representation-plan.json) was committed before the Laya
screen finished. The new entry points are
`scripts/official39_representation_pilot.py` and
`scripts/official39_laya_inference.py`. Laya runs in an isolated optional
environment; the main project's dependencies and earlier experiment source
files are unchanged. Model weights and row-level input files remain under
ignored `outputs/`, not in Git.
