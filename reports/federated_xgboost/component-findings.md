# Cyclic federated XGBoost matches compressed shared Adam at a quarter of the payload

**After tuning on a fit-row holdout, cyclic federated XGBoost passes the
development rule.** Its three-seed mean macro-F1 at the 5% alarm target is
0.6240, against 0.6219 for compressed shared Adam (block128). It needs about
55.6 MB of modeled payload instead of 240 MB, and 200 client visits instead of
2,160 synchronizations. The class profile differs, however: DoS recall is 25
points lower at every seed.

This is exploratory evidence. The development rows had already scored two
earlier cyclic attempts (v1 and v2), so this study is development-adaptive in
motivation, even though stages 1–2 selected only on a fit-row holdout.
Confirmation, original validation and test rows were not opened.

## Design

The [plan](component-plan.json), including its two recorded amendments, split
the work into three stages:

1. **Base learner, centrally.** Variants were trained on 80% of fit rows and
   scored on a hashed 20% holdout (99,440 rows; 123 Web-based and 66 Brute
   Force rows). The threshold was calibrated to 5% benign false alerts on the
   holdout itself; this is a selection metric only.
2. **Cyclic scheme.** Three base-learner configurations, four schemes, seeds
   7 and 17, each client training on its own 80% of rows.
3. **Final check.** The selected combination was trained on all fit rows at
   seeds 7, 17 and 27, then scored once on the frozen development
   calibration/audit split.

Training runs in a numpy/xgboost-only process, because importing torch
alongside xgboost segfaults on this macOS build. Scoring reuses the existing
alarm code in a separate process.

## Stage 1: the repository defaults were already near-optimal

| Variant | Holdout macro-F1 | DoS | Web | Brute | Model |
| --- | --- | --- | --- | --- | --- |
| base (depth 6, eta 0.05, 200 rounds, sqrt weights) | 0.6983 | 0.746 | 0.163 | 0.273 | 3.91 MB |
| 400 rounds | 0.6994 | 0.743 | 0.163 | 0.258 | 8.11 MB |
| depth 8 / eta 0.1 / min_child_weight 5 | 0.6949–0.6983 | ~0.745 | 0.14–0.15 | 0.273 | 3.6–7.1 MB |
| multi_output_tree (one tree for all classes) | 0.6882 | 0.743 | 0.138 | 0.258 | 1.72 MB |
| balanced class weights | 0.6549 | 0.901 | 0.439 | 0.409 | 3.98 MB |
| class weights n^-0.75 | 0.6864 | 0.844 | 0.293 | 0.318 | 3.95 MB |

Class weighting is the main lever for DoS and rare-class recall. Balanced
weights raise them sharply but cost precision and macro-F1.

## Stage 2: scheme tweaks barely matter; federation costs ~6 points

On the holdout, cyclic training scores about 0.640 against 0.698 centrally.
A learning rate decaying across passes (0.6397), the earlier one-round-per-visit
scheme (0.6391) and 0.8 row subsampling (0.6390) are effectively tied. Visiting
clients in proportion to their size makes DoS collapse worse (0.24–0.45 recall),
which suggests that large DDoS-heavy clients dominate the boosting. The plan's
rule selected base + eta_decay.

## Stage 3: development result

| Seed | Cyclic XGBoost | Block128 | Difference | Actual FAR | Payload |
| --- | --- | --- | --- | --- | --- |
| 7 | 0.6264 | 0.6241 | +0.23 pt | 5.43% | 56.2 MB |
| 17 | 0.6260 | 0.6164 | +0.96 pt | 5.45% | 55.4 MB |
| 27 | 0.6197 | 0.6252 | −0.55 pt | 5.42% | 55.3 MB |
| **Mean** | **0.6240** | **0.6219** | +0.21 pt | | |

The rule required every seed within one point of block128, FAR within 0.5
points, payload at most FedAvg's 88 MB and a mean at least block128's. All
conditions pass. The serialized model is about 2.8 MB, against 20 KB for the
neural network.

Mean class recall at 5% (%):

| | Benign | DDoS | DoS | Recon | Web | Brute | Spoof | Mirai |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cyclic XGBoost | 94.6 | 94.1 | 46.6 | 51.9 | 6.0 | 14.4 | 66.6 | 99.1 |
| Block128 | 94.7 | 83.7 | 72.2 | 47.1 | 9.5 | 19.2 | 56.3 | 99.5 |

Macro-F1 is similar for different reasons. Trees keep more DDoS, Spoofing and
Recon, but lose DoS (flagged at every seed: −23 to −30 points) and some rare
recall. At 1% FAR block128 is ahead (0.5781 vs 0.5585); at 10% and 20% the
trees lead (0.6449 vs 0.6367, 0.6513 vs 0.6404).

## What this does and does not establish

- **Cost:** comparable aggregate quality at about 23% of block128 payload,
  63% of FedAvg's, with 200 visits. Visits are sequential, so wall-clock
  latency scales with client count; this is not measured.
- **Not a strict improvement:** DoS and rare-class detection are worse, and
  the inference model is about 140 times larger than the neural model.
- **Evidence strength:** one partition and development-adaptive history. A
  confirmation claim needs a newly reserved population.

Reproduce: `python -m src.eval.federated_xgboost_components {central,cyclic,final,score}`
(see the docstring); stage-3 command:
`... final --config base --scheme eta_decay`, then `... score`.
