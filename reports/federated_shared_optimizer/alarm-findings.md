# Shared gradient gains persist at comparable false alert rates

14 October 2026. **Shared-gradient Adam retains a large macro-F1 advantage over FedAvg after calibrating both to the same benign alarm target.** At the primary 5% calibration target, audit macro-F1 is 0.6246 for shared Adam, 0.5137 for FedAvg and 0.6456 for centralized-light Adam. The actual audit false-alert rates are 5.43%, 5.70% and 5.33%, respectively. Shared Adam therefore gains 11.09 macro-F1 percentage points over FedAvg while raising slightly fewer benign alerts in this comparison.

This is a useful direction, not a completed solution. Shared Adam remains 2.10 macro-F1 points below centralized-light, rare-class recall is poor, and its modeled training payload remains twenty times FedAvg's. MIME's advantage over FedAvg largely disappears when the alarm target is controlled. Focus subsequent work on the shared-gradient reference rather than expanding MIME or adding another classifier.

## The comparison rules

The [protocol](alarm-protocol.md), [plan](alarm-plan.json), inference-only implementation and synthetic tests were committed as `c16f0c6` before calculating alarm-target results. The five checkpoints and their learning rates were fixed from the previous experiment. No new weights, model fits, loss functions or learning rates were selected.

The existing development set was split by a label-independent hash of standardized input vectors into 200,022 calibration rows and 199,530 audit rows. Exact duplicate inputs stayed together: the two parts share zero input-fingerprint groups. Calibration contains 10,562 benign rows; audit contains 10,631. Thresholds used only the calibration benign scores, never audit labels or attack-class performance.

For each model, the score is its highest attack logit minus its benign logit. Predict the highest-scoring attack category only if this margin exceeds the threshold; otherwise predict Benign. The zero threshold reproduces the original argmax predictions exactly. This is one automatic decision offset, **not a new classifier, an abstention rule or a human-review process**. It changes the attack-versus-benign boundary but cannot change the ordering of attack categories.

Both parts of this development set previously contributed to learning-rate selection, and the rows originated from historical training data. Splitting them now separates threshold fitting from scoring, but does not make the audit an independent final holdout. Results remain exploratory, with one training seed and one client partition. Reserved confirmation, original validation and original test arrays stayed unopened.

## Results at the primary alarm target

| Fixed saved model | Audit macro-F1 | Actual audit false alerts | False alerts among 10,631 benign rows |
| --- | ---: | ---: | ---: |
| Centralized-light Adam | 0.6456 | 5.33% | 567 |
| Shared-gradient Adam | 0.6246 | 5.43% | 577 |
| FedAvg with reset Adam | 0.5137 | 5.70% | 606 |
| Central MIME base updater | 0.6367 | 5.18% | 551 |
| Full MIME | 0.5153 | 5.24% | 557 |

Every method allowed at most 528 false alerts among the 10,562 calibration benign rows, using the predeclared order-statistic threshold with boundary ties kept benign. Calibration FAR was 4.999%. **The audit rates are not exactly 5%, and all five exceed that target.** There was no second adjustment on audit labels. This illustrates why an empirical calibration allowance is not a deployment guarantee.

The shared-versus-central audit FAR difference is 0.094 percentage points, and the shared-versus-FedAvg difference is −0.273 points. Both satisfy the declared 0.5-point comparability tolerance. These are comparable realized alarm burdens, not mathematically identical false-alert rates.

At this operating point, shared Adam closes 84.1% of the FedAvg-to-centralized macro-F1 gap. Do not substitute the earlier 95.1% figure: that used full-development, default-argmax outcomes, whereas this comparison uses the audit subset and calibrated thresholds. The remaining 2.10-point gap also does not meet the earlier 2.00-point band by rounding. The original training continuation screen remains failed; this diagnostic does not rewrite it.

## The advantage is not limited to one target

| Calibration FAR target | Central Adam F1 | Shared Adam F1 | FedAvg F1 | Full MIME F1 |
| --- | ---: | ---: | ---: | ---: |
| 1% | 0.6125 | 0.5839 | 0.4729 | 0.4653 |
| 5% primary | 0.6456 | 0.6246 | 0.5137 | 0.5153 |
| 10% | 0.6606 | 0.6378 | 0.5289 | 0.5327 |
| 20% | 0.6582 | 0.6407 | 0.5446 | 0.5401 |

Shared Adam exceeds FedAvg by 9.60–11.10 macro-F1 percentage points across all four fixed targets. Their realized audit FARs meet the 0.5-point comparison tolerance at each target. This supports an advantage beyond simply setting a more aggressive default alarm boundary, though it does not isolate optimizer memory from synchronization frequency or other update differences.

One comparison must be flagged: at the 10% target, shared Adam has 10.22% actual FAR versus centralized Adam's 10.77%, a 0.555-point difference exceeding the declared tolerance. That pair is not treated as tightly alarm-matched. No target was changed or selected after inspecting this table; 5% remains primary. The [complete evidence](alarm-results.json) includes actual FARs, thresholds and every class metric for all five models at all four targets.

MIME differs from FedAvg by −0.76, +0.16, +0.38 and −0.46 macro-F1 percentage points at the four targets. It shows no consistent advantage here. Its former default-argmax advantage was therefore strongly dependent on comparing different alarm operating points. This does not prove that every MIME configuration fails; it argues against spending more effort on this particular configuration.

## Rare classes remain weak

Correct eight-class recall at the primary 5% calibration target is:

| Class | Central Adam | Shared Adam | FedAvg | Full MIME |
| --- | ---: | ---: | ---: | ---: |
| Benign | 94.67% | 94.57% | 94.30% | 94.76% |
| DDoS | 87.70% | 83.80% | 99.42% | 83.39% |
| DoS | 64.73% | 72.45% | 17.98% | 58.83% |
| Recon | 50.53% | 43.84% | 44.30% | 32.51% |
| Web-based | 11.85% | 9.63% | 0.74% | 0.00% |
| Brute Force | 24.62% | 18.46% | 19.23% | 17.69% |
| Spoofing | 68.58% | 63.54% | 23.16% | 21.01% |
| Mirai | 99.50% | 99.47% | 99.03% | 98.79% |

Shared Adam's largest gains over FedAvg are DoS and Spoofing. It does not win every class: DDoS recall falls, Recon is similar, and Brute Force is slightly worse. A higher aggregate F1 should not hide those regressions.

Shared Adam correctly identifies only 26 of 270 Web-based examples and 24 of 130 Brute Force examples. It flags 41.48% and 40.77% of those classes as some attack, but many receive the wrong category. Those broader detection rates are not substitutes for eight-class recall. A benign threshold cannot repair the relative scores among attack categories, so it cannot by itself solve this remaining weakness.

## What to do next

Keep the investigation narrow. The next useful experiment is a separately frozen robustness study of **centralized-light Adam, shared-gradient Adam and FedAvg**, using paired training seeds 17 and 27 with the already selected learning rates and the same work budgets. Repeat the same grouped calibration rule and primary 5% target without additional tuning. This would test whether the current operating-point advantage survives initialization changes; it would still not establish new-session generalization.

That follow-up would be a new robustness question, not a retroactive pass of the earlier training screen or authorization to score reserved confirmation. Keep MIME and adaptive-anchor experiments on hold. If the shared-gradient advantage repeats, investigate communication reduction next with a fixed quality/false-alert tolerance. Do not claim novelty from shared-gradient Adam itself; it is an established control that now provides a more credible starting point for an extension.

The twenty-fold modeled payload penalty remains unchanged: threshold adjustment affects prediction decisions, not the previous training traffic. At 5.43%, benign false alerts also remain substantial. No cloud computation is necessary for the next local checks, and no deployment-readiness claim follows from these results.

## Verification and artifacts

All five saved models exactly reproduce their earlier full-development default metrics. An independent sorted-order and direct-logit replay reproduces all twenty calibrated operating points and all five default audit outcomes. The diagnostic performs 1,997,760 forward-example evaluations, zero backward examples and zero new model fits; saved-logit replay requires no additional model inference.

The [public receipt](alarm-results.json) records plan, data, input-group, model-checkpoint and source hashes, class counts, confusion matrices, thresholds and comparison arithmetic. Row-level logits, masks and checkpoints remain in ignored local output folders. Original comments and unrelated user changes are preserved. No cloud machine was used and the branch remains local.
