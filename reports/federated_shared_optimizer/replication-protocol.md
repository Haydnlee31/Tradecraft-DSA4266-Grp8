# Shared optimizer training seed replication

The alarm-matched discovery comparison put shared-gradient Adam about 2.10 macro-F1 points behind centralized-light Adam and 11.09 points above reset-client-Adam FedAvg. Two additional training seeds will check whether that pattern repeats before we invest in reducing communication.

## Fixed comparison

Run seeds 17 and 27 for centralized Adam, shared-gradient Adam and FedAvg. Keep the selected learning rates at 0.001, 0.003 and 0.003 respectively. Each run uses 21.6 million backward examples, giving 129.6 million new backward examples across six runs. Reuse the published seed 7 endpoints without retraining them. Seed 7 was used to select the settings and remains discovery evidence.

All models use the same 39 inputs, 64 and 32 hidden units, LayerNorm, fixed-denominator square-root weighted loss, 500,000 fitting rows and 20 historical virtual clients. Changing a seed changes initialization and minibatch draws, not client ownership. Within a seed, all three models start from identical weights; both federated methods see the same client sample multiset in a different update order. Centralized Adam samples the pooled fitting set.

The equal-example budget does not equalize optimizer updates or communication. Shared Adam performs 2,160 synchronizations, whereas FedAvg performs 108 rounds of 20 local updates. Their modeled payloads remain 1,761,177,600 and 88,058,880 bytes respectively. This is simulated federation, not measured network traffic or a privacy guarantee.

## Evaluation without another search

Score only the final endpoint. Reuse the exact existing grouped calibration/audit split of the development data. For each model, freeze thresholds using calibration benign rows at 1%, 5%, 10% and 20% false-alert targets. The primary comparison remains 5%. Report actual audit false-alert rates; flag paired rates differing by more than 0.5 percentage points instead of retuning to force equality.

Report each seed, paired macro-F1 differences, confusion matrices and all eight class recalls. Summarize the three available seeds using means, sample standard deviations and ranges, while showing the two new replications separately. Three seeds on the same development rows do not establish statistical significance, equivalence or generalization to new captures.

## Decision after the runs

A positive shared-versus-FedAvg macro-F1 difference in both new seeds at comparable actual false-alert rates would support a focused communication-reduction experiment. An inconsistent effect would require diagnosis before adding a new method. Report the centralized gap as observed; neither rounding nor this alarm-matched question changes the previously failed uncalibrated continuation gate. Rare-class failures remain failures even if overall macro-F1 improves.

Reserved confirmation arrays and the original validation and test splits stay closed. No cloud compute, new classifier, architecture search or human review is part of this stage. The machine-readable rules are in [replication-plan.json](replication-plan.json); the updater and alarm code remain unchanged.
