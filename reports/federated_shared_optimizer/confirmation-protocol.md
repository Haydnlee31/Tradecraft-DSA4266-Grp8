# Locked confirmation: preparation before the final comparison

14 October 2026. This stage freezes the comparison; **it does not score the reserved cohort**. The blockwise candidate passed the development safeguards for all three training seeds. The remaining question is whether that result carries to rows reserved for this extension, without any more model or threshold choices.

## What is fixed

Use the existing twelve endpoints: centralized-light Adam, uncompressed shared-gradient Adam, FedAvg with client Adam and blockwise four-bit shared Adam, each at seeds 7, 17 and 27. Keep the trained models, fit-only scaler, 39-feature input, eight-class mapping, client partition and historical training budgets unchanged. The compressed endpoint is the model actually delivered to clients, not the server's hidden full-precision target.

Each checkpoint has four existing development-calibrated thresholds: 1%, 5%, 10% and 20% benign false-alert targets. The primary comparison remains the **5% calibration target**. Freeze all 48 thresholds now, including their original benign sample counts and tie rules. On confirmation, the actual false-alert rates can differ from these targets; recalibration would answer a different question and is prohibited.

The [machine-readable plan](confirmation-plan.json) pins the audited parent evidence and records these rules. The [preparation runner](../../src/eval/shared_optimizer_confirmation_prepare.py) derives paths and identities from that evidence, hashes every checkpoint, and replays all twelve models on the previously used development data. Exact logits, default decisions and all 48 operating points must match. This checks deployment-state and threshold handling without new training or new selection.

## Data boundary and reconstruction

The manifest already records 398,502 reserved rows, including 500 Web-based and 260 Brute Force examples. Those counts are existing metadata, not newly observed model outcomes. This preparation reads the manifest and fit-only scaler, but does not open the reserved row-ID file, raw training shards, staging cache, original validation or test arrays.

The temporary raw-training staging cache has no recorded checksum. Do not use it as trusted confirmation input. After authorization, reconstruct the cohort from the existing checksum-verified nested training Parquet shards, using the frozen row IDs and original ordering. Apply the already fitted scaler and verify counts, labels, row identity and standardized input-group separation before inference. Stream bounded batches; there is no need to load the full official dataset.

These rows were reserved for this extension but belong to previously researched training data. They are not an independent external dataset or unseen-capture-session test. The original project test was evaluated earlier; it must not be reopened for this extension's tuning. State this history when interpreting the final results.

## How the final result will be judged

At the primary target, each seed must retain all five existing safeguards:

| Comparison | Fixed limit |
| --- | --- |
| Macro-F1 lost to same-seed uncompressed shared Adam | At most 0.01 |
| Absolute actual false-alert difference from shared Adam | At most 0.005, or 0.5 percentage points |
| Largest recall loss in any class from shared Adam | At most 0.05 |
| Macro-F1 gain over same-seed FedAvg | At least 0.08 |
| Serialized payload / historical uncompressed shared payload | At most 0.15 |

All three seeds must pass. An average, different threshold, selected seed or new checkpoint cannot rescue a failure. The centralized-light gap will be reported per seed and in aggregate; this is not a new equivalence test. Actual false-alert differences against centralized-light and FedAvg must also be reported. If they exceed 0.5 percentage points, label those comparisons as not alarm-matched even if the inherited compression screen passes.

Publish all endpoints, default argmax results and secondary targets, not just the best-looking table. Include all class recalls and rare-class correct counts. Distinguish detecting an attack from correctly naming its category. Seed variability is descriptive: the three models share one evaluation cohort and one client partition, so they are not three independent datasets.

Payload is the existing serialized application-byte accounting, not measured network traffic or edge latency. The candidate still synchronizes 2,160 times, versus FedAvg's 108 rounds. Compression does not remove that cost or establish privacy.

## Finish without another search loop

1. Freeze and verify the twelve endpoints locally, leaving confirmation closed. Review the readiness receipt.
2. After explicit authorization, reconstruct and evaluate the same reserved cohort once for all twelve endpoints using the frozen rules. No retraining or threshold refitting.
3. Write the final centralized-versus-federated trade-off conclusion, whether the safeguards pass or fail. If they fail, narrow the claim; do not tune on the outcome.

Preparation evaluates 4,794,624 development examples across the twelve models. The later confirmation inference budget is 4,782,024 examples. These are forward-only evaluations, not additional training. CPU inference with two threads is the frozen path; no new cloud session or larger GPU is required. Private arrays, logits and checkpoints remain outside Git; only code, rules, hashes and aggregate evidence belong in the repository.
