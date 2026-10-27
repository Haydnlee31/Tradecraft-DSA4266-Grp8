# Interpretation of the matched comparison

These conclusions describe the fixed-setting, 33-run supplement in
[results.md](results.md). Scores are three-seed mean macro-F1 at a 5%
development-calibration false-alert target. The realized confirmation false-alert
rates range from 5.18% to 5.37%; no threshold was refitted on confirmation.
The confirmation cohort was evaluated previously, so this is retrospective
evidence rather than another independent test.

## Main finding

Shared-gradient Adam achieves 0.6361 confirmation macro-F1, compared with 0.5187
for FedAvg-Adam and 0.6535 for centralized Adam. Its compressed version achieves
0.6325, reproducing the original result exactly on the original runtime. **0.6325
is macro-F1, not recall.** Compression loses 0.36 F1 percentage points relative
to shared-gradient Adam and remains 2.10 points below centralized Adam.

The strongest interpretation is that synchronized gradients with persistent
server optimizer state substantially narrow the observed centralized–federated
gap under this protocol. This comparison changes both synchronization frequency
and optimizer-state handling, so it does not isolate persistent Adam memory as
the sole cause. Nor does it establish a new optimizer or general superiority
over optimized implementations of all other federated methods.

The trade-off remains material. Shared-gradient variants use 2,160 synchronization
steps versus FedAvg's 108. Recorded compressed application payload is 240.26 MB,
an 86.36% reduction from uncompressed shared-gradient Adam's 1,761.18 MB, but
still approximately 2.73 times FedAvg's 88.06 MB. Compression counts serializer
metadata and initialization, while uncompressed accounting uses raw FP32
vectors; neither measures transport overhead or physical-network traffic.

## Updated method summaries for the thesis

- **FedAvg:** The reset-client-Adam baseline achieves 0.5187 macro-F1 with
  relatively low communication. It recognizes DDoS and Mirai well but has
  weak DoS, Spoofing and Web-based recall under the fixed non-IID partition.
- **FedProx:** The two fixed coefficients improve macro-F1 to 0.5241 (mu=0.01)
  and 0.5407 (mu=0.1). The latter is a 2.20-point improvement over FedAvg-Adam.
  Actual false alerts remain comparable after development-only calibration.
  The earlier claim of substantially higher false alerts belongs to the earlier
  experiment and operating point; it should not describe this matched table.
- **SCAFFOLD:** Plain-SGD SCAFFOLD reaches 0.5151 versus 0.5010 for matched
  FedAvg-SGD, improving in all three paired seeds. DoS recall rises from 34.85%
  to 64.87%, but DDoS recall falls from 94.67% to 82.15%; Web-based recall is
  zero. False alerts are comparable here. It remains below centralized SGD
  (0.6068). Its benefit must be judged against the SGD control, not attributed
  by comparing it solely with FedAvg-Adam.
- **Full MIME with an Adam-like base:** It reaches 0.5067 versus 0.6532 for its
  centralized base updater. It does not improve mean macro-F1 over FedAvg-Adam
  in this budget and has appreciable seed variation (sample SD 0.0272). Full
  anchor and paired correction gradients consume its equal backward budget,
  allowing 24 rounds and 48 logical communication barriers. This is a finding
  about the tested finite-work setting, not a claim that MIME cannot converge.
- **Shared-gradient Adam:** It substantially improves macro-F1 to 0.6361 and
  restores much of DoS and Spoofing recognition, at the cost of frequent
  synchronization and large messages. It remains a simulated federation with
  client-owned data and server-side gradient aggregation.
- **Compressed shared-gradient Adam:** The fixed block128, four-bit,
  bidirectional residual-feedback implementation preserves most of that gain
  at 0.6325 while reducing message size. Earlier unsuccessful codecs explain
  its development history, but are not additional arms in this matched table.

## Rare attacks are still a limitation

Compressed shared-gradient Adam obtains 9.87% Web-based recall and 26.28% Brute
Force recall, compared with centralized Adam's 15.60% and 30.00%. It improves
several categories over FedAvg but does not solve rare-class recognition.
The study therefore supports a quality–communication trade-off, not deployment
readiness or reliable detection of every attack category. The confirmation
cohort contains only 500 Web-based and 260 Brute Force examples; three training
seeds do not create additional independent attack observations.

## What the thesis can and cannot claim

Use the [methodology draft](methodology-draft.md) for cohort construction,
architecture, optimizers, work accounting, threshold calibration and hardware.
All methods use the same fit rows, features, objective, partition and backward-
example budget. They do not have equal communication, optimizer calls, FLOPs,
wall time or hyperparameter-search effort. Report both mean and sample SD,
retain all fixed configurations and avoid selecting only the favorable seed.

This supplement ran on an Apple M3 **CPU**, not its GPU or a cloud Tesla T4.
Historical cloud experiments should have a separate hardware paragraph.
Similar classification metrics across environments cannot establish equivalent
runtime, energy use or numerical trajectories. No cloud rerun is needed to
describe these actual experiments truthfully.

The original four-arm result remains the historical confirmation evidence.
This broader table adds common-protocol context, but reusing already exposed
confirmation rows cannot establish a fresh generalization claim. Do not present
these optimizer results as the separate centralized-heavy comparison, or as
evidence for unseen devices, capture sessions or a physically deployed network.

Before submission, integrate this material with the thesis's earlier baseline
section and make the experimental phases explicit. Further model searching is
not required to support the conclusions above. A new hardware-performance or
external-generalization claim would require a separately designed experiment.
