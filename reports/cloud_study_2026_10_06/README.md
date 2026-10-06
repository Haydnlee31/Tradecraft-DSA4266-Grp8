# Cloud FedAvg study and FedProx handoff — 2026-10-06

## Outcome

The cloud pipeline is operational on the user's RONIN G4dn/T4 environment, but
none of the matched light-model configurations is operationally ready. Non-IID
FedAvg repeatedly misses rare attacks across training seeds and client partitions.
Preserve these results as the baseline; the next optional extension is FedProx,
not a larger model or a replacement of the original study.

The [auditor](audit.py) verified **19 completed runs in seven supplied archives**:
four two-step recovery pilots, two 60-step references, two tuning runs, one loss
screen, two matched controls, six model-seed confirmations and two partition checks.
It checks source/data provenance against the pre-cloud snapshot, archive and
checkpoint hashes, saved best-state tensors, history/selection consistency, all
per-class recalls, confusion-derived F1/accuracy/false positives, and null test
metrics. [Machine-readable evidence](results/summary.json) contains every run's
metrics, curves, environment and archive SHA-256. This is an artifact audit, not
independent inference on the checkpoint. Scalers are hashed, not deserialized.

Reproduce from the repository root with the original private archives:

```bash
python -m reports.cloud_study_2026_10_06.audit \
  --archive-dir /path/to/archives --output outputs/cloud-audit-recheck
```

Raw datasets, checkpoint/scaler files and backup archives are intentionally not
committed. A clone alone cannot reproduce training or inference without the private
sample files. The user verified the pilot archive persisted through a cloud
stop/start, with the same SHA-256 as the local copy. GPU execution and exact
pause/resume parity were measured on the original FedAvg code, not yet on FedProx.

## Matched 60-step model-seed comparison

All use the light 64–32 MLP, LayerNorm, dropout 0.2, square-root-weighted CE,
Adam LR 0.001, batch 512, weight decay 1e-5. Federated runs use full participation
of 20 simulated clients, one local epoch and freshly reset client Adam each round.
Partition seed is 0; non-IID alpha is 0.5. All use train/validation only.

| Validation metric | Central light | IID light | Non-IID light |
| --- | ---: | ---: | ---: |
| Macro-F1, seeds 7/8/9 mean ± sample SD | 0.6872 ± 0.0128 | 0.6345 ± 0.0045 | 0.5559 ± 0.0063 |
| Macro-F1, new seeds 8/9 only | 0.6865 | 0.6324 | 0.5576 |
| Benign false positives, three-seed mean | 20.99% | 25.50% | 4.27% |
| Benign recall | 79.01% | 74.50% | 95.73% |
| DDoS recall | 91.85% | 78.85% | 99.18% |
| DoS recall | 83.69% | 77.93% | 15.62% |
| Recon recall | 75.38% | 75.82% | 49.77% |
| Web recall | 30.89% | 14.23% | 4.25% |
| Brute Force recall | 19.61% | 15.04% | 14.80% |
| Spoofing recall | 76.57% | 72.74% | 47.83% |
| Mirai recall | 99.44% | 99.28% | 99.15% |

The ordering is consistent across seeds. Seed 7 informed the confirmation design,
so report new seeds separately. Sample SD is not a confidence interval. Equal
epoch/round counts are not equal optimization trajectories. IID models select
rounds 59/59/60, so this is a fixed-budget comparison, not proof of convergence.
The central model's 19.61% mean Brute Force recall is below the existing 20%
research gate; rounding cannot turn it into a pass. No gate was relaxed.

## Cloud screens (seed 7, partition 0)

| Experiment | Macro-F1 | Web recall | Brute Force recall | False positives | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Heavy reference, BatchNorm/dropout 0.3 | 0.6686 | 37.58% | 23.83% | 21.44% | Retain reference |
| Heavy LR 0.0003 | 0.6647 | 32.17% | 22.38% | 21.38% | Neither promotion gate |
| Non-IID reference, 60 rounds | 0.5525 | 3.18% | 14.80% | 3.56% | Retain reference |
| Non-IID 120 rounds | 0.5769 | 4.94% | 15.16% | 4.22% | Neither promotion gate |
| Non-IID inverse-frequency CE, 60 rounds | 0.5567 | 28.98% | 29.24% | 3.40% | Neither promotion gate |

Longer non-IID training improves DoS and Spoofing recall but reduces DDoS recall
by 2.77 percentage points, beyond the two-point guardrail. Its first 60 rounds
matched the reference in the earlier paired archive review. Stronger loss weighting
raises rare-class recall, but Web/Brute Force precision is only 11.97%/7.26%; it
also loses DDoS, Recon and Spoofing recall. These are useful trade-offs, not promoted
defaults. Central-light seed 7 beats heavy seed 7 on macro-F1, but differing
normalization/dropout and limited matched heavy replication prevent a pure capacity claim.

## Partition robustness (model seed 7, 60 rounds)

| Metric | Partition 0 | Partition 1 | Partition 2 |
| --- | ---: | ---: | ---: |
| Macro-F1 | 0.5525 | 0.5448 | 0.5427 |
| Benign false positives | 3.56% | 2.62% | 5.37% |
| DoS recall | 16.55% | 24.44% | 30.46% |
| Web recall | 3.18% | 0.16% | 9.87% |
| Brute Force recall | 14.80% | 14.80% | 0.00% |
| Web classified benign | 72.29% | 79.46% | 70.06% |
| Brute Force classified benign | 70.76% | 75.09% | 61.73% |

All eight class recalls are retained in the evidence JSON. Partition 1 detects
one of 628 Web rows; partition 2 makes no Brute Force predictions. No easiest
partition is selected as a benchmark. In partition 0, clients with no Web samples
carry 35.67% of sample-count aggregation weight; clients with no Brute Force samples
carry 41.46%. This supports a heterogeneity hypothesis, not proof that aggregation
alone causes the failure. Partition 2 selects round 60 and still improves; the
three partitions are not an exhaustive or fully converged robustness study.

## Compute and scope

Original cloud revision: `3ef7202f59ba4e7a468342b9a1ad65d22a343630`.
Cloud Python 3.12, PyTorch 2.7.0+cu128, Flower 1.39.0, Tesla T4; manifests retain
the exact recorded versions. CUBLAS_WORKSPACE_CONFIG was recorded as `:4096:8`.
Typical 60-step training plus full validation took about 8–10 minutes. This excludes
setup, checkpoint IO, transfer, idle machine time and provider billing overhead.
Peak allocated PyTorch GPU memory is not total GPU memory. No causal CPU/GPU
speedup or Jetson edge measurements are claimed. Monetary costs have not been
reconciled against a provider invoice.

All runs use the same sampled partial Kaggle release and validation split. The
historical test split has been inspected in older work; it cannot become fresh
confirmation by relabeling it. Current cloud tuning did not evaluate it. Federation
remains single-machine simulation with pooled preprocessing/global class weights;
no physical distribution, privacy guarantee or per-device vulnerability follows.

## Next: optional, bounded FedProx extension

[Local verification](local_verification.json): all 46 unit tests and dependency
checks passed. The actual-data two-round CPU pilot passed mu=0 equivalence and
positive-mu recovery. Its default FedAvg final weights and validation metrics
also matched the saved pre-cloud CPU run exactly. The old Trainer implementation
was independently compared against the default new Trainer for three synthetic
epochs with dropout and matched exactly. None of these checks establishes CUDA
correctness of the new FedProx path or a model-quality gain.

See [FEDPROX.md](../../src/models/FEDPROX.md) and
[frozen screen plan](../../configs/cloud_fedprox_plan.json). The earlier local
12-run plan is not extended or overwritten. This is a new explicitly scoped plan:
one same-code FedAvg bridge plus two FedProx candidates, after correctness pilots.
No experiment was launched on the cloud by this handoff. Keep the GPU stopped
while local verification is performed. The original configured-model report and
all failed screening evidence remain unchanged.
