# Full-run measured results

Fixed sampled splits; three seeds per configuration. Maximum 30 epochs/rounds, patience 5, batch 512, Adam 0.001, weight decay 0.00001, sqrt-weighted CE. FL uses 20 clients, full participation, one local epoch and global train class weights. Checkpoint selection uses validation macro-F1 only.

FL runs use the actual client training helpers and Flower's sample-weighted aggregation, executed sequentially on this Mac. This measures learning, not distributed-system throughput, communication overhead, privacy or edge-hardware performance.

## Aggregate metrics

± denotes sample standard deviation across seeds (not a confidence interval).

| Configuration | Seeds | Validation macro-F1 | Test macro-F1 | Test accuracy | Parameters | Parameter KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| centralized-heavy | 3 | 0.6558 ± 0.0119 | 0.6580 ± 0.0126 | 0.8220 | 198,984 | 777.28 |
| centralized-light | 3 | 0.6458 ± 0.0034 | 0.6466 ± 0.0036 | 0.8078 | 5,544 | 21.66 |
| FL IID control | 3 | 0.5877 ± 0.0068 | 0.5927 ± 0.0066 | 0.7739 | 5,544 | 21.66 |
| FL Dirichlet α=0.5 | 3 | 0.4409 ± 0.0267 | 0.4412 ± 0.0304 | 0.7464 | 5,544 | 21.66 |

## Test recall by class

| Class | centralized-heavy | centralized-light | FL IID control | FL Dirichlet α=0.5 |
| --- | ---: | ---: | ---: | ---: |
| Benign | 0.737 ± 0.014 | 0.737 ± 0.005 | 0.648 ± 0.027 | 0.867 ± 0.119 |
| DDoS | 0.786 ± 0.006 | 0.760 ± 0.022 | 0.702 ± 0.004 | 0.951 ± 0.084 |
| DoS | 0.811 ± 0.010 | 0.807 ± 0.010 | 0.834 ± 0.007 | 0.080 ± 0.134 |
| Recon | 0.787 ± 0.011 | 0.728 ± 0.023 | 0.772 ± 0.030 | 0.357 ± 0.117 |
| Web-based | 0.254 ± 0.031 | 0.214 ± 0.063 | 0.015 ± 0.004 | 0.006 ± 0.011 |
| Brute Force | 0.309 ± 0.007 | 0.207 ± 0.014 | 0.192 ± 0.000 | 0.063 ± 0.109 |
| Spoofing | 0.756 ± 0.021 | 0.791 ± 0.024 | 0.665 ± 0.016 | 0.275 ± 0.143 |
| Mirai | 0.993 ± 0.000 | 0.993 ± 0.000 | 0.992 ± 0.000 | 0.987 ± 0.005 |

## Alert interpretation and confusions

Treating any non-Benign prediction as an alert is only a diagnostic view. An attack assigned the wrong attack category is still detected by that binary rule.

- centralized-heavy: attack-detection recall 99.42%; benign false-alert rate 26.30%. Web-based → Recon: 42.6%; Brute Force → Recon: 42.2%; Web-based → Spoofing: 28.2%; DDoS → DoS: 21.3%.
- centralized-light: attack-detection recall 99.36%; benign false-alert rate 26.34%. Web-based → Spoofing: 43.4%; Brute Force → Recon: 41.3%; Web-based → Recon: 31.9%; Brute Force → Spoofing: 26.0%.
- FL IID control: attack-detection recall 98.90%; benign false-alert rate 35.22%. Web-based → Recon: 58.0%; Brute Force → Recon: 49.0%; DDoS → DoS: 29.5%; Web-based → Spoofing: 26.3%.
- FL Dirichlet α=0.5: attack-detection recall 96.18%; benign false-alert rate 13.34%. DoS → DDoS: 91.6%; Web-based → Benign: 68.1%; Spoofing → Benign: 60.6%; Brute Force → Benign: 60.3%.

## Run completion

| Report | Selected epoch/round | Executed | Budget reached? |
| --- | ---: | ---: | --- |
| centralized_heavy_sqrt_weighted_ce_seed0.json | 13 | 18 | no |
| centralized_heavy_sqrt_weighted_ce_seed1.json | 14 | 19 | no |
| centralized_heavy_sqrt_weighted_ce_seed2.json | 29 | 30 | yes |
| centralized_light_sqrt_weighted_ce_seed0.json | 27 | 30 | yes |
| centralized_light_sqrt_weighted_ce_seed1.json | 18 | 23 | no |
| centralized_light_sqrt_weighted_ce_seed2.json | 17 | 22 | no |
| federated_light_sqrt_weighted_ce_a0.5_n20_seed0.json | 7 | 12 | no |
| federated_light_sqrt_weighted_ce_a0.5_n20_seed1.json | 30 | 30 | yes |
| federated_light_sqrt_weighted_ce_a0.5_n20_seed2.json | 2 | 7 | no |
| federated_light_sqrt_weighted_ce_iid_n20_seed0.json | 30 | 30 | yes |
| federated_light_sqrt_weighted_ce_iid_n20_seed1.json | 30 | 30 | yes |
| federated_light_sqrt_weighted_ce_iid_n20_seed2.json | 13 | 18 | no |

## Limits on interpretation

- This is the existing partial Kaggle mirror/sample, not all official CICIoT2023 traffic. Within-split duplicates and unknown upstream session/device relationships remain; an exact-overlap screen is not proof of independence.
- Only one loss and one non-IID alpha are confirmed here; do not claim a universal architecture winner. Training seeds also change Dirichlet assignments, so FL variance combines both sources.
- Heavy/light differ in width, depth AND dropout (0.3 versus 0.2). This compares configured model families, not a pure parameter-count intervention.
- Each FL client resets Adam each round. Centralized Adam retains moments. Equal row-pass budgets do not make their optimization trajectories identical.
- Global scaling, class counts and server validation are simulation conveniences; no privacy guarantee follows.
- Parameter size excludes buffers, activations, framework memory and serialization. No edge latency/power was measured.
- Decision thresholds are illustrative research gates; passing them is not deployment approval. Future architecture/loss changes should be selected on validation; these already-inspected test results are exploratory for subsequent iterations.
