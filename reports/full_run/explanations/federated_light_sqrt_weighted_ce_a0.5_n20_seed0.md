# Explanation audit: federated_light_sqrt_weighted_ce_a0.5_n20_seed0

Exploratory validation evidence only. These are class-logit attributions, not probabilities.

| True class | Correct / sampled | Leading features for this class's logit |
| --- | ---: | --- |
| Benign | 15 / 16 | Weight, IAT, Duration, Number, rst_count |
| DDoS | 16 / 16 | psh_flag_number, Magnitue, ICMP, Weight, Number |
| DoS | 0 / 16 | Variance, Tot sum, syn_count, Min, Protocol Type |
| Recon | 6 / 16 | Variance, ack_flag_number, Protocol Type, IAT, Duration |
| Web-based | 0 / 16 | Weight, IAT, ack_count, Protocol Type, Variance |
| Brute Force | 3 / 16 | IAT, Variance, SSH, Number, Weight |
| Spoofing | 9 / 16 | IAT, Weight, Number, Variance, AVG |
| Mirai | 16 / 16 | Min, Protocol Type, Magnitue, syn_count, Variance |

## Policy interpretation

Use these feature groups to prioritize telemetry and analyst investigation. Inspect the corresponding flow counters alongside packet/application logs; compare flagged traffic with normal-service baselines. Do not turn a top-feature list into a deny rule: direction, thresholds, false-positive cost and stability have not been established. Rare-class misses require independent application/authentication monitoring.

Mean absolute additivity residual: 0.0840 logits; 95th percentile: 0.2221 logits.

Method: https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html
