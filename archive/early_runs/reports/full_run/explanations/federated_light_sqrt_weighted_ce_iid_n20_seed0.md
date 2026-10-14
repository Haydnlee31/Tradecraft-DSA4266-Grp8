# Explanation audit: federated_light_sqrt_weighted_ce_iid_n20_seed0

Exploratory validation evidence only. These are class-logit attributions, not probabilities.

| True class | Correct / sampled | Leading features for this class's logit |
| --- | ---: | --- |
| Benign | 12 / 16 | Weight, IAT, Number, rst_count, Variance |
| DDoS | 10 / 16 | psh_flag_number, Weight, ICMP, IAT, Protocol Type |
| DoS | 12 / 16 | Weight, IAT, TCP, Variance, syn_count |
| Recon | 13 / 16 | Variance, IAT, Weight, Number, fin_count |
| Web-based | 1 / 16 | Weight, IAT, Number, Variance, ack_count |
| Brute Force | 3 / 16 | SSH, IAT, Weight, Variance, Number |
| Spoofing | 13 / 16 | Weight, IAT, Number, Header_Length, syn_count |
| Mirai | 16 / 16 | Protocol Type, Min, Magnitue, TCP, Variance |

## Policy interpretation

Use these feature groups to prioritize telemetry and analyst investigation. Inspect the corresponding flow counters alongside packet/application logs; compare flagged traffic with normal-service baselines. Do not turn a top-feature list into a deny rule: direction, thresholds, false-positive cost and stability have not been established. Rare-class misses require independent application/authentication monitoring.

Mean absolute additivity residual: 0.2855 logits; 95th percentile: 0.7919 logits.

Method: https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html
