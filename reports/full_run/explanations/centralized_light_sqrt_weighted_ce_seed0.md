# Explanation audit: centralized_light_sqrt_weighted_ce_seed0

Exploratory validation evidence only. These are class-logit attributions, not probabilities.

| True class | Correct / sampled | Leading features for this class's logit |
| --- | ---: | --- |
| Benign | 14 / 16 | rst_count, Weight, IAT, Number, Header_Length |
| DDoS | 10 / 16 | Max, psh_flag_number, Radius, Weight, Magnitue |
| DoS | 11 / 16 | Weight, IAT, Magnitue, syn_flag_number, Variance |
| Recon | 13 / 16 | IAT, Variance, Number, Protocol Type, Weight |
| Web-based | 7 / 16 | Weight, IAT, Number, Variance, ack_count |
| Brute Force | 3 / 16 | SSH, Weight, IAT, Number, Variance |
| Spoofing | 15 / 16 | Weight, Number, IAT, Header_Length, rst_count |
| Mirai | 16 / 16 | Protocol Type, Magnitue, Min, Header_Length, AVG |

## Policy interpretation

Use these feature groups to prioritize telemetry and analyst investigation. Inspect the corresponding flow counters alongside packet/application logs; compare flagged traffic with normal-service baselines. Do not turn a top-feature list into a deny rule: direction, thresholds, false-positive cost and stability have not been established. Rare-class misses require independent application/authentication monitoring.

Mean absolute additivity residual: 0.6236 logits; 95th percentile: 1.7204 logits.

Method: https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html
