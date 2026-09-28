# Explanation audit: centralized_heavy_sqrt_weighted_ce_seed0

Exploratory validation evidence only. These are class-logit attributions, not probabilities.

| True class | Correct / sampled | Leading features for this class's logit |
| --- | ---: | --- |
| Benign | 13 / 16 | rst_count, Weight, Number, IAT, urg_count |
| DDoS | 12 / 16 | psh_flag_number, ICMP, Weight, Protocol Type, Max |
| DoS | 11 / 16 | Magnitue, Weight, fin_count, Number, Max |
| Recon | 15 / 16 | Variance, rst_count, Header_Length, fin_count, Magnitue |
| Web-based | 4 / 16 | Number, IAT, Weight, Variance, flow_duration |
| Brute Force | 5 / 16 | SSH, Weight, Number, IAT, syn_count |
| Spoofing | 13 / 16 | Weight, Number, IAT, Header_Length, Variance |
| Mirai | 16 / 16 | Protocol Type, Magnitue, TCP, Min, Radius |

## Policy interpretation

Use these feature groups to prioritize telemetry and analyst investigation. Inspect the corresponding flow counters alongside packet/application logs; compare flagged traffic with normal-service baselines. Do not turn a top-feature list into a deny rule: direction, thresholds, false-positive cost and stability have not been established. Rare-class misses require independent application/authentication monitoring.

Mean absolute additivity residual: 0.5628 logits; 95th percentile: 1.5886 logits.

Method: https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html
