# Bounded extensions: development comparison

All rows below use the same 199,530-row development audit, with thresholds fitted on the other development half. Values are three-seed means. Confirmation is shown separately. Results are exploratory, not a fresh external test.

| Family | Method | Macro-F1 ± seed SD | Actual FAR | Payload MB | Syncs/visits |
| --- | --- | --- | --- | --- | --- |
| primary | central_adam | 0.6414 ± 0.0053 | 5.30% | 0.00 | 0 |
| primary | fedavg_adam | 0.5112 ± 0.0041 | 5.40% | 88.06 | 108 |
| primary | fedprox001 | 0.5162 ± 0.0045 | 5.45% | 88.06 | 108 |
| primary | fedprox01 | 0.5334 ± 0.0030 | 5.44% | 88.06 | 108 |
| primary | central_sgd | 0.6011 ± 0.0039 | 5.31% | 0.00 | 0 |
| primary | fedavg_sgd | 0.4963 ± 0.0172 | 5.28% | 88.06 | 108 |
| primary | scaffold | 0.5112 ± 0.0160 | 5.22% | 176.12 | 108 |
| primary | central_mime_base | 0.6393 ± 0.0027 | 5.25% | 0.00 | 0 |
| primary | mime | 0.4996 ± 0.0298 | 5.28% | 58.71 | 24 |
| primary | shared_adam | 0.6236 ± 0.0024 | 5.37% | 1761.18 | 2160 |
| primary | block128 | 0.6219 ± 0.0048 | 5.33% | 240.26 | 2160 |
| neural_fedlc | fedavg_adam_control | 0.5112 ± 0.0041 | 5.40% | 88.06 | 108 |
| neural_fedlc | fedavg_adam_fedlc | 0.5223 ± 0.0071 | 5.53% | 88.06 | 108 |
| neural_fedlc | block128_control | 0.6219 ± 0.0048 | 5.33% | 240.26 | 2160 |
| neural_fedlc | block128_fedlc | 0.6202 ± 0.0051 | 5.35% | 240.26 | 2160 |
| ple | central_ple | 0.6554 ± 0.0025 | 5.29% | 0.00 | 0 |
| ple | compressed_ple | 0.6495 ± 0.0037 | 5.24% | 752.29 | 2160 |
| ple | compressed_raw_wide | 0.6198 ± 0.0027 | 5.44% | 750.73 | 2160 |
| trees | central_xgb | 0.6697 ± 0.0000 | 5.69% | 0.00 | 0 |
| trees | cyclic_xgb | 0.6240 ± 0.0038 | 5.43% | 542.95 | 200 |
| trees | cyclic_fedlc | 0.6334 ± 0.0037 | 5.42% | 529.44 | 200 |

Neural rows use 21.6M backward examples; larger PLE networks require more computation per example. Tree rows match 200 boosting rounds, not compute: central sees 100M row-visits versus 5M cyclic. Tree payload is full-model UBJSON uplink + downlink; historical cached-tree estimates (~54 MB for FedLC) are preserved in JSON but are not an implemented transport. No numbers here measure real network latency.

## Class recall

| Method | Benign | DDoS | DoS | Recon | Web-based | Brute Force | Spoofing | Mirai |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| central_adam | 94.7% | 86.1% | 67.8% | 49.3% | 13.8% | 22.8% | 66.8% | 99.6% |
| fedavg_adam | 94.6% | 99.1% | 20.5% | 42.4% | 0.4% | 19.7% | 21.0% | 99.0% |
| fedprox001 | 94.6% | 99.0% | 21.3% | 42.5% | 0.7% | 21.3% | 21.5% | 99.1% |
| fedprox01 | 94.6% | 97.8% | 27.6% | 44.0% | 0.9% | 19.7% | 25.6% | 99.2% |
| central_sgd | 94.7% | 86.7% | 65.1% | 45.3% | 3.2% | 22.1% | 49.9% | 99.4% |
| fedavg_sgd | 94.7% | 94.7% | 34.7% | 33.1% | 0.1% | 12.8% | 19.7% | 98.8% |
| scaffold | 94.8% | 82.1% | 64.8% | 31.3% | 0.0% | 12.6% | 23.5% | 98.8% |
| central_mime_base | 94.8% | 86.7% | 66.9% | 48.3% | 11.6% | 21.3% | 69.6% | 99.7% |
| mime | 94.7% | 82.2% | 62.1% | 29.1% | 0.4% | 17.9% | 14.6% | 98.9% |
| shared_adam | 94.6% | 82.8% | 73.8% | 46.1% | 10.0% | 20.5% | 59.5% | 99.5% |
| block128 | 94.7% | 83.7% | 72.2% | 47.1% | 9.5% | 19.2% | 56.3% | 99.5% |
| fedavg_adam_control | 94.6% | 99.1% | 20.5% | 42.4% | 0.4% | 19.7% | 21.0% | 99.0% |
| fedavg_adam_fedlc | 94.5% | 98.7% | 23.1% | 42.6% | 0.7% | 23.3% | 24.3% | 99.2% |
| block128_control | 94.7% | 83.7% | 72.2% | 47.1% | 9.5% | 19.2% | 56.3% | 99.5% |
| block128_fedlc | 94.7% | 78.2% | 80.5% | 46.7% | 13.2% | 22.6% | 54.3% | 99.5% |
| central_ple | 94.7% | 86.8% | 69.2% | 53.1% | 16.7% | 21.3% | 71.3% | 99.7% |
| compressed_ple | 94.8% | 84.4% | 73.5% | 53.3% | 8.4% | 21.3% | 70.6% | 99.6% |
| compressed_raw_wide | 94.6% | 87.0% | 65.8% | 47.2% | 10.0% | 19.0% | 55.9% | 99.6% |
| central_xgb | 94.3% | 80.3% | 77.8% | 58.2% | 11.9% | 19.2% | 79.6% | 99.6% |
| cyclic_xgb | 94.6% | 94.1% | 46.6% | 51.9% | 6.0% | 14.4% | 66.6% | 99.1% |
| cyclic_fedlc | 94.6% | 87.1% | 64.9% | 52.5% | 6.5% | 15.6% | 68.2% | 99.2% |

## Paired comparisons at the same seed

Differences are treatment minus comparator. FAR differences are percentage points, not percentages of the comparator. Tree/central and PLE/raw contrasts retain the work and capacity limitations above.

| Treatment vs comparator | Mean macro-F1 difference | Largest absolute FAR difference |
| --- | --- | --- |
| cyclic_fedlc vs cyclic_xgb | +0.0094 | 0.03 pp |
| compressed_ple vs block128 | +0.0276 | 0.34 pp |
| compressed_ple vs compressed_raw_wide | +0.0296 | 0.42 pp |
| central_ple vs compressed_ple | +0.0059 | 0.54 pp |
| central_xgb vs cyclic_fedlc | +0.0363 | 0.29 pp |

## Existing confirmation replay

These are replays of already exposed frozen models and thresholds, not additional model selection. New central PLE controls and neural FedLC are not evaluated on confirmation.

| Candidate | Macro-F1 | Actual FAR | Exact historical seeds |
| --- | --- | --- | --- |
| ple | 0.6603 | 5.31% | 3/3 |
| xgboost | 0.6456 | 5.42% | 3/3 |

## Neural FedLC predeclared assessment

```json
{
  "fedavg_adam": {
    "per_seed": [
      {
        "seed": 7,
        "macro_f1_delta": 0.005098000648667056,
        "far_delta": -0.0012228388674631052,
        "mean_target_recall_delta": 0.013840977037028251,
        "mean_target_f1_delta": 0.0013181714484776796,
        "safeguards": {
          "macro_f1": true,
          "far": true,
          "other_class_recall": true
        },
        "safeguards_pass": true
      },
      {
        "seed": 17,
        "macro_f1_delta": 0.017160600772483492,
        "far_delta": 0.0017872260370614956,
        "mean_target_recall_delta": 0.025414309526603212,
        "mean_target_f1_delta": 0.028072477676165553,
        "safeguards": {
          "macro_f1": true,
          "far": true,
          "other_class_recall": true
        },
        "safeguards_pass": true
      },
      {
        "seed": 27,
        "macro_f1_delta": 0.01115270376370181,
        "far_delta": 0.003292258489323685,
        "mean_target_recall_delta": 0.0334323870495396,
        "mean_target_f1_delta": 0.0198210145837965,
        "safeguards": {
          "macro_f1": true,
          "far": true,
          "other_class_recall": true
        },
        "safeguards_pass": true
      }
    ],
    "mean_target_recall_gain": 0.02422922453772369,
    "mean_target_f1_gain": 0.016403887902813243,
    "every_seed_passes_safeguards": true,
    "supports_fedlc": true
  },
  "block128": {
    "per_seed": [
      {
        "seed": 7,
        "macro_f1_delta": -0.005084128425017043,
        "far_delta": -0.0019753550935941444,
        "mean_target_recall_delta": 0.004633200221662377,
        "mean_target_f1_delta": 0.0017366309839973633,
        "safeguards": {
          "macro_f1": true,
          "far": true,
          "other_class_recall": false
        },
        "safeguards_pass": false
      },
      {
        "seed": 17,
        "macro_f1_delta": -0.0005709127825650473,
        "far_delta": -0.001410967923995865,
        "mean_target_recall_delta": 0.028273024385935056,
        "mean_target_f1_delta": 0.007547550213310883,
        "safeguards": {
          "macro_f1": true,
          "far": true,
          "other_class_recall": true
        },
        "safeguards_pass": true
      },
      {
        "seed": 27,
        "macro_f1_delta": 0.0004998048606714711,
        "far_delta": 0.0038566456589219644,
        "mean_target_recall_delta": 0.01810582399815781,
        "mean_target_f1_delta": 0.0037539685679323087,
        "safeguards": {
          "macro_f1": true,
          "far": true,
          "other_class_recall": true
        },
        "safeguards_pass": true
      }
    ],
    "mean_target_recall_gain": 0.017004016201918413,
    "mean_target_f1_gain": 0.004346049921746852,
    "every_seed_passes_safeguards": false,
    "supports_fedlc": false
  }
}
```

## Verification

```json
{
  "neural_control_bridges_exact": 6,
  "ple_model_and_threshold_bridges_exact": 3,
  "tree_model_and_logits_bridges_exact": 3,
  "fresh_runs": 30,
  "confirmation_endpoints_replayed": 6,
  "confusion_endpoints_audited": 276
}
```
