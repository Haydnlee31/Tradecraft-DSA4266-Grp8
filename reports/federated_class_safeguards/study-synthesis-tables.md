# Official39 class failures and costs

Generated from existing audited counts. Three training seeds; no new inference or training.

## Primary final test results

| Setting | Macro-F1 mean and SD | False alerts per 10000 benign | Training examples | Optimizer steps | GPU train plus validation seconds mean | FP32 model bytes | Estimated model exchange bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized light | 0.656732 ± 0.000911 | 1749.3 | 40,000,000 | 78,140 | 299.53 | 20,384 | 0 |
| IID federated | 0.611692 ± 0.003491 | 1908.2 | 40,000,000 | 78,400 | 310.30 | 20,384 | 16,307,200 |
| Controlled non-IID | 0.404938 ± 0.003283 | 25.0 | 40,000,000 | 78,340 | 310.72 | 20,384 | 16,307,200 |

Exchange counts both directions over 20 full-participation rounds; excludes protocol and data collection costs. Central model exchange is zero, NOT zero data-transfer cost.

## Every attack category and its two kinds of mistake

Percentages are equal-weight means across seeds; the three outcome columns sum to 100 for each class, up to rounding.

| Setting | True attack | Correct category | Predicted benign | Wrong attack category |
| --- | --- | ---: | ---: | ---: |
| Centralized light | DDoS | 84.799% | 0.001% | 15.200% |
| Centralized light | DoS | 70.738% | 0.001% | 29.261% |
| Centralized light | Recon | 73.925% | 21.945% | 4.129% |
| Centralized light | Web-based | 8.002% | 30.762% | 61.236% |
| Centralized light | Brute Force | 26.285% | 33.333% | 40.382% |
| Centralized light | Spoofing | 71.037% | 19.605% | 9.358% |
| Centralized light | Mirai | 99.543% | 0.001% | 0.456% |
| IID federated | DDoS | 84.697% | 0.002% | 15.301% |
| IID federated | DoS | 67.230% | 0.003% | 32.767% |
| IID federated | Recon | 72.731% | 21.772% | 5.497% |
| IID federated | Web-based | 0.752% | 29.285% | 69.963% |
| IID federated | Brute Force | 25.344% | 33.511% | 41.145% |
| IID federated | Spoofing | 57.423% | 27.850% | 14.727% |
| IID federated | Mirai | 99.408% | 0.002% | 0.590% |
| Controlled non-IID | DDoS | 98.768% | 0.009% | 1.223% |
| Controlled non-IID | DoS | 21.013% | 0.025% | 78.962% |
| Controlled non-IID | Recon | 19.825% | 79.178% | 0.997% |
| Controlled non-IID | Web-based | 0.000% | 97.552% | 2.448% |
| Controlled non-IID | Brute Force | 0.000% | 96.921% | 3.079% |
| Controlled non-IID | Spoofing | 0.925% | 98.288% | 0.787% |
| Controlled non-IID | Mirai | 98.806% | 0.079% | 1.115% |

## All class precision recall and F1

| Setting | Class | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| Centralized light | Benign | 78.215% | 82.507% | 0.802863 |
| Centralized light | DDoS | 90.538% | 84.799% | 0.875346 |
| Centralized light | DoS | 58.850% | 70.738% | 0.640502 |
| Centralized light | Recon | 70.980% | 73.925% | 0.723269 |
| Centralized light | Web-based | 46.284% | 8.002% | 0.135298 |
| Centralized light | Brute Force | 40.566% | 26.285% | 0.318203 |
| Centralized light | Spoofing | 82.335% | 71.037% | 0.762065 |
| Centralized light | Mirai | 99.720% | 99.543% | 0.996309 |
| IID federated | Benign | 75.447% | 80.918% | 0.780854 |
| IID federated | DDoS | 89.491% | 84.697% | 0.870235 |
| IID federated | DoS | 57.260% | 67.230% | 0.618212 |
| IID federated | Recon | 66.949% | 72.731% | 0.697173 |
| IID federated | Web-based | 13.002% | 0.752% | 0.014223 |
| IID federated | Brute Force | 29.480% | 25.344% | 0.271241 |
| IID federated | Spoofing | 74.042% | 57.423% | 0.646679 |
| IID federated | Mirai | 99.576% | 99.408% | 0.994918 |
| Controlled non-IID | Benign | 51.418% | 99.750% | 0.678570 |
| Controlled non-IID | DDoS | 80.319% | 98.768% | 0.885917 |
| Controlled non-IID | DoS | 84.217% | 21.013% | 0.335242 |
| Controlled non-IID | Recon | 95.032% | 19.825% | 0.327687 |
| Controlled non-IID | Web-based | 0.000% | 0.000% | 0.000000 |
| Controlled non-IID | Brute Force | 0.000% | 0.000% | 0.000000 |
| Controlled non-IID | Spoofing | 67.880% | 0.925% | 0.018114 |
| Controlled non-IID | Mirai | 99.995% | 98.806% | 0.993972 |

## Which classes account for the score gap?

Arithmetic contributions in macro-F1 percentage points, not causal effects. Each column sums to its total gap; negative contributions favor the second setting.

| Class | Central minus IID | IID minus non-IID | Central minus non-IID |
| --- | ---: | ---: | ---: |
| Benign | 0.275 | 1.279 | 1.554 |
| DDoS | 0.064 | -0.196 | -0.132 |
| DoS | 0.279 | 3.537 | 3.816 |
| Recon | 0.326 | 4.619 | 4.945 |
| Web-based | 1.513 | 0.178 | 1.691 |
| Brute Force | 0.587 | 3.391 | 3.978 |
| Spoofing | 1.442 | 7.857 | 9.299 |
| Mirai | 0.017 | 0.012 | 0.029 |
| Total | 4.504 | 20.675 | 25.179 |

## Later SGD comparison is training only

Do NOT rank these against the final-test table above. These are one-seed frozen warm-start endpoints.

| Arm | Argmax training macro-F1 | Training false alerts | 5 percent target audit macro-F1 | Achieved audit false alerts | Training examples | Optimizer calls | Estimated exchange bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| central_sgd | 0.606799 | 18.383% | 0.579812 | 4.991% | 5,120,000 | 10,000 | 0 |
| fedavg_sgd | 0.571436 | 15.103% | 0.541743 | 4.999% | 5,120,000 | 10,000 | 8,153,600 |
| scaffold | 0.593971 | 24.447% | 0.551147 | 4.972% | 5,120,000 | 10,000 | 16,307,200 |

The audit rows were held out from threshold fitting only, NOT from model fitting. All models remain research-only.
