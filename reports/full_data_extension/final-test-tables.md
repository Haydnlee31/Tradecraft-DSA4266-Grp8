# Official39 final test tables

Generated from the frozen evidence. Equal-weight means across training seeds 7, 17 and 27;
sample SD is training-seed variation, not uncertainty across independent datasets.
False alerts are the fraction of true benign flows predicted as attacks.

Full39 is the primary reference; both masks are sensitivity controls, not newly selected winners.

## Primary shared test population

Rows: 2,040,729.

| Setting | Arm | Macro-F1 mean and SD | Macro-F1 range | Mean false alerts |
| --- | --- | ---: | ---: | ---: |
| Centralized light | full39 | 0.656732 ± 0.000911 | 0.655880–0.657692 | 17.493% |
| Centralized light | number_masked | 0.652802 ± 0.004877 | 0.647179–0.655879 | 19.024% |
| Centralized light | number_total_masked | 0.658452 ± 0.002107 | 0.656019–0.659670 | 17.741% |
| IID federated | full39 | 0.611692 ± 0.003491 | 0.607670–0.613933 | 19.082% |
| IID federated | number_masked | 0.607025 ± 0.004033 | 0.603761–0.611533 | 18.865% |
| IID federated | number_total_masked | 0.603903 ± 0.003522 | 0.601364–0.607923 | 19.083% |
| Controlled non-IID | full39 | 0.404938 ± 0.003283 | 0.401499–0.408037 | 0.250% |
| Controlled non-IID | number_masked | 0.405026 ± 0.002413 | 0.402243–0.406525 | 0.170% |
| Controlled non-IID | number_total_masked | 0.407403 ± 0.002214 | 0.404856–0.408867 | 0.408% |

### Class recall for full39

| Class | Support per model | Centralized light | IID federated | Controlled non-IID |
| --- | ---: | ---: | ---: | ---: |
| Benign | 109,133 | 82.507% | 80.918% | 99.750% |
| DDoS | 1,203,157 | 84.799% | 84.697% | 98.768% |
| DoS | 364,574 | 70.738% | 67.230% | 21.013% |
| Recon | 68,363 | 73.925% | 72.731% | 19.825% |
| Web-based | 2,437 | 8.002% | 0.752% | 0.000% |
| Brute Force | 1,310 | 26.285% | 25.344% | 0.000% |
| Spoofing | 45,471 | 71.037% | 57.423% | 0.925% |
| Mirai | 246,284 | 99.543% | 99.408% | 98.806% |

### Class recall for number masked

| Class | Support per model | Centralized light | IID federated | Controlled non-IID |
| --- | ---: | ---: | ---: | ---: |
| Benign | 109,133 | 80.976% | 81.135% | 99.830% |
| DDoS | 1,203,157 | 83.940% | 83.894% | 98.406% |
| DoS | 364,574 | 72.574% | 68.973% | 24.221% |
| Recon | 68,363 | 75.133% | 71.964% | 18.437% |
| Web-based | 2,437 | 7.482% | 0.000% | 0.000% |
| Brute Force | 1,310 | 27.354% | 25.420% | 0.000% |
| Spoofing | 45,471 | 69.772% | 56.824% | 0.001% |
| Mirai | 246,284 | 99.516% | 99.372% | 98.586% |

### Class recall for number total masked

| Class | Support per model | Centralized light | IID federated | Controlled non-IID |
| --- | ---: | ---: | ---: | ---: |
| Benign | 109,133 | 82.259% | 80.917% | 99.592% |
| DDoS | 1,203,157 | 84.776% | 83.735% | 98.627% |
| DoS | 364,574 | 70.668% | 68.968% | 21.962% |
| Recon | 68,363 | 74.417% | 71.984% | 18.743% |
| Web-based | 2,437 | 7.181% | 0.000% | 0.000% |
| Brute Force | 1,310 | 27.379% | 25.522% | 0.000% |
| Spoofing | 45,471 | 74.779% | 55.411% | 1.830% |
| Mirai | 246,284 | 99.517% | 99.291% | 98.908% |

### Paired mask differences

Masked minus same-seed full39. All differences are percentage points, not relative percentages.

| Setting and mask | F1 seed 7 | F1 seed 17 | F1 seed 27 | Mean F1 change | Mean false-alert change |
| --- | ---: | ---: | ---: | ---: | ---: |
| light/number_masked-minus-full39 | -0.944 | -0.053 | -0.181 | -0.393 | +1.531 |
| light/number_total_masked-minus-full39 | -0.060 | +0.379 | +0.198 | +0.172 | +0.248 |
| iid/number_masked-minus-full39 | -0.194 | -0.815 | -0.391 | -0.467 | -0.217 |
| iid/number_total_masked-minus-full39 | -0.555 | -1.257 | -0.525 | -0.779 | +0.001 |
| dirichlet/number_masked-minus-full39 | +0.074 | +0.103 | -0.151 | +0.009 | -0.080 |
| dirichlet/number_total_masked-minus-full39 | +0.737 | -0.042 | +0.045 | +0.247 | +0.158 |

## Secondary original test population

Rows: 2,060,864.

| Setting | Arm | Macro-F1 mean and SD | Macro-F1 range | Mean false alerts |
| --- | --- | ---: | ---: | ---: |
| Centralized light | full39 | 0.656712 ± 0.000905 | 0.655866–0.657666 | 17.492% |
| Centralized light | number_masked | 0.652801 ± 0.004892 | 0.647162–0.655908 | 19.023% |
| Centralized light | number_total_masked | 0.658447 ± 0.002121 | 0.655998–0.659681 | 17.740% |
| IID federated | full39 | 0.611765 ± 0.003493 | 0.607741–0.614011 | 19.081% |
| IID federated | number_masked | 0.607094 ± 0.004033 | 0.603832–0.611604 | 18.864% |
| IID federated | number_total_masked | 0.603976 ± 0.003532 | 0.601436–0.608010 | 19.082% |
| Controlled non-IID | full39 | 0.403901 ± 0.003245 | 0.400526–0.406998 | 0.250% |
| Controlled non-IID | number_masked | 0.403942 ± 0.002395 | 0.401181–0.405442 | 0.170% |
| Controlled non-IID | number_total_masked | 0.406347 ± 0.002213 | 0.403802–0.407813 | 0.408% |

### Class recall for full39

| Class | Support per model | Centralized light | IID federated | Controlled non-IID |
| --- | ---: | ---: | ---: | ---: |
| Benign | 109,137 | 82.508% | 80.919% | 99.750% |
| DDoS | 1,213,933 | 84.526% | 84.450% | 98.779% |
| DoS | 373,680 | 71.046% | 67.579% | 20.525% |
| Recon | 68,364 | 73.924% | 72.730% | 19.825% |
| Web-based | 2,437 | 8.002% | 0.752% | 0.000% |
| Brute Force | 1,310 | 26.285% | 25.344% | 0.000% |
| Spoofing | 45,485 | 71.046% | 57.436% | 0.925% |
| Mirai | 246,518 | 99.543% | 99.409% | 98.807% |

### Class recall for number masked

| Class | Support per model | Centralized light | IID federated | Controlled non-IID |
| --- | ---: | ---: | ---: | ---: |
| Benign | 109,137 | 80.977% | 81.136% | 99.830% |
| DDoS | 1,213,933 | 83.652% | 83.631% | 98.419% |
| DoS | 373,680 | 72.914% | 69.332% | 23.668% |
| Recon | 68,364 | 75.132% | 71.963% | 18.437% |
| Web-based | 2,437 | 7.482% | 0.000% | 0.000% |
| Brute Force | 1,310 | 27.354% | 25.420% | 0.000% |
| Spoofing | 45,485 | 69.781% | 56.838% | 0.001% |
| Mirai | 246,518 | 99.517% | 99.373% | 98.587% |

### Class recall for number total masked

| Class | Support per model | Centralized light | IID federated | Controlled non-IID |
| --- | ---: | ---: | ---: | ---: |
| Benign | 109,137 | 82.260% | 80.918% | 99.592% |
| DDoS | 1,213,933 | 84.501% | 83.471% | 98.639% |
| DoS | 373,680 | 70.996% | 69.331% | 21.451% |
| Recon | 68,364 | 74.416% | 71.983% | 18.743% |
| Web-based | 2,437 | 7.181% | 0.000% | 0.000% |
| Brute Force | 1,310 | 27.379% | 25.522% | 0.000% |
| Spoofing | 45,485 | 74.787% | 55.421% | 1.830% |
| Mirai | 246,518 | 99.518% | 99.291% | 98.909% |

### Paired mask differences

Masked minus same-seed full39. All differences are percentage points, not relative percentages.

| Setting and mask | F1 seed 7 | F1 seed 17 | F1 seed 27 | Mean F1 change | Mean false-alert change |
| --- | ---: | ---: | ---: | ---: | ---: |
| light/number_masked-minus-full39 | -0.944 | -0.053 | -0.176 | -0.391 | +1.530 |
| light/number_total_masked-minus-full39 | -0.061 | +0.380 | +0.201 | +0.174 | +0.248 |
| iid/number_masked-minus-full39 | -0.194 | -0.817 | -0.391 | -0.467 | -0.217 |
| iid/number_total_masked-minus-full39 | -0.553 | -1.257 | -0.526 | -0.779 | +0.001 |
| dirichlet/number_masked-minus-full39 | +0.065 | +0.103 | -0.156 | +0.004 | -0.080 |
| dirichlet/number_total_masked-minus-full39 | +0.729 | -0.038 | +0.043 | +0.245 | +0.158 |

## Per-seed full39 primary attack errors

Counts are not averaged or pooled across seeds. Each row partitions the true attack class.

| Setting | Seed | True class | Support | Correct category | Predicted benign | Wrong attack category |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Centralized light | 7 | DDoS | 1,203,157 | 988,364 | 9 | 214,784 |
| Centralized light | 7 | DoS | 364,574 | 276,247 | 4 | 88,323 |
| Centralized light | 7 | Recon | 68,363 | 49,892 | 15,471 | 3,000 |
| Centralized light | 7 | Web-based | 2,437 | 192 | 777 | 1,468 |
| Centralized light | 7 | Brute Force | 1,310 | 375 | 418 | 517 |
| Centralized light | 7 | Spoofing | 45,471 | 32,864 | 8,431 | 4,176 |
| Centralized light | 7 | Mirai | 246,284 | 245,317 | 2 | 965 |
| Centralized light | 17 | DDoS | 1,203,157 | 1,011,469 | 12 | 191,676 |
| Centralized light | 17 | DoS | 364,574 | 263,936 | 5 | 100,633 |
| Centralized light | 17 | Recon | 68,363 | 48,447 | 16,338 | 3,578 |
| Centralized light | 17 | Web-based | 2,437 | 221 | 859 | 1,357 |
| Centralized light | 17 | Brute Force | 1,310 | 323 | 476 | 511 |
| Centralized light | 17 | Spoofing | 45,471 | 32,287 | 9,725 | 3,459 |
| Centralized light | 17 | Mirai | 246,284 | 245,358 | 2 | 924 |
| Centralized light | 27 | DDoS | 1,203,157 | 1,060,947 | 15 | 142,195 |
| Centralized light | 27 | DoS | 364,574 | 233,497 | 2 | 131,075 |
| Centralized light | 27 | Recon | 68,363 | 53,274 | 13,198 | 1,891 |
| Centralized light | 27 | Web-based | 2,437 | 172 | 613 | 1,652 |
| Centralized light | 27 | Brute Force | 1,310 | 335 | 416 | 559 |
| Centralized light | 27 | Spoofing | 45,471 | 31,753 | 8,588 | 5,130 |
| Centralized light | 27 | Mirai | 246,284 | 244,797 | 4 | 1,483 |
| IID federated | 7 | DDoS | 1,203,157 | 1,012,631 | 16 | 190,510 |
| IID federated | 7 | DoS | 364,574 | 249,931 | 7 | 114,636 |
| IID federated | 7 | Recon | 68,363 | 49,506 | 14,946 | 3,911 |
| IID federated | 7 | Web-based | 2,437 | 0 | 718 | 1,719 |
| IID federated | 7 | Brute Force | 1,310 | 326 | 453 | 531 |
| IID federated | 7 | Spoofing | 45,471 | 26,801 | 12,300 | 6,370 |
| IID federated | 7 | Mirai | 246,284 | 244,769 | 5 | 1,510 |
| IID federated | 17 | DDoS | 1,203,157 | 1,010,588 | 30 | 192,539 |
| IID federated | 17 | DoS | 364,574 | 249,303 | 13 | 115,258 |
| IID federated | 17 | Recon | 68,363 | 49,605 | 15,131 | 3,627 |
| IID federated | 17 | Web-based | 2,437 | 55 | 714 | 1,668 |
| IID federated | 17 | Brute Force | 1,310 | 333 | 433 | 544 |
| IID federated | 17 | Spoofing | 45,471 | 25,838 | 12,858 | 6,775 |
| IID federated | 17 | Mirai | 246,284 | 244,916 | 9 | 1,359 |
| IID federated | 27 | DDoS | 1,203,157 | 1,033,907 | 21 | 169,229 |
| IID federated | 27 | DoS | 364,574 | 236,071 | 13 | 128,490 |
| IID federated | 27 | Recon | 68,363 | 50,053 | 14,575 | 3,735 |
| IID federated | 27 | Web-based | 2,437 | 0 | 709 | 1,728 |
| IID federated | 27 | Brute Force | 1,310 | 337 | 431 | 542 |
| IID federated | 27 | Spoofing | 45,471 | 25,693 | 12,833 | 6,945 |
| IID federated | 27 | Mirai | 246,284 | 244,793 | 4 | 1,487 |
| Controlled non-IID | 7 | DDoS | 1,203,157 | 1,191,843 | 105 | 11,209 |
| Controlled non-IID | 7 | DoS | 364,574 | 65,398 | 92 | 299,084 |
| Controlled non-IID | 7 | Recon | 68,363 | 14,961 | 52,746 | 656 |
| Controlled non-IID | 7 | Web-based | 2,437 | 0 | 2,359 | 78 |
| Controlled non-IID | 7 | Brute Force | 1,310 | 0 | 1,248 | 62 |
| Controlled non-IID | 7 | Spoofing | 45,471 | 67 | 45,148 | 256 |
| Controlled non-IID | 7 | Mirai | 246,284 | 243,277 | 212 | 2,795 |
| Controlled non-IID | 17 | DDoS | 1,203,157 | 1,182,701 | 110 | 20,346 |
| Controlled non-IID | 17 | DoS | 364,574 | 88,092 | 100 | 276,382 |
| Controlled non-IID | 17 | Recon | 68,363 | 12,318 | 55,307 | 738 |
| Controlled non-IID | 17 | Web-based | 2,437 | 0 | 2,391 | 46 |
| Controlled non-IID | 17 | Brute Force | 1,310 | 0 | 1,286 | 24 |
| Controlled non-IID | 17 | Spoofing | 45,471 | 193 | 44,878 | 400 |
| Controlled non-IID | 17 | Mirai | 246,284 | 243,435 | 216 | 2,633 |
| Controlled non-IID | 27 | DDoS | 1,203,157 | 1,190,474 | 107 | 12,576 |
| Controlled non-IID | 27 | DoS | 364,574 | 76,330 | 82 | 288,162 |
| Controlled non-IID | 27 | Recon | 68,363 | 13,380 | 54,333 | 650 |
| Controlled non-IID | 27 | Web-based | 2,437 | 0 | 2,382 | 55 |
| Controlled non-IID | 27 | Brute Force | 1,310 | 0 | 1,275 | 35 |
| Controlled non-IID | 27 | Spoofing | 45,471 | 1,002 | 44,052 | 417 |
| Controlled non-IID | 27 | Mirai | 246,284 | 243,319 | 154 | 2,811 |

All seeds, all arms, both populations, precision/recall/F1, confusion matrices and error counts
remain in `final-test-results.json`. This appendix neither authorizes deployment nor changes thresholds.
