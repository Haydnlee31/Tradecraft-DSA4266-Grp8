# What the centralized–federated study establishes

Updated 13 October 2026. **A: class-by-class failure synthesis and B: performance–cost comparison are complete.** Existing final-test counts were audited and consolidated; the nine frozen reference models were benchmarked locally in two independent CPU processes. No model was trained, no cloud instance was used, and no original validation or test arrays were opened for this work.

The most useful new synthesis is that **the non-IID deficit is not mainly a rare-class problem**. Spoofing and Recon together account for 56.6% of its macro-F1 gap to centralized light; Web-based and Brute Force together account for 22.5%. These are arithmetic contributions to a score difference, not causal attributions. Improving federation's overall optimization remains a meaningful research direction even if the rare classes remain difficult. There is not yet evidence of a federated method beating the matched centralized model.

## The main comparison and its limits

The official39 extension compares the same 5,096-parameter LayerNorm network in centralized-light, IID-federated and controlled non-IID-federated settings. All primary references use the full 39 inputs, two million training rows, the fixed step-20 endpoint and training seeds 7, 17 and 27. Federation uses 20 simulated clients on one machine. These are the nine predeclared full39 references, not the best models selected from the masked-feature sensitivity experiments.

| Final-test measure | Centralized light | IID federated | Controlled non-IID |
| --- | ---: | ---: | ---: |
| Macro-F1, mean ± sample SD | 0.65673 ± 0.00091 | 0.61169 ± 0.00349 | 0.40494 ± 0.00328 |
| Benign false-alert rate | 17.49% | 19.08% | 0.25% |
| False alerts per 10,000 benign flows | 1,749 | 1,908 | 25 |
| Gap below centralized macro-F1 | — | 4.50 percentage points | 25.18 percentage points |

These are within-collection, duplicate-grouped results, not unseen capture-session or device-level performance. Training-seed spread is not uncertainty across independent datasets or client partitions: the partition seed is fixed. Rates per 10,000 simply rescale observed conditional rates; they are not traffic forecasts. The very low non-IID false-alert rate is accompanied by extensive attack misses.

The official39 experiment has **no matched centralized-heavy arm**. Keep the earlier 46-feature heavy/light study separate; differences in data, preprocessing and architecture prevent a combined leaderboard or a claim that light replaces heavy. The [completed-study handoff](../full_data_extension/final-closeout.md) retains those boundaries.

## A. Different categories fail in different ways

All eight mean final-test recalls are shown below. The [generated tables](study-synthesis-tables.md) also contain every class's precision and F1, per-class error partitions, work counts and arithmetic score-gap contributions; the [machine-readable evidence](study-synthesis-results.json) retains each seed separately.

| Class | Centralized light | IID federated | Controlled non-IID |
| --- | ---: | ---: | ---: |
| Benign | 82.51% | 80.92% | 99.75% |
| DDoS | 84.80% | 84.70% | 98.77% |
| DoS | 70.74% | 67.23% | 21.01% |
| Recon | 73.93% | 72.73% | 19.83% |
| Web-based | 8.00% | 0.75% | 0.00% |
| Brute Force | 26.28% | 25.34% | 0.00% |
| Spoofing | 71.04% | 57.42% | 0.93% |
| Mirai | 99.54% | 99.41% | 98.81% |

**DDoS and Mirai demonstrate real coverage, but cannot stand in for the whole task.** Their high non-IID recall coexists with severe failures elsewhere. DDoS recall can also rise when other categories are incorrectly called DDoS; precision and the confusion matrix remain necessary.

**DoS is primarily an attack-category confusion, not a failure to raise any alarm.** Non-IID correctly categorizes only 21.01% of DoS flows, but sends just 0.025% to Benign. Almost all remaining errors are wrong attack categories, chiefly DDoS. Changing a benign-versus-attack threshold does not directly repair that distinction. Some alarm coverage is useful context, but it is not correct eight-class detection.

**Recon and Spoofing are major sources of the federated gap.** Non-IID sends 79.18% of Recon and 98.29% of Spoofing to Benign. Their low recalls are not explained by the two rarest classes being scarce. Spoofing contributes 9.30 macro-F1 percentage points and Recon 4.94 points to the 25.18-point centralized–non-IID difference. This gives a future update experiment a broader, more realistic target than “solve rare attacks.”

**Web-based has a shared learning problem plus an additional federated deficit.** Even centralized light gets only 8.00% correct; it sends 30.76% to Benign and 61.24% to another attack category. Non-IID sends 97.55% to Benign. A prior balanced-panel diagnostic raised validation Web-based recall to 51.07%, but precision was only 3.80% and benign false alerts reached 29.52%. That shows usable signal under another recipe, not a solution under realistic class prevalence. [Panel diagnostic](../full_data_extension/panel-validation-findings.md).

**Brute Force combines weak global learning with evidence of lost local signal.** Centralized and IID recall remain around one quarter, while non-IID sends 96.92% to Benign. In a separate earlier one-round validation probe, individual clients recovered roughly one quarter of Brute Force with low benign-probe false alerts, yet aggregation recovered none. Other clients' impressive attack recall came with extreme false alerts. A row-weighted probability ensemble also failed. This supports investigating aggregation and optimization, but not the claim that every local client is a good detector destroyed by averaging. [Aggregation probe](../full_data_extension/aggregation-mechanism-probe.md).

**Benign traffic exposes the operational cost.** Centralized light's stronger category coverage still causes about 1,749 false alarms per 10,000 benign flows at its fixed argmax decision rule. Neither its higher macro-F1 nor non-IID's conservatism supports deployment. The policy conclusion is to distinguish “wrong attack category,” “missed attack” and “false benign alarm,” rather than treating every error as interchangeable.

### What the later update experiments add—and do not add

The later class-gradient, finite-step and SCAFFOLD work is **training-only mechanism evidence**, not an improved final-test result. It must not be inserted into the table above.

- Class-loss safeguards could improve a rare-class loss without making that class the winning prediction. Lower cross-entropy is not automatically useful class recall. [Finite-step findings](finite-step-findings.md).
- SCAFFOLD reduced the endpoint client-update dispersion proxy by about 71%. Training macro-F1 improved from 0.57144 to 0.59397, but false alerts rose from 15.10% to 24.45%. This was a failed trade-off under the frozen screen, not a solved drift problem. [Matched SGD experiment](scaffold-findings.md).
- At the separately predeclared 5% alarm target, SCAFFOLD retained a smaller gain: 0.55115 versus FedAvg's 0.54174, while centralized-SGD reached 0.57981. Brute Force fell from 116 to 111 correct out of 621 audit rows; Web-based stayed at zero. Those rows were excluded from threshold fitting, **not model fitting**. [Operating-point diagnostic](operating-points-findings.md).

Thus drift correction has a measurable effect, but its benefit depends on the category and alarm budget. These findings do not prove that federation cannot work, that rare classes are unlearnable, or that more GPU capacity would fix them.

## B. Most of the federated cost is in training, not inference

| Quantity | Centralized light | IID federated | Controlled non-IID |
| --- | ---: | ---: | ---: |
| Parameters / FP32 parameter storage | 5,096 / 20,384 bytes | Same | Same |
| Processed training examples per run | 40,000,000 | 40,000,000 | 40,000,000 |
| Optimizer calls per run | 78,140 | 78,400 | 78,340 |
| Historical GPU training + validation, mean seconds | 299.53 | 310.30 | 310.72 |
| Estimated FP32 model exchange over 20 rounds | 0 | 16,307,200 bytes | 16,307,200 bytes |
| New local CPU batch-one time, ms | 0.02044 | 0.02026 | 0.02049 |
| New local CPU batch-512 time, ms per whole batch | 0.09581 | 0.09548 | 0.09504 |

The timing rows above summarize medians across three model seeds and two independent benchmark processes. Their individual repeat means are preserved in the [benchmark evidence](cost-benchmark-results.json). This is a descriptive summary, not six independent hardware replications or a statistical speed ranking.

**Measured locally:** Apple M3, 16 GiB RAM, CPU only, two intra-op threads and one inter-op thread; Python 3.13.9, PyTorch 2.14.0 and NumPy 2.5.3. Each model/batch case had 30 warmup calls and nine timing repetitions. One repetition averaged 1,000 batch-one calls or 100 batch-512 calls. Every model used the same prepared float32 training inputs, with case order rotated between repetitions. Both complete processes agreed on every non-timing field, including model states and prediction fingerprints.

Across models and processes, batch-one medians ranged from **0.01997 to 0.02117 ms**; batch-512 medians ranged from **0.09130 to 0.10059 ms**. The models share the same inference architecture. Small apparent lane differences should not be promoted as an inference advantage; batch timing variation between processes is larger than those differences.

The timed region is warm-cache PyTorch forward plus argmax. It excludes feature extraction, scaling, input-tensor creation, loading, transfers and service overhead. These are not per-request tail percentiles, end-to-end packet latency, measured Jetson performance or energy measurements. Parameter storage—about **19.91 KiB**—is not total process RAM or checkpoint size. The recorded 16 GiB is installed host RAM, not model memory consumption.

**Measured historically:** the GPU row is training plus full validation from the original cloud runs. It excludes setup and checkpoint I/O, is not a billed-cost receipt, and comes from a single-machine simulator rather than a distributed network. Equal processed examples do not mean identical update trajectories, optimizer counts or unique-row coverage.

**Counted, not network-measured:** FedAvg sends one parameter vector down and one up for each of 20 clients per round: `20,384 × 20 × 20 × 2 = 16,307,200` bytes, about 15.55 MiB. This excludes protocol overhead, retries, optimizer metadata, initialization and validation traffic. Central model exchange is zero, but collecting centralized raw data has an unknown transfer cost—not zero cost. No privacy guarantee follows from sending gradients rather than raw rows.

The later ten-round SCAFFOLD diagnostic doubles this model/control payload relative to its ten-round FedAvg control: 16,307,200 versus 8,153,600 bytes. Its extra server-plus-client control tensors occupy 428,064 bytes in the simulator. The inference network remains unchanged. That is a clear thesis result: gradient correction can move costs and modest benefits into training without adding inference complexity.

## The remaining idea worth investigating

The historical federated Adam recipe initializes a new client optimizer each round, whereas centralized Adam retains its optimizer history. The SCAFFOLD comparison tested plain SGD; it did **not** settle whether sharing adaptive optimizer information helps. This is a plausible mechanism, not an isolated cause already demonstrated by the evidence.

The [focused literature-backed proposal](../Central%20aligned%20federated%20updates.md) recommends one sequence:

1. Establish a numerical control: aggregate same-weights client gradients and apply persistent server Adam; compare with the *same weighted centralized accumulated-batch update*. This is not equivalent to sequential centralized minibatch Adam, and potentially frequent communication must be counted.
2. Test established **MIME-Adam**, which combines shared optimizer statistics with local gradient correction, against newly matched centralized and FedAvg controls. Do not rename it as a novel algorithm.
3. Only if that mechanism helps, evaluate a bounded adaptive local-block rule against fixed-block MIME. The proposed question is whether fresh gradient-disagreement measurements can reduce synchronization while retaining the benefit. Related adaptive-communication methods already exist; novelty would require a distinct, successful and carefully compared contribution.

Your revised priority—approach centralized macro-F1 even without solving rare classes—is reasonable for this extension. It does not require a human-review layer or another classifier search. Keep all eight recalls and false alerts visible, predeclare what “close” means, and do not relax the old failed screens retrospectively. The proposal creates no claim of better performance and launches no new experiment.

Any next fit must have a newly frozen from-scratch development protocol and a separate confirmation boundary. Repartitioning historical training rows does not create wholly independent evidence after extensive prior work; preserve that limitation. Keep the completed original final test closed to method selection. The immediate deliverable is now the consolidated study, not another cloud session.

## Reproducibility and handoff

The [synthesis implementation](../../scripts/official39_study_synthesis.py) rebuilds A and the historical portion of B from committed aggregate counts without private arrays or dependencies beyond Python's standard library. The [benchmark implementation](../../src/eval/official39_cost_benchmark.py) and [measurement plan](cost-benchmark-plan.json) were committed as `7edfdd2` before measurement. The receipt preserves that execution commit and source hashes. Two passes performed only forward evaluation; there were zero optimizer or backward calls.

Portable checks, from the repository root:

```bash
.venv/bin/python -m scripts.official39_study_synthesis
.venv/bin/python -m unittest tests.test_study_synthesis tests.test_official39_cost_benchmark tests.test_study_synthesis_evidence -q
```

Private archives, training inputs and original model tensors remain outside Git. Existing comments and unrelated working-tree changes are preserved. The report is explanatory evidence for the centralized-versus-federated thesis, not authorization for deployment or a new training sweep.

Verification: all **17 focused tests** passed. The full committed regression suite passed **668 tests in 55.935 seconds** in an isolated archive of commit `361214e`, excluding unrelated uncommitted files. Public aggregate reconstruction also passed with Python's site packages disabled. Both reports' local links resolve. This is local verification, not a claim of a new GitHub CI run; no branch push was performed for this handoff.
