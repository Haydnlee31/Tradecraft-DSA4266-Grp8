# Federated loss safeguards improve stability but not enough rare attack detection

The refined correction passes its training-loss safety checks, but **does not
pass the five-round detection screen**. It produces two correct Brute Force
training predictions where matched FedAvg produces none, while Web-based
recall remains zero. Overall macro-F1 is lower than FedAvg. Stop this recipe
here; the evidence does not justify longer training, cloud compute, or a claim
that the rare-class problem is solved.

The study still uses one small eight-output network and simulated federated
learning. There is no new inference classifier, language model or human review.
All results below concern the same two-million-row **training population**.
Validation and test arrays were not opened. The previously completed final
test remains closed; these experiments are exploratory fitting diagnostics.

## The finite update now has an actual loss check

The [first safeguard](diagnostic-findings.md) corrected directions predicted to
harm Benign, Web-based or Brute Force, but actual benign loss still increased
at two inspected states. A direction that looks safe at the starting point
need not remain safe after a finite move through a nonlinear network.

The refinement adds a fixed small descent margin and checks the actual result.
It finds the closest update to FedAvg with each normalized protected gradient
having a dot product at most `−0.01 × ||FedAvg displacement||`. Clients then
compute loss sums and counts locally at the candidate weights. The server
accepts the first passing scale among `1, 0.5, 0.25, 0.125`; it does not select
the highest F1. Each protected mean loss must strictly decrease by more than
0.000001 and satisfy the fixed sufficient-decrease rule. Numerical rounding,
excessive expansion and near-zero progress are checked explicitly.

This is an adaptation of established Armijo-style backtracking to three
training losses, not a newly invented line-search principle or a convergence
guarantee for this nonconvex network. The margin and finite search limit are
engineering choices frozen before execution. ([Boyd and Vandenberghe,
slide 9.10](https://web.stanford.edu/class/ee364a/lectures/unconstrained.pdf))

The same independent source checkpoints after rounds 2, 5 and 20 were tested.
Their ordinary FedAvg weights, client work, gradients and training metrics
reproduced the previous diagnostic exactly. The accepted loss changes were:

| Source round | Accepted scale | Benign CE change | Web-based CE change | Brute Force CE change |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.5 | −0.000579 | −0.082879 | −0.089371 |
| 5 | 1.0 | −0.001108 | −0.138068 | −0.112544 |
| 20 | 1.0 | −0.001959 | −0.052137 | −0.050608 |

Negative changes mean lower cross-entropy, not percentage improvements. At
round 2, the full step increased benign CE by 0.004308 and was rejected. Its
half-step passed. Accepted updates retained 49.84%, 100.06% and 100.06% of the
ordinary displacement norm: this was not a zero-update safety trick. These
three single updates still had zero recall for both rare classes.

## Five paired rounds expose the remaining trade-off

After the one-step checks passed, a separate five-round plan was frozen before
running. Both arms started from identical round-20 weights, with identical
client assignments, round/client seeds, minibatch order and reset-Adam local
epochs. One arm retained FedAvg; the other applied the unchanged correction
after each round. This tests rescue of an already collapsed model, not
training the method from scratch.

All five corrected updates passed at full scale. Benign, Web-based and Brute
Force training CE decreased at every accepted step. Nevertheless, at the
fixed round-25 endpoint:

| Training metric | Paired FedAvg | Corrected update |
| --- | ---: | ---: |
| Macro-F1 | 0.417309 | 0.411712 |
| Benign false-alert rate | 0.4946% | 0.2167% |
| Web-based recall | 0 / 2,391 = 0% | 0 / 2,391 = 0% |
| Brute Force recall | 0 / 1,269 = 0% | 2 / 1,269 = 0.1576% |

The two Brute Force detections appeared gradually: one at round 24, two at
round 25. They are a small fitting change, not reliable evidence of improved
generalization. The predeclared screen required a rare-class recall gain of
at least one percentage point, no loss of recall on the other rare class,
macro-F1 loss no larger than 0.5 percentage point, and false-alert increase no
larger than 0.1 percentage point. The correction missed the recall threshold
and lost **0.5597 percentage point** of macro-F1. It passed the false-alert
condition. No intermediate round was selected instead of the endpoint.

All eight endpoint recalls are shown here, in percent:

| Class | FedAvg | Corrected |
| --- | ---: | ---: |
| Benign | 99.5054 | 99.7833 |
| DDoS | 98.8650 | 98.8728 |
| DoS | 19.5023 | 19.4038 |
| Recon | 22.3260 | 19.4819 |
| Web-based | 0.0000 | 0.0000 |
| Brute Force | 0.0000 | 0.1576 |
| Spoofing | 5.1140 | 4.7670 |
| Mirai | 98.9139 | 98.9172 |

Protecting benign loss does not protect the other attack categories. Recon
recall fell by 2.8441 percentage points relative to FedAvg, while Spoofing also
weakened. The correction is buying more conservative benign behavior without
enough rare-attack benefit. Decreasing true-class loss can raise a class's
probability while leaving another class as the winner; loss progress and
correct classifications are different outcomes.

## Training overhead remains a real cost

Per pass, the three independent one-step checks processed six million local
training examples, 11,751 optimizer steps, six million gradient-probe examples,
18 million full training-evaluation forwards and 768,656 protected-risk
forwards. The paired five-round trial processed 20 million local training
examples, 39,170 optimizer steps, ten million gradient-probe examples,
20 million full training-evaluation forwards and 1,098,080 protected-risk
forwards. These are repeated exposures to the two-million-row cohort, not
additional unique data. Each experiment was independently replayed.

A prospective three-class implementation needs up to 61,152 extra FP32
gradient bytes per client per round, plus counts. The five accepted candidate
checks would additionally broadcast 2,038,400 model bytes across all 20 clients
and return 9,600 bytes of loss sums/counts. These are payload estimates,
excluding transport overhead; no network latency was measured. The diagnostic
computes all eight gradients, accumulates in FP64 and runs clients on one
machine. A distributed FP32 exchange would need its own numerical checks.

Raw flows need not leave clients in the proposed protocol, but gradient and
class-count sharing still raises privacy concerns. No secure aggregation,
differential privacy or physical edge deployment is claimed. Model parameter
count and inference architecture are unchanged; the extra costs are training
and communication costs.

## Stop this recipe and test local drift correction next

Do not enlarge the margin, loosen benign protection or add more rounds after
seeing this result. These findings reject this fixed short rescue recipe as
a compelling improvement, not every possible gradient-based federated method.
They also do not establish that a from-scratch version would fail.

The next focused candidate is **SCAFFOLD with matched FedAvg-SGD controls**.
It corrects client drift during local optimization rather than only repairing
the final aggregate. Its published update supplies a principled starting
point; improvement on this dataset remains a hypothesis. Switching from reset
Adam to SGD must not be mistaken for the effect of the correction itself.
Keep the architecture, cohort and partition fixed, include centralized-SGD
as a reference, and verify the weighted-client formulation before running.
Only a bounded local mechanism trial should come next—not another classifier
search or a cloud campaign. ([SCAFFOLD, ICML 2020](https://proceedings.mlr.press/v119/karimireddy20a.html))

## Reproducible evidence

The [finite-step plan](finite-step-plan.json) was committed before its real
data run. The [five-round plan](trajectory-plan.json) was committed only after
the first diagnostic passed, and before the paired trajectory began. Original
comments, runners and failed results remain untouched.

Aggregate receipts are in [finite-step results](finite-step-results.json) and
[paired trajectory results](trajectory-results.json). They retain all class
metrics, confusion matrices, loss checks, work counts and source hashes. Private
data and checkpoint files are not committed. Independent replays matched all
non-timing fields. Timings are local audit timings, not hardware benchmarks.
