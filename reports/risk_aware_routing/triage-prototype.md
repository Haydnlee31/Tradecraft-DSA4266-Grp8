# Local read only triage prototype

The prototype shows a saved classifier prediction alongside a separate
**Needs review** or **Not flagged** status. It does not change the predicted
class, block traffic or assume that a reviewer resolves the case.
“Not flagged” is not a safety guarantee.

Open the generated [local preview](../../outputs/official39-triage-preview-v1/index.html).
It is a self-contained HTML file; no server, internet connection, API key or
cloud machine is needed. On your Mac, from the repository root:

```bash
open outputs/official39-triage-preview-v1/index.html
```

The file is private and ignored by git. Teammates receiving only the source
code will not have your row preview or saved model caches.

## What to inspect

Use the four selectors to change the base lane, training seed, review score
and illustrative calibration ceiling. All six base/seed pairs and all 54
previous review policies are included. The initial IID, seed 7, confidence,
5% selection is a demonstration, not a chosen operating point or best seed.

- Population cards show referrals across all 408,027 internal-audit flows,
  not the tiny preview sample. Overruns remain visible; no rows are trimmed
  to make a later cohort appear compliant with its calibration ceiling.
- The preview contains the first eight flagged and first eight unflagged
  rows, where available, in original order. It is not ranked by risk or
  selected by true class. Row references are dataset positions, not devices.
- The saved class prediction and uncalibrated maximum softmax are distinct
  from the review score and flag. The learned repair/harm score is not an
  attack probability or a calibrated safety estimate.
- Historical evaluation shows the unchanged base macro-F1, benign
  false-alert rate, every class's recall, review coverage and unresolved
  errors. These are previously measured aggregate outcomes, not a new
  evaluation or improved recall attributed to reviewing a flow.

For a useful demonstration, change IID seed 7 to seed 17 while retaining
confidence and the 5% ceiling. The latter displays a small workload overrun.
Then compare non-IID learned and confidence scores: their actual workloads
differ, even at the same nominal ceiling. Neither comparison approves a
deployment policy.

## Implementation boundary

The [builder](../../scripts/official39_triage.py) loads only pinned validation
features, panel membership, frozen probability/prediction caches and trees.
It does not open ground-truth row arrays or the historical final test, fit a
model, rerun neural inference, or recalibrate a threshold. Historical class
outcomes are read from the already-verified aggregate evidence, separately
from review selection.

The [review function](../../src/eval/triage_preview.py) has no true-label or
expert-prediction argument. It preserves classifier IDs and applies the
fixed score threshold without sorting or later-cohort quota trimming. The
browser only selects among precomputed displays; it does not run a model or
send data anywhere. HTML embeds escaped JSON and uses text nodes rather than
data-derived markup; its content policy disallows external connections.

Generated `index.html` and `preview.json` contain a small private prediction
sample. The builder requires a fresh directory beneath ignored `outputs/`
and refuses to overwrite an existing preview. No row previews are committed.
`checks.json` records source hashes, unchanged prediction hashes, flag hashes,
scenario counts and agreement with the frozen review workload.

## Verification

All 54 policies reproduce their historical referral counts, rates and budget
flags exactly. A monitored local run opened only the 15 allowed arrays: the
validation feature matrix, retained-row mapping, saved partitions, and six
prediction/probability cache pairs. No ground-truth or test arrays were opened.

Two complete runs produce byte-identical HTML, preview JSON and checks JSON.
All **420 tests** pass in 33.6 seconds in a clean repository snapshot, excluding
unrelated uncommitted work; 16 tests cover this prototype. The environment
passes `pip check`, and the embedded JavaScript passes its syntax check.

The browser automation tool rejects local `file://` pages, so visual layout
and browser interaction were not inspected through that tool. No browser
permission or security setting was changed. Open the local preview to check
its appearance on your screen. Static HTML structure, script syntax, escaping
and Python routing behaviour are checked separately.

Run locally, with the previous private caches and the recorded CPU environment:

```bash
.venv/bin/python -m unittest tests.test_triage_preview -q
.venv/bin/python scripts/official39_triage.py \
  --output outputs/official39-triage-preview-reproduction
```

Optional `--router-run`, `--packed` and `--panel` arguments locate the earlier
private inputs. Their identities must still match the verified evidence.
No dependencies were added, and historical training/evaluation code remains
unchanged.

## What comes next

Use this prototype to agree with your teammates on the intended workflow:
who reviews flagged flows, what workload is acceptable, and what happens
when a frozen threshold exceeds that workload. Those choices remain open;
there is no approved reviewer or operating budget in this demo.

If a review-assisted study is useful, a later experiment must measure actual
reviewer outcomes, mistakes, time and cost. If the deliverable must remain a
fully automatic classifier, report the [trade-off findings](tradeoff-findings.md)
without treating referrals as detections. This prototype creates no new
accuracy evidence, independent evaluation, edge measurement or reason to
restart cloud compute.
