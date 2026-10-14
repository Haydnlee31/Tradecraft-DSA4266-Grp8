# Integrated Tradecraft thesis draft

[Read the integrated manuscript](thesis.md). It connects the original heavy/light
comparison, official39 study, explanation work and confirmed shared-optimizer
extension. This is an academic content draft, not a university-formatted final
submission. No template or page limit was found in the repository. Author names,
institutional front matter and required citation style still need team review.

The draft preserves three distinct evaluation settings. The legacy comparison
uses validation-selected configurations; official39 uses frozen final-test
endpoints; the optimizer extension uses reserved confirmation with frozen alarm
thresholds. A higher score in one setting is not a measured improvement over a
score from another population.

## Visual contents

The manuscript includes fifteen figures. Six are newly generated here: the
three-study map, original quality/size comparison, original recall heatmap,
saved learning curves, official39 class-gap decomposition and original SHAP
numerical diagnostics. Nine figures are reused from the
[reviewed optimizer chapter](../federated_shared_optimizer/thesis/README.md).
All have PNG and SVG versions; all captions state the relevant population and
interpretation limits. The PDF has navigation bookmarks for section headings.

[Figure data](figures/figure-data.json), [figure manifest](figures/manifest.json)
and [manuscript manifest](thesis.manifest.json) retain source and output hashes.
The learning-curve [portable history](evidence/legacy-histories.json) preserves
all 720 validation points from the original twelve result logs. Every source
log matched its previously committed SHA-256 before export. It contains no
traffic rows, checkpoints or per-row predictions. New optimizer convergence
curves were not invented: those runs did not log per-step validation scores.

## Editing and rebuilding

Edit `parts/01_context_and_baselines.md` and `parts/02_discussion.md` for the
thesis-wide material. Sections 12-19 are assembled from sections 1-8 of the
existing optimizer chapter, with only heading/figure numbers and relative links
changed. Do not separately edit those copies in the assembled manuscript.

Run from the repository root using the existing environment:

```bash
python -m scripts.study_thesis figures --output outputs/thesis-figures-rebuild-v1
python -m scripts.study_thesis assemble --output outputs/thesis-rebuild-v1/thesis.md
python -m unittest tests.test_study_thesis tests.test_shared_optimizer_thesis_figures -v
```

The assembled Markdown is intended to live in `reports/thesis/`, where its
relative links resolve. A scratch rebuild verifies text equality; it is not a
self-contained copy of the figure assets. Existing outputs are never overwritten.
The optional `history` command exports checksummed saved result logs only; it is
not needed for ordinary figure or document rebuilding from a clone.

Use a document-only environment containing ReportLab to render the review PDF:

```bash
python scripts/render_thesis_pdf.py \
  --source reports/thesis/thesis.md \
  --output output/pdf/tradecraft-integrated-thesis-draft.pdf \
  --title "Tradecraft: Centralized and Federated Learning for IoT Intrusion Classification" \
  --footer-label "TRADECRAFT  |  Integrated thesis draft"
```

The PDF is ignored by Git and delivered separately. After content or renderer
changes, visually inspect every page; text extraction alone does not verify
layout. No experimental training, new test inference or cloud resources are
required for this writing stage.

## Submission review still needed

1. Apply the course template, page limit, author details and required references
   style once supplied. Confirm any declaration about AI assistance required by
   the course rather than inventing one.
2. Have teammates verify descriptions of their original implementations and the
   scoped empirical claims. The draft is not a claim that experiments or prose
   were independently authored or verified by every teammate.
3. Keep operational deployment and physical-edge claims out unless separately
   measured. The final candidate improves the tested FedAvg control; it does not
   solve rare-class detection or establish a novel optimization theorem.
4. Preserve the local extension branch until the user chooses to push it. Review
   integration into the shared project branch separately; do not overwrite
   unrelated manuscript or teammate work.
