# Illustrated thesis chapter

This is an editable methods, results and discussion chapter for the completed
shared-optimizer extension, not a claim that the entire thesis is finished.
[Read the chapter](chapter.md). A PDF review copy is generated separately and is
not part of the committed evidence. The nine figures use existing committed
aggregate results: no training, new inference, threshold fitting or cloud work
is needed to reproduce them.

## Figure guide

| Chapter figure | Export | What the reader should learn |
| --- | --- | --- |
| 1 | [Training design](figures/01_design.png) | Which computation changes, which inference network stays fixed, and where fitting, development and confirmation belong. |
| 2 | [Class support](figures/09_class_support.png) | How few rare-category examples underpin the results; counts use a log axis. |
| 3 | [Primary performance](figures/02_primary.png) | Macro-F1 and actual benign false alerts, showing each seed as well as the mean. |
| 4 | [Recall heatmap](figures/03_recall.png) | Which classes improve and which remain poorly recognized. |
| 5 | [Confusion matrices](figures/04_confusion.png) | Where errors go, using a shared scale and true-class normalization. |
| 6 | [Rare-class errors](figures/05_rare_errors.png) | Correct category, wrong attack category and missed attack are distinct outcomes. |
| 7 | [Communication trade-off](figures/06_communication.png) | Compression reduces counted bytes but does not reduce synchronization frequency. |
| 8 | [Frozen operating points](figures/07_operating_points.png) | How alarm burden changes the comparison at four already-fixed thresholds; this is not an ROC curve. |
| 9 | [Historical baseline](figures/08_historical_context.png) | Earlier centralized/IID/non-IID results motivate the extension but belong to a different population and protocol. |

Every PNG has a same-named SVG for editable, resolution-independent thesis use.
Figure captions are in the chapter and [provenance manifest](figures/manifest.json).
[Numerical figure data](figures/figure-data.json) preserve method, seed and class
ordering, all confusion counts and the derived plotted values.

Do not combine Figure 9 with Figure 3 into one leaderboard. Do not present the
current shared-optimizer comparison as a matched heavy-model experiment, a
rare-class solution, physical-edge deployment evidence or a novel optimization
theorem. Error bars in Figure 7 are sample standard deviations over three training
seeds; the ranges in Figure 3 are not confidence intervals.

## Rebuild without accessing training data

From the repository root, use the project's existing Python environment:

```bash
python -m scripts.shared_optimizer_thesis_figures --output outputs/thesis-figures-rebuild-v1
python -m unittest tests.test_shared_optimizer_thesis_figures -v
```

The generator refuses an existing destination. It verifies exact source hashes
and the confirmation audit, then creates nine PNG/SVG pairs, derived JSON and a
manifest of all output hashes. It imports neither PyTorch nor an evaluator and
does not load datasets or checkpoints. PNG rendering may vary with the graphics
library or font versions; numerical JSON is the portable source of truth.

The three aggregate inputs are pinned in
[`scripts/shared_optimizer_thesis_figures.py`](../../../scripts/shared_optimizer_thesis_figures.py).
Do not update those pins silently to accommodate different experimental evidence.
Revisions to the experiment require a newly labeled report.

## PDF review copy

The optional renderer uses ReportLab only. Keep document dependencies separate
from the training environment if ReportLab is not already available. For example,
create a document-only virtual environment, install ReportLab there, then run:

```bash
python scripts/render_thesis_pdf.py \
  --source reports/federated_shared_optimizer/thesis/chapter.md \
  --output output/pdf/tradecraft-federated-thesis-chapter.pdf
```

The PDF is a review artifact; the Markdown and SVGs remain the editable source.
The renderer preserves an existing PDF rather than overwriting it. After any text
or layout change, render and visually inspect every PDF page before distribution.
The PDF does not include raw data, model checkpoints or private row-level outputs.

## Integration into the full thesis

1. Keep the original 46-feature heavy/light/federated study under its original
   dataset and protocol. Reuse its audited tables and figures in that chapter.
2. Present the official39 extension separately, explaining the feature schema,
   duplicate-group controls and within-collection generalization limits. Figure 9
   provides the concise final baseline context; detailed scaling and ablation
   experiments can be placed in an appendix rather than retold as a search diary.
3. Use the new chapter for the optimizer question, fixed confirmation protocol,
   class-level findings and communication accounting.
4. Keep explanation methods correctly named. The previous official39 integrated
   gradients analysis is not SHAP on the twelve new optimizer checkpoints. Do not
   turn feature associations into causal or per-device security claims.
5. Finish the thesis-wide introduction, literature review and conclusion after
   agreeing the scope and departmental template with the team. No further tuning
   on the now-open confirmation cohort is part of writing this chapter.

There are older training/convergence plots elsewhere in the repository, but the
latest updater runs saved endpoints and work counters rather than validation
scores at every step. A truthful convergence curve for these exact runs cannot
be reconstructed from counters. Rerunning final experiments solely to obtain
prettier curves is unnecessary. If a later, separately designed development
experiment genuinely needs convergence evidence, add logging before that run and
label it as a new experiment rather than retrofitting the confirmed endpoints.

## Verification

The figure pack is checked against source identities, exact confusion supports,
row normalization, rare-class outcome totals, cohort sizes and communication
counters. Its tests also rebuild all figures while blocking NumPy array loading
and verify that existing exports are preserved. The PDF is visually reviewed
separately because document dependencies are not required for training or CI.
