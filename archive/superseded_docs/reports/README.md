# Study results and thesis guide

Start with the [integrated thesis](thesis/README.md): it connects the original
heavy/light comparison, the official39 study and the confirmed shared-optimizer
extension. The three populations have different protocols and must not be treated
as one accuracy leaderboard.

| Study | Main findings | Evidence |
| --- | --- | --- |
| Original configured heavy/light/federated models | [Three-seed comparison](confirmation_2026_10_05/README.md) | [Aggregate results](confirmation_2026_10_05/results/summary.json) |
| Official39 centralized and federated light models | [Final closeout](full_data_extension/final-closeout.md) | [Class and cost synthesis](federated_class_safeguards/study-synthesis-tables.md) |
| Shared optimizer and compression | [Reserved confirmation findings](federated_shared_optimizer/confirmation-findings.md) | [Audited results](federated_shared_optimizer/confirmation-results.json) |
| Explanations | [Original SHAP audit](explanations_2026_10_05/README.md) and [official39 analysis](full_data_extension/explanation-study.md) | Feature-level interpretation, not device vulnerability or causal rules |

The dated convergence, loss, normalization, screening, cloud and diagnostic
folders are supporting evidence. They remain at their existing paths because
active summaries and receipt checks depend on them. Completed evaluations are
not instructions to keep tuning on their exposed test sets.

Retired classifier searches and superseded early results are in the
[historical archive](../archive/README.md). New scratch runs belong under ignored
`outputs/`, not beside published report receipts. PDFs are review copies under
ignored `output/pdf/`; editable thesis text and figures are checked in.
