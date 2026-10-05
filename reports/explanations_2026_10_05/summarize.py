"""Compile saved SHAP artifacts; keep sampled explanations and population errors separate."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('outputs/explain-confirmation-2026-10-05'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit = json.loads((args.input / 'summary.json').read_text())
    sampling = json.loads((args.input / 'sampling.json').read_text())
    results, classes, features = audit['results'], audit['classes'], audit['feature_columns']
    lanes = ('heavy', 'light', 'iid', 'dirichlet')
    args.output.mkdir(parents=True, exist_ok=False)
    # The compact report preserves numerical evidence; raw attribution tensors
    # stay in the original output directory for signed-case and repeat audits.
    (args.output / 'summary.json').write_text(json.dumps(audit, indent=2))
    (args.output / 'sampling.json').write_text(json.dumps(sampling, indent=2))
    lines = ['# SHAP numerical audit and failure evidence', '',
             f"Shared core: {audit['core_count']} validation rows; separate targeted supplement: {audit['supplement_count']} rows. Background: {audit['background_count']} training rows. Two expected-gradients repeats with {audit['nsamples_per_repeat']} integration samples each. Attributions are in logit units.", '',
             '## Approximation diagnostics', '',
             '| Model / seed | Mean abs residual | p95 abs residual | Mean abs residual / mean abs output difference | Mean MC top-5 overlap across classes |',
             '|---|---:|---:|---:|---:|']
    for key, r in results.items():
        mc = np.mean([v['mc_repeat_top5_overlap'] for v in r['classes'].values()])
        lines.append(f"| {key} | {r['mean_abs_residual_logits']:.3f} | {r['p95_abs_residual_logits']:.3f} | {100*r['relative_mean_abs_residual']:.1f}% | {100*mc:.1f}% |")
    lines += ['', 'Residuals measure deviation from logit difference to the sampled-background mean, not predictive error. Ratios are aggregate diagnostics, not per-case guarantees. Repeat overlap measures numerical sampling stability; it does not validate causal meaning.', '',
              '## Core true-class-logit features and training-seed stability', '',
              '| Model | True class | Leading features (normalized importance averaged across seeds) | Features in every seed top 5 | Mean pairwise top-5 overlap |', '|---|---|---|---|---:|']
    for lane in lanes:
        for name in classes:
            scores = [np.array(results[f'{lane}-seed-{s}']['classes'][name]['mean_abs_attributions']) for s in (0, 1, 2)]
            normed = [a / max(a.sum(), 1e-12) for a in scores]
            order = np.argsort(-np.mean(normed, axis=0), kind='stable')[:5]
            common = set.intersection(*[set(np.argsort(-a, kind='stable')[:5]) for a in scores])
            stable = audit['stability'][lane][name]['mean_top5_overlap']
            lines.append(f"| {lane} | {name} | {', '.join(features[i] for i in order)} | {', '.join(features[i] for i in order if i in common) or 'None'} | {100*stable:.1f}% |")
    lines += ['', 'These means use only the shared balanced core, eight rows per true class. Absolute importance is not feature direction or a rule threshold; normalization prevents raw logit scales from dominating across seeds.', '',
              '## Full-validation failure destinations (not SHAP sample estimates)', '',
              '| Model / seed | True class | Correct / total | Most common wrong destinations (counts) |', '|---|---|---:|---|']
    destinations = {}
    for key, r in results.items():
        matrix = np.array(r['full_validation_metrics']['confusion_matrix'])
        destinations[key] = {}
        for name in ('Benign', 'Web-based', 'Brute Force', 'DoS'):
            i = classes.index(name)
            wrong = sorted([(classes[j], int(matrix[i, j])) for j in range(len(classes)) if j != i and matrix[i, j]], key=lambda item: -item[1])
            destinations[key][name] = {'correct': int(matrix[i, i]), 'total': int(matrix[i].sum()), 'wrong_destinations': wrong}
            lines.append(f"| {key} | {name} | {matrix[i,i]} / {matrix[i].sum()} | {', '.join(f'{c}: {n}' for c,n in wrong[:3])} |")
    lines += ['', '## Targeted signed examples: seed 1 for every lane', '',
              'Seed 1 is used consistently here, not selected for favorable explanations. These are illustrative individual rows, not class-wide causal conclusions. A positive contribution favors the wrong predicted class over the true class relative to the training background. Several residuals are comparable to or exceed the observed decision margin (notably heavy Web/DoS and IID Web/DoS below): do not use those cases as precise local explanations. Low aggregate residual does not guarantee local fidelity.', '',
              '| Model | Case | Validation row | True → predicted | Logit margin | Residual | Leading signed margin contributions |', '|---|---|---:|---|---:|---:|---|']
    for lane in lanes:
        key = lane + '-seed-1'
        for group in ('false_alert', 'miss_Web-based', 'miss_Brute Force', 'miss_DoS'):
            row = sampling['groups'][key][group]['selected_row']
            if row is None:
                continue
            case = next(c for c in results[key]['cases'] if c['row'] == row)
            features_text = ', '.join(f"{f['feature']} {f['signed_attribution']:+.2f}" for f in case['margin_top_features'][:3])
            lines.append(f"| {lane} | {group} | {row} | {case['class']} → {case['predicted_class']} | {case['predicted_minus_true_logit']:.2f} | {case['margin_residual']:.2f} | {features_text} |")
    lines += ['', 'Row identities/source files, all 12 models’ individual cases, numerical diagnostics and full confusion matrices are in [summary.json](summary.json) and [sampling.json](sampling.json). The raw NPZ arrays retain signed attributions and both Monte Carlo repeats.', '',
              '## Limitations', ''] + ['- ' + item for item in audit['limitations']]
    lines += ['- The 128-row training background contains no Web or Brute Force examples in this draw; it represents this sampled training distribution, not every attack or deployment traffic.',
              '- No independent background/sample sensitivity experiment was run. Cross-seed agreement on one shared sample does not establish population-wide explanation stability.', '',
              'Method reference: [SHAP GradientExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html).', '',
              '![Training-seed feature overlap](stability.png)', '']
    (args.output / 'README.md').write_text('\n'.join(lines))
    (args.output / 'failure_destinations.json').write_text(json.dumps(destinations, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = np.array([[audit['stability'][lane][name]['mean_top5_overlap'] for name in classes] for lane in lanes])
    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    im = ax.imshow(data, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(range(len(classes)), classes, rotation=25, ha='right')
    ax.set_yticks(range(len(lanes)), lanes)
    for i, j in itertools.product(range(len(lanes)), range(len(classes))):
        ax.text(j, i, f'{100*data[i,j]:.0f}%', ha='center', va='center', color='white' if data[i,j] > .65 else 'black')
    ax.set_title('Mean pairwise seed overlap of top-5 features\nShared balanced core; true-class logits; not causal validity')
    fig.colorbar(im, ax=ax, label='Top-5 overlap')
    fig.savefig(args.output / 'stability.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
