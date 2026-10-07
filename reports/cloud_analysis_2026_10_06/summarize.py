"""Publish aggregate evidence only; checkpoints, rows and SHAP arrays stay private."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.data.label_map import CLASSES
from src.explain.research import sha, load_model, logits, confusion
from src.models.dataset import to_arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    audit = json.loads((a.input/'audit.json').read_text())
    ex = json.loads((a.input/'explanations/summary.json').read_text())
    sampling = json.loads((a.input/'explanations/sampling.json').read_text())
    previous = json.loads(Path('reports/cloud_study_2026_10_06/results/summary.json').read_text())
    study = json.loads((a.input/'study.json').read_text())
    evidence = {'fedprox_audit': audit, 'explanation_provenance': {k:v for k,v in ex.items() if k != 'results'},
                'background_class_counts': sampling['background_class_counts'], 'models': {}, 'learning_curves': {}}
    lines = ['# Cloud diagnostic analysis — 2026-10-06', '',
        '## Decision', '',
        'Keep the cloud stopped. Neither FedProx candidate passes the frozen promotion gates. '
        'This report diagnoses the existing models; it does not select a deployable model or authorize another sweep.', '',
        'The artifact audit reproduced the historical FedAvg bridge exactly: all round metrics/task losses '
        'and the selected best tensors. The local CPU explanation run independently reproduced the full '
        'validation confusion matrix for all 12 cloud checkpoints before computing attributions. No test data was opened.', '',
        '## FedProx screen', '',
        '| Configuration | Macro-F1 | F1 delta | False alerts | Gate A | Gate B | Training + validation seconds |',
        '|---|---:|---:|---:|---|---|---:|']
    for key,r in audit['runs'].items():
        lines.append(f"| {key} | {r['metrics']['macro_f1']:.5f} | {r['macro_f1_delta']:+.5f} | {100*r['metrics']['benign_false_alert_rate']:.2f}% | {r['gate_a']} | {r['gate_b']} | {r['seconds']:.1f} |")
    lines += ['', 'Gate A: F1 gain ≥0.01, false-alert increase ≤0.02, every class recall loss ≤0.02. '
              'Gate B requires a 0.05 absolute false-alert reduction and is impossible from this 0.03555 baseline. '
              'The reference row is not a candidate. Criteria were not relaxed after seeing results.', '',
              '## Matched light-model evidence', '',
              '| Lane | Seeds | Macro-F1 mean ± sample SD | Mean false alerts |', '|---|---|---:|---:|']
    for lane in ('light','iid','dirichlet'):
        results = [ex['results'][f'{lane}-seed-{seed}']['full_validation_metrics'] for seed in (7,8,9)]
        scores = [r['macro_f1'] for r in results]
        lines.append(f"| {lane} | 7/8/9 | {np.mean(scores):.5f} ± {np.std(scores, ddof=1):.5f} | {100*np.mean([r['benign_false_alert_rate'] for r in results]):.2f}% |")
    lines += ['', 'Seed 7 informed earlier screening; seeds 8/9 are the new confirmation seeds. '
              'The heavy reference has only seed 7 here and different normalization/dropout. '
              'It cannot establish a pure model-capacity effect. See the earlier cloud study for all per-class means.', '',
              '## Learning curves: budget limits, not convergence claims', '',
              '| Run | Best step | Final step | Macro-F1 gain over last 10 steps |', '|---|---:|---:|---:|']
    for key,r in previous['runs'].items():
        curve = r['curve']
        if len(curve) < 11:
            continue
        gain = curve[-1]['macro_f1'] - curve[-11]['macro_f1']
        evidence['learning_curves'][key] = {'best_step':r['best_step'], 'last10_gain':gain, 'curve':curve}
        lines.append(f"| {key.removeprefix('outputs/')} | {r['best_step']} | {len(curve)} | {gain:+.5f} |")
    for key,r in audit['runs'].items():
        lines.append(f"| {key} | {r['best_step']} | 60 | {r['last10_f1_gain']:+.5f} |")
    lines += ['', 'A positive tail gain is descriptive, not proof that more rounds will satisfy the safety gates. '
              'The existing 120-round FedAvg experiment already failed promotion. Do not restart it or expand FedProx coefficients.', '',
              '## Expected-gradients numerical quality', '',
              f"All models share {ex['core_count']} balanced validation examples plus {ex['supplement_count']} error-enriched examples; "
              f"background: {ex['background_count']} training rows. Two Monte Carlo repeats, {ex['nsamples_per_repeat']} samples each. "
              'These are diagnostic samples, not population performance estimates.', '',
              '| Model | Mean absolute residual / mean absolute output difference | Mean repeat top-5 overlap |',
              '|---|---:|---:|']
    for key,r in ex['results'].items():
        repeat = np.mean([x['mc_repeat_top5_overlap'] for x in r['classes'].values()])
        compact = {k:v for k,v in r.items() if k != 'cases'}
        matrix = np.asarray(r['full_validation_metrics']['confusion_matrix'])
        compact['failure_destinations'] = {}
        for i,c in enumerate(CLASSES):
            compact['failure_destinations'][c] = sorted(
                [{'predicted':CLASSES[j], 'count':int(matrix[i,j]), 'fraction':float(matrix[i,j]/matrix[i].sum())}
                 for j in range(len(CLASSES)) if i != j and matrix[i,j]], key=lambda x:-x['count'])
        # Count unreliable signed cases; do not silently turn them into policies.
        errors = [c for c in r['cases'] if not c['correct']]
        compact['error_cases'] = len(errors)
        compact['error_cases_residual_exceeds_margin'] = sum(abs(c['margin_residual']) >= abs(c['predicted_minus_true_logit']) for c in errors)
        evidence['models'][key] = compact
        lines.append(f"| {key} | {100*r['relative_mean_abs_residual']:.1f}% | {100*repeat:.1f}% |")
    lines += ['', 'High residual means the approximation does not closely reconstruct the explained logit difference. '
              'High repeat overlap alone is not sufficient: both repeats may share approximation bias. '
              'No precise SHAP-derived thresholds or automated blocking rules are justified.', '',
              '## Rare-class errors on the full validation split', '',
              '| Model | True class | Recall | Leading wrong destination (fraction of true class) |', '|---|---|---:|---|']
    for key,r in evidence['models'].items():
        for c in ('Web-based','Brute Force','DoS'):
            wrong = r['failure_destinations'][c][0]
            recall = r['full_validation_metrics']['per_class'][c]['recall']
            lines.append(f"| {key} | {c} | {100*recall:.2f}% | {wrong['predicted']}: {100*wrong['fraction']:.2f}% |")
    lines += ['', '## Rare-class feature hypotheses (shared balanced core)', '',
              '| Model | True class | Top five absolute true-class-logit features | Repeat overlap |', '|---|---|---|---:|']
    selected_features = set()
    for key,r in ex['results'].items():
        if not key.endswith('-7'):
            continue
        for c in ('Web-based','Brute Force','DoS'):
            v = r['classes'][c]
            top = [f['feature'] for f in v['top_features'][:5]]
            selected_features.update(top)
            lines.append(f"| {key} | {c} | {', '.join(top)} | {100*v['mc_repeat_top5_overlap']:.0f}% |")
    # Descriptive distribution checks on bounded sampled splits, never raw full CSVs.
    # Quantile overlap is not a statistical test or evidence of causality/leakage.
    evidence['feature_quantiles'] = {}
    for split in ('train','val'):
        path = Path('data/splits') / f'{split}.parquet'
        if sha(path) != ex['split_sha256'][split]:
            raise ValueError('Data changed since explanation run')
        frame = pl.read_parquet(path)
        evidence['feature_quantiles'][split] = {}
        for c in ('Benign','Web-based','Brute Force','DoS','DDoS'):
            subset = frame.filter(pl.col('class') == c)
            evidence['feature_quantiles'][split][c] = {'rows':len(subset), 'features': {
                f: [subset[f].quantile(q) for q in (.1,.5,.9)] for f in sorted(selected_features)}}
    # Raw labels are diagnostic slices of the fixed eight-class target, not a
    # new 33-class training task. Small slice counts must remain visible.
    torch.set_num_threads(2)
    validation = pl.read_parquet('data/splits/val.parquet')
    raw_labels = validation['label'].to_numpy()
    rare_labels = validation.filter(pl.col('class').is_in(['Web-based','Brute Force']))['label'].unique().sort().to_list()
    evidence['rare_raw_label_slices'] = {}
    correct_bf = {}
    for lane,item in study['lanes'].items():
        for seed,run in item['runs'].items():
            key = f'{lane}-seed-{seed}'
            model,scaler,manifest,report = load_model(run['path'])
            x,y = to_arrays(validation,scaler,manifest['feature_columns'])
            pred = logits(model,x).argmax(1)
            if confusion(y,pred).tolist() != report['validation_metrics']['confusion_matrix']:
                raise ValueError('Prediction mismatch during subgroup diagnostics')
            correct_bf[key] = set(np.flatnonzero((y == CLASSES.index('Brute Force')) & (pred == y)).tolist())
            evidence['rare_raw_label_slices'][key] = {}
            for label in rare_labels:
                mask = raw_labels == label
                evidence['rare_raw_label_slices'][key][label] = {'support':int(mask.sum()),
                    'eight_class_correct':int((pred[mask] == y[mask]).sum()),
                    'predicted_class_counts':dict(zip(CLASSES,np.bincount(pred[mask],minlength=len(CLASSES)).tolist()))}
    keys = ['dirichlet-seed-7','fedprox_mu001-seed-7','fedprox_mu01-seed-7']
    evidence['fedprox_brute_force_correct_overlap'] = {
        'counts':{k:len(correct_bf[k]) for k in keys},
        'same_correct_rows':all(correct_bf[k] == correct_bf[keys[0]] for k in keys[1:])}
    lines += ['', '### Rare raw-label slices (still predicting eight classes)', '',
              '| Raw label | Validation support | Central light seed 7 correct | FedAvg seed 7 correct | FedProx 0.1 correct |',
              '|---|---:|---:|---:|---:|']
    for label in rare_labels:
        rows = [evidence['rare_raw_label_slices'][k][label] for k in ('light-seed-7','dirichlet-seed-7','fedprox_mu01-seed-7')]
        lines.append(f"| {label} | {rows[0]['support']} | {rows[0]['eight_class_correct']} | {rows[1]['eight_class_correct']} | {rows[2]['eight_class_correct']} |")
    lines += ['', f"FedAvg and both FedProx candidates detect the same Brute Force rows: {evidence['fedprox_brute_force_correct_overlap']['same_correct_rows']}. "
              'Counts and all model/raw-label prediction distributions are preserved in evidence.json. '
              'These validation slices are descriptive and small; they do not justify subclass-level performance guarantees.', '']
    lines += ['## Distribution warning before any further scaling', '',
              '| Class | Train median IAT | Validation median IAT | Train median Number / Weight | Validation median Number / Weight |',
              '|---|---:|---:|---|---|']
    for c in ('Benign','Web-based','Brute Force'):
        t = evidence['feature_quantiles']['train'][c]['features']
        v = evidence['feature_quantiles']['val'][c]['features']
        lines.append(f"| {c} | {t['IAT'][1]:.6g} | {v['IAT'][1]:.6g} | {t['Number'][1]} / {t['Weight'][1]} | {v['Number'][1]} / {v['Weight'][1]} |")
    lines += ['', 'These features also rank highly in the explanation sample. Their very different medians '
              'are consistent with changing mixtures of feature regimes across the upstream splits, not proof '
              'of corruption, leakage or causality. IAT units/semantics have not been independently verified. '
              'Benign and Brute Force have overlapping Number/Weight quantiles. DoS and DDoS both have '
              '10th/50th/90th quantiles of Number=9.5 and Weight=141.55. Such overlap does not establish '
              'indistinguishability in the complete feature space. Preserve the upstream splits; do not pool or resplit them.', '',
              '## Explanation limits that need a local follow-up', '',
              'The sampled natural-prevalence background contains no Web-based or Brute Force rows. '
              'A stratified-background sensitivity comparison would change the reference population and must '
              'be labelled as such, not silently substituted to obtain prettier explanations.', '',
              '| Model | Error cases with absolute residual ≥ decision margin / sampled errors |', '|---|---:|']
    for key,r in evidence['models'].items():
        lines.append(f"| {key} | {r['error_cases_residual_exceeds_margin']} / {r['error_cases']} |")
    lines += ['', 'These cases are unsuitable for precise signed local-rule interpretation at this integration budget. '
              'Global repeat stability does not rescue an unreliable individual explanation.', '',
              '| Lane | Web top-5 seed overlap | Brute Force overlap | DoS overlap |', '|---|---:|---:|---:|']
    for lane in ('light','iid','dirichlet'):
        values = [ex['stability'][lane][c]['mean_top5_overlap'] for c in ('Web-based','Brute Force','DoS')]
        lines.append(f"| {lane} | {values[0]:.1%} | {values[1]:.1%} | {values[2]:.1%} |")
    lines += ['', 'Feature quantiles (10th/50th/90th) for the above feature union on train and validation '
              'are saved in evidence.json, stratified by Benign/Web/Brute Force/DoS/DDoS. '
              'They are descriptive overlap checks, not proof of distribution equality or a causal mechanism.', '',
              '## Policy interpretation and next step', '',
              '- Treat predictions as analyst-review signals, not automatic blocking decisions: no model is operationally approved.',
              '- A non-IID benign prediction is not evidence that Web or Brute Force activity is safe. Independent application/authentication telemetry would be needed for such decisions; it is not supplied by these flow CSVs.',
              '- DoS/DDoS confusion motivates reviewing flood-related flow features, not assigning device-specific vulnerabilities.',
              '- Inspect residuals, seed stability and background sensitivity before making feature-specific recommendations. No physical edge latency/power claims are made.',
              '- Preserve the failed screens. Credits remaining are not evidence that more parameters will solve these errors.', '',
              '## Limits and provenance', '',
              'Only seed 7 is available for heavy and FedProx here; their between-seed stability is null, not zero. '
              'The three light-model lanes use seeds 7/8/9. Attributions are on logits with a shared natural-prevalence '
              'training background; correlated flow statistics and off-manifold interpolations limit interpretation. '
              'A second background/sample sensitivity study has not been run. Historical test inspection remains disclosed '
              'in the original study; the test split cannot be called untouched across the entire project.', '',
              'Aggregate evidence, source/package hashes, all class metrics and numerical diagnostics: [evidence.json](evidence.json). '
              'Private rows, scaler/checkpoint files and attribution arrays remain under ignored outputs/.', '']
    a.output.mkdir(parents=True)
    (a.output/'evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    (a.output/'README.md').write_text('\n'.join(lines))
    print('Published aggregate diagnostic report:', a.output)


if __name__ == '__main__':
    main()
