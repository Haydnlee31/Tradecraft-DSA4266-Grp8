"""Validation-only adapter for the frozen configured-model confirmation study.

Reuse the existing decision engine and gates; do not select a new recipe, replace
non-IID with IID, or infer deployment readiness from illustrative research gates.
Only saved artifacts are read. No dataset or model object is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.data.label_map import CLASSES
from src.eval.decision import Criteria, build_decision, render_markdown


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    """Use explicit exceptions so checks cannot disappear under python -O."""
    if not condition:
        raise ValueError(message)


def validate_metrics(metrics):
    """Recompute gates from counts, catching NaN, stale summaries and bad labels."""
    matrix = np.asarray(metrics['confusion_matrix'], dtype=float)
    require(matrix.shape == (len(CLASSES), len(CLASSES)), 'Confusion matrix shape mismatch')
    require(np.isfinite(matrix).all() and (matrix >= 0).all() and (matrix == np.floor(matrix)).all(), 'Invalid confusion counts')
    support, predicted = matrix.sum(1), matrix.sum(0)
    require((support > 0).all(), 'Every validation class needs support')
    recall = np.diag(matrix) / support
    precision = np.divide(np.diag(matrix), predicted, out=np.zeros(len(CLASSES)), where=predicted != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(CLASSES)), where=(precision + recall) != 0)
    expected = {'macro_f1': float(f1.mean()), 'accuracy': float(np.trace(matrix) / matrix.sum()),
                'benign_false_alert_rate': float(1 - recall[CLASSES.index('Benign')])}
    for key, value in expected.items():
        require(math.isfinite(float(metrics[key])) and math.isclose(float(metrics[key]), value, abs_tol=1e-10), f'Inconsistent {key}')
    require(set(metrics['per_class_recall']) == set(CLASSES), 'Recall class schema mismatch')
    for i, name in enumerate(CLASSES):
        require(math.isclose(float(metrics['per_class_recall'][name]), recall[i], abs_tol=1e-10), f'Inconsistent recall: {name}')
    return matrix.astype(np.int64)


def error_breakdown(matrix):
    """Partition each attack row into correct, benign miss and wrong attack type."""
    benign = CLASSES.index('Benign')
    result = {}
    for i, name in enumerate(CLASSES):
        total = int(matrix[i].sum())
        correct, to_benign = int(matrix[i, i]), int(matrix[i, benign])
        if i == benign:
            result[name] = {'support': total, 'correct': correct, 'false_alerts': total - correct,
                            'false_alert_rate': (total - correct) / total}
        else:
            wrong_attack = total - correct - to_benign
            result[name] = {'support': total, 'correct_category': correct,
                            'predicted_benign': to_benign, 'wrong_attack_category': wrong_attack,
                            'category_recall': correct / total, 'attack_to_benign_rate': to_benign / total,
                            'wrong_attack_category_rate': wrong_attack / total}
    return result


def normalize_confirmation(study, plan):
    """Translate verified frozen recipes without inventing test measurements."""
    require(study.get('test_evaluated') is False, 'Expected validation-only study')
    require(set(study['lanes']) == set(plan['lanes']) == {'heavy', 'light', 'iid', 'dirichlet'}, 'Missing or extra lane')
    require(plan['training_seeds'] == [0, 1, 2] and plan['new_training_seeds'] == [1, 2], 'Unexpected seed protocol')
    reports, diagnostics, reference = [], {}, None
    for lane, item in study['lanes'].items():
        require(set(item['runs']) == {'0', '1', '2'}, f'Missing/duplicate/extra seed: {lane}')
        recipe, lane_manifest = plan['lanes'][lane], None
        for seed_text, run in item['runs'].items():
            seed, m = int(seed_text), run['manifest']
            settings = m['settings']
            require(settings['lane'] == lane and settings['seed'] == seed, 'Seed/lane mismatch')
            require(run['reused_screening_seed'] == (seed == 0), 'Screening seed annotation mismatch')
            require(m['classes'] == CLASSES, 'Canonical class order mismatch')
            for k, v in plan['common'].items():
                require(settings[k] == v, f'Frozen setting changed: {k}')
            require(settings['normalization'] == recipe['normalization'] and
                    m['model_config']['normalization'] == recipe['normalization'] and
                    m['model_config']['dropout'] == recipe['dropout'], 'Frozen model recipe changed')
            require(settings['dropout'] in (None, recipe['dropout']), 'Dropout argument mismatch')
            require(set(m['split_sha256']) == {'train', 'val'}, 'Unexpected split access')
            if reference is None:
                reference = m
            for k in ('source_sha256', 'split_sha256', 'packages', 'feature_columns', 'scaler_sha256', 'train_class_counts'):
                require(m[k] == reference[k], f'Incompatible provenance: {k}')
            if lane_manifest is None:
                lane_manifest = m
            require(m['model_config'] == lane_manifest['model_config'] and m['partition_sha256'] == lane_manifest['partition_sha256'], 'Configuration/partition drift across seeds')
            require(isinstance(run['num_parameters'], int) and run['num_parameters'] > 0, 'Invalid parameter count')
            matrix = validate_metrics(run['validation_metrics'])
            key = f'{lane}-seed-{seed}'
            diagnostics[key] = error_breakdown(matrix)
            federated = lane in ('iid', 'dirichlet')
            # Only one frozen configuration per lane/partition is admitted. The
            # legacy history field is a schema adapter, not a fabricated run.
            report = {'variant': 'heavy' if lane == 'heavy' else 'light',
                      'lane': 'federated' if federated else 'centralized',
                      'config': m['model_config'], 'loss': settings['loss'],
                      'num_parameters': run['num_parameters'],
                      'parameter_bytes': run['num_parameters'] * 4,
                      'args': settings, 'history': [{'val_macro_f1': run['validation_metrics']['macro_f1']}],
                      'validation_metrics': run['validation_metrics'], 'test_metrics': None,
                      '_source': run['path'], '_research_key': key}
            if federated:
                report.update(partitioner=lane, alpha=settings['alpha'], num_clients=settings['clients'],
                              local_epochs=m['local_epochs'], fraction_train=m['fraction_train'],
                              class_weights='global', strategy='FedAvg')
            reports.append(report)
    return reports, diagnostics


def load_frozen(summary_path, plan_path):
    """Pin summary to plan and raw run receipts, without opening any data splits."""
    study, plan = read(summary_path), read(plan_path)
    require(study['plan_sha256'] == sha(plan_path), 'Plan hash mismatch')
    launch = read(Path(plan['output_root']) / 'launch.json')
    require(launch['plan_sha256'] == sha(plan_path), 'Pre-run receipt mismatch')
    for item in study['lanes'].values():
        for run in item['runs'].values():
            folder = Path(run['path'])
            require(sha(folder / 'result.json') == run['result_sha256'], 'Raw result hash mismatch')
            require(read(folder / 'environment.json') == run['manifest'], 'Manifest mismatch')
            raw = read(folder / 'result.json')
            require(raw['test_metrics'] is None and raw['steps'] == 60 and read(folder / 'status.json')['status'] == 'complete', 'Incomplete/non-validation run')
            require([h['step'] for h in raw['history']] == list(range(1, 61)), 'Incomplete history')
            best = max(raw['history'], key=lambda h: h['validation_metrics']['macro_f1'])
            require(best['step'] == raw['best_step'] == run['best_step'] and best['validation_metrics'] == raw['validation_metrics'], 'Best-checkpoint selection mismatch')
            require(raw['validation_metrics'] == run['validation_metrics'] and raw['num_parameters'] == run['num_parameters'], 'Summary metrics/size mismatch')
            require(raw['parameter_bytes'] == run['num_parameters'] * 4, 'Expected saved float32 parameter byte count')
            require(sha(folder / 'best.pt') == raw['checkpoint_sha256'] == run['checkpoint_sha256'], 'Checkpoint hash mismatch')
            require(sha(folder / 'scaler.joblib') == run['manifest']['scaler_sha256'], 'Scaler hash mismatch')
            if run['manifest']['partition_sha256'] is not None:
                require(sha(folder / 'assignments.npz') == run['manifest']['partition_sha256'], 'Partition hash mismatch')
    reports, diagnostics = normalize_confirmation(study, plan)
    return study, plan, reports, diagnostics


def build_research_decision(study, plan, reports, diagnostics, require_federated=False):
    criteria = Criteria(require_federated=require_federated)
    decision = build_decision(reports, criteria)
    # New seeds alone do not satisfy the unchanged three-seed gate. Present the
    # incomplete sensitivity view explicitly rather than relaxing it to two.
    fresh = build_decision([r for r in reports if r['args']['seed'] in plan['new_training_seeds']], criteria)
    seed_warnings = []
    for report in reports:
        weak = [name for name, value in report['validation_metrics']['per_class_recall'].items()
                if value < criteria.min_class_recall]
        if weak:
            seed_warnings.append(f"{report['_research_key']}: individual-seed recall below {criteria.min_class_recall:.2f} for {', '.join(weak)}. Existing gates use means, not per-seed minima.")
    return {'scope': 'configured-model validation research decision', 'deployment_authorized': False,
            'test_evaluated': False, 'decision': decision, 'new_seed_sensitivity': fresh,
            'error_breakdown_by_seed': diagnostics, 'frozen_plan': plan,
            'warnings': [*plan['limitations'], *seed_warnings,
                         'Decision gates are unchanged illustrative research preferences, not operational standards.',
                         'No false-alert-rate ceiling exists in the inherited gates; high false alerts remain a warning, not a hidden new gate.',
                         'All-seed eligibility includes reused screening seed 0; new-seed sensitivity retains the three-seed requirement and is incomplete.',
                         'IID remains a diagnostic control and cannot replace the non-IID target.',
                         'Model MiB describes saved float32 parameter tensors, not runtime RAM or edge feasibility.']}


def render_research(payload):
    d = payload['decision']
    lines = ['# Frozen validation-only research decision', '',
             '**Research comparison only. No deployment is authorized.**', '',
             f"Gate-based research choice: **{d['recommended_lane'] or 'none'}**. Engine status: `{d['status']}`.", '',
             'The frozen recipes are not re-tuned. Default non-IID selection and all existing metric, size and seed gates are preserved.', '',
             render_markdown(d), '', '## New-seed-only sensitivity', '',
             f"Status: `{payload['new_seed_sensitivity']['status']}`; provisional research choice: `{payload['new_seed_sensitivity']['provisional_recommendation']}`. Only seeds 1/2 are new, so the unchanged requirement of three seeds is not met.", '',
             '## All candidates, including IID diagnostic control', '',
             '| Lane / partition | Seeds | Mean validation F1 | Worst mean class recall |', '|---|---|---:|---:|']
    for c in d['all_candidates']:
        lines.append(f"| {c['lane']} / {c.get('partitioner', 'central')} | {c['seeds']} | {c['validation_macro_f1_mean']:.4f} | {c['worst_validation_class_recall_mean']:.4f} |")
    lines += ['', '## Every class: mean validation recall', '', '| Class | Heavy | Light | IID control | Non-IID target |', '|---|---:|---:|---:|---:|']
    candidates = {('heavy' if c['lane'] == 'centralized-heavy' else 'light' if c['lane'] == 'centralized-light' else c['partitioner']): c for c in d['all_candidates']}
    for name in CLASSES:
        lines.append('| ' + name + ' | ' + ' | '.join(f"{100*candidates[k]['validation_per_class_recall_mean'][name]:.2f}%" for k in ('heavy', 'light', 'iid', 'dirichlet')) + ' |')
    lines += ['', '## Error types: full validation, each seed', '',
              'Each attack row is partitioned into correct category, predicted benign, and wrong attack category. These supplemental counts do not replace the eight-class target or gate metrics.', '',
              '| Model / seed | True class | Support | Correct category | Predicted benign | Wrong attack category |', '|---|---|---:|---:|---:|---:|']
    for key, breakdown in payload['error_breakdown_by_seed'].items():
        for name, row in breakdown.items():
            if name != 'Benign':
                lines.append(f"| {key} | {name} | {row['support']} | {row['correct_category']} | {row['predicted_benign']} | {row['wrong_attack_category']} |")
    lines += ['', '## False alerts: each seed', '', '| Model / seed | Benign support | False alerts | Rate |', '|---|---:|---:|---:|']
    for key, breakdown in payload['error_breakdown_by_seed'].items():
        row = breakdown['Benign']
        lines.append(f"| {key} | {row['support']} | {row['false_alerts']} | {100*row['false_alert_rate']:.2f}% |")
    lines += ['', '## Interpretation safeguards', ''] + ['- ' + w for w in payload['warnings']]
    lines += ['', '## Human-review policy hypotheses', '',
              '- Do not let a model benign output suppress independent web/application or authentication alerts.',
              '- Review Recon/Spoofing alerts with service context rather than automatically blocking traffic.',
              '- Preserve attack alerts while treating uncertain DoS/DDoS subtypes separately.',
              '- SHAP feature rankings do not establish causal rules, thresholds, or device vulnerability.',
              '- These are investigation proposals; no operational policy or decision threshold is changed.', '']
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirmation', type=Path, default=Path('reports/confirmation_2026_10_05/results/summary.json'))
    p.add_argument('--plan', type=Path, default=Path('configs/local_confirmation_plan.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-federated', action='store_true')
    p.add_argument('--strict', action='store_true')
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    study, plan, reports, diagnostics = load_frozen(args.confirmation, args.plan)
    payload = build_research_decision(study, plan, reports, diagnostics, args.require_federated)
    payload['provenance'] = {'confirmation_sha256': sha(args.confirmation), 'plan_sha256': sha(args.plan),
                             'adapter_sha256': sha(__file__), 'engine_sha256': sha(Path(__file__).with_name('decision.py'))}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json = args.output / 'decision.json'
    write_json.write_text(json.dumps(payload, indent=2))
    (args.output / 'README.md').write_text(render_research(payload))
    print(json.dumps({'status': payload['decision']['status'], 'research_recommendation': payload['decision']['recommended_lane'],
                      'deployment_authorized': False, 'output': str(args.output)}, indent=2))
    if args.strict and payload['decision']['status'] != 'ready':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
