"""Failure-focused, validation-only expected-gradients audit of frozen models.

The balanced core is shared by every model/seed and is kept separate from an
error-enriched supplement. Neither is a natural-prevalence performance estimate.
Saved training scalers are verified and reused: this module never fits a scaler.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch

from src.data.label_map import CLASSES
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.dataset import feature_columns, to_arrays


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def load_model(folder):
    """Load only trusted local project artifacts, failing closed on mismatches."""
    folder = Path(folder)
    m, r = read(folder / 'environment.json'), read(folder / 'result.json')
    if read(folder / 'status.json')['status'] != 'complete' or r['test_metrics'] is not None:
        raise ValueError('Expected a completed validation-only research run')
    if sha(folder / 'best.pt') != r['checkpoint_sha256'] or sha(folder / 'scaler.joblib') != m['scaler_sha256']:
        raise ValueError('Checkpoint/scaler hash mismatch')
    # Explanation code is new, but inference-critical source must remain exact.
    for file in ('src/models/architectures.py', 'src/models/dataset.py', 'src/data/label_map.py'):
        if sha(file) != m['source_sha256'][file]:
            raise ValueError(f'Inference source changed: {file}')
    saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    if (saved['classes'] != CLASSES or saved['feature_columns'] != m['feature_columns']
            or saved['seed'] != m['settings']['seed'] or saved['best_step'] != r['best_step']):
        raise ValueError('Checkpoint schema/selection mismatch')
    config = dict(saved['config'])
    config['hidden_dims'] = list(config['hidden_dims'])
    if config != m['model_config']:
        raise ValueError('Checkpoint model configuration mismatch')
    config['hidden_dims'] = tuple(config['hidden_dims'])
    model = MLPClassifier(saved['num_features'], saved['num_classes'], MLPConfig(**config))
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval()
    scaler = joblib.load(folder / 'scaler.joblib')
    if scaler.n_features_in_ != len(saved['feature_columns']):
        raise ValueError('Scaler feature count mismatch')
    return model, scaler, m, r


def logits(model, x, batch_size=512):
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i + batch_size])).numpy()
                               for i in range(0, len(x), batch_size)])


def confusion(y, prediction):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(matrix, (y, prediction), 1)
    return matrix


def select_rows(y, predictions, per_class, seed):
    """Uniform within-class core, plus one seeded example per error type/model.

    Returning the union means every checkpoint sees identical rows. Reasons are
    recorded even when multiple models select the same row. Empty error/correct
    groups are reported instead of inventing examples of nonexistent detections.
    """
    rng = np.random.default_rng(seed)
    core = np.concatenate([rng.choice(np.flatnonzero(y == i), min(per_class, (y == i).sum()), replace=False)
                           for i in range(len(CLASSES))]).astype(int)
    reasons, groups = {}, {}
    benign = CLASSES.index('Benign')
    for key, pred in predictions.items():
        masks = {'false_alert': (y == benign) & (pred != benign), 'correct': pred == y}
        for name in ('Web-based', 'Brute Force', 'DoS'):
            masks['miss_' + name] = (y == CLASSES.index(name)) & (pred != y)
        groups[key] = {}
        for name, mask in masks.items():
            eligible = np.flatnonzero(mask)
            row = int(rng.choice(eligible)) if len(eligible) else None
            groups[key][name] = {'eligible_validation_count': len(eligible), 'selected_row': row}
            if row is not None:
                reasons.setdefault(str(row), []).append(key + ':' + name)
    supplement = sorted(set(map(int, reasons)) - set(core.tolist()))
    return core, np.asarray(core.tolist() + supplement, dtype=int), reasons, groups


def top_features(values, features, count=8):
    order = np.argsort(-np.asarray(values), kind='stable')[:count]
    return [{'feature': features[i], 'value': float(values[i])} for i in order]


def overlap(a, b, k=5):
    return len(set(np.argsort(-a, kind='stable')[:k]) & set(np.argsort(-b, kind='stable')[:k])) / k


def attribution_summary(values, repeat, outputs, baseline, y, features, core_count):
    """Describe true-class logits separately from predicted-minus-true margins."""
    residual = outputs - baseline - values.sum(axis=1)
    classes = {}
    core_y = y[:core_count]
    for i, name in enumerate(CLASSES):
        mask = core_y == i
        a = np.abs(values[:core_count][mask, :, i]).mean(0)
        b = np.abs(repeat[:core_count][mask, :, i]).mean(0)
        classes[name] = {'core_count': int(mask.sum()), 'mean_abs_attributions': a.tolist(),
                         'top_features': top_features(a, features),
                         'mc_repeat_top5_overlap': overlap(a, b)}
    denom = float(np.abs(outputs - baseline).mean())
    return {'classes': classes, 'mean_abs_residual_logits': float(np.abs(residual).mean()),
            'p95_abs_residual_logits': float(np.quantile(np.abs(residual), .95)),
            'mean_abs_output_minus_background': denom,
            'relative_mean_abs_residual': float(np.abs(residual).mean() / max(denom, 1e-12)),
            'mc_repeat_mean_abs_attribution_difference': float(np.abs(values - repeat).mean())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirmation', type=Path, default=Path('reports/confirmation_2026_10_05/results/summary.json'))
    p.add_argument('--plan', type=Path, default=Path('configs/local_confirmation_plan.json'))
    p.add_argument('--splits', type=Path, default=Path('data/splits'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-class', type=int, default=8)
    p.add_argument('--background', type=int, default=128)
    p.add_argument('--nsamples', type=int, default=256)
    p.add_argument('--sample-seed', type=int, default=4266)
    args = p.parse_args(argv)
    if min(args.per_class, args.background, args.nsamples) < 1:
        raise ValueError('Sample counts must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    study = read(args.confirmation)
    if study['plan_sha256'] != sha(args.plan):
        raise ValueError('Frozen confirmation plan changed')
    torch.set_num_threads(2)
    # Only the already bounded sampled train/validation splits are opened.
    frames = {s: pl.read_parquet(args.splits / f'{s}.parquet') for s in ('train', 'val')}
    split_hashes = {s: sha(args.splits / f'{s}.parquet') for s in frames}
    models, predictions, manifests, reports = {}, {}, {}, {}
    x = y = features = reference_scaler = None
    for lane, item in study['lanes'].items():
        for seed, run in item['runs'].items():
            key = lane + '-seed-' + seed
            folder = Path(run['path'])
            if sha(folder / 'result.json') != run['result_sha256']:
                raise ValueError('Confirmation result changed')
            model, scaler, manifest, report = load_model(folder)
            if manifest['split_sha256'] != split_hashes:
                raise ValueError('Train/validation data changed')
            if feature_columns(frames['train']) != manifest['feature_columns'] or feature_columns(frames['val']) != manifest['feature_columns']:
                raise ValueError('Data feature order changed')
            if features is None:
                features = manifest['feature_columns']
                reference_scaler = scaler
                x, y = to_arrays(frames['val'], scaler, features)
            elif manifest['scaler_sha256'] != next(iter(manifests.values()))['scaler_sha256']:
                raise ValueError('Models do not share an identical scaler')
            pred = logits(model, x).argmax(1)
            if confusion(y, pred).tolist() != report['validation_metrics']['confusion_matrix']:
                raise ValueError(f'Full validation prediction parity failed: {key}')
            models[key], predictions[key], manifests[key], reports[key] = model, pred, manifest, report
            print(f'{key}: full validation confusion reproduced', flush=True)
    core, rows, reasons, groups = select_rows(y, predictions, args.per_class, args.sample_seed)
    rng = np.random.default_rng(args.sample_seed)
    background_rows = rng.choice(frames['train'].height, min(args.background, frames['train'].height), replace=False)
    background, _ = to_arrays(frames['train'][background_rows.tolist()], reference_scaler, features)
    selected_x, selected_y = x[rows], y[rows]
    row_info = [{'row': int(row), 'class': CLASSES[y[row]],
                 'source_file': frames['val']['source_file'][int(row)] if 'source_file' in frames['val'].columns else None,
                 'raw_label': frames['val']['label'][int(row)],
                 'cohort': 'balanced_core' if i < len(core) else 'error_supplement',
                 'selection_reasons': reasons.get(str(row), [])} for i, row in enumerate(rows)]
    args.output.mkdir(parents=True, exist_ok=False)
    sampling = {'sample_seed': args.sample_seed, 'core_rows': core.tolist(), 'rows': row_info,
                'background_train_rows': background_rows.tolist(), 'groups': groups,
                'background_class_counts': frames['train'][background_rows.tolist()].group_by('class').len().to_dicts(),
                'full_validation_class_counts': {name: int((y == i).sum()) for i, name in enumerate(CLASSES)}}
    write(args.output / 'sampling.json', sampling)
    np.savez_compressed(args.output / 'inputs.npz', background=background, x=selected_x, y=selected_y, validation_rows=rows)
    import shap
    results = {}
    for key, model in models.items():
        tensor = torch.from_numpy(selected_x)
        explainer = shap.GradientExplainer(model, torch.from_numpy(background), batch_size=128)
        values = []
        for mc_seed in (0, 1):
            v = np.asarray(explainer.shap_values(tensor, nsamples=args.nsamples, rseed=mc_seed))
            if v.shape != (len(rows), len(features), len(CLASSES)) or not np.isfinite(v).all():
                raise ValueError('Invalid attribution shape/values')
            values.append(v)
            print(f'{key}: expected gradients repeat {mc_seed} complete ({len(rows)} rows)', flush=True)
        outputs, baseline = logits(model, selected_x), logits(model, background).mean(0)
        # Average two independent Monte Carlo integrations; preserve both for QA.
        mean_values = (values[0] + values[1]) / 2
        result = attribution_summary(mean_values, values[1], outputs, baseline, selected_y, features, len(core))
        # Compare the two genuinely separate repeats (not mean versus a repeat).
        mc = attribution_summary(values[0], values[1], outputs, baseline, selected_y, features, len(core))
        for name in CLASSES:
            result['classes'][name]['mc_repeat_top5_overlap'] = mc['classes'][name]['mc_repeat_top5_overlap']
        result['mc_repeat_mean_abs_attribution_difference'] = mc['mc_repeat_mean_abs_attribution_difference']
        cases = []
        for i, info in enumerate(row_info):
            true, pred = int(selected_y[i]), int(outputs[i].argmax())
            # For an error, a positive attribution to predicted-minus-true logit
            # supports the erroneous preference relative to this background.
            margin_values = mean_values[i, :, pred] - mean_values[i, :, true]
            ordered = np.argsort(-np.abs(margin_values), kind='stable')[:8]
            cases.append({**info, 'predicted_class': CLASSES[pred], 'correct': true == pred,
                          'predicted_minus_true_logit': float(outputs[i, pred] - outputs[i, true]),
                          'margin_residual': float(outputs[i, pred] - outputs[i, true] - baseline[pred] + baseline[true] - margin_values.sum()),
                          'margin_top_features': [{'feature': features[j], 'signed_attribution': float(margin_values[j]),
                                                   'standardized_input': float(selected_x[i, j])} for j in ordered] if true != pred else []})
        result.update(cases=cases, checkpoint_sha256=reports[key]['checkpoint_sha256'],
                      full_validation_metrics=reports[key]['validation_metrics'])
        write(args.output / f'{key}.json', result)
        np.savez_compressed(args.output / f'{key}.npz', values=mean_values, repeat_0=values[0], repeat_1=values[1], logits=outputs, reference_logits=baseline)
        results[key] = result
    stability = {}
    for lane in study['lanes']:
        stability[lane] = {}
        for name in CLASSES:
            pairs = [overlap(np.array(results[f'{lane}-seed-{a}']['classes'][name]['mean_abs_attributions']),
                             np.array(results[f'{lane}-seed-{b}']['classes'][name]['mean_abs_attributions']))
                     for a, b in itertools.combinations((0, 1, 2), 2)]
            stability[lane][name] = {'pairwise_top5_overlap_01_02_12': pairs, 'mean_top5_overlap': float(np.mean(pairs))}
    payload = {'method': 'SHAP GradientExplainer expected gradients on logits', 'classes': CLASSES,
               'feature_columns': features, 'nsamples_per_repeat': args.nsamples, 'mc_seeds': [0, 1],
               'core_count': len(core), 'supplement_count': len(rows) - len(core), 'background_count': len(background),
               'test_evaluated': False, 'confirmation_sha256': sha(args.confirmation), 'plan_sha256': sha(args.plan),
               'split_sha256': split_hashes, 'code_sha256': sha(__file__),
               'packages': {p: importlib.metadata.version(p) for p in ('shap', 'torch', 'numpy', 'polars', 'scikit-learn')},
               'stability': stability, 'results': results,
               'limitations': ['Balanced core is not population-weighted; error-enriched supplement is separate.',
                              'Small shared background/sample; correlated features and off-manifold interpolations limit interpretation.',
                              'Logit attribution is model behavior, not calibrated probability, causality, thresholds or device vulnerability.',
                              'Monte Carlo residual/repeat diagnostics must qualify every interpretation; no automated policy deployment.']}
    write(args.output / 'summary.json', payload)
    print(f'Completed explanation audit: {args.output}', flush=True)


if __name__ == '__main__':
    main()
