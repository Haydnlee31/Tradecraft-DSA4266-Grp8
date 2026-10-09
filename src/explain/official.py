"""Prepare official-39 error explanations and run a bounded numerical pilot.

This is NOT training, a threshold search, or a finished SHAP interpretation.
Version 1 requires historical GPU confusion parity. Version 2 explicitly freezes
a separate CPU explanation reference after a reported v1 mismatch; it requires
exact repeat of every CPU prediction, not a relaxed GPU tolerance. The original
46-feature explanation path and all model/decision defaults remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import tarfile

import numpy as np
import torch

from src.data.label_map import BENIGN, CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.eval.official_evidence import close_tree, metrics_from_counts, safe_path
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.research import atomic_json

ROSTER = [(lane, seed) for lane in ('light', 'iid') for seed in (7, 17, 27)] + [('dirichlet', 7)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_member(archive, name, limit=10_000_000):
    """Read explicitly pinned members, never extract an archive onto disk."""
    require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe archive member')
    matches = [m for m in archive.getmembers() if m.name == name]
    require(len(matches) == 1 and matches[0].isfile(), 'Missing/duplicate/non-file archive member')
    require(matches[0].size <= limit, 'Oversized checkpoint or receipt')
    return archive.extractfile(matches[0]).read()


def validate_plan(plan, manifest):
    protocols = {'official39-explanation-preparation-v1': 'exact',
                 'official39-explanation-cpu-reference-v2': 'frozen_cpu_reference'}
    require(plan['protocol'] in protocols, 'Unknown protocol')
    require(plan['classes'] == manifest['classes'] == CLASSES, 'Class order mismatch')
    require([(s['lane'], s['seed']) for s in plan['models']] == ROSTER, 'Frozen seven-model roster required')
    require(len({s['id'] for s in plan['models']}) == len(ROSTER), 'Duplicate model ID')
    require(all(s['id'] == f"{s['lane']}-seed{s['seed']}" for s in plan['models']), 'Unsafe or inconsistent model ID')
    require(manifest['status'] == 'complete' and manifest['test_opened'] is False, 'Completed train/val pack required')
    require(len(manifest['features']) == len(set(manifest['features'])) == 39, 'Official 39-feature schema required')
    require(manifest['features'] == plan['features'], 'Feature order mismatch')
    require(manifest['train_rows'] == plan['train_rows'] and manifest['val_rows'] == plan['validation_rows'], 'Row counts changed')
    require(0 < plan['train_rows'] <= 2_000_000 and 0 < plan['validation_rows'] <= 2_100_000, 'Data cap exceeded')
    require(manifest['train_class_counts'] == plan['train_class_counts'], 'Training class counts changed')
    require(set(plan['validation_class_counts']) == set(CLASSES)
            and all(type(n) is int and n > 0 for n in plan['validation_class_counts'].values())
            and sum(plan['validation_class_counts'].values()) == plan['validation_rows'], 'Validation counts invalid')
    expected = dict(core_per_class=8, background_rows=128, sample_seed=42663907,
                    error_rows_per_group=1, maximum_explanation_rows=225,
                    pilot_rows_per_class=1, pilot_nsamples=512, mc_seeds=[71, 72],
                    inference_batch_size=512, gradient_batch_size=128, threads=2,
                    checkpoint_step=20, confusion_parity=protocols[plan['protocol']], method='expected_gradients_logits')
    require(plan['settings'] == expected, 'Frozen preparation budget or method changed')
    for flag in ('test_evaluated', 'model_trained', 'model_promoted', 'deployment_authorized', 'full_attribution_run'):
        require(plan[flag] is False, f'Unexpected action: {flag}')
    require(plan['quality_limits'] == dict(relative_mean_abs_residual=.10,
                                           relative_repeat_difference=.25,
                                           case_abs_residual=.25, case_relative_residual=.10), 'QA limits changed')


def load_frozen(spec, archive_root, plan, manifest):
    archive_spec = plan['archives'][spec['archive']]
    path = safe_path(archive_root, archive_spec['file'])
    require(sha256(path) == archive_spec['sha256'], 'Archive checksum mismatch')
    with tarfile.open(path) as archive:
        raw = {name: read_member(archive, spec['member_root']+'/'+name)
               for name in ('environment.json', 'status.json', 'result.json', 'last.pt')}
    for name, content in raw.items():
        require(hashlib.sha256(content).hexdigest() == spec['files_sha256'][name], f'Changed model artifact: {name}')
    env, status, result = (json.loads(raw[n]) for n in ('environment.json', 'status.json', 'result.json'))
    require(status == {'status': 'complete', 'step': 20} and result['test_metrics'] is None
            and env['test_evaluated'] is False, 'Completed validation-only run required')
    require(env['classes'] == CLASSES and env['features'] == manifest['features']
            and env['packed_manifest_sha256'] == plan['packed_manifest_sha256'], 'Model/data provenance mismatch')
    require(env['settings'] == dict(plan['training_settings'], lane=spec['lane'], seed=spec['seed']), 'Training recipe changed')
    require(env['model_config'] == plan['model_config'], 'Architecture mismatch')
    # Only inference-critical sources must match the training snapshot. Additive
    # report code may evolve. LF normalization permits Windows text checkouts,
    # but does not waive any executable-source difference.
    for name in ('models/architectures.py', 'data/label_map.py'):
        source = Path(__file__).parents[1]/name
        raw_source = source.read_bytes()
        variants = (raw_source, raw_source.replace(b'\r\n', b'\n'))
        require(env['source_sha256'][name] in [hashlib.sha256(v).hexdigest() for v in variants], 'Inference source changed')
    # Verify the bytes BEFORE restricted deserialization. Never load best.pt:
    # its selection point differs from the final-at-budget comparison.
    saved = torch.load(io.BytesIO(raw['last.pt']), map_location='cpu', weights_only=True)
    require(saved['environment_sha256'] == spec['files_sha256']['environment.json']
            and saved['step'] == 20 and len(saved['history']) == 20
            and saved['history'] == result['history'], 'Checkpoint endpoint/history mismatch')
    require([h['step'] for h in saved['history']] == list(range(1, 21)), 'Incomplete history')
    expected = result['history'][-1]['validation_metrics']
    require(expected == result['final_validation_metrics'], 'Not the final endpoint')
    computed, _ = metrics_from_counts(expected['confusion_matrix'], plan['validation_class_counts'])
    close_tree(expected, computed)
    for key in ('examples_processed', 'optimizer_steps'):
        require(sum(h[key] for h in saved['history']) == result[key] == spec[key], 'Training work mismatch')
    require(result['num_parameters'] == 5096, 'Unexpected parameter count')
    require(saved['partition_sha256'] == spec['partition_sha256'], 'Partition provenance mismatch')
    if spec['lane'] == 'dirichlet':
        require(env['partition_control'] == spec['partition_control'], 'Frozen client ownership changed')
    model = MLPClassifier(39, len(CLASSES), MLPConfig(**env['model_config']))
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    require(model.num_parameters() == result['num_parameters'], 'Model size mismatch')
    return model, expected


def scan_validation(model, data, batch_size):
    """Only labels/predictions scale with row count; features stay disk-backed."""
    prediction = np.empty(len(data.arrays['val'][1]), dtype=np.int8)
    cm, offset = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64), 0
    with torch.inference_mode():
        for x, y in data.batches('val', batch_size):
            scores = model(torch.from_numpy(x)).numpy()
            require(np.isfinite(scores).all(), 'Nonfinite validation logits')
            pred = scores.argmax(1)
            prediction[offset:offset+len(y)] = pred
            np.add.at(cm, (y, pred), 1)
            offset += len(y)
    require(offset == len(prediction), 'Incomplete validation pass')
    return prediction, cm


def prediction_hash(prediction):
    """One byte per row in packed order; no dependence on native integer width."""
    return hashlib.sha256(np.asarray(prediction, dtype=np.uint8).tobytes()).hexdigest()


def audit_bridges(data_root, archive_root, plan_path, output):
    """Diagnose historical parity only; never invoke SHAP or select examples.

    A confusion-count difference gives a MINIMUM number of label reassignments.
    Archived per-row GPU predictions were not saved, so it cannot identify the
    exact differing flows or exclude cancelling per-row changes.
    """
    data_root, output = Path(data_root), Path(output)
    require(not output.exists(), 'Refusing to overwrite bridge evidence')
    plan = json.loads(Path(plan_path).read_text())
    require(sha256(data_root/'manifest.json') == plan['packed_manifest_sha256'], 'Packed manifest checksum mismatch')
    manifest = json.loads((data_root/'manifest.json').read_text())
    validate_plan(plan, manifest)
    data = PackedData(data_root)
    torch.set_num_threads(plan['settings']['threads'])
    torch.use_deterministic_algorithms(True)
    result = dict(status='complete', purpose='CPU/GPU confusion audit; no feature interpretation',
                  plan_sha256=sha256(plan_path), packed_manifest_sha256=plan['packed_manifest_sha256'],
                  script_sha256=sha256(__file__), device='cpu', platform=platform.platform(), classes=CLASSES,
                  inference_batch_size=plan['settings']['inference_batch_size'], threads=plan['settings']['threads'],
                  packages={p: importlib.metadata.version(p) for p in ('torch', 'numpy')},
                  test_evaluated=False, model_trained=False, attribution_run=False, models={})
    for spec in plan['models']:
        model, expected = load_frozen(spec, archive_root, plan, manifest)
        pred, cm = scan_validation(model, data, plan['settings']['inference_batch_size'])
        observed, _ = metrics_from_counts(cm.tolist(), plan['validation_class_counts'])
        delta = cm-np.asarray(expected['confusion_matrix'], dtype=np.int64)
        result['models'][spec['id']] = dict(checkpoint_sha256=spec['files_sha256']['last.pt'],
            historical_confusion_exact=not bool(delta.any()), cpu_validation_metrics=observed,
            historical_validation_metrics=expected, cpu_prediction_sha256=prediction_hash(pred),
            cpu_minus_historical_confusion=delta.tolist(),
            minimum_reassignments_per_true_class=dict(zip(CLASSES, map(int, np.abs(delta).sum(1)//2))))
        print(f"{spec['id']}: historical confusion exact={not bool(delta.any())}", flush=True)
    result['all_historical_confusions_exact'] = all(m['historical_confusion_exact'] for m in result['models'].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(result, output)
    return result


def freeze_rows(train_y, val_y, predictions, settings):
    """Shared balanced core plus a separately labelled union of error cases.

    Pick one row uniformly from each nonempty class/error group per model.
    Correct detections and absent groups are recorded too, including zero rare
    detections. Each group has an independent seed, so model iteration order
    cannot silently change another model's selection.
    """
    seed = settings['sample_seed']
    require(settings['error_rows_per_group'] == 1, 'Only the one-case supplement is reviewed')
    for labels in (train_y, val_y):
        require(np.issubdtype(labels.dtype, np.integer) and len(labels) > 0
                and labels.min() >= 0 and labels.max() < len(CLASSES), 'Invalid labels')
    core = []
    for c in range(len(CLASSES)):
        pool = np.flatnonzero(val_y == c)
        require(len(pool) >= settings['core_per_class'], 'Insufficient core support')
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, c]))
        core.extend(sorted(map(int, rng.choice(pool, settings['core_per_class'], replace=False))))
    groups, reasons = {}, {}
    benign = CLASSES.index(BENIGN)
    for key in sorted(predictions):
        pred = predictions[key]
        require(pred.shape == val_y.shape and np.issubdtype(pred.dtype, np.integer)
                and pred.min() >= 0 and pred.max() < len(CLASSES), 'Invalid predictions')
        key_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')
        groups[key] = {}
        for c, name in enumerate(CLASSES):
            masks = {'correct': pred == c}
            if c == benign:
                masks['false_alert'] = pred != benign
            else:
                masks.update(missed_as_benign=pred == benign,
                             wrong_attack=(pred != benign) & (pred != c))
            for index, (kind, mask) in enumerate(masks.items()):
                pool = np.flatnonzero((val_y == c) & mask)
                rng = np.random.default_rng(np.random.SeedSequence([seed, 1, key_seed, c, index]))
                row = int(rng.choice(pool)) if len(pool) else None
                group = name+'/'+kind
                groups[key][group] = dict(eligible_count=len(pool), selected_row=row)
                if row is not None:
                    reasons.setdefault(str(row), []).append(key+'/'+group)
    supplement = sorted(set(map(int, reasons))-set(core))
    rows = core+supplement
    require(len(rows) == len(set(rows)) and len(rows) <= settings['maximum_explanation_rows'], 'Explanation row cap exceeded')
    require(len(train_y) >= settings['background_rows'], 'Insufficient training background')
    background = sorted(map(int, np.random.default_rng(np.random.SeedSequence([seed, 2])).choice(
        len(train_y), settings['background_rows'], replace=False)))
    # Fixed first selected core row per class; not the easiest or cleanest case.
    pilot = [core[c*settings['core_per_class']] for c in range(len(CLASSES))]
    return dict(core_rows=core, supplement_rows=supplement, validation_rows=rows,
                background_train_rows=background, pilot_validation_rows=pilot,
                selection_reasons=reasons, error_groups=groups,
                background_class_counts=dict(zip(CLASSES, map(int, np.bincount(train_y[background], minlength=len(CLASSES))))))


def numerical_quality(repeats, outputs, baseline, limits):
    """Heuristic numerical screens, NOT statistical confidence or model gates.

    Averaging repeats does not make them independent background samples. Both
    repeat error and completeness are needed: two similar, biased numerical
    approximations must not receive an automatic interpretation certificate.
    """
    a, b = map(np.asarray, repeats)
    require(a.shape == b.shape and a.ndim == 3 and a.shape[0] == len(outputs)
            and a.shape[2] == outputs.shape[1] == len(CLASSES), 'Invalid attribution shape')
    require(all(np.isfinite(v).all() for v in (a, b, outputs, baseline)), 'Nonfinite attribution output')
    mean = (a+b)/2
    delta = outputs-baseline
    residual = delta-mean.sum(1)
    relative = float(np.abs(residual).mean()/max(float(np.abs(delta).mean()), 1e-12))
    repeat_difference = float(np.abs(a-b).mean()/max(float(np.abs(mean).mean()), 1e-12))
    ok_cases = np.abs(residual) <= limits['case_abs_residual']+limits['case_relative_residual']*np.abs(delta)
    passes = (relative <= limits['relative_mean_abs_residual']
              and repeat_difference <= limits['relative_repeat_difference'] and bool(ok_cases.all()))
    return dict(mean_abs_residual_logits=float(np.abs(residual).mean()),
                maximum_abs_residual_logits=float(np.abs(residual).max()),
                relative_mean_abs_residual=relative, relative_repeat_difference=repeat_difference,
                passing_output_count=int(ok_cases.sum()), output_count=int(ok_cases.size),
                numerical_screen_passed=passes, residuals=residual.tolist())


def pilot(model, background, x, settings, limits):
    import shap
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tensor, reference = torch.from_numpy(x), torch.from_numpy(background)
    with torch.no_grad():
        outputs = model(tensor).numpy()
        baseline = model(reference).numpy().mean(0)
    explainer = shap.GradientExplainer(model, reference, batch_size=settings['gradient_batch_size'], local_smoothing=0)
    repeats = [np.asarray(explainer.shap_values(tensor, nsamples=settings['pilot_nsamples'], rseed=s))
               for s in settings['mc_seeds']]
    require(all(v.shape == (len(x), 39, len(CLASSES)) for v in repeats), 'Unsupported SHAP multi-output shape')
    require(all(torch.equal(v, model.state_dict()[k]) for k, v in original.items()), 'Explanation mutated model state')
    qa = numerical_quality(repeats, outputs, baseline, limits)
    return qa, dict(repeat_0=repeats[0], repeat_1=repeats[1], logits=outputs, reference_logits=baseline)


def prepare(data_root, archive_root, plan_path, output, cpu_reference_path=None):
    data_root, output = Path(data_root), Path(output)
    require(not output.exists(), 'Refusing to overwrite preparation evidence')
    plan = json.loads(Path(plan_path).read_text())
    require(sha256(data_root/'manifest.json') == plan['packed_manifest_sha256'], 'Packed manifest checksum mismatch')
    manifest = json.loads((data_root/'manifest.json').read_text())
    validate_plan(plan, manifest)
    cpu_reference = None
    if plan['settings']['confusion_parity'] == 'frozen_cpu_reference':
        require(cpu_reference_path is not None and sha256(cpu_reference_path) == plan['cpu_reference_sha256'],
                'Frozen CPU reference checksum mismatch')
        cpu_reference = json.loads(Path(cpu_reference_path).read_text())
        require(cpu_reference['status'] == 'complete' and cpu_reference['device'] == 'cpu'
                and cpu_reference['test_evaluated'] is False and cpu_reference['model_trained'] is False
                and cpu_reference['attribution_run'] is False and cpu_reference['classes'] == CLASSES
                and cpu_reference['packed_manifest_sha256'] == plan['packed_manifest_sha256']
                and cpu_reference['plan_sha256'] == plan['superseded_plan_sha256']
                and cpu_reference['inference_batch_size'] == plan['settings']['inference_batch_size']
                and cpu_reference['threads'] == plan['settings']['threads']
                and set(cpu_reference['models']) == {s['id'] for s in plan['models']}, 'CPU reference provenance mismatch')
        require(cpu_reference['packages'] == {p: importlib.metadata.version(p) for p in ('torch', 'numpy')},
                'CPU reference package versions changed')
    else:
        require(cpu_reference_path is None, 'Version 1 cannot silently use a CPU reference')
    data = PackedData(data_root)  # Only train/val are supported, mmap_mode='r'.
    counts = {c: int((data.arrays['val'][1] == i).sum()) for i, c in enumerate(CLASSES)}
    require(counts == plan['validation_class_counts'], 'Validation class counts changed')
    require({c: int((data.arrays['train'][1] == i).sum()) for i, c in enumerate(CLASSES)}
            == plan['train_class_counts'], 'Training class counts changed')
    scaler = json.loads((data_root/'scaler.json').read_text())
    require(scaler['fit_split'] == 'train' and scaler['features'] == manifest['features']
            and scaler['n_samples_seen'] == manifest['train_rows'], 'Train-only scaler required')
    torch.set_num_threads(plan['settings']['threads'])
    torch.use_deterministic_algorithms(True)
    models, predictions, bridges = {}, {}, {}
    # Complete every identity/parity check BEFORE using selected examples or SHAP.
    # Version 1 retains its strict historical check; version 2 changes the stated
    # inference reference, preserves GPU results and uses no numerical tolerance.
    for spec in plan['models']:
        model, expected = load_frozen(spec, archive_root, plan, manifest)
        pred, matrix = scan_validation(model, data, plan['settings']['inference_batch_size'])
        historical_exact = matrix.tolist() == expected['confusion_matrix']
        if cpu_reference is None:
            require(historical_exact, f"Exact full-validation bridge failed: {spec['id']}")
            observed = expected
        else:
            ref = cpu_reference['models'][spec['id']]
            require(ref['checkpoint_sha256'] == spec['files_sha256']['last.pt']
                    and ref['historical_validation_metrics'] == expected, 'CPU reference checkpoint/history mismatch')
            observed, _ = metrics_from_counts(matrix.tolist(), plan['validation_class_counts'])
            close_tree(observed, ref['cpu_validation_metrics'])
            require(prediction_hash(pred) == ref['cpu_prediction_sha256'], 'Full CPU prediction replay mismatch')
        models[spec['id']], predictions[spec['id']] = model, pred
        bridges[spec['id']] = dict(explanation_reference_exact=True, historical_confusion_exact=historical_exact,
                                   cpu_validation_metrics=observed, historical_validation_metrics=expected,
                                   cpu_prediction_sha256=prediction_hash(pred),
                                   checkpoint_sha256=spec['files_sha256']['last.pt'])
        print(f"{spec['id']}: explanation reference exact; historical GPU confusion exact={historical_exact}", flush=True)
    sampling = freeze_rows(data.arrays['train'][1], data.arrays['val'][1], predictions, plan['settings'])
    x = np.array(data.arrays['val'][0][sampling['validation_rows']], copy=True)
    background = np.array(data.arrays['train'][0][sampling['background_train_rows']], copy=True)
    pilot_x = np.array(data.arrays['val'][0][sampling['pilot_validation_rows']], copy=True)
    require(all(np.isfinite(a).all() for a in (x, background, pilot_x)), 'Nonfinite sampled inputs')
    output.mkdir(parents=True, exist_ok=False)
    sampling['selected_predictions'] = {k: v[sampling['validation_rows']].tolist() for k, v in predictions.items()}
    atomic_json(sampling, output/'sampling.json')
    np.savez_compressed(output/'inputs.npz', background=background, x=x, pilot_x=pilot_x,
                        y=data.arrays['val'][1][sampling['validation_rows']],
                        validation_rows=np.asarray(sampling['validation_rows']))
    result = dict(status='pilot_incomplete', protocol=plan['protocol'], plan_sha256=sha256(plan_path),
                  packed_manifest_sha256=plan['packed_manifest_sha256'], source_sha256={
                      'explain/official.py': sha256(__file__), 'data/official_packed.py': sha256(Path(__file__).parents[1]/'data/official_packed.py')},
                  classes=CLASSES, features=manifest['features'], device='cpu', platform=platform.platform(),
                  packages={p: importlib.metadata.version(p) for p in ('torch', 'shap', 'numpy')},
                  cpu_reference_sha256=sha256(cpu_reference_path) if cpu_reference_path else None,
                  test_evaluated=False, model_trained=False, model_promoted=False, deployment_authorized=False,
                  full_attribution_run=False, feature_interpretation_authorized=False,
                  background_rows=len(background), core_rows=len(sampling['core_rows']),
                  supplement_rows=len(sampling['supplement_rows']), pilot_rows=len(pilot_x),
                  sampling_sha256=sha256(output/'sampling.json'), inputs_sha256=sha256(output/'inputs.npz'),
                  bridges=bridges, pilot={}, quality_limits=plan['quality_limits'])
    atomic_json(result, output/'receipt.json')
    for key, model in models.items():
        qa, arrays = pilot(model, background, pilot_x, plan['settings'], plan['quality_limits'])
        np.savez_compressed(output/f'{key}-pilot.npz', **arrays)
        result['pilot'][key] = {**qa, 'arrays_sha256': sha256(output/f'{key}-pilot.npz')}
        atomic_json(result, output/'receipt.json')
        print(f"{key}: numerical pilot {'PASS' if qa['numerical_screen_passed'] else 'CAUTION'}", flush=True)
    result['all_numerical_screens_passed'] = all(q['numerical_screen_passed'] for q in result['pilot'].values())
    result['status'] = 'complete' if result['all_numerical_screens_passed'] else 'numerical_review_required'
    result['next'] = ('Review preparation before full explanation; no full run launched' if result['all_numerical_screens_passed']
                      else 'Stop interpretation; review integration error and repeat sensitivity before any full explanation')
    atomic_json(result, output/'receipt.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'archive-root', 'plan-path', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--audit-bridge-only', action='store_true', help='JSON diagnostic only; no sampling/SHAP')
    parser.add_argument('--cpu-reference-path', type=Path, help='Required only for the explicit v2 CPU protocol')
    args = vars(parser.parse_args())
    audit = args.pop('audit_bridge_only')
    if audit:
        require(args.pop('cpu_reference_path') is None, 'Audit does not consume a CPU reference')
        audit_bridges(**args)
    else:
        prepare(**args)


if __name__ == '__main__':
    main()
