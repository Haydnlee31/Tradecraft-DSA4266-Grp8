"""Bounded deterministic integration on the frozen official-39 explanation pilot.

No model changes or SHAP fallbacks. Average integrated gradients over EVERY
original background row. Keep quadrature convergence, Monte Carlo disagreement,
and model accuracy distinct. Raw per-feature values stay in ignored NPZ files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import platform

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.explain import official, official_convergence as mc
from src.models.research import atomic_json

require = official.require


def quadrature(segments):
    """32-point Gauss-Legendre on each equal-width segment of [0, 1].

    Composite rules keep the polynomial degree below NumPy's documented
    tested limit and refine the path partition, rather than use degree 128.
    They do not guarantee exact integration through neural-network kinks.
    """
    require(type(segments) is int and segments in (2, 4), 'Only the two reviewed resolutions are supported')
    nodes, weights = np.polynomial.legendre.leggauss(32)
    alpha = np.concatenate([(nodes+1)/(2*segments)+i/segments for i in range(segments)])
    weight = np.tile(weights/(2*segments), segments)
    require(np.isfinite(alpha).all() and (np.diff(alpha) > 0).all()
            and (alpha > 0).all() and (alpha < 1).all()
            and (weight > 0).all() and abs(weight.sum()-1) < 1e-12, 'Invalid quadrature rule')
    return alpha, weight


def integrate(model, background, x, segments, batch_size=128):
    """Stream path batches; gradients use unchanged float32 weights.

    For each background b: (x-b) * integral grad f(b + alpha*(x-b)) d alpha.
    Sum in float64 then divide by the exact background count. Do NOT adjust
    feature values to force completeness. autograd.grad does not train weights.
    The reviewed eval-mode MLP is pointwise; summing batch outputs therefore
    yields independent input gradients, not a batch-coupled attribution.
    """
    require(not any(m.training for m in model.modules()), 'Evaluation mode required')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()), 'Unchanged float32 CPU model required')
    require(x.ndim == background.ndim == 2 and x.shape[1] == background.shape[1] == 39
            and 0 < len(x) <= 8 and 0 < len(background) <= 128
            and x.dtype == background.dtype == np.float32
            and np.isfinite(x).all() and np.isfinite(background).all(), 'Invalid integration inputs')
    require(type(batch_size) is int and 1 <= batch_size <= 128, 'Gradient batch cap exceeded')
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    alpha, weight = quadrature(segments)
    n, b = len(alpha), len(background)
    values = np.zeros((len(x), 39, len(CLASSES)), dtype=np.float64)
    for row, case in enumerate(x):
        delta = case-background
        for start in range(0, b*n, batch_size):
            flat = np.arange(start, min(start+batch_size, b*n))
            bg, node = flat//n, flat % n
            # Float32 interpolation/gradients, float64 weighted accumulation.
            points = background[bg]+alpha[node, None].astype(np.float32)*delta[bg]
            tensor = torch.from_numpy(points).requires_grad_(True)
            scores = model(tensor)
            require(scores.shape == (len(flat), len(CLASSES)) and bool(torch.isfinite(scores).all()), 'Invalid path scores')
            factor = delta[bg].astype(np.float64)*weight[node, None]/b
            for c in range(len(CLASSES)):
                gradient = torch.autograd.grad(scores[:, c].sum(), tensor,
                                               retain_graph=c < len(CLASSES)-1)[0].detach().numpy()
                require(np.isfinite(gradient).all(), 'Nonfinite input gradient')
                values[row, :, c] += (gradient.astype(np.float64)*factor).sum(0)
    require(all(torch.equal(v, model.state_dict()[k]) for k, v in state.items()), 'Integration mutated model state')
    require(all((old is None and p.grad is None) or (old is not None and p.grad is not None and torch.equal(old, p.grad))
                for old, p in zip(grads, model.parameters())), 'Integration accumulated parameter gradients')
    require(np.isfinite(values).all(), 'Nonfinite integrated attribution')
    return values


def completeness(values, delta, limits):
    """Original residual tolerances, without inventing Monte Carlo repeats."""
    require(values.ndim == delta.ndim+1 and values.shape[:1]+values.shape[2:] == delta.shape
            and values.shape[1] == 39 and delta.size > 0
            and np.isfinite(values).all() and np.isfinite(delta).all(), 'Invalid completeness arrays')
    residual = delta-values.sum(1)
    threshold = limits['case_abs_residual']+limits['case_relative_residual']*np.abs(delta)
    passed = np.abs(residual) <= threshold
    relative = float(np.abs(residual).mean()/max(float(np.abs(delta).mean()), 1e-12))
    return dict(mean_abs_residual_logits=float(np.abs(residual).mean()),
                maximum_abs_residual_logits=float(np.abs(residual).max()), relative_mean_abs_residual=relative,
                passing_output_count=int(passed.sum()), output_count=int(passed.size),
                completeness_screen_passed=bool(passed.all()) and relative <= limits['relative_mean_abs_residual'],
                residuals=residual.tolist(), residual_limits=threshold.tolist())


def agreement(coarse, fine, limits=None):
    """Featurewise difference, so errors cancelling in the sum remain visible.

    Denominators use fine-grid attribution magnitude, not logit magnitude.
    A small epsilon handles all-zero attributions without certifying uncertainty.
    No pass/fail is assigned to the descriptive Monte Carlo comparison.
    """
    require(coarse.shape == fine.shape and coarse.ndim in (2, 3) and coarse.shape[1] == 39
            and coarse.size > 0 and np.isfinite(coarse).all() and np.isfinite(fine).all(), 'Invalid agreement arrays')
    difference = np.abs(coarse-fine)
    ratios = difference.sum(1)/np.maximum(np.abs(fine).sum(1), 1e-12)
    relative = float(difference.mean()/max(float(np.abs(fine).mean()), 1e-12))
    return dict(mean_abs_difference=float(difference.mean()), maximum_abs_feature_difference=float(difference.max()),
                relative_mean_abs_difference=relative, maximum_per_output_relative_l1_difference=float(ratios.max()),
                per_output_relative_l1_difference=ratios.tolist(),
                resolution_screen_passed=None if limits is None else
                relative <= limits['relative_mean_abs_difference'] and bool((ratios <= limits['per_output_relative_l1_difference']).all()))


def diagnose(coarse, fine, baseline, labels, plan):
    logits, ref = baseline['logits'], baseline['reference_logits']
    limits, resolution = plan['completeness_limits'], plan['resolution_limits']
    scores = {str(nodes): completeness(values, logits-ref, limits) for nodes, values in ((64, coarse), (128, fine))}
    result = dict(scores=scores, resolution_agreement=agreement(coarse, fine, resolution),
                  versus_mc2048=agreement((baseline['repeat_0']+baseline['repeat_1'])/2, fine),
                  error_margins={})
    pred = logits.argmax(1)
    rows = np.flatnonzero(pred != labels)
    if len(rows):
        p, y = pred[rows], labels[rows]
        delta = (logits[rows, p]-logits[rows, y])-(ref[p]-ref[y])
        a, b = (v[rows, :, p]-v[rows, :, y] for v in (coarse, fine))
        result['error_margins'] = dict(count=len(rows), pilot_positions=rows.tolist(),
            true_classes=[CLASSES[int(c)] for c in y], predicted_classes=[CLASSES[int(c)] for c in p],
            coarse=completeness(a, delta, limits), fine=completeness(b, delta, limits),
            resolution_agreement=agreement(a, b, resolution))
    else:
        result['error_margins'] = dict(count=0, coarse=None, fine=None, resolution_agreement=None)
    margin_ok = not len(rows) or (result['error_margins']['coarse']['completeness_screen_passed']
        and result['error_margins']['fine']['completeness_screen_passed']
        and result['error_margins']['resolution_agreement']['resolution_screen_passed'])
    result['method_screen_passed'] = all(q['completeness_screen_passed'] for q in scores.values()) and \
        result['resolution_agreement']['resolution_screen_passed'] and margin_ok
    return result


def validate_plan(plan, previous):
    require(plan['protocol'] == 'official39-explanation-integration-v1', 'Unknown integration protocol')
    require(plan['settings'] == dict(method='background_averaged_composite_gauss_legendre_integrated_gradients',
        nodes_per_segment=32, segments=[2, 4], path_nodes=[64, 128], gradient_batch_size=128, threads=2,
        pilot_rows=8, background_rows=128, model_dtype='float32', accumulator_dtype='float64'), 'Integration scope changed')
    require(plan['completeness_limits'] == {k: v for k, v in previous['quality_limits'].items()
                                           if k != 'relative_repeat_difference'}, 'Original residual thresholds changed')
    require(plan['resolution_limits'] == dict(relative_mean_abs_difference=.10, per_output_relative_l1_difference=.25),
            'Resolution thresholds changed')
    require(all(plan[k] is False for k in mc.FLAGS), 'Unreviewed action')


def run(preparation_root, convergence_root, archive_root, manifest_path, preparation_plan_path,
        convergence_plan_path, plan_path, output):
    output, prep_root, conv_root = Path(output), Path(preparation_root), Path(convergence_root)
    require(not output.exists(), 'Refusing to overwrite integration evidence')
    plan, conv, prep = map(mc.read_json, (plan_path, convergence_plan_path, preparation_plan_path))
    require(sha256(convergence_plan_path) == plan['convergence_plan_sha256']
            and sha256(conv_root/'receipt.json') == plan['convergence_receipt_sha256'], 'Convergence evidence changed')
    validate_plan(plan, conv)
    mc.validate_plan(conv, prep)
    require(sha256(preparation_plan_path) == conv['preparation_plan_sha256']
            and sha256(prep_root/'receipt.json') == conv['preparation_receipt_sha256'], 'Preparation evidence changed')
    prior, prepared, manifest = map(mc.read_json, (conv_root/'receipt.json', prep_root/'receipt.json', manifest_path))
    require(sha256(manifest_path) == prior['packed_manifest_sha256'] == prep['packed_manifest_sha256'], 'Manifest changed')
    official.validate_plan(prep, manifest)
    packages = {p: importlib.metadata.version(p) for p in ('torch', 'shap', 'numpy')}
    require(packages == prior['packages'] == prepared['packages'], 'Package versions changed')
    for module in (official, mc):
        name = 'explain/'+Path(module.__file__).name
        raw = Path(module.__file__).read_bytes()
        require(prior['source_sha256'][name] in [hashlib.sha256(v).hexdigest() for v in (raw, raw.replace(b'\r\n', b'\n'))],
                'Previous calculation source changed')
    for r in (prior, prepared):
        require(r['status'] in ('complete', 'numerical_review_required') and r['device'] == 'cpu'
                and r['classes'] == CLASSES and all(r[k] is False for k in mc.FLAGS), 'Invalid prior scope')
        for name, key in (('inputs.npz', 'inputs_sha256'), ('sampling.json', 'sampling_sha256')):
            require(sha256(prep_root/name) == r[key], 'Frozen pilot inputs changed')
    require(prior['plan_sha256'] == sha256(convergence_plan_path)
            and prior['preparation_plan_sha256'] == prepared['plan_sha256'] == sha256(preparation_plan_path)
            and prior['preparation_receipt_sha256'] == sha256(prep_root/'receipt.json')
            and prior['cpu_reference_sha256'] == prepared['cpu_reference_sha256'] == prep['cpu_reference_sha256'], 'Reference chain changed')
    inputs, sampling = mc.read_arrays(prep_root/'inputs.npz'), mc.read_json(prep_root/'sampling.json')
    x, background = inputs['pilot_x'], inputs['background']
    positions = [sampling['validation_rows'].index(row) for row in sampling['pilot_validation_rows']]
    labels = inputs['y'][positions]
    require(x.shape == (8, 39) and background.shape == (128, 39)
            and np.array_equal(inputs['x'][positions], x) and labels.tolist() == list(range(8))
            and inputs['validation_rows'].tolist() == sampling['validation_rows'], 'Frozen pilot shape/order changed')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    models, baselines = {}, {}
    require(set(prior['models']) == {s['id'] for s in prep['models']}, 'Model roster changed')
    # Verify ALL saved models, Monte Carlo arrays and their diagnostics before
    # integrating the first one. No full dataset feature array is reopened.
    for spec in prep['models']:
        key = spec['id']
        record = prior['models'][key]
        path = conv_root/f'{key}-2048.npz'
        require(sha256(path) == record['comparison_arrays_sha256'], 'Monte Carlo arrays changed')
        arrays = mc.read_arrays(path)
        require(mc.array_quality(arrays, conv['quality_limits']) == record['comparison']
                and mc.contrast_quality(arrays, labels, conv['quality_limits']) == record['comparison_contrasts'],
                'Prior numerical checks changed')
        require(record['checkpoint_sha256'] == prepared['bridges'][key]['checkpoint_sha256'] == spec['files_sha256']['last.pt']
                and prepared['bridges'][key]['explanation_reference_exact'] is True, 'Checkpoint reference changed')
        model, historical = official.load_frozen(spec, archive_root, prep, manifest)
        require(historical == prepared['bridges'][key]['historical_validation_metrics'], 'Historical measurements changed')
        with torch.no_grad():
            logits = model(torch.from_numpy(x)).numpy()
            ref = model(torch.from_numpy(background)).numpy().mean(0)
        require(np.array_equal(logits, arrays['logits']) and np.array_equal(ref, arrays['reference_logits'])
                and np.array_equal(logits.argmax(1), np.asarray(sampling['selected_predictions'][key])[positions]),
                'Exact pilot logit bridge failed')
        models[key], baselines[key] = model, arrays
    output.mkdir(parents=True, exist_ok=False)
    result = dict(protocol=plan['protocol'], status='incomplete', plan_sha256=sha256(plan_path),
        convergence_receipt_sha256=sha256(conv_root/'receipt.json'), preparation_receipt_sha256=sha256(prep_root/'receipt.json'),
        inputs_sha256=sha256(prep_root/'inputs.npz'), sampling_sha256=sha256(prep_root/'sampling.json'),
        packed_manifest_sha256=sha256(manifest_path), cpu_reference_sha256=prior['cpu_reference_sha256'],
        source_sha256={'explain/official_integration.py': sha256(__file__),
                      'explain/official.py': sha256(official.__file__), 'explain/official_convergence.py': sha256(mc.__file__)},
        packages=packages, platform=platform.platform(), device='cpu', classes=CLASSES, settings=plan['settings'],
        completeness_limits=plan['completeness_limits'], resolution_limits=plan['resolution_limits'],
        full_validation_rerun=False, models={}, **dict.fromkeys(mc.FLAGS, False))
    atomic_json(result, output/'receipt.json')
    for key, model in models.items():
        print(f'{key}: deterministic 64-point integration', flush=True)
        coarse = integrate(model, background, x, 2)
        print(f'{key}: deterministic 128-point integration', flush=True)
        fine = integrate(model, background, x, 4)
        summary = diagnose(coarse, fine, baselines[key], labels, plan)
        np.savez_compressed(output/f'{key}-integration.npz', coarse=coarse, fine=fine,
                            logits=baselines[key]['logits'], reference_logits=baselines[key]['reference_logits'])
        result['models'][key] = dict(summary, checkpoint_sha256=prior['models'][key]['checkpoint_sha256'],
            monte_carlo_arrays_sha256=prior['models'][key]['comparison_arrays_sha256'],
            arrays_sha256=sha256(output/f'{key}-integration.npz'), path_forward_rows=196608, output_gradient_rows=1572864)
        atomic_json(result, output/'receipt.json')
        print(f"{key}: fine residual checks {summary['scores']['128']['passing_output_count']}/64; method screen={summary['method_screen_passed']}", flush=True)
    result['all_method_screens_passed'] = all(m['method_screen_passed'] for m in result['models'].values())
    result['status'] = 'complete' if result['all_method_screens_passed'] else 'numerical_review_required'
    result['next'] = ('Review method and define a full-run protocol; no feature interpretation yet' if result['all_method_screens_passed']
                      else 'Keep warnings; review unresolved numerical error without automatically expanding the calculation')
    atomic_json(result, output/'receipt.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation-root', 'convergence-root', 'archive-root', 'manifest-path', 'preparation-plan-path',
                 'convergence-plan-path', 'plan-path', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
