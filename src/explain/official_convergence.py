"""Compare 512 and 2048 expected-gradient draws on an unchanged small pilot.

Reuse the saved, byte-pinned pilot inputs rather than resample or reopen the
large dataset. The packed manifest is metadata only. Every model and baseline
array is verified before any new attribution; no test data is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.explain import official
from src.models.research import atomic_json

require = official.require
FLAGS = ('test_evaluated', 'model_trained', 'model_promoted', 'deployment_authorized',
         'full_attribution_run', 'feature_interpretation_authorized')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_arrays(path):
    # Context management matters on Windows: do not retain an open ZIP handle.
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def validate_plan(plan, previous):
    require(plan['protocol'] == 'official39-explanation-convergence-v1', 'Unknown convergence protocol')
    require(previous['protocol'] == 'official39-explanation-cpu-reference-v2', 'Frozen CPU preparation required')
    require(plan['baseline_nsamples'] == previous['settings']['pilot_nsamples'] == 512
            and plan['comparison_nsamples'] == 2048, 'Unreviewed sample budget')
    require(plan['mc_seeds'] == previous['settings']['mc_seeds'] == [71, 72], 'Monte Carlo seeds changed')
    require(plan['pilot_rows'] == 8 and plan['background_rows'] == 128, 'Pilot scope changed')
    require(plan['quality_limits'] == previous['quality_limits'], 'Numerical criteria changed')
    require(all(plan[flag] is False for flag in FLAGS), 'Unreviewed action')


def array_quality(arrays, limits):
    require(set(arrays) == {'repeat_0', 'repeat_1', 'logits', 'reference_logits'}, 'Unexpected pilot arrays')
    require(arrays['logits'].shape == (8, 8) and arrays['reference_logits'].shape == (8,), 'Invalid logit shape')
    require(all(arrays[k].shape == (8, 39, 8) for k in ('repeat_0', 'repeat_1')), 'Invalid attribution shape')
    return official.numerical_quality([arrays['repeat_0'], arrays['repeat_1']],
                                      arrays['logits'], arrays['reference_logits'], limits)


def contrast_quality(arrays, labels, limits):
    """Explain a mistake's predicted-minus-true logit margin, not its probability.

    Difference of the two class attributions targets the difference of their
    centered scores. Do not include correct cases: their self-contrast is zero
    by construction and would artificially improve an aggregate check.
    """
    labels = np.asarray(labels)
    require(labels.shape == (8,) and np.issubdtype(labels.dtype, np.integer)
            and labels.min() >= 0 and labels.max() < 8, 'Invalid pilot labels')
    pred = arrays['logits'].argmax(1)
    rows = np.flatnonzero(pred != labels)
    if not len(rows):
        return dict(error_case_count=0, passing_case_count=0, cases=[],
                    mean_abs_residual_logits=None, relative_mean_abs_residual=None,
                    relative_repeat_difference=None, diagnostic_screen_passed=None)
    p, y = pred[rows], labels[rows]
    a, b = [arrays[k][rows, :, p]-arrays[k][rows, :, y] for k in ('repeat_0', 'repeat_1')]
    margin = arrays['logits'][rows, p]-arrays['logits'][rows, y]
    reference = arrays['reference_logits'][p]-arrays['reference_logits'][y]
    delta = margin-reference
    mean = (a+b)/2
    residual = delta-mean.sum(1)
    threshold = limits['case_abs_residual']+limits['case_relative_residual']*np.abs(delta)
    passed = np.abs(residual) <= threshold
    relative = float(np.abs(residual).mean()/max(float(np.abs(delta).mean()), 1e-12))
    disagreement = float(np.abs(a-b).mean()/max(float(np.abs(mean).mean()), 1e-12))
    cases = [dict(pilot_position=int(row), true_class=CLASSES[int(y[i])], predicted_class=CLASSES[int(p[i])],
                  logit_margin=float(margin[i]), background_margin=float(reference[i]),
                  centered_margin=float(delta[i]), residual_logits=float(residual[i]),
                  residual_limit=float(threshold[i]), residual_screen_passed=bool(passed[i]))
             for i, row in enumerate(rows)]
    return dict(error_case_count=len(rows), passing_case_count=int(passed.sum()), cases=cases,
                mean_abs_residual_logits=float(np.abs(residual).mean()), relative_mean_abs_residual=relative,
                relative_repeat_difference=disagreement,
                diagnostic_screen_passed=bool(passed.all()) and relative <= limits['relative_mean_abs_residual']
                and disagreement <= limits['relative_repeat_difference'])


def compare_arrays(old, new, labels, limits):
    require(np.array_equal(old['logits'], new['logits'])
            and np.array_equal(old['reference_logits'], new['reference_logits']), 'Pilot logits changed')
    before, after = (array_quality(a, limits) for a in (old, new))
    old_abs, new_abs = (np.abs(q['residuals']) for q in (before, after))
    threshold = limits['case_abs_residual']+limits['case_relative_residual']*np.abs(old['logits']-old['reference_logits'])
    old_pass, new_pass = old_abs <= threshold, new_abs <= threshold
    return dict(baseline=before, comparison=after,
                baseline_contrasts=contrast_quality(old, labels, limits),
                comparison_contrasts=contrast_quality(new, labels, limits),
                output_transitions=dict(failed_to_pass=int((~old_pass & new_pass).sum()),
                                        passed_to_fail=int((old_pass & ~new_pass).sum()),
                                        still_failed=int((~old_pass & ~new_pass).sum()),
                                        still_passed=int((old_pass & new_pass).sum())),
                absolute_residual_improved_outputs=int((new_abs < old_abs).sum()),
                absolute_residual_worsened_outputs=int((new_abs > old_abs).sum()),
                exact_pilot_logits=True)


def run(preparation_root, archive_root, manifest_path, preparation_plan_path, plan_path, output):
    root, output = Path(preparation_root), Path(output)
    require(not output.exists(), 'Refusing to overwrite convergence evidence')
    plan, previous = read_json(plan_path), read_json(preparation_plan_path)
    require(sha256(preparation_plan_path) == plan['preparation_plan_sha256'], 'Preparation plan checksum changed')
    require(sha256(root/'receipt.json') == plan['preparation_receipt_sha256'], 'Preparation receipt checksum changed')
    validate_plan(plan, previous)
    receipt = read_json(root/'receipt.json')
    manifest = read_json(manifest_path)
    require(sha256(manifest_path) == previous['packed_manifest_sha256'] == receipt['packed_manifest_sha256'],
            'Packed manifest changed')
    official.validate_plan(previous, manifest)
    require(receipt['plan_sha256'] == sha256(preparation_plan_path)
            and receipt['cpu_reference_sha256'] == previous['cpu_reference_sha256']
            and receipt['status'] in ('complete', 'numerical_review_required')
            and receipt['device'] == 'cpu' and receipt['classes'] == CLASSES
            and receipt['features'] == previous['features']
            and receipt['pilot_rows'] == 8 and receipt['background_rows'] == 128
            and receipt['quality_limits'] == plan['quality_limits']
            and all(receipt[flag] is False for flag in FLAGS), 'Preparation provenance mismatch')
    packages = {p: importlib.metadata.version(p) for p in ('torch', 'shap', 'numpy')}
    require(packages == receipt['packages'], 'Explanation package versions changed')
    source = Path(official.__file__).read_bytes()
    require(receipt['source_sha256']['explain/official.py'] in
            [hashlib.sha256(b).hexdigest() for b in (source, source.replace(b'\r\n', b'\n'))], 'Pilot implementation changed')
    for name, key in (('inputs.npz', 'inputs_sha256'), ('sampling.json', 'sampling_sha256')):
        require(sha256(root/name) == receipt[key], 'Frozen inputs or sampling changed')
    inputs, sampling = read_arrays(root/'inputs.npz'), read_json(root/'sampling.json')
    require(set(inputs) == {'background', 'x', 'pilot_x', 'y', 'validation_rows'}, 'Unexpected input arrays')
    require(inputs['background'].shape == (128, 39) and inputs['pilot_x'].shape == (8, 39)
            and inputs['x'].shape == (len(sampling['validation_rows']), 39)
            and inputs['y'].shape == (len(inputs['x']),)
            and inputs['validation_rows'].tolist() == sampling['validation_rows'], 'Input shape/order changed')
    require(all(inputs[k].dtype == np.float32 and np.isfinite(inputs[k]).all()
                for k in ('background', 'pilot_x', 'x')), 'Invalid frozen features')
    positions = [sampling['validation_rows'].index(row) for row in sampling['pilot_validation_rows']]
    require(len(positions) == len(set(positions)) == 8
            and np.array_equal(inputs['x'][positions], inputs['pilot_x']), 'Pilot is not the frozen core subset')
    labels = inputs['y'][positions]
    require(labels.tolist() == list(range(8)), 'Pilot must contain one case per canonical class')
    roster = {spec['id'] for spec in previous['models']}
    require(set(receipt['pilot']) == set(receipt['bridges']) == roster, 'Missing baseline model')
    torch.set_num_threads(previous['settings']['threads'])
    torch.use_deterministic_algorithms(True)
    models, baselines = {}, {}
    # Check every saved model and every baseline file before the first SHAP call.
    # This does NOT rerun full validation; it inherits the separately pinned
    # full-validation CPU reference and verifies the exact small-pilot logits.
    for spec in previous['models']:
        key = spec['id']
        bridge = receipt['bridges'][key]
        require(bridge['explanation_reference_exact'] is True
                and bridge['checkpoint_sha256'] == spec['files_sha256']['last.pt'], 'Checkpoint bridge changed')
        path = root/f'{key}-pilot.npz'
        require(sha256(path) == receipt['pilot'][key]['arrays_sha256'], 'Baseline attribution checksum changed')
        arrays = read_arrays(path)
        qa = array_quality(arrays, plan['quality_limits'])
        require(qa == {k: v for k, v in receipt['pilot'][key].items() if k != 'arrays_sha256'}, 'Baseline numerical checks changed')
        model, historical = official.load_frozen(spec, archive_root, previous, manifest)
        require(historical == bridge['historical_validation_metrics'], 'Historical model metrics changed')
        with torch.no_grad():
            logits = model(torch.from_numpy(inputs['pilot_x'])).numpy()
            baseline = model(torch.from_numpy(inputs['background'])).numpy().mean(0)
        require(np.array_equal(logits, arrays['logits']) and np.array_equal(baseline, arrays['reference_logits']),
                f'Exact pilot logit bridge failed: {key}')
        expected_pred = np.asarray(sampling['selected_predictions'][key])[positions]
        require(np.array_equal(logits.argmax(1), expected_pred), 'Pilot prediction reference changed')
        models[key], baselines[key] = model, arrays
    output.mkdir(parents=True, exist_ok=False)
    result = dict(protocol=plan['protocol'], status='incomplete', plan_sha256=sha256(plan_path),
                  preparation_plan_sha256=sha256(preparation_plan_path),
                  preparation_receipt_sha256=sha256(root/'receipt.json'),
                  inputs_sha256=receipt['inputs_sha256'], sampling_sha256=receipt['sampling_sha256'],
                  packed_manifest_sha256=receipt['packed_manifest_sha256'], cpu_reference_sha256=receipt['cpu_reference_sha256'],
                  source_sha256={'explain/official_convergence.py': sha256(__file__),
                                 'explain/official.py': sha256(official.__file__)},
                  device='cpu', platform=platform.platform(), packages=packages, classes=CLASSES,
                  baseline_nsamples=512, comparison_nsamples=2048, mc_seeds=plan['mc_seeds'],
                  pilot_rows=8, background_rows=128, quality_limits=plan['quality_limits'],
                  full_validation_rerun=False, models={}, **dict.fromkeys(FLAGS, False))
    atomic_json(result, output/'receipt.json')
    settings = dict(previous['settings'], pilot_nsamples=plan['comparison_nsamples'])
    for key, model in models.items():
        print(f'{key}: starting 2048-draw comparison', flush=True)
        qa, arrays = official.pilot(model, inputs['background'], inputs['pilot_x'], settings, plan['quality_limits'])
        comparison = compare_arrays(baselines[key], arrays, labels, plan['quality_limits'])
        require(qa == comparison['comparison'], 'Comparison numerical checks disagree')
        np.savez_compressed(output/f'{key}-2048.npz', **arrays)
        result['models'][key] = dict(comparison,
            checkpoint_sha256=receipt['bridges'][key]['checkpoint_sha256'],
            baseline_arrays_sha256=receipt['pilot'][key]['arrays_sha256'],
            comparison_arrays_sha256=sha256(output/f'{key}-2048.npz'))
        atomic_json(result, output/'receipt.json')
        print(f"{key}: {qa['passing_output_count']}/64 local residual checks pass", flush=True)
    result['all_comparison_screens_passed'] = all(m['comparison']['numerical_screen_passed'] for m in result['models'].values())
    result['status'] = 'complete' if result['all_comparison_screens_passed'] else 'numerical_review_required'
    result['next'] = ('Review pilot before defining full-attribution scope' if result['all_comparison_screens_passed']
                      else 'Review integration or attribution method; do not expand sample budget or interpret features automatically')
    atomic_json(result, output/'receipt.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation-root', 'archive-root', 'manifest-path', 'preparation-plan-path', 'plan-path', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
