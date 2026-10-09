"""Bounded full explanation study using the unchanged deterministic pilot method.

Only the frozen 219-row snapshot is opened. Keep the balanced core and the
purposefully selected supplement separate. A failed numerical screen withholds
the whole stratum's rankings, not just inconvenient examples. This is model
behavior analysis, never training, test evaluation or policy deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from itertools import combinations
from pathlib import Path
import platform

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.explain import official, official_convergence as mc, official_integration as ig
from src.models.research import atomic_json

require = official.require


def validate_plan(plan, previous):
    require(plan['protocol'] == 'official39-explanation-study-v1', 'Unknown study protocol')
    require(plan['settings'] == dict(cases=219, core_cases=64, supplement_cases=155,
        background_rows=128, case_batch_size=8, path_nodes=[64, 128], gradient_batch_size=128, threads=2),
        'Frozen study scope changed')
    require(plan['work_budget'] == dict(models=7, path_forward_rows_per_model=219*128*192,
        path_forward_rows_total=7*219*128*192, output_gradient_rows_total=8*7*219*128*192), 'Work budget changed')
    require(plan['full_attribution_run'] is True and plan['descriptive_feature_summaries'] is True
        and all(plan[k] is False for k in mc.FLAGS if k != 'full_attribution_run'), 'Unreviewed study action')
    require(previous['protocol'] == 'official39-explanation-integration-v1', 'Wrong parent protocol')


def screen(coarse, fine, delta, plan):
    """Retain global AND per-output checks; expose individual failures."""
    a = ig.completeness(coarse, delta, plan['completeness_limits'])
    b = ig.completeness(fine, delta, plan['completeness_limits'])
    resolution = ig.agreement(coarse, fine, plan['resolution_limits'])
    mask = ((np.abs(a['residuals']) <= np.asarray(a['residual_limits']))
            & (np.abs(b['residuals']) <= np.asarray(b['residual_limits']))
            & (np.asarray(resolution['per_output_relative_l1_difference'])
               <= plan['resolution_limits']['per_output_relative_l1_difference']))
    return dict(coarse=a, fine=b, resolution=resolution, per_output_pass=mask.tolist(),
                passed=a['completeness_screen_passed'] and b['completeness_screen_passed']
                and resolution['resolution_screen_passed'])


def rank(values, features):
    """Absolute magnitude measures influence; signed mean retains direction.

    Neither quantity measures causal effect or classification correctness.
    Stable feature-order tie breaking also makes zero/tied fixtures reproducible.
    """
    require(values.ndim == 2 and values.shape[1] == len(features) and len(values) > 0
            and np.isfinite(values).all(), 'Invalid ranking values')
    absolute, signed = np.abs(values).mean(0), values.mean(0)
    order = np.argsort(-absolute, kind='stable')
    return dict(count=len(values), mean_absolute=absolute.tolist(), mean_signed=signed.tolist(),
                top_five=[features[i] for i in order[:5]])


def describe_stratum(coarse, fine, logits, reference, labels, rows, features, plan):
    """All cases remain in QA. Summaries are all-or-nothing for this stratum."""
    score = screen(coarse, fine, logits-reference, plan)
    pred = logits.argmax(1)
    errors = np.flatnonzero(pred != labels)
    margin, margins = None, None
    if len(errors):
        p, y = pred[errors], labels[errors]
        margins = fine[errors, :, p]-fine[errors, :, y]
        coarse_margins = coarse[errors, :, p]-coarse[errors, :, y]
        delta = (logits[errors, p]-logits[errors, y])-(reference[p]-reference[y])
        margin = screen(coarse_margins, margins, delta, plan)
    passed = score['passed'] and (margin is None or margin['passed'])
    case_pass = np.asarray(score['per_output_pass']).all(1)
    if margin is not None:
        case_pass[errors] &= np.asarray(margin['per_output_pass'])
    result = dict(count=len(rows), error_count=len(errors), scores=score, error_margins=margin,
        error_validation_rows=np.asarray(rows)[errors].tolist(), passed=passed,
        cases=[dict(validation_row=int(row), true_class=CLASSES[int(labels[i])],
                    predicted_class=CLASSES[int(pred[i])], individual_checks_passed=bool(case_pass[i]))
               for i, row in enumerate(rows)], descriptive=None)
    if passed:
        own = fine[np.arange(len(labels)), :, labels]
        result['descriptive'] = dict(own_true_score=rank(own, features), by_true_class={}, error_pairs={})
        for c, name in enumerate(CLASSES):
            selected = labels == c
            if selected.any():
                result['descriptive']['by_true_class'][name] = rank(fine[selected, :, c], features)
        for y, p in sorted(set(zip(labels[errors].tolist(), pred[errors].tolist()))):
            selected = (labels[errors] == y) & (pred[errors] == p)
            result['descriptive']['error_pairs'][CLASSES[y]+' -> '+CLASSES[p]] = rank(margins[selected], features)
    return result


def seed_summary(models, features):
    """Equal seed weights after per-model magnitude normalization, core only.

    Withhold a lane if even one core stratum fails; do not cherry-pick seeds.
    Supplement rows and single-seed non-IID are not mixed into these averages.
    """
    summary = {}
    for lane, seeds in (('light', (7, 17, 27)), ('iid', (7, 17, 27)), ('dirichlet', (7,))):
        keys = [f'{lane}-seed{s}' for s in seeds]
        if not all(models[k]['core']['passed'] for k in keys):
            summary[lane] = dict(status='withheld_numerical_failure', seeds=list(seeds))
            continue
        targets = {}
        for target in ['balanced_core']+CLASSES:
            entries = [models[k]['core']['descriptive']['own_true_score'] if target == 'balanced_core'
                       else models[k]['core']['descriptive']['by_true_class'][target] for k in keys]
            vectors = np.array([r['mean_absolute'] for r in entries])
            totals = vectors.sum(1)
            # All-zero vectors have no feature ranking evidence, even though
            # their numerical completeness could pass for a constant model.
            if not (totals > 0).all():
                targets[target] = dict(status='zero_attribution_magnitude')
                continue
            normalized = vectors/totals[:, None]
            mean = normalized.mean(0)
            top_sets = [set(e['top_five']) for e in entries]
            targets[target] = dict(status='descriptive', normalized_mean=mean.tolist(),
                normalized_sample_sd=None if len(keys) == 1 else normalized.std(0, ddof=1).tolist(),
                top_five=[features[i] for i in np.argsort(-mean, kind='stable')[:5]],
                all_seed_top_five_intersection=[f for f in features if f in set.intersection(*top_sets)],
                pairwise_top_five_jaccard=[len(a & b)/len(a | b) for a, b in combinations(top_sets, 2)])
        summary[lane] = dict(status='descriptive', seeds=list(seeds), targets=targets)
    return summary


def load_evidence(preparation_root, convergence_root, integration_root, archive_root, manifest_path,
                  preparation_plan_path, convergence_plan_path, integration_plan_path, plan_path):
    """Verify every input/model before any long integration starts."""
    prep_root, conv_root, int_root = map(Path, (preparation_root, convergence_root, integration_root))
    plan, parent, conv, prep = map(mc.read_json, (plan_path, integration_plan_path,
                                                 convergence_plan_path, preparation_plan_path))
    validate_plan(plan, parent)
    ig.validate_plan(parent, conv)
    mc.validate_plan(conv, prep)
    for path, expected in ((integration_plan_path, plan['integration_plan_sha256']),
        (int_root/'receipt.json', plan['integration_receipt_sha256']),
        (convergence_plan_path, parent['convergence_plan_sha256']),
        (conv_root/'receipt.json', parent['convergence_receipt_sha256']),
        (preparation_plan_path, conv['preparation_plan_sha256']),
        (prep_root/'receipt.json', conv['preparation_receipt_sha256']),
        (manifest_path, prep['packed_manifest_sha256'])):
        require(sha256(path) == expected, f'Evidence checksum changed: {Path(path).name}')
    prior, prepared, manifest = map(mc.read_json, (int_root/'receipt.json', prep_root/'receipt.json', manifest_path))
    official.validate_plan(prep, manifest)
    require(prior['status'] == 'complete' and prior['all_method_screens_passed'] is True
            and prior['plan_sha256'] == sha256(integration_plan_path)
            and prior['preparation_receipt_sha256'] == sha256(prep_root/'receipt.json')
            and prior['convergence_receipt_sha256'] == sha256(conv_root/'receipt.json')
            and prior['cpu_reference_sha256'] == prep['cpu_reference_sha256']
            and prior['device'] == 'cpu' and prior['classes'] == CLASSES
            and all(prior[k] is False for k in mc.FLAGS), 'Invalid passing pilot')
    packages = {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'shap')}
    require(packages == prior['packages'] == prepared['packages'], 'Explanation packages changed')
    for module in (official, mc, ig):
        raw = Path(module.__file__).read_bytes()
        require(prior['source_sha256']['explain/'+Path(module.__file__).name] in
                [hashlib.sha256(v).hexdigest() for v in (raw, raw.replace(b'\r\n', b'\n'))], 'Pilot source changed')
    for filename, field in (('inputs.npz', 'inputs_sha256'), ('sampling.json', 'sampling_sha256')):
        require(sha256(prep_root/filename) == prior[field] == prepared[field], 'Frozen study inputs changed')
    inputs, sampling = mc.read_arrays(prep_root/'inputs.npz'), mc.read_json(prep_root/'sampling.json')
    x, background, y = inputs['x'], inputs['background'], inputs['y']
    rows = sampling['validation_rows']
    require(x.shape == (219, 39) and background.shape == (128, 39) and y.shape == (219,)
        and np.issubdtype(y.dtype, np.integer) and y.min() >= 0 and y.max() < 8
        and all(v.dtype == np.float32 and np.isfinite(v).all() for v in (x, background))
        and inputs['validation_rows'].tolist() == rows and len(set(rows)) == 219
        and rows == sampling['core_rows']+sampling['supplement_rows']
        and len(sampling['core_rows']) == 64 and len(sampling['supplement_rows']) == 155
        and np.bincount(y[:64], minlength=8).tolist() == [8]*8, 'Frozen study shape/order changed')
    positions = [rows.index(r) for r in sampling['pilot_validation_rows']]
    require(len(positions) == 8 and all(p < 64 for p in positions)
        and y[positions].tolist() == list(range(8))
        and np.array_equal(x[positions], inputs['pilot_x']), 'Pilot subset changed')
    require(set(prior['models']) == {s['id'] for s in prep['models']} == set(sampling['selected_predictions']),
            'Model roster changed')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    models, pilots, endpoints = {}, {}, {}
    for spec in prep['models']:
        key = spec['id']
        record = prior['models'][key]
        path = int_root/f'{key}-integration.npz'
        require(sha256(path) == record['arrays_sha256'], 'Pilot arrays changed')
        pilot = mc.read_arrays(path)
        mc_path = conv_root/f'{key}-2048.npz'
        require(sha256(mc_path) == record['monte_carlo_arrays_sha256'], 'Monte Carlo arrays changed')
        baseline = mc.read_arrays(mc_path)
        require(ig.diagnose(pilot['coarse'], pilot['fine'], baseline, y[positions], parent)
                == {k: record[k] for k in ('scores', 'resolution_agreement', 'versus_mc2048', 'error_margins', 'method_screen_passed')}
                and record['method_screen_passed'] is True, 'Pilot diagnostics changed')
        require(record['checkpoint_sha256'] == spec['files_sha256']['last.pt'], 'Checkpoint reference changed')
        model, historical = official.load_frozen(spec, archive_root, prep, manifest)
        require(historical == prepared['bridges'][key]['historical_validation_metrics'], 'Historical metrics changed')
        with torch.no_grad():
            logits = model(torch.from_numpy(x)).numpy()
            pilot_logits = model(torch.from_numpy(inputs['pilot_x'])).numpy()
            reference = model(torch.from_numpy(background)).numpy().mean(0)
        require(np.array_equal(pilot_logits, pilot['logits']) and np.array_equal(reference, pilot['reference_logits'])
            and np.array_equal(pilot_logits, baseline['logits'])
            and np.array_equal(logits.argmax(1), sampling['selected_predictions'][key]), 'Frozen prediction/logit bridge failed')
        models[key], pilots[key], endpoints[key] = model, pilot, (logits, reference)
    return plan, parent, prep, prior, inputs, sampling, models, pilots, endpoints, packages


def run(preparation_root, convergence_root, integration_root, archive_root, manifest_path,
        preparation_plan_path, convergence_plan_path, integration_plan_path, plan_path, output):
    output = Path(output)
    require(not output.exists(), 'Refusing to overwrite study evidence')
    plan, parent, prep, prior, inputs, sampling, models, pilots, endpoints, packages = load_evidence(
        preparation_root, convergence_root, integration_root, archive_root, manifest_path,
        preparation_plan_path, convergence_plan_path, integration_plan_path, plan_path)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(protocol=plan['protocol'], status='incomplete', plan_sha256=sha256(plan_path),
        integration_receipt_sha256=plan['integration_receipt_sha256'],
        inputs_sha256=prior['inputs_sha256'], sampling_sha256=prior['sampling_sha256'],
        packed_manifest_sha256=sha256(manifest_path), cpu_reference_sha256=prior['cpu_reference_sha256'],
        source_sha256=dict(prior['source_sha256'], **{'explain/official_study.py': sha256(__file__)}),
        device='cpu', platform=platform.platform(), packages=packages, classes=CLASSES, features=prep['features'],
        settings=plan['settings'], work_budget=plan['work_budget'], completeness_limits=parent['completeness_limits'],
        resolution_limits=parent['resolution_limits'], full_validation_rerun=False,
        **{k: plan[k] for k in mc.FLAGS}, models={})
    atomic_json(result, output/'receipt.json')
    x, y, background = inputs['x'], inputs['y'], inputs['background']
    rows = sampling['validation_rows']
    positions = [rows.index(r) for r in sampling['pilot_validation_rows']]
    for key, model in models.items():
        values = {}
        for segments, name in ((2, 'coarse'), (4, 'fine')):
            chunks = []
            for start in range(0, len(x), 8):
                chunks.append(ig.integrate(model, background, x[start:start+8], segments))
                print(f'{key}: {segments*32} points, {min(start+8, len(x))}/{len(x)} cases', flush=True)
            values[name] = np.concatenate(chunks)
            require(np.array_equal(values[name][positions], pilots[key][name]), 'Full-run pilot attribution replay failed')
        logits, reference = endpoints[key]
        np.savez_compressed(output/f'{key}-study.npz', **values, logits=logits, reference_logits=reference)
        record = dict(checkpoint_sha256=prior['models'][key]['checkpoint_sha256'],
            arrays_sha256=sha256(output/f'{key}-study.npz'), pilot_attributions_exact=True,
            path_forward_rows=219*128*192, output_gradient_rows=8*219*128*192)
        for name, region in (('core', slice(0, 64)), ('supplement', slice(64, 219))):
            record[name] = describe_stratum(values['coarse'][region], values['fine'][region],
                logits[region], reference, y[region], rows[region], prep['features'], parent)
        result['models'][key] = record
        atomic_json(result, output/'receipt.json')
        print(f"{key}: core QA={record['core']['passed']}, supplement QA={record['supplement']['passed']}", flush=True)
    result['all_numerical_screens_passed'] = all(m[s]['passed'] for m in result['models'].values() for s in ('core', 'supplement'))
    result['status'] = 'complete' if result['all_numerical_screens_passed'] else 'numerical_review_required'
    result['seed_summary'] = seed_summary(result['models'], prep['features'])
    result['next'] = 'Review conditional descriptive results and limitations; no automatic policy or training change'
    atomic_json(result, output/'receipt.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation-root', 'convergence-root', 'integration-root', 'archive-root', 'manifest-path',
                 'preparation-plan-path', 'convergence-plan-path', 'integration-plan-path', 'plan-path', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
