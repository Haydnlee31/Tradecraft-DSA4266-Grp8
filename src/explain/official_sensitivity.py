"""Local reference sensitivity and training-only feature redundancy audit.

No model retraining or feature removal. Read training arrays in bounded chunks,
then explain only the eight already frozen pilot cases against one newly seeded
training background. Related-feature sums describe existing attributions; they
are not grouped Shapley calculations or an ablation experiment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.explain import official, official_convergence as mc, official_integration as ig, official_study as study
from src.models.research import atomic_json

require = official.require
GROUPS = {'link_indicators': ['ARP', 'IPv', 'LLC'], 'size_summaries': ['AVG', 'Tot size']}


def validate_plan(plan):
    require(plan['protocol'] == 'official39-explanation-sensitivity-v1', 'Unknown sensitivity protocol')
    require(plan['settings'] == dict(training_rows=2000000, scan_batch_size=8192,
        background_rows=128, background_seed=42663908, cases='same eight original pilot rows, one per class',
        path_nodes=[64, 128], threads=2, correlation_threshold=.98, identity_tolerance_scaled=1e-5,
        number_value_tolerance_unscaled=1e-4, sign_magnitude_floor=1e-6), 'Sensitivity scope changed')
    require(plan['feature_groups'] == GROUPS and all(plan[k] is False for k in mc.FLAGS), 'Unreviewed action')
    require(plan['work_budget'] == dict(models=7, path_forward_rows=7*8*128*192,
        output_gradient_rows=8*7*8*128*192), 'Sensitivity budget changed')


def select_background(n, excluded, size=128, seed=42663908):
    """Uniform without replacement among eligible rows, with no label peeking.

    Rejection draws avoid materializing an N-row index array. Excluding the old
    background makes the new reference disjoint, not statistically independent
    of the same finite dataset. Sorting fixes summation order.
    """
    blocked = set(map(int, excluded))
    require(type(n) is int and 0 < size <= n-len(blocked)
        and all(0 <= i < n for i in blocked), 'Invalid background population')
    rng, selected = np.random.default_rng(seed), set()
    while len(selected) < size:
        row = int(rng.integers(n))
        if row not in blocked:
            selected.add(row)
    return np.array(sorted(selected), dtype=np.int64)


@contextmanager
def training_arrays(root, manifest):
    """Never instantiate PackedData here: it also opens validation arrays."""
    root, arrays = Path(root), []
    # Check every allowed input before opening the first array.
    for name in ('train_x.npy', 'train_y.npy'):
        require(sha256(root/name) == manifest['files'][name], 'Training file checksum changed')
    try:
        for name in ('train_x.npy', 'train_y.npy'):
            arrays.append(np.load(root/name, mmap_mode='r', allow_pickle=False))
        x, y = arrays
        require(x.shape == (manifest['train_rows'], 39) and y.shape == (len(x),)
                and x.dtype == np.float32 and y.dtype == np.int64, 'Training shape/dtype changed')
        yield x, y
    finally:
        # Explicitly close memmaps so Windows can release fixture directories.
        for array in arrays:
            array._mmap.close()


def scan_training(x, y, scaler, features, settings):
    """Streaming moments and predeclared identities, with no classifier fit.

    Number is approximately unscaled from the packed float32 values. Its
    10/100 bins use a fixed tolerance; other values are kept, never rounded into
    whichever bin supports a hypothesis. Equality tests use stored model inputs.
    """
    require(len(features) == len(set(features)) == 39 and x.shape == (len(y), 39) and len(y) > 0,
            'Invalid training audit shape')
    sums, cross = np.zeros(39), np.zeros((39, 39))
    feature_min, feature_max = np.full(39, np.inf), np.full(39, -np.inf)
    counts, bins = np.zeros(8, np.int64), np.zeros((8, 3), np.int64)
    minimum, maximum = np.full(8, np.inf), np.full(8, -np.inf)
    identities = [('IPv', 'LLC', 1), ('AVG', 'Tot size', 1), ('ARP', 'IPv', -1)]
    identity_counts, identity_max = np.zeros(3, np.int64), np.zeros(3)
    f = features.index('Number')
    for start in range(0, len(y), settings['scan_batch_size']):
        end = min(start+settings['scan_batch_size'], len(y))
        a, labels = np.array(x[start:end], dtype=np.float64), np.array(y[start:end], copy=True)
        require(np.isfinite(a).all() and np.issubdtype(labels.dtype, np.integer)
                and labels.min() >= 0 and labels.max() < 8, 'Invalid training values')
        sums += a.sum(0)
        cross += a.T@a
        feature_min = np.minimum(feature_min, a.min(0))
        feature_max = np.maximum(feature_max, a.max(0))
        counts += np.bincount(labels, minlength=8)
        number = a[:, f]*scaler['scale'][f]+scaler['mean'][f]
        b = np.full(len(a), 2, np.int64)
        b[np.abs(number-10) <= settings['number_value_tolerance_unscaled']] = 0
        b[np.abs(number-100) <= settings['number_value_tolerance_unscaled']] = 1
        np.add.at(bins, (labels, b), 1)
        np.minimum.at(minimum, labels, number)
        np.maximum.at(maximum, labels, number)
        for k, (left, right, sign) in enumerate(identities):
            residual = np.abs(a[:, features.index(left)]-sign*a[:, features.index(right)])
            identity_counts[k] += np.count_nonzero(residual <= settings['identity_tolerance_scaled'])
            identity_max[k] = max(identity_max[k], float(residual.max()))
    covariance = cross/len(y)-np.outer(sums/len(y), sums/len(y))
    std = np.sqrt(np.maximum(np.diag(covariance), 0))
    # Constants are determined by extrema, not a cancellation-prone variance.
    std[feature_min == feature_max] = 0
    correlated = []
    for i in range(39):
        for j in range(i+1, 39):
            if std[i] > 1e-12 and std[j] > 1e-12:
                r = float(np.clip(covariance[i, j]/(std[i]*std[j]), -1, 1))
                if abs(r) >= settings['correlation_threshold']:
                    correlated.append(dict(left=features[i], right=features[j], pearson_r=r))
    return dict(rows=len(y), classes=CLASSES, class_counts=dict(zip(CLASSES, map(int, counts))),
        number_by_class={name: dict(count=int(counts[c]), near_10=int(bins[c, 0]), near_100=int(bins[c, 1]),
            other=int(bins[c, 2]), minimum=None if counts[c] == 0 else float(minimum[c]),
            maximum=None if counts[c] == 0 else float(maximum[c])) for c, name in enumerate(CLASSES)},
        high_correlations=correlated, constant_features=[features[i] for i in range(39) if std[i] <= 1e-12],
        identities=[dict(left=a, right=b, right_multiplier=s, passing_rows=int(identity_counts[k]),
                         maximum_absolute_scaled_residual=float(identity_max[k])) for k, (a, b, s) in enumerate(identities)])


def grouped(values, features):
    """Aggregate signed contributions BEFORE magnitude, exposing cancellation."""
    result = {}
    for name, members in GROUPS.items():
        a = values[:, [features.index(f) for f in members], :]
        gross, net = np.abs(a).sum(1), np.abs(a.sum(1))
        total = float(gross.sum())
        result[name] = dict(features=members, mean_gross_magnitude=float(gross.mean()),
            mean_net_magnitude=float(net.mean()), magnitude_cancellation_fraction=None if total == 0 else
            float(np.clip(1-net.sum()/total, 0, 1)))
    return result


def compare_backgrounds(old, new, logits, old_reference, new_reference, labels, features, floor):
    """Different references change the estimand: no invented agreement gate."""
    rows = np.arange(len(labels))
    old_own, new_own = old[rows, :, labels], new[rows, :, labels]
    ranks = [study.rank(v, features) for v in (old_own, new_own)]
    top = [set(r['top_five']) for r in ranks]
    number = features.index('Number')
    errors = np.flatnonzero(logits.argmax(1) != labels)
    number_cases = []
    for i in rows:
        a, b = float(old_own[i, number]), float(new_own[i, number])
        number_cases.append(dict(pilot_position=int(i), true_class=CLASSES[int(labels[i])],
            old_true_score_contribution=a, new_true_score_contribution=b,
            sign_comparable=abs(a) > floor and abs(b) > floor,
            sign_changed=bool(np.sign(a) != np.sign(b)) if abs(a) > floor and abs(b) > floor else None))
    error_cases = []
    for i in errors:
        p, y = int(logits[i].argmax()), int(labels[i])
        a, b = old[i, :, p]-old[i, :, y], new[i, :, p]-new[i, :, y]
        error_cases.append(dict(pilot_position=int(i), true_class=CLASSES[y], predicted_class=CLASSES[p],
            unchanged_logit_margin=float(logits[i, p]-logits[i, y]),
            old_background_margin=float(old_reference[p]-old_reference[y]),
            new_background_margin=float(new_reference[p]-new_reference[y]),
            old_number_contribution=float(a[number]), new_number_contribution=float(b[number])))
    return dict(score_feature_difference=ig.agreement(old, new),
        reference_logit_shift=(new_reference-old_reference).tolist(),
        own_true_score=dict(old=ranks[0], new=ranks[1], top_five_jaccard=len(top[0] & top[1])/len(top[0] | top[1])),
        number_cases=number_cases, error_cases=error_cases,
        groups=dict(old=grouped(old, features), new=grouped(new, features)))


def run(preparation_root, convergence_root, integration_root, study_root, archive_root, packed_root,
        preparation_plan_path, convergence_plan_path, integration_plan_path, study_plan_path, plan_path, output):
    output, packed, study_root = Path(output), Path(packed_root), Path(study_root)
    require(not output.exists(), 'Refusing to overwrite sensitivity evidence')
    plan = mc.read_json(plan_path)
    validate_plan(plan)
    require(sha256(study_plan_path) == plan['study_plan_sha256']
        and sha256(study_root/'receipt.json') == plan['study_receipt_sha256'], 'Study evidence changed')
    saved = mc.read_json(study_root/'receipt.json')
    require(saved['status'] == 'complete' and saved['all_numerical_screens_passed'] is True
            and saved['plan_sha256'] == sha256(study_plan_path), 'Passing study required')
    for module in (official, mc, ig, study):
        raw = Path(module.__file__).read_bytes()
        require(saved['source_sha256']['explain/'+Path(module.__file__).name] in
            [hashlib.sha256(v).hexdigest() for v in (raw, raw.replace(b'\r\n', b'\n'))], 'Study source changed')
    _, parent, prep, prior, inputs, sampling, models, pilots, _, packages = study.load_evidence(
        preparation_root, convergence_root, integration_root, archive_root, packed/'manifest.json',
        preparation_plan_path, convergence_plan_path, integration_plan_path, study_plan_path)
    manifest, scaler = mc.read_json(packed/'manifest.json'), mc.read_json(packed/'scaler.json')
    require(manifest['train_rows'] == plan['settings']['training_rows']
        and sha256(packed/'scaler.json') == manifest['scaler_sha256']
        and scaler['features'] == prep['features'] and scaler['fit_split'] == 'train'
        and scaler['n_samples_seen'] == manifest['train_rows'], 'Training scaler/cohort changed')
    features = prep['features']
    positions = [sampling['validation_rows'].index(r) for r in sampling['pilot_validation_rows']]
    labels = inputs['y'][positions]
    group_audit = {}
    require(set(saved['models']) == set(models), 'Study roster changed')
    # Audit all saved arrays before reading training features or integrating.
    for key in models:
        path = study_root/f'{key}-study.npz'
        require(sha256(path) == saved['models'][key]['arrays_sha256']
            and saved['models'][key]['checkpoint_sha256'] == prior['models'][key]['checkpoint_sha256'], 'Study arrays changed')
        a = mc.read_arrays(path)
        require(np.array_equal(a['coarse'][positions], pilots[key]['coarse'])
            and np.array_equal(a['fine'][positions], pilots[key]['fine']), 'Saved pilot subset changed')
        group_audit[key] = {name: grouped(a['fine'][region], features)
            for name, region in (('core', slice(0, 64)), ('supplement', slice(64, 219)))}
    with training_arrays(packed, manifest) as (train_x, train_y):
        old_rows = sampling['background_train_rows']
        require(np.array_equal(train_x[old_rows], inputs['background']), 'Original training background changed')
        selected = select_background(len(train_y), old_rows)
        background = np.array(train_x[selected], copy=True)
        background_labels = np.array(train_y[selected], copy=True)
        training = scan_training(train_x, train_y, scaler, features, plan['settings'])
        require(training['class_counts'] == manifest['train_class_counts'], 'Training class counts changed')
    require(np.isfinite(background).all(), 'Invalid new background')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'background.npz', background=background, labels=background_labels, training_rows=selected)
    result = dict(protocol=plan['protocol'], status='incomplete', plan_sha256=sha256(plan_path),
        study_receipt_sha256=sha256(study_root/'receipt.json'), packed_manifest_sha256=sha256(packed/'manifest.json'),
        inputs_sha256=prior['inputs_sha256'], sampling_sha256=prior['sampling_sha256'],
        source_sha256=dict(saved['source_sha256'], **{'explain/official_sensitivity.py': sha256(__file__)}),
        device='cpu', packages=packages, classes=CLASSES, features=features, settings=plan['settings'],
        work_budget=plan['work_budget'], background_sha256=sha256(output/'background.npz'),
        background_rows=selected.tolist(), original_background_counts=sampling['background_class_counts'],
        new_background_counts=dict(zip(CLASSES, map(int, np.bincount(background_labels, minlength=8)))),
        training_audit=training, existing_group_audit=group_audit, models={}, **dict.fromkeys(mc.FLAGS, False))
    atomic_json(result, output/'receipt.json')
    for key, model in models.items():
        old = pilots[key]
        with torch.no_grad():
            logits = model(torch.from_numpy(inputs['pilot_x'])).numpy()
            reference = model(torch.from_numpy(background)).numpy().mean(0)
        require(np.array_equal(logits, old['logits']), 'Pilot logits changed')
        print(f'{key}: new uniform background, 64 then 128 path points', flush=True)
        coarse, fine = (ig.integrate(model, background, inputs['pilot_x'], s) for s in (2, 4))
        qa = study.describe_stratum(coarse, fine, logits, reference, labels,
            sampling['pilot_validation_rows'], features, parent)
        original_qa = study.describe_stratum(old['coarse'], old['fine'], logits, old['reference_logits'],
            labels, sampling['pilot_validation_rows'], features, parent)
        passed = qa['passed'] and original_qa['passed']
        comparison = compare_backgrounds(old['fine'], fine, logits, old['reference_logits'], reference,
            labels, features, plan['settings']['sign_magnitude_floor']) if passed else None
        np.savez_compressed(output/f'{key}-sensitivity.npz', coarse=coarse, fine=fine, logits=logits, reference_logits=reference)
        result['models'][key] = dict(checkpoint_sha256=saved['models'][key]['checkpoint_sha256'],
            arrays_sha256=sha256(output/f'{key}-sensitivity.npz'), numerical_qa=qa,
            original_numerical_qa=original_qa, comparison_available=passed, comparison=comparison)
        atomic_json(result, output/'receipt.json')
        print(f'{key}: numerical QA={passed}; background robustness is NOT a pass/fail test', flush=True)
    result['all_numerical_screens_passed'] = all(r['comparison_available'] for r in result['models'].values())
    result['status'] = 'complete' if result['all_numerical_screens_passed'] else 'numerical_review_required'
    result['next'] = 'Review training-feature associations and matched-reference sensitivity; no automatic training or policy change'
    atomic_json(result, output/'receipt.json')
    return result


def publish_verified(root, replay_root, output):
    """Publish measured evidence only after a byte-identical independent run.

    Replay checks reproducibility, not independent sampling uncertainty. It
    repeats the same seeds and background, never selects a more favorable one.
    """
    root, replay, output = map(Path, (root, replay_root, output))
    require(root.resolve() != replay.resolve(), 'Independent replay folder required')
    require(not output.exists(), 'Refusing to overwrite published sensitivity evidence')
    require(sha256(root/'receipt.json') == sha256(replay/'receipt.json'), 'Sensitivity replay changed')
    r = mc.read_json(root/'receipt.json')
    require(r['protocol'] == 'official39-explanation-sensitivity-v1'
        and r['status'] in ('complete', 'numerical_review_required')
        and all(r[k] is False for k in mc.FLAGS)
        and set(r['models']) == {f'{lane}-seed{s}' for lane, s in official.ROSTER}, 'Invalid sensitivity receipt')
    files = {'background.npz': r['background_sha256']}
    files.update({k+'-sensitivity.npz': v['arrays_sha256'] for k, v in r['models'].items()})
    for name, digest in files.items():
        require(sha256(root/name) == digest == sha256(replay/name), 'Sensitivity array replay changed')
    published = dict(r, source_receipt_sha256=sha256(root/'receipt.json'),
        audit=dict(independent_receipt_replay_exact=True, all_eight_npz_files_replay_exact=True,
                   independent_uncertainty_estimate=False))
    atomic_json(published, output)
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preparation-root', 'convergence-root', 'integration-root', 'study-root', 'archive-root', 'packed-root',
                 'preparation-plan-path', 'convergence-plan-path', 'integration-plan-path', 'study-plan-path', 'plan-path', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
