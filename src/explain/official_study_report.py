"""Audit saved study arrays and publish compact, failure-preserving evidence.

No integration, model loading, dataset scanning or feature-value publication.
Recompute every diagnostic and ranking from the saved attribution arrays before
summarizing them. Complete raw arrays stay in ignored outputs, pinned by SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.explain import official, official_convergence as mc, official_integration as ig, official_study as study
from src.models.research import atomic_json

require = official.require


def compact_screen(screen, rows, outputs):
    """Keep every failed row/output, not just totals that can hide outliers."""
    failures = []
    passed = np.asarray(screen['per_output_pass'])
    for index in np.argwhere(~passed):
        i = tuple(index)
        failures.append(dict(validation_row=int(rows[i[0]]),
            output=outputs[i[1]] if len(i) == 2 else 'predicted_minus_true',
            coarse_residual=float(np.asarray(screen['coarse']['residuals'])[i]),
            fine_residual=float(np.asarray(screen['fine']['residuals'])[i]),
            residual_limit=float(np.asarray(screen['fine']['residual_limits'])[i]),
            relative_l1_resolution_difference=float(np.asarray(screen['resolution']['per_output_relative_l1_difference'])[i])))
    return dict(passed=screen['passed'], individual_pass_count=int(passed.sum()), output_count=int(passed.size),
        coarse={k: v for k, v in screen['coarse'].items() if k not in ('residuals', 'residual_limits')},
        fine={k: v for k, v in screen['fine'].items() if k not in ('residuals', 'residual_limits')},
        resolution={k: v for k, v in screen['resolution'].items() if k != 'per_output_relative_l1_difference'},
        failed_outputs=failures)


def compact_rank(ranking, features):
    return dict(count=ranking['count'], total_mean_absolute=float(sum(ranking['mean_absolute'])),
        top_five=[dict(feature=name, mean_absolute=ranking['mean_absolute'][features.index(name)],
                       mean_signed=ranking['mean_signed'][features.index(name)]) for name in ranking['top_five']])


def publish(study_root, preparation_root, plan_path, integration_plan_path, output):
    root, prep_root, output = Path(study_root), Path(preparation_root), Path(output)
    require(not output.exists(), 'Refusing to overwrite published study evidence')
    receipt, plan, parent = map(mc.read_json, (root/'receipt.json', plan_path, integration_plan_path))
    study.validate_plan(plan, parent)
    require(receipt['protocol'] == plan['protocol'] and receipt['plan_sha256'] == sha256(plan_path)
        and plan['integration_plan_sha256'] == sha256(integration_plan_path)
        and receipt['integration_receipt_sha256'] == plan['integration_receipt_sha256']
        and receipt['status'] in ('complete', 'numerical_review_required')
        and all(receipt[k] == plan[k] for k in mc.FLAGS)
        and receipt['classes'] == CLASSES and receipt['full_validation_rerun'] is False, 'Study scope changed')
    for module in (official, mc, ig, study):
        raw = Path(module.__file__).read_bytes()
        require(receipt['source_sha256']['explain/'+Path(module.__file__).name] in
            [hashlib.sha256(v).hexdigest() for v in (raw, raw.replace(b'\r\n', b'\n'))], 'Study source changed')
    require(sha256(prep_root/'inputs.npz') == receipt['inputs_sha256']
        and sha256(prep_root/'sampling.json') == receipt['sampling_sha256'], 'Frozen inputs changed')
    inputs, sampling = mc.read_arrays(prep_root/'inputs.npz'), mc.read_json(prep_root/'sampling.json')
    features = receipt['features']
    rows, labels = sampling['validation_rows'], inputs['y']
    require(inputs['validation_rows'].tolist() == rows and len(rows) == len(labels) == 219
        and len(features) == len(set(features)) == 39 and rows[:64] == sampling['core_rows'], 'Study shape changed')
    require(set(receipt['models']) == {f'{lane}-seed{s}' for lane, s in official.ROSTER}, 'Study roster changed')
    published = {k: v for k, v in receipt.items() if k != 'models'}
    published.update(source_receipt_sha256=sha256(root/'receipt.json'), publisher_sha256=sha256(__file__), models={})
    for key, record in receipt['models'].items():
        path = root/f'{key}-study.npz'
        require(sha256(path) == record['arrays_sha256'], 'Study arrays changed')
        a = mc.read_arrays(path)
        require(a['coarse'].shape == a['fine'].shape == (219, 39, 8)
            and a['logits'].shape == (219, 8) and a['reference_logits'].shape == (8,)
            and all(np.isfinite(v).all() for v in a.values()), 'Invalid study arrays')
        compact = {k: v for k, v in record.items() if k not in ('core', 'supplement')}
        for name, region in (('core', slice(0, 64)), ('supplement', slice(64, 219))):
            recalculated = study.describe_stratum(a['coarse'][region], a['fine'][region], a['logits'][region],
                a['reference_logits'], labels[region], rows[region], features, parent)
            require(recalculated == record[name], 'Study diagnostics or rankings changed')
            r = recalculated
            compact[name] = dict(count=r['count'], error_count=r['error_count'], passed=r['passed'],
                individual_passing_cases=sum(c['individual_checks_passed'] for c in r['cases']),
                scores=compact_screen(r['scores'], rows[region], CLASSES),
                error_margins=None if r['error_margins'] is None else
                    compact_screen(r['error_margins'], r['error_validation_rows'], []), descriptive=None)
            if r['descriptive'] is not None:
                desc = r['descriptive']
                compact[name]['descriptive'] = dict(own_true_score=compact_rank(desc['own_true_score'], features),
                    by_true_class={k: compact_rank(v, features) for k, v in desc['by_true_class'].items()},
                    error_pairs={k: compact_rank(v, features) for k, v in desc['error_pairs'].items()})
        published['models'][key] = compact
    require(study.seed_summary(receipt['models'], features) == receipt['seed_summary'], 'Seed summary changed')
    all_pass = all(m[s]['passed'] for m in receipt['models'].values() for s in ('core', 'supplement'))
    require(receipt['all_numerical_screens_passed'] == all_pass
        and receipt['status'] == ('complete' if all_pass else 'numerical_review_required'), 'Completion status changed')
    published['audit'] = dict(all_saved_arrays_verified=True, all_diagnostics_recomputed=True,
                             all_descriptive_rankings_recomputed=True, test_opened=False, integration_run=False)
    atomic_json(published, output)
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study-root', 'preparation-root', 'plan-path', 'integration-plan-path', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    result = publish(**vars(parser.parse_args()))
    print(result['status'])


if __name__ == '__main__':
    main()
