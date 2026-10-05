"""Audit the frozen configured-model study and report training-seed stability.

Only saved artifacts are read. --output must be new. Seed 0 is reused screening
evidence, so aggregates for new seeds 1/2 are always presented separately.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values),
            'min': min(values), 'max': max(values), 'n': len(values)}


def aggregate(runs, seeds, classes):
    result = {name: stats([runs[str(s)]['validation_metrics'][name] for s in seeds])
              for name in ('macro_f1', 'benign_false_alert_rate')}
    result['per_class'] = {c: {name: stats([runs[str(s)]['validation_metrics']['per_class'][c][name] for s in seeds])
                             for name in ('precision', 'recall', 'f1')} for c in classes}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('configs/local_confirmation_plan.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    root = Path(plan['output_root'])
    launch = read(root / 'launch.json')
    assert launch['plan_sha256'] == digest(args.plan) and launch['plan'] == plan
    results, histories, reference = {}, {}, None
    for lane, recipe in plan['lanes'].items():
        runs, seed_zero_manifest = {}, None
        for seed in plan['training_seeds']:
            folder = Path(recipe['seed_zero_path']) if seed == 0 else root / lane / f'seed-{seed}'
            r, m = read(folder / 'result.json'), read(folder / 'environment.json')
            assert read(folder / 'status.json')['status'] == 'complete', (lane, seed)
            assert r['steps'] == plan['common']['epochs'] and r['test_metrics'] is None
            assert set(m['split_sha256']) == {'train', 'val'}
            assert m['settings']['lane'] == lane and m['settings']['seed'] == seed
            for key, value in plan['common'].items():
                assert m['settings'][key] == value, (lane, seed, key)
            assert m['settings']['normalization'] == recipe['normalization']
            assert m['model_config']['normalization'] == recipe['normalization']
            assert m['model_config']['dropout'] == recipe['dropout']
            # Seed-zero CLI used default dropout=None; compare effective model
            # config instead of treating the explicit equivalent value as drift.
            assert m['settings']['dropout'] in (None, recipe['dropout'])
            assert [h['step'] for h in r['history']] == list(range(1, 61))
            selected = max(r['history'], key=lambda h: h['validation_metrics']['macro_f1'])
            assert selected['step'] == r['best_step'] and selected['validation_metrics'] == r['validation_metrics']
            assert digest(folder / 'best.pt') == r['checkpoint_sha256']
            assert digest(folder / 'scaler.joblib') == m['scaler_sha256']
            if lane in ('iid', 'dirichlet'):
                assert digest(folder / 'assignments.npz') == m['partition_sha256']
            assert m['source_sha256'] == launch['source_sha256']
            if reference is None:
                reference = m
            for key in ('source_sha256', 'split_sha256', 'packages', 'feature_columns', 'classes', 'scaler_sha256', 'train_class_counts'):
                assert m[key] == reference[key], (lane, seed, key)
            if seed_zero_manifest is None:
                seed_zero_manifest = m
            else:
                for key in ('model_config', 'partition_sha256'):
                    assert m[key] == seed_zero_manifest[key], (lane, seed, key)
                assert {k: v for k, v in m['settings'].items() if k not in ('seed', 'dropout')} == {k: v for k, v in seed_zero_manifest['settings'].items() if k not in ('seed', 'dropout')}
            runs[str(seed)] = {key: r[key] for key in ('best_step', 'validation_metrics', 'checkpoint_sha256', 'num_parameters', 'step_seconds_sum', 'examples_processed', 'optimizer_steps')}
            runs[str(seed)].update(path=str(folder), reused_screening_seed=seed == 0, manifest=m, result_sha256=digest(folder / 'result.json'))
            histories[lane, seed] = r['history']
        results[lane] = {'runs': runs,
                         'all_seeds_0_1_2': aggregate(runs, plan['training_seeds'], reference['classes']),
                         'new_seeds_1_2': aggregate(runs, plan['new_training_seeds'], reference['classes'])}
    summary = {'plan_sha256': digest(args.plan), 'plan': plan, 'test_evaluated': False,
               'new_runs_completed': 8, 'screening_seed_zero_runs_reused': 4, 'lanes': results}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Configured-model training-seed stability — 2026-10-05', '',
             'Validation only. Eight new runs (seeds 1/2) plus four reused screening runs (seed 0). All use 60 steps and partition seed 0. No cross-seed winner is selected.', '',
             '| Model | Seed 0 (screen) | Seed 1 | Seed 2 | All-seed F1 mean ± sample SD | New-seed F1 mean ± sample SD |',
             '|---|---:|---:|---:|---:|---:|']
    for lane, data in results.items():
        scores = [data['runs'][str(s)]['validation_metrics']['macro_f1'] for s in (0, 1, 2)]
        a, b = [data[group]['macro_f1'] for group in ('all_seeds_0_1_2', 'new_seeds_1_2')]
        lines.append(f"| {lane} | {scores[0]:.4f} | {scores[1]:.4f} | {scores[2]:.4f} | {a['mean']:.4f} ± {a['sample_sd']:.4f} | {b['mean']:.4f} ± {b['sample_sd']:.4f} |")
    lines += ['', '## Every run: selected step and benign false alerts', '', '| Model | Seed | Selected step | False alerts |', '|---|---:|---:|---:|']
    for lane, data in results.items():
        for seed, r in data['runs'].items():
            lines.append(f"| {lane} | {seed} | {r['best_step']} | {100*r['validation_metrics']['benign_false_alert_rate']:.2f}% |")
    for group, title in (('all_seeds_0_1_2', 'All seeds 0/1/2'), ('new_seeds_1_2', 'New seeds 1/2 only')):
        lines += ['', f'## {title}: class recall and false alerts (mean ± sample SD)', '', '| Metric | Heavy | Light | IID | Non-IID |', '|---|---:|---:|---:|---:|']
        for name in ['benign_false_alert_rate'] + reference['classes']:
            values = []
            for lane in results:
                a = results[lane][group]
                s = a[name] if name == 'benign_false_alert_rate' else a['per_class'][name]['recall']
                values.append(f"{100*s['mean']:.2f}% ± {100*s['sample_sd']:.2f} pp")
            lines.append('| ' + name + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## Every seed: per-class recall', '', '| Model / seed | ' + ' | '.join(reference['classes']) + ' |', '|---|' + '---:|' * len(reference['classes'])]
    for lane, data in results.items():
        for seed, r in data['runs'].items():
            lines.append(f'| {lane} / {seed} | ' + ' | '.join(f"{100*r['validation_metrics']['per_class_recall'][c]:.2f}%" for c in reference['classes']) + ' |')
    lines += ['', '## Interpretation limits', ''] + ['- ' + item for item in plan['limitations']]
    lines += ['- Sample SD is not a confidence interval; two new seeds cannot establish broad robustness.',
              '- Both seeds retain validation-based checkpoint selection. No new independent data were evaluated.',
              '- Normalization/dropout differ across configured models; these are not isolated federation/capacity effects.',
              '- Host timing includes training/validation, excludes setup/checkpoint IO. New jobs ran in pairs under uncontrolled host load; no causal speed or edge-hardware comparison is supported.', '',
              'All completion, configuration, source/data/package, selected-checkpoint, scaler and fixed-partition checks passed. Full per-class precision/recall/F1, confusion matrices and manifests: [summary.json](summary.json).', '',
              '![Seed trajectories](curves.png)', '']
    (args.output / 'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, lane in zip(axes.flat, results):
        for seed in (0, 1, 2):
            h = histories[lane, seed]
            ax.plot([r['step'] for r in h], [r['validation_metrics']['macro_f1'] for r in h], linestyle='--' if seed == 0 else '-', label=f'Seed {seed}' + (' (screen)' if seed == 0 else ''))
        ax.set(title=lane, xlabel='Epoch' if lane in ('heavy', 'light') else 'Round', ylabel='Validation macro-F1', ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Frozen configured-model comparison — fixed partition seed 0')
    fig.savefig(args.output / 'curves.png', dpi=150)
    plt.close(fig)
    print(json.dumps({lane: {group: {name: data[group][name] for name in ('macro_f1', 'benign_false_alert_rate')} for group in ('all_seeds_0_1_2', 'new_seeds_1_2')} for lane, data in results.items()}, indent=2))


if __name__ == '__main__':
    main()
