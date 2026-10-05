"""Verify all twelve validation screens and summarize the final dropout ablation.

Run from the repository root with a new --output directory. Only saved artifacts
are read: no model checkpoints are loaded and no datasets are opened.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/mlp-tuning'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan_path = Path('configs/local_tuning_plan.json')
    plan = read(plan_path)
    spec = importlib.util.spec_from_file_location('convergence', Path(__file__).resolve().parents[1] / 'convergence_2026_10_03/analyze.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    runs, histories, reference = {}, {}, None
    for stage in plan['stages']:
        for lane in stage['lanes']:
            key = stage['name'] + '/' + lane
            folder = args.root / key
            r, m = read(folder / 'result.json'), read(folder / 'environment.json')
            assert read(folder / 'status.json')['status'] == 'complete', key
            assert r['steps'] == 60 and r['test_metrics'] is None, key
            assert set(m['split_sha256']) == {'train', 'val'}, key
            assert m['settings']['lane'] == lane, key
            for name, value in {**plan['common'], **stage['overrides']}.items():
                assert m['settings'][name] == value, (key, name)
            assert [h['step'] for h in r['history']] == list(range(1, 61)), key
            selected = helper.best(r['history'])
            assert selected['step'] == r['best_step'], key
            assert selected['validation_metrics'] == r['validation_metrics'], key
            assert digest(folder / 'best.pt') == r['checkpoint_sha256'], key
            assert digest(folder / 'scaler.joblib') == m['scaler_sha256'], key
            if lane in ('iid', 'dirichlet'):
                assert digest(folder / 'assignments.npz') == m['partition_sha256'], key
            if reference is None:
                reference = m
            for field in ('source_sha256', 'split_sha256', 'packages', 'feature_columns', 'classes', 'scaler_sha256', 'train_class_counts'):
                assert m[field] == reference[field], (key, field)
            runs[key] = {name: r[name] for name in ('best_step', 'validation_metrics', 'checkpoint_sha256', 'num_parameters', 'step_seconds_sum', 'examples_processed', 'optimizer_steps')}
            runs[key].update(manifest=m, result_sha256=digest(folder / 'result.json'))
            histories[key] = r['history']
    assert len(runs) == plan['max_screen_runs'] == 12
    # Audit every ablation against its same-lane convergence control.
    for stage in plan['stages'][1:]:
        allowed = set(stage['overrides'])
        for lane in stage['lanes']:
            candidate, control = runs[stage['name'] + '/' + lane], runs['convergence/' + lane]
            a, b = control['manifest'], candidate['manifest']
            for section in ('settings', 'model_config'):
                assert {k: v for k, v in a[section].items() if k not in allowed} == {k: v for k, v in b[section].items() if k not in allowed}
            assert a['partition_sha256'] == b['partition_sha256']
            candidate['change_vs_control'] = helper.compare(control, candidate, plan)
    control, candidate = runs['convergence/heavy'], runs['heavy_dropout/heavy']
    assert control['manifest']['model_config']['dropout'] == 0.3
    assert candidate['manifest']['model_config']['dropout'] == 0.2
    summary = {'screen_runs_completed': 12, 'screen_runs_remaining': 0,
               'confirmation_runs_completed': 0, 'test_evaluated': False,
               'plan_sha256': digest(plan_path), 'runs': runs}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Completed validation screening — 2026-10-05', '',
             'Twelve planned runs, seed 0, fixed partition seed 0, 60 epochs/rounds. Each score selects the first-best validation macro-F1 checkpoint; no test evaluation.', '',
             '| Run | Best step | Macro-F1 | False alerts | Web recall | Brute Force recall | Gate A / B vs control |',
             '|---|---:|---:|---:|---:|---:|---|']
    for key, run in runs.items():
        m = run['validation_metrics']
        d = run.get('change_vs_control')
        gate = f"{d['promotion_gate_a']} / {d['promotion_gate_b']}" if d else 'Control'
        lines.append(f"| {key} | {run['best_step']} | {m['macro_f1']:.4f} | {100*m['benign_false_alert_rate']:.2f}% | {100*m['per_class_recall']['Web-based']:.2f}% | {100*m['per_class_recall']['Brute Force']:.2f}% | {gate} |")
    lines += ['', 'Gate A requires ≥1 pp macro-F1 gain, ≤2 pp false-alert increase and ≤2 pp recall loss for every class. Gate B requires ≥5 pp false-alert reduction, ≤1 pp macro-F1 loss and ≤2 pp Web/Brute Force recall loss. These are project preferences, not significance or deployment tests.', '',
              '## Final dropout ablation: every class recall', '', '| Class | Dropout 0.3 | Dropout 0.2 | Change (pp) |', '|---|---:|---:|---:|']
    for name in reference['classes']:
        a, b = [run['validation_metrics']['per_class_recall'][name] for run in (control, candidate)]
        lines.append(f'| {name} | {100*a:.2f}% | {100*b:.2f}% | {100*(b-a):+.2f} |')
    lines += ['', 'Full per-class precision/recall/F1, confusion matrices, provenance and compute for every run are in [summary.json](summary.json).', '',
              '## Checks and limits', '', '- Verified every run against the frozen plan, full history, best-step selection, checkpoint/scaler/partition hashes and shared source/data/package provenance.',
              '- Each ablation differs only in its declared settings/model-configuration override from its same-lane control.',
              '- Single-seed validation selection is exploratory. Reusing seed 0 in later summaries does not make it independent confirmation.',
              '- CPU timing includes training/validation but excludes setup/checkpoint IO; host load differs between runs. It is not an edge measurement or causal speed comparison.', '',
              '![Heavy dropout validation curves](curves.png)', '']
    (args.output / 'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for key, label in (('convergence/heavy', 'Dropout 0.3'), ('heavy_dropout/heavy', 'Dropout 0.2')):
        h = histories[key]
        for ax, metric in zip(axes, ('macro_f1', 'benign_false_alert_rate')):
            ax.plot([r['step'] for r in h], [r['validation_metrics'][metric] for r in h], label=label)
            ax.set(xlabel='Epoch', ylabel=metric)
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('Heavy dropout ablation — seed 0; validation only')
    fig.savefig(args.output / 'curves.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'best_step': candidate['best_step'], 'metrics': candidate['validation_metrics'], 'change': candidate['change_vs_control']}, indent=2))


if __name__ == '__main__':
    main()
