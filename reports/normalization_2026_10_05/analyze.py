"""Verify normalization controls and summarize saved validation-only artifacts.

Run from the repository root; --output must name a new directory. This script
does not load checkpoints or read training, validation, or test datasets.
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
    # Share the frozen gate implementation with the earlier convergence report.
    spec = importlib.util.spec_from_file_location('convergence', Path(__file__).resolve().parents[1] / 'convergence_2026_10_03/analyze.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    lanes = ('light', 'iid', 'dirichlet')
    results, histories = {}, {}
    for lane in lanes:
        pair, manifests = {}, {}
        for stage, norm in (('convergence', 'batch'), ('normalization', 'layer')):
            folder = args.root / stage / lane
            result, manifest = read(folder / 'result.json'), read(folder / 'environment.json')
            assert read(folder / 'status.json')['status'] == 'complete'
            assert result['steps'] == 60 and result['test_metrics'] is None
            assert set(manifest['split_sha256']) == {'train', 'val'}
            for key, value in {**plan['common'], 'normalization': norm}.items():
                assert manifest['settings'][key] == value, (lane, stage, key)
            assert [h['step'] for h in result['history']] == list(range(1, 61))
            best = helper.best(result['history'])
            assert best['step'] == result['best_step']
            assert best['validation_metrics'] == result['validation_metrics']
            assert digest(folder / 'best.pt') == result['checkpoint_sha256']
            assert digest(folder / 'scaler.joblib') == manifest['scaler_sha256']
            if lane != 'light':
                assert digest(folder / 'assignments.npz') == manifest['partition_sha256']
            pair[stage] = {k: result[k] for k in ('best_step', 'validation_metrics', 'checkpoint_sha256', 'step_seconds_sum', 'examples_processed', 'optimizer_steps', 'process_peak_rss_bytes', 'num_parameters')}
            pair[stage]['result_sha256'] = digest(folder / 'result.json')
            manifests[stage] = manifest
            histories[lane, stage] = result['history']
        before, after = manifests['convergence'], manifests['normalization']
        # Normalization changes state structure, but no other architecture setting.
        for section in ('settings', 'model_config'):
            assert {k: v for k, v in before[section].items() if k != 'normalization'} == {k: v for k, v in after[section].items() if k != 'normalization'}, (lane, section)
        for key in ('source_sha256', 'split_sha256', 'packages', 'feature_columns', 'classes', 'train_class_counts', 'scaler_sha256', 'partition_sha256'):
            assert before[key] == after[key], (lane, key)
        pair['change'] = helper.compare(pair['convergence'], pair['normalization'], plan)
        pair['manifests'] = manifests
        results[lane] = pair
    gaps = {}
    for stage in ('convergence', 'normalization'):
        central = results['light'][stage]['validation_metrics']['macro_f1']
        gaps[stage] = {lane: central - results[lane][stage]['validation_metrics']['macro_f1'] for lane in ('iid', 'dirichlet')}
    summary = {'stage': 'normalization', 'test_evaluated': False, 'screen_runs_completed': 11,
               'screen_runs_remaining': plan['max_screen_runs'] - 11,
               'plan_sha256': digest(plan_path), 'lanes': results, 'central_minus_federated_macro_f1': gaps}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Normalization comparison — 2026-10-05', '',
             'Validation only; model/partition seed 0; 60 epochs or rounds; weighted CE; no early stopping. Select first-best validation macro-F1 within each run.', '',
             '| Lane | BatchNorm F1 (step) | LayerNorm F1 (step) | Gain (pp) | Gate A / B |',
             '|---|---:|---:|---:|---|']
    for lane, pair in results.items():
        a, b, d = pair['convergence'], pair['normalization'], pair['change']
        lines.append(f"| {lane} | {a['validation_metrics']['macro_f1']:.4f} ({a['best_step']}) | {b['validation_metrics']['macro_f1']:.4f} ({b['best_step']}) | {100*d['macro_f1_gain']:+.2f} | {d['promotion_gate_a']} / {d['promotion_gate_b']} |")
    lines += ['', 'Gates are project preferences, not significance tests. A: macro-F1 gain ≥1 pp, false-alert increase ≤2 pp, each class recall drop ≤2 pp. B: false-alert reduction ≥5 pp, macro-F1 drop ≤1 pp, Web/Brute Force recall drops ≤2 pp.', '',
              '## Selected-checkpoint recall and false alerts', '',
              '| Lane / norm | False alerts | ' + ' | '.join(before['classes']) + ' |',
              '|---|' + '---:|' * (1 + len(before['classes']))]
    for lane, pair in results.items():
        for stage, label in (('convergence', 'BN'), ('normalization', 'LN')):
            m = pair[stage]['validation_metrics']
            values = [m['benign_false_alert_rate']] + [m['per_class_recall'][c] for c in before['classes']]
            lines.append(f'| {lane} / {label} | ' + ' | '.join(f'{100*v:.2f}%' for v in values) + ' |')
    lines += ['', '## Central-light minus federated macro-F1 gap', '', '| Norm | IID gap | Non-IID gap |', '|---|---:|---:|']
    for stage, label in (('convergence', 'BN'), ('normalization', 'LN')):
        lines.append(f"| {label} | {100*gaps[stage]['iid']:.2f} pp | {100*gaps[stage]['dirichlet']:.2f} pp |")
    lines += ['', '## Measured local compute', '', '| Lane / norm | Training + validation seconds | Examples processed | Optimizer steps | Parameters |', '|---|---:|---:|---:|---:|']
    for lane, pair in results.items():
        for stage, label in (('convergence', 'BN'), ('normalization', 'LN')):
            d = pair[stage]
            lines.append(f"| {lane} / {label} | {d['step_seconds_sum']:.1f} | {d['examples_processed']:,} | {d['optimizer_steps']:,} | {d['num_parameters']:,} |")
    lines += ['', 'Host timing excludes setup/checkpoint IO. Controls were measured separately under uncontrolled host load; these are not causal speed comparisons or edge measurements.', '',
              '## Verification', '', '- Complete histories, selected metrics and checkpoint/scaler/partition hashes verified.',
              '- Paired source, data, package versions, features, class counts, scaler and partitions match; only normalization differs in settings/model configuration.',
              '- No test evaluation. Single-seed validation selection is exploratory, not significance evidence.',
              '- Eleven of twelve screening runs complete; no confirmation runs in this stage.', '',
              'Full class precision/recall/F1, confusion matrices and manifests: [summary.json](summary.json).', '',
              '![Validation trajectories](curves.png)', '']
    (args.output / 'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, lane in zip(axes, lanes):
        for stage, label in (('convergence', 'BatchNorm'), ('normalization', 'LayerNorm')):
            history = histories[lane, stage]
            ax.plot([r['step'] for r in history], [r['validation_metrics']['macro_f1'] for r in history], label=label)
        ax.set(title=lane, xlabel='Epoch' if lane == 'light' else 'Round', ylabel='Validation macro-F1', ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Normalization ablation — fixed seed 0; validation only')
    fig.savefig(args.output / 'curves.png', dpi=150)
    plt.close(fig)
    print(json.dumps({lane: {**pair['change'], 'ln_best_step': pair['normalization']['best_step'], 'ln_macro_f1': pair['normalization']['validation_metrics']['macro_f1']} for lane, pair in results.items()}, indent=2))


if __name__ == '__main__':
    main()
