"""Verify paired loss controls and generate validation-only comparison artifacts.

Run from the repository root. The output directory must not already exist.
No checkpoint is loaded and no training/test data is read by this analysis.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/mlp-tuning'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan_path = Path('configs/local_tuning_plan.json')
    plan = read(plan_path)
    # Reuse the exact promotion rules used for the convergence report.
    spec = importlib.util.spec_from_file_location('convergence', Path(__file__).resolve().parents[1] / 'convergence_2026_10_03/analyze.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    lanes = ('heavy', 'light', 'iid', 'dirichlet')
    results, histories = {}, {}
    for lane in lanes:
        pair, manifests = {}, {}
        for stage in ('convergence', 'loss'):
            folder = args.root / stage / lane
            result, manifest = read(folder / 'result.json'), read(folder / 'environment.json')
            assert read(folder / 'status.json')['status'] == 'complete'
            assert result['steps'] == 60 and result['test_metrics'] is None
            assert set(manifest['split_sha256']) == {'train', 'val'}
            expected = {**plan['common'], 'loss': 'ce' if stage == 'loss' else 'sqrt_weighted_ce'}
            for key, value in expected.items():
                assert manifest['settings'][key] == value, (lane, stage, key)
            assert [row['step'] for row in result['history']] == list(range(1, 61))
            selected = previous.best(result['history'])
            assert selected['step'] == result['best_step']
            assert selected['validation_metrics'] == result['validation_metrics']
            assert digest(folder / 'best.pt') == result['checkpoint_sha256']
            assert digest(folder / 'scaler.joblib') == manifest['scaler_sha256']
            if lane in ('iid', 'dirichlet'):
                assert digest(folder / 'assignments.npz') == manifest['partition_sha256']
            pair[stage] = {k: result[k] for k in ('best_step', 'validation_metrics', 'checkpoint_sha256', 'step_seconds_sum', 'examples_processed', 'optimizer_steps', 'process_peak_rss_bytes')}
            pair[stage]['result_sha256'] = digest(folder / 'result.json')
            manifests[stage] = manifest
            histories[lane, stage] = result['history']
        before, after = manifests['convergence'], manifests['loss']
        # Loss must be the only configuration difference in each paired run.
        assert {k: v for k, v in before['settings'].items() if k != 'loss'} == {k: v for k, v in after['settings'].items() if k != 'loss'}
        for key in ('source_sha256', 'split_sha256', 'packages', 'feature_columns', 'classes', 'model_config', 'train_class_counts', 'scaler_sha256', 'partition_sha256'):
            assert before[key] == after[key], (lane, key)
        pair['change'] = previous.compare(pair['convergence'], pair['loss'], plan)
        pair['manifests'] = manifests
        results[lane] = pair
    summary = {'stage': 'loss', 'test_evaluated': False, 'screen_runs_completed': 8,
               'screen_runs_remaining': plan['max_screen_runs'] - 8,
               'plan_sha256': digest(plan_path), 'lanes': results}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['# Loss-function comparison — 2026-10-05', '',
             'Validation only, model/partition seed 0, 60 epochs or rounds, no early stopping. Both losses select their first best validation macro-F1 checkpoint. CE means unweighted cross-entropy; weighted means square-root class-weighted CE.', '',
             '| Lane | Weighted F1 (step) | CE F1 (step) | Gain (pp) | Gate A / B |',
             '|---|---:|---:|---:|---|']
    for lane, pair in results.items():
        a, b, d = pair['convergence'], pair['loss'], pair['change']
        lines.append(f"| {lane} | {a['validation_metrics']['macro_f1']:.4f} ({a['best_step']}) | {b['validation_metrics']['macro_f1']:.4f} ({b['best_step']}) | {100*d['macro_f1_gain']:+.2f} | {d['promotion_gate_a']} / {d['promotion_gate_b']} |")
    lines += ['', 'Gates are project preferences, not statistical significance or deployment tests. A: gain ≥1 pp macro-F1, false-alert increase ≤2 pp, every class recall drop ≤2 pp. B: false-alert reduction ≥5 pp, macro-F1 drop ≤1 pp, Web/Brute Force recall drops ≤2 pp.', '', '## Selected-checkpoint recall and false alerts', '',
              '| Lane / loss | False alerts | ' + ' | '.join(before['classes']) + ' |',
              '|---|' + '---:|' * (1 + len(before['classes']))]
    for lane, pair in results.items():
        for stage, label in (('convergence', 'weighted'), ('loss', 'CE')):
            metrics = pair[stage]['validation_metrics']
            values = [metrics['benign_false_alert_rate']] + [metrics['per_class_recall'][c] for c in before['classes']]
            lines.append(f'| {lane} / {label} | ' + ' | '.join(f'{100*v:.2f}%' for v in values) + ' |')
    lines += ['', 'Full precision, recall, F1, confusion matrices, deltas and provenance are in [summary.json](summary.json).', '',
              '## Measured local training cost', '',
              '| Lane / loss | Train + validation seconds | Examples processed | Optimizer steps |', '|---|---:|---:|---:|']
    for lane, pair in results.items():
        for stage in ('convergence', 'loss'):
            d = pair[stage]
            lines.append(f"| {lane} / {stage} | {d['step_seconds_sum']:.1f} | {d['examples_processed']:,} | {d['optimizer_steps']:,} |")
    lines += ['', 'Timing excludes setup/checkpoint IO and is measured on this CPU host, not edge hardware. Epochs and FL rounds are not compute-equivalent.', '',
              '## Verification and limitations', '',
              '- Verified complete 60-step histories, best-checkpoint selection and checkpoint/scaler/partition hashes.',
              '- Paired settings differ only in loss; source, data, packages, features, model configuration, scaler and partitions match.',
              '- Single-seed validation selection is exploratory. No significance claim, no test access, no cloud use.',
              '- Eight of twelve screening runs completed; four remain. No multi-seed confirmations yet.',
              '- All source comments and prior outputs are preserved; no model/training code changes were needed.', '',
              '![Validation macro-F1 trajectories](curves.png)', '']
    (args.output / 'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, lane in zip(axes.flat, lanes):
        for stage, label in (('convergence', 'Weighted'), ('loss', 'CE')):
            history = histories[lane, stage]
            ax.plot([r['step'] for r in history], [r['validation_metrics']['macro_f1'] for r in history], label=label)
        ax.set(title=lane, xlabel='Epoch' if lane in ('heavy', 'light') else 'Round', ylabel='Validation macro-F1', ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Loss ablation — fixed seed 0; validation only')
    fig.savefig(args.output / 'curves.png', dpi=150)
    plt.close(fig)
    print(json.dumps({lane: {**pair['change'], 'ce_best_step': pair['loss']['best_step'], 'ce_macro_f1': pair['loss']['validation_metrics']['macro_f1']} for lane, pair in results.items()}, indent=2))


if __name__ == '__main__':
    main()
