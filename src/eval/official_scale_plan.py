"""Freeze a two-size, full-epoch experiment without training or reading test data.

The primary work control matches processed training examples, NOT exact optimizer
updates. Exact update counts are computed from real client sizes and singleton
batch handling. Repartitioning is deterministic but not nested across sizes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.data.official_inventory import sha256
from src.data.label_map import CLASSES
from src.data.official_packed import PackedData
from src.models.official_streaming import partitions
from src.models.research import atomic_json


def batches_per_pass(n, batch_size=512):
    if n < 2 or batch_size < 2:
        raise ValueError('At least two rows and batch size >=2 required')
    return (n+batch_size-1)//batch_size - int(n > batch_size and n % batch_size == 1)


def budgets(sizes, epochs):
    return {'steps': epochs, 'examples_processed': sum(sizes)*epochs,
            'optimizer_steps': sum(batches_per_pass(n) for n in sizes)*epochs}


def plan(small, large, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    roots = [Path(small), Path(large)]
    data = [PackedData(p) for p in roots]
    a, b = [d.manifest for d in data]
    if [a['train_rows'], b['train_rows']] != [500000, 2000000]:
        raise ValueError('This reviewed plan is restricted to 500k and 2M cohorts')
    for key in ('parent_manifest', 'subset_manifest', 'subset_recipe', 'scaler_check'):
        if a['source_sha256'][key] != b['source_sha256'][key]:
            raise ValueError('Cohorts do not share the verified source/subset recipe')
    if (a['features'] != b['features'] or a['classes'] != b['classes']
            or a['val_rows'] != b['val_rows'] or a['files']['val_y.npy'] != b['files']['val_y.npy']):
        raise ValueError('Validation population/schema mismatch')
    for root, d in zip(roots, data):
        scaler = json.loads((root/'scaler.json').read_text())
        if scaler['fit_split'] != 'train' or scaler['n_samples_seen'] != d.manifest['train_rows']:
            raise ValueError('Cohort-specific training-only scaler required')
    result = {'status': 'prepared_not_trained', 'planner_sha256': sha256(__file__),
              'numpy': np.__version__, 'test_opened': False,
              'validation_population': 'same parent split and identical ordered label hash; features scaled separately',
              'work_control': 'exact processed examples, NOT exact updates, rounds, time or communication',
              'settings': {'seed': 7, 'partition_seed': 7, 'clients': 20, 'alpha': .5,
                           'batch_size': 512, 'lr': .001, 'weight_decay': 1e-5,
                           'loss': 'sqrt_weighted_ce', 'loss_reduction': 'batch_weight_sum',
                           'normalization': 'layer', 'local_max_batches': 0,
                           'total_update_budget': 0, 'threads': 2, 'device': 'cuda'},
              'cohorts': {}, 'comparisons': {}, 'partition_profiles': {}}
    profiles = {}
    for root, d in zip(roots, data):
        n = d.manifest['train_rows']
        result['cohorts'][str(n)] = {'path': str(root), 'manifest_sha256': sha256(root/'manifest.json'),
                                    'scaler_sha256': sha256(root/'scaler.json'),
                                    'train_class_counts': d.manifest['train_class_counts']}
        profiles[n] = {}
        for lane in ('light', 'iid', 'dirichlet'):
            if lane == 'light':
                sizes = [n]
            else:
                parts, _ = partitions(d.arrays['train'][1], lane, 20, 7, .5)
                counts = np.array([np.bincount(d.arrays['train'][1][p], minlength=8) for p in parts])
                profiles[n][lane] = counts
                sizes = list(map(len, parts))
                result['partition_profiles'][f'{n}/{lane}'] = {
                    'client_class_counts': counts.tolist(), 'min_rows': min(sizes), 'max_rows': max(sizes),
                    'clients_holding_class': dict(zip(CLASSES, map(int, (counts>0).sum(axis=0))))}
            result['comparisons'][f'{n}/{lane}'] = {
                'fixed_epoch_endpoint': budgets(sizes, 20),
                'matched_examples_endpoint': budgets(sizes, 20 if n == 500000 else 5)}
    result['partition_caveat'] = 'Same recipe/seed, not nested client assignments. Do not attribute FL differences solely to row count.'
    # Quantify how much each class moves among numbered clients. This is not a
    # distance between physical devices: clients are simulated and IDs arbitrary.
    result['class_allocation_total_variation'] = {}
    for lane in ('iid', 'dirichlet'):
        x, y = profiles[500000][lane], profiles[2000000][lane]
        tv = .5*np.abs(x/x.sum(axis=0)-y/y.sum(axis=0)).sum(axis=0)
        result['class_allocation_total_variation'][lane] = dict(zip(CLASSES, map(float, tv)))
    result['run_policy'] = {
        'new_training_runs': 2, 'lanes': ['light', 'iid'],
        'deferred_lane': 'dirichlet',
        'deferred_reason': 'Observed non-IID mixtures change substantially across sizes; design a profile-controlled comparison before launch',
        'large_epochs': 20, 'pause_after': 5,
        'checkpoint_preservation': 'Preserve last.pt and environment/history at step 5 before resuming the SAME run to step 20',
        'primary_metrics': 'final-at-budget metrics, all eight recalls and precision plus benign false alerts',
        'best_checkpoint_metrics': 'descriptive only; unequal selection opportunities across work-matched endpoints',
        'automatic_extension': False, 'five_million_authorized': False,
        'interpretation': 'Exploratory single-seed within-collection scaling including cohort-specific preprocessing and repartitioning'}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(result, output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--small', type=Path, required=True)
    p.add_argument('--large', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    plan(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
