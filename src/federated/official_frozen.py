"""Load reviewed nested client assignments without redrawing a partition.

Paths are not identity: content hashes bind the cohort, plan and assignments.
This lets an unchanged handoff move from a laptop to cloud storage, while a
changed plan or row assignment invalidates checkpoint recovery.
"""
import json
from pathlib import Path

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.federated.partition import check_exact_cover


def load_frozen(root, data, clients):
    root = Path(root)
    plan_path = root/'plan.json'
    plan_hash = sha256(plan_path)
    plan = json.loads(plan_path.read_text())
    if (plan['status'] != 'prepared_not_trained' or plan['classes'] != CLASSES
            or plan['test_opened'] or not plan['class_support_preserved']):
        raise ValueError('Unreviewed frozen partition plan')
    # Fixed filenames avoid accepting arbitrary paths supplied in a plan.
    filenames = {'500k-assignments.npz', '2m-assignments.npz', 'small_to_large.npy'}
    if set(plan['files']) != filenames:
        raise ValueError('Unexpected frozen partition files')
    for name in filenames:
        if sha256(root/name) != plan['files'][name]:
            raise ValueError(f'Frozen partition checksum changed: {name}')
    data_hash = sha256(data.root/'manifest.json')
    matches = [name for name, digest in plan['packed_manifest_sha256'].items() if digest == data_hash]
    if len(matches) != 1 or matches[0] not in ('500k', '2m'):
        raise ValueError('Frozen partition belongs to a different cohort')
    cohort = matches[0]
    filename = f'{cohort}-assignments.npz'
    with np.load(root/filename, allow_pickle=False) as saved:
        if set(saved.files) != {f'client_{i}' for i in range(clients)}:
            raise ValueError('Frozen client count differs from training settings')
        parts = [saved[f'client_{i}'].copy() for i in range(clients)]
    for rows in parts:
        if (rows.ndim != 1 or rows.dtype != np.int64 or len(rows) < 2
                or np.any(rows[1:] <= rows[:-1])):
            raise ValueError('Frozen rows must be sorted unique int64 indices, at least two per client')
    check_exact_cover(parts, data.manifest['train_rows'])
    expected = plan['anchor_counts' if cohort == '500k' else 'expanded_counts']
    counts = [np.bincount(data.arrays['train'][1][rows], minlength=len(CLASSES)).tolist() for rows in parts]
    if counts != expected:
        raise ValueError('Frozen class counts differ from packed labels')
    if sha256(plan_path) != plan_hash:
        raise ValueError('Frozen plan changed while loading')
    return parts, {'policy': 'nested-anchor-class-shares-v1', 'cohort': cohort,
                   'plan_sha256': plan_hash, 'assignment_sha256': plan['files'][filename]}
