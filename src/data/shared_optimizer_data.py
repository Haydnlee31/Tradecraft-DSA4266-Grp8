"""Training-only data boundary for the shared-optimizer performance screen.

Historical training rows are not independent new data. We nevertheless reserve
new development/confirmation boundaries before training and fit the scaler ONLY
on the new fit cohort. Never instantiate PackedData: it opens original validation.
Raw shards are streamed into an ignored disk-backed staging array, not RAM.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_quality import fingerprints
from src.data.official_subsets import SubsetBatches
from src.federated.shared_adam import require
from src.models.research import atomic_json

ROOT = Path(__file__).parents[2]
PLAN = ROOT / 'reports/federated_shared_optimizer/performance-plan.json'


def digest_array(values):
    """Use void bytes, not fixed strings, so trailing zero bytes are retained."""
    return np.frombuffer(b''.join(fingerprints(values)), dtype='V32').copy()


def split_rows(digests, domain, fit_rows):
    # The official unique-vector view must really be unique. Fail rather than
    # splitting a repeated raw vector or arbitrarily breaking a duplicate group.
    require(len(np.unique(digests)) == len(digests), 'Raw feature vectors are not unique')
    keys = np.array([hashlib.sha256(domain.encode()+d.tobytes()).digest()
                     for d in digests], dtype='V32')
    values = np.frombuffer(keys.tobytes(), dtype='>u8').reshape(-1, 4)[:, 0]
    split = np.where(values < (1 << 64)*3//5, 0,
                     np.where(values < (1 << 64)*4//5, 1, 2)).astype(np.int8)
    eligible = np.flatnonzero(split == 0)
    require(len(eligible) >= fit_rows, 'Not enough fit-eligible rows')
    chosen = eligible[np.argsort(keys[eligible])[:fit_rows]]
    split[eligible] = -1  # Unused fit-eligible rows never enter evaluation.
    split[chosen] = 0
    return split


def remove_precision_collisions(split, hashes):
    """Fit has priority, then development, then confirmation; ignore unused rows.

    Float32 conversion can merge originally distinct real-valued vectors. Drop
    lower-priority held-out copies without changing fit membership or its scaler.
    This label-independent rule is frozen before any predictions are produced.
    """
    result = split.copy()
    seen = np.unique(hashes[result == 0])
    removed = {}
    for label, name in ((1, 'development'), (2, 'confirmation')):
        rows = np.flatnonzero(result == label)
        collides = np.isin(hashes[rows], seen)
        result[rows[collides]] = -1
        removed[name] = int(collides.sum())
        seen = np.union1d(seen, hashes[rows[~collides]])
    return result, removed


def fit_scaler(raw, rows, batch_size=8192):
    scaler = StandardScaler()
    for start in range(0, len(rows), batch_size):
        scaler.partial_fit(raw[rows[start:start+batch_size]])
    return scaler


def owner_vector(parts, size):
    owner = np.full(size, -1, dtype=np.int16)
    for client, rows in enumerate(parts):
        require(rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer)
                and len(rows) > 0 and len(np.unique(rows)) == len(rows)
                and int(rows.min()) >= 0 and int(rows.max()) < size,
                'Invalid client row IDs')
        require(bool((owner[rows] == -1).all()), 'Overlapping client ownership')
        owner[rows] = client
    require(bool((owner >= 0).all()), 'Missing client ownership')
    return owner


def prepare(parent, subsets, historical_y, historical_x, historical_scaler, partition, output, plan):
    output = Path(output)
    require(not output.exists(), 'Preserve existing prepared data; choose a new destination')
    for path, key in ((Path(parent)/'manifest.json', 'parent_manifest_sha256'),
                      (Path(subsets)/'manifest.json', 'subset_manifest_sha256'),
                      (historical_y, 'historical_train_y_sha256'),
                      (historical_x, 'historical_train_x_sha256'),
                      (historical_scaler, 'historical_scaler_sha256'), (partition, 'partition_sha256')):
        require(sha256(path) == plan[key], f'Input identity differs: {key}')
    data = SubsetBatches(parent, subsets, plan['historical_training_rows'])
    old_y = np.load(historical_y, mmap_mode='r', allow_pickle=False)
    old_x = np.load(historical_x, mmap_mode='r', allow_pickle=False)
    old_scaler = json.loads(Path(historical_scaler).read_text())
    require(old_y.shape == (data.rows,), 'Historical label shape differs')
    with np.load(partition, allow_pickle=False) as z:
        require(set(z.files) == {f'client_{i}' for i in range(plan['clients'])}, 'Client keys differ')
        owner = owner_vector([z[f'client_{i}'] for i in range(plan['clients'])], data.rows)
    output.mkdir(parents=True)
    atomic_json({'status': 'preparing', 'test_opened': False}, output/'manifest.json')
    raw = np.lib.format.open_memmap(output/'raw-training-staging.npy', mode='w+',
                                    dtype='<f8', shape=(data.rows, len(data.features)))
    digests = np.empty(data.rows, dtype='V32')
    offset = 0
    for x, y in data.raw_batches():
        end = offset+len(y)
        require(np.array_equal(y, old_y[offset:end]), 'Historical row order labels differ')
        # Reproduce old preprocessing ONLY to prove exact row order/ownership.
        # This old scaler never supplies the new model's inputs or loss weights.
        expected = ((x-np.array(old_scaler['mean']))/np.array(old_scaler['scale'])).astype(np.float32)
        require(np.array_equal(expected, old_x[offset:end]), 'Historical row feature identity differs')
        raw[offset:end] = x
        digests[offset:end] = digest_array(x)
        offset = end
    require(offset == data.rows, 'Raw view row count differs')
    raw.flush()
    split = split_rows(digests, plan['split_domain'], plan['fit_rows'])
    fit_ids = np.flatnonzero(split == 0)
    scaler = fit_scaler(raw, fit_ids)
    precision_hashes = np.empty(data.rows, dtype='V32')
    # Only fingerprints, not predictions, are computed on reserved rows.
    for start in range(0, data.rows, 8192):
        x = scaler.transform(raw[start:start+8192]).astype(np.float32)
        require(bool(np.isfinite(x).all()), 'Nonfinite standardized input')
        precision_hashes[start:start+len(x)] = digest_array(x)
    split, removed = remove_precision_collisions(split, precision_hashes)
    counts, files = {}, {}
    for label, name in ((0, 'fit'), (1, 'development'), (2, 'confirmation')):
        rows = np.flatnonzero(split == label)
        np.save(output/f'{name}_row_ids.npy', rows, allow_pickle=False)
        counts[name] = dict(zip(CLASSES, np.bincount(old_y[rows], minlength=len(CLASSES)).tolist()))
        require(min(counts[name].values()) > 0, f'{name} omits a class')
        if name == 'confirmation':
            # Reserved row IDs only: there is deliberately no confirmation_x/y.
            continue
        xout = np.lib.format.open_memmap(output/f'{name}_x.npy', mode='w+', dtype='<f4',
                                         shape=(len(rows), len(data.features)))
        for start in range(0, len(rows), 8192):
            xout[start:start+8192] = scaler.transform(raw[rows[start:start+8192]]).astype(np.float32)
        xout.flush()
        np.save(output/f'{name}_y.npy', np.array(old_y[rows]), allow_pickle=False)
    client_ids = owner[fit_ids]
    parts = [np.flatnonzero(client_ids == i) for i in range(plan['clients'])]
    owner_vector(parts, len(fit_ids))
    np.savez_compressed(output/'clients.npz', **{f'client_{i}': r for i, r in enumerate(parts)})
    atomic_json({'mean': scaler.mean_.tolist(), 'var': scaler.var_.tolist(),
                 'scale': scaler.scale_.tolist(), 'n_samples_seen': int(scaler.n_samples_seen_)},
                output/'scaler.json')
    for path in sorted(output.iterdir()):
        if path.name not in ('manifest.json', 'raw-training-staging.npy'):
            files[path.name] = sha256(path)
    result = {'status': 'complete', 'plan_sha256': sha256(PLAN), 'builder_sha256': sha256(__file__),
              'features': data.features, 'classes': CLASSES, 'files': files, 'counts': counts,
              'fit_client_counts': [len(r) for r in parts], 'float32_collision_rows_removed': removed,
              'raw_unique_vectors': data.rows, 'split_domain': plan['split_domain'],
              'source': {k: v for k, v in plan.items() if k.endswith('_sha256')},
              'original_validation_opened': False, 'test_opened': False,
              'confirmation_scored': False, 'model_trained': False}
    atomic_json(result, output/'manifest.json')
    return result


class DevelopmentData:
    """Whitelist two arrays per allowed split; no generic holdout selector."""
    def __init__(self, root, plan):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        m = self.manifest
        require(m['status'] == 'complete' and m['plan_sha256'] == sha256(PLAN)
                and m['classes'] == CLASSES, 'Prepared-data contract differs')
        for name, expected in m['files'].items():
            require(Path(name).name == name and sha256(self.root/name) == expected, 'Data checksum differs')
        self.arrays = {}
        for name in ('fit', 'development'):
            x, y = [np.load(self.root/f'{name}_{suffix}.npy', mmap_mode='r', allow_pickle=False)
                    for suffix in ('x', 'y')]
            require(x.shape == (len(y), len(m['features'])) and x.dtype == np.float32
                    and y.dtype == np.int64, 'Prepared array shape or dtype differs')
            require(np.bincount(y, minlength=len(CLASSES)).tolist() == list(m['counts'][name].values()),
                    'Prepared class counts differ')
            self.arrays[name] = (x, y)
        with np.load(self.root/'clients.npz', allow_pickle=False) as z:
            self.parts = [z[f'client_{i}'] for i in range(plan['clients'])]
        owner_vector(self.parts, len(self.arrays['fit'][1]))
        require([len(p) for p in self.parts] == m['fit_client_counts'], 'Fit ownership differs')

    def batch(self, rows):
        import torch
        x, y = self.arrays['fit']
        return torch.from_numpy(np.array(x[rows], copy=True)), torch.from_numpy(np.array(y[rows], copy=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--subsets', type=Path, required=True)
    parser.add_argument('--historical-y', type=Path, required=True)
    parser.add_argument('--historical-x', type=Path, required=True)
    parser.add_argument('--historical-scaler', type=Path, required=True)
    parser.add_argument('--partition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args), plan=json.loads(PLAN.read_text())), indent=2))


if __name__ == '__main__':
    main()
