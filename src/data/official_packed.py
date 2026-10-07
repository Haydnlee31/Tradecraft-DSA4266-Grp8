"""Pack a selected official-39 cohort into disk-backed model-input arrays.

This removes thousands of Parquet opens per epoch. Only a batch of feature rows
is copied into Python RAM; the operating system manages mapped file pages. Global
shuffle/client indices still cost O(N) RAM (8 bytes per index), with an explicit
five-million training-row cap. This is not constant total process memory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_loader import OfficialBatches
from src.data.official_subsets import SubsetBatches
from src.data.official_shards import save_json


def prepare(parent, subsets, checks, size, output):
    parent, subsets, checks, output = map(Path, (parent, subsets, checks, output))
    if not 1 <= size <= 5000000:
        raise ValueError('Reviewed training cap is five million rows')
    train = SubsetBatches(parent, subsets, size)
    check = json.loads((checks/'check.json').read_text())
    scaler_path = checks/f'scaler-{size}.json'
    if (check['status'] != 'complete' or check['parent_manifest_sha256'] != sha256(parent/'manifest.json')
            or check['subset_manifest_sha256'] != sha256(subsets/'manifest.json')
            or check['subset_recipe_sha256'] != sha256(subsets/'recipe.json')
            or check['sizes'][str(size)]['scaler_sha256'] != sha256(scaler_path)):
        raise ValueError('Subset/scaler check receipt mismatch')
    receipt = json.loads(scaler_path.read_text())
    if receipt['features'] != train.features or receipt['n_samples_seen'] != size or receipt['fit_split'] != 'train':
        raise ValueError('Scaler was not fitted on this training subset')
    scaler = StandardScaler()
    scaler.mean_, scaler.var_, scaler.scale_ = (np.asarray(receipt[key]) for key in ('mean', 'var', 'scale'))
    scaler.n_features_in_, scaler.n_samples_seen_ = 39, size
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'scaler.json', receipt)
    manifest = {'status': 'incomplete', 'features': train.features, 'classes': CLASSES,
                'subset_size': size, 'train_class_counts': train.class_counts, 'files': {},
                'source_sha256': {'parent_manifest': sha256(parent/'manifest.json'),
                                  'subset_manifest': sha256(subsets/'manifest.json'),
                                  'subset_recipe': sha256(subsets/'recipe.json'),
                                  'scaler_check': sha256(checks/'check.json')},
                'builder_sha256': sha256(__file__), 'scaler_sha256': sha256(output/'scaler.json'),
                'test_opened': False, 'feature_dtype': 'float32 after float64 train-only scaling',
                'row_order': 'source view shard order, then record order; shuffle indices at training time'}
    save_json(output/'manifest.json', manifest)
    for split, data in (('train', train), ('val', OfficialBatches(parent, 'val'))):
        x = np.lib.format.open_memmap(output/f'{split}_x.npy', mode='w+', dtype=np.float32, shape=(data.rows, 39))
        y = np.lib.format.open_memmap(output/f'{split}_y.npy', mode='w+', dtype=np.int64, shape=(data.rows,))
        offset = 0
        for xb, yb in data.batches(scaler):
            n = len(yb)
            x[offset:offset+n], y[offset:offset+n] = xb, yb
            offset += n
        x.flush()
        y.flush()
        if offset != data.rows:
            raise ValueError('Packed row count mismatch')
        del x, y
        # Independent source replay checks values/order and catches write mistakes.
        x, y = (np.load(output/f'{split}_{key}.npy', mmap_mode='r') for key in ('x', 'y'))
        offset = 0
        for xb, yb in data.batches(scaler):
            n = len(yb)
            if not np.array_equal(x[offset:offset+n], xb) or not np.array_equal(y[offset:offset+n], yb):
                raise ValueError('Packed array differs from source/scaler replay')
            offset += n
        for key in ('x', 'y'):
            name = f'{split}_{key}.npy'
            manifest['files'][name] = sha256(output/name)
        manifest[f'{split}_rows'] = data.rows
        print(f'{split}: {data.rows:,} packed rows match source replay', flush=True)
    manifest['status'] = 'complete'
    save_json(output/'manifest.json', manifest)
    return manifest


class PackedData:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        m = self.manifest
        if m['status'] != 'complete' or len(m['features']) != 39 or m['classes'] != CLASSES:
            raise ValueError('Completed official-39 packed data required')
        if sha256(self.root/'scaler.json') != m['scaler_sha256']:
            raise ValueError('Packed scaler checksum changed')
        self.arrays = {}
        for split in ('train', 'val'):
            pair = []
            for key in ('x', 'y'):
                name = f'{split}_{key}.npy'
                if sha256(self.root/name) != m['files'][name]:
                    raise ValueError('Packed array checksum changed')
                pair.append(np.load(self.root/name, mmap_mode='r', allow_pickle=False))
            x, y = pair
            if x.shape != (m[f'{split}_rows'], 39) or y.shape != (len(x),) or x.dtype != np.float32 or y.dtype != np.int64:
                raise ValueError('Packed shape/dtype mismatch')
            self.arrays[split] = pair

    def batches(self, split, batch_size, seed=None, rows=None):
        if split not in ('train', 'val') or batch_size < 2:
            raise ValueError('Only train/val and batch size >=2 are supported')
        x, y = self.arrays[split]
        # Permute indices, not the feature matrix; every selected row appears once.
        order = np.arange(len(y)) if rows is None else np.array(rows, dtype=np.int64, copy=True)
        if seed is not None:
            np.random.default_rng(seed).shuffle(order)
        start = 0
        while start < len(order):
            end = min(start+batch_size, len(order))
            # Merge a final singleton into the previous batch for BatchNorm. No
            # row is dropped; the largest batch can be batch_size+1.
            if len(order)-end == 1:
                end += 1
            take = order[start:end]
            yield np.array(x[take], copy=True), np.array(y[take], copy=True)
            start = end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--subsets', type=Path, required=True)
    parser.add_argument('--checks', type=Path, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.parent, args.subsets, args.checks, args.size, args.output)


if __name__ == '__main__':
    main()
