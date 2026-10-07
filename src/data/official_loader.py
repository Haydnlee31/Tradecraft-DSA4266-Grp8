"""Bounded-memory official-39 batches; independent of the legacy 46 inputs.

This is a data-access primitive, not a finished cloud training pipeline. Training
subset selection must happen BEFORE fitting its scaler. Test access is blocked
unless a caller explicitly authorizes final evaluation. Metadata never becomes
a model feature. Optional shuffling permutes shards and each bounded buffer;
it is reproducible but is NOT a uniform shuffle of the entire dataset.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256


class OfficialBatches:
    def __init__(self, root, split, batch_size=8192, allow_test=False):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest['status'] != 'complete':
            raise ValueError('Only completed, verified materializations can be loaded')
        protocol_path = self.root / 'protocol.json'
        protocol = json.loads(protocol_path.read_text())
        if (sha256(protocol_path) != self.manifest['protocol_sha256']
                or protocol['features'] != self.manifest['features']
                or protocol['counts_by_split_and_class'] != self.manifest['counts_by_split_and_class']):
            raise ValueError('Materialization disagrees with frozen protocol')
        if split not in ('train', 'val', 'test') or batch_size < 1:
            raise ValueError('Invalid split or batch size')
        if split == 'test' and not allow_test:
            raise ValueError('Test data is sealed until explicit final evaluation')
        self.split, self.batch_size = split, batch_size
        self.features = self.manifest['features']
        if len(self.features) != 39 or self.manifest['classes'] != CLASSES:
            raise ValueError('Unexpected official-39 feature or class contract')
        self.shards = [shard for record in self.manifest['files'] for shard in record['shards'] if shard['split'] == split]
        self.rows = sum(shard['rows'] for shard in self.shards)
        for shard in self.shards:
            if not (self.root / shard['path']).resolve().is_relative_to(self.root):
                raise ValueError('Shard path escapes materialization directory')

    def raw_batches(self, seed=None):
        order = np.arange(len(self.shards))
        if seed is not None:
            np.random.default_rng(seed).shuffle(order)
        for index in order:
            path = self.root / self.shards[index]['path']
            if sha256(path) != self.shards[index]['sha256']:
                raise ValueError('Shard checksum changed since materialization')
            for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=self.batch_size, columns=self.features + ['_class_id']):
                x = np.column_stack([batch.column(name).to_numpy() for name in self.features])
                y = batch.column('_class_id').to_numpy().astype(np.int64)
                if not np.isfinite(x).all():
                    raise ValueError('Nonfinite materialized features')
                yield x, y

    def fit_scaler(self):
        if self.split != 'train':
            raise ValueError('Scalers may only be fitted on the selected training data')
        scaler = StandardScaler()
        for x, _ in self.raw_batches():
            scaler.partial_fit(x)
        if not hasattr(scaler, 'mean_'):
            raise ValueError('Empty training data')
        return scaler

    def batches(self, scaler, seed=None, shuffle_buffer_rows=65536):
        """Yield float32 standardized features and int64 targets, without loss.

        Passing a seed activates bounded block shuffling. Shuffled source shards
        mix capture order, but class-correlated windows can remain; a balanced
        subset/sampler is a separate training-design gate. No mid-epoch resume
        promise: recreating this iterator with the same seed restarts its epoch.
        """
        if shuffle_buffer_rows < self.batch_size:
            raise ValueError('Shuffle buffer must hold at least one output batch')
        rng = np.random.default_rng(seed)
        xs, ys, filled = [], [], 0

        def emit():
            x, y = np.concatenate(xs), np.concatenate(ys)
            order = rng.permutation(len(y)) if seed is not None else np.arange(len(y))
            for start in range(0, len(y), self.batch_size):
                take = order[start:start+self.batch_size]
                # Scaling in float64 BEFORE narrowing avoids needless precision loss.
                standardized = scaler.transform(x[take]).astype(np.float32)
                if not np.isfinite(standardized).all():
                    raise ValueError('Nonfinite standardized model inputs')
                yield standardized, y[take]

        for x, y in self.raw_batches(seed=seed):
            offset = 0
            while offset < len(y):
                n = min(shuffle_buffer_rows-filled, len(y)-offset)
                xs.append(x[offset:offset+n])
                ys.append(y[offset:offset+n])
                offset += n
                filled += n
                if filled == shuffle_buffer_rows:
                    yield from emit()
                    xs, ys, filled = [], [], 0
        if filled:
            yield from emit()
