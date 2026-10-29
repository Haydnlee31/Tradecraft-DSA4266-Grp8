"""Read-only, memory-mapped adapter for the prepared optimizer fit/dev bundle.

Adapted from extensions/fedntd/data.py (same whitelist and checks) so this
extension does not depend on an unmerged branch. It never rebuilds splits or
scalers and never opens confirmation or original validation/test arrays, even
when their checksums appear in the upstream manifest.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from extensions.fedlc.core import require
from src.data.label_map import CLASSES

ROOT = Path(__file__).resolve().parents[2]
PARENT_PLAN = ROOT/'reports/federated_shared_optimizer/performance-plan.json'
ALLOWED_FILES = ('fit_x.npy', 'fit_y.npy', 'development_x.npy', 'development_y.npy',
                 'fit_row_ids.npy', 'development_row_ids.npy', 'clients.npz', 'scaler.json')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class DevelopmentData:
    def __init__(self, root, base, parent_plan=PARENT_PLAN, features=39):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        m = self.manifest
        require(m['status'] == 'complete' and m['plan_sha256'] == sha256(parent_plan)
                and m['classes'] == CLASSES, 'Prepared-data contract differs')
        require(len(m['features']) == features and len(set(m['features'])) == features,
                'Feature schema differs')
        self.file_hashes = {}
        for name in ALLOWED_FILES:
            path = self.root/name
            require(path.resolve().parent == self.root and not path.is_symlink(),
                    'Prepared files must be regular in-directory files')
            actual = sha256(path)
            require(actual == m['files'][name], f'Data checksum differs: {name}')
            self.file_hashes[name] = actual
        self.arrays, self.row_ids = {}, {}
        for split in ('fit', 'development'):
            x, y, rows = [np.load(self.root/f'{split}_{suffix}.npy', mmap_mode='r', allow_pickle=False)
                          for suffix in ('x', 'y', 'row_ids')]
            require(x.shape == (len(y), features) and x.dtype == np.float32
                    and y.ndim == 1 and y.dtype == np.int64 and len(y) > 0,
                    'Prepared array shape/dtype differs')
            require(bool(((y >= 0) & (y < len(CLASSES))).all()), 'Invalid labels')
            require(np.bincount(y, minlength=len(CLASSES)).tolist()
                    == [m['counts'][split][c] for c in CLASSES], 'Class counts differ')
            require(rows.shape == y.shape and np.issubdtype(rows.dtype, np.integer)
                    and bool((rows >= 0).all()) and len(np.unique(rows)) == len(rows),
                    'Invalid original row IDs')
            for start in range(0, len(y), 8192):
                require(bool(np.isfinite(x[start:start+8192]).all()), 'Nonfinite prepared features')
            self.arrays[split] = (x, y)
            self.row_ids[split] = rows
        require(len(self.arrays['fit'][1]) == base['fit_rows'], 'Fit size differs')
        require(not np.intersect1d(*self.row_ids.values()).size, 'Fit/development row overlap')
        with np.load(self.root/'clients.npz', allow_pickle=False) as z:
            require(set(z.files) == {f'client_{i}' for i in range(base['clients'])}, 'Client keys differ')
            self.parts = [z[f'client_{i}'].copy() for i in range(base['clients'])]
        owner = np.zeros(base['fit_rows'], dtype=np.int8)
        for rows in self.parts:
            require(rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer) and len(rows) > 0
                    and len(np.unique(rows)) == len(rows) and int(rows.min()) >= 0
                    and int(rows.max()) < len(owner), 'Invalid client rows')
            require(bool((owner[rows] == 0).all()), 'Overlapping client ownership')
            owner[rows] = 1
        require(bool((owner == 1).all()) and [len(p) for p in self.parts] == m['fit_client_counts'],
                'Client ownership must exactly cover fit')

    def batch(self, rows):
        return tuple(torch.from_numpy(np.array(a[rows], copy=True)) for a in self.arrays['fit'])
