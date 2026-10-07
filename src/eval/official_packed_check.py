"""Check packed cohort batches and proposed client partitions without training."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.data.official_shards import save_json
from src.models.official_streaming import partitions
from src.models.research import memory_metrics


def check(root, output, clients=20, seed=7, alpha=.5):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Use a new packed-check receipt')
    started = time.monotonic()
    data = PackedData(root)
    result = {'status': 'incomplete', 'packed_manifest_sha256': sha256(root/'manifest.json'),
              'checker_sha256': sha256(__file__), 'test_opened': False, 'model_trained': False,
              'cuda_checked': False, 'batch_size': 512, 'seed': seed, 'clients': clients,
              'alpha': alpha, 'counts': {}, 'partitions': {}}
    for split in ('train', 'val'):
        counts = np.zeros(8, dtype=np.int64)
        for x, y in data.batches(split, 512, seed if split == 'train' else None):
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite packed batch')
            counts += np.bincount(y, minlength=8)
        if int(counts.sum()) != data.manifest[f'{split}_rows']:
            raise ValueError('Packed loader row count mismatch')
        result['counts'][split] = dict(zip(CLASSES, map(int, counts)))
    if result['counts']['train'] != data.manifest['train_class_counts']:
        raise ValueError('Packed training class counts mismatch')
    y = data.arrays['train'][1]
    for lane in ('iid', 'dirichlet'):
        parts, attempts = partitions(y, lane, clients, seed, alpha)
        counts = np.array([np.bincount(y[rows], minlength=8) for rows in parts])
        result['partitions'][lane] = {'attempts': attempts, 'exact_cover': True,
                                      'min_client_rows': min(map(len, parts)), 'max_client_rows': max(map(len, parts)),
                                      'client_class_counts': counts.tolist(),
                                      'clients_holding_class': dict(zip(CLASSES, map(int, (counts > 0).sum(axis=0))))}
    result.update(status='complete', elapsed_seconds=time.monotonic()-started, **memory_metrics('cpu'))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.data, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'partitions'}, indent=2))


if __name__ == '__main__':
    main()
