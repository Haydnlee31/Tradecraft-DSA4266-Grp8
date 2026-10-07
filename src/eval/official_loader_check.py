"""Exercise all train/validation shards locally, without training or test access.

The temporary full-training scaler is diagnostic only and is not saved for future
learning-curve runs: each selected training subset must fit its own scaler.
"""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_loader import OfficialBatches
from src.data.official_shards import save_json


def check(root, output, batch_size=8192, buffer_rows=65536):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Use a new loader-check receipt')
    started = time.monotonic()
    train = OfficialBatches(root, 'train', batch_size=batch_size)
    print('Fitting diagnostic scaler on training only...', flush=True)
    scaler = train.fit_scaler()
    fit_seconds = time.monotonic() - started
    result = {'manifest_sha256': sha256(root/'manifest.json'), 'checker_sha256': sha256(__file__),
              'protocol_sha256': sha256(root/'protocol.json'),
              'materialized_source_files': len(train.manifest['files']),
              'materialized_shards': sum(len(r['shards']) for r in train.manifest['files']),
              'materialized_counts': train.manifest['counts_by_split_and_class'],
              'exact_digest_disjointness_verified': train.manifest['exact_digest_disjointness_verified'],
              'loader_sha256': sha256(Path(__file__).parents[1]/'data'/'official_loader.py'),
              'status': 'incomplete', 'test_opened': False, 'test_evaluated': False,
              'model_trained': False, 'batch_size': batch_size, 'shuffle_buffer_rows': buffer_rows,
              'diagnostic_scaler_saved': False, 'scaler_fit_rows': int(scaler.n_samples_seen_),
              'scaler_fit_seconds': fit_seconds, 'splits': {}}
    for split in ('train', 'val'):
        data = train if split == 'train' else OfficialBatches(root, split, batch_size=batch_size)
        counts = np.zeros(len(CLASSES), dtype=np.int64)
        maximum = 0
        split_started = time.monotonic()
        for x, y in data.batches(scaler, seed=426639 if split == 'train' else None,
                                 shuffle_buffer_rows=buffer_rows):
            if x.dtype != np.float32 or x.shape[1] != 39 or y.dtype != np.int64:
                raise ValueError('Model input contract failed')
            counts += np.bincount(y, minlength=len(CLASSES))
            maximum = max(maximum, len(y))
        observed = dict(zip(CLASSES, map(int, counts)))
        if observed != data.manifest['counts_by_split_and_class'][split]:
            raise ValueError('Loader dropped or repeated rows')
        result['splits'][split] = {'counts': observed, 'rows': int(counts.sum()),
                                   'max_batch_rows': maximum, 'seconds': time.monotonic()-split_started}
        print(f'{split}: verified {int(counts.sum()):,} finite standardized rows', flush=True)
    if result['scaler_fit_rows'] != train.rows:
        raise ValueError('Scaler saw an unexpected number of training rows')
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['process_peak_rss_bytes'] = int(peak if platform.system() == 'Darwin' else peak*1024)
    except ImportError:
        result['process_peak_rss_bytes'] = None  # Not available on every platform.
    result.update(status='complete', elapsed_seconds=time.monotonic()-started,
                  platform=platform.platform(), python=platform.python_version())
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.data, args.output), indent=2))


if __name__ == '__main__':
    main()
