"""Reconstruct training exposure from archived seeded client shuffles, without training.

Only training labels and index arrays are used. The validation and test splits
are not read. This is processed-example exposure, not an effective gradient weight.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.federated.partition import check_exact_cover
from src.models.research import atomic_json
from src.utils.seed import client_seed


def selected_rows(rows, batch_size, seed, cap):
    """Mirror PackedData shuffling and singleton-tail merging without feature IO."""
    if batch_size < 2 or cap < 0:
        raise ValueError('Invalid batch size or cap')
    order = np.array(rows, dtype=np.int64, copy=True)
    np.random.default_rng(seed).shuffle(order)
    end = min(len(order), cap*batch_size) if cap else len(order)
    if len(order)-end == 1:
        end += 1
    return order[:end]


def summarize(labels, exposure):
    result = {}
    for code, name in enumerate(CLASSES):
        values = exposure[labels == code]
        result[name] = {'assigned_unique': len(values), 'processed_examples': int(values.sum()),
                        'seen_unique': int(np.count_nonzero(values)),
                        'unseen_unique': int(np.count_nonzero(values == 0)),
                        'minimum_repeats': int(values.min()) if len(values) else None,
                        'maximum_repeats': int(values.max()) if len(values) else None}
    return result


def audit(data_root, archive, output):
    data_root, output = Path(data_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((data_root/'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['classes'] != CLASSES:
        raise ValueError('Completed packed data required')
    if sha256(data_root/'train_y.npy') != manifest['files']['train_y.npy']:
        raise ValueError('Training label checksum mismatch')
    labels = np.load(data_root/'train_y.npy', mmap_mode='r', allow_pickle=False)
    if labels.shape != (manifest['train_rows'],) or labels.dtype != np.int64:
        raise ValueError('Invalid training labels')
    result = {'status': 'incomplete', 'archive_sha256': sha256(archive),
              'packed_manifest_sha256': sha256(data_root/'manifest.json'),
              'auditor_sha256': sha256(__file__), 'numpy': np.__version__,
              'validation_read': False, 'test_read': False, 'runs': {}}
    with tarfile.open(archive) as bundle:
        for folder in ('official39-bounded-bridge-v1', 'official39-bounded-comparison-v1'):
            for lane in ('iid', 'dirichlet'):
                root = f'outputs/{folder}/{lane}'
                def read(name):
                    return bundle.extractfile(root+'/'+name).read()
                env = json.loads(read('environment.json'))
                r = json.loads(read('result.json'))
                settings = env['settings']
                if json.loads(read('status.json')) != {'status': 'complete', 'step': settings['epochs']}:
                    raise ValueError('Completed run required')
                if len(r['history']) != settings['epochs']:
                    raise ValueError('Missing training history')
                # These two sources define the actual row selection being
                # reconstructed. Do not claim exact replay if they have changed.
                for name in ('data/official_packed.py', 'utils/seed.py'):
                    if sha256(Path(__file__).parents[1]/name) != env['source_sha256'][name]:
                        raise ValueError('Batching/seeding source changed')
                if env['packed_manifest_sha256'] != result['packed_manifest_sha256']:
                    raise ValueError('Data provenance mismatch')
                if env['packages']['numpy'] != np.__version__:
                    raise ValueError('Reconstruction requires the recorded NumPy version')
                raw = read('assignments.npz')
                info = json.loads(read('partition.json'))
                if hashlib.sha256(raw).hexdigest() != info['sha256']:
                    raise ValueError('Assignment checksum mismatch')
                with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
                    parts = [saved[f'client_{i}'].copy() for i in range(settings['clients'])]
                check_exact_cover(parts, len(labels))
                if [np.bincount(labels[p], minlength=8).tolist() for p in parts] != info['counts']:
                    raise ValueError('Client label counts mismatch')
                exposure = np.zeros(len(labels), dtype=np.int64)
                for expected_step, h in enumerate(r['history'], 1):
                    if h['step'] != expected_step or len(h['client_work']) != len(parts):
                        raise ValueError('Invalid history or client participation')
                    for client, rows in enumerate(parts):
                        chosen = selected_rows(rows, settings['batch_size'],
                            client_seed(settings['seed'], client, h['step']), h['local_batch_cap'])
                        work = h['client_work'][client]
                        if work['client'] != client or len(chosen) != work['examples_processed']:
                            raise ValueError('Reconstructed exposure differs from recorded work')
                        exposure[chosen] += 1
                if int(exposure.sum()) != r['examples_processed']:
                    raise ValueError('Total exposure mismatch')
                result['runs'][folder+'/'+lane] = {
                    'settings': settings, 'class_exposure': summarize(labels, exposure),
                    'client_class_exposure': [summarize(labels[p], exposure[p]) for p in parts],
                    'examples_processed': int(exposure.sum()),
                    'exposure_sha256': hashlib.sha256(exposure.astype('<i8').tobytes()).hexdigest()}
    result['status'] = 'complete'
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(result, output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    audit(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
