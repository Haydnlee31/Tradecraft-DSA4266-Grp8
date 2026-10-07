"""Verify each nested training view and save its own train-only scaler receipt."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import sklearn

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_loader import OfficialBatches
from src.data.official_subsets import SubsetBatches
from src.data.official_shards import save_json


def check(parent, subsets, output):
    parent, subsets, output = Path(parent), Path(subsets), Path(output)
    if output.exists():
        raise FileExistsError('Use a new subset-check output')
    recipe = json.loads((subsets/'recipe.json').read_text())
    output.mkdir(parents=True)
    result = {'status': 'incomplete', 'subset_recipe_sha256': sha256(subsets/'recipe.json'),
              'subset_manifest_sha256': sha256(subsets/'manifest.json'),
              'parent_manifest_sha256': sha256(parent/'manifest.json'),
              'checker_sha256': sha256(__file__), 'sklearn_version': sklearn.__version__,
              'test_opened': False, 'model_trained': False, 'training_ready': False, 'sizes': {}}
    save_json(output/'check.json', result)
    for size in recipe['sizes']:
        started = time.monotonic()
        data = SubsetBatches(parent, subsets, size)
        scaler = data.fit_scaler()
        if int(scaler.n_samples_seen_) != size:
            raise ValueError('Scaler saw a different training size')
        counts = np.zeros(len(CLASSES), dtype=np.int64)
        for x, y in data.batches(scaler, seed=426639):
            if x.shape[1] != 39 or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError('Subset model input contract failed')
            counts += np.bincount(y, minlength=len(CLASSES))
        observed = dict(zip(CLASSES, map(int, counts)))
        if observed != data.class_counts:
            raise ValueError('Subset loader dropped or repeated rows')
        # Validation is transformed with THIS training scaler, never fitted on.
        validation = OfficialBatches(parent, 'val')
        val_counts = np.zeros(len(CLASSES), dtype=np.int64)
        for x, y in validation.batches(scaler):
            val_counts += np.bincount(y, minlength=len(CLASSES))
        if dict(zip(CLASSES, map(int, val_counts))) != validation.manifest['counts_by_split_and_class']['val']:
            raise ValueError('Validation count mismatch')
        scaler_receipt = {'kind': 'sklearn.StandardScaler', 'sklearn_version': sklearn.__version__,
                          'subset_recipe_sha256': result['subset_recipe_sha256'],
                          'subset_manifest_sha256': result['subset_manifest_sha256'],
                          'parent_manifest_sha256': result['parent_manifest_sha256'],
                          'subset_size': size, 'features': data.features, 'fit_split': 'train',
                          'n_samples_seen': int(scaler.n_samples_seen_),
                          'n_features_in': int(scaler.n_features_in_),
                          'mean': scaler.mean_.tolist(), 'var': scaler.var_.tolist(),
                          'scale': scaler.scale_.tolist()}
        scaler_path = output/f'scaler-{size}.json'
        save_json(scaler_path, scaler_receipt)
        result['sizes'][str(size)] = {'class_counts': observed, 'scaler_fit_rows': size,
                                     'scaler_path': scaler_path.name, 'scaler_sha256': sha256(scaler_path),
                                     'validation_rows_checked': int(val_counts.sum()),
                                     'seconds': time.monotonic()-started}
        save_json(output/'check.json', result)
        print(f'{size:,}: all training rows and {int(val_counts.sum()):,} validation rows passed', flush=True)
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['process_peak_rss_bytes'] = int(peak if platform.system() == 'Darwin' else peak*1024)
    except ImportError:
        result['process_peak_rss_bytes'] = None
    result.update(status='complete', next_gate='matched streaming trainers, client partitions and checkpoint recovery')
    save_json(output/'check.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--subsets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.parent, args.subsets, args.output), indent=2))


if __name__ == '__main__':
    main()
