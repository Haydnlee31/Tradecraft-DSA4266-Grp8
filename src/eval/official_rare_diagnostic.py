"""Read-only CPU diagnosis of final 2M checkpoints, never a training/tuning run.

Rare classes are evaluated completely; other classes use a fixed bounded probe.
Probe precision is deliberately omitted because stratification changes prevalence.
Univariate training AUCs describe feature separation, not held-out model quality,
feature importance, or proof of an irreducible error floor.
"""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.research import atomic_json

RARE = ('Web-based', 'Brute Force')
COMPETITORS = ('Benign', 'Recon', 'Spoofing')


def rank_summary(logits, target):
    """Ranks use descending logits; probabilities are uncalibrated model scores."""
    logits = np.asarray(logits)
    if (logits.ndim != 2 or logits.shape[1] != len(CLASSES) or not len(logits)
            or not 0 <= target < len(CLASSES) or not np.isfinite(logits).all()):
        raise ValueError('Finite nonempty logit matrix required')
    rank = np.argsort(-logits, axis=1, kind='stable')
    ranks = np.argmax(rank == target, axis=1) + 1
    other = logits.copy()
    other[:, target] = -np.inf
    margin = logits[:, target] - other.max(axis=1)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability = exp[:, target] / exp.sum(axis=1)
    return {'rows': len(logits), 'recall': float((ranks == 1).mean()),
            'top2_recall': float((ranks <= 2).mean()), 'top3_recall': float((ranks <= 3).mean()),
            'median_true_vs_best_other_logit': float(np.median(margin)),
            'mean_true_class_uncalibrated_score': float(probability.mean()),
            'predicted_counts': dict(zip(CLASSES, map(int, np.bincount(rank[:, 0], minlength=len(CLASSES)))))}


def select_rows(labels, cap=5000):
    if cap < 1:
        raise ValueError('Positive probe cap required')
    rng = np.random.default_rng(426639)
    selected = {}
    for i, name in enumerate(CLASSES):
        rows = np.flatnonzero(labels == i)
        # Include ALL rare examples, not just easy or randomly lucky examples.
        selected[name] = rows if name in RARE else np.sort(rng.choice(rows, min(cap, len(rows)), replace=False))
    return selected


def rare_packed_overlap(arrays):
    """Exact numeric float32 matches involving rare rows, using bounded scans.

    Raw float64 deduplication does not rule out new equal vectors after float32
    packing. This observes model-input collisions, not source/session leakage.
    Only rare-vector keys and 8-class counts are retained; full arrays stay mapped.
    """
    rare_keys, counts = {}, {}
    for split, (x, y) in arrays.items():
        for name in RARE:
            keys = []
            for row in x[np.flatnonzero(y == CLASSES.index(name))]:
                row = row.copy()
                row[row == 0] = 0  # Canonicalize signed zero for numeric equality.
                key = row.tobytes()
                counts.setdefault(key, {s: np.zeros(len(CLASSES), dtype=np.int64) for s in arrays})
                keys.append(key)
            rare_keys[split, name] = keys
    for split, (x, y) in arrays.items():
        for start in range(0, len(y), 8192):
            block = np.array(x[start:start+8192], copy=True)
            block[block == 0] = 0
            for row, label in zip(block, y[start:start+8192]):
                key = row.tobytes()
                if key in counts:
                    counts[key][split][int(label)] += 1
    result = {}
    for (split, name), keys in rare_keys.items():
        c = CLASSES.index(name)
        result[f'{split}/{name}'] = {'rows': len(keys), 'unique_packed_vectors': len(set(keys))}
        for pool in arrays:
            result[f'{split}/{name}'][f'rows_matching_any_{pool}_vector'] = sum(counts[k][pool].sum() > 0 for k in keys)
            result[f'{split}/{name}'][f'rows_matching_other_class_in_{pool}'] = sum(counts[k][pool].sum() > counts[k][pool][c] for k in keys)
        result[f'{split}/{name}'] = {k: int(v) for k, v in result[f'{split}/{name}'].items()}
    return result


def diagnose(data_root, reference, confirmation, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data = PackedData(data_root)
    if data.manifest['train_rows'] != 2000000:
        raise ValueError('This diagnostic is frozen for the 2M cohort')
    rows = {split: select_rows(data.arrays[split][1]) for split in ('train', 'val')}
    result = {'status': 'complete', 'test_opened': False, 'model_trained': False,
              'device': 'cpu', 'script_sha256': sha256(__file__),
              'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scikit-learn')},
              'packed_manifest_sha256': sha256(Path(data_root)/'manifest.json'),
              'archive_sha256': {'reference': sha256(reference), 'confirmation': sha256(confirmation)},
              'scope': 'All rare train/val rows; up to 5000 rows per other class; no probe precision',
              'models': {}, 'checkpoint_sha256': {}, 'training_feature_separation': {}}
    result['rare_float32_collisions'] = rare_packed_overlap(data.arrays)
    torch.set_num_threads(2)
    for seed in (7, 17, 27):
        with tarfile.open(reference if seed == 7 else confirmation) as archive:
            for lane in ('light', 'iid'):
                root = ('outputs/official39-2m-reference-v1/'+lane if seed == 7 else
                        f'outputs/official39-scaling-confirmation-v1/2m/{lane}-seed{seed}')
                env_bytes = archive.extractfile(root+'/environment.json').read()
                env = json.loads(env_bytes)
                checkpoint_bytes = archive.extractfile(root+'/last.pt').read()
                checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
                result['checkpoint_sha256'][f'{lane}/seed{seed}'] = hashlib.sha256(checkpoint_bytes).hexdigest()
                if (checkpoint['environment_sha256'] != hashlib.sha256(env_bytes).hexdigest()
                        or env['packed_manifest_sha256'] != result['packed_manifest_sha256']
                        or checkpoint['step'] != 20 or env['settings']['seed'] != seed
                        or env['settings']['lane'] != lane or env['test_evaluated']):
                    raise ValueError('Checkpoint provenance mismatch')
                model = MLPClassifier(39, len(CLASSES), MLPConfig(**env['model_config']))
                model.load_state_dict(checkpoint['model'])
                model.eval()
                measured = {}
                with torch.inference_mode():
                    for split in rows:
                        measured[split] = {}
                        for name, indices in rows[split].items():
                            logits = np.concatenate([model(torch.from_numpy(x)).numpy()
                                for x, _ in data.batches(split, 512, rows=indices)])
                            measured[split][name] = rank_summary(logits, CLASSES.index(name))
                result['models'][f'{lane}/seed{seed}'] = measured
                # Allow at most one changed rare prediction across CPU/GPU math;
                # this is an inference diagnostic, not a CUDA recovery guarantee.
                for name in RARE:
                    expected = checkpoint['history'][-1]['validation_metrics']['per_class'][name]['recall']
                    observed = measured['val'][name]
                    if abs(expected-observed['recall']) > 1/observed['rows'] + 1e-12:
                        raise ValueError('CPU rare recall disagrees with archived GPU endpoint')
                print(f'{lane} seed {seed}: train/validation probes complete', flush=True)
    # Train-only one-feature comparisons against the major confusion destinations.
    # Direction is made symmetric so a small feature value can also separate classes.
    x = data.arrays['train'][0]
    for rare in RARE:
        for other in COMPETITORS:
            a, b = np.asarray(x[rows['train'][rare]]), np.asarray(x[rows['train'][other]])
            labels = np.r_[np.ones(len(a)), np.zeros(len(b))]
            pair = {}
            for col, name in enumerate(data.manifest['features']):
                auc = roc_auc_score(labels, np.r_[a[:, col], b[:, col]])
                pair[name] = {'symmetric_train_auc': float(max(auc, 1-auc)),
                              'rare_q10_q50_q90': np.quantile(a[:, col], [.1, .5, .9]).tolist(),
                              'other_q10_q50_q90': np.quantile(b[:, col], [.1, .5, .9]).tolist()}
            result['training_feature_separation'][f'{rare} vs {other}'] = pair
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(result, output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'reference', 'confirmation', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    diagnose(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
