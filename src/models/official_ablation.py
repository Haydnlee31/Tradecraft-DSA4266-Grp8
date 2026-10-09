"""Opt-in standardized-input masks and a shared, single-forward validation view.

The model still has 39 input slots. Zero means the training mean in standardized
coordinates. Never apply a mask only at evaluation: the same transform is used
for each training batch, full validation batch, and subsequent inference.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_ablation_panel import load_panel, write_json


ARMS = {'full39': [], 'number_masked': ['Number'], 'number_total_masked': ['Number', 'Tot sum']}


def transform_inputs(x, features, arm):
    if arm not in ARMS or len(features) != 39 or len(set(features)) != 39:
        raise ValueError('Unknown ablation arm/feature schema')
    if x.ndim != 2 or x.shape[1] != 39 or x.dtype != np.float32:
        raise ValueError('Expected a float32 batch with 39 standardized inputs')
    if any(name not in features for name in ARMS[arm]):
        raise ValueError('Masked feature is missing')
    # Copy even if the caller hands us a view into a read-only mmap. Input packs,
    # shared validation data, and another client's batch must never be mutated.
    transformed = np.array(x, copy=True)
    for name in ARMS[arm]:
        transformed[:, features.index(name)] = 0.
    return transformed


class AblationData:
    def __init__(self, base, arm, panel_root):
        if arm not in ARMS:
            raise ValueError('Unknown ablation arm')
        self.base, self.arm = base, arm
        self.root, self.manifest, self.arrays = base.root, base.manifest, base.arrays
        features = self.manifest['features']
        # Validate names even before the first batch and before any output write.
        transform_inputs(np.empty((0, 39), np.float32), features, arm)
        self.panel_mask, panel_info = load_panel(panel_root, base)
        self.metadata = dict(protocol='official39-input-mask-v1', arm=arm,
                             zero_after_scaling=ARMS[arm], features=features, input_slots=39,
                             value=0., applies_to='train_validation_and_inference',
                             panel=panel_info, primary_checkpoint='last.pt',
                             secondary_checkpoint='best.pt selected by all-validation macro-F1',
                             evaluation_policy='one full-validation forward; panel counts use the same logits')

    def batches(self, split, batch_size, seed=None, rows=None):
        for x, y in self.base.batches(split, batch_size, seed, rows):
            yield transform_inputs(x, self.manifest['features'], self.arm), y


def initial_state(model, output, resume):
    """Hash initial values, not a Torch archive whose metadata may vary."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, list(array.shape), str(array.dtype)]).encode())
        digest.update(array.tobytes())
    record = dict(model_state_sha256=digest.hexdigest(), num_parameters=model.num_parameters())
    path = Path(output)/'initialization.json'
    if resume:
        if json.loads(path.read_text()) != record:
            raise ValueError('Initial model state changed')
    else:
        write_json(path, record)
    return record


def error_counts(metrics):
    cm = np.asarray(metrics['confusion_matrix'], dtype=np.int64)
    benign = CLASSES.index('Benign')
    attack = np.arange(len(CLASSES)) != benign
    missed = int(cm[attack, benign].sum())
    errors = int(cm[attack].sum()-cm.diagonal()[attack].sum())
    return dict(attacks_missed_as_benign=missed, attacks_given_wrong_attack_category=errors-missed)


def evaluate_views(model, batches, device, panel_mask):
    """Avoid re-batching retained rows: all views use identical forward logits.

    The original all-validation reduction is kept exactly. The second view
    only filters already computed predictions/loss inputs. It consumes no RNG,
    performs no extra model forward, and cannot alter training or early stopping
    (the runner has no early stopping). Validation-best remains secondary.
    """
    from src.models.official_streaming import confusion_metrics
    if panel_mask.ndim != 1 or panel_mask.dtype != bool or not panel_mask.any():
        raise ValueError('Nonempty boolean validation panel required')
    model.eval()
    full_cm = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    panel_cm = np.zeros_like(full_cm)
    full_loss, panel_loss, offset = 0., 0., 0
    with torch.no_grad():
        for x, y in batches:
            take = panel_mask[offset:offset+len(y)]
            if len(take) != len(y):
                raise ValueError('Validation iterator exceeds panel length')
            logits = model(torch.from_numpy(x).to(device))
            targets = torch.from_numpy(y).to(device)
            loss = torch.nn.functional.cross_entropy(logits, targets, reduction='sum')
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite validation loss')
            predicted = logits.argmax(1).cpu().numpy()
            full_loss += loss.item()
            full_cm += np.bincount(y*len(CLASSES)+predicted, minlength=len(CLASSES)**2).reshape(full_cm.shape)
            if take.any():
                selected = torch.from_numpy(take.copy()).to(device)
                selected_loss = torch.nn.functional.cross_entropy(logits[selected], targets[selected], reduction='sum')
                if not torch.isfinite(selected_loss):
                    raise ValueError('Nonfinite panel validation loss')
                panel_loss += selected_loss.item()
                panel_cm += np.bincount(y[take]*len(CLASSES)+predicted[take], minlength=len(CLASSES)**2).reshape(panel_cm.shape)
            offset += len(y)
    if offset != len(panel_mask) or not full_cm.sum() or not panel_cm.sum():
        raise ValueError('Validation iterator does not cover the shared panel')
    return (full_loss/int(full_cm.sum()), confusion_metrics(full_cm),
            panel_loss/int(panel_cm.sum()), confusion_metrics(panel_cm))
