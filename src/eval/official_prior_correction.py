"""One fixed prior correction on six frozen models, with no training or search.

Equal-class exposure deliberately supports rare attacks. Correcting those scores
back toward the original cohort proportions changes that decision trade-off;
it is not a guaranteed improvement or a calibration certificate. Class priors
come only from the hashed TRAINING manifest, never validation labels/predictions.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.official_panel_validation import load_validation, validation_batches, summarize
from src.eval.official_training_fit import state_hash
from src.eval.research_decision import error_breakdown
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_streaming import evaluate
from src.models.research import atomic_json

CONDITIONS = ('panel_control', 'majority_diversity')


def prior_offsets(counts):
    """Exact positive integer counts, no validation-derived estimate or smoothing.

    For uniform optimization exposure q(c)=1/8, Bayes prior reweighting gives
    score(c)=logit(c)+log(p(c)/q(c)). The -log(q) term is common to all classes,
    so this is the proposed logit+log(p) rule, up to an irrelevant constant.
    Retaining q makes equal training counts yield an exactly zero offset.
    Compute logs in float64, then freeze the actual float32 inference offsets.
    """
    if (set(counts) != set(CLASSES)
            or any(type(counts[c]) is not int or counts[c] <= 0 for c in CLASSES)):
        raise ValueError('Positive integer training counts for every canonical class required')
    values = np.asarray([counts[c] for c in CLASSES], dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(values.sum()):
        raise ValueError('Finite training counts required')
    prior = values/values.sum()
    offsets = np.log(prior*len(CLASSES)).astype(np.float32)
    if not np.isfinite(offsets).all():
        raise ValueError('Finite prior offsets required')
    return prior, offsets


class PriorAdjustedModel(torch.nn.Module):
    """Inference wrapper only: it never modifies the saved network parameters."""
    def __init__(self, model, offsets):
        super().__init__()
        values = np.asarray(offsets, dtype=np.float32)
        if values.shape != (len(CLASSES),) or not np.isfinite(values).all():
            raise ValueError('One finite offset per class required')
        self.model = model
        self.register_buffer('offsets', torch.from_numpy(values.copy()))

    def forward(self, x):
        return self.model(x)+self.offsets


def check_protocol(plan, reference, manifest):
    if (plan['protocol'] != 'official39-fixed-training-prior-correction-v1'
            or plan['conditions'] != list(CONDITIONS)
            or plan['score_rule'] != 'logit_plus_log_training_prior_over_uniform_exposure_prior'
            or plan['prior_source'] != 'packed_training_manifest_only' or plan['coefficient'] != 1.
            or plan['numerics'] != 'float64_log_ratio_cast_to_float32_before_logit_addition'
            or plan['require_all_raw_bridges_before_correction'] is not True
            or plan['model_trained'] is not False or plan['test_opened'] is not False
            or plan['model_promoted'] is not False or not 1 <= plan['threads'] <= 2
            or not 1 <= plan['evaluation_batch_size'] <= 8192
            or not 1 <= len(plan['seeds']) <= 3 or len(set(plan['seeds'])) != len(plan['seeds'])
            or not 1 <= plan['epochs'] <= 300
            or set(plan['validation_class_counts']) != set(CLASSES)
            or min(plan['validation_class_counts'].values()) <= 0
            or sum(plan['validation_class_counts'].values()) != plan['validation_rows']):
        raise ValueError('Plan exceeds the fixed inference-only prior check')
    if (reference['status'] != 'complete' or reference['device'] != 'cpu'
            or reference['plan_sha256'] != plan['reference_plan_sha256']
            or reference['all_control_bridges_exact'] is not True or reference['all_pairs_matched'] is not True
            or reference['test_opened'] is not False or reference['model_promoted'] is not False
            or reference['validation_opened'] is not True
            or reference['mlp_settings']['seeds'] != plan['seeds']
            or reference['mlp_settings']['epochs'] != plan['epochs']
            or reference['plan']['evaluation_batch_size'] != plan['evaluation_batch_size']
            or reference['classes'] != CLASSES or manifest['classes'] != CLASSES
            or manifest['status'] != 'complete' or manifest['test_opened'] is not False
            or len(manifest['features']) != 39 or len(set(manifest['features'])) != 39
            or manifest['val_rows'] != plan['validation_rows']):
        raise ValueError('Reference or packed provenance mismatch')
    prior_offsets(manifest['train_class_counts'])
    if sum(manifest['train_class_counts'].values()) != manifest['train_rows']:
        raise ValueError('Training counts do not sum to the recorded cohort')
    expected = [(c, s, f'{c}_seed{s}') for c in CONDITIONS for s in plan['seeds']]
    if ([(r['condition'], r['seed'], r['name']) for r in reference['fits']] != expected
            or [(r['condition'], r['seed'], r['name']) for r in reference['validation_results']] != expected):
        raise ValueError('Incomplete or duplicated frozen model coverage')
    for entry, validation in zip(reference['fits'], reference['validation_results']):
        fit, config = entry['training_fit'], entry['training_fit']['model_config']
        exposure = entry['exposure']['per_class']
        if (config['hidden_dims'] != [64, 32] or config['dropout'] != .2 or config['normalization'] != 'layer'
                or fit['seed'] != entry['seed'] or fit['num_parameters'] != 5096
                or len(fit['class_weights']) != len(CLASSES)
                or not np.allclose(fit['class_weights'], np.ones(len(CLASSES)), rtol=0, atol=1e-7)
                or len(set(fit['class_weights'])) != 1
                or len({exposure[c]['examples_processed'] for c in CLASSES}) != 1
                or min(exposure[c]['examples_processed'] for c in CLASSES) <= 0
                or sum(exposure[c]['examples_processed'] for c in CLASSES) != fit['examples_processed']
                or any(exposure[c]['available_rows'] != manifest['train_class_counts'][c] for c in CLASSES)
                or {c: validation['validation_metrics']['per_class'][c]['support'] for c in CLASSES}
                   != plan['validation_class_counts']):
            raise ValueError('Uniform training exposure or original cohort counts not verified')


def load_model(root, entry, reference, manifest):
    """Hash and validate metadata before restricted weights-only model loading."""
    root = Path(root).resolve()
    name = entry['checkpoint']
    if Path(name).name != name or not (root/name).resolve().is_relative_to(root):
        raise ValueError('Invalid checkpoint path')
    if sha256(root/name) != entry['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    saved = torch.load(root/name, map_location='cpu', weights_only=True)
    fit = entry['training_fit']
    if (json.loads(json.dumps(saved['model_config'])) != fit['model_config']
            or saved['condition'] != entry['condition'] or saved['seed'] != entry['seed']
            or saved['epochs'] != reference['mlp_settings']['epochs']
            or saved['panel_sha256'] != reference['panel_sha256']
            or saved['features'] != manifest['features'] or saved['classes'] != CLASSES):
        raise ValueError('Checkpoint metadata mismatch')
    model = MLPClassifier(39, len(CLASSES), MLPConfig(**saved['model_config']))
    model.load_state_dict(saved['state_dict'], strict=True)
    if state_hash(model) != fit['final_state_sha256'] or model.num_parameters() != fit['num_parameters']:
        raise ValueError('Final model parameters differ from the frozen fit')
    model.eval()
    return model


def diagnose(data_root, reference_root, plan_path, output):
    data_root, reference_root, output = Path(data_root), Path(reference_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(Path(plan_path).read_text())
    reference_path = reference_root/'receipt.json'
    if sha256(reference_path) != plan['reference_receipt_sha256']:
        raise ValueError('Reference checksum mismatch')
    reference = json.loads(reference_path.read_text())
    if sha256(data_root/'manifest.json') != reference['packed_manifest_sha256']:
        raise ValueError('Packed manifest checksum mismatch')
    manifest = json.loads((data_root/'manifest.json').read_text())
    check_protocol(plan, reference, manifest)
    if sha256(data_root/'scaler.json') != manifest['scaler_sha256']:
        raise ValueError('Scaler checksum mismatch')
    source_root, sources = Path(__file__).parents[1], dict(reference['source_sha256'])
    for name, old in sources.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(source_root/name) != old:
            raise ValueError('Frozen shared source changed')
    for name in ('eval/official_prior_correction.py', 'eval/research_decision.py', 'eval/decision.py'):
        sources[name] = sha256(source_root/name)
    # These are already recorded training counts. Do not open or rehash train
    # arrays, and never iterate generically over all manifest split files.
    prior, offsets = prior_offsets(manifest['train_class_counts'])
    models = [load_model(reference_root, e, reference, manifest) for e in reference['fits']]
    receipt = {'status': 'incomplete', 'purpose': plan['purpose'], 'plan': plan,
               'plan_sha256': sha256(plan_path), 'source_sha256': sources,
               'packed_manifest_sha256': reference['packed_manifest_sha256'],
               'reference_receipt_sha256': plan['reference_receipt_sha256'],
               'classes': CLASSES, 'device': 'cpu', 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scikit-learn', 'threadpoolctl')},
               'training_class_counts': manifest['train_class_counts'], 'target_prior': dict(zip(CLASSES, prior.tolist())),
               'source_exposure_prior': dict.fromkeys(CLASSES, 1/len(CLASSES)),
               'applied_offsets_float32': dict(zip(CLASSES, offsets.tolist())),
               'training_arrays_opened': False, 'model_trained': False, 'test_opened': False,
               'model_promoted': False, 'independent_assessment': False,
               'all_raw_bridges_exact': False, 'validation_opened': True, 'raw_results': [], 'corrected_results': []}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(receipt, output/'receipt.json')

    def check_inputs():
        if (sha256(plan_path) != receipt['plan_sha256']
                or sha256(reference_path) != plan['reference_receipt_sha256']
                or sha256(data_root/'manifest.json') != receipt['packed_manifest_sha256']
                or sha256(data_root/'scaler.json') != manifest['scaler_sha256']
                or any(sha256(source_root/p) != h for p, h in sources.items())
                or any(sha256(reference_root/e['checkpoint']) != e['checkpoint_sha256'] for e in reference['fits'])):
            raise ValueError('Frozen inputs changed during inference')

    x, y = load_validation(data_root, manifest, plan)
    torch.set_num_threads(plan['threads'])
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=plan['threads']):
        for mode in ('raw', 'corrected'):
            if mode == 'corrected':
                if receipt['all_raw_bridges_exact'] is not True:
                    raise ValueError('All raw bridges must pass before corrected inference')
                check_inputs()
            for entry, old, model in zip(reference['fits'], reference['validation_results'], models):
                scorer = model if mode == 'raw' else PriorAdjustedModel(model, offsets)
                ce, metrics = evaluate(scorer, validation_batches(x, y, plan['evaluation_batch_size']), 'cpu')
                if mode == 'raw' and (ce != old['validation_cross_entropy'] or metrics != old['validation_metrics']):
                    raise ValueError('Raw bridge failed; no prior correction was evaluated')
                if state_hash(model) != entry['training_fit']['final_state_sha256']:
                    raise ValueError('Inference changed model parameters')
                receipt[mode+'_results'].append({'name': entry['name'], 'condition': entry['condition'],
                    'seed': entry['seed'], 'mode': mode, 'checkpoint_sha256': entry['checkpoint_sha256'],
                    'final_state_sha256': entry['training_fit']['final_state_sha256'],
                    'validation_cross_entropy': ce, 'validation_metrics': metrics,
                    'error_breakdown': error_breakdown(np.asarray(metrics['confusion_matrix']))})
                atomic_json(receipt, output/'receipt.json')
                print(f'{mode} {entry["name"]}: macro-F1={metrics["macro_f1"]:.6f} FAR={metrics["benign_false_alert_rate"]:.4%}', flush=True)
            if mode == 'raw':
                receipt['all_raw_bridges_exact'] = True
                atomic_json(receipt, output/'receipt.json')
    check_inputs()
    for name in ('val_x.npy', 'val_y.npy'):
        if sha256(data_root/name) != manifest['files'][name]:
            raise ValueError('Validation array changed during inference')
    receipt['summary'] = {mode: summarize(receipt[mode+'_results']) for mode in ('raw', 'corrected')}
    receipt['status'] = 'complete'
    atomic_json(receipt, output/'receipt.json')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'reference-root', 'plan-path', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    diagnose(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
