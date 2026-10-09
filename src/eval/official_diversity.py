"""One matched-work training-diversity control, not a new model search.

The small balanced panel's class order and rare row IDs stay fixed. Only the
other six classes draw from larger TRAINING pools. This changes coverage and
repetition of those examples, not class prevalence, rare exposure or updates.
The archived fitter is untouched; the shared optimizer step is reused and the
new control loop must reproduce its complete historical record exactly.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval.official_dropout_fit import comparable
from src.eval.official_panel_validation import load_validation, validation_batches, summarize
from src.eval.official_training_fit import load_training_panel, panel_batches, state_hash
from src.models.architectures import LIGHT_CONFIG, MLPClassifier
from src.models.official_losses import build_official_criterion
from src.models.official_streaming import train_epoch, evaluate
from src.models.research import atomic_json, atomic_save
from src.utils.seed import set_seed

RARE_CLASSES = ('Web-based', 'Brute Force')
CONDITIONS = ('panel_control', 'majority_diversity')


class ClassCycle:
    """Visit each source row once before reshuffling this class's own pool.

    Crossing a cycle boundary can repeat a row within one batch. That is valid:
    the guarantee is per full pool traversal, not batch-level deduplication.
    Independent NumPy generators never consume the model's dropout RNG.
    """
    def __init__(self, pool, seed, class_id, namespace):
        self.pool = np.asarray(pool, dtype=np.int64)
        if self.pool.ndim != 1 or not len(self.pool) or len(np.unique(self.pool)) != len(self.pool):
            raise ValueError('A nonempty unique row pool is required')
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, namespace, class_id]))
        self.order, self.position = self.rng.permutation(self.pool), 0

    def take(self, count):
        if count < 0:
            raise ValueError('Nonnegative draw count required')
        chunks = []
        while count:
            if self.position == len(self.order):
                self.order = self.rng.permutation(self.pool)
                self.position = 0
            n = min(count, len(self.order)-self.position)
            chunks.append(self.order[self.position:self.position+n])
            self.position += n
            count -= n
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


class RowSchedule:
    """Stream row IDs with an auditable identical label/rare-position sequence."""
    def __init__(self, labels, panel_rows, seed, diverse, namespace=426639):
        self.labels, self.rows, self.seed = labels, np.asarray(panel_rows), seed
        if (self.rows.ndim != 1 or not len(self.rows) or not np.issubdtype(self.rows.dtype, np.integer)
                or np.any(self.rows < 0) or np.any(self.rows >= len(labels))
                or len(np.unique(self.rows)) != len(self.rows)):
            raise ValueError('Unique in-range panel rows required')
        self.panel_y = np.array(labels[self.rows], copy=True)
        counts = np.bincount(self.panel_y, minlength=len(CLASSES))
        if len(counts) != len(CLASSES) or min(counts) <= 0 or len(set(counts)) != 1:
            raise ValueError('Balanced eight-class panel required')
        self.rare_ids = [CLASSES.index(c) for c in RARE_CLASSES]
        self.cycles = {c: ClassCycle(np.flatnonzero(labels == c), seed, c, namespace)
                       for c in range(len(CLASSES)) if diverse and c not in self.rare_ids}
        self.exposure = np.zeros(len(labels), dtype=np.int64)
        self.label_hash, self.rare_hash, self.row_hash = (hashlib.sha256() for _ in range(3))
        self.batches = 0

    def epoch(self, epoch, batch_size):
        # Reusing panel_batches preserves shuffle seeds AND its singleton-tail
        # merging rule. It treats the one-column source IDs as a tiny feature.
        for row_column, target in panel_batches(self.rows[:, None], self.panel_y, batch_size, self.seed+epoch):
            rows = row_column[:, 0].copy()
            for c, cycle in self.cycles.items():
                mask = target == c
                rows[mask] = cycle.take(int(mask.sum()))
            if not np.array_equal(self.labels[rows], target):
                raise ValueError('Replacement changed batch class sequence')
            rare = np.isin(target, self.rare_ids)
            size = np.asarray([len(rows)], dtype='<i8').tobytes()
            self.label_hash.update(size+target.astype('<i8').tobytes())
            # The -1 placeholders preserve exact positions, not only a bag of
            # rare row IDs. Per-batch lengths also bind the update boundaries.
            self.rare_hash.update(size+np.where(rare, rows, -1).astype('<i8').tobytes())
            self.row_hash.update(size+rows.astype('<i8').tobytes())
            np.add.at(self.exposure, rows, 1)
            self.batches += 1
            yield rows, target

    def summary(self):
        classes = {}
        for c, name in enumerate(CLASSES):
            counts = self.exposure[self.labels == c]
            seen = counts[counts > 0]
            classes[name] = {'available_rows': len(counts), 'unique_rows_seen': len(seen),
                             'examples_processed': int(counts.sum()),
                             'min_exposure_among_seen': int(seen.min()) if len(seen) else 0,
                             'max_exposure': int(counts.max())}
        return {'label_sequence_sha256': self.label_hash.hexdigest(),
                'rare_position_sequence_sha256': self.rare_hash.hexdigest(),
                'row_sequence_sha256': self.row_hash.hexdigest(),
                'optimizer_steps': self.batches, 'per_class': classes}


def fit_condition(x, y, rows, settings, seed, diverse, namespace, historical=None):
    """Same neural recipe and observation schedule, with explicit row exposure."""
    panel_x, panel_y = np.array(x[rows], copy=True), np.array(y[rows], copy=True)
    schedule = RowSchedule(y, rows, seed, diverse, namespace)
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    config = replace(LIGHT_CONFIG, normalization=settings['normalization'])
    model = MLPClassifier(39, len(CLASSES), config)
    initial = state_hash(model)
    counts = dict(zip(CLASSES, map(int, np.bincount(panel_y, minlength=len(CLASSES)))))
    criterion = build_official_criterion(settings['loss'], counts, settings['loss_reduction'])
    if not torch.equal(criterion.weight, criterion.weight[0].expand(len(CLASSES))):
        raise ValueError('Matched balanced exposure requires uniform class weights')
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    observations, examples, updates = [], 0, 0
    start = time.perf_counter()

    def batches(epoch):
        for take, target in schedule.epoch(epoch, settings['batch_size']):
            xb = np.array(x[take], copy=True)
            if not np.isfinite(xb).all():
                raise ValueError('Nonfinite selected training features')
            yield xb, target

    for epoch in range(1, settings['epochs']+1):
        loss, seen, steps = train_epoch(model, optimizer, criterion, batches(epoch), 'cpu')
        if seen != len(rows):
            raise ValueError('Incomplete scheduled exposure')
        examples += seen
        updates += steps
        if epoch in settings['report_epochs']:
            # This is always the ORIGINAL panel, not the whole changing pool.
            # For treatment it measures fit on the reference panel, not accuracy
            # over all examples seen during training or generalization.
            ce, metrics = evaluate(model, panel_batches(panel_x, panel_y, 2048), 'cpu')
            observations.append({'epoch': epoch, 'train_batch_mean_loss': loss,
                                 'train_eval_cross_entropy': ce, 'training_panel_metrics': metrics})
            print(f'{"diverse" if diverse else "control"} seed={seed} epoch={epoch}: panel macro-F1={metrics["macro_f1"]:.6f}', flush=True)
        if historical and epoch == historical.get('prefix_epoch'):
            if state_hash(model) != historical['prefix_state_sha256']:
                raise ValueError('Control prefix differs from the historical parameters')
    batches_per_epoch = ((len(rows)+settings['batch_size']-1)//settings['batch_size']
                         - int(len(rows) > settings['batch_size'] and len(rows) % settings['batch_size'] == 1))
    if (examples != len(rows)*settings['epochs'] or updates != batches_per_epoch*settings['epochs']
            or updates != schedule.batches):
        raise ValueError('Matched neural budget differs')
    result = {'seed': seed, 'model_config': asdict(config), 'num_parameters': model.num_parameters(),
              'initial_state_sha256': initial, 'final_state_sha256': state_hash(model),
              'examples_processed': examples, 'optimizer_steps': updates,
              'class_weights': criterion.weight.tolist(), 'observations': observations,
              'elapsed_seconds': time.perf_counter()-start}
    if historical and 'prefix_epoch' in historical:
        result.update(prefix_bridge_exact=True, prefix_epoch=historical['prefix_epoch'],
                      prefix_state_sha256=historical['prefix_state_sha256'])
    return result, model, schedule


def check_pair(control, treatment):
    for key in ('seed', 'model_config', 'initial_state_sha256', 'num_parameters',
                'examples_processed', 'optimizer_steps', 'class_weights'):
        if comparable(control['training_fit'])[key] != comparable(treatment['training_fit'])[key]:
            raise ValueError(f'Paired neural recipe or budget differs: {key}')
    for key in ('label_sequence_sha256', 'rare_position_sequence_sha256', 'optimizer_steps'):
        if control['exposure'][key] != treatment['exposure'][key]:
            raise ValueError(f'Paired exposure differs: {key}')
    for name in CLASSES:
        a, b = (r['exposure']['per_class'][name] for r in (control, treatment))
        if a['examples_processed'] != b['examples_processed']:
            raise ValueError('Class exposure changed')
        if name in RARE_CLASSES and a != b:
            raise ValueError('Rare row coverage changed')
        if name not in RARE_CLASSES and b['unique_rows_seen'] != min(b['available_rows'], b['examples_processed']):
            raise ValueError('Class cycle did not maximize coverage before reuse')


def check_protocol(plan, reference, training):
    if (plan['protocol'] != 'official39-matched-majority-diversity-v1'
            or plan['conditions'] != list(CONDITIONS) or plan['frozen_rare_classes'] != list(RARE_CLASSES)
            or plan['sampling'] != 'independent_seeded_class_cycles_without_replacement_until_exhausted'
            or plan['sampling_seed_namespace'] != 426639
            or plan['require_all_control_bridges_before_treatment'] is not True
            or plan['require_all_training_before_validation'] is not True
            or plan['test_opened'] is not False or plan['model_promoted'] is not False
            or plan['prediction_rule'] != 'argmax'
            or not 1 <= plan['evaluation_batch_size'] <= 8192):
        raise ValueError('Plan exceeds the one-factor diversity control')
    if (reference['status'] != 'complete' or reference['all_training_bridges_exact'] is not True
            or reference['test_opened'] is not False or reference['model_promoted'] is not False
            or reference['device'] != 'cpu' or reference['plan_sha256'] != plan['reference_plan_sha256']
            or reference['plan']['reference_receipt_sha256'] != plan['training_reference_sha256']
            or training['status'] != 'complete' or training['test_opened'] is not False
            or training['panel_sha256'] != reference['panel_sha256']
            or training['packed_manifest_sha256'] != reference['packed_manifest_sha256']):
        raise ValueError('Reference provenance mismatch')
    settings = training['mlp_settings']
    controls = [f for f in reference['neural_fits'] if f['condition'] == 'small_dropout']
    if (not 1 <= settings['epochs'] <= 300 or not 1 <= training['base_plan']['threads'] <= 2
            or not 2 <= settings['batch_size'] <= 512
            or settings['normalization'] != 'layer' or settings['loss'] != 'sqrt_weighted_ce'
            or settings['loss_reduction'] != 'batch_weight_sum'
            or settings['class_weight_counts'] != 'balanced_panel_only'
            or [e['training_fit']['seed'] for e in controls] != settings['seeds']
            or not 1 <= len(settings['seeds']) <= 3 or len(set(settings['seeds'])) != len(settings['seeds'])
            or settings['epochs'] != reference['plan']['epochs']
            or settings['seeds'] != reference['plan']['seeds']
            or settings['report_epochs'] != sorted(set(settings['report_epochs']))
            or settings['report_epochs'][-1] != settings['epochs']
            or plan['evaluation_batch_size'] != reference['plan']['evaluation_batch_size']):
        raise ValueError('Frozen settings or seed coverage mismatch')
    for entry in controls:
        config = entry['training_fit']['model_config']
        if config['hidden_dims'] != [64, 32] or config['dropout'] != .2 or config['normalization'] != 'layer':
            raise ValueError('Only the original small dropout model is allowed')
    return controls, settings


def diagnose(data_root, reference_path, training_reference_path, plan_path, output):
    data_root, output = Path(data_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(Path(plan_path).read_text())
    if (sha256(reference_path) != plan['reference_receipt_sha256']
            or sha256(training_reference_path) != plan['training_reference_sha256']):
        raise ValueError('Reference checksum mismatch')
    reference = json.loads(Path(reference_path).read_text())
    training = json.loads(Path(training_reference_path).read_text())
    controls, settings = check_protocol(plan, reference, training)
    if sha256(data_root/'manifest.json') != reference['packed_manifest_sha256']:
        raise ValueError('Packed manifest mismatch')
    source_root = Path(__file__).parents[1]
    sources = dict(reference['source_sha256'])
    for name, old_hash in sources.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(source_root/name) != old_hash:
            raise ValueError('Frozen shared source changed')
    sources['eval/official_diversity.py'] = sha256(__file__)
    panel = training['base_plan']['panel']
    px, py, rows, manifest = load_training_panel(data_root, panel['per_class'], panel['seed'])
    panel_hash = hashlib.sha256(rows.astype('<i8').tobytes()+px.astype('<f4').tobytes()+py.astype('<i8').tobytes()).hexdigest()
    if panel_hash != reference['panel_sha256'] or manifest['train_rows'] != training['base_plan']['expected_training_rows']:
        raise ValueError('Training panel mismatch')
    # Only scalar labels and row IDs are held across batches. Never copy the
    # full two-million-row feature matrix into memory.
    x = np.load(data_root/'train_x.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(data_root/'train_y.npy', mmap_mode='r', allow_pickle=False)
    receipt = {'status': 'incomplete', 'purpose': plan['purpose'], 'plan': plan,
               'plan_sha256': sha256(plan_path), 'panel_sha256': panel_hash,
               'packed_manifest_sha256': reference['packed_manifest_sha256'], 'source_sha256': sources,
               'classes': CLASSES, 'mlp_settings': settings, 'panel_rows': rows.tolist(),
               'device': 'cpu', 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scikit-learn', 'threadpoolctl')},
               'all_control_bridges_exact': False, 'all_pairs_matched': False,
               'validation_opened': False, 'test_opened': False, 'model_promoted': False,
               'fits': [], 'validation_results': []}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(receipt, output/'receipt.json')

    def check_inputs():
        if (sha256(plan_path) != receipt['plan_sha256']
                or sha256(reference_path) != plan['reference_receipt_sha256']
                or sha256(training_reference_path) != plan['training_reference_sha256']
                or sha256(data_root/'manifest.json') != receipt['packed_manifest_sha256']
                or sha256(data_root/'scaler.json') != manifest['scaler_sha256']
                or any(sha256(source_root/p) != h for p, h in sources.items())):
            raise ValueError('Frozen inputs changed during comparison')

    torch.set_num_threads(training['base_plan']['threads'])
    models = []
    with threadpool_limits(limits=training['base_plan']['threads']):
        for condition in CONDITIONS:
            diverse = condition == 'majority_diversity'
            if diverse and receipt['all_control_bridges_exact'] is not True:
                raise ValueError('All controls must bridge before any treatment')
            for old in controls:
                seed = old['training_fit']['seed']
                fit, model, schedule = fit_condition(x, y, rows, settings, seed, diverse,
                    plan['sampling_seed_namespace'], historical=None if diverse else old['training_fit'])
                if not diverse and comparable(fit) != comparable(old['training_fit']):
                    raise ValueError('Control training bridge failed; no treatment or validation allowed')
                name = f'{condition}_seed{seed}'
                counts_path, checkpoint = output/(name+'-exposure.npy'), output/(name+'.pt')
                np.save(counts_path, schedule.exposure, allow_pickle=False)
                atomic_save({'state_dict': model.state_dict(), 'model_config': fit['model_config'],
                             'features': manifest['features'], 'classes': CLASSES, 'seed': seed,
                             'epochs': settings['epochs'], 'condition': condition, 'panel_sha256': panel_hash}, checkpoint)
                entry = {'name': name, 'condition': condition, 'seed': seed, 'training_fit': fit,
                         'exposure': schedule.summary(), 'exposure_file': counts_path.name,
                         'exposure_sha256': sha256(counts_path), 'checkpoint': checkpoint.name,
                         'checkpoint_sha256': sha256(checkpoint)}
                if diverse:
                    check_pair(next(e for e in receipt['fits'] if e['condition'] == 'panel_control' and e['seed'] == seed), entry)
                receipt['fits'].append(entry)
                models.append((entry, model))
                del schedule
                atomic_json(receipt, output/'receipt.json')
            if not diverse:
                receipt['all_control_bridges_exact'] = True
                check_inputs()
                atomic_json(receipt, output/'receipt.json')
        receipt['all_pairs_matched'] = True
        check_inputs()
        receipt['validation_opened'] = True
        atomic_json(receipt, output/'receipt.json')
        vx, vy = load_validation(data_root, manifest, reference['plan'])
        old_validation = {e['seed']: e for e in reference['validation_results'] if e['condition'] == 'small_dropout'}
        # Controls appear first: every old validation matrix is checked before
        # the first treatment is scored. No scores choose a training endpoint.
        for entry, model in models:
            ce, metrics = evaluate(model, validation_batches(vx, vy, plan['evaluation_batch_size']), 'cpu')
            if entry['condition'] == 'panel_control':
                old = old_validation[entry['seed']]
                if ce != old['validation_cross_entropy'] or metrics != old['validation_metrics']:
                    raise ValueError('Control validation bridge failed; treatment scoring blocked')
            if state_hash(model) != entry['training_fit']['final_state_sha256']:
                raise ValueError('Evaluation changed model parameters')
            receipt['validation_results'].append({'name': entry['name'], 'condition': entry['condition'],
                'seed': entry['seed'], 'validation_cross_entropy': ce, 'validation_metrics': metrics})
            print(f'{entry["name"]}: VALIDATION macro-F1={metrics["macro_f1"]:.6f} FAR={metrics["benign_false_alert_rate"]:.4%}', flush=True)
            atomic_json(receipt, output/'receipt.json')
    check_inputs()
    for name in ('train_x.npy', 'train_y.npy', 'val_x.npy', 'val_y.npy'):
        if sha256(data_root/name) != manifest['files'][name]:
            raise ValueError('Allowed data array changed during comparison')
    for entry in receipt['fits']:
        if (sha256(output/entry['exposure_file']) != entry['exposure_sha256']
                or sha256(output/entry['checkpoint']) != entry['checkpoint_sha256']):
            raise ValueError('Generated evidence changed during comparison')
    receipt['summary'] = summarize(receipt['validation_results'])
    receipt['status'] = 'complete'
    atomic_json(receipt, output/'receipt.json')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'reference-path', 'training-reference-path', 'plan-path', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    diagnose(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
