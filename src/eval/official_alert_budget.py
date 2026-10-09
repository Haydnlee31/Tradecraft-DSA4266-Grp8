"""A single frozen-model alert gate at predeclared benign false-alert budgets.

This is an EXPLORATORY same-validation diagnostic. Using benign validation
scores to set a cutoff and then measuring on that same split does NOT establish
an independent false-alert guarantee. The existing production decision engine
and its research gates are untouched. No optimizer or training loader is used.
"""
import argparse
from fractions import Fraction
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
from src.eval.official_panel_validation import load_validation, validation_batches, summarize
from src.eval.official_training_fit import state_hash
from src.eval.research_decision import error_breakdown
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_streaming import confusion_metrics
from src.models.research import atomic_json

BENIGN = CLASSES.index('Benign')
CONDITIONS = ('small_dropout', 'small_no_dropout', 'wide_dropout', 'wide_no_dropout')


def check_protocol(plan, reference):
    if (plan['protocol'] != 'official39-frozen-alert-budget-diagnostic-v1'
            or plan['conditions'] != list(CONDITIONS) or plan['primary_condition'] != 'small_dropout'
            or plan['benign_false_alert_budgets'] != [.001, .01, .05]
            or plan['score'] != 'max_attack_logit_minus_benign_logit'
            or plan['threshold_fit_split'] != 'validation'
            or plan['assessment_split'] != 'same_validation_exploratory_only'
            or plan['require_all_argmax_bridges'] is not True
            or plan['test_opened'] is not False or plan['model_trained'] is not False
            or plan['model_promoted'] is not False
            or plan['rare_score_classes'] != ['Web-based', 'Brute Force']
            or plan['score_quantiles'] != [.1, .5, .9, .99]
            or not 1 <= plan['threads'] <= 2 or not 1 <= plan['evaluation_batch_size'] <= 8192):
        raise ValueError('Plan exceeds the frozen alert-budget diagnostic')
    if (reference['status'] != 'complete' or reference['all_training_bridges_exact'] is not True
            or reference['validation_opened'] is not True or reference['test_opened'] is not False
            or reference['model_promoted'] is not False or reference['device'] != 'cpu'
            or reference['plan_sha256'] != plan['reference_plan_sha256']
            or reference['plan']['seeds'] != plan['seeds']
            or not 1 <= len(plan['seeds']) <= 3 or len(set(plan['seeds'])) != len(plan['seeds'])
            or reference['plan']['evaluation_batch_size'] != plan['evaluation_batch_size']):
        raise ValueError('Reference is not the approved frozen validation comparison')
    expected = [(c, s) for c in CONDITIONS for s in plan['seeds']]
    actual = [(r['condition'], r['training_fit']['seed']) for r in reference['neural_fits']]
    if actual != expected or len({r['name'] for r in reference['neural_fits']}) != len(expected):
        raise ValueError('Missing or duplicated checkpoint coverage')
    validation_names = [r['name'] for r in reference['validation_results'] if r['condition'] in CONDITIONS]
    if validation_names != [r['name'] for r in reference['neural_fits']]:
        raise ValueError('Missing or duplicated argmax reference coverage')


def load_checkpoint(root, entry, reference, manifest):
    """Verify content and metadata before restricted weights-only loading."""
    root = Path(root).resolve()
    name = entry['checkpoint']
    if Path(name).name != name or not (root/name).resolve().is_relative_to(root):
        raise ValueError('Invalid checkpoint path')
    path = root/name
    if sha256(path) != entry['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    fit = entry['training_fit']
    if (json.loads(json.dumps(saved['model_config'])) != fit['model_config']
            or saved['panel_sha256'] != reference['panel_sha256']
            or saved['features'] != manifest['features'] or saved['classes'] != CLASSES
            or saved['seed'] != fit['seed'] or saved['epochs'] != reference['plan']['epochs']):
        raise ValueError('Checkpoint metadata mismatch')
    model = MLPClassifier(len(manifest['features']), len(CLASSES), MLPConfig(**saved['model_config']))
    model.load_state_dict(saved['state_dict'], strict=True)
    if state_hash(model) != fit['final_state_sha256'] or model.num_parameters() != fit['num_parameters']:
        raise ValueError('Final parameter hash or size mismatch')
    model.eval()
    return model


def collect_scores(model, x, y, batch_size, rare_classes, score_path, winner_path):
    """Save only scalar margins, rare-class scores and attack winners, not logits.

    At threshold zero the strict margin rule must reproduce argmax exactly.
    Benign is first in the canonical label order, so an attack/benign tie is
    correctly benign. Attack/attack ties retain canonical argmax ordering.
    """
    if BENIGN != 0:
        raise ValueError('The argmax tie bridge requires the canonical benign-first schema')
    attack_ids = [i for i in range(len(CLASSES)) if i != BENIGN]
    rare_ids = [CLASSES.index(c) for c in rare_classes]
    scores = np.lib.format.open_memmap(score_path, mode='w+', dtype=np.float32, shape=(len(y), 1+len(rare_ids)))
    winners = np.lib.format.open_memmap(winner_path, mode='w+', dtype=np.uint8, shape=(len(y),))
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    offset, loss_sum = 0, 0.
    with torch.no_grad():
        for xb, yb in validation_batches(x, y, batch_size):
            logits = model(torch.from_numpy(xb))
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite model scores')
            loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(yb), reduction='sum')
            loss_sum += loss.item()
            pred = logits.argmax(1).numpy()
            best_score, local_winner = logits[:, attack_ids].max(1)
            best = np.asarray(attack_ids, dtype=np.uint8)[local_winner.numpy()]
            margin = (best_score-logits[:, BENIGN]).numpy()
            if not np.array_equal(pred, np.where(margin > 0., best, BENIGN)):
                raise ValueError('Zero-margin gate does not reproduce raw argmax')
            end = offset+len(yb)
            scores[offset:end, 0] = margin
            scores[offset:end, 1:] = logits.softmax(1)[:, rare_ids].numpy()
            winners[offset:end] = best
            matrix += np.bincount(yb*len(CLASSES)+pred, minlength=len(CLASSES)**2).reshape(matrix.shape)
            offset = end
    if offset != len(y):
        raise ValueError('Incomplete scoring coverage')
    scores.flush()
    winners.flush()
    del scores, winners
    return loss_sum/offset, confusion_metrics(matrix)


def threshold_for_budget(benign_margins, budget):
    """Conservative exact order statistic, with a strict > comparison for ties.

    For K allowed alerts, the (K+1)th highest benign score is the cutoff. Ties
    may leave unused budget; do not break ties using labels or row order.
    Nonnegative thresholds can only suppress existing argmax attack alerts.
    """
    values = np.asarray(benign_margins)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 <= budget < 1:
        raise ValueError('Finite benign scores and a budget in [0, 1) required')
    allowed = int(Fraction(str(budget))*len(values))
    index = len(values)-allowed-1
    threshold = max(0., float(np.partition(values, index)[index]))
    observed = int(np.count_nonzero(values > threshold))
    if observed > allowed:
        raise ValueError('Empirical benign budget exceeded')
    return {'threshold': threshold, 'allowed_benign_alerts': allowed,
            'observed_benign_alerts': observed, 'benign_support': len(values)}


def gated_metrics(margins, winners, labels, threshold, batch_size):
    if not np.isfinite(threshold) or threshold < 0 or not 1 <= batch_size <= 8192:
        raise ValueError('A finite nonnegative gate and bounded batches are required')
    if not len(labels) or len(margins) != len(labels) or len(winners) != len(labels):
        raise ValueError('Score and label lengths differ')
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for start in range(0, len(labels), batch_size):
        stop = start+batch_size
        y = labels[start:stop]
        winner = winners[start:stop]
        margin = margins[start:stop]
        if (not np.isfinite(margin).all() or np.any(winner <= BENIGN)
                or np.any(winner >= len(CLASSES)) or np.any(y < 0) or np.any(y >= len(CLASSES))):
            raise ValueError('Invalid saved scores, winners or labels')
        pred = np.where(margin > threshold, winner, BENIGN).astype(np.int64)
        matrix += np.bincount(y*len(CLASSES)+pred, minlength=len(CLASSES)**2).reshape(matrix.shape)
    return confusion_metrics(matrix)


def score_distributions(scores, labels, rare_classes, quantiles):
    """Softmax values are descriptive scores, not calibrated risk estimates."""
    result = {}
    for column, name in enumerate(rare_classes, 1):
        target = CLASSES.index(name)
        groups = {'true_class': labels == target, 'benign': labels == BENIGN,
                  'other_attacks': (labels != target) & (labels != BENIGN)}
        result[name] = {group: {'rows': int(mask.sum()),
                               'quantiles': dict(zip(map(str, quantiles), map(float, np.quantile(scores[mask, column], quantiles))))}
                        for group, mask in groups.items()}
    return result


def diagnose(data_root, reference_root, plan_path, output):
    output, data_root, reference_root = Path(output), Path(data_root), Path(reference_root)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(Path(plan_path).read_text())
    reference_path = reference_root/'receipt.json'
    if sha256(reference_path) != plan['reference_receipt_sha256']:
        raise ValueError('Reference checksum mismatch')
    reference = json.loads(reference_path.read_text())
    check_protocol(plan, reference)
    if sha256(data_root/'manifest.json') != reference['packed_manifest_sha256']:
        raise ValueError('Packed manifest mismatch')
    manifest = json.loads((data_root/'manifest.json').read_text())
    if (manifest['classes'] != CLASSES or manifest['test_opened'] is not False
            or sha256(data_root/'scaler.json') != manifest['scaler_sha256']):
        raise ValueError('Class, scaler or split provenance mismatch')
    source_root, sources = Path(__file__).parents[1], {}
    for name, old_hash in reference['source_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid source path')
        sources[name] = sha256(source_root/path)
        if sources[name] != old_hash:
            raise ValueError('Frozen shared source changed')
    for name in ('eval/official_alert_budget.py', 'eval/research_decision.py', 'eval/decision.py'):
        sources[name] = sha256(source_root/name)
    # Check EVERY checkpoint before data scoring. This is tiny model memory,
    # not feature-matrix loading or retraining.
    models = [load_checkpoint(reference_root, e, reference, manifest) for e in reference['neural_fits']]
    receipt = {'status': 'incomplete', 'purpose': plan['purpose'], 'plan': plan,
               'plan_sha256': sha256(plan_path), 'reference_receipt_sha256': plan['reference_receipt_sha256'],
               'packed_manifest_sha256': reference['packed_manifest_sha256'], 'source_sha256': sources,
               'device': 'cpu', 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scikit-learn', 'threadpoolctl')},
               'test_opened': False, 'training_arrays_opened': False, 'model_trained': False,
               'model_promoted': False, 'independent_assessment': False,
               'validation_opened': True, 'all_argmax_bridges_exact': False, 'models': [], 'operating_points': []}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(receipt, output/'receipt.json')
    x, y = load_validation(data_root, manifest, reference['plan'])
    originals = {r['name']: r for r in reference['validation_results']}
    torch.set_num_threads(plan['threads'])
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=plan['threads']):
        for entry, model in zip(reference['neural_fits'], models):
            start = time.perf_counter()
            name = entry['name']
            # Check canonical names separately from the checkpoint allowlist;
            # output filenames must never carry paths supplied by a receipt.
            expected_name = f'{entry["condition"]}_seed{entry["training_fit"]["seed"]}'
            if name != expected_name:
                raise ValueError('Noncanonical model name')
            score_path, winner_path = output/(name+'-scores.npy'), output/(name+'-winners.npy')
            ce, metrics = collect_scores(model, x, y, plan['evaluation_batch_size'],
                                         plan['rare_score_classes'], score_path, winner_path)
            old = originals[name]
            if ce != old['validation_cross_entropy'] or metrics != old['validation_metrics']:
                raise ValueError('Raw argmax bridge failed; no thresholds were selected')
            if state_hash(model) != entry['training_fit']['final_state_sha256']:
                raise ValueError('Scoring changed model parameters')
            receipt['models'].append({'name': name, 'condition': entry['condition'], 'seed': entry['training_fit']['seed'],
                'checkpoint_sha256': entry['checkpoint_sha256'], 'final_state_sha256': entry['training_fit']['final_state_sha256'],
                'argmax_cross_entropy': ce, 'argmax_metrics': metrics,
                'score_file': score_path.name, 'score_sha256': sha256(score_path),
                'winner_file': winner_path.name, 'winner_sha256': sha256(winner_path),
                'elapsed_seconds': time.perf_counter()-start})
            atomic_json(receipt, output/'receipt.json')
            print(f'{name}: PASS frozen argmax bridge', flush=True)
    receipt['all_argmax_bridges_exact'] = True
    atomic_json(receipt, output/'receipt.json')
    for entry in receipt['models']:
        scores = np.load(output/entry['score_file'], mmap_mode='r', allow_pickle=False)
        winners = np.load(output/entry['winner_file'], mmap_mode='r', allow_pickle=False)
        if gated_metrics(scores[:, 0], winners, y, 0., plan['evaluation_batch_size']) != entry['argmax_metrics']:
            raise ValueError('Saved-score zero-threshold bridge failed')
        entry['rare_score_distributions'] = score_distributions(scores, y, plan['rare_score_classes'], plan['score_quantiles'])
        benign_margins = scores[y == BENIGN, 0]
        for budget in plan['benign_false_alert_budgets']:
            cutoff = threshold_for_budget(benign_margins, budget)
            metrics = gated_metrics(scores[:, 0], winners, y, cutoff['threshold'], plan['evaluation_batch_size'])
            matrix = np.asarray(metrics['confusion_matrix'])
            if int(matrix[BENIGN].sum()-matrix[BENIGN, BENIGN]) != cutoff['observed_benign_alerts']:
                raise ValueError('Gate confusion counts differ from the empirical budget')
            # Suppression cannot make a previously missed attack category correct.
            if any(metrics['per_class'][c]['recall'] > entry['argmax_metrics']['per_class'][c]['recall']
                   for c in CLASSES if c != CLASSES[BENIGN]):
                raise ValueError('Gate changed an attack category instead of suppressing it')
            receipt['operating_points'].append({'name': entry['name'], 'condition': entry['condition'],
                'seed': entry['seed'], 'budget': budget, **cutoff, 'validation_metrics': metrics,
                'error_breakdown': error_breakdown(matrix)})
            print(f'{entry["name"]} budget={budget:.1%}: F1={metrics["macro_f1"]:.6f}', flush=True)
        del scores, winners, benign_margins
        atomic_json(receipt, output/'receipt.json')
    if (sha256(plan_path) != receipt['plan_sha256'] or sha256(reference_path) != plan['reference_receipt_sha256']
            or sha256(data_root/'manifest.json') != receipt['packed_manifest_sha256']
            or any(sha256(source_root/p) != h for p, h in sources.items())
            or any(sha256(reference_root/e['checkpoint']) != e['checkpoint_sha256'] for e in reference['neural_fits'])):
        raise ValueError('Frozen inputs or source changed during diagnosis')
    for name in ('val_x.npy', 'val_y.npy'):
        if sha256(data_root/name) != manifest['files'][name]:
            raise ValueError('Validation data changed during diagnosis')
    for entry in receipt['models']:
        for kind in ('score', 'winner'):
            if sha256(output/entry[kind+'_file']) != entry[kind+'_sha256']:
                raise ValueError('Saved diagnostic scores changed during diagnosis')
    receipt['summary_by_budget'] = {str(b): summarize([r for r in receipt['operating_points'] if r['budget'] == b])
                                    for b in plan['benign_false_alert_budgets']}
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
