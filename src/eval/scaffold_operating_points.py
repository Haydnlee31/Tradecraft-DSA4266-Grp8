"""Frozen-model alarm calibration on TRAIN only; never another training run.

Calibration and audit groups were BOTH seen during model fitting. Splitting them
here prevents threshold fitting on audit labels, not training-data optimism.
Only the benign/attack boundary changes; attack-category ranking stays fixed.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import federated_class_safeguard as historical
from src.eval.federated_class_safeguard import TrainOnlyData, require, state_digest
from src.models import architectures, official_streaming
from src.models.architectures import MLPClassifier, MLPConfig
from src.models.official_streaming import confusion_metrics
from src.models.research import atomic_json

ARMS = ('central_sgd', 'fedavg_sgd', 'scaffold')
BENIGN = CLASSES.index('Benign')
ATTACKS = np.array([i for i in range(len(CLASSES)) if i != BENIGN])
DOMAIN = b'tradecraft-scaffold-operating-points-v1:426650:'


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def calibration_groups(x, batch_size=8192):
    """Hash canonical model-input bytes, not labels or scores, into two halves.

    Float32 rounding can merge originally distinct records. Hash at the actual
    model precision, normalize signed zero, and keep every exact vector together
    even when its labels differ. Hash membership is order-independent; it is not
    exact class stratification or a claim of session/near-duplicate independence.
    Process chunks rather than copying the full memory-mapped feature matrix.
    """
    require(x.ndim == 2 and x.dtype == np.float32 and len(x) > 0 and batch_size > 0,
            'Nonempty float32 feature matrix required')
    mask = np.empty(len(x), dtype=bool)
    for start in range(0, len(x), batch_size):
        values = np.array(x[start:start+batch_size], dtype='<f4', copy=True)
        require(np.isfinite(values).all(), 'Nonfinite split feature')
        values[values == 0] = 0.  # -0.0 and +0.0 are the same model input.
        mask[start:start+len(values)] = [
            hashlib.sha256(DOMAIN + row.tobytes()).digest()[0] < 128 for row in values]
    return mask


def conservative_threshold(benign_scores, target_basis_points):
    """Permit at most floor(n * target/10000) calibration false alerts.

    Predict attack only for score STRICTLY GREATER than the threshold. Setting
    it to the boundary order statistic excludes boundary ties, so calibration
    FAR cannot exceed the target. This is an empirical rule, not a population
    FAR guarantee. Integer basis points avoid floating-point floor ambiguity.
    """
    scores = np.asarray(benign_scores, dtype=np.float64)
    require(scores.ndim == 1 and len(scores) > 0 and np.isfinite(scores).all(),
            'Finite benign calibration scores required')
    require(type(target_basis_points) is int and 0 <= target_basis_points < 10000,
            'Target must be integer basis points in [0,10000)')
    allowed = len(scores)*target_basis_points//10000
    index = len(scores)-allowed-1
    threshold = float(np.partition(scores, index)[index])
    alerts = int(np.count_nonzero(scores > threshold))
    require(alerts <= allowed, 'Calibration false-alert budget exceeded')
    return {'threshold': threshold, 'benign_rows': len(scores),
            'allowed_false_alerts': allowed, 'false_alerts': alerts,
            'boundary_ties': int(np.count_nonzero(scores == threshold)),
            'achieved_false_alert_rate': alerts/len(scores)}


def classify(scores, best_attack, threshold):
    scores, best_attack = np.asarray(scores), np.asarray(best_attack)
    require(scores.ndim == 1 and scores.shape == best_attack.shape
            and np.isfinite(scores).all() and np.isfinite(threshold)
            and np.isin(best_attack, ATTACKS).all(), 'Invalid decision inputs')
    return np.where(scores > threshold, best_attack, BENIGN)


def metrics(labels, predictions):
    require(labels.shape == predictions.shape and labels.ndim == 1 and len(labels) > 0
            and np.isin(labels, np.arange(8)).all()
            and np.isin(predictions, np.arange(8)).all(), 'Invalid metric inputs')
    cm = np.bincount(labels*8+predictions, minlength=64).reshape(8, 8)
    return confusion_metrics(cm)


def frozen_scores(model, data, batch_size=2048):
    """One forward per training row; no optimizer, backward or parameter writes."""
    model.eval()
    before = state_digest(model)
    n = len(data.arrays['train'][1])
    scores, attacks, predictions = np.empty(n, np.float64), np.empty(n, np.int64), np.empty(n, np.int64)
    start = 0
    with torch.inference_mode():
        for x, _ in data.batches('train', batch_size):
            logits = model(torch.from_numpy(x)).numpy()
            require(logits.shape == (len(x), 8) and np.isfinite(logits).all(), 'Invalid logits')
            # Canonical class order breaks ties between attacks, as argmax does.
            attack = ATTACKS[logits[:, ATTACKS].argmax(1)]
            end = start+len(x)
            attacks[start:end] = attack
            predictions[start:end] = logits.argmax(1)
            # Subtract in float64 so tiny FP32 differences retain their sign.
            scores[start:end] = logits[np.arange(len(x)), attack].astype(np.float64)-logits[:, BENIGN]
            start = end
    require(start == n and state_digest(model) == before, 'Rows missing or frozen model changed')
    # Benign is first in the canonical class order, hence ties go to benign.
    require(BENIGN == 0 and np.array_equal(classify(scores, attacks, 0.), predictions),
            'Zero-threshold rule must exactly reproduce ordinary argmax')
    return scores, attacks, predictions


def describe_arm(labels, calibration, scores, attacks, predictions, targets):
    audit = ~calibration
    require(calibration.dtype == bool and calibration.shape == labels.shape
            and all(np.all(np.bincount(labels[m], minlength=8) > 0) for m in (calibration, audit)),
            'Both fixed partitions must contain all classes; do not retry another split')
    # Only these benign calibration scores may affect the threshold. Audit
    # labels and attack labels are used to report metrics, not optimize a rule.
    benign_scores = scores[calibration & (labels == BENIGN)]
    result = {'scores_sha256': array_digest(scores.astype('<f8')),
              'best_attack_sha256': array_digest(attacks.astype('<i8')),
              'argmax_all_training': metrics(labels, predictions),
              'argmax_audit': metrics(labels[audit], predictions[audit]),
              'attack_rank_recall_ceiling': {}, 'targets': {}}
    for c in ATTACKS:
        take = audit & (labels == c)
        correct = int(np.count_nonzero(attacks[take] == c))
        result['attack_rank_recall_ceiling'][CLASSES[c]] = {
            'correct_attack_rank': correct, 'support': int(take.sum()), 'recall': correct/int(take.sum())}
    for target in targets:
        fitted = conservative_threshold(benign_scores, target)
        pred = classify(scores[audit], attacks[audit], fitted['threshold'])
        m = metrics(labels[audit], pred)
        cm = np.asarray(m['confusion_matrix'])
        result['targets'][str(target)] = {
            'target_false_alert_rate': target/10000, 'calibration': fitted, 'audit': m,
            # Binary alarm recall must never be mistaken for correct 8-class recall.
            'audit_attack_alarm_recall': {CLASSES[c]: float(1-cm[c, BENIGN]/cm[c].sum()) for c in ATTACKS}}
    return result


def primary_diagnostic(arms, plan):
    """A new descriptive screen cannot reverse the earlier failed training gate."""
    target, limits = str(plan['primary_target_basis_points']), plan['diagnostic_screen']
    avg, method = [arms[a]['targets'][target]['audit'] for a in ('fedavg_sgd', 'scaffold')]
    rare = {c: method['per_class'][c]['recall']-avg['per_class'][c]['recall']
            for c in historical.PROTECTED[1:]}
    macro = method['macro_f1']-avg['macro_f1']
    far = method['benign_false_alert_rate']-avg['benign_false_alert_rate']
    comparable = abs(far) <= limits['max_audit_far_difference'] and all(
        abs(m['benign_false_alert_rate']-int(target)/10000) <= limits['max_audit_target_deviation']
        for m in (avg, method))
    eligible = comparable and macro >= limits['min_macro_gain'] and min(rare.values()) >= 0
    return {'primary_target_basis_points': int(target), 'macro_f1_gain': macro,
            'rare_recall_gains': rare, 'audit_false_alert_difference': far,
            'audit_operating_points_comparable_under_plan': comparable,
            'eligible_for_separately_planned_confirmation': eligible,
            'prior_training_screen_overturned': False, 'generalization_established': False,
            'new_training_launched': False}


def run(data_root, endpoint, evidence_path, plan_path, output):
    started = time.perf_counter()
    output, endpoint, evidence_path, plan_path = map(Path, (output, endpoint, evidence_path, plan_path))
    require(not output.exists(), 'Output exists; preserve it')
    plan = json.loads(plan_path.read_text())
    require(plan['arms'] == list(ARMS) and plan['targets_basis_points'] == [100, 500, 1000]
            and plan['primary_target_basis_points'] == 500 and plan['split_domain'] == DOMAIN.decode(),
            'Changed diagnostic scope')
    require(sha256(endpoint) == plan['endpoint_sha256'] and sha256(evidence_path) == plan['evidence_sha256']
            and sha256(Path(data_root)/'manifest.json') == plan['packed_manifest_sha256'], 'Changed frozen input')
    evidence = json.loads(evidence_path.read_text())['primary']
    require(evidence['status'] == 'complete' and evidence['rounds'][-1]['step'] == 30
            and not evidence['screen']['safety_screen_passed'], 'Wrong source trial')
    data = TrainOnlyData(data_root)
    saved = torch.load(endpoint, weights_only=True, map_location='cpu')
    require(saved['step'] == 30 and set(saved['models']) == set(ARMS)
            and saved['plan_sha256'] == evidence['plan_sha256'], 'Wrong endpoint')
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    models = {}
    for arm in ARMS:
        model = MLPClassifier(39, 8, MLPConfig(**plan['model_config']))
        model.load_state_dict(saved['models'][arm], strict=True)
        require(state_digest(model) == plan['model_sha256'][arm]
                == evidence['rounds'][-1]['arms'][arm]['model_sha256'], 'Wrong frozen model')
        models[arm] = model
    x, labels = data.arrays['train']
    calibration = calibration_groups(x)
    counts = {name: np.bincount(labels[mask], minlength=8).tolist()
              for name, mask in (('calibration', calibration), ('audit', ~calibration))}
    require(all(min(c) > 0 for c in counts.values()), 'Empty class; stop, never reseed')
    result = {'status': 'running', 'plan_sha256': sha256(plan_path), 'script_sha256': sha256(__file__),
              'helper_sha256': sha256(historical.__file__), 'architecture_sha256': sha256(architectures.__file__),
              'metrics_source_sha256': sha256(official_streaming.__file__),
              'endpoint_sha256': sha256(endpoint), 'evidence_sha256': sha256(evidence_path),
              'packed_manifest_sha256': plan['packed_manifest_sha256'], 'class_order': CLASSES,
              'partition': {'domain': DOMAIN.decode(), 'calibration_mask_sha256': array_digest(calibration),
                            'counts': counts, 'all_rows_previously_available_for_model_fitting': True},
              'device': 'cpu', 'threads': 2, 'platform': platform.platform(),
              'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy')},
              'validation_rows_opened': False, 'test_opened': False, 'test_evaluated': False,
              'cloud_used': False, 'optimizer_steps': 0, 'backward_batches': 0,
              'forward_examples': 0, 'arms': {}}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(result, output/'result.json')
    for arm, model in models.items():
        values = frozen_scores(model, data)
        described = describe_arm(labels, calibration, *values, plan['targets_basis_points'])
        require(described['argmax_all_training'] == evidence['rounds'][-1]['arms'][arm]['evaluation']['metrics'],
                'Historical full-training endpoint bridge differs')
        described.update(model_sha256=state_digest(model), argmax_historical_bridge_exact=True,
                         zero_threshold_argmax_bridge_exact=True)
        result['arms'][arm] = described
        result['forward_examples'] += len(labels)
        print(f'{arm}: frozen argmax bridge exact; all three operating points measured', flush=True)
        atomic_json(result, output/'result.json')
    require(result['forward_examples'] == 6_000_000 and sha256(endpoint) == result['endpoint_sha256']
            and sha256(evidence_path) == result['evidence_sha256'], 'Changed source or work budget')
    result.update(status='complete', frozen_inputs_unchanged=True,
                  diagnostic=primary_diagnostic(result['arms'], plan), elapsed_seconds=time.perf_counter()-started)
    atomic_json(result, output/'result.json')
    print(json.dumps(result['diagnostic'], indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'endpoint', 'evidence-path', 'plan-path', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
