"""A bounded finite-step check of federated class-gradient safeguards.

This is three independent counterfactual updates, NOT a new training run.
Keep the historical diagnostic unchanged: its failed result is still evidence.
The server-side correction uses class gradients; candidate checks need only
client-local loss sums/counts. Actual recall improvement is not guaranteed.
"""
import argparse
import copy
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import federated_class_safeguard as original
from src.eval.federated_class_safeguard import (
    PROTECTED, TrainOnlyData, require, vector, moved_model, state_digest,
    class_gradients, fedavg_round, training_evaluation, checkpoint,
)
from src.models.research import atomic_json
from src.utils.seed import set_seed


def margin_projection(delta, gradients, fraction):
    """Closest d with normalized class-gradient dot products <= -margin.

    The margin is fraction * ||FedAvg move||, so rescaling one loss does not
    alter the constraints. Unlike a cone projection, this can expand the norm
    or be infeasible (e.g. exactly opposing gradients). Never call a zero move
    a successful strict-descent solution. An uncertified solve is a rejection,
    not proof of mathematical infeasibility.
    """
    delta, g = np.asarray(delta, dtype=np.float64), np.asarray(gradients, dtype=np.float64)
    require(delta.ndim == 1 and g.ndim == 2 and g.shape[1] == len(delta)
            and 1 <= len(g) <= 3 and np.isfinite(delta).all() and np.isfinite(g).all()
            and np.isfinite(fraction) and 0 < fraction < 1, 'Invalid margin projection inputs')
    size, norms = np.linalg.norm(delta), np.linalg.norm(g, axis=1)
    require(np.isfinite(size) and np.isfinite(norms).all(), 'Projection norm overflow')
    if size == 0 or np.any(norms == 0):
        return None, {'status': 'rejected_stationary_input'}
    u, margin = g/norms[:, None], float(fraction*size)
    tol = 1e-10*size
    candidates = []
    for n in range(len(g)+1):
        for active in itertools.combinations(range(len(g)), n):
            a = u[list(active)]
            # KKT stationarity: d = delta - A.T lambda. Active constraints
            # satisfy A d = -margin, with lambda >= 0 for inequalities <= 0.
            lam = np.linalg.lstsq(a @ a.T, a @ delta+margin, rcond=1e-12)[0] if n else np.zeros(0)
            d = delta-a.T @ lam
            residual = max(0., float(np.max(u @ d+margin)),
                           float(np.max(np.abs(a @ d+margin))) if n else 0.,
                           float(-np.min(lam)) if n else 0.)
            if residual <= tol:
                candidates.append((float(np.dot(d-delta, d-delta)), d, active, residual))
    if not candidates:
        return None, {'status': 'rejected_no_certified_solution', 'margin': margin}
    _, d, active, residual = min(candidates, key=lambda item: item[0])
    return d, {'status': 'certified', 'margin': margin, 'active_constraints': list(active),
               'kkt_residual': residual, 'normalized_dots': (u @ d).tolist()}


def client_risk(model, data, parts, batch_size=2048):
    """Each virtual client returns protected-class CE SUMS and COUNTS only.

    Empty client/classes contribute (0, 0), not a fabricated zero mean. Batch
    forward uses FP32, summation FP64. Use identical batching for the baseline
    and every candidate so a changed batch arrangement cannot drive acceptance.
    A real deployment would execute this loop at clients, not centralize rows.
    """
    work, labels = copy.deepcopy(model).eval(), data.arrays['train'][1]
    sums, counts = np.zeros((len(parts), 3)), np.zeros((len(parts), 3), dtype=np.int64)
    batches = 0
    with torch.inference_mode():
        for client, rows in enumerate(parts):
            for j, name in enumerate(PROTECTED):
                selected = rows[labels[rows] == CLASSES.index(name)]
                counts[client, j] = len(selected)
                for x, y in data.batches('train', batch_size, rows=selected):
                    terms = F.cross_entropy(work(torch.from_numpy(x)), torch.from_numpy(y), reduction='none')
                    require(bool(torch.isfinite(terms).all()), 'Nonfinite client risk')
                    sums[client, j] += float(terms.double().sum())
                    batches += 1
    total = counts.sum(axis=0)
    require(np.all(total > 0), 'Protected classes need training examples')
    return {'mean_class_ce': (sums.sum(axis=0)/total).tolist(),
            'client_loss_sums': sums.tolist(), 'client_counts': counts.tolist(),
            'forward_examples': int(total.sum()), 'forward_batches': batches}


def bounded_search(model, delta, gradients, baseline, evaluate, plan):
    """First passing step, not the best-scoring step; at most four proposals.

    evaluate is the client-risk protocol. It never sees validation, F1 or recall.
    Check the displacement actually stored in FP32, not just the ideal FP64 QP.
    Rejection returns None; callers may NOT silently continue with FedAvg.
    """
    require(plan['scales'] == [1., .5, .25, .125]
            and 0 < plan['armijo_fraction'] < .5
            and plan['loss_tolerance'] > 0
            and 0 < plan['min_retained_norm_fraction'] <= 1
            and 1 <= plan['max_direction_norm_ratio'] <= 2, 'Invalid bounded search recipe')
    d, solver = margin_projection(delta, gradients, plan['margin_fraction'])
    info = {'accepted': False, 'solver': solver, 'trials': []}
    if d is None:
        info['reason'] = solver['status']
        return None, info
    size = float(np.linalg.norm(delta))
    ratio = float(np.linalg.norm(d)/size)
    info['direction_norm_ratio'] = ratio
    if ratio > plan['max_direction_norm_ratio']:
        info['reason'] = 'rejected_direction_expansion'
        return None, info
    before = np.asarray(baseline['mean_class_ce'])
    require(before.shape == (len(PROTECTED),) and np.isfinite(before).all(), 'Invalid baseline risk')
    for scale in plan['scales']:
        candidate = moved_model(model, scale*d)
        applied = vector(candidate)-vector(model)
        predictions = gradients @ applied
        retained = float(np.linalg.norm(applied)/size)
        risk = evaluate(candidate)
        changes = np.asarray(risk['mean_class_ce'])-before
        require(changes.shape == before.shape and np.isfinite(changes).all(), 'Invalid candidate risk')
        threshold = plan['armijo_fraction']*predictions+plan['loss_tolerance']
        # The independent strict-decrease check prevents the tolerance from
        # admitting unchanged or slightly increasing losses as "safe progress".
        accepted = bool(np.all(predictions < 0) and np.all(changes < -plan['loss_tolerance'])
                        and np.all(changes <= threshold)
                        and retained >= plan['min_retained_norm_fraction'])
        trial = {'scale': scale, 'accepted': accepted, 'retained_norm_fraction': retained,
                 'model_sha256': state_digest(candidate), 'risk': risk,
                 'predicted_ce_changes': predictions.tolist(), 'actual_ce_changes': changes.tolist(),
                 'armijo_upper_bounds': threshold.tolist(),
                 'rounding_displacement_norm': float(np.linalg.norm(applied-scale*d))}
        info['trials'].append(trial)
        if accepted:
            info.update(accepted=True, reason='first_passing_scale', accepted_scale=scale)
            return candidate, info
    info['reason'] = 'rejected_all_four_scales'
    return None, info


def continuation_gate(cases, plan):
    safe, useful = [], []
    for c in cases:
        trials = c['search']['trials']
        accepted = c['search']['accepted'] and bool(trials) and trials[-1]['accepted']
        safe.append(bool(accepted))
        harmed = [i for i, name in enumerate(PROTECTED) if name != PROTECTED[0]
                  and c['fedavg_predicted_protected_ce_changes'][i] > plan['loss_tolerance']
                  and c['fedavg_actual_protected_ce_changes'][i] > plan['loss_tolerance']]
        helps = accepted and any(trials[-1]['actual_ce_changes'][i] <
                                c['fedavg_actual_protected_ce_changes'][i]-plan['loss_tolerance'] for i in harmed)
        useful.append(bool(helps))
    passed = all(safe) and sum(useful) >= plan['min_useful_checkpoints']
    return {'decision': 'eligible_for_separately_planned_short_trajectory' if passed else 'stop_no_full_training',
            'safe_checkpoint_flags': safe, 'useful_checkpoint_flags': useful,
            'full_training_launched': False, 'not_a_performance_or_generalization_claim': True}


def inspect(model, data, parts, settings, step, prior, plan):
    source = state_digest(model)
    gradients, gradient_work = class_gradients(model, data, parts)
    ordinary, _, client_work = fedavg_round(model, data, parts, settings, step+1)
    delta = vector(ordinary)-vector(model)
    before, avg = [training_evaluation(m, data) for m in (model, ordinary)]
    # Detect a changed baseline before interpreting any refinement. Same CPU
    # runtime is deliberately required for this local development comparison.
    require(source == prior['source_model_sha256'] and state_digest(ordinary) == prior['fedavg_model_sha256']
            and before == prior['before'] and avg == prior['fedavg']
            and gradient_work == prior['gradient_work'] and client_work == prior['client_work'],
            'Historical diagnostic bridge differs')
    for i, name in enumerate(CLASSES):
        require(float(gradients[i] @ delta) == prior['per_class'][name]['fedavg_predicted_ce_change']
                and float(np.linalg.norm(gradients[i])) == prior['per_class'][name]['gradient_norm'],
                'Historical gradient bridge differs')
    baseline = client_risk(model, data, parts)
    ids = [CLASSES.index(c) for c in PROTECTED]
    accepted, search = bounded_search(model, delta, gradients[ids], baseline,
                                     lambda m: client_risk(m, data, parts), plan)
    corrected = training_evaluation(accepted, data) if accepted is not None else None
    require(state_digest(model) == source, 'Source checkpoint was mutated')
    checks = len(search['trials'])
    # Prospective payload estimates, NOT measured network traffic. The baseline
    # weights are already at clients; extra candidate broadcasts are explicit.
    # Gradients may be sent FP32 (3 * parameter_bytes); Python simulation sums
    # in FP64. Scalars use 3 float64 sums + 3 int64 counts = 48 bytes per reply.
    communication = {'prospective_extra_gradient_bytes_per_client': 3*model.parameter_bytes(),
                     'diagnostic_all8_gradient_bytes_per_client_fp32_equivalent': 8*model.parameter_bytes(),
                     'candidate_checks': checks,
                     'extra_model_broadcast_bytes_all_clients': checks*len(parts)*model.parameter_bytes(),
                     'risk_reply_bytes_all_clients': (1+checks)*len(parts)*48,
                     'measured_network_traffic': False, 'protocol_overhead_included': False}
    return {'source_step': step, 'source_model_sha256': source,
            'fedavg_model_sha256': state_digest(ordinary), 'baseline_bridge_exact': True,
            'source_model_unchanged': True, 'gradient_work': gradient_work, 'client_work': client_work,
            'ordinary_delta_norm': float(np.linalg.norm(delta)),
            'before': before, 'fedavg': avg, 'accepted_evaluation': corrected,
            'fedavg_predicted_protected_ce_changes': (gradients[ids] @ delta).tolist(),
            'fedavg_actual_protected_ce_changes': (np.array(avg['mean_class_ce'])[ids]-np.array(before['mean_class_ce'])[ids]).tolist(),
            'baseline_client_risk': baseline, 'search': search, 'communication': communication,
            'training_forward_evaluation_examples': data.manifest['train_rows']*(2+(accepted is not None)),
            'protected_risk_forward_examples': baseline['forward_examples']+
                sum(t['risk']['forward_examples'] for t in search['trials'])}


def run(data_root, archive, plan_path, previous_evidence, output):
    output, archive, plan_path = Path(output), Path(archive), Path(plan_path)
    require(not output.exists(), 'Output exists; preserve it')
    plan = json.loads(plan_path.read_text())
    require(plan['protected_classes'] == list(PROTECTED), 'Changed protected classes')
    require(sha256(previous_evidence) == plan['previous_evidence_sha256']
            and sha256(original.__file__) == plan['previous_runner_sha256'], 'Previous evidence changed')
    prior = json.loads(Path(previous_evidence).read_text())['primary']
    require(prior['status'] == 'complete' and prior['gate']['decision'] == 'stop_no_full_training',
            'Expected failed first-order diagnostic')
    require(sha256(archive) == plan['archive_sha256']
            and sha256(Path(data_root)/'manifest.json') == plan['packed_manifest_sha256'], 'Changed archive/data')
    require([p['step'] for p in plan['checkpoints']] == [c['source_step'] for c in prior['cases']], 'Changed states')
    data = TrainOnlyData(data_root)
    torch.set_num_threads(2)
    set_seed(7)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'plan_sha256': sha256(plan_path), 'script_sha256': sha256(__file__),
               'previous_runner_sha256': sha256(original.__file__), 'previous_evidence_sha256': sha256(previous_evidence),
               'archive_sha256': sha256(archive), 'packed_manifest_sha256': plan['packed_manifest_sha256'],
               'device': 'cpu', 'threads': 2, 'seed': 7, 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'flwr')},
               'class_order': CLASSES, 'protected_classes': list(PROTECTED),
               'validation_rows_opened': False, 'test_opened': False, 'test_evaluated': False, 'cases': []}
    atomic_json(receipt, output/'result.json')
    for choice, prior_case in zip(plan['checkpoints'], prior['cases']):
        started = time.perf_counter()
        print(f"Finite-step diagnostic at frozen round {choice['step']}; training only", flush=True)
        model, parts, settings, provenance = checkpoint(archive, choice['member_root'], choice['step'], data)
        require(provenance == prior_case['provenance'], 'Source provenance differs')
        case = inspect(model, data, parts, settings, choice['step'], prior_case, plan)
        case['provenance'], case['elapsed_seconds'] = provenance, time.perf_counter()-started
        receipt['cases'].append(case)
        atomic_json(receipt, output/'result.json')
        print(f"step={choice['step']} accepted={case['search']['accepted']} "
              f"scale={case['search'].get('accepted_scale')} trials={len(case['search']['trials'])}", flush=True)
    cases = receipt['cases']
    receipt['fixed_work'] = {
        'client_training_examples': sum(w['examples_processed'] for c in cases for w in c['client_work']),
        'client_optimizer_steps': sum(w['optimizer_steps'] for c in cases for w in c['client_work']),
        'gradient_examples': sum(c['gradient_work']['gradient_examples'] for c in cases)}
    receipt['bounded_work'] = {k: sum(c[k] for c in cases) for k in
                              ('training_forward_evaluation_examples', 'protected_risk_forward_examples')}
    receipt['bounded_work']['candidate_checks'] = sum(len(c['search']['trials']) for c in cases)
    require(receipt['fixed_work'] == plan['fixed_work'] and
            all(v <= plan['max_work'][k] for k, v in receipt['bounded_work'].items()), 'Work budget mismatch')
    require(sha256(archive) == plan['archive_sha256'], 'Source archive changed')
    receipt.update(status='complete', source_archive_unchanged=True, gate=continuation_gate(cases, plan))
    atomic_json(receipt, output/'result.json')
    print(json.dumps(receipt['gate'], indent=2), flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('data-root', 'archive', 'plan-path', 'previous-evidence', 'output'):
        p.add_argument('--'+arg, type=Path, required=True)
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
