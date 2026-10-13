"""At most five paired rounds to test whether a safe update starts detecting rare classes.

This is a training-only rescue diagnostic from one archived collapsed state.
The historical classifier, optimizer, partitions and all correction settings
stay fixed. If the correction rejects a round, STOP instead of changing it.
"""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import federated_finite_step as finite
from src.eval import federated_class_safeguard as original
from src.eval.federated_class_safeguard import (
    PROTECTED, TrainOnlyData, require, checkpoint, state_digest, vector,
    class_gradients, fedavg_round, training_evaluation,
)
from src.eval.federated_finite_step import client_risk, bounded_search
from src.models.research import atomic_json, atomic_save
from src.utils.seed import set_seed


def development_screen(rounds, plan):
    """Last scheduled endpoint, never choose a favorable intermediate result."""
    complete = len(rounds) == plan['rounds'] and all(r['search']['accepted'] for r in rounds)
    if not complete:
        return {'decision': 'stop_rejected_update', 'completed_all_rounds': False,
                'full_training_launched': False}
    a, b = [rounds[-1][arm]['metrics'] for arm in ('fedavg', 'corrected')]
    gains = {c: b['per_class'][c]['recall']-a['per_class'][c]['recall'] for c in PROTECTED[1:]}
    macro = b['macro_f1']-a['macro_f1']
    alerts = b['benign_false_alert_rate']-a['benign_false_alert_rate']
    s = plan['screen']
    passed = (max(gains.values()) >= s['min_rare_recall_gain'] and min(gains.values()) >= 0
              and macro >= -s['max_macro_f1_drop'] and alerts <= s['max_false_alert_increase'])
    return {'decision': 'eligible_for_separately_planned_development_validation' if passed
            else 'stop_no_detection_gain_within_budget', 'completed_all_rounds': True,
            'rare_recall_gains': gains, 'macro_f1_change': macro, 'false_alert_change': alerts,
            'full_training_launched': False, 'not_a_generalization_claim': True}


def paired_round(models, data, parts, settings, step, recipe):
    """One matched local epoch per arm; only the corrected arm has extra probes."""
    start = {arm: state_digest(m) for arm, m in models.items()}
    gradients, gw = class_gradients(models['corrected'], data, parts)
    ordinary, _, aw = fedavg_round(models['fedavg'], data, parts, settings, step)
    proposal, _, cw = fedavg_round(models['corrected'], data, parts, settings, step)
    baseline = client_risk(models['corrected'], data, parts)
    ids = [CLASSES.index(c) for c in PROTECTED]
    accepted, search = bounded_search(models['corrected'], vector(proposal)-vector(models['corrected']),
                                     gradients[ids], baseline, lambda m: client_risk(m, data, parts), recipe)
    require(all(state_digest(models[arm]) == start[arm] for arm in models), 'Input models were mutated')
    evaluations = {'fedavg': training_evaluation(ordinary, data),
                   'corrected': training_evaluation(accepted, data) if accepted is not None else None}
    work = {'client_training_examples': sum(w['examples_processed'] for w in aw+cw),
            'client_optimizer_steps': sum(w['optimizer_steps'] for w in aw+cw),
            'gradient_examples': gw['gradient_examples'],
            'training_forward_evaluation_examples': data.manifest['train_rows']*(1+(accepted is not None)),
            'protected_risk_forward_examples': baseline['forward_examples']+
                sum(t['risk']['forward_examples'] for t in search['trials']),
            'candidate_checks': len(search['trials'])}
    result = {'round': step, 'starting_model_sha256': start,
              'fedavg_model_sha256': state_digest(ordinary),
              'proposal_model_sha256': state_digest(proposal),
              'corrected_model_sha256': state_digest(accepted) if accepted is not None else None,
              'gradient_work': gw, 'client_work': {'fedavg': aw, 'corrected': cw},
              'baseline_client_risk': baseline, 'search': search, 'work': work, **evaluations}
    # Retain the last accepted weights if rejected, but never label them as a
    # successfully trained next round. The caller must stop on that rejection.
    return {'fedavg': ordinary, 'corrected': accepted if accepted is not None else models['corrected']}, result


def verify_first_step(result, prior):
    require(result['starting_model_sha256'] == dict.fromkeys(('fedavg', 'corrected'), prior['source_model_sha256'])
            and result['fedavg_model_sha256'] == result['proposal_model_sha256'] == prior['fedavg_model_sha256']
            and result['corrected_model_sha256'] == prior['search']['trials'][-1]['model_sha256']
            and result['gradient_work'] == prior['gradient_work']
            and all(w == prior['client_work'] for w in result['client_work'].values())
            and result['baseline_client_risk'] == prior['baseline_client_risk']
            and result['search'] == prior['search'] and result['fedavg'] == prior['fedavg']
            and result['corrected'] == prior['accepted_evaluation'], 'First-step bridge differs')


def run(data_root, archive, plan_path, recipe_path, evidence_path, output):
    output, archive = Path(output), Path(archive)
    require(not output.exists(), 'Output exists; preserve it')
    plan, recipe = [json.loads(Path(p).read_text()) for p in (plan_path, recipe_path)]
    require(plan['rounds'] == 5 and plan['source_step'] == 20, 'Only frozen five-round rescue authorized')
    require(sha256(evidence_path) == plan['finite_step_evidence_sha256']
            and sha256(finite.__file__) == plan['finite_step_runner_sha256'], 'Finite-step evidence changed')
    evidence = json.loads(Path(evidence_path).read_text())['primary']
    require(evidence['gate']['decision'] == 'eligible_for_separately_planned_short_trajectory'
            and sha256(recipe_path) == evidence['plan_sha256']
            and sha256(original.__file__) == evidence['previous_runner_sha256'], 'Changed correction recipe')
    require(sha256(archive) == plan['archive_sha256']
            and sha256(Path(data_root)/'manifest.json') == plan['packed_manifest_sha256'], 'Changed data/source')
    data = TrainOnlyData(data_root)
    torch.set_num_threads(2)
    set_seed(7)
    torch.use_deterministic_algorithms(True)
    model, parts, settings, provenance = checkpoint(archive, plan['source_member_root'], plan['source_step'], data)
    prior = next(c for c in evidence['cases'] if c['source_step'] == plan['source_step'])
    require(provenance == prior['provenance'], 'Wrong source checkpoint')
    models = {'fedavg': copy.deepcopy(model), 'corrected': copy.deepcopy(model)}
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'plan_sha256': sha256(plan_path), 'script_sha256': sha256(__file__),
               'recipe_sha256': sha256(recipe_path), 'finite_step_runner_sha256': sha256(finite.__file__),
               'finite_step_evidence_sha256': sha256(evidence_path), 'source_provenance': provenance,
               'archive_sha256': sha256(archive), 'packed_manifest_sha256': plan['packed_manifest_sha256'],
               'settings': settings, 'class_order': CLASSES, 'initial_training_evaluation': prior['before'],
               'device': 'cpu', 'threads': 2, 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'flwr')},
               'validation_rows_opened': False, 'test_opened': False, 'test_evaluated': False,
               'cloud_used': False, 'rounds': [], 'work': dict.fromkeys(plan['max_work'], 0)}
    atomic_json(receipt, output/'result.json')
    for step in range(plan['source_step']+1, plan['source_step']+plan['rounds']+1):
        started = time.perf_counter()
        print(f'Paired training-only round {step}', flush=True)
        models, result = paired_round(models, data, parts, settings, step, recipe)
        if not receipt['rounds']:
            verify_first_step(result, prior)
            receipt['first_step_bridge_exact'] = True
        result['elapsed_seconds'] = time.perf_counter()-started
        receipt['rounds'].append(result)
        for key, value in result['work'].items():
            receipt['work'][key] += value
            require(receipt['work'][key] <= plan['max_work'][key], 'Exceeded paired budget')
        atomic_json(receipt, output/'result.json')
        # Audit snapshots, not a claimed/resume-tested recovery interface.
        for arm, m in models.items():
            atomic_save({'model': m.state_dict(), 'model_sha256': state_digest(m),
                         'accepted_round': step if arm == 'fedavg' or result['search']['accepted'] else step-1},
                        output/(arm+'-last.pt'))
        print(f"round={step} accepted={result['search']['accepted']} "
              f"scale={result['search'].get('accepted_scale')}", flush=True)
        if not result['search']['accepted']:
            break
    require(state_digest(model) == prior['source_model_sha256']
            and sha256(archive) == plan['archive_sha256'], 'Frozen source changed')
    receipt['screen'] = development_screen(receipt['rounds'], plan)
    receipt['status'] = 'complete' if receipt['screen']['completed_all_rounds'] else 'stopped_rejected_update'
    receipt['source_unchanged'] = True
    atomic_json(receipt, output/'result.json')
    print(json.dumps(receipt['screen'], indent=2), flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('data-root', 'archive', 'plan-path', 'recipe-path', 'evidence-path', 'output'):
        p.add_argument('--'+arg, type=Path, required=True)
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
