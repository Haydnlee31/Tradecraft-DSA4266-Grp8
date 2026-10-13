"""Frozen local SCAFFOLD mechanism trial; training data only, no model search."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.eval import federated_class_safeguard as historical
from src.eval.federated_class_safeguard import TrainOnlyData, checkpoint, require, state_digest, training_evaluation
from src.federated import scaffold
from src.models.research import atomic_json, atomic_save
from src.utils.seed import set_seed

ARMS = ('central_sgd', 'fedavg_sgd', 'scaffold')


def screen(rounds, plan):
    """Separate aggregate promise from rare-class promise; neither is validation."""
    require(len(rounds) == plan['rounds'], 'Only the fixed final endpoint may be screened')
    avg, method = [rounds[-1]['arms'][a]['evaluation']['metrics'] for a in ('fedavg_sgd', 'scaffold')]
    rare = {c: method['per_class'][c]['recall']-avg['per_class'][c]['recall']
            for c in historical.PROTECTED[1:]}
    macro = method['macro_f1']-avg['macro_f1']
    alerts = method['benign_false_alert_rate']-avg['benign_false_alert_rate']
    limits = plan['screen']
    safety = min(rare.values()) >= 0 and alerts <= limits['max_false_alert_increase']
    aggregate = safety and macro >= limits['macro_gain']
    rare_progress = safety and max(rare.values()) >= limits['rare_gain'] and macro >= -limits['max_macro_drop_for_rare_gain']
    return {'decision': 'eligible_for_separately_planned_confirmation' if aggregate or rare_progress
            else 'stop_no_convincing_gain_at_frozen_recipe',
            'macro_f1_gain': macro, 'rare_recall_gains': rare, 'false_alert_change': alerts,
            'aggregate_screen_passed': aggregate, 'rare_class_screen_passed': rare_progress,
            'safety_screen_passed': safety, 'full_training_launched': False,
            'not_a_generalization_or_novelty_claim': True}


def verify_zero_bridge(avg, method):
    require(avg['model_sha256'] == method['model_sha256'], 'Zero-control model bridge differs')
    for a, b in zip(avg['clients'], method['clients']):
        require(all(a[k] == b[k] for k in a), 'Zero-control local work/sample bridge differs')
    require(len(avg['clients']) == len(method['clients']), 'Changed client count')


def run(data_root, archive, plan_path, output):
    output, archive, plan_path = Path(output), Path(archive), Path(plan_path)
    require(not output.exists(), 'Output exists; preserve it')
    plan = json.loads(plan_path.read_text())
    require(plan['arms'] == list(ARMS) and plan['rounds'] == 10 and plan['source_step'] == 20
            and plan['clients'] == 20, 'Wrong bounded trial scope')
    require(sha256(archive) == plan['archive_sha256']
            and sha256(Path(data_root)/'manifest.json') == plan['packed_manifest_sha256'], 'Changed data/source')
    data = TrainOnlyData(data_root)
    torch.set_num_threads(2)
    set_seed(plan['recipe']['seed'])
    torch.use_deterministic_algorithms(True)
    source, parts, archived_settings, provenance = checkpoint(archive, plan['source_member_root'], 20, data)
    require(len(parts) == plan['clients'], 'Wrong partition count')
    models = {arm: copy.deepcopy(source) for arm in ARMS}
    controls = scaffold.initialize_controls(source, len(parts))
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'plan_sha256': sha256(plan_path), 'script_sha256': sha256(__file__),
               'scaffold_source_sha256': sha256(scaffold.__file__), 'historical_helper_sha256': sha256(historical.__file__),
               'archive_sha256': sha256(archive), 'packed_manifest_sha256': plan['packed_manifest_sha256'],
               'provenance': provenance, 'archived_settings_not_current_recipe': archived_settings,
               'recipe': plan['recipe'], 'class_order': CLASSES, 'source_model_sha256': state_digest(source),
               'initial_evaluation': training_evaluation(source, data),
               'device': 'cpu', 'threads': 2, 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'flwr')},
               'validation_rows_opened': False, 'test_opened': False, 'test_evaluated': False,
               'cloud_used': False, 'rounds': [], 'work_per_arm': {
                   a: {'client_training_examples': 0, 'optimizer_steps': 0} for a in ARMS},
               'training_forward_evaluation_examples': data.manifest['train_rows']}
    atomic_json(receipt, output/'result.json')
    for step in range(21, 21+plan['rounds']):
        started = time.perf_counter()
        print(f'SGD comparison at continuation {step-20}/{plan["rounds"]}; training only', flush=True)
        row = {'step': step, 'starting_model_sha256': {a: state_digest(m) for a, m in models.items()}, 'arms': {}}
        models['central_sgd'], row['arms']['central_sgd'] = scaffold.central_round(
            models['central_sgd'], data, len(parts), plan['recipe'], step)
        models['fedavg_sgd'], _, row['arms']['fedavg_sgd'] = scaffold.federated_round(
            models['fedavg_sgd'], data, parts, plan['recipe'], step)
        models['scaffold'], controls, row['arms']['scaffold'] = scaffold.federated_round(
            models['scaffold'], data, parts, plan['recipe'], step, controls)
        if step == 21:
            verify_zero_bridge(row['arms']['fedavg_sgd'], row['arms']['scaffold'])
            receipt['zero_control_first_round_exact'] = True
        # Paired FL samples must match in EVERY round, not only the zero bridge.
        for a, b in zip(row['arms']['fedavg_sgd']['clients'], row['arms']['scaffold']['clients']):
            for key in ('sampled_rows_sha256', 'sampled_class_counts', 'examples_processed', 'optimizer_steps', 'seed'):
                require(a[key] == b[key], 'Paired client sample/work mismatch')
        for arm, model in models.items():
            result = row['arms'][arm]
            work = result['clients'] if 'clients' in result else [result]
            for metric, field in (('client_training_examples', 'examples_processed'), ('optimizer_steps', 'optimizer_steps')):
                receipt['work_per_arm'][arm][metric] += sum(w[field] for w in work)
            result['evaluation'] = training_evaluation(model, data)
            receipt['training_forward_evaluation_examples'] += data.manifest['train_rows']
            print(f"  {arm}: macro_f1={result['evaluation']['metrics']['macro_f1']:.6f}", flush=True)
        row['elapsed_seconds'] = time.perf_counter()-started
        receipt['rounds'].append(row)
        atomic_json(receipt, output/'result.json')
        # Keep ALL controls for a reproducible boundary state. This entry point
        # intentionally has no resume/extension switch or best-checkpoint pick.
        atomic_save({'step': step, 'models': {a: m.state_dict() for a, m in models.items()}, 'controls': controls,
                     'plan_sha256': receipt['plan_sha256'], 'scaffold_source_sha256': receipt['scaffold_source_sha256']},
                    output/'last.pt')
    require(all(w == plan['work_per_arm'] for w in receipt['work_per_arm'].values()), 'Per-arm budget mismatch')
    total = {k: sum(w[k] for w in receipt['work_per_arm'].values()) for k in plan['work_per_arm']}
    total['training_forward_evaluation_examples'] = receipt['training_forward_evaluation_examples']
    require(total == plan['total_work'], 'Total budget mismatch')
    require(state_digest(source) == receipt['source_model_sha256'] and sha256(archive) == plan['archive_sha256'],
            'Source checkpoint changed')
    # Extra FP32 control vector upload AND broadcast per client per round.
    # These are payload estimates; physical network transfer is not measured.
    payload = source.parameter_bytes()
    receipt.update(status='complete', source_unchanged=True, total_work=total,
                   screen=screen(receipt['rounds'], plan), communication={
                       'parameter_bytes': payload, 'additional_control_bytes_per_client_per_round_each_direction': payload,
                       'extra_control_upload_bytes_all_rounds': payload*len(parts)*plan['rounds'],
                       'extra_control_broadcast_bytes_all_rounds': payload*len(parts)*plan['rounds'],
                       'persistent_control_bytes_per_client': payload,
                       'simulator_all_control_bytes': payload*(len(parts)+1),
                       'transport_overhead_included': False, 'measured_network_traffic': False})
    atomic_json(receipt, output/'result.json')
    print(json.dumps(receipt['screen'], indent=2), flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('data-root', 'archive', 'plan-path', 'output'):
        p.add_argument('--'+arg, type=Path, required=True)
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
