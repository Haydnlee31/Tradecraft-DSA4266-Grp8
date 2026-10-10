"""Bounded seed confirmation around the unchanged official39 training runner.

This file deliberately lives outside src/: adding an orchestration entry point
must not change the training-source inventory or invalidate existing recovery.
Each explicit run command schedules nine models for ONE seed, not a search.
No score can change the schedule. The final test is never opened.
"""
import argparse
import copy
from contextlib import contextmanager
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.data.official_ablation_panel import load_panel
from src.federated.official_frozen import load_frozen
from src.models import official_streaming as training
from src.models.official_ablation import ARMS, error_counts
from src.models.research import atomic_json
from src.eval.official_streaming_recovery import same

REPORTS = ROOT / 'reports/full_data_extension'
DEFAULT_PLAN = REPORTS / 'ablation-confirmation-plan.json'
LANES = ('light', 'iid', 'dirichlet')
SEEDS = (17, 27)


def require(condition, message):
    # Unlike assert, these guards also run if Python is launched with -O.
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_inventory():
    return {p.relative_to(ROOT / 'src').as_posix(): sha256(p)
            for p in sorted((ROOT / 'src').rglob('*.py'))}


def cases(seed):
    require(seed in SEEDS, 'Only prespecified confirmation seeds 17 and 27 are allowed')
    return [(lane, arm) for lane in LANES for arm in ARMS]


def load_plan(path):
    p = read(path)
    require(p['protocol'] == 'official39-ablation-confirmation-v1', 'Unknown protocol')
    require(p['confirmation_seeds'] == list(SEEDS) and p['lanes'] == list(LANES)
            and p['arms'] == list(ARMS), 'Confirmation schedule changed')
    study_path = REPORTS / 'shortcut-ablation-plan.json'
    evidence_path = REPORTS / 'ablation-seed7-results.json'
    require(sha256(study_path) == p['parent_plan_sha256'], 'Parent design changed')
    require(sha256(evidence_path) == p['seed7_evidence_sha256'], 'Seed-7 evidence changed')
    study = read(study_path)
    require(p['settings'] == study['settings'], 'Training settings changed')
    require(p['work_per_run'] == study['work_per_arm_seed'], 'Work budgets changed')
    require(p['packed_manifest_sha256'] == study['packed_manifest_sha256']
            and p['partitions'] == study['partitions'], 'Cohort or partition policy changed')
    require(p['maximum_new_runs'] == 18 and p['new_examples_processed'] == 720000000
            and p['new_optimizer_steps'] == 1409280, 'Total work budget changed')
    require(p['primary_endpoint'] == 'final_step_20_on_shared_panel'
            and p['test_evaluation_allowed'] is False
            and p['score_based_stopping_allowed'] is False, 'Endpoint or stopping policy changed')
    require(read(evidence_path)['archive_sha256'] == p['seed7_archive_sha256'],
            'Archive differs from reviewed evidence')
    return p


def preflight(plan_path, archive, data, panel, partition_root, verify_runtime=True):
    """Read-only provenance and GPU/data checks; this never constructs a model."""
    p = load_plan(plan_path)
    require(sha256(archive) == p['seed7_archive_sha256'], 'Seed-7 archive checksum changed')
    # Read exact regular members, never extract an archive into the worktree.
    with tarfile.open(archive) as saved:
        members = saved.getmembers()
        require(len({m.name for m in members}) == len(members), 'Duplicate archive members')
        def member(name):
            m = saved.getmember(name)
            require(m.isfile() and m.size < 2**20, 'Unexpected metadata member')
            return json.load(saved.extractfile(m))
        refs = {lane: member(f'outputs/official39-ablation-bridges-v1/{lane}/environment.json')
                for lane in LANES}
        for root in ('gpu-recovery', 'bridges', 'masked'):
            checks = member(f'outputs/official39-ablation-{root}-v1/checks.json')
            require(checks['status'] == 'complete' and checks['test_evaluated'] is False,
                    'Previous cloud gates incomplete')
    inventory = source_inventory()
    if verify_runtime:
        require(torch.cuda.is_available(), 'CUDA is unavailable; do not fall back to CPU')
    for lane, env in refs.items():
        require(env['source_sha256'] == inventory, 'Training source inventory changed')
        require(env['packed_manifest_sha256'] == p['packed_manifest_sha256'], 'Reference cohort changed')
        expected_settings = {k: p['settings'][k] for k in env['settings']
                             if k not in ('lane', 'epochs', 'seed')}
        expected_settings.update(lane=lane, epochs=20, seed=7)
        require(env['settings'] == expected_settings, 'Reference settings differ from frozen plan')
        require(env['input_transform']['arm'] == 'full39', 'Expected full-feature reference')
        if verify_runtime:
            require(env['device_name'] == torch.cuda.get_device_name(0), 'GPU differs from reviewed image')
            require(env['platform'] == platform.platform(), 'Operating-system image changed')
            require(env['cublas_workspace_config'] == os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    'CUBLAS_WORKSPACE_CONFIG differs from reference')
            for package, version in env['packages'].items():
                require(importlib.metadata.version(package) == version, f'Package changed: {package}')
    require(sha256(Path(data) / 'manifest.json') == p['packed_manifest_sha256'], 'Packed cohort changed')
    packed = PackedData(data)
    try:
        _, panel_info = load_panel(panel, packed)
        _, frozen_info = load_frozen(partition_root, packed, 20)
        require(panel_info['receipt_sha256'] == p['panel_receipt_sha256'], 'Panel receipt changed')
        for env in refs.values():
            require(env['input_transform']['panel'] == panel_info, 'Panel differs from seed 7')
        require(refs['dirichlet']['partition_control'] == frozen_info, 'Frozen assignments changed')
        # IID must also retain its exact client membership, not merely the seed label.
        parts, _ = training.partitions(packed.arrays['train'][1], 'iid', 20, 7, .5)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **{f'client_{i}': rows for i, rows in enumerate(parts)})
        import hashlib
        require(hashlib.sha256(buffer.getvalue()).hexdigest() == p['partitions']['iid_assignment_sha256'],
                'IID assignment bytes differ from seed 7')
    finally:
        for pair in packed.arrays.values():
            for array in pair:
                array._mmap.close()
    return {'protocol': p['protocol'], 'plan_sha256': sha256(plan_path),
            'orchestrator_sha256': sha256(__file__), 'seed7_archive_sha256': sha256(archive),
            'references': refs, 'work_per_run': p['work_per_run'],
            'test_evaluated': False, 'model_trained_by_preflight': False, 'cuda_checked': verify_runtime}


def expected_environment(context, lane, arm, seed):
    env = copy.deepcopy(context['references'][lane])
    env['settings']['seed'] = seed
    env['input_transform']['arm'] = arm
    env['input_transform']['zero_after_scaling'] = ARMS[arm]
    return env


def verify_run(path, expected, budget, initial=None):
    """Recheck artifacts before accepting or skipping a completed run on resume."""
    path = Path(path)
    env, result = read(path / 'environment.json'), read(path / 'result.json')
    epochs = expected['settings']['epochs']
    require(env == expected, 'Run environment differs from the matched design')
    require(read(path / 'status.json') == {'status': 'complete', 'step': epochs}, 'Run incomplete')
    require(result['test_metrics'] is None and env['test_evaluated'] is False, 'Test evaluation detected')
    require(result['examples_processed'] == budget['examples_processed']
            and result['optimizer_steps'] == budget['optimizer_steps'], 'Training budget mismatch')
    require(result['num_parameters'] == 5096 and result['edge_hardware_measurements'] is False,
            'Architecture or measurement scope changed')
    init = read(path / 'initialization.json')
    require(init['num_parameters'] == 5096 and (initial is None or initial == init),
            'Initial model differs across feature arms')
    state = torch.load(path / 'last.pt', map_location='cpu', weights_only=True)
    require(state['environment_sha256'] == sha256(path / 'environment.json'), 'Checkpoint environment changed')
    require(state['step'] == epochs and state['history'] == result['history'], 'Checkpoint/history mismatch')
    require(state['input_transform'] == env['input_transform'] == result['input_transform'], 'Mask metadata mismatch')
    history = result['history']
    require(len(history) == epochs and [h['step'] for h in history] == list(range(1, epochs + 1)),
            'History does not cover the fixed budget')
    require(sum(h['examples_processed'] for h in history) == budget['examples_processed']
            and sum(h['optimizer_steps'] for h in history) == budget['optimizer_steps'], 'History work mismatch')
    lane = env['settings']['lane']
    for h in history:
        require(h['examples_processed'] * epochs == budget['examples_processed']
                and h['optimizer_steps'] * epochs == budget['optimizer_steps'], 'Per-step work changed')
        for key in ('validation_metrics', 'panel_validation_metrics'):
            metrics = h[key]
            cm = np.asarray(metrics['confusion_matrix'])
            require(cm.shape == (8, 8) and np.issubdtype(cm.dtype, np.integer) and np.all(cm >= 0),
                    'Invalid confusion matrix')
            require(training.confusion_metrics(cm) == metrics, 'Metrics differ from confusion matrix')
        supports = {c: h['panel_validation_metrics']['per_class'][c]['support'] for c in CLASSES}
        require(supports == env['input_transform']['panel']['retained_class_counts'], 'Panel class support changed')
        full_cm = np.asarray(h['validation_metrics']['confusion_matrix'])
        panel_cm = np.asarray(h['panel_validation_metrics']['confusion_matrix'])
        panel_info = env['input_transform']['panel']
        require(full_cm.sum() == panel_info['retained_rows'] + panel_info['excluded_rows']
                and np.all(full_cm >= panel_cm), 'Original validation view changed')
        if lane != 'light':
            work = h['client_work']
            require(len(work) == env['settings']['clients'] and h['local_batch_cap'] == 0
                    and [w['client'] for w in work] == list(range(len(work)))
                    and all(w['examples_processed'] == w['assigned_rows'] for w in work)
                    and sum(w['examples_processed'] for w in work) == h['examples_processed']
                    and sum(w['optimizer_steps'] for w in work) == h['optimizer_steps'],
                    'Full client participation/work mismatch')
    require(result['primary_checkpoint'] == 'last.pt'
            and result['primary_validation_metrics'] == result['final_panel_validation_metrics']
            == history[-1]['panel_validation_metrics'], 'Primary endpoint was substituted')
    require(result['final_validation_metrics'] == history[-1]['validation_metrics'], 'All-validation endpoint mismatch')
    for scope, metrics in [('panel', result['primary_validation_metrics']),
                           ('all_validation', result['final_validation_metrics'])]:
        require(result[f'final_{scope}_error_counts'] == error_counts(metrics), 'Error categories changed')
    best = max(range(epochs), key=lambda i: history[i]['validation_metrics']['macro_f1'])
    require(result['best_step'] == state['best_step'] == best + 1
            and result['validation_metrics'] == history[best]['validation_metrics']
            and state['best_score'] == history[best]['validation_metrics']['macro_f1'], 'Secondary best endpoint mismatch')
    best_state = torch.load(path / 'best.pt', map_location='cpu', weights_only=True)
    require(same(best_state['model'], state['best']) and best_state['input_transform'] == env['input_transform']
            and best_state['best_step'] == state['best_step'],
            'Saved best model differs from checkpoint')
    digest = sha256(path / 'assignments.npz') if lane != 'light' else None
    require(state['partition_sha256'] == digest, 'Checkpoint assignment identity changed')
    return {'verified': True, 'initialization': init, 'assignment_sha256': digest,
            'artifact_sha256': {name: sha256(path / name) for name in
                               ('last.pt', 'best.pt', 'result.json', 'environment.json', 'initialization.json', 'status.json')},
            'examples_processed': result['examples_processed'], 'optimizer_steps': result['optimizer_steps'],
            'final_panel_metrics': result['primary_validation_metrics'],
            'final_panel_error_counts': result['final_panel_error_counts'],
            'training_validation_seconds': sum(h['elapsed_seconds'] for h in history)}


@contextmanager
def exclusive_seed_lock(root, seed):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f'seed{seed}.lock'
    # An exclusive file also works on Windows. A hard process kill can leave a
    # stale lock: inspect the job before removing that one file, never its runs.
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps({'pid': os.getpid(), 'host': platform.node(), 'seed': seed}))
    try:
        yield
    finally:
        lock.unlink()


def execute_seed(context, seed, output, data, panel, partition_root, resume=False):
    """All nine cases are fixed up front; no result-driven branching is allowed."""
    schedule = cases(seed)
    root = Path(output) / f'seed{seed}'
    identity = dict(context, seed=seed, schedule=[list(case) for case in schedule])
    if seed == 27:
        previous = Path(output) / 'seed17'
        expected_previous = dict(context, seed=17, schedule=[list(case) for case in cases(17)])
        require(read(previous / 'session.json') == expected_previous, 'Seed-17 session differs')
        prior_receipt = read(previous / 'checks.json')
        require(prior_receipt['status'] == 'complete' and len(prior_receipt['runs']) == 9
                and prior_receipt['session_sha256'] == sha256(previous / 'session.json'),
                'Complete all nine seed-17 cases before seed 27; do not gate by scores')
        for lane, arm in cases(17):
            proof = verify_run(previous / f'{lane}-{arm}', expected_environment(context, lane, arm, 17),
                               context['work_per_run'][lane], prior_receipt['runs'][f'{lane}-full39']['initialization'])
            require(proof == prior_receipt['runs'][f'{lane}-{arm}'], 'Seed-17 evidence changed')
    with exclusive_seed_lock(output, seed):
        if resume:
            require(read(root / 'session.json') == identity, 'Plan, wrapper, source or runtime changed on resume')
            receipt = read(root / 'checks.json')
            require(receipt['session_sha256'] == sha256(root / 'session.json'), 'Session identity changed')
        else:
            root.mkdir(exist_ok=False)
            atomic_json(identity, root / 'session.json')
            receipt = {'status': 'incomplete', 'seed': seed, 'test_evaluated': False,
                       'session_sha256': sha256(root / 'session.json'), 'runs': {}}
            atomic_json(receipt, root / 'checks.json')
        require(set(receipt['runs']).issubset({f'{l}-{a}' for l, a in schedule}), 'Unexpected completed run')
        paired_initial, paired_partition = {}, {}
        for lane, arm in schedule:
            name = f'{lane}-{arm}'
            target = root / name
            expected = expected_environment(context, lane, arm, seed)
            budget = context['work_per_run'][lane]
            recorded = receipt['runs'].get(name)
            if recorded is not None:
                for filename, digest in recorded['artifact_sha256'].items():
                    require(sha256(target / filename) == digest, 'Completed artifact changed; refusing to skip')
            done = target.exists() and (target / 'status.json').exists() and read(target / 'status.json')['status'] == 'complete'
            if not done:
                require(recorded is None, 'Completed receipt points to an incomplete run')
                recovering = target.exists()
                require(not recovering or (resume and (target / 'last.pt').is_file()),
                        'Partial run lacks a recoverable checkpoint; inspect without overwriting')
                training.run(data_root=data, output=target, ablation_arm=arm, validation_panel=panel,
                             partition_root=partition_root if lane == 'dirichlet' else None,
                             resume=recovering, **expected['settings'])
            proof = verify_run(target, expected, budget, paired_initial.get(lane))
            paired_initial[lane] = proof['initialization']
            digest = proof['assignment_sha256']
            if lane in paired_partition:
                require(digest == paired_partition[lane], 'Client membership differs across arms')
            if lane != 'light':
                require(digest == context['assignment_sha256'][lane], 'Client membership differs from seed 7')
            paired_partition[lane] = digest
            if recorded is not None:
                require(recorded == proof, 'Completed evidence changed on verification')
            receipt['runs'][name] = proof
            atomic_json(receipt, root / 'checks.json')
            print(f'PASS: seed={seed} {name}; pairing, budget and final panel verified', flush=True)
        receipt.update(status='complete', total_examples_processed=sum(r['examples_processed'] for r in receipt['runs'].values()),
                       total_optimizer_steps=sum(r['optimizer_steps'] for r in receipt['runs'].values()))
        require(receipt['total_examples_processed'] == 3 * sum(b['examples_processed'] for b in context['work_per_run'].values()),
                'Seed batch exposure budget mismatch')
        require(receipt['total_optimizer_steps'] == 3 * sum(b['optimizer_steps'] for b in context['work_per_run'].values()),
                'Seed batch optimizer budget mismatch')
        atomic_json(receipt, root / 'checks.json')
    return receipt


def describe(values):
    return {'values': values, 'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values),
            'min': min(values), 'max': max(values)}


def paired_summary(records):
    """Three fixed seeds are descriptive replication, not proof of equivalence."""
    seeds = (7, 17, 27)
    require(set(records) == {(s, l, a) for s in seeds for l in LANES for a in ARMS},
            'Need all 27 endpoints; do not summarize only favorable seeds or arms')
    comparisons = {}
    for lane in LANES:
        for arm in ('number_masked', 'number_total_masked'):
            pairs = [(records[s, lane, arm], records[s, lane, 'full39']) for s in seeds]
            comparisons[f'{lane}/{arm}-minus-full39'] = {
                key: describe([a[key] - b[key] for a, b in pairs])
                for key in ('macro_f1', 'benign_false_alert_rate')}
            comparisons[f'{lane}/{arm}-minus-full39']['per_class'] = {
                c: {key: describe([a['per_class'][c][key] - b['per_class'][c][key] for a, b in pairs])
                    for key in ('recall', 'precision', 'f1')} for c in CLASSES}
            comparisons[f'{lane}/{arm}-minus-full39']['error_counts'] = {
                key: describe([error_counts(a)[key] - error_counts(b)[key] for a, b in pairs])
                for key in ('attacks_missed_as_benign', 'attacks_given_wrong_attack_category')}
    return {'seed_order': list(seeds), 'paired_mask_minus_full39': comparisons,
            'seed_roles': {'7': 'exploratory reference', '17': 'prospective confirmation', '27': 'prospective confirmation'},
            'interpretation': 'Descriptive paired differences at final step 20; three seeds, fixed partitions, reused within-collection validation; no equivalence or deployment claim'}


def summarize(plan_path, output):
    p = load_plan(plan_path)
    evidence = read(REPORTS / 'ablation-seed7-results.json')
    records = {(7, lane, arm): evidence['runs'][f'{lane}/{arm}']['final_panel'] for lane in LANES for arm in ARMS}
    endpoints = {}
    for lane in LANES:
        for arm in ARMS:
            old = evidence['runs'][f'{lane}/{arm}']
            endpoints[f'seed7/{lane}/{arm}'] = {
                'final_panel_metrics': old['final_panel'], 'final_panel_error_counts': old['panel_error_counts'],
                'examples_processed': old['examples_processed'], 'optimizer_steps': old['optimizer_steps'],
                'training_validation_seconds': old['training_validation_seconds'],
                'artifact_sha256': {name: old[key] for name, key in
                                    [('last.pt', 'checkpoint_sha256'), ('environment.json', 'environment_sha256'),
                                     ('result.json', 'result_sha256')]}}
    receipts = {}
    for seed in SEEDS:
        root = Path(output) / f'seed{seed}'
        context = read(root / 'session.json')
        receipt = read(root / 'checks.json')
        require(context['plan_sha256'] == sha256(plan_path) and context['orchestrator_sha256'] == sha256(__file__),
                'Summary plan or wrapper differs from execution')
        require(context['seed'] == seed and context['work_per_run'] == p['work_per_run']
                and context['seed7_archive_sha256'] == p['seed7_archive_sha256'], 'Summary identity changed')
        require(receipt['status'] == 'complete' and receipt['test_evaluated'] is False
                and receipt['session_sha256'] == sha256(root / 'session.json') and len(receipt['runs']) == 9,
                'Confirmation incomplete')
        for lane, arm in cases(seed):
            proof = verify_run(root / f'{lane}-{arm}', expected_environment(context, lane, arm, seed),
                               context['work_per_run'][lane], receipt['runs'][f'{lane}-full39']['initialization'])
            require(proof == receipt['runs'][f'{lane}-{arm}'], 'Confirmation evidence changed')
            if lane != 'light':
                require(proof['assignment_sha256'] == p['partitions'][lane + '_assignment_sha256'], 'Partition changed')
            records[seed, lane, arm] = proof['final_panel_metrics']
            endpoints[f'seed{seed}/{lane}/{arm}'] = proof
        receipts[str(seed)] = sha256(root / 'checks.json')
    return dict(status='complete', test_evaluated=False, plan_sha256=sha256(plan_path),
                confirmation_receipts_sha256=receipts, endpoints=endpoints, **paired_summary(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run', 'summarize'))
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--seed7-archive', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--panel', type=Path)
    parser.add_argument('--partition-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--metadata-only', action='store_true',
                        help='Check only: verify source/data/evidence without claiming CUDA or runtime readiness')
    args = parser.parse_args()
    require(not args.metadata_only or args.action == 'check', 'Metadata-only is not a training mode')
    if args.action == 'summarize':
        require(args.output is not None and not args.resume and args.seed is None, 'Summary requires output, not seed/resume')
        destination = args.output / 'summary.json'
        require(not destination.exists(), 'Summary already exists; preserve it')
        result = summarize(args.plan, args.output)
        atomic_json(result, destination)
        print(f'PASS: all 27 fixed endpoints summarized in {destination}')
        return
    require(all(v is not None for v in (args.seed7_archive, args.data, args.panel, args.partition_root)),
            'Supply archive, data, panel and partition-root')
    context = preflight(args.plan, args.seed7_archive, args.data, args.panel, args.partition_root,
                        verify_runtime=not args.metadata_only)
    p = load_plan(args.plan)
    context['assignment_sha256'] = {lane: p['partitions'][lane + '_assignment_sha256'] for lane in ('iid', 'dirichlet')}
    if args.action == 'check':
        require(args.seed is None and not args.resume, 'Check does not accept seed/resume')
        print(json.dumps({'status': 'metadata_verified_runtime_pending' if args.metadata_only else 'ready_for_explicit_run',
                          'model_trained': False, 'test_evaluated': False, 'cuda_checked': context['cuda_checked'],
                          'plan_sha256': context['plan_sha256'], 'next': 'One explicit run command per seed: 17, then 27'}, indent=2))
    else:
        require(args.seed is not None and args.output is not None, 'Run requires seed and output')
        receipt = execute_seed(context, args.seed, args.output, args.data, args.panel, args.partition_root, args.resume)
        print(json.dumps({k: v for k, v in receipt.items() if k != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
