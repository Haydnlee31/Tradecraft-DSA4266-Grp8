"""Frozen-checkpoint evaluation with separate validation and sealed-test gates.

No training, fitting, threshold search or candidate selection is implemented.
Run validation-replay first. Test preparation and final scoring each require a
different explicit opt-in; neither is implied by completing validation replay.
Source files live outside src/ to preserve historical training identities.
"""
import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts import official39_closeout as closeout
from src.data.label_map import CLASSES
from src.data.official_ablation_panel import load_panel
from src.data.official_shortcut_audit import packed_arrays
from src.eval.official_evidence import close_tree, digest, metrics_from_counts, require
from src.models.architectures import MLPClassifier, config_for_variant
from src.models.official_ablation import ARMS, transform_inputs, evaluate_views

PLAN = closeout.FINAL_PLAN
BATCH_SIZE = 512


def read(path):
    return closeout.read(path)


def save(path, value):
    """Atomic UTF-8/LF receipts, only inside an explicitly owned output folder."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    temporary.replace(path)


def source_identity():
    names = ['scripts/official39_final_evaluation.py', 'scripts/official39_test_panel.py',
             'scripts/official39_closeout.py', 'src/data/label_map.py',
             'src/data/official_loader.py', 'src/data/official_shortcut_audit.py',
             'src/data/official_ablation_panel.py', 'src/models/architectures.py',
             'src/models/official_ablation.py', 'src/eval/official_evidence.py']
    return {name: digest(ROOT / name) for name in names}


def load_plan():
    plan = read(PLAN)
    closeout.audit_final_plan(plan)
    return plan


def configure(device):
    require(device in ('cpu', 'cuda'), 'Use explicit CPU or CUDA, never automatic fallback')
    if device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(device=device, device_name=torch.cuda.get_device_name(0) if device == 'cuda' else platform.machine(),
                python=platform.python_version(), platform=platform.platform(), torch=str(torch.__version__),
                numpy=np.__version__, torch_cuda=torch.version.cuda, threads=2, batch_size=BATCH_SIZE,
                deterministic=True, float32_matmul_precision='highest', tf32=False,
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def verify_pack(root, plan):
    root = Path(root)
    identity = plan['data_identity']
    require(digest(root / 'manifest.json') == identity['packed_manifest_sha256'], 'Packed manifest changed')
    manifest = read(root / 'manifest.json')
    require(manifest['files'] == identity['train_validation_arrays_sha256']
            and digest(root / 'scaler.json') == identity['scaler_sha256'], 'Inputs/scaler changed')
    require(manifest['features'] == identity['features'] and manifest['classes'] == CLASSES
            and manifest['train_rows'] == 2000000 and manifest['val_rows'] == 2059284, 'Pack schema changed')
    return manifest


def batches(x, y, features, arm, batch_size=BATCH_SIZE):
    """Sequential, bounded copies; match the historical singleton-tail policy."""
    require(x.shape == (len(y), 39) and x.dtype == np.float32 and y.dtype == np.int64,
            'Expected standardized float32 inputs and int64 targets')
    require(batch_size >= 2, 'Batch size must be at least two')
    start = 0
    while start < len(y):
        end = min(start + batch_size, len(y))
        if len(y) - end == 1:
            end += 1
        raw, labels = np.array(x[start:end], copy=True), np.array(y[start:end], copy=True)
        require(np.isfinite(raw).all() and np.all((labels >= 0) & (labels < len(CLASSES))),
                'Nonfinite feature or invalid target, including masked columns')
        yield transform_inputs(raw, features, arm), labels
        start = end


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name, list(array.shape), str(array.dtype)]).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def score(model, iterator, device, retained):
    """One forward per batch, two populations from identical argmax predictions."""
    require(retained.ndim == 1 and retained.dtype == bool and retained.any(), 'Invalid retained-row mask')
    before = state_digest(model)
    model.eval()
    full = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    panel = np.zeros_like(full)
    offset, forwards = 0, 0
    with torch.no_grad():
        for x, y in iterator:
            require(len(x) == len(y) and len(y) > 0 and offset + len(y) <= len(retained), 'Batch coverage mismatch')
            require(x.shape == (len(y), 39) and x.dtype == np.float32 and np.isfinite(x).all()
                    and y.ndim == 1 and y.dtype == np.int64 and np.all((y >= 0) & (y < len(CLASSES))),
                    'Invalid inference batch')
            logits = model(torch.from_numpy(x).to(device))
            require(logits.shape == (len(y), len(CLASSES)) and torch.isfinite(logits).all().item(),
                    'Invalid model output; no silent replacement')
            # torch.argmax returns the first canonical index on an exact tie.
            predicted = logits.argmax(1).cpu().numpy()
            selected = retained[offset:offset + len(y)]
            full += np.bincount(y * len(CLASSES) + predicted, minlength=len(CLASSES)**2).reshape(full.shape)
            panel += np.bincount(y[selected] * len(CLASSES) + predicted[selected],
                                 minlength=len(CLASSES)**2).reshape(panel.shape)
            offset += len(y)
            forwards += 1
    require(offset == len(retained) and state_digest(model) == before, 'Coverage or model state changed')
    require(int(panel.sum()) == int(retained.sum()), 'Panel rows lost')
    result = dict(rows=offset, primary_rows=int(retained.sum()), forward_batches=forwards,
                  model_state_sha256=before, weights_unchanged=True)
    for name, matrix in [('all', full), ('panel', panel)]:
        supports = dict(zip(CLASSES, map(int, matrix.sum(1))))
        metrics, errors = metrics_from_counts(matrix.tolist(), supports)
        result[name] = dict(metrics=metrics, errors=errors)
    return result


class Artifacts:
    """Read only exact hash-pinned archive members; never extract or execute code."""
    def __init__(self, root, candidates):
        self.stack, self.archives, self.members = ExitStack(), {}, {}
        try:
            specs = {c['artifact']['archive']: c['artifact']['archive_sha256'] for c in candidates}
            for name, expected in specs.items():
                require(Path(name).name == name, 'Archive must be a basename')
                path = Path(root) / name
                require(digest(path) == expected, 'Archive checksum changed')
                archive = self.stack.enter_context(tarfile.open(path))
                members = archive.getmembers()
                require(len(members) <= 2000 and len({m.name for m in members}) == len(members),
                        'Unexpected or duplicate archive entries')
                self.archives[name], self.members[name] = archive, {m.name: m for m in members}
        except BaseException:
            self.close()
            raise

    def close(self):
        self.stack.close()

    def member(self, artifact, kind, filename):
        name = artifact[kind]
        require(not name.startswith('/') and '..' not in Path(name).parts and '\\' not in name,
                'Unsafe member name')
        member = self.members[artifact['archive']][name]
        require(member.isfile() and member.size < 12_000_000, 'Invalid model artifact member')
        raw = self.archives[artifact['archive']].extractfile(member).read()
        require(hashlib.sha256(raw).hexdigest() == artifact['sha256'][filename], 'Artifact checksum changed')
        return raw

    def load(self, candidate, plan, device):
        spec = candidate['artifact']
        env_raw = self.member(spec, 'environment_member', 'environment.json')
        env = closeout.read_json(env_raw)
        result = closeout.read_json(self.member(spec, 'result_member', 'result.json'))
        require(env['classes'] == CLASSES and env['features'] == plan['data_identity']['features']
                and env['packed_manifest_sha256'] == plan['data_identity']['packed_manifest_sha256'], 'Checkpoint cohort changed')
        settings = env['settings']
        require(settings['lane'] == candidate['lane'] and settings['seed'] == candidate['seed']
                and settings['epochs'] == 20 and settings['batch_size'] == BATCH_SIZE, 'Wrong lane/seed/budget')
        transform = env['input_transform']
        require(transform['arm'] == candidate['arm'] and transform['zero_after_scaling'] == ARMS[candidate['arm']]
                and transform['features'] == env['features'] and transform['value'] == 0., 'Wrong input transform')
        require(env['model_config'] == dict(name='centralized_light', hidden_dims=[64, 32], dropout=.2, normalization='layer'),
                'Model architecture changed')
        for name in ('models/architectures.py', 'models/official_ablation.py'):
            require(digest(ROOT / 'src' / name) == env['source_sha256'][name], 'Inference source changed since training')
        require(env['test_evaluated'] is False and result['test_metrics'] is None
                and result['primary_checkpoint'] == 'last.pt', 'Wrong training or selection scope')
        checkpoint = torch.load(io.BytesIO(self.member(spec, 'checkpoint_member', 'last.pt')),
                                map_location='cpu', weights_only=True)
        require(checkpoint['step'] == 20 and checkpoint['environment_sha256'] == hashlib.sha256(env_raw).hexdigest()
                and checkpoint['input_transform'] == transform and checkpoint['history'] == result['history'],
                'Checkpoint identity/history mismatch')
        require(result['final_validation_metrics'] == checkpoint['history'][-1]['validation_metrics']
                and result['primary_validation_metrics'] == checkpoint['history'][-1]['panel_validation_metrics'],
                'Not the final endpoint')
        model = MLPClassifier(39, len(CLASSES), replace(config_for_variant('light'), normalization='layer'))
        model.load_state_dict(checkpoint['model'], strict=True)
        require(model.num_parameters() == 5096, 'Parameter count changed')
        return model.to(device), env, result


@contextmanager
def session(output, identity, resume):
    """Exclusive output ownership and model-boundary recovery, never overwrite."""
    output = Path(output)
    if resume:
        require(read(output / 'session.json') == identity, 'Session identity changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        save(output / 'session.json', identity)
    lock = output / 'running.lock'
    # A hard kill may leave this file: inspect the host/PID before any manual
    # removal. Never delete locks/checkpoints merely because SSH disconnected.
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(pid=os.getpid(), host=platform.node())))
    try:
        yield output
    finally:
        lock.unlink()


def verify_scored(record, identity, candidate):
    require(record['session_sha256'] == identity and record['candidate'] == candidate
            and record['status'] == 'complete' and record['weights_unchanged'] is True, 'Saved result identity changed')
    for name in ('all', 'panel'):
        metrics = record[name]['metrics']
        expected, errors = metrics_from_counts(metrics['confusion_matrix'],
                                               {c: metrics['per_class'][c]['support'] for c in CLASSES})
        close_tree(metrics, expected)
        close_tree(record[name]['errors'], errors)
    require(sum(v['support'] for v in record['all']['metrics']['per_class'].values()) == record['rows']
            and sum(v['support'] for v in record['panel']['metrics']['per_class'].values()) == record['primary_rows'],
            'Saved scoring counts changed')
    require(np.all(np.asarray(record['panel']['metrics']['confusion_matrix']) <=
                   np.asarray(record['all']['metrics']['confusion_matrix'])), 'Panel is not a subset of original rows')


class Journal:
    """Resume only committed, checksum-matching cases; never adopt orphan files.

    A crash between saving a case and committing its checksum leaves an orphan.
    Stop for inspection in that rare case rather than silently accepting or
    overwriting it. Normal interruption between cases needs only --resume.
    """
    def __init__(self, folder, session_hash, resume):
        self.folder, self.path = folder, folder / 'progress.json'
        if resume:
            self.state = read(self.path)
            require(self.state['session_sha256'] == session_hash, 'Journal identity changed')
        else:
            self.state = dict(session_sha256=session_hash, cases={})
            save(self.path, self.state)
        actual = {p.name for p in folder.glob('case-*.json')}
        require(actual == set(self.state['cases']), 'Orphan or missing case; preserve output for inspection')
        require(actual <= {f'case-{i:02d}.json' for i in range(27)}, 'Unexpected case file')
        for name, expected in self.state['cases'].items():
            require(digest(folder / name) == expected, 'Completed case checksum changed')

    def get(self, target):
        return read(target) if target.name in self.state['cases'] else None

    def commit(self, target, record):
        require(not target.exists() and target.name not in self.state['cases'], 'Preserve completed case')
        save(target, record)
        self.state['cases'][target.name] = digest(target)
        save(self.path, self.state)


def verify_comparison(record):
    require(set(record['historical_comparison']) == {'all', 'panel'}, 'Incomplete historical comparison')
    require(type(record['same_runtime_reference_exact']) is bool, 'Missing same-runtime reference')
    for result in record['historical_comparison'].values():
        require(type(result['exact']) is bool and type(result['confusion_l1_difference']) is int
                and result['confusion_l1_difference'] >= 0
                and result['exact'] == (result['confusion_l1_difference'] == 0), 'Invalid historical comparison')
        # Counts must be exactly identical. The historical NumPy F1 reduction
        # and the stdlib evidence reduction can differ in their last float bit.
        require(not result['exact'] or abs(result['macro_f1_difference']) <= 1e-12, 'Inconsistent exact comparison')


def compare_archive(scored, original):
    result = {}
    for name, key in [('all', 'final_validation_metrics'), ('panel', 'primary_validation_metrics')]:
        current = np.asarray(scored[name]['metrics']['confusion_matrix'], np.int64)
        historical = np.asarray(original[key]['confusion_matrix'], np.int64)
        result[name] = dict(exact=bool(np.array_equal(current, historical)),
                            confusion_l1_difference=int(np.abs(current - historical).sum()),
                            macro_f1_difference=scored[name]['metrics']['macro_f1'] - original[key]['macro_f1'])
    return result


def validation_replay(archives, packed, panel_root, output, device='cpu', resume=False):
    plan, runtime = load_plan(), configure(device)
    manifest = verify_pack(packed, plan)
    identity = dict(protocol='official39-validation-replay-v1', plan_sha256=digest(PLAN),
                    source_sha256=source_identity(), runtime=runtime, test_opened=False, model_trained=False,
                    test_evaluated=False, candidates=plan['candidates'])
    with packed_arrays(Path(packed), manifest) as arrays:
        # The existing panel loader expects only these three data attributes.
        from types import SimpleNamespace
        keep, panel_info = load_panel(panel_root, SimpleNamespace(root=Path(packed), arrays=arrays, manifest=manifest))
        require(panel_info['receipt_sha256'] == read(closeout.REPORTS / 'ablation-confirmation-plan.json')['panel_receipt_sha256'],
                'Validation panel changed')
        identity['validation_panel'] = panel_info
        with session(output, identity, resume) as folder:
            session_hash = digest(folder / 'session.json')
            journal = Journal(folder, session_hash, resume)
            store = Artifacts(archives, plan['candidates'])
            try:
                records, hashes = [], {}
                for index, candidate in enumerate(plan['candidates']):
                    target = folder / f'case-{index:02d}.json'
                    record = journal.get(target)
                    if record is not None:
                        verify_scored(record, session_hash, candidate)
                    else:
                        model, env, original = store.load(candidate, plan, device)
                        require(env['input_transform']['panel'] == panel_info, 'Candidate panel changed')
                        x, y = arrays['val']
                        started = time.perf_counter()
                        record = score(model, batches(x, y, manifest['features'], candidate['arm']), device, keep)
                        # An independent pass through the unchanged training
                        # evaluator distinguishes new-code errors from CPU/CUDA
                        # numeric differences. Never relax the historical gate.
                        _, full_ref, _, panel_ref = evaluate_views(model,
                            batches(x, y, manifest['features'], candidate['arm']), device, keep)
                        local_exact = (record['all']['metrics']['confusion_matrix'] == full_ref['confusion_matrix']
                                       and record['panel']['metrics']['confusion_matrix'] == panel_ref['confusion_matrix'])
                        record.update(status='complete', candidate=candidate, session_sha256=session_hash,
                                      test_evaluated=False, same_runtime_reference_exact=local_exact,
                                      historical_comparison=compare_archive(record, original),
                                      replay_seconds=time.perf_counter() - started)
                        verify_scored(record, session_hash, candidate)
                        verify_comparison(record)
                        journal.commit(target, record)
                        del model
                    verify_comparison(record)
                    records.append(record)
                    hashes[target.name] = digest(target)
                    historical = all(v['exact'] for v in record['historical_comparison'].values())
                    print(f'{candidate["id"]}: local_reference={record["same_runtime_reference_exact"]} archived_CUDA_counts={historical}', flush=True)
                ready = all(r['same_runtime_reference_exact'] and all(v['exact'] for v in r['historical_comparison'].values()) for r in records)
                receipt = dict(status='complete' if ready else 'blocked_replay_mismatch',
                               session_sha256=session_hash, cases=hashes, candidates_verified=len(records),
                               exact_same_runtime_references=sum(r['same_runtime_reference_exact'] for r in records),
                               exact_historical_endpoints=sum(all(v['exact'] for v in r['historical_comparison'].values()) for r in records),
                               runtime=runtime, source_sha256=identity['source_sha256'], plan_sha256=digest(PLAN),
                               test_opened=False, test_evaluated=False, model_trained=False,
                               test_scoring_ready=ready, summed_replay_seconds=sum(r['replay_seconds'] for r in records))
                save(folder / 'checks.json', receipt)
                return receipt
            finally:
                store.close()


def verify_replay(root, runtime):
    """Recheck actual case files, not just a Boolean in the top-level receipt."""
    root = Path(root)
    plan, receipt, identity = load_plan(), read(root / 'checks.json'), read(root / 'session.json')
    require(receipt['status'] == 'complete' and receipt['test_scoring_ready'] is True
            and receipt['plan_sha256'] == digest(PLAN) and receipt['source_sha256'] == source_identity()
            and receipt['runtime'] == runtime and receipt['test_evaluated'] is False, 'Validation gate not satisfied')
    require(receipt['session_sha256'] == digest(root / 'session.json')
            and identity['candidates'] == plan['candidates'] and identity['runtime'] == runtime
            and identity['source_sha256'] == source_identity(), 'Replay identity changed')
    require(identity['test_opened'] is False and identity['test_evaluated'] is False
            and receipt['test_opened'] is False and receipt['model_trained'] is False, 'Wrong replay scope')
    require(set(receipt['cases']) == {f'case-{i:02d}.json' for i in range(27)}, 'Incomplete replay case set')
    require(Journal(root, receipt['session_sha256'], True).state['cases'] == receipt['cases'], 'Journal differs from receipt')
    evidence = read(closeout.EVIDENCE)
    expected_panel = {(r['seed'], r['group'], r['condition']): r['metrics']['confusion_matrix'] for r in evidence['records']}
    for index, candidate in enumerate(plan['candidates']):
        name = f'case-{index:02d}.json'
        require(digest(root / name) == receipt['cases'][name], 'Replay artifact changed')
        r = read(root / name)
        verify_scored(r, receipt['session_sha256'], candidate)
        verify_comparison(r)
        require(r['rows'] == 2059284 and r['primary_rows'] == 2047805
                and r['panel']['metrics']['confusion_matrix'] == expected_panel[candidate['seed'], candidate['lane'], candidate['arm']],
                'Replay does not match pinned primary evidence')
        require(r['test_evaluated'] is False and r['same_runtime_reference_exact'] is True
                and all(v['exact'] is True for v in r['historical_comparison'].values()), 'Replay difference unresolved')
    return digest(root / 'checks.json')


def test_evaluation(archives, prepared, replay, output, device='cpu', resume=False, allow_final_test=False):
    require(allow_final_test is True, 'Explicit final-test approval required; test stays closed')
    from scripts.official39_test_panel import prepared_arrays
    plan, runtime = load_plan(), configure(device)
    replay_hash = verify_replay(replay, runtime)
    identity = dict(protocol='official39-final-execution-v1', plan_sha256=digest(PLAN),
                    source_sha256=source_identity(), runtime=runtime, validation_replay_sha256=replay_hash,
                    preparation_receipt_sha256=digest(Path(prepared) / 'receipt.json'),
                    explicit_final_test_opt_in=True, model_trained=False, candidates=plan['candidates'])
    with prepared_arrays(prepared, plan) as (x, y, keep, prep):
        with session(output, identity, resume) as folder:
            session_hash = digest(folder / 'session.json')
            journal = Journal(folder, session_hash, resume)
            store = Artifacts(archives, plan['candidates'])
            try:
                records, hashes = [], {}
                for index, candidate in enumerate(plan['candidates']):
                    target = folder / f'case-{index:02d}.json'
                    record = journal.get(target)
                    fresh = record is None
                    if fresh:
                        model, _, _ = store.load(candidate, plan, device)
                        record = score(model, batches(x, y, plan['data_identity']['features'], candidate['arm']), device, keep)
                        record.update(status='complete', candidate=candidate, session_sha256=session_hash, test_evaluated=True)
                        del model
                    verify_scored(record, session_hash, candidate)
                    require(record['test_evaluated'] is True and record['rows'] == prep['rows']
                            and record['primary_rows'] == prep['panel']['retained_rows'], 'Test population changed')
                    for scope, supports in [('all', prep['class_counts']), ('panel', prep['panel']['retained_class_counts'])]:
                        require({c: record[scope]['metrics']['per_class'][c]['support'] for c in CLASSES} == supports,
                                'Scored test supports changed')
                    if fresh:
                        journal.commit(target, record)
                    records.append(record)
                    hashes[target.name] = digest(target)
                    print(f'Completed fixed case {index+1}/27: {candidate["id"]}', flush=True)
                summaries = {}
                for scope in ('all', 'panel'):
                    view = [dict(seed=r['candidate']['seed'], group=r['candidate']['lane'], condition=r['candidate']['arm'],
                                 metrics=r[scope]['metrics'], errors=r[scope]['errors']) for r in records]
                    groups = {}
                    for lane in closeout.LANES:
                        for arm in closeout.ARMS:
                            chosen = sorted((r for r in view if r['group'] == lane and r['condition'] == arm), key=lambda r: r['seed'])
                            groups[f'{lane}/{arm}'] = {k: closeout.describe([r['metrics'][k] for r in chosen])
                                                      for k in ('macro_f1', 'benign_false_alert_rate')}
                            groups[f'{lane}/{arm}']['per_class'] = {c: {k: closeout.describe([r['metrics']['per_class'][c][k] for r in chosen])
                                                                      for k in ('precision', 'recall', 'f1')} for c in CLASSES}
                    summaries[scope] = dict(groups=groups, paired=closeout.paired_analysis({'records': view}))
                result = dict(status='complete', session_sha256=session_hash, cases=hashes, candidates=27,
                              test_evaluated=True, model_trained=False, model_promoted=False, deployment_authorized=False,
                              primary_population='panel', secondary_population='all', seed_order=list(closeout.SEEDS), summary=summaries)
                save(folder / 'checks.json', result)
                return result
            finally:
                store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validation-replay', 'evaluate-test'))
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packed', type=Path)
    parser.add_argument('--validation-panel', type=Path)
    parser.add_argument('--prepared-test', type=Path)
    parser.add_argument('--validation-replay', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-final-test-evaluation', action='store_true')
    args = parser.parse_args()
    if args.action == 'validation-replay':
        require(not args.allow_final_test_evaluation and args.prepared_test is None and args.validation_replay is None,
                'Validation replay cannot access test artifacts')
        require(args.packed is not None and args.validation_panel is not None, 'Supply packed data and validation panel')
        result = validation_replay(args.archives, args.packed, args.validation_panel, args.output, args.device, args.resume)
    else:
        require(args.prepared_test is not None and args.validation_replay is not None
                and args.packed is None and args.validation_panel is None, 'Supply prepared-test and validation-replay only')
        result = test_evaluation(args.archives, args.prepared_test, args.validation_replay, args.output,
                                 args.device, args.resume, args.allow_final_test_evaluation)
    print(json.dumps({k: v for k, v in result.items() if k not in ('summary', 'source_sha256', 'cases')}, indent=2))


if __name__ == '__main__':
    main()
