"""Measure frozen official39 inference cost on local TRAIN inputs only.

Quality comes from already published aggregate evidence, never from scoring the
benchmark inputs. Timings are warm-cache forward+argmax, not edge deployment or
end-to-end flow latency. Repetition measures timing variability, not accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch

from scripts import official39_final_evaluation as frozen
from scripts import official39_final_report as final
from src.eval.official_evidence import digest, require
from src.models.research import atomic_json

ROOT = Path(__file__).parents[2]
PLAN = ROOT/'reports/federated_class_safeguards/cost-benchmark-plan.json'


def hardware():
    """Only non-identifying CPU/RAM fields; never collect hardware serial IDs."""
    result = {'platform': platform.platform(), 'architecture': platform.machine(),
              'cpu_model': platform.processor() or 'unavailable', 'ram_bytes': None}
    if platform.system() == 'Darwin':
        for key, name in (('cpu_model', 'machdep.cpu.brand_string'), ('ram_bytes', 'hw.memsize')):
            query = subprocess.run(['sysctl', '-n', name], capture_output=True, text=True, timeout=5)
            if query.returncode == 0:
                result[key] = int(query.stdout.strip()) if key == 'ram_bytes' else query.stdout.strip()
    return result


def timing_summary(values, batch_size):
    require(len(values) > 0 and all(np.isfinite(x) and x > 0 for x in values), 'Invalid timing samples')
    median = statistics.median(values)
    return {'repeat_mean_call_ms': values, 'median_mean_call_ms': median,
            'min_mean_call_ms': min(values), 'max_mean_call_ms': max(values),
            'amortized_ms_per_flow': median/batch_size, 'flows_per_second': 1000*batch_size/median,
            'tail_request_latency_measured': False}


def benchmark_cases(models, inputs, plan):
    """Rotate ALL model/batch cases, keeping tensor preparation outside timing."""
    require(all(not m.training for m in models.values()), 'Frozen eval-mode models required')
    batches = {size: [inputs[start:start+size] for start in range(0, len(inputs), size)]
               for size in plan['batch_sizes']}
    require(all(all(len(x) == size for x in bs) for size, bs in batches.items()), 'Partial benchmark batch')
    cases = [(name, size) for name in models for size in plan['batch_sizes']]
    samples = {case: [] for case in cases}
    work = {'warmup_forward_examples': 0, 'timed_forward_examples': 0, 'timed_calls': 0}
    with torch.inference_mode():
        for name, size in cases:
            for i in range(plan['warmup_calls_per_case']):
                models[name](batches[size][i % len(batches[size])]).argmax(1)
                work['warmup_forward_examples'] += size
        for repeat in range(plan['repeats']):
            order = cases[repeat % len(cases):]+cases[:repeat % len(cases)]
            for name, size in order:
                model, pool = models[name], batches[size]
                count = plan['calls_per_repeat'][str(size)]
                started = time.perf_counter_ns()
                for i in range(count):
                    prediction = model(pool[i % len(pool)]).argmax(1)
                elapsed = time.perf_counter_ns()-started
                require(prediction.shape == (size,), 'Prediction shape changed')
                samples[name, size].append(elapsed/1e6/count)
                work['timed_calls'] += count
                work['timed_forward_examples'] += count*size
    return {name: {str(size): timing_summary(samples[name, size], size) for size in plan['batch_sizes']}
            for name in models}, work


def fingerprints(models, inputs):
    # A bounded training-input fingerprint is a provenance check, not metrics.
    with torch.inference_mode():
        return {name: hashlib.sha256(model(inputs[:512]).argmax(1).numpy().astype('<i8').tobytes()).hexdigest()
                for name, model in models.items()}


def run(data_root, archive_root, output, plan_path=PLAN):
    output, data_root, plan_path = Path(output), Path(data_root), Path(plan_path)
    require(not output.exists(), 'Output exists; preserve it')
    settings = json.loads(plan_path.read_text())
    require(settings['device'] == 'cpu' and settings['batch_sizes'] == [1, 512]
            and settings['repeats'] == 9 and settings['input_rows'] == 4096, 'Changed benchmark scope')
    plan = frozen.load_plan()
    evidence = json.loads(final.EVIDENCE.read_text())
    final.audit(evidence)
    candidates = [c for c in plan['candidates'] if c['arm'] == 'full39']
    require(len(candidates) == 9, 'All nine references required')
    # Do NOT use PackedData or validation verification helpers here: they can
    # open validation arrays. This benchmark needs training features only.
    require(digest(data_root/'manifest.json') == plan['data_identity']['packed_manifest_sha256']
            and digest(data_root/'train_x.npy') == plan['data_identity']['train_validation_arrays_sha256']['train_x.npy'],
            'Training input identity changed')
    x = np.load(data_root/'train_x.npy', mmap_mode='r', allow_pickle=False)
    require(x.shape == (2000000, 39) and x.dtype == np.float32, 'Wrong training features')
    rows = np.random.default_rng(settings['input_seed']).choice(len(x), settings['input_rows'], replace=False)
    inputs = torch.from_numpy(np.array(x[rows], copy=True))
    require(bool(torch.isfinite(inputs).all()), 'Nonfinite benchmark inputs')
    torch.set_num_threads(settings['threads'])
    torch.set_num_interop_threads(settings['interop_threads'])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    records = {r['candidate']['id']: r for name, r in evidence['documents'].items()
               if name.startswith(final.FINAL+'case-')}
    models, identities = {}, {}
    artifacts = frozen.Artifacts(archive_root, candidates)
    try:
        for candidate in candidates:
            model, _, _ = artifacts.load(candidate, plan, 'cpu')
            model.eval()
            name, state = candidate['id'], frozen.state_digest(model)
            require(state == records[name]['model_state_sha256'], 'Weights differ from final reference')
            models[name] = model
            identities[name] = {'model_state_sha256': state, 'num_parameters': model.num_parameters(),
                                'parameter_bytes': model.parameter_bytes(), 'artifact': candidate['artifact']}
    finally:
        artifacts.close()
    before = fingerprints(models, inputs)
    measurements, work = benchmark_cases(models, inputs, settings)
    require(before == fingerprints(models, inputs), 'Frozen input predictions changed')
    require(all(frozen.state_digest(m) == identities[name]['model_state_sha256'] for name, m in models.items()),
            'Frozen weights changed')
    result = {'status': 'complete', 'plan_sha256': digest(plan_path), 'script_sha256': digest(__file__),
              'final_evidence_sha256': digest(final.EVIDENCE), 'final_plan_sha256': digest(frozen.PLAN),
              'loader_sha256': digest(frozen.__file__), 'architecture_sha256': digest(ROOT/'src/models/architectures.py'),
              'packed_manifest_sha256': plan['data_identity']['packed_manifest_sha256'],
              'hardware': hardware(), 'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                  'numpy': np.__version__, 'device': 'cpu', 'threads': torch.get_num_threads(),
                  'interop_threads': torch.get_num_interop_threads(), 'matmul_precision': torch.get_float32_matmul_precision()},
              'input_rows_sha256': hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),
              'input_tensor_sha256': hashlib.sha256(inputs.numpy().astype('<f4').tobytes()).hexdigest(),
              'models': identities, 'prediction_fingerprints': before, 'measurements': measurements,
              'work': dict(work, fingerprint_forward_examples=2*512*len(models), optimizer_steps=0, backward_batches=0),
              'weights_unchanged': True, 'validation_rows_opened': False, 'test_rows_opened': False,
              'test_scored': False, 'training_labels_opened': False, 'quality_scoring_performed': False,
              'cloud_used': False, 'edge_hardware_measured': False, 'timing_scope': settings['timed_scope']}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(result, output/'result.json')
    for name, values in measurements.items():
        print(f"{name}: batch1 {values['1']['median_mean_call_ms']:.5f} ms; batch512 {values['512']['median_mean_call_ms']:.5f} ms", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('data-root', 'archive-root', 'output'):
        p.add_argument('--'+arg, required=True, type=Path)
    run(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
