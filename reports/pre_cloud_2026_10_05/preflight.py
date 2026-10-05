"""Final local pre-cloud checks; never launch cloud resources or read test data.

Fresh actual-data CPU recovery pilots use two steps, not a new model screen.
Output directories must be new. Inventory hashes identify separately transferred
data/checkpoints; no raw data or model weights are copied into the repository.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

import torch


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.runs.mkdir(parents=True, exist_ok=False)
    def command(cmd, name):
        result = subprocess.run(cmd, text=True, capture_output=True)
        # Strip terminal colours and trailing spaces from versioned evidence logs.
        log = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout + result.stderr)
        (args.output / f'{name}.log').write_text('\n'.join(line.rstrip() for line in log.splitlines()) + '\n')
        if result.returncode:
            raise RuntimeError(f'{name} failed; see saved log')
    command([sys.executable, '-m', 'pip', 'check'], 'dependencies')
    command([sys.executable, '-m', 'unittest', 'discover', '-v'], 'tests')
    checks = {}
    for lane, norm in (('heavy', 'batch'), ('dirichlet', 'layer')):
        common = [sys.executable, '-m', 'src.models.research', '--lane', lane,
                  '--normalization', norm, '--epochs', '2', '--seed', '7',
                  '--partition-seed', '0', '--patience', '0', '--device', 'cpu',
                  '--threads', '2', '--batch-size', '512', '--clients', '20',
                  '--alpha', '0.5', '--lr', '0.001', '--weight-decay', '0.00001',
                  '--loss', 'sqrt_weighted_ce']
        full, resumed = args.runs / f'{lane}-full', args.runs / f'{lane}-resumed'
        command(common + ['--output', str(full)], lane + '-full')
        command(common + ['--output', str(resumed), '--stop-after', '1'], lane + '-pause')
        if read(resumed / 'status.json')['status'] != 'paused':
            raise ValueError('Expected a paused checkpoint')
        command(common + ['--output', str(resumed), '--resume'], lane + '-resume')
        a = torch.load(full / 'last.pt', map_location='cpu', weights_only=True)
        b = torch.load(resumed / 'last.pt', map_location='cpu', weights_only=True)
        if a['step'] != 2 or b['step'] != 2:
            raise ValueError('Incomplete pilot')
        for key in a['model']:
            torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
        ar, br = read(full / 'result.json'), read(resumed / 'result.json')
        if ar['validation_metrics'] != br['validation_metrics'] or ar['test_metrics'] is not None or br['test_metrics'] is not None:
            raise ValueError('Resume metric parity failed')
        checks[lane] = {'normalization': norm, 'steps': 2, 'exact_final_model_state_match': True,
                        'exact_validation_metrics_match': True, 'test_evaluated': False,
                        'run_directory': str(full), 'resumed_directory': str(resumed)}
        print(f'{lane}: actual-data CPU resume parity passed', flush=True)
    study = read('reports/confirmation_2026_10_05/results/summary.json')
    paths = [Path('data/splits/train.parquet'), Path('data/splits/val.parquet')]
    # Include complete recovery directories in transfer guidance; inventory all
    # retained files rather than pretending Git contains ignored checkpoint data.
    for data in study['lanes'].values():
        for run in data['runs'].values():
            paths.extend(p for p in Path(run['path']).rglob('*') if p.is_file())
    paths.extend(p for p in Path('outputs/explain-confirmation-2026-10-05').rglob('*') if p.is_file())
    inventory = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(paths))]
    summary = {'local_tests_passed': True, 'dependency_check_passed': True,
               'actual_data_cpu_recovery': checks, 'cuda_available': torch.cuda.is_available(),
               'cuda_training_verified': False, 'cloud_resources_launched': False,
               'readiness': 'local checkpoint for a bounded cloud pilot; not full-scale or deployment clearance',
               'python': sys.version, 'platform': platform.platform(),
               'packages': {p: importlib.metadata.version(p) for p in ('torch', 'flwr', 'numpy', 'polars', 'scikit-learn', 'shap')},
               'current_source_sha256': {str(p): sha(p) for p in sorted(Path('src').rglob('*.py'))},
               'transfer_inventory': inventory, 'transfer_bytes': sum(p['bytes'] for p in inventory),
               'test_data_in_inventory': False}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('Pre-cloud local checks completed', flush=True)


if __name__ == '__main__':
    main()
