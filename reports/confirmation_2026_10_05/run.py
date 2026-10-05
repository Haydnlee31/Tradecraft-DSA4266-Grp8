"""Execute only the frozen, user-authorized eight confirmation jobs.

Default is command preview. --execute requires a new output root; interrupted
individual runs can be recovered with the research runner's --resume workflow.
Two foreground child processes at most, each with two PyTorch CPU threads.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    plan_path = Path('configs/local_confirmation_plan.json')
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    commands = []
    source = None
    for lane, recipe in plan['lanes'].items():
        folder = Path(recipe['seed_zero_path'])
        m = json.loads((folder / 'environment.json').read_text())
        r = json.loads((folder / 'result.json').read_text())
        assert r['steps'] == 60 and r['test_metrics'] is None
        assert hashlib.sha256((folder / 'best.pt').read_bytes()).hexdigest() == r['checkpoint_sha256']
        assert m['model_config']['dropout'] == recipe['dropout']
        assert m['model_config']['normalization'] == recipe['normalization']
        assert all(m['settings'][k] == v for k, v in plan['common'].items())
        if source is None:
            source = m['source_sha256']
        assert source == m['source_sha256']
        for seed in plan['new_training_seeds']:
            output = Path(plan['output_root']) / lane / f'seed-{seed}'
            settings = {**plan['common'], 'normalization': recipe['normalization'], 'dropout': recipe['dropout'], 'seed': seed}
            cmd = [sys.executable, '-u', '-m', 'src.models.research', '--lane', lane, '--output', str(output)]
            for k, v in settings.items():
                cmd.extend(['--' + k.replace('_', '-'), str(v)])
            commands.append(cmd)
    assert len(commands) == plan['new_run_count'] == 8
    assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest() == v for k, v in source.items())
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
    if not args.execute:
        return
    root = Path(plan['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    # This receipt predates all training results and identifies the frozen plan.
    (root / 'launch.json').write_text(json.dumps({'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(), 'plan': plan, 'commands': commands, 'source_sha256': source, 'max_concurrent_jobs': 2}, indent=2))
    def run(cmd):
        subprocess.run(cmd, check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        # Keep only one pair in flight, so a failure prevents later pairs starting.
        for start in range(0, len(commands), 2):
            list(pool.map(run, commands[start:start + 2]))


if __name__ == '__main__':
    main()
