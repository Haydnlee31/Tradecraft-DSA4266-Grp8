"""Supported entry point for the frozen primary thesis study.

This delegates to existing, hash-pinned implementations. It does not alter
hyperparameters, introduce model selection, or merge the MimeLite supplement.
Imports are lazy so --help and input setup do not require a loaded ML runtime.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVIDENCE = ROOT / 'reports/methodology_comparison/results.json'


def call(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def compare(actual, reference):
    """Compare scientific endpoints, excluding elapsed time and file packaging.

Model hashes cover numerical weights; logits and sample hashes cover predictions
and training streams. Serialized checkpoint bytes may depend on packaging, so
they are checked by the training/evaluation runner, not equated across runs here.
"""
    from scripts.methodology_report import summarize
    summarize(actual)
    expected = {(r['seed'], r['arm']): r for r in reference['results']}
    fields = ('settings', 'work', 'initial_model_sha256', 'model_sha256',
              'sample_chain_sha256', 'development_logits_sha256', 'development',
              'recovery_exact', 'recovery_backward_examples',
              'confirmation_logits_sha256', 'confirmation')
    checks = []
    for row in actual['results']:
        key = (row['seed'], row['arm'])
        match = {name: row[name] == expected[key][name] for name in fields}
        if not all(match.values()):
            raise ValueError(f'Endpoint differs for {key}: {[k for k,v in match.items() if not v]}')
        checks.append({'seed': key[0], 'arm': key[1], 'exact': match})
    return {'status': 'exact', 'endpoints': len(checks), 'checks': checks,
            'scope': 'Strict original-runtime replication; no cross-platform equality claim'}


def render(run, output):
    """Rebuild from a run or committed evidence without training or scoring."""
    if run is not None:
        call('scripts/methodology_report.py', '--run', run, '--output', output)
    else:
        with tempfile.TemporaryDirectory(prefix='tradecraft-published-') as directory:
            source = Path(directory)
            evidence = json.loads(EVIDENCE.read_text(encoding='utf-8'))
            (source/'summary.json').write_text(json.dumps(evidence['summary'], indent=2), encoding='utf-8')
            call('scripts/methodology_report.py', '--run', source, '--output', output)
    # The historical renderer links README.md. Supply a local explanation so a
    # standalone report directory does not contain an unresolved guide link.
    guide = Path(output)/'README.md'
    if not guide.exists():
        guide.write_text('Rebuilt primary study report. Protocol and commands:\n'
                         'https://github.com/Haydnlee31/Tradecraft-DSA4266-Grp8/blob/cleanup/focused-study/docs/reproduction.md\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('setup', 'check', 'audit', 'train', 'evaluate',
                                          'compare', 'report', 'reproduce'))
    parser.add_argument('--archive', type=Path, help='Optional pinned data archive for setup/reproduce')
    parser.add_argument('--run', type=Path, help='Existing run for evaluate/compare or report')
    parser.add_argument('--output', type=Path, help='New training or report directory')
    args = parser.parse_args()
    # Resolve caller-relative paths before subprocesses switch to the repo root.
    for key in ('archive', 'run', 'output'):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.action in ('train', 'report', 'reproduce') and args.output is None:
        parser.error('--output is required')
    if args.action in ('evaluate', 'compare') and args.run is None:
        parser.error('--run is required')
    if args.archive and args.action not in ('setup', 'reproduce'):
        parser.error('--archive applies only to setup/reproduce')
    if args.run and args.action not in ('evaluate', 'compare', 'report'):
        parser.error('--run applies only to evaluate/compare/report')
    if args.output and args.action not in ('train', 'report', 'reproduce'):
        parser.error('--output applies only to train/report/reproduce')
    if args.output and args.output.exists():
        parser.error('Preserve existing output; choose a new directory')
    if args.action in ('setup', 'reproduce'):
        command = ['scripts/reproduce_shared_adam.py', 'setup', '--python', sys.executable]
        if args.archive:
            command += ['--archive', args.archive]
        call(*command)
    if args.action in ('check', 'reproduce'):
        call('-m', 'src.eval.methodology_comparison', 'check')
    if args.action == 'audit':
        from scripts.methodology_report import summarize
        analysis = summarize(json.loads(EVIDENCE.read_text(encoding='utf-8'))['summary'])
        print(json.dumps({'status': 'complete', 'training_performed': False,
                          'confusion_endpoints_checked': analysis['confusion_endpoints_checked']}, indent=2))
    if args.action in ('train', 'reproduce'):
        call('-u', '-m', 'src.eval.methodology_comparison', 'train', '--output', args.output)
    run = args.output if args.action == 'reproduce' else args.run
    if args.action in ('evaluate', 'reproduce'):
        call('-m', 'src.eval.methodology_comparison', 'evaluate', '--output', run)
    if args.action in ('compare', 'reproduce'):
        result = compare(json.loads((run/'summary.json').read_text(encoding='utf-8')),
                         json.loads(EVIDENCE.read_text(encoding='utf-8'))['summary'])
        (run/'reproduction-checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(f"PASS: all {result['endpoints']} endpoints reproduce the primary report exactly", flush=True)
    if args.action == 'report':
        render(args.run, args.output)
    elif args.action == 'reproduce':
        render(run, run/'report')


if __name__ == '__main__':
    main()
