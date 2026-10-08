"""Compile paired-seed scaling evidence from explicit archive members.

Read JSON only and verify reported work and confusion-derived metrics. The rare
diagnostic separately reads weights with PyTorch's restricted weights-only loader.
No archive extraction, test data access, or training is performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import tarfile

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.models.official_streaming import confusion_metrics
from src.models.research import atomic_json


def describe(values):
    return {'values': values, 'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values), 'min': min(values), 'max': max(values)}


def review(baseline, reference, confirmation, plan, output, source_ref=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads(Path(plan).read_text())
    if source_ref is not None and re.fullmatch(r'[0-9a-f]{40}', source_ref) is None:
        raise ValueError('source_ref must be a full immutable Git commit hash')
    source_cache = {}
    archives = dict(baseline=baseline, reference=reference, confirmation=confirmation)
    receipt = {'status': 'complete', 'test_opened': False, 'model_trained': False,
               'script_sha256': sha256(__file__), 'plan_sha256': sha256(plan),
               'verified_source_ref': source_ref or 'working_tree',
               'archive_sha256': {k: sha256(v) for k, v in archives.items()},
               'seed_order': [7, 17, 27], 'runs': {}, 'summaries': {}, 'paired_differences': {}}
    data = {}
    for seed in receipt['seed_order']:
        for cohort in protocol['cohorts']:
            for lane in protocol['lanes']:
                if seed == 7:
                    arc = baseline if cohort == '500k' else reference
                    root = protocol['reuse_seed7'][cohort+'_root']+'/'+lane
                else:
                    arc = confirmation
                    root = protocol['output_template'].format(cohort=cohort, lane=lane, seed=seed)
                with tarfile.open(arc) as t:
                    def read(name):
                        return json.load(t.extractfile(root+'/'+name))
                    result, env, status = read('result.json'), read('environment.json'), read('status.json')
                spec = protocol['cohorts'][cohort]
                if (status != {'status': 'complete', 'step': 20} or result['test_metrics'] is not None
                        or env['test_evaluated'] or env['packed_manifest_sha256'] != spec['manifest_sha256']
                        or result['examples_processed'] != spec['final_examples']
                        or result['optimizer_steps'] != spec['final_updates'][lane]):
                    raise ValueError('Run status, budget or data provenance mismatch')
                expected = dict(protocol['settings'], lane=lane, seed=seed, epochs=20)
                if env['settings'] != expected:
                    raise ValueError('Training setting mismatch')
                # Additive analysis files may be absent from an old manifest; all
                # source files it actually records must still match their content.
                for name, digest in env['source_sha256'].items():
                    relative = Path(name)
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('Invalid recorded source path')
                    if name not in source_cache:
                        if source_ref is None:
                            source_cache[name] = sha256(Path(__file__).parents[1]/name)
                        else:
                            # Old run evidence remains reproducible after the
                            # runner evolves: validate against its pinned source,
                            # never silently waive a training-source mismatch.
                            source = subprocess.run(['git', 'show', f'{source_ref}:src/{name}'],
                                cwd=Path(__file__).parents[2], check=True, capture_output=True).stdout
                            source_cache[name] = hashlib.sha256(source).hexdigest()
                    if source_cache[name] != digest:
                        raise ValueError(f'Recorded source changed: {name}')
                history = result['history']
                if len(history) != 20 or [h['step'] for h in history] != list(range(1, 21)):
                    raise ValueError('Incomplete training history')
                if sum(h['optimizer_steps'] for h in history) != result['optimizer_steps']:
                    raise ValueError('Work history differs from total')
                for h in history:
                    if h['examples_processed'] != spec['rows']:
                        raise ValueError('Wrong round exposure')
                    if lane == 'iid' and (len(h['client_work']) != 20 or
                        any(c['examples_processed'] != c['assigned_rows'] for c in h['client_work'])):
                        raise ValueError('Incomplete participation')
                    metrics = h['validation_metrics']
                    recomputed = confusion_metrics(np.array(metrics['confusion_matrix']))
                    if recomputed != metrics:
                        raise ValueError('Metrics disagree with confusion matrix')
                if result['final_validation_metrics'] != history[-1]['validation_metrics']:
                    raise ValueError('Final endpoint differs from history')
                data[seed, cohort, lane] = result
                receipt['runs'][f'{cohort}/{lane}/seed{seed}'] = {
                    'final': result['final_validation_metrics'],
                    'step5': history[4]['validation_metrics'] if cohort == '2m' else None,
                    'best_macro_f1': result['validation_metrics']['macro_f1'], 'best_step': result['best_step'],
                    'examples_processed': result['examples_processed'], 'optimizer_steps': result['optimizer_steps'],
                    'training_validation_seconds': sum(h['elapsed_seconds'] for h in history)}
    for lane in protocol['lanes']:
        for cohort, step in [('500k', 20), ('2m', 5), ('2m', 20)]:
            ms = [data[s, cohort, lane]['history'][step-1]['validation_metrics'] for s in receipt['seed_order']]
            stats = {k: describe([m[k] for m in ms]) for k in ('macro_f1', 'benign_false_alert_rate')}
            stats['per_class'] = {c: {k: describe([m['per_class'][c][k] for m in ms])
                                        for k in ('recall', 'precision', 'f1')} for c in CLASSES}
            receipt['summaries'][f'{lane}/{cohort}/step{step}'] = stats
        for step in (5, 20):
            pairs = [(data[s, '2m', lane]['history'][step-1]['validation_metrics'],
                      data[s, '500k', lane]['final_validation_metrics']) for s in receipt['seed_order']]
            stats = {k: describe([a[k]-b[k] for a, b in pairs]) for k in ('macro_f1', 'benign_false_alert_rate')}
            stats['per_class'] = {c: {k: describe([a['per_class'][c][k]-b['per_class'][c][k] for a, b in pairs])
                                        for k in ('recall', 'precision', 'f1')} for c in CLASSES}
            receipt['paired_differences'][f'{lane}/2m-step{step}-minus-500k-step20'] = stats
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt, output)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'reference', 'confirmation', 'plan', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source-ref', help='Pinned full commit for historical source verification')
    review(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
