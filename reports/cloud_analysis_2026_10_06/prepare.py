"""Audit trusted cloud backups and stage private artifacts for local explanations.

No tar extraction is used: only explicitly named regular files are copied.
Checkpoints use weights_only loading; scaler pickles are hashed here and loaded
only by the existing explanation code for these trusted project backups.
"""
import argparse
import io
import json
from pathlib import Path
import tarfile

import torch

from reports.cloud_study_2026_10_06.audit import digest, require, metric_check
from src.data.label_map import CLASSES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), 'Use a fresh output directory')
    previous_path = Path('reports/cloud_study_2026_10_06/results/summary.json')
    previous = json.loads(previous_path.read_text())
    plan_path = Path('configs/cloud_fedprox_plan.json')
    plan = json.loads(plan_path.read_text())
    archives = {}

    def raw(stage, name):
        if stage not in archives:
            path = a.archives / f'tradecraft-cloud-{stage}-backup.tgz'
            archive = tarfile.open(path)
            require(len(archive.getnames()) == len(set(archive.getnames())), 'Duplicate members')
            archives[stage] = archive
        member = archives[stage].getmember(name)
        require(member.isfile() and member.size < 100_000_000, 'Unexpected artifact')
        return archives[stage].extractfile(member).read()

    def j(stage, base, name):
        return json.loads(raw(stage, base + '/' + name))

    runs = {}
    for name in ('fedavg_bridge', 'fedprox_mu001', 'fedprox_mu01'):
        base = f'outputs/cloud-fedprox/{name}/dirichlet'
        r, e, h = [j('fedprox', base, f) for f in ('result.json', 'environment.json', 'history.json')]
        require(j('fedprox', base, 'status.json') == {'status': 'complete', 'step': 60}, 'Incomplete')
        require(r['test_metrics'] is None and len(h) == 60 and r['history'] == h, 'History/test mismatch')
        require([x['step'] for x in h] == list(range(1, 61)), 'Invalid round sequence')
        for entry in h:
            metric_check(entry['validation_metrics'])
        best = max(h, key=lambda x: x['validation_metrics']['macro_f1'])
        require(best['step'] == r['best_step'] and best['validation_metrics'] == r['validation_metrics'], 'Selection mismatch')
        expected = dict(plan['common'], lane='dirichlet', **next(s['overrides'] for s in plan['stages'] if s['name'] == name))
        require(all(e['settings'][k] == v for k, v in expected.items()), 'Plan settings drift')
        require(e['classes'] == CLASSES, 'Class order changed')
        for filename, sha in [('best.pt', r['checkpoint_sha256']), ('scaler.joblib', e['scaler_sha256']), ('assignments.npz', e['partition_sha256'])]:
            require(digest(raw('fedprox', base+'/'+filename)) == sha, 'Artifact hash mismatch')
        b = torch.load(io.BytesIO(raw('fedprox', base+'/best.pt')), weights_only=True, map_location='cpu')
        last = torch.load(io.BytesIO(raw('fedprox', base+'/last.pt')), weights_only=True, map_location='cpu')
        require(last['step'] == 60 and b['best_step'] == r['best_step'], 'Checkpoint step mismatch')
        for k, v in b['model_state_dict'].items():
            torch.testing.assert_close(v, last['best_state'][k], rtol=0, atol=0)
        runs[name] = dict(result=r, manifest=e, path=base)
    reference = runs['fedavg_bridge']
    for run in runs.values():
        for key in ('packages', 'source_sha256', 'split_sha256', 'scaler_sha256', 'partition_sha256', 'cublas_workspace_config'):
            require(run['manifest'][key] == reference['manifest'][key], f'{key} drift')
    old_base = 'outputs/cloud-reference/dirichlet-seed7'
    old = j('reference', old_base, 'result.json')
    require(old['validation_metrics'] == reference['result']['validation_metrics'], 'Historical metric mismatch')
    for old_step, new_step in zip(old['history'], reference['result']['history']):
        for key in ('step', 'validation_metrics', 'val_loss', 'train_batch_mean_task_loss', 'examples_processed', 'optimizer_steps'):
            require(old_step[key] == new_step[key], f'Historical history mismatch: {key}')
    old_weights = torch.load(io.BytesIO(raw('reference', old_base+'/best.pt')), weights_only=True, map_location='cpu')['model_state_dict']
    new_weights = torch.load(io.BytesIO(raw('fedprox', reference['path']+'/best.pt')), weights_only=True, map_location='cpu')['model_state_dict']
    for key in old_weights:
        torch.testing.assert_close(old_weights[key], new_weights[key], rtol=0, atol=0)
    require(json.loads(raw('fedprox', 'configs/cloud_fedprox_plan.json')) == plan, 'Archived plan drift')
    summary = {'historical_bridge_metrics_history_best_tensors_exact': True, 'test_evaluated': False,
               'previous_audit_sha256': digest(previous_path.read_bytes()), 'runs': {}}
    baseline = reference['result']['validation_metrics']
    for name, run in runs.items():
        r = run['result']; m = r['validation_metrics']; curve = r['history']
        df1 = m['macro_f1'] - baseline['macro_f1']
        dfpr = m['benign_false_alert_rate'] - baseline['benign_false_alert_rate']
        dr = {c: m['per_class_recall'][c] - baseline['per_class_recall'][c] for c in CLASSES}
        summary['runs'][name] = {'metrics': m, 'best_step': r['best_step'], 'seconds': r['step_seconds_sum'],
            'macro_f1_delta': df1, 'false_alert_delta': dfpr, 'recall_deltas': dr,
            'gate_a': df1 >= .01 and dfpr <= .02 and min(dr.values()) >= -.02,
            'gate_b': dfpr <= -.05 and df1 >= -.01 and min(dr[c] for c in ('Web-based', 'Brute Force')) >= -.02,
            'last10_f1_gain': curve[-1]['validation_metrics']['macro_f1'] - curve[-11]['validation_metrics']['macro_f1'],
            'attack_to_benign': {c: m['confusion_matrix'][CLASSES.index(c)][0] / m['per_class'][c]['support'] for c in ('Web-based', 'Brute Force', 'DoS')}}
    # Primary matched models plus two explicitly diagnostic, non-promoted candidates.
    selected = [('reference', 'heavy', 7, 'outputs/cloud-reference/heavy-seed7')]
    for lane in ('light', 'iid', 'dirichlet'):
        base = old_base if lane == 'dirichlet' else f'outputs/cloud-controls/{lane}-layernorm-seed7'
        selected.append(('reference' if lane == 'dirichlet' else 'controls', lane, 7, base))
        for seed in (8, 9):
            selected.append(('confirmation', lane, seed, f'outputs/cloud-confirmation/{lane}-seed{seed}'))
    selected += [('fedprox', name, 7, runs[name]['path']) for name in ('fedprox_mu001', 'fedprox_mu01')]
    a.output.mkdir(parents=True)
    study = {'plan_sha256': digest(plan_path.read_bytes()), 'lanes': {}}
    for stage, lane, seed, base in selected:
        if stage != 'fedprox':
            require(j(stage, base, 'environment.json') == previous['runs'][base]['manifest'], 'Previous manifest mismatch')
        folder = a.output / 'models' / f'{lane}-seed{seed}'
        folder.mkdir(parents=True)
        for name in ('best.pt', 'scaler.joblib', 'environment.json', 'result.json', 'status.json'):
            (folder/name).write_bytes(raw(stage, base+'/'+name))
        study['lanes'].setdefault(lane, {'runs': {}})['runs'][str(seed)] = {'path': str(folder), 'result_sha256': digest((folder/'result.json').read_bytes())}
    summary['archives'] = {}
    for stage, archive in archives.items():
        path = a.archives / f'tradecraft-cloud-{stage}-backup.tgz'
        summary['archives'][stage] = digest(path.read_bytes())
        if stage in previous['archives']:
            require(summary['archives'][stage] == previous['archives'][stage]['sha256'], 'Previous archive changed')
        archive.close()
    for name, obj in [('audit.json', summary), ('study.json', study)]:
        (a.output/name).write_text(json.dumps(obj, indent=2)+'\n')
    print(json.dumps(summary['runs'], indent=2))
    print('PASS archive audit and exact historical bridge; staged 12 models')


if __name__ == '__main__':
    main()
