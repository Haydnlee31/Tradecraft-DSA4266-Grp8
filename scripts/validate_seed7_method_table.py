"""Freshly reproduce the four traceable rows of the requested development table.

This does not implement MimeLite or turn unverified numbers into reference data.
It uses the existing frozen training kernels and never scores confirmation/test.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.eval import methodology_comparison as study
from src.models.research import atomic_json

EXPECTED = {
    'central_adam': (.6456, 5.33, 0, 0),
    'shared_adam': (.6246, 5.43, 2160, 1761177600),
    'mime': (.5153, 5.24, 24, 58705920),
    'fedavg_adam': (.5137, 5.70, 108, 88058880),
}


def run(output):
    old = study.old
    output = old.require_fresh_output(output)
    plan, base, data, alarm, x, y, calibration = study.preflight(old.DATA)
    base = {**base, 'seed': 7}
    identity = study.contract()
    reference_path = ROOT/'reports/methodology_comparison/results.json'
    reference = json.loads(reference_path.read_text())['summary']['results']
    criterion = old.build_official_criterion('sqrt_weighted_ce', data.manifest['counts']['fit'], 'fixed_train_mean')
    summary = {
        'status': 'running', 'purpose': 'Verify supplied seed7 development-audit table',
        'runtime': study.portable.runtime_record(), 'source_contract': identity,
        'validation_script_sha256': old.sha256(__file__),
        'reference_sha256': old.sha256(reference_path),
        'data_manifest_sha256': alarm['data_manifest_sha256'],
        'seed': 7, 'fit_rows': len(data.arrays['fit'][1]),
        'development_rows': len(y), 'calibration_rows': int(calibration.sum()),
        'audit_rows': int((~calibration).sum()),
        'confirmation_evaluated': False, 'original_validation_evaluated': False,
        'test_evaluated': False, 'results': [],
        'unverified_rows': [
            {'method': 'MimeLite, 5 local steps', 'macro_f1_claim': .5843, 'false_alert_percent_claim': 5.61},
            {'method': 'MimeLite, 20 local steps', 'macro_f1_claim': .5477, 'false_alert_percent_claim': 5.64},
        ],
        'unverified_reason': 'No executable MimeLite pipeline or matching saved result found in searched repository branches; not replaced with a guessed implementation.',
    }
    atomic_json(summary, output/'summary.json')
    for arm, claim in EXPECTED.items():
        print('START', arm, flush=True)
        path = output/arm
        state, seconds, recovery_work = study.train_one(
            data, base, arm, plan['arms'][arm], path, identity, criterion)
        logits = old.rep.infer(state['model'], x, alarm['inference_batch_size'])
        np.save(path/'development_logits.npy', logits, allow_pickle=False)
        scores = old.alarms.evaluate_scores(logits, y, calibration, [100, 500, 1000, 2000])
        old.replication.independent_points(logits, y, calibration, scores['operating_points'])
        primary = next(p for p in scores['operating_points'] if p['calibration']['target_basis_points'] == 500)['audit_metrics']
        ref = next(r for r in reference if r['arm'] == arm and r['seed'] == 7)
        model_hash = old.state_digest(state['model'])
        logits_hash = old.sha256(path/'development_logits.npy')
        checks = {
            'model_exact': model_hash == ref['model_sha256'],
            'logits_exact': logits_hash == ref['development_logits_sha256'],
            'all_development_metrics_and_thresholds_exact': scores == ref['development'],
            'sample_stream_exact': state['sample_chain_sha256'] == ref['sample_chain_sha256'],
            'work_exact': state['work'] == ref['work'],
            'supplied_macro_f1_matches': round(primary['macro_f1'], 4) == claim[0],
            'supplied_false_alerts_match': round(100*primary['benign_false_alert_rate'], 2) == claim[1],
            'supplied_rounds_match': state['work']['synchronization_rounds'] == claim[2],
            'supplied_payload_matches': state['work']['estimated_payload_bytes'] == claim[3],
            'serialized_recovery_exact': True,
        }
        result = {'arm': arm, 'checks': checks, 'work': state['work'],
                  'model_sha256': model_hash, 'development_logits_sha256': logits_hash,
                  'sample_chain_sha256': state['sample_chain_sha256'],
                  'primary_metrics': primary, 'development': scores,
                  'training_seconds': seconds, 'additional_recovery_backward_examples': recovery_work}
        atomic_json(result, path/'result.json')
        summary['results'].append(result)
        atomic_json(summary, output/'summary.json')
        old.require(all(checks.values()), f'Mismatch for {arm}: {checks}')
        print(f"PASS {arm}: F1={primary['macro_f1']:.9f}; FAR={100*primary['benign_false_alert_rate']:.6f}%", flush=True)
    old.require(study.contract() == identity, 'Frozen source changed during validation')
    summary['status'] = 'partial_validation_complete'
    summary['verified_rows'] = len(summary['results'])
    summary['total_training_backward_examples'] = sum(r['work']['backward_examples'] for r in summary['results'])
    atomic_json(summary, output/'summary.json')
    print('PASS: four existing rows reproduced; two MimeLite rows remain unverified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New output directory; existing outputs are preserved')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run(args.output)
