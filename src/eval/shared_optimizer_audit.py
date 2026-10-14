"""Replay saved development endpoints without training or opening other splits."""
import argparse
import json
from pathlib import Path

import torch

from src.data.official_inventory import sha256
from src.data.shared_optimizer_data import PLAN, DevelopmentData
from src.eval.federated_class_safeguard import state_digest
from src.eval.shared_optimizer_performance import initial_model, evaluate_development, sources, work_plan, screen
from src.federated.shared_adam import require
from src.models.research import atomic_json


def audit(data_root, run_root, output):
    output, run_root = Path(output), Path(run_root)
    require(not output.exists(), 'Preserve existing audit')
    summary = json.loads((run_root/'summary.json').read_text())
    plan = json.loads(PLAN.read_text())
    require(summary['plan'] == plan and summary['contract']['plan_sha256'] == sha256(PLAN)
            and summary['contract']['sources'] == sources(), 'Source or plan identity differs')
    require(summary['contract']['data_manifest_sha256'] == sha256(Path(data_root)/'manifest.json'),
            'Data identity differs')
    torch.set_num_threads(plan['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = DevelopmentData(data_root,plan)
    expected = {(a,lr) for a in plan['arms'] for lr in plan['learning_rates']}
    require(len(summary['results']) == len(expected) and
            {(r['arm'],r['learning_rate']) for r in summary['results']} == expected, 'Missing/duplicate candidate')
    checks = []
    for result in summary['results']:
        arm, lr = result['arm'], result['learning_rate']
        checkpoint = run_root/f'{arm}-lr{lr}'/'last.pt'
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        require(saved['contract'] == result['contract'] == summary['contract'], 'Checkpoint contract differs')
        require(saved['state']['unit'] == result['units'] == work_plan(plan,arm)[0], 'Incomplete endpoint')
        require(saved['state']['work'] == result['work'] and
                result['work']['backward_examples'] == plan['backward_examples_per_arm'], 'Work counters differ')
        require(saved['state']['sample_chain_sha256'] == result['sample_chain_sha256'], 'Sampling receipt differs')
        model = initial_model(plan)
        require(state_digest(model) == result['initial_model_sha256'], 'Initialization differs')
        model.load_state_dict(saved['state']['model'])
        require(state_digest(model) == result['final_model_sha256'], 'Final weights differ')
        metrics = evaluate_development(model,data)
        require(metrics == result['development_metrics'], 'Development replay differs')
        checks.append({'arm':arm,'learning_rate':lr,'checkpoint_sha256':sha256(checkpoint),
                       'development_replay_exact':True,'backward_examples':result['work']['backward_examples']})
        print(f'PASS saved endpoint {arm} lr={lr}',flush=True)
    selected = {a:max((r for r in summary['results'] if r['arm']==a),
                     key=lambda r:(r['development_metrics']['macro_f1'],-r['learning_rate'])) for a in plan['arms']}
    require(summary['screen']==screen(selected,plan), 'Screen decision differs')
    require(summary['selected_learning_rates']=={a:r['learning_rate'] for a,r in selected.items()}, 'Selection differs')
    require(summary['total_backward_examples']==plan['total_backward_examples']==
            sum(r['work']['backward_examples'] for r in summary['results']), 'Total work differs')
    result = {'status':'complete','summary_sha256':sha256(run_root/'summary.json'),
              'auditor_sha256':sha256(__file__),'checks':checks,
              'development_evaluation_examples':len(checks)*len(data.arrays['development'][1]),
              'additional_backward_examples':0,'confirmation_scored':False,
              'original_validation_opened':False,'test_opened':False,'test_evaluated':False,
              'scope':'Same-runtime saved-checkpoint inference replay, not an independent training seed'}
    atomic_json(result,output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',dest='data_root',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(audit(**vars(args)),indent=2))


if __name__ == '__main__':
    main()
