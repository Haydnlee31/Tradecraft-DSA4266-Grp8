import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

workspace = Path(__file__).resolve().parents[3]
checkout = workspace/'outputs/Tradecraft-reproduce'
sys.path.insert(0, str(checkout))
from extensions.fedntd.data import DevelopmentData, sha256
from extensions.fedntd.run import initial_model
from src.data.label_map import CLASSES
from src.eval.federated_class_safeguard import state_digest

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
run_dir = checkout/'extensions/fedntd/runs/seed7-screen-v1'
summary = json.loads((run_dir/'summary.json').read_text())
contract = summary['contract']
plan = contract['plan']
for name, expected in contract['sources'].items():
    assert sha256(checkout/name) == expected, name
data = DevelopmentData(checkout/'outputs/shared-optimizer-development-data-v1', plan)
assert data.file_hashes == contract['files']
assert sha256(data.root/'manifest.json') == contract['data_manifest_sha256']
assert sha256(checkout/'extensions/fedntd/plan.json') == contract['plan_sha256']
historical = json.loads((checkout/'reports/federated_shared_optimizer/performance-results.json').read_text())
fedavg = next(r for r in historical['summary']['results']
              if r['arm'] == 'fedavg_adam' and r['learning_rate'] == .003)
results = []
for r in summary['results']:
    checkpoint = run_dir/f'beta-{r["beta"]:g}'/'last.pt'
    assert sha256(checkpoint) == r['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['contract'] == contract and saved['round'] == 108
    assert saved['work'] == r['work'] and saved['sample_chain_sha256'] == r['sample_chain_sha256']
    model = initial_model(plan)
    model.load_state_dict(saved['model'])
    model.eval()
    x, y = data.arrays['development']
    predictions = []
    with torch.no_grad():
        for start in range(0, len(y), 8192):
            predictions.append(model(torch.from_numpy(np.array(x[start:start+8192], copy=True))).argmax(1).numpy())
    predicted = np.concatenate(predictions)
    cm = confusion_matrix(y, predicted, labels=np.arange(len(CLASSES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=np.arange(len(CLASSES)), zero_division=0)
    m = r['development_metrics']
    assert cm.tolist() == m['confusion_matrix']
    assert abs(float(f1.mean())-m['macro_f1']) < 1e-14
    for i, c in enumerate(CLASSES):
        for key, value in zip(('precision','recall','f1','support'),
                              (precision[i],recall[i],f1[i],support[i])):
            assert abs(float(value)-m['per_class'][c][key]) < 1e-14
    benign = CLASSES.index('Benign')
    assert abs(float((predicted[y == benign] != benign).mean())-m['benign_false_alert_rate']) < 1e-14
    if r['beta'] == 0:
        assert {k:v for k,v in m.items() if k != 'decision_rule'} == fedavg['development_metrics']
        assert state_digest(model) == fedavg['final_model_sha256']
        assert r['sample_chain_sha256'] == fedavg['sample_chain_sha256']
    assert r['work']['backward_examples'] == 21600000
    assert r['work']['teacher_forward_examples'] == (21600000 if r['beta'] else 0)
    assert r['work']['estimated_payload_bytes'] == 88058880
    results.append({k:r[k] for k in ('beta','temperature','rounds','work','training_seconds_sim_host',
                                    'sample_chain_sha256','checkpoint_sha256','development_metrics',
                                    'num_parameters','parameter_bytes')})
assert len({r['sample_chain_sha256'] for r in results}) == 1
reproduction = json.loads((checkout/'outputs/local-reproduction-v1/summary.json').read_text())
assert reproduction['status'] == 'complete' and reproduction['new_training_runs'] == 3
receipt = {
    'status':'verified',
    'scope':'Seed-7 FedNTD exploratory development screen; argmax, no alarm tuning',
    'data':{'fit_rows':len(data.arrays['fit'][1]),'development_rows':len(data.arrays['development'][1]),
            'clients':len(data.parts),'features':len(data.manifest['features'])},
    'runtime':contract['runtime'],
    'plan':plan,
    'input_manifest_sha256':contract['data_manifest_sha256'],
    'run_summary_sha256':sha256(run_dir/'summary.json'),
    'audit_script_sha256':sha256(__file__),
    'checks':{'source_and_data_hashes_match':True,'all_three_checkpoints_replayed':True,
              'independent_sklearn_metrics_match':True,'beta_zero_original_weights_metrics_samples_exact':True,
              'all_arms_identical_sample_chain':True,'work_and_payload_counts_match':True},
    'results':results,
    'comparisons_to_paired_fedavg':summary['comparisons_to_paired_fedavg'],
    'baseline_reproduction':{'mean_macro_f1':reproduction['mean_macro_f1'],
        'summary_sha256':sha256(checkout/'outputs/local-reproduction-v1/summary.json'),
        'seeds':[{'seed':r['seed'],'macro_f1':r['primary']['macro_f1'],
                  'full_training_state_exact':r['full_training_state_exact'],
                  'confirmation_metrics_exact':r['confirmation_metrics_exact']} for r in reproduction['results']],
        'scope':'Previously published Shared Adam confirmation results with 5% target calibration; separate from FedNTD comparison'},
    'fedntd_test_opened':False,'fedntd_confirmation_scored':False,
    'audit_new_training_examples':0,'audit_development_inference_examples':3*len(data.arrays['development'][1]),
    'limitations':['One training seed and one fixed client partition.',
                  'Development cohort has informed earlier research.',
                  'No independent holdout or calibrated-FAR FedNTD comparison.',
                  'Simulation-host time; modeled communication bytes, no edge/network measurements.']}
destination = workspace/'reports/extensions/fedntd'
destination.mkdir(parents=True, exist_ok=True)
(destination/'real-data-results.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
print(json.dumps({'status':receipt['status'],'checks':receipt['checks'],
                 'arms':[{'beta':r['beta'],'macro_f1':r['development_metrics']['macro_f1'],
                          'far':r['development_metrics']['benign_false_alert_rate'],
                          'seconds':r['training_seconds_sim_host']} for r in results]},indent=2))
