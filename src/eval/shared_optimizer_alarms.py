"""Development-only alarm-target diagnostic for already selected frozen models.

This is a deterministic benign-logit offset, not another learned classifier.
Calibration uses benign labels from one grouped half; evaluation uses the other.
Both halves were previously exposed during model selection: no fresh-test claim.
Historical modules are imported but never edited or used to open held-out data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.shared_optimizer_data import ROOT, digest_array
from src.eval.federated_class_safeguard import state_digest
from src.eval.shared_optimizer_performance import initial_model, sources as training_sources
from src.federated.shared_adam import require
from src.models.official_streaming import confusion_metrics
from src.models.research import atomic_json

PLAN = ROOT/'reports/federated_shared_optimizer/alarm-plan.json'
EVIDENCE = ROOT/'reports/federated_shared_optimizer/performance-results.json'
BENIGN = CLASSES.index('Benign')
ATTACKS = np.array([i for i in range(len(CLASSES)) if i != BENIGN])


def read_contract():
    plan = json.loads(PLAN.read_text())
    require(sha256(EVIDENCE) == plan['performance_evidence_sha256'], 'Performance evidence changed')
    evidence = json.loads(EVIDENCE.read_text())
    s = evidence['summary']
    require(s['selected_learning_rates'] == plan['selected_learning_rates'], 'Selected models changed')
    require(s['contract']['sources'] == training_sources(), 'Training source identity differs')
    require(s['contract']['data_manifest_sha256'] == plan['data_manifest_sha256'], 'Data identity differs')
    return plan, evidence


def load_development(root, plan, evidence):
    """Only three permitted arrays: no generic split selector or fit-array load."""
    root = Path(root)
    require(sha256(root/'manifest.json') == plan['data_manifest_sha256'], 'Data manifest differs')
    manifest = json.loads((root/'manifest.json').read_text())
    require(manifest == evidence['summary']['data'], 'Prepared data receipt differs')
    arrays = []
    for name in ('development_x.npy', 'development_y.npy', 'development_row_ids.npy'):
        require(sha256(root/name) == manifest['files'][name], f'Development input differs: {name}')
        arrays.append(np.load(root/name, mmap_mode='r', allow_pickle=False))
    x, y, row_ids = arrays
    n = plan['development_rows']
    require(x.shape == (n,len(manifest['features'])) and x.dtype == np.float32
            and y.shape == row_ids.shape == (n,) and y.dtype == row_ids.dtype == np.int64,
            'Development shape/dtype differs')
    require(np.bincount(y,minlength=len(CLASSES)).tolist() ==
            [manifest['counts']['development'][c] for c in CLASSES], 'Development labels differ')
    require(len(np.unique(row_ids)) == n, 'Repeated source row IDs')
    return x, y, row_ids


def grouped_split(x, domain, batch_size=8192):
    hashes = np.empty(len(x), dtype='V32')
    for start in range(0,len(x),batch_size):
        hashes[start:start+batch_size] = digest_array(x[start:start+batch_size])
    # Canonical signed-zero fingerprints preserve exact model-input groups.
    calibration = np.array([hashlib.sha256(domain.encode()+h.tobytes()).digest()[0] % 2 == 0
                            for h in hashes],dtype=bool)
    require(calibration.any() and (~calibration).any(), 'Empty calibration/audit partition')
    a, b = np.unique(hashes[calibration]), np.unique(hashes[~calibration])
    require(len(np.intersect1d(a,b)) == 0, 'Feature group crosses the boundary')
    return calibration, {'calibration_unique_input_groups':len(a), 'audit_unique_input_groups':len(b),
                         'cross_split_input_groups':0,
                         'input_digest_sequence_sha256':hashlib.sha256(hashes.tobytes()).hexdigest()}


def margins(logits):
    require(logits.ndim == 2 and logits.shape[1] == len(CLASSES) and len(logits) > 0
            and bool(np.isfinite(logits).all()), 'Invalid logits')
    attack_class = ATTACKS[logits[:,ATTACKS].argmax(axis=1)]
    attack_logit = logits[np.arange(len(logits)),attack_class].astype(np.float64)
    score = attack_logit-logits[:,BENIGN].astype(np.float64)
    require(bool(np.isfinite(score).all()), 'Nonfinite margins')
    return score, attack_class


def calibrate(benign_scores, target_basis_points):
    """Exact finite-sample count cap; NOT a population-FAR guarantee."""
    scores = np.asarray(benign_scores,dtype=np.float64)
    require(scores.ndim == 1 and len(scores) > 0 and bool(np.isfinite(scores).all()), 'Invalid benign scores')
    require(type(target_basis_points) is int and 0 <= target_basis_points < 10000, 'Invalid FAR target')
    allowed = len(scores)*target_basis_points//10000
    rank = len(scores)-allowed-1
    threshold = float(np.partition(scores,rank)[rank])
    observed = int((scores > threshold).sum())
    require(observed <= allowed, 'Calibration cap exceeded')
    return {'target_basis_points':target_basis_points, 'threshold':threshold,
            'calibration_benign_rows':len(scores), 'allowed_false_alerts':allowed,
            'observed_false_alerts':observed, 'calibration_false_alert_rate':observed/len(scores),
            'boundary_ties':int((scores == threshold).sum()), 'ascending_order_statistic_index':rank}


def decisions(scores, attack_class, threshold):
    require(np.isfinite(threshold), 'Nonfinite threshold')
    require(scores.shape == attack_class.shape and bool(np.isfinite(scores).all())
            and bool(np.isin(attack_class,ATTACKS).all()), 'Invalid score/class vectors')
    return np.where(scores > threshold,attack_class,BENIGN)


def metrics(y, predicted):
    require(y.shape == predicted.shape and y.ndim == 1 and len(y) > 0, 'Invalid targets')
    require(bool(((y >= 0) & (y < len(CLASSES))).all())
            and bool(((predicted >= 0) & (predicted < len(CLASSES))).all()), 'Invalid class ID')
    cm = np.bincount(y*len(CLASSES)+predicted,minlength=len(CLASSES)**2).reshape(len(CLASSES),len(CLASSES))
    result = confusion_metrics(cm)
    require(bool((cm.sum(axis=1) > 0).all()), 'Evaluation omits a class')
    attack_rows = int(cm[ATTACKS].sum())
    result['attack_detection_recall'] = float(1-cm[ATTACKS,BENIGN].sum()/attack_rows)
    result['macro_attack_class_recall'] = float(np.mean([result['per_class'][CLASSES[i]]['recall'] for i in ATTACKS]))
    result['per_attack_detection_recall'] = {CLASSES[i]:float(1-cm[i,BENIGN]/cm[i].sum()) for i in ATTACKS}
    return result


def evaluate_scores(logits, y, calibration, targets):
    """Threshold fitting never sees audit labels; no search over attack outcomes."""
    require(y.shape == calibration.shape == (len(logits),) and calibration.dtype == bool, 'Invalid split mask')
    score, attack_class = margins(logits)
    zero_predictions = decisions(score,attack_class,0.)
    require(np.array_equal(zero_predictions,logits.argmax(axis=1)), 'Zero-margin argmax bridge differs')
    # Freeze every threshold BEFORE producing any audit outcomes.
    frozen = [calibrate(score[calibration & (y == BENIGN)],target) for target in targets]
    audit_rows = ~calibration
    outcomes = [{'calibration':c, 'audit_metrics':metrics(y[audit_rows],
                    decisions(score[audit_rows],attack_class[audit_rows],c['threshold']))} for c in frozen]
    return {'zero_margin_argmax_exact':True,
            'full_development_default_metrics':confusion_metrics(np.bincount(
                 y*len(CLASSES)+zero_predictions,minlength=len(CLASSES)**2).reshape(len(CLASSES),len(CLASSES))),
            'audit_default_metrics':metrics(y[audit_rows],zero_predictions[audit_rows]),
            'operating_points':outcomes}


def comparisons(results, plan):
    by_arm = {r['arm']:r for r in results}
    output = []
    for target in plan['false_alert_targets_basis_points']:
        points = {a:next(p['audit_metrics'] for p in r['operating_points']
                        if p['calibration']['target_basis_points'] == target) for a,r in by_arm.items()}
        for candidate in ('shared_adam','mime'):
            for reference in ('central_adam','fedavg_adam'):
                c, r = points[candidate],points[reference]
                far_gap = c['benign_false_alert_rate']-r['benign_false_alert_rate']
                output.append({'target_basis_points':target,'candidate':candidate,'reference':reference,
                   'macro_f1_difference':c['macro_f1']-r['macro_f1'],
                   'audit_far_difference':far_gap,
                   'audit_far_within_comparability_tolerance':abs(far_gap) <= plan['actual_far_comparability_basis_points']/10000,
                   'macro_attack_class_recall_difference':c['macro_attack_class_recall']-r['macro_attack_class_recall'],
                   'per_class_recall_difference':{name:c['per_class'][name]['recall']-r['per_class'][name]['recall'] for name in CLASSES}})
    return output


def source_hashes():
    names = ['src/eval/shared_optimizer_alarms.py','src/models/official_streaming.py',
             'src/data/shared_optimizer_data.py','src/data/official_quality.py','src/data/label_map.py',
             'src/eval/shared_optimizer_performance.py','src/models/architectures.py']
    return {name:sha256(ROOT/name) for name in names}


def run(data_root, run_root, output):
    plan, evidence = read_contract()
    plan_digest = sha256(PLAN)
    output, run_root = Path(output),Path(run_root)
    require(not output.exists(), 'Preserve prior diagnostic output; use a new destination')
    torch.set_num_threads(plan['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    x,y,row_ids = load_development(data_root,plan,evidence)
    calibration,split_receipt = grouped_split(x,plan['split_domain'])
    for mask in (calibration,~calibration):
        require(bool((np.bincount(y[mask],minlength=len(CLASSES)) > 0).all()), 'Split omits a class')
    output.mkdir(parents=True)
    atomic_json({'status':'running','training_performed':False,'test_opened':False},output/'status.json')
    np.save(output/'calibration_mask.npy',calibration,allow_pickle=False)
    split_receipt.update(calibration_mask_sha256=sha256(output/'calibration_mask.npy'),
        counts={name:dict(zip(CLASSES,np.bincount(y[mask],minlength=len(CLASSES)).tolist()))
                for name,mask in (('calibration',calibration),('audit',~calibration))},
        source_row_ids_sha256={name:hashlib.sha256(row_ids[mask].astype('<i8').tobytes()).hexdigest()
                for name,mask in (('calibration',calibration),('audit',~calibration))})
    source = source_hashes()
    results = []
    for arm,lr in plan['selected_learning_rates'].items():
        previous = next(r for r in evidence['summary']['results'] if r['arm'] == arm and r['learning_rate'] == lr)
        checkpoint_hash = next(r['checkpoint_sha256'] for r in evidence['audit']['checks']
                                if r['arm'] == arm and r['learning_rate'] == lr)
        checkpoint = run_root/f'{arm}-lr{lr}'/'last.pt'
        require(sha256(checkpoint) == checkpoint_hash, 'Selected checkpoint differs')
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        require(saved['contract'] == previous['contract'], 'Checkpoint contract differs')
        model = initial_model(evidence['summary']['plan'])
        model.load_state_dict(saved['state']['model'])
        require(state_digest(model) == previous['final_model_sha256'], 'Model state differs')
        model.eval()
        logits_path = output/f'{arm}-logits.npy'
        logits = np.lib.format.open_memmap(logits_path,mode='w+',dtype='<f4',shape=(len(y),len(CLASSES)))
        with torch.no_grad():
            for start in range(0,len(y),plan['inference_batch_size']):
                batch = torch.from_numpy(np.array(x[start:start+plan['inference_batch_size']],copy=True))
                logits[start:start+len(batch)] = model(batch).numpy()
        logits.flush()
        evaluated = evaluate_scores(logits,y,calibration,plan['false_alert_targets_basis_points'])
        require(evaluated['full_development_default_metrics'] == previous['development_metrics'],
                'Historical default endpoint no longer matches')
        result = {'arm':arm,'learning_rate':lr,'checkpoint_sha256':checkpoint_hash,
                  'final_model_sha256':previous['final_model_sha256'],'logits_sha256':sha256(logits_path),
                  'historical_default_endpoint_exact':True,**evaluated}
        atomic_json(result,output/f'{arm}-result.json')
        results.append(result)
        primary = next(p for p in evaluated['operating_points']
                       if p['calibration']['target_basis_points'] == plan['primary_target_basis_points'])['audit_metrics']
        print(f'PASS {arm}: primary macro-F1={primary["macro_f1"]:.6f}, actual FAR={primary["benign_false_alert_rate"]:.4%}',flush=True)
    require(source_hashes() == source and sha256(PLAN) == plan_digest,'Source/plan changed during diagnostic')
    summary = {'status':'complete','plan':plan,'plan_sha256':plan_digest,'sources':source,'split':split_receipt,
       'results':results,'comparisons':comparisons(results,plan),'forward_examples':len(y)*len(results),
       'backward_examples':0,'new_model_fits':0,'elapsed_seconds':time.perf_counter()-started,
       'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__},
       'training_performed':False,'previous_screen_overridden':False,'confirmation_opened':False,
       'confirmation_scored':False,'original_validation_opened':False,'test_opened':False,'test_evaluated':False}
    atomic_json(summary,output/'summary.json')
    atomic_json({'status':'complete','test_opened':False,'training_performed':False},output/'status.json')
    return summary


def replay(data_root, output, receipt):
    """Independent implementation using stored logits and frozen thresholds.

    No new model inference, fitting, or calibration is performed. Direct logit
    comparisons recheck the compact score path, including its decision ties.
    """
    output,receipt = Path(output),Path(receipt)
    require(not receipt.exists(), 'Preserve replay receipt')
    plan,evidence = read_contract()
    summary = json.loads((output/'summary.json').read_text())
    require(summary['plan'] == plan and summary['plan_sha256'] == sha256(PLAN)
            and summary['sources'] == source_hashes(), 'Diagnostic contract differs')
    require(len(summary['results']) == len(plan['selected_learning_rates']) and
            {r['arm']:r['learning_rate'] for r in summary['results']} == plan['selected_learning_rates'],
            'Missing/duplicate selected model')
    x,y,_ = load_development(data_root,plan,evidence)
    calibration,group_receipt = grouped_split(x,plan['split_domain'])
    require(sha256(output/'calibration_mask.npy') == summary['split']['calibration_mask_sha256'], 'Split receipt changed')
    require(np.array_equal(calibration,np.load(output/'calibration_mask.npy',allow_pickle=False)), 'Split replay differs')
    require(all(summary['split'][k] == v for k,v in group_receipt.items()), 'Grouped split differs')
    checks = []
    for r in summary['results']:
        require(len(r['operating_points']) == len(plan['false_alert_targets_basis_points']) and
                [p['calibration']['target_basis_points'] for p in r['operating_points']] ==
                plan['false_alert_targets_basis_points'], 'Missing/duplicate operating point')
        path = output/f'{r["arm"]}-logits.npy'
        require(sha256(path) == r['logits_sha256'],'Saved logits differ')
        logits = np.load(path,mmap_mode='r',allow_pickle=False)
        winners = ATTACKS[logits[:,ATTACKS].argmax(axis=1)]
        top = logits[np.arange(len(y)),winners].astype(np.float64)
        benign = logits[:,BENIGN].astype(np.float64)
        for point in r['operating_points']:
            c = point['calibration']
            benign_margins = (top-benign)[calibration & (y == BENIGN)]
            # Independent sorted-order and direct-logit reference, no np.partition.
            sorted_values = np.sort(benign_margins)
            allowed = len(sorted_values)*c['target_basis_points']//10000
            require(float(sorted_values[len(sorted_values)-allowed-1]) == c['threshold'],'Threshold order statistic differs')
            alert = top > benign+c['threshold']
            # Addition/subtraction can differ at a floating point boundary. Here
            # original logits are FP32 and the reference operates in FP64; require
            # agreement rather than hiding such a difference with a tolerance.
            predicted = np.where(alert,winners,BENIGN)
            require(metrics(y[~calibration],predicted[~calibration]) == point['audit_metrics'],'Audit prediction replay differs')
            observed = int((alert & calibration & (y == BENIGN)).sum())
            require(observed == c['observed_false_alerts'] <= allowed,'Calibration replay differs')
        require(metrics(y[~calibration],logits[~calibration].argmax(axis=1)) == r['audit_default_metrics'], 'Default audit differs')
        checks.append({'arm':r['arm'],'operating_points_exact':len(r['operating_points']),'default_exact':True})
    require(summary['comparisons'] == comparisons(summary['results'],plan), 'Comparison arithmetic differs')
    result = {'status':'complete','summary_sha256':sha256(output/'summary.json'),'source_sha256':sha256(__file__),
              'checks':checks,'replay_exact':True,'new_forward_examples':0,'backward_examples':0,
              'confirmation_opened':False,'confirmation_scored':False,'original_validation_opened':False,
              'test_opened':False,'test_evaluated':False}
    atomic_json(result,receipt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','replay'))
    parser.add_argument('--data',dest='data_root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-root',type=Path)
    parser.add_argument('--receipt',type=Path)
    args = parser.parse_args()
    if args.action == 'run':
        require(args.run_root is not None and args.receipt is None,'Run needs --run-root only')
        result = run(args.data_root,args.run_root,args.output)
    else:
        require(args.receipt is not None and args.run_root is None,'Replay needs --receipt only')
        result = replay(args.data_root,args.output,args.receipt)
    print(json.dumps({'status':result['status'],'backward_examples':result['backward_examples'],
                      'test_opened':result['test_opened']},indent=2))


if __name__ == '__main__':
    main()
