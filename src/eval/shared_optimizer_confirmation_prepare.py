"""Lock the twelve confirmation endpoints using development-only replay.

There is intentionally no confirmation-scoring action here. Reading an existing
manifest is allowed; opening reserved row IDs, features or labels is not. The
next evaluation must consume this lock without changing models or thresholds.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_subsets import SubsetBatches
from src.data.shared_optimizer_data import ROOT
from src.eval import shared_optimizer_alarms as alarms
from src.eval import shared_optimizer_block_replication as previous
from src.eval import shared_optimizer_performance as training
from src.eval import shared_optimizer_replication as rep
from src.eval.federated_class_safeguard import state_digest
from src.federated.shared_adam import require
from src.models.official_streaming import confusion_metrics
from src.models.research import atomic_json

PLAN = ROOT/'reports/federated_shared_optimizer/confirmation-plan.json'
PARENT = PLAN.parent/'block-replication-results.json'
ARMS = ('central_adam','shared_adam','fedavg_adam','block128')


def sources():
    names = ('src/eval/shared_optimizer_confirmation_prepare.py','src/data/official_subsets.py',
             'src/data/official_loader.py','src/data/official_inventory.py')
    return {**previous.sources(),**{name:sha256(ROOT/name) for name in names}}


def read_contract(require_runtime=False):
    plan = json.loads(PLAN.read_text())
    old,base,alarm_plan,performance,_,_ = previous.read_contract(require_runtime=require_runtime)
    require(sha256(PARENT) == plan['parent_evidence_sha256'], 'Replication evidence differs')
    evidence = json.loads(PARENT.read_text()); s,a = evidence['summary'],evidence['audit']
    require(s['plan'] == old and s['contract'] == previous.contract(), 'Replication contract differs')
    require(a['status'] == 'complete' and a['summary_sha256'] ==
            hashlib.sha256(json.dumps(s,indent=2).encode()).hexdigest(), 'Replication not audited')
    require(s['assessment']['all_seeds_pass'] and
            s['assessment']['candidate_for_confirmation_preparation'] == 'block128', 'Candidate not eligible')
    if require_runtime:
        require(all(s['runtime'][k] == v for k,v in previous.runtime().items()), 'Runtime differs')
    require(plan['seeds'] == [7,17,27] and plan['arms'] == list(ARMS), 'Frozen roster differs')
    for key in ('screen','primary_target_basis_points','false_alert_targets_basis_points','device','threads'):
        require(plan[key] == old[key], f'Fixed criterion differs: {key}')
    require(plan['data_manifest_sha256'] == alarm_plan['data_manifest_sha256'] and
            plan['development_rows'] == alarm_plan['development_rows'], 'Data identity differs')
    require(plan['confirmation_rows'] == sum(performance['summary']['data']['counts']['confirmation'].values()),
            'Reserved row count differs')
    require(plan['expected_endpoints'] == 12 and plan['expected_locked_operating_points'] == 48 and
            plan['preparation_forward_examples'] == 12*plan['development_rows'] and
            plan['planned_confirmation_forward_examples'] == 12*plan['confirmation_rows'], 'Forward budget differs')
    require(plan['require_every_seed_to_pass'] is True and all(plan[k] is False for k in ('training_permitted','threshold_refitting_permitted',
            'confirmation_scoring_permitted','original_validation_scoring_permitted','test_scoring_permitted')),
            'Preparation access boundary changed')
    return plan,base,alarm_plan,performance,evidence


def endpoint_records(evidence):
    rows = [*evidence['summary']['references'],*evidence['summary']['results']]
    require(len(rows) == 12 and {(r['seed'],r['arm']) for r in rows} ==
            {(seed,arm) for seed in (7,17,27) for arm in ARMS}, 'Missing/duplicate endpoint')
    return {(r['seed'],r['arm']):r for r in rows}


def registry(evidence,plan):
    """All choices derive from audited evidence; there is no model selection."""
    records = endpoint_records(evidence); entries = []
    for seed in plan['seeds']:
        for arm in plan['arms']:
            r = records[seed,arm]
            if arm == 'block128':
                folder = ('outputs/shared-optimizer-block-compression-v1/block128' if seed == 7 else
                          f'outputs/shared-optimizer-block-replication-v1/seed{seed}-block128')
                logits = f'{folder}/logits.npy'
                model_key,model_sha = 'visible',r['delivered_model_sha256']
            elif seed == 7:
                folder = f'outputs/shared-optimizer-performance-v1/{arm}-lr{r["learning_rate"]}'
                logits = f'outputs/shared-optimizer-alarm-diagnostic-v1/{arm}-logits.npy'
                model_key,model_sha = 'model',r['final_model_sha256']
            else:
                folder = f'outputs/shared-optimizer-replication-v1/seed{seed}-{arm}'
                logits = f'{folder}/logits.npy'
                model_key,model_sha = 'model',r['final_model_sha256']
            calibrations = [copy.deepcopy(p['calibration']) for p in r['operating_points']]
            require([c['target_basis_points'] for c in calibrations] == plan['false_alert_targets_basis_points'] and
                    all(np.isfinite(c['threshold']) for c in calibrations), 'Frozen thresholds differ')
            entries.append({'seed':seed,'arm':arm,'checkpoint':f'{folder}/last.pt',
                'checkpoint_sha256':r['checkpoint_sha256'],'state_key':model_key,'model_sha256':model_sha,
                'development_logits':logits,'development_logits_sha256':r['logits_sha256'],
                'initial_model_sha256':r['initial_model_sha256'],'sample_chain_sha256':r['sample_chain_sha256'],
                'learning_rate':r['learning_rate'],'training_work':r['work'],'calibrations':calibrations})
    return entries


def within_root(root,relative):
    root = Path(root).resolve(); p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'Unsafe artifact path')
    resolved = (root/p).resolve()
    require(resolved.is_relative_to(root), 'Artifact path escapes project')
    return resolved


def load_model(entry,base,project_root):
    checkpoint = within_root(project_root,entry['checkpoint'])
    require(sha256(checkpoint) == entry['checkpoint_sha256'], 'Checkpoint checksum differs')
    require(entry['state_key'] == ('visible' if entry['arm'] == 'block128' else 'model'), 'Wrong deployment state')
    model = training.initial_model({**base,'seed':entry['seed']})
    require(state_digest(model) == entry['initial_model_sha256'], 'Initialization differs')
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['state'][entry['state_key']],strict=True)
    require(state_digest(model) == entry['model_sha256'], 'Delivered model differs')
    require(saved['state']['sample_chain_sha256'] == entry['sample_chain_sha256'] and
            saved['state']['work'] == entry['training_work'], 'Checkpoint work or samples differ')
    return model.eval()


def metadata_preflight(data_root,parent,subsets,base,performance,plan):
    """Read only manifests, recipes and the fit-only scaler, never raw shards."""
    root = Path(data_root)
    require(sha256(root/'manifest.json') == plan['data_manifest_sha256'], 'Prepared manifest differs')
    m = json.loads((root/'manifest.json').read_text())
    require(m == performance['summary']['data'], 'Prepared receipt differs')
    require(sha256(root/'scaler.json') == m['files']['scaler.json'], 'Fit scaler differs')
    scaler = json.loads((root/'scaler.json').read_text())
    require(scaler['n_samples_seen'] == base['fit_rows'] and len(scaler['mean']) == len(m['features']) and
            len(scaler['scale']) == len(m['features']) and np.isfinite(scaler['mean']).all() and
            np.isfinite(scaler['scale']).all() and (np.asarray(scaler['scale']) > 0).all(), 'Invalid fit scaler')
    require(sha256(Path(parent)/'manifest.json') == base['parent_manifest_sha256'] and
            sha256(Path(subsets)/'manifest.json') == base['subset_manifest_sha256'], 'Historical source differs')
    # Constructor validates metadata only. raw_batches() must NOT be called in
    # preparation, since those training shards also contain the reserved rows.
    data = SubsetBatches(parent,subsets,base['historical_training_rows'])
    require(data.features == m['features'] and data.rows == base['historical_training_rows'], 'Source feature schema differs')
    return {'data_manifest_sha256':plan['data_manifest_sha256'],'scaler_sha256':m['files']['scaler.json'],
        'confirmation_row_ids_sha256':m['files']['confirmation_row_ids.npy'],
        'confirmation_counts':m['counts']['confirmation'],'confirmation_rows':sum(m['counts']['confirmation'].values()),
        'features':m['features'],'classes':CLASSES,'historical_training_rows':data.rows,
        'source_parent_manifest_sha256':sha256(Path(parent)/'manifest.json'),
        'source_parent_protocol_sha256':sha256(Path(parent)/'protocol.json'),
        'source_subset_manifest_sha256':sha256(Path(subsets)/'manifest.json'),
        'source_subset_recipe_sha256':sha256(Path(subsets)/'recipe.json'),
        'source_shard_count':len(data.shards),'raw_shards_opened':False,'reserved_row_ids_opened':False,
        'unhashed_staging_cache_used':False}


def frozen_replay(logits,y,calibration,calibrations):
    """Check stored thresholds without calling any threshold-fitting helper."""
    score,winners = alarms.margins(logits)
    default = alarms.decisions(score,winners,0.)
    require(np.array_equal(default,logits.argmax(axis=1)), 'Default decision bridge differs')
    benign_scores = score[calibration & (y == alarms.BENIGN)]
    ordered = np.sort(benign_scores)
    points = []
    for c in calibrations:
        n = len(ordered); allowed = n*c['target_basis_points']//10000
        require(c['calibration_benign_rows'] == n and c['allowed_false_alerts'] == allowed and
                c['ascending_order_statistic_index'] == n-allowed-1 and
                c['threshold'] == float(ordered[n-allowed-1]), 'Stored threshold no longer matches calibration')
        observed = int((benign_scores > c['threshold']).sum())
        require(observed == c['observed_false_alerts'] <= allowed and
                c['calibration_false_alert_rate'] == observed/n and
                c['boundary_ties'] == int((benign_scores == c['threshold']).sum()), 'Calibration counts differ')
        predicted = alarms.decisions(score,winners,c['threshold'])
        # Independent direct-logit decision verifies the margin implementation.
        direct = np.where(logits[np.arange(len(y)),winners].astype(np.float64) >
                          logits[:,alarms.BENIGN].astype(np.float64)+c['threshold'],winners,alarms.BENIGN)
        require(np.array_equal(predicted,direct), 'Independent frozen decisions differ')
        points.append({'calibration':copy.deepcopy(c),'audit_metrics':alarms.metrics(y[~calibration],predicted[~calibration])})
    return {'zero_margin_argmax_exact':True,'full_development_default_metrics':confusion_metrics(
        np.bincount(y*len(CLASSES)+default,minlength=len(CLASSES)**2).reshape(len(CLASSES),len(CLASSES))),
        'audit_default_metrics':alarms.metrics(y[~calibration],default[~calibration]),'operating_points':points}


def peak_rss():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if platform.system() == 'Darwin' else value*1024)
    except (ImportError,AttributeError):
        return None


def prepare(data_root,parent,subsets,output,project_root=ROOT):
    plan,base,alarm_plan,performance,evidence = read_contract(require_runtime=True)
    output = Path(output)
    require(not output.exists(), 'Preserve existing preparation; use a new destination')
    started = time.perf_counter()
    source = sources(); plan_sha = sha256(PLAN)
    source_receipt = metadata_preflight(data_root,parent,subsets,base,performance,plan)
    entries = registry(evidence,plan)
    x,y,_ = alarms.load_development(data_root,alarm_plan,performance)
    calibration,groups = alarms.grouped_split(x,alarm_plan['split_domain'])
    require(all(evidence['summary']['split'][k] == v for k,v in groups.items()), 'Development groups differ')
    output.mkdir(parents=True)
    atomic_json({'status':'preparing','confirmation_opened':False},output/'status.json')
    lock = {'version':plan['version'],'plan':plan,'plan_sha256':plan_sha,'sources':source,
        'parent_evidence_sha256':sha256(PARENT),'data':source_receipt,'models':entries,
        'development_split':evidence['summary']['split'],'runtime':previous.runtime(),
        'confirmation_scoring_authorized':False,'training_permitted':False,'threshold_refitting_permitted':False}
    atomic_json(lock,output/'lock.json')
    np.save(output/'calibration_mask.npy',calibration,allow_pickle=False)
    require(sha256(output/'calibration_mask.npy') == evidence['summary']['split']['calibration_mask_sha256'], 'Mask differs')
    records = endpoint_records(evidence); checks = []; inference_seconds = 0.
    for entry in entries:
        model = load_model(entry,base,project_root)
        old_path = within_root(project_root,entry['development_logits'])
        require(sha256(old_path) == entry['development_logits_sha256'], 'Historical logits differ')
        before = time.perf_counter()
        logits = rep.infer(model,x,alarm_plan['inference_batch_size'])
        elapsed = time.perf_counter()-before; inference_seconds += elapsed
        require(np.array_equal(logits,np.load(old_path,mmap_mode='r',allow_pickle=False)), 'Checkpoint inference differs')
        evaluated = frozen_replay(logits,y,calibration,entry['calibrations'])
        old = records[entry['seed'],entry['arm']]
        require(all(old[k] == v for k,v in evaluated.items()), 'Historical endpoint differs')
        checks.append({'seed':entry['seed'],'arm':entry['arm'],'checkpoint_sha256':entry['checkpoint_sha256'],
            'model_sha256':state_digest(model),'checkpoint_inference_exact':True,'historical_endpoint_exact':True,
            'independent_frozen_operating_points':len(entry['calibrations']),'inference_seconds':elapsed})
        print(f'PASS seed={entry["seed"]} {entry["arm"]}: checkpoint, logits and frozen thresholds exact',flush=True)
    require(sources() == source and sha256(PLAN) == plan_sha, 'Source or plan changed')
    require(len(checks) == plan['expected_endpoints'] and
            sum(c['independent_frozen_operating_points'] for c in checks) == plan['expected_locked_operating_points'],
            'Replay budget differs')
    result = {'status':'ready_for_explicit_confirmation_authorization','lock_sha256':sha256(output/'lock.json'),
        'source_sha256':sha256(__file__),'checks':checks,'forward_examples':len(checks)*len(y),
        'planned_confirmation_forward_examples':len(checks)*source_receipt['confirmation_rows'],
        'new_model_fits':0,'backward_examples':0,'inference_seconds':inference_seconds,
        'elapsed_seconds':time.perf_counter()-started,'process_peak_rss_bytes':peak_rss(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'thresholds_refitted':False,'confirmation_opened':False,'confirmation_scored':False,
        'reserved_row_ids_opened':False,'raw_shards_opened':False,'original_validation_opened':False,
        'test_opened':False,'test_evaluated':False,'cloud_used':False}
    require(result['forward_examples'] == plan['preparation_forward_examples'] and
            result['planned_confirmation_forward_examples'] == plan['planned_confirmation_forward_examples'], 'Forward count differs')
    atomic_json(result,output/'checks.json')
    atomic_json({'status':result['status'],'confirmation_opened':False},output/'status.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--subsets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    r = prepare(args.data,args.parent,args.subsets,args.output)
    print(json.dumps({k:r[k] for k in ('status','forward_examples','inference_seconds','confirmation_opened','cloud_used')},indent=2))


if __name__ == '__main__':
    main()
