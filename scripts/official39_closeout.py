"""Portable confirmation review and a closed final-evaluation protocol.

This entry point reads JSON only. It cannot open data arrays, load a checkpoint,
train a model or evaluate the test set. It lives outside src/ so the historical
training-source inventory and recovery contracts stay unchanged.
"""
import argparse
import copy
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.label_map import BENIGN, CLASSES
from src.eval.official_evidence import close_tree, digest, export, read_json, require, summarize

REPORTS = ROOT / 'reports/full_data_extension'
EVIDENCE_PLAN = REPORTS / 'ablation-confirmation-evidence-plan.json'
EVIDENCE = REPORTS / 'ablation-confirmation-results.json'
FINAL_PLAN = REPORTS / 'final-evaluation-plan.json'
LANES = ('light', 'iid', 'dirichlet')
ARMS = ('full39', 'number_masked', 'number_total_masked')
SEEDS = (7, 17, 27)


def read(path):
    return read_json(Path(path).read_bytes())


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def describe(values):
    # Variation across three training seeds is not uncertainty across datasets.
    return dict(values=values, mean=statistics.mean(values),
                sample_sd=statistics.stdev(values), min=min(values), max=max(values))


def paired_analysis(evidence):
    records = {(r['seed'], r['group'], r['condition']): r for r in evidence['records']}
    result = {}
    for lane in LANES:
        for arm in ARMS[1:]:
            pairs = [(records[s, lane, arm], records[s, lane, 'full39']) for s in SEEDS]
            item = {k: describe([a['metrics'][k] - b['metrics'][k] for a, b in pairs])
                    for k in ('macro_f1', 'benign_false_alert_rate')}
            item['per_class'] = {
                c: {k: describe([a['metrics']['per_class'][c][k] - b['metrics']['per_class'][c][k]
                                for a, b in pairs]) for k in ('precision', 'recall', 'f1')}
                for c in CLASSES}
            item['attack_error_counts'] = {
                k: describe([sum(a['errors'][c][k] - b['errors'][c][k] for c in CLASSES if c != BENIGN)
                             for a, b in pairs]) for k in ('predicted_benign', 'wrong_attack_category')}
            # Keep prospective replication separate from the exploratory seed
            # that motivated this experiment; do not hide a failed replication.
            item['prospective_seeds_17_27'] = {
                k: describe(item[k]['values'][1:]) for k in ('macro_f1', 'benign_false_alert_rate')}
            result[f'{lane}/{arm}-minus-full39'] = item
    return result


def audit_evidence(evidence):
    plan = read(EVIDENCE_PLAN)
    require(evidence['plan_sha256'] == digest(EVIDENCE_PLAN), 'Evidence plan changed')
    for key in ('scope', 'classes', 'validation_class_counts', 'groups', 'comparisons'):
        close_tree(evidence[key], plan[key])
    require(set(evidence['groups']) == set(LANES), 'Missing lane')
    for spec in evidence['groups'].values():
        require(spec['conditions'] == {arm: list(SEEDS) for arm in ARMS}, 'Missing arm or seed')
    require(sum(evidence['validation_class_counts'].values()) == 2047805, 'Shared panel changed')
    require(len(evidence['records']) == len(plan['records']) == 27, 'All 27 endpoints required')
    indexed = {r['id']: r for r in evidence['records']}
    require(len(indexed) == 27, 'Duplicate endpoint')
    training_plan = read(REPORTS / 'ablation-confirmation-plan.json')
    for spec in plan['records']:
        record = indexed[spec['id']]
        close_tree({k: v for k, v in record.items() if k not in ('metrics', 'errors')}, spec)
        require(record['endpoint'] == 20 and record['cohort_rows'] == 2000000
                and record['num_parameters'] == 5096, 'Endpoint or architecture changed')
        for key, value in training_plan['work_per_run'][record['group']].items():
            require(record[key] == value, 'Work budget mismatch')
        require(math.isfinite(record['training_validation_seconds'])
                and record['training_validation_seconds'] > 0, 'Invalid measured duration')
        artifact = record['artifact']
        require(valid_digest(artifact['archive_sha256'])
                and all(valid_digest(v) for v in artifact['sha256'].values()), 'Invalid artifact digest')
        for key in ('checkpoint_member', 'environment_member', 'result_member'):
            member = PurePosixPath(artifact[key])
            require(not member.is_absolute() and '..' not in member.parts
                    and '\\' not in artifact[key], 'Unsafe artifact member')
        require(artifact['checkpoint_member'].endswith('/last.pt'), 'Must retain final checkpoint')
    require(set(evidence['source_evidence']) == set(plan['sources']), 'Unexpected evidence source')
    for key, spec in plan['sources'].items():
        entry = evidence['source_evidence'][key]
        close_tree({k: v for k, v in entry.items() if k != 'json_sha256'}, spec)
        require(valid_digest(entry['json_sha256']), 'Invalid source JSON digest')
    close_tree(evidence['summary'], summarize(evidence))
    close_tree(evidence['paired_analysis'], paired_analysis(evidence))
    return {'status': 'verified', 'endpoints': 27, 'prospective_runs': 18,
            'model_trained': False, 'test_evaluated': False, 'deployment_authorized': False,
            'scope': 'Portable JSON provenance/metric audit, not fresh checkpoint or dataset verification'}


def final_policy():
    """One research comparison, with no newly tuned alerting rule or winner."""
    return {
        'prediction': 'argmax_logits_first_canonical_class_on_tie',
        'checkpoint': 'last.pt_at_step_20',
        'primary_models': 'all_nine_full39_lane_seed_combinations',
        'secondary_models': 'all_eighteen_masked_lane_seed_combinations',
        'seed_selection': 'report_all_7_17_27_no_best_seed_or_ensemble',
        'preprocessing': 'reuse_frozen_2m_train_scaler_float64_then_float32_and_training_arm_mask',
        'primary_population': 'shared_test_panel_excluding_input_matches_to_train_or_validation_under_any_arm',
        'matching': 'label_blind_exact_float32_vectors_signed_zero_canonicalized',
        'test_duplicates': 'retain_original_float64_unique_row_unit_after_filtering_report_projection_groups',
        'secondary_population': 'all_original_test_rows_same_forward_predictions',
        'empty_class': 'stop_before_scoring_if_any_shared_panel_class_has_zero_support_no_redraw',
        'metrics': ['macro_f1', 'benign_false_alert_rate', 'all_class_precision_recall_f1_support',
                    'confusion_matrix', 'per_class_benign_misses_and_wrong_attack_categories'],
        'aggregation': 'per_seed_then_mean_sample_sd_range_and_paired_differences_no_pooled_headline',
        'threshold_tuning_allowed': False,
        'calibration_or_prior_adjustment_allowed': False,
        'additional_training_allowed': False,
        'score_based_stopping_allowed': False,
        'automatic_model_promotion_allowed': False,
        'post_test_retuning_allowed': False,
    }


def candidates(evidence):
    return [dict(id=r['id'], lane=r['group'], arm=r['condition'], seed=r['seed'],
                 role='primary_reference' if r['condition'] == 'full39' else 'secondary_sensitivity',
                 artifact=copy.deepcopy(r['artifact'])) for r in evidence['records']]


def operating_requirements():
    return dict(scope='research comparison, not operational deployment', numerical_deployment_thresholds=None,
                automatic_allow_or_block=False, rare_class_blind_spots_must_be_displayed=True,
                existing_legacy46_decision_defaults_changed=False)


def audit_final_plan(plan, evidence_path=EVIDENCE):
    evidence = read(evidence_path)
    audit_evidence(evidence)
    require(plan['protocol'] == 'official39-final-evaluation-v1'
            and plan['status'] == 'prepared_test_closed', 'Unexpected final protocol status')
    require(plan['evidence_sha256'] == digest(evidence_path), 'Final candidate evidence changed')
    require(plan['classes'] == CLASSES and plan['training_seeds'] == list(SEEDS), 'Class/seed policy changed')
    for flag in ('test_data_opened_during_preparation', 'test_evaluated', 'test_access_authorized',
                 'cloud_authorized', 'deployment_authorized', 'runner_implemented'):
        require(plan[flag] is False, f'Unapproved action: {flag}')
    close_tree(plan['policy'], final_policy())
    close_tree(plan['candidates'], candidates(evidence))
    close_tree(plan['operating_requirements'], operating_requirements())
    split_path = REPORTS / 'results/protocol.json'
    split = read(split_path)
    data = plan['data_identity']
    panel = read(REPORTS / 'ablation-panel-results.json')
    loader = read(REPORTS / 'loader-check.json')
    require(data['parent_shards_manifest_sha256'] == loader['manifest_sha256'], 'Parent shards changed')
    require(data['split_protocol_sha256'] == digest(split_path), 'Split protocol changed')
    require(data['test_class_counts_before_overlap_filter'] == split['counts_by_split_and_class']['test'],
            'Test population changed')
    require(data['test_rows_before_overlap_filter'] == sum(split['counts_by_split_and_class']['test'].values()),
            'Test row count changed')
    require(data['features'] == split['features'] and len(data['features']) == 39, 'Feature schema changed')
    require(data['packed_manifest_sha256'] == panel['packed_manifest_sha256']
            and data['scaler_sha256'] == panel['scaler_sha256']
            and data['train_validation_arrays_sha256'] == panel['input_files_sha256'], 'Model input identity changed')
    require(data['shared_test_panel_receipt_sha256'] is None
            and data['shared_test_panel_rows'] is None, 'Do not fabricate a prepared test panel')
    require(data['reference_row_counts'] == {'selected_train': 2000000, 'all_validation': 2059284},
            'Overlap reference populations changed')
    require(plan['approval_gates'] == [
        'explicit_user_approval_before_reading_test_shards',
        'bounded_label_blind_overlap_audit_and_frozen_shared_panel_receipt',
        'synthetic_runner_tests_and_validation_checkpoint_replay',
        'explicit_user_approval_for_one_locked_test_evaluation'], 'Final evaluation gates changed')
    return dict(status='prepared_test_closed', candidates=27, primary_references=9,
                secondary_sensitivity=18, test_evaluated=False, test_access_authorized=False,
                next='Prepare/test runner on synthetic and validation inputs; request approval before test access')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'audit', 'check-final-plan'))
    parser.add_argument('--archive-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evidence', type=Path, default=EVIDENCE)
    parser.add_argument('--plan', type=Path, default=FINAL_PLAN)
    args = parser.parse_args()
    if args.action == 'build':
        require(args.archive_dir is not None and args.output is not None, 'Supply archive-dir and fresh output')
        require(not args.output.exists(), 'Preserve existing evidence; choose a fresh output')
        evidence = export(EVIDENCE_PLAN, ROOT, args.archive_dir)
        evidence['paired_analysis'] = paired_analysis(evidence)
        result = audit_evidence(evidence)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    elif args.action == 'audit':
        result = audit_evidence(read(args.evidence))
    else:
        result = audit_final_plan(read(args.plan), args.evidence)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
