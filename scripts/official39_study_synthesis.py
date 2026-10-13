"""Audit and synthesize existing final results and costs; no new inference.

Class errors partition each attack into correct category, benign miss and wrong
attack category. Seed means remain descriptive: correlated flows are not new
independent replications. Communication is a tensor accounting model, not traffic.
"""
import argparse
import json
from pathlib import Path
import statistics

from scripts import official39_final_report as final, official39_closeout as closeout
from src.data.label_map import CLASSES, BENIGN
from src.eval.official_evidence import digest, require, metrics_from_counts

ROOT = Path(__file__).parents[1]
REPORTS = ROOT/'reports/federated_class_safeguards'


def communication(parameter_bytes, clients, rounds, extra_vectors_each_direction=0):
    """Count uncompressed FP32 model/control payloads, not protocol packets.

    Count one point-to-point broadcast and upload per participating client per
    round. Exclude raw-data transfer, TLS/serialization, validation, retries,
    setup and optimizer metadata. Central parameter exchange is zero; its data
    collection cost is unknown, not zero. No measured-network claim follows.
    """
    require(all(type(v) is int and v >= 0 for v in (parameter_bytes, clients, rounds, extra_vectors_each_direction)),
            'Invalid payload inputs')
    base = parameter_bytes*clients*rounds
    each = base*(1+extra_vectors_each_direction)
    return {'upload_bytes': each, 'broadcast_bytes': each, 'both_directions_bytes': 2*each,
            'extra_bytes_both_directions': 2*base*extra_vectors_each_direction,
            'measured_network_traffic': False, 'raw_data_collection_bytes': None,
            'transport_overhead_included': False}


def error_partition(cm):
    supports = {name: sum(cm[i]) for i, name in enumerate(CLASSES)}
    metrics, errors = metrics_from_counts(cm, supports)
    for name, e in errors.items():
        if name == BENIGN:
            e['false_alerts_per_10000_benign'] = e['false_alert_rate']*10000
        else:
            require(e['correct_category']+e['predicted_benign']+e['wrong_attack_category'] == e['support'],
                    'Attack errors do not partition class support')
            e['any_attack_alarm_recall'] = 1-e['attack_to_benign_rate']
    return metrics, errors


def build():
    evidence = closeout.read(final.EVIDENCE)
    final.audit(evidence)
    history = closeout.read(closeout.EVIDENCE)
    closeout.audit_evidence(history)
    historical = {r['id']: r for r in history['records']}
    records = []
    for name, r in evidence['documents'].items():
        if not name.startswith(final.FINAL+'case-') or r['candidate']['arm'] != 'full39':
            continue
        candidate = r['candidate']
        metrics, errors = error_partition(r['panel']['metrics']['confusion_matrix'])
        source = historical[candidate['id']]
        costs = {k: source[k] for k in ('num_parameters', 'examples_processed', 'optimizer_steps', 'training_validation_seconds')}
        costs['fp32_parameter_bytes'] = 4*source['num_parameters']
        costs['communication'] = communication(costs['fp32_parameter_bytes'],
                                               0 if candidate['lane'] == 'light' else 20, 20)
        records.append({'id': candidate['id'], 'lane': candidate['lane'], 'seed': candidate['seed'],
                        'metrics': metrics, 'errors': errors, 'costs': costs})
    require(len(records) == 9, 'Need every primary model')
    records.sort(key=lambda r: (r['lane'], r['seed']))
    groups = {}
    for lane in closeout.LANES:
        selected = [r for r in records if r['lane'] == lane]
        require([r['seed'] for r in selected] == [7, 17, 27], 'Missing or duplicate seed')
        groups[lane] = {
            'metrics': {k: closeout.describe([r['metrics'][k] for r in selected])
                        for k in ('macro_f1', 'benign_false_alert_rate')},
            'per_class': {c: {key: closeout.describe([r['metrics']['per_class'][c][key] for r in selected])
                              for key in ('precision', 'recall', 'f1')} for c in CLASSES},
            'error_rates': {c: {key: closeout.describe([r['errors'][c][key] for r in selected])
                                for key in (('false_alerts_per_10000_benign',) if c == BENIGN else
                                            ('category_recall', 'attack_to_benign_rate', 'wrong_attack_category_rate',
                                             'any_attack_alarm_recall'))} for c in CLASSES},
            'costs': {k: closeout.describe([r['costs'][k] for r in selected]) for k in
                      ('num_parameters', 'fp32_parameter_bytes', 'examples_processed', 'optimizer_steps', 'training_validation_seconds')},
            'communication': selected[0]['costs']['communication'],
        }
    # Same training-seed labels describe matched comparisons; the resulting gap
    # is not an isolated causal effect of data heterogeneity or a significance test.
    indexed = {(r['lane'], r['seed']): r for r in records}
    paired = {f'{a}_minus_{b}': closeout.describe([
        indexed[a, seed]['metrics']['macro_f1']-indexed[b, seed]['metrics']['macro_f1']
        for seed in closeout.SEEDS]) for a, b in (('light', 'iid'), ('iid', 'dirichlet'), ('light', 'dirichlet'))}
    sg = closeout.read(REPORTS/'scaffold-results.json')['primary']
    op = closeout.read(REPORTS/'operating-points-results.json')['primary']
    arms = {}
    for arm, result in sg['rounds'][-1]['arms'].items():
        arms[arm] = {
            'scope': 'TRAINING ONLY, warm-started continuation, one seed; not comparable to final-test table',
            'argmax_training_metrics': result['evaluation']['metrics'],
            'matched_5pct_training_audit_metrics': op['arms'][arm]['targets']['500']['audit'],
            'work': sg['work_per_arm'][arm],
            'fp32_parameter_bytes': sg['communication']['parameter_bytes'],
            'communication': communication(sg['communication']['parameter_bytes'],
                0 if arm == 'central_sgd' else 20, 10, 1 if arm == 'scaffold' else 0),
        }
    sources = [final.EVIDENCE, closeout.EVIDENCE, REPORTS/'scaffold-results.json', REPORTS/'operating-points-results.json']
    return {'status': 'complete', 'script_sha256': digest(__file__),
            'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
            'scope': 'Official39 fixed final-test aggregate evidence; later training-only diagnostics kept separate',
            'class_order': CLASSES, 'source_final_test_evaluated': True,
            'new_test_inference': False, 'dataset_arrays_opened': False, 'model_trained': False,
            'records': records, 'groups': groups, 'paired_macro_f1_gaps': paired,
            'training_only_sgd_diagnostic': arms,
            'interpretation': {
                'seed_spread': 'Sample SD/range across training seeds, not confidence intervals across datasets or partitions',
                'rates_per_10000': 'Rescaled empirical conditional rates, not forecasts of future network traffic',
                'cloud_times': 'Historical GPU training PLUS validation, excludes setup/checkpoint IO and billing/idle time; no central/federated real-network speedup claim',
                'payload': 'Estimated uncompressed FP32 model/control tensors; no distributed-network measurements',
                'comparison': 'No matched official39 heavy lane; original46 results stay separate; no deployment promotion',
                'inference': 'See separate local CPU benchmark; parameter storage is not process RAM or checkpoint size'},
            'cloud_used': False, 'deployment_authorized': False}


def tables(r):
    lines = ['# Official39 class failures and costs', '',
             'Generated from existing audited counts. Three training seeds; no new inference or training.', '',
             '## Primary final test results', '',
             '| Setting | Macro-F1 mean and SD | False alerts per 10000 benign | Training examples | Optimizer steps | GPU train plus validation seconds mean | FP32 model bytes | Estimated model exchange bytes |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for lane, g in r['groups'].items():
        m, costs = g['metrics']['macro_f1'], g['costs']
        lines.append(f"| {final.LANE_NAMES[lane]} | {m['mean']:.6f} ± {m['sample_sd']:.6f} | {g['error_rates'][BENIGN]['false_alerts_per_10000_benign']['mean']:.1f} | {costs['examples_processed']['mean']:,.0f} | {costs['optimizer_steps']['mean']:,.0f} | {costs['training_validation_seconds']['mean']:.2f} | {costs['fp32_parameter_bytes']['mean']:,.0f} | {g['communication']['both_directions_bytes']:,} |")
    lines += ['', 'Exchange counts both directions over20 full-participation rounds; excludes protocol and data collection costs. Central model exchange is zero, NOT zero data-transfer cost.', '',
              '## Every attack category and its two kinds of mistake', '',
              'Percentages are equal-weight means across seeds; the three outcome columns sum to100 for each class.', '',
              '| Setting | True attack | Correct category | Predicted benign | Wrong attack category |',
              '| --- | --- | ---: | ---: | ---: |']
    for lane, g in r['groups'].items():
        for c in CLASSES:
            if c == BENIGN: continue
            v = g['error_rates'][c]
            lines.append(f"| {final.LANE_NAMES[lane]} | {c} | {100*v['category_recall']['mean']:.3f}% | {100*v['attack_to_benign_rate']['mean']:.3f}% | {100*v['wrong_attack_category_rate']['mean']:.3f}% |")
    lines += ['', '## All class precision recall and F1', '',
              '| Setting | Class | Precision | Recall | F1 |', '| --- | --- | ---: | ---: | ---: |']
    for lane, g in r['groups'].items():
        for c, m in g['per_class'].items():
            lines.append(f"| {final.LANE_NAMES[lane]} | {c} | {100*m['precision']['mean']:.3f}% | {100*m['recall']['mean']:.3f}% | {m['f1']['mean']:.6f} |")
    lines += ['', '## Later SGD comparison is training only', '',
              'Do NOT rank these against the final-test table above. These are one-seed frozen warm-start endpoints.', '',
              '| Arm | Argmax training macro-F1 | Training false alerts | 5 percent target audit macro-F1 | Achieved audit false alerts | Training examples | Optimizer calls | Estimated exchange bytes |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm, row in r['training_only_sgd_diagnostic'].items():
        a, b, work = row['argmax_training_metrics'], row['matched_5pct_training_audit_metrics'], row['work']
        lines.append(f"| {arm} | {a['macro_f1']:.6f} | {100*a['benign_false_alert_rate']:.3f}% | {b['macro_f1']:.6f} | {100*b['benign_false_alert_rate']:.3f}% | {work['client_training_examples']:,} | {work['optimizer_steps']:,} | {row['communication']['both_directions_bytes']:,} |")
    lines += ['', 'The audit rows were held out from threshold fitting only, NOT from model fitting. All models remain research-only.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    r = build()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        final.write_new(args.output/'result.json', final.encode(r))
        final.write_new(args.output/'tables.md', tables(r).encode())
    else:
        print(tables(r), end='')


if __name__ == '__main__':
    main()
