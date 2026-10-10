"""Publish and audit aggregate router evidence without opening any flow arrays."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts import official39_routing_gate as runner
from src.eval.official_evidence import digest, metrics_from_counts, require

frozen, gate = runner.frozen, runner.gate


def audit(record):
    require(record['status'] == 'complete' and record['scope'] == 'reused_validation_exploratory_internal_audit'
            and record['model_trained'] is True and record['learned_gate_trained'] is True
            and all(record[k] is False for k in runner.FALSE_FLAGS), 'Wrong experiment scope')
    runner.validate_plan(record['identity']['plan'])
    require(record['identity']['plan_sha256'] == digest(runner.PLAN)
            and all(digest(ROOT/n) == h for n, h in record['identity']['source_sha256'].items()), 'Plan/source identity changed')
    require(len(record['cache_checks']) == 12 and
            all(c['prediction_replay_exact'] and c['weights_unchanged'] for c in record['cache_checks']), 'Missing neural replay')
    expected = [(seed, lane) for seed in (7, 17, 27) for lane in ('iid', 'dirichlet')]
    require([(p['seed'], p['lane']) for p in record['prepared_pairs']] == expected
            and [(p['seed'], p['lane']) for p in record['pairs']] == expected, 'Incomplete gate pairs')
    partition = record['split']['partitions']
    require(set(partition) == {'fit', 'calibration', 'audit'}
            and sum(v['rows'] for v in partition.values()) == record['validation_panel']['retained_rows'], 'Partition counts differ')
    for name, part in partition.items():
        require(sum(part['classes'].values()) == part['rows'] and all(v >= 25 for v in part['classes'].values())
                and 0 < part['unique_groups'] <= part['rows'], 'Invalid partition support')
    require(all(sum(part['classes'][c] for part in partition.values()) == record['validation_panel']['retained_class_counts'][c]
                for c in gate.CLASSES), 'Partition class totals differ')
    supports, rows = partition['audit']['classes'], partition['audit']['rows']
    for prepared, pair in zip(record['prepared_pairs'], record['pairs']):
        require(0 < prepared['nodes'] <= 63 and prepared['depth'] <= 5
                and sum(prepared['target_counts']) == record['split']['fit_sample_rows'], 'Invalid gate fit')
        for standalone in [pair['base'], *pair['always'].values()]:
            m, e = metrics_from_counts(standalone['metrics']['confusion_matrix'], supports)
            require(m == standalone['metrics'] and e == standalone['errors'], 'Standalone metrics changed')
        base = pair['base']['metrics']['confusion_matrix']
        require(len(prepared['policies']) == len(pair['policies']) == 21, 'Incomplete policy comparisons')
        seen = set()
        for selected, result in zip(prepared['policies'], pair['policies']):
            key = (result['rule'], result['kind'], result['budget_percent'])
            require(key == (selected['rule'], selected['kind'], selected['budget_percent']) and key not in seen
                    and result['threshold'] == selected['selection']['threshold'], 'Policy changed after calibration')
            seen.add(key)
            chosen = selected['selection']
            require(chosen['calls'] <= partition['calibration']['rows']*result['budget_percent']//100
                    and (result['kind'] != 'guarded' or chosen['false_alerts'] <= selected['calibration_base_false_alerts']),
                    'Invalid calibration constraint')
            m, e = metrics_from_counts(result['confusion_matrix'], supports)
            require(result['rows'] == rows and result['metrics'] == m and result['errors'] == e
                    and result['macro_f1'] == gate.macro_f1(np.asarray(result['confusion_matrix'])), 'Audit metrics changed')
            require(result['false_alerts'] == e[gate.BENIGN]['false_alerts']
                    and result['benign_guard_respected'] == (result['false_alerts'] <= gate.false_alerts(np.asarray(base)))
                    and result['budget_respected'] == (result['calls'] <= rows*result['budget_percent']//100)
                    and result['escalation_rate'] == result['calls']/rows, 'Reported constraint changed')
            routing = result['per_class_routing']
            for k, total in [('escalated', 'calls'), ('corrected', 'corrected_rows'), ('damaged', 'damaged_rows')]:
                require(sum(v[k] for v in routing.values()) == result[total], 'Routing totals differ')
            for c, name in enumerate(gate.CLASSES):
                v = routing[name]
                require(v['support'] == supports[name] and 0 <= v['corrected']+v['damaged'] <= v['escalated'] <= v['support']
                        and result['confusion_matrix'][c][c] == base[c][c]+v['corrected']-v['damaged'], 'Correction/harm counts differ')
    require(record['summary'] == runner.summarize(record['pairs']), 'Summary or continuation screen changed')
    return dict(status='verified', gates=6, frozen_neural_models=12, audit_rows=rows,
                test_opened=False, cloud_authorized=False,
                scope='Aggregate provenance and count audit, not new model inference')


def publish(output, destination):
    output, destination = Path(output), Path(destination)
    require(not destination.exists(), 'Preserve existing published evidence')
    record = frozen.read(output/'receipt.json')
    audit(record)
    require(digest(output/'calibration.json') == record['calibration_sha256'], 'Calibration freeze changed')
    locked = frozen.read(output/'calibration.json')
    require(locked['status'] == 'frozen_before_internal_audit' and 'pairs' not in locked
            and locked['prepared_pairs'] == record['prepared_pairs'], 'Missing pre-audit freeze')
    for c in record['cache_checks']:
        for name, value in c['files'].items():
            require(Path(name).name == name and digest(output/name) == value, 'Private prediction cache changed')
    for p in record['prepared_pairs']:
        require(p['gate'] == f'seed{p["seed"]}-{p["lane"]}-gate.json'
                and digest(output/p['gate']) == p['gate_sha256'], 'Private gate changed')
    for name, value in record['split']['files'].items():
        require(name in ('partitions.npy', 'fit_rows.npy', 'group_keys.npy') and digest(output/name) == value, 'Private split changed')
    export = dict(record, source_receipt_sha256=digest(output/'receipt.json'), publisher_sha256=digest(__file__))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # No individual predictions, probabilities, inputs or tree parameters are
    # copied: the receipt already consists entirely of aggregate evidence.
    import json
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(export, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['publish', 'audit'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        require(args.output is not None, 'Private output directory required')
        publish(args.output, args.evidence)
    print(audit(frozen.read(args.evidence)))


if __name__ == '__main__':
    main()
