"""Consolidate official-39 receipts without loading data or trained models.

This is an evidence ledger, NOT a replacement for the legacy decision engine.
Groups have explicitly curated scopes and endpoints; there is no cross-study
leaderboard, automatic winner, new threshold, training or deployment action.
The export reads only pinned JSON files/members. A teammate can subsequently
audit the published counts and summaries without downloading the dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

from src.data.label_map import BENIGN, CLASSES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(raw):
    """Reject duplicate keys and non-standard NaN/Infinity rather than hiding them."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f'Non-finite JSON value: {value}')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def at(value, pointer):
    """Resolve explicit JSON pointers, including indices in recorded histories."""
    if pointer == '':
        return value
    require(pointer.startswith('/'), 'Expected a JSON pointer')
    for part in pointer[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        if isinstance(value, list):
            require(part.isdigit(), 'Expected a non-negative list index')
            value = value[int(part)]
        else:
            value = value[part]
    return value


def metrics_from_counts(matrix, supports):
    """Rows are true classes, columns are predicted classes in canonical order.

    Count mistakes explicitly: missing an attack as benign differs from naming
    the wrong attack. Neither can be repaired by reporting accuracy alone.
    """
    n = len(CLASSES)
    require(isinstance(matrix, list) and len(matrix) == n, 'Wrong matrix shape')
    require(all(isinstance(row, list) and len(row) == n for row in matrix), 'Wrong matrix shape')
    require(all(type(x) is int and x >= 0 for row in matrix for x in row), 'Invalid confusion counts')
    require(set(supports) == set(CLASSES), 'Wrong support classes')
    require(all(type(supports[c]) is int and supports[c] > 0 for c in CLASSES), 'Invalid supports')
    totals = [sum(row) for row in matrix]
    require(totals == [supports[c] for c in CLASSES], 'Validation population changed')
    predicted = [sum(row[j] for row in matrix) for j in range(n)]
    per_class, errors = {}, {}
    benign = CLASSES.index(BENIGN)
    for i, name in enumerate(CLASSES):
        correct, total = matrix[i][i], totals[i]
        precision = correct / predicted[i] if predicted[i] else 0.
        recall = correct / total
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        per_class[name] = dict(precision=precision, recall=recall, f1=f1, support=total)
        if name == BENIGN:
            errors[name] = dict(support=total, correct=correct, false_alerts=total-correct,
                                false_alert_rate=(total-correct)/total)
        else:
            missed = matrix[i][benign]
            wrong = total-correct-missed
            errors[name] = dict(support=total, correct_category=correct,
                                predicted_benign=missed, wrong_attack_category=wrong,
                                category_recall=recall, attack_to_benign_rate=missed/total,
                                wrong_attack_category_rate=wrong/total)
    metrics = dict(macro_f1=statistics.fmean(v['f1'] for v in per_class.values()),
                   accuracy=sum(matrix[i][i] for i in range(n))/sum(totals),
                   benign_false_alert_rate=errors[BENIGN]['false_alert_rate'],
                   per_class=per_class, confusion_matrix=matrix)
    return metrics, errors


def close_tree(actual, expected):
    """Only floating-point roundoff is allowed; missing classes are an error."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Metric schema mismatch')
        for key in expected:
            close_tree(actual[key], expected[key])
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'Metric list mismatch')
        for a, b in zip(actual, expected):
            close_tree(a, b)
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), 'Metric value mismatch')
    else:
        require(type(actual) is type(expected) and actual == expected, 'Metric value mismatch')


def describe(values):
    # A single seed cannot estimate between-seed variability. Zero SD would
    # misleadingly make a single-seed diagnostic look perfectly stable.
    return dict(values=values, mean=statistics.fmean(values),
                sample_sd=statistics.stdev(values) if len(values) > 1 else None)


def summarize(evidence):
    require(evidence['classes'] == CLASSES, 'Canonical class order required')
    require(evidence['scope'] == 'official39_validation_only', 'Do not mix feature studies')
    for flag in ('dataset_opened', 'checkpoint_opened', 'model_trained',
                 'test_evaluated', 'deployment_authorized', 'model_promoted'):
        require(evidence[flag] is False, f'Unexpected action: {flag}')
    groups, records = evidence['groups'], evidence['records']
    require(len({r['id'] for r in records}) == len(records), 'Duplicate record ID')
    expected = {(g, c, s) for g, spec in groups.items()
                for c, seeds in spec['conditions'].items() for s in seeds}
    actual = [(r['group'], r['condition'], r['seed']) for r in records]
    require(len(set(actual)) == len(actual) and set(actual) == expected, 'Missing/duplicate/extra condition or seed')
    indexed = {}
    for r in records:
        metrics, errors = metrics_from_counts(r['metrics']['confusion_matrix'], evidence['validation_class_counts'])
        close_tree(r['metrics'], metrics)
        close_tree(r['errors'], errors)
        indexed[r['group'], r['condition'], r['seed']] = metrics
    summaries = {}
    for group, spec in groups.items():
        summaries[group] = {}
        for condition, seeds in spec['conditions'].items():
            require(seeds and len(set(seeds)) == len(seeds), 'Invalid expected seeds')
            ms = [indexed[group, condition, s] for s in seeds]
            item = {key: describe([m[key] for m in ms]) for key in ('macro_f1', 'benign_false_alert_rate')}
            item['seeds'] = seeds
            item['per_class'] = {c: {key: describe([m['per_class'][c][key] for m in ms])
                                     for key in ('recall', 'precision', 'f1')} for c in CLASSES}
            summaries[group][condition] = item
    comparisons = {}
    for pair in evidence['comparisons']:
        group, left, right = pair['group'], pair['left'], pair['right']
        require(pair['id'] not in comparisons, 'Duplicate comparison ID')
        seeds = groups[group]['conditions'][left]
        require(seeds == groups[group]['conditions'][right], 'Paired seeds must match exactly')
        # Differences are computed within seeds; pooled confusion counts would
        # answer a different question and conceal seed-to-seed variation.
        comparisons[pair['id']] = {
            'scope': pair['scope'], 'seeds': seeds,
            **{key: describe([indexed[group, left, s][key]-indexed[group, right, s][key]
                              for s in seeds]) for key in ('macro_f1', 'benign_false_alert_rate')}}
    return {'groups': summaries, 'paired_differences': comparisons}


def safe_path(root, relative):
    root, path = Path(root).resolve(), Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe source path')
    resolved = (root/path).resolve()
    require(resolved.is_relative_to(root), 'Source escapes its root')
    return resolved


def export(plan_path, repository, archive_dir):
    plan = read_json(Path(plan_path).read_bytes())
    sources, hashes, provenance = {}, {}, {}
    for key, spec in plan['sources'].items():
        root = archive_dir if 'member' in spec else repository
        require(root is not None, 'Archive directory required to rebuild source evidence')
        path = safe_path(root, spec['path'])
        if path not in hashes:
            hashes[path] = digest(path)
        require(hashes[path] == spec['sha256'], f'Source checksum mismatch: {key}')
        if 'member' in spec:
            member = spec['member']
            require(not Path(member).is_absolute() and '..' not in Path(member).parts, 'Unsafe archive member')
            with tarfile.open(path) as archive:
                matches = [m for m in archive.getmembers() if m.name == member]
                require(len(matches) == 1 and matches[0].isfile(), 'Missing/duplicate/non-file member')
                require(matches[0].size <= 10_000_000 and member.endswith('.json'), 'JSON receipt required')
                raw = archive.extractfile(matches[0]).read()
        else:
            require(path.suffix == '.json' and path.stat().st_size <= 10_000_000, 'JSON receipt required')
            raw = path.read_bytes()
        sources[key] = read_json(raw)
        for pointer, expected in spec.get('guards', {}).items():
            close_tree(at(sources[key], pointer), expected)
        provenance[key] = {**spec, 'json_sha256': hashlib.sha256(raw).hexdigest()}
    result = {key: plan[key] for key in ('scope', 'classes', 'validation_class_counts', 'groups', 'comparisons')}
    result.update(plan_sha256=digest(plan_path), source_evidence=provenance, records=[],
                  dataset_opened=False, checkpoint_opened=False, model_trained=False,
                  test_evaluated=False, deployment_authorized=False, model_promoted=False)
    for spec in plan['records']:
        selected = at(sources[spec['source']], spec['pointer'])
        metrics, errors = metrics_from_counts(selected['confusion_matrix'], plan['validation_class_counts'])
        if spec['format'] == 'metrics':
            close_tree(selected, metrics)
        else:
            require(spec['format'] == 'counts', 'Unknown metric format')
        if spec['pointer'].startswith('/history/'):
            # Historical bridges differ in schema, so use their recorded
            # histories, not a guessed epochs*batch formula for work counts.
            history = sources[spec['source']]['history']
            step = spec['endpoint']
            require(spec['pointer'] == f'/history/{step-1}/validation_metrics', 'Wrong budget endpoint')
            require([h['step'] for h in history] == list(range(1, len(history)+1)), 'Incomplete history')
            for counter in ('examples_processed', 'optimizer_steps'):
                require(sum(h[counter] for h in history[:step]) == spec[counter], 'Endpoint work mismatch')
                require(sum(h[counter] for h in history) == sources[spec['source']][counter], 'Total work mismatch')
        result['records'].append({**spec, 'metrics': metrics, 'errors': errors})
    result['summary'] = summarize(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', type=Path, help='Rebuild from pinned source JSON receipts')
    mode.add_argument('--audit', type=Path, help='Check published counts and summaries only')
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--archive-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.audit:
        evidence = read_json(args.audit.read_bytes())
        close_tree(evidence['summary'], summarize(evidence))
        print(f"PASS: {len(evidence['records'])} validation endpoints; no dataset or model access")
        return
    require(args.output is not None, '--output required for export')
    require(not args.output.exists(), 'Refusing to overwrite evidence')
    evidence = export(args.plan, args.repository, args.archive_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    print(f"Exported {len(evidence['records'])} validation endpoints to {args.output}")


if __name__ == '__main__':
    main()
