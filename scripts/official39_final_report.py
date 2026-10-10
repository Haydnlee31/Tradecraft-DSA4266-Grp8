"""Publish and inspect the frozen official39 final test without running models.

Only aggregate JSON is read. No Torch, NumPy, dataset, checkpoint, threshold
search or cloud connection is needed. The legacy46 decision engine is not
called: its validation gates and heavy-model comparison are a different study.
All output files are created exclusively, so rebuilding cannot erase evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import official39_closeout as c
from src.data.label_map import BENIGN, CLASSES
from src.eval.official_evidence import close_tree, digest, metrics_from_counts, read_json, require

# These are the reviewed final artifacts, not user-selectable model winners.
ARCHIVE = 'tradecraft-official39-final-test-backup.tgz'
ARCHIVE_SHA256 = '91e240ffe1bbc07e2a4573a03d1112bd600c6bb8148ff49b01f97e2ca416431e'
CHECKS_SHA256 = 'ca8b2679e5b8255737fe6b4bfac0d9a75e4a2cba3d054ea370639a51a0a976d1'
EVIDENCE = c.REPORTS / 'final-test-results.json'
FINAL = 'outputs/official39-final-test-v1/'
REPLAY = 'outputs/official39-final-validation-replay-cuda-v1/'
PREP = 'outputs/official39-test-panel-v1/receipt.json'
PLAN = 'reports/full_data_extension/final-evaluation-plan.json'
CASES = {f'case-{i:02d}.json' for i in range(27)}
MEMBERS = {prefix + name for prefix in (FINAL, REPLAY)
           for name in CASES | {'checks.json', 'session.json', 'progress.json'}} | {PREP, PLAN}
LANE_NAMES = {'light': 'Centralized light', 'iid': 'IID federated', 'dirichlet': 'Controlled non-IID'}


def encode(value):
    """The archived runner wrote this exact UTF-8/LF format, including key order.

    Keeping the JSON documents intact lets a portable export reconstruct and
    verify their original byte hashes, without embedding executable code.
    """
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def json_digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def build(archive):
    require(digest(archive) == ARCHIVE_SHA256, 'Final archive checksum changed')
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        require(len(members) == 64 and len({m.name for m in members}) == 64,
                'Unexpected or duplicate archive member')
        require(sum(m.size for m in members) < 2_000_000, 'Oversized archive')
        for m in members:
            p = PurePosixPath(m.name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in m.name
                    and (m.isfile() or m.isdir()), 'Unsafe archive member')
        require({m.name for m in members if m.isfile()} == MEMBERS, 'Unexpected JSON member set')
        require({m.name for m in members if m.isdir()} == {FINAL.rstrip('/'), REPLAY.rstrip('/')},
                'Unexpected directory member')
        documents = {}
        for name in sorted(MEMBERS):
            raw = handle.extractfile(name).read()
            value = read_json(raw)
            require(encode(value) == raw, 'Original JSON bytes cannot be reconstructed')
            documents[name] = value
    evidence = dict(protocol='official39-final-evidence-v1',
                    archive=dict(name=ARCHIVE, sha256=ARCHIVE_SHA256), documents=documents)
    audit(evidence)
    return evidence


def validate_record(record, candidate, session_hash, supports, test_evaluated):
    """Recompute every metric and distinguish benign misses from wrong labels."""
    require(record['candidate'] == candidate and record['session_sha256'] == session_hash
            and record['status'] == 'complete', 'Case identity or completion changed')
    require(record['weights_unchanged'] is True and c.valid_digest(record['model_state_sha256']),
            'Invalid model-state evidence')
    require(record['test_evaluated'] is test_evaluated, 'Wrong evaluation scope')
    require(record['rows'] == sum(supports['all'].values())
            and record['primary_rows'] == sum(supports['panel'].values()), 'Population changed')
    # The historical iterator joins a singleton tail to the preceding batch.
    rows = record['rows']
    require(record['forward_batches'] == (rows + 511) // 512 - (rows % 512 == 1),
            'Forward coverage changed')
    for scope in ('all', 'panel'):
        metrics, errors = metrics_from_counts(record[scope]['metrics']['confusion_matrix'], supports[scope])
        close_tree(record[scope]['metrics'], metrics)
        close_tree(record[scope]['errors'], errors)
    full, panel = (record[s]['metrics']['confusion_matrix'] for s in ('all', 'panel'))
    require(all(a >= b for row_a, row_b in zip(full, panel) for a, b in zip(row_a, row_b)),
            'Primary matrix is not a subset of the original population')


def summarize(records):
    """Equal seed weights, sample SD and within-seed contrasts; never pool rows."""
    result = {}
    for scope in ('all', 'panel'):
        view = [dict(seed=r['candidate']['seed'], group=r['candidate']['lane'],
                     condition=r['candidate']['arm'], metrics=r[scope]['metrics'], errors=r[scope]['errors'])
                for r in records]
        groups = {}
        for lane in c.LANES:
            for arm in c.ARMS:
                chosen = sorted((r for r in view if r['group'] == lane and r['condition'] == arm),
                                key=lambda r: r['seed'])
                require([r['seed'] for r in chosen] == list(c.SEEDS), 'Missing/duplicate/extra training seed')
                item = {key: c.describe([r['metrics'][key] for r in chosen])
                        for key in ('macro_f1', 'benign_false_alert_rate')}
                item['per_class'] = {name: {key: c.describe([r['metrics']['per_class'][name][key] for r in chosen])
                                          for key in ('precision', 'recall', 'f1')} for name in CLASSES}
                groups[f'{lane}/{arm}'] = item
        result[scope] = dict(groups=groups, paired=c.paired_analysis({'records': view}))
    return result


def audit(evidence):
    """Audit receipts and recorded predictions, not re-inference or data integrity.

    A pinned top-level final receipt binds the session and every final case.
    The previously published CUDA receipt binds the replay cases. Even jointly
    changing a metric and its local case checksum cannot bypass those anchors.
    """
    require(set(evidence) == {'protocol', 'archive', 'documents'}
            and evidence['protocol'] == 'official39-final-evidence-v1', 'Wrong evidence schema')
    require(evidence['archive'] == dict(name=ARCHIVE, sha256=ARCHIVE_SHA256), 'Archive identity changed')
    d = evidence['documents']
    require(set(d) == MEMBERS, 'Missing/extra evidence documents')
    plan, prep = d[PLAN], d[PREP]
    c.audit_final_plan(plan)
    require(encode(plan) == c.FINAL_PLAN.read_bytes(), 'Frozen final plan changed')
    require(encode(prep) == (c.REPORTS / 'test-panel-preparation.json').read_bytes(), 'Prepared population changed')
    f, fs = d[FINAL + 'checks.json'], d[FINAL + 'session.json']
    v, vs = d[REPLAY + 'checks.json'], d[REPLAY + 'session.json']
    require(json_digest(f) == CHECKS_SHA256, 'Reviewed final receipt changed')
    require(encode(v) == (c.REPORTS / 'final-validation-cuda-checks.json').read_bytes(), 'Reviewed CUDA receipt changed')
    require(fs['protocol'] == 'official39-final-execution-v1' and fs['explicit_final_test_opt_in'] is True,
            'Final test was not explicitly authorized')
    require(fs['model_trained'] is False and vs['model_trained'] is False, 'Unexpected training')
    require(fs['candidates'] == vs['candidates'] == plan['candidates'], 'Candidate selection changed')
    require(fs['plan_sha256'] == vs['plan_sha256'] == v['plan_sha256'] == prep['plan_sha256'] == json_digest(plan),
            'Plan identity changed')
    require(fs['source_sha256'] == vs['source_sha256'] == v['source_sha256'] == prep['source_sha256'],
            'Implementation identity changed')
    for name, expected in fs['source_sha256'].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and digest(path) == expected, 'Frozen source changed')
    require(fs['runtime'] == vs['runtime'] == v['runtime'] and fs['runtime']['device'] == 'cuda',
            'Validated runtime changed')
    require(fs['validation_replay_sha256'] == json_digest(v)
            and fs['preparation_receipt_sha256'] == json_digest(prep), 'Approval gate identity changed')
    require(f['status'] == v['status'] == 'complete' and f['candidates'] == 27
            and f['test_evaluated'] is True and v['test_evaluated'] is False
            and v['test_opened'] is False and v['test_scoring_ready'] is True, 'Incomplete final evaluation or replay')
    require(v['exact_same_runtime_references'] == v['exact_historical_endpoints'] == v['candidates_verified'] == 27,
            'Incomplete historical replay')
    require(all(f[k] is False for k in ('model_trained', 'model_promoted', 'deployment_authorized')),
            'Unexpected training or promotion')
    require(f['primary_population'] == 'panel' and f['secondary_population'] == 'all'
            and f['seed_order'] == list(c.SEEDS), 'Reporting population or seeds changed')
    previous = c.read(c.EVIDENCE)
    previous_by_id = {r['id']: r for r in previous['records']}
    validation_panel = c.read(c.REPORTS / 'ablation-panel-results.json')
    validation_supports = dict(
        all={name: validation_panel['retained_class_counts'][name]
                   + validation_panel['excluded_class_counts'][name] for name in CLASSES},
        panel=previous['validation_class_counts'])
    test_supports = dict(all=prep['class_counts'], panel=prep['panel']['retained_class_counts'])
    records = []
    for prefix, check, supports in ((FINAL, f, test_supports), (REPLAY, v, validation_supports)):
        session_hash = json_digest(d[prefix + 'session.json'])
        require(check['session_sha256'] == session_hash, 'Session checksum changed')
        journal = d[prefix + 'progress.json']
        require(journal['session_sha256'] == session_hash and journal['cases'] == check['cases']
                and set(check['cases']) == CASES, 'Incomplete/mismatched recovery journal')
        for i, candidate in enumerate(plan['candidates']):
            name = f'case-{i:02d}.json'
            r = d[prefix + name]
            require(json_digest(r) == check['cases'][name], 'Case checksum changed')
            validate_record(r, candidate, session_hash, supports, prefix == FINAL)
            if prefix == FINAL:
                require(r['model_state_sha256'] == d[REPLAY + name]['model_state_sha256'],
                        'Final weights differ from the replayed model')
                records.append(r)
            else:
                require(r['same_runtime_reference_exact'] is True
                        and set(r['historical_comparison']) == {'all', 'panel'}, 'Replay comparison changed')
                require(all(x['exact'] is True and x['confusion_l1_difference'] == 0
                            and abs(x['macro_f1_difference']) <= 1e-12
                            for x in r['historical_comparison'].values()), 'Historical replay mismatch')
                close_tree(r['panel']['metrics'], previous_by_id[candidate['id']]['metrics'])
                close_tree(r['panel']['errors'], previous_by_id[candidate['id']]['errors'])
    close_tree(f['summary'], summarize(records))
    return dict(status='verified', candidates=27, primary_references=9, sensitivity_controls=18,
                primary_rows=sum(test_supports['panel'].values()), original_rows=sum(test_supports['all'].values()),
                metric_views_recomputed=108, source_test_evaluated=True,
                inference_performed=False, model_trained=False, deployment_authorized=False,
                scope='JSON provenance and count audit; not new prediction or dataset verification')


def research_decision(evidence):
    """A separate research-facing decision view, not an operational classifier.

    No retrospective numerical gates or winning masked model are introduced.
    References were chosen before test access. Show all eight classes and both
    kinds of attack error even when aggregate F1 or false alerts look favorable.
    """
    audit(evidence)
    d = evidence['documents']
    groups = d[FINAL + 'checks.json']['summary']['panel']['groups']
    records = [d[FINAL + f'case-{i:02d}.json'] for i in range(27)]
    references = {}
    for lane in c.LANES:
        rr = sorted((r for r in records if r['candidate']['lane'] == lane and r['candidate']['arm'] == 'full39'),
                    key=lambda r: r['candidate']['seed'])
        references[lane] = dict(metrics=groups[lane + '/full39'],
            errors_by_seed={str(r['candidate']['seed']): r['panel']['errors'] for r in rr},
            zero_category_recall_in_all_seeds=[name for name in CLASSES
                if all(r['panel']['metrics']['per_class'][name]['recall'] == 0 for r in rr)])
    return dict(scope='official39 final-test research decision', source_test_evaluated=True,
                evidence_checks_sha256=CHECKS_SHA256, research_references=references,
                primary_population='shared_test_panel', automatic_allow_or_block=False,
                deployment_authorized=False, model_promoted=False, numerical_deployment_thresholds=None,
                legacy46_defaults_changed=False, post_test_retuning_allowed=False,
                conclusion='Retain the predeclared full39 research references; no operational model promotion.',
                warnings=[
                    'Low benign false alerts do not establish useful detection; inspect attack-as-benign counts.',
                    'Wrong attack category and a benign miss are different security-response risks.',
                    'Three training seeds share a fixed client partition and collection, not independent deployments.',
                    'No matched official39 heavy model, physical-edge measurements or privacy guarantee.',
                    'Keep the original 46-feature study separate; test results cannot tune its defaults.',
                    'The final holdout is now exposed and cannot serve as a fresh test for further tuning.'],
                explanation_context=dict(
                    method='background-averaged integrated gradients, not exact conditional SHAP',
                    population='219 selected validation cases; not final-test explanations',
                    model_scope='centralized and IID seeds 7/17/27; non-IID seed 7 only',
                    findings='Number and protocol/flow features recur, but background and redundant-feature sensitivity remain.',
                    human_review_hypotheses=[
                        'Do not use a benign model output to suppress independent authentication or web alerts.',
                        'Review protocol/flag/size patterns with service context; no attribution-derived blocking threshold.',
                        'Treat the Number/windowing concern as a dataset/extractor audit question, not a device vulnerability.'],
                    sources=['reports/full_data_extension/explanation-study.md',
                             'reports/full_data_extension/explanation-sensitivity.md',
                             'reports/full_data_extension/ablation-confirmation-findings.md']))


def tables(evidence):
    """Generate the appendix directly from audited counts, including all controls."""
    audit(evidence)
    d = evidence['documents']
    summary = d[FINAL + 'checks.json']['summary']
    lines = ['# Official39 final test tables', '',
             'Generated from the frozen evidence. Equal-weight means across training seeds 7, 17 and 27;',
             'sample SD is training-seed variation, not uncertainty across independent datasets.',
             'False alerts are the fraction of true benign flows predicted as attacks.', '',
             'Full39 is the primary reference; both masks are sensitivity controls, not newly selected winners.', '']
    for scope, title in [('panel', 'Primary shared test population'), ('all', 'Secondary original test population')]:
        supports = d[PREP]['panel']['retained_class_counts'] if scope == 'panel' else d[PREP]['class_counts']
        lines += ['## ' + title, '', f'Rows: {sum(supports.values()):,}.', '',
                  '| Setting | Arm | Macro-F1 mean and SD | Macro-F1 range | Mean false alerts |',
                  '| --- | --- | ---: | ---: | ---: |']
        for lane in c.LANES:
            for arm in c.ARMS:
                g = summary[scope]['groups'][f'{lane}/{arm}']
                f = g['macro_f1']
                lines.append(f'| {LANE_NAMES[lane]} | {arm} | {f["mean"]:.6f} ± {f["sample_sd"]:.6f} | '
                             f'{f["min"]:.6f}–{f["max"]:.6f} | {100*g["benign_false_alert_rate"]["mean"]:.3f}% |')
        for arm in c.ARMS:
            lines += ['', '### Class recall for ' + arm.replace('_', ' '), '',
                      '| Class | Support per model | Centralized light | IID federated | Controlled non-IID |',
                      '| --- | ---: | ---: | ---: | ---: |']
            for name in CLASSES:
                values = [100*summary[scope]['groups'][f'{lane}/{arm}']['per_class'][name]['recall']['mean'] for lane in c.LANES]
                lines.append(f'| {name} | {supports[name]:,} | ' + ' | '.join(f'{x:.3f}%' for x in values) + ' |')
        lines += ['', '### Paired mask differences', '',
                  'Masked minus same-seed full39. All differences are percentage points, not relative percentages.', '',
                  '| Setting and mask | F1 seed 7 | F1 seed 17 | F1 seed 27 | Mean F1 change | Mean false-alert change |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name, p in summary[scope]['paired'].items():
            values = [*p['macro_f1']['values'], p['macro_f1']['mean'], p['benign_false_alert_rate']['mean']]
            lines.append('| ' + name + ' | ' + ' | '.join(f'{100*x:+.3f}' for x in values) + ' |')
        lines += ['']
    lines += ['## Per-seed full39 primary attack errors', '',
              'Counts are not averaged or pooled across seeds. Each row partitions the true attack class.', '',
              '| Setting | Seed | True class | Support | Correct category | Predicted benign | Wrong attack category |',
              '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for i in range(27):
        r = d[FINAL + f'case-{i:02d}.json']
        spec = r['candidate']
        if spec['arm'] == 'full39':
            for name in CLASSES:
                if name != BENIGN:
                    e = r['panel']['errors'][name]
                    counts = [e[k] for k in ('support', 'correct_category', 'predicted_benign', 'wrong_attack_category')]
                    lines.append(f'| {LANE_NAMES[spec["lane"]]} | {spec["seed"]} | {name} | '
                                 + ' | '.join(f'{x:,}' for x in counts) + ' |')
    lines += ['', 'All seeds, all arms, both populations, precision/recall/F1, confusion matrices and error counts',
              'remain in `final-test-results.json`. This appendix neither authorizes deployment nor changes thresholds.', '']
    return '\n'.join(lines)


def write_new(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'audit', 'decision', 'tables'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--evidence', type=Path, default=EVIDENCE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    require(args.action == 'build' or args.archive is None, 'Only build reads an archive')
    if args.action == 'build':
        require(args.archive is not None and args.output is not None, 'Supply archive and fresh output')
        require(not args.output.exists(), 'Preserve existing evidence; choose a fresh output')
        evidence = build(args.archive)
        write_new(args.output, encode(evidence))
        print(json.dumps(audit(evidence), indent=2))
    else:
        evidence = c.read(args.evidence)
        if args.action == 'audit':
            require(args.output is None, 'Audit is read-only; no output file')
            print(json.dumps(audit(evidence), indent=2))
        else:
            result = tables(evidence) if args.action == 'tables' else research_decision(evidence)
            content = result.encode('utf-8') if isinstance(result, str) else encode(result)
            if args.output is None:
                print(content.decode('utf-8'), end='')
            else:
                write_new(args.output, content)
                print(f'Created {args.output}; no training, inference or deployment action')


if __name__ == '__main__':
    main()
