"""Select a bounded automatic score rule locally, without refitting classifiers.

All rows come from reused development validation. Calibration chooses the
rule; the unchanged development audit checks that frozen choice. Neither is a
fresh independent test. There is deliberately no original-test or cloud mode.
"""
import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from scripts import official39_balanced_tree as prior
from src.eval import score_adjustment as rules
from src.eval.official_evidence import digest, require

ablation, expert, frozen, headroom = prior.ablation, prior.expert, prior.frozen, prior.headroom
PLAN = ROOT/'reports/risk_aware_routing/score-adjustment-plan.json'
REFERENCE = ROOT/'reports/risk_aware_routing/balanced-tree-results.json'
SOURCES = ['src/eval/score_adjustment.py', 'scripts/official39_score_adjustment.py']
FALSE_FLAGS = ('new_models_fitted', 'test_opened', 'test_evaluated', 'cloud_authorized',
               'human_review_required', 'deployment_authorized', 'independent_confirmation')
TRUE_FLAGS = ('calibration_used', 'decision_rule_search')


def validate_plan(p):
    require(p == frozen.read(PLAN) and p['protocol'] == 'official39-score-adjustment-v1'
            and p['scope'] == 'reused_validation_exploratory_internal_audit'
            and p['reference_sha256'] == digest(REFERENCE) and p['seeds'] == rules.SEEDS
            and p['methods'] == list(rules.METHODS) and p['exponents'] == list(rules.EXPONENTS)
            and p['arm'] == prior.ARM and p['device'] == 'cpu' and p['threads'] == 2
            and p['batch_size'] == 8192 and p['fit_rows'] == 270326 and p['audit_rows'] == 408027
            and p['original_calibration_rows'] == 409919
            and all(p[k] is False for k in FALSE_FLAGS) and all(p[k] is True for k in TRUE_FLAGS),
            'Unreviewed decision-rule experiment')


def probability_rows(model, x, rows, features):
    """Inference only: canonical columns, bounded batches, immutable model."""
    require(np.array_equal(model.classes_, np.arange(8)), 'Wrong probability column order')
    ablation.ablation.rows_ok(rows, len(x))
    before = prior.balanced.expert.state_hash(model)
    p = np.empty((len(rows), 8), dtype=np.float64)
    for start in range(0, len(rows), 8192):
        ids = rows[start:start+8192]
        p[start:start+len(ids)] = model.predict_proba(ablation.ablation.project(x[ids], features, prior.ARM))
    rules.probabilities_ok(p)
    require(prior.balanced.expert.state_hash(model) == before
            and ablation.ablation.masked_splits(model, features, prior.ARM) == 0,
            'Inference changed model or used masked inputs')
    return p


def calibration_panel(x, features, fit_population, fit_rows, calibration, audit, output):
    # Two exact, label-free screens. The helper's generic "audit" output
    # fields describe its candidate input, which is CALIBRATION in both calls.
    # In the second call the reference population is the fixed audit, not fit.
    keep_fit, excluded_fit, first = ablation.ablation.collision_panel(
        x, features, fit_population, fit_rows, calibration, output/'fit-vectors.sqlite')
    keep, excluded_audit, second = ablation.ablation.collision_panel(
        x, features, audit, audit, keep_fit, output/'audit-vectors.sqlite')
    require(len(keep) > 0, 'No disjoint calibration rows remain')
    files = {}
    for name, rows in [('calibration-rows.npy', keep), ('excluded-fit.npy', excluded_fit),
                        ('excluded-audit.npy', excluded_audit)]:
        np.save(output/name, rows.astype('<i8'), allow_pickle=False)
        files[name] = digest(output/name)
    return keep, dict(against_fit=first, against_audit=second, files=files,
                     retained_rows=len(keep), original_rows=len(calibration),
                     excluded_fit=len(excluded_fit), excluded_audit=len(excluded_audit))


def verify_panel(p, ref):
    require(p['original_rows'] == 409919 and p['retained_rows'] > 0
            and p['retained_rows']+p['excluded_fit']+p['excluded_audit'] == p['original_rows']
            and set(p['files']) == {'calibration-rows.npy', 'excluded-fit.npy', 'excluded-audit.npy'}
            and set(p['classes']) == set(rules.CLASSES)
            and sum(p['classes'].values()) == p['retained_rows']
            and all(type(p['classes'][c]) is int and 0 < p['classes'][c] <= ref['split']['partitions']['calibration']['classes'][c]
                    for c in rules.CLASSES), 'Calibration cohort changed')
    for scan, population, sample, original, excluded in [
        (p['against_fit'], 1229859, 270326, 409919, p['excluded_fit']),
        (p['against_audit'], 408027, 408027, 409919-p['excluded_fit'], p['excluded_audit'])]:
        require(scan['fit_population_rows'] == population and scan['fit_sample_rows'] == sample
                and scan['original_audit_rows'] == original and scan['excluded_rows'] == excluded
                and scan['retained_rows'] == original-excluded and type(excluded) is int and excluded >= 0
                and scan['drop_features'] == ablation.ablation.ARMS[prior.ARM]
                and scan['labels_used_for_membership'] is False and scan['split_reassigned'] is False
                and 0 < scan['joint_projection_unique_fit_keys'] <= population
                and scan['matches_to_sampled_fit_rows'] >= 0 and scan['matches_only_to_unsampled_fit_rows'] >= 0
                and scan['matches_to_sampled_fit_rows']+scan['matches_only_to_unsampled_fit_rows'] == excluded,
                'Collision accounting changed')


def freeze_content(r):
    # The file is written before any new audit probabilities or labels are
    # scored. It contains all selection inputs, but no new audit metrics.
    return dict(status='decision_frozen_before_audit_scoring', identity=r['identity'],
        calibration_panel=r['calibration_panel'], calibration=r['calibration'], selection=r['selection'])


def audit_record(r):
    require(r['status'] == 'complete' and all(r[k] is False for k in FALSE_FLAGS)
            and all(r[k] is True for k in TRUE_FLAGS), 'Wrong result scope')
    validate_plan(r['identity']['plan'])
    ref = frozen.read(REFERENCE)
    prior.audit_record(ref)
    sources = dict(ref['identity']['source_sha256'])
    sources.update({n: digest(ROOT/n) for n in SOURCES})
    require(r['identity']['source_sha256'] == sources and r['identity']['plan_sha256'] == digest(PLAN)
            and r['identity']['runtime'] == ref['identity']['runtime']
            and r['validation_panel'] == ref['validation_panel'] and r['split'] == ref['split']
            and r['audit_panel'] == ref['common_panel'] and r['new_model_count'] == 0,
            'Identity, cohort or work changed')
    verify_panel(r['calibration_panel'], ref)
    pop = ref['models'][0]['fit_population_counts']
    require(r['fit_population_counts'] == pop and [v['seed'] for v in r['calibration']] == rules.SEEDS,
            'Fitting prior or seed coverage changed')
    for case in r['calibration']:
        require(set(case['scores']) == {v['id'] for v in rules.candidates()}
                and set(case['probability_sha256']) == set(rules.METHODS), 'Candidate coverage changed')
        for score in case['scores'].values():
            prior.prior.verify_score(score, r['calibration_panel']['classes'], pop)
    require(r['selection'] == rules.select(r['calibration']), 'Calibration selection changed')
    require([v['seed'] for v in r['audit']] == rules.SEEDS, 'Missing audit seed')
    for case, old in zip(r['audit'], ref['cases']):
        require(set(case['scores']) == {'baseline', 'equal_raw', 'selected'}
                and set(case['probability_sha256']) == set(rules.METHODS), 'Unplanned audit search')
        for score in case['scores'].values():
            prior.prior.verify_score(score, ref['common_panel']['retained_classes'], pop)
        require(case['scores']['baseline'] == old['audit']['sqrt_population']
                and case['scores']['equal_raw'] == old['audit']['equal_class'], 'Historical audit bridge differs')
        if r['selection']['fallback_to_baseline']:
            require(case['scores']['selected'] == case['scores']['baseline'], 'Fallback changed predictions')
    require(r['summary'] == rules.summarize(r['audit'], r['selection']), 'Summary changed')
    return dict(status='verified', new_models_fitted=0, candidate_rules=10,
        eligible_alternatives=r['selection']['eligible_alternatives'],
        selected_rule=r['selection']['selected_rule']['id'],
        continuation_screen_passed=r['summary']['continuation_screen_passed'], test_opened=False)


def verify_output(output, r):
    require(digest(output/'decision-frozen.json') == r['decision_frozen_sha256']
            and frozen.read(output/'decision-frozen.json') == freeze_content(r)
            and frozen.read(output/'calibration-panel.json') == r['calibration_panel'], 'Frozen decision changed')
    for name, sha in r['calibration_panel']['files'].items():
        require(Path(name).name == name and digest(output/name) == sha, 'Calibration membership changed')


def run(cache, packed, panel, trees, balanced, output):
    cache, packed, panel, trees, balanced, output = map(Path, (cache, packed, panel, trees, balanced, output))
    require(not output.exists(), 'Preserve existing output; use a fresh directory')
    p, ref, tree_ref = frozen.read(PLAN), frozen.read(REFERENCE), frozen.read(prior.prior.REFERENCE)
    validate_plan(p)
    prior.audit_record(ref)
    ablation.verify_output(trees, tree_ref)
    prior.verify_output(balanced, ref)
    require(digest(trees/'receipt.json') == tree_ref['source_receipt_sha256']
            and digest(balanced/'receipt.json') == ref['source_receipt_sha256'], 'Wrong frozen model roots')
    cache_ref = frozen.read(expert.previous.PREVIOUS)
    expert.previous.verify_private(cache, cache_ref)
    runtime, parent = frozen.configure('cpu'), frozen.load_plan()
    require(runtime == ref['identity']['runtime'], 'Use historical local CPU runtime')
    manifest = frozen.verify_pack(packed, parent)
    sources = dict(ref['identity']['source_sha256'])
    sources.update({n: digest(ROOT/n) for n in SOURCES})
    identity = dict(plan=p, plan_sha256=digest(PLAN), source_sha256=sources, runtime=runtime)
    started = time.perf_counter()
    with frozen.session(output, identity, resume=False), threadpool_limits(limits=2):
        r = dict(status='preparing', identity=identity, new_model_count=0,
            audit_panel=ref['common_panel'], validation_panel=ref['validation_panel'], split=ref['split'],
            **dict.fromkeys(FALSE_FLAGS, False), **dict.fromkeys(TRUE_FLAGS, True))
        frozen.save(output/'receipt.json', r)
        with headroom.validation_only(packed, manifest) as (x, y):
            features = manifest['features']
            keep, info = headroom.load_panel(panel, SimpleNamespace(root=packed, arrays={'val': (x, y)}, manifest=manifest))
            require(info == ref['validation_panel'], 'Validation cohort changed')
            rows = np.flatnonzero(keep)
            split, chosen, keys = [np.load(cache/n, allow_pickle=False)
                                   for n in ('partitions.npy', 'fit_rows.npy', 'group_keys.npy')]
            expert.previous.fusion.check_split(rows, split, chosen, keys)
            audit = np.load(trees/'retained_rows.npy', allow_pickle=False)
            require(np.array_equal(audit, rows[split == 2]) and len(audit) == 408027, 'Audit population changed')
            calibration, cp = calibration_panel(x, features, rows[split == 0], rows[chosen], rows[split == 1], audit, output)
            # Membership is already fixed without labels. Counts are recorded
            # now only to require all eight classes and audit the denominators.
            cp['classes'] = ablation.counts(y[calibration])
            verify_panel(cp, ref)
            frozen.save(output/'calibration-panel.json', cp)
            r['calibration_panel'] = cp
            pop = np.bincount(y[rows[split == 0]], minlength=8)
            require(pop.tolist() == ref['models'][0]['fit_population_counts'], 'Fitting prior changed')
            r['fit_population_counts'] = pop.tolist()
            models = {}
            for seed in rules.SEEDS:
                a = next(m for m in tree_ref['models'] if m['seed'] == seed and m['arm'] == prior.ARM)
                b = next(m for m in ref['models'] if m['seed'] == seed)
                models[seed] = dict(sqrt_population=expert.load_owned(trees, a), equal_class=expert.load_owned(balanced, b))
            r['calibration'] = []
            for seed in rules.SEEDS:
                case = dict(seed=seed, scores={}, probability_sha256={})
                for method, model in models[seed].items():
                    probabilities = probability_rows(model, x, calibration, features)
                    case['probability_sha256'][method] = ablation.hash_vector(probabilities)
                    for rule in rules.candidates():
                        if rule['method'] == method:
                            pred = rules.predict(probabilities, pop, rule['exponent'])
                            case['scores'][rule['id']] = prior.prior.scored(y[calibration], pred, pop)
                r['calibration'].append(case)
                print(f'CALIBRATION: seed={seed}; fixed 10-rule comparison complete', flush=True)
            r['selection'] = rules.select(r['calibration'])
            frozen.save(output/'decision-frozen.json', freeze_content(r))
            r['decision_frozen_sha256'] = digest(output/'decision-frozen.json')
            print(f'FROZEN DECISION: {r["selection"]["selected_rule"]}; eligible alternatives={r["selection"]["eligible_alternatives"]}', flush=True)
            r['audit'] = []
            chosen_rule = r['selection']['selected_rule']
            for seed in rules.SEEDS:
                case = dict(seed=seed, scores={}, probability_sha256={})
                for method, name in zip(rules.METHODS, ('baseline', 'equal_raw')):
                    probabilities = probability_rows(models[seed][method], x, audit, features)
                    case['probability_sha256'][method] = ablation.hash_vector(probabilities)
                    case['scores'][name] = prior.prior.scored(y[audit], rules.predict(probabilities, pop, 0.), pop)
                    if method == chosen_rule['method']:
                        case['scores']['selected'] = prior.prior.scored(y[audit], rules.predict(probabilities, pop, chosen_rule['exponent']), pop)
                r['audit'].append(case)
                print(f'AUDIT: seed={seed}; frozen rule and two historical controls scored', flush=True)
            r['summary'] = rules.summarize(r['audit'], r['selection'])
        expert.previous.verify_private(cache, cache_ref)
        ablation.verify_output(trees, tree_ref)
        prior.verify_output(balanced, ref)
        require(frozen.verify_pack(packed, parent) == manifest
                and digest(panel/'receipt.json') == info['receipt_sha256']
                and all(digest(panel/n) == h for n, h in info['files'].items()), 'Source inputs changed')
        verify_output(output, r)
        r.update(status='complete', elapsed_seconds=time.perf_counter()-started)
        audit_record(r)
        frozen.save(output/'receipt.json', r)
    return audit_record(r)


def publish(first, replay, destination):
    first, replay, destination = map(Path, (first, replay, destination))
    require(first.resolve() != replay.resolve() and not destination.exists(), 'Require independent replay and fresh publication')
    a, b = [frozen.read(p/'receipt.json') for p in (first, replay)]
    for path, record in ((first, a), (replay, b)):
        audit_record(record)
        verify_output(path, record)
    require(expert.without_timing(a) == expert.without_timing(b), 'Non-timing replay differs')
    frozen.save(destination, dict(a, non_timing_replay_exact=True,
        source_receipt_sha256=digest(first/'receipt.json'), replay_receipt_sha256=digest(replay/'receipt.json')))
    return audit_record(a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run_parser = sub.add_parser('run')
    for key in ('cache', 'packed', 'panel', 'trees', 'balanced', 'output'):
        run_parser.add_argument('--'+key, type=Path, required=True)
    pub = sub.add_parser('publish')
    for key in ('first', 'replay', 'destination'):
        pub.add_argument('--'+key, type=Path, required=True)
    sub.add_parser('audit').add_argument('--record', type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    result = run(**args) if command == 'run' else publish(**args) if command == 'publish' else audit_record(frozen.read(args['record']))
    import json
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
