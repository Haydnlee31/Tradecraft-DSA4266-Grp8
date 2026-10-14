"""Build a private, local, read-only preview from frozen validation caches.

No ground-truth array is opened. Historical aggregate outcomes are displayed
separately; they never select thresholds, review flags, or preview examples.
Every prior review scenario is replayed, not just a favourable seed or rule.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts import official39_policy_tradeoffs as policy
from src.eval import triage_preview as view
from src.eval.official_evidence import digest, require

EVIDENCE = ROOT/'reports/risk_aware_routing/tradeoff-results.json'
EVIDENCE_SHA256 = '9bd7c68af8492e62704af6b6921743d4c540b763e77ed0520a4f81bafdf466ac'
SOURCES = ('scripts/official39_triage.py', 'src/eval/triage_preview.py', 'src/eval/triage_preview.html')
FALSE_FLAGS = ('ground_truth_rows_opened', 'model_trained', 'neural_inference_performed',
               'test_opened', 'test_evaluated', 'cloud_used', 'external_api_called',
               'classification_labels_changed', 'human_review_performed', 'automatic_blocking',
               'operational_budget_approved')


def load_evidence():
    require(digest(EVIDENCE) == EVIDENCE_SHA256, 'Reviewed trade-off evidence changed')
    record = policy.frozen.read(EVIDENCE)
    policy.audit(record)
    return record


def replay_pair(seed, lane, rows, predictions, probabilities, scores, evidence_pair):
    """Only cached inference-available values decide review status.

    The historical labels are absent from this interface. Aggregate outcomes
    are copied after the independently replayed flag count passes its bridge.
    """
    require((seed, lane) == (evidence_pair['seed'], evidence_pair['lane']), 'Unpaired evidence')
    require(probabilities.shape == (len(rows), len(view.CLASSES)) and probabilities.dtype == np.float32
            and np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1))
            and np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-6), 'Invalid saved probabilities')
    require(set(scores) == set(view.METHODS), 'Missing score controls')
    before = policy.prior.vector_hash(predictions)
    result = dict(seed=seed, lane=lane,
        base_metrics={k: evidence_pair['base']['metrics'][k] for k in ('macro_f1', 'benign_false_alert_rate', 'per_class')},
        scenarios=[])
    checks = []
    require([(r['rule'], r['budget_percent']) for r in evidence_pair['reviews']] ==
            [(m, b) for m in view.METHODS for b in (1, 5, 10)], 'Incomplete review policies')
    for old in evidence_pair['reviews']:
        method, budget = old['rule'], old['budget_percent']
        preview, flags = view.preview_rows(rows, predictions, probabilities.max(1), scores[method], old['threshold'], method)
        work = view.workload(flags, budget)
        require(work['referrals'] == old['calls'] and work['rows'] == old['rows']
                and work['referral_rate'] == old['review_rate'] and work['budget_respected'] == old['budget_respected'],
                'Review count does not match frozen evidence')
        require(policy.prior.vector_hash(predictions) == before, 'Classifier predictions changed')
        result['scenarios'].append(dict(rule=method, budget_percent=budget, threshold=old['threshold'],
            workload=work, preview=preview,
            historical_review={k: old[k] for k in ('base_errors_referred', 'benign_referrals', 'per_class')}))
        checks.append(dict(rule=method, budget_percent=budget, threshold=old['threshold'],
            referrals=work['referrals'], budget_respected=work['budget_respected'], over_cap=work['over_cap'],
            flags_sha256=policy.prior.vector_hash(flags), historical_count_exact=True))
    return result, dict(seed=seed, lane=lane, prediction_sha256=before, predictions_unchanged=True, scenarios=checks)


def run(router_run, packed, panel, output):
    root, packed, panel, output = map(Path, (router_run, packed, panel, output))
    # Private rows belong only in ignored outputs, never reports or the repo
    # root. Resolve symlinks first and reject inputs/ancestors as destinations.
    output = output.resolve()
    require(output.is_relative_to((ROOT/'outputs').resolve()) and output != (ROOT/'outputs').resolve(),
            'Private preview output must be beneath repository outputs/')
    require(not output.exists(), 'Preserve existing preview; use a fresh output path')
    evidence, previous = load_evidence(), policy.frozen.read(policy.PREVIOUS)
    policy.verify_private(root, previous)
    runtime = policy.frozen.configure('cpu')
    require(runtime == evidence['identity']['runtime'], 'Use recorded local CPU runtime')
    source_hashes = {n: digest(ROOT/n) for n in SOURCES}
    parent = policy.frozen.load_plan()
    manifest = policy.frozen.verify_pack(packed, parent)
    info = evidence['validation_panel']
    require(digest(panel/'receipt.json') == info['receipt_sha256']
            and all(digest(panel/n) == h for n, h in info['files'].items()), 'Panel identity changed')
    # This allowlist is deliberately val_x only: not even hashing val_y opens
    # ground-truth rows in this workflow. Packed metadata supplies its schema.
    require(digest(packed/'val_x.npy') == manifest['files']['val_x.npy'], 'Validation features changed')
    with ExitStack() as stack:
        def mmap(path):
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            stack.callback(a._mmap.close)
            return a
        retained = mmap(panel/'retained_rows.npy')
        split = mmap(root/'partitions.npy')
        x = mmap(packed/'val_x.npy')
        require(retained.dtype == np.int64 and retained.shape == (info['retained_rows'],)
                and np.all(retained[1:] > retained[:-1]) and retained[0] >= 0 and retained[-1] < manifest['val_rows']
                and split.dtype == np.uint8 and split.shape == retained.shape and np.isin(split, [0, 1, 2]).all()
                and x.dtype == np.float32 and x.shape == (manifest['val_rows'], 39), 'Invalid cached input schema')
        indices = np.flatnonzero(split == 2)
        require(policy.prior.vector_hash(indices) == evidence['split']['partitions']['audit']['indices_sha256'], 'Audit membership changed')
        rows = np.array(retained[indices], copy=True)
        payload = dict(protocol='official39-triage-preview-v1', read_only=True,
            evidence_sha256=EVIDENCE_SHA256, classes=view.CLASSES, ground_truth_rows_opened=False,
            operational_budget_approved=False, sample_rule='first eight flagged and eight unflagged, source order, no labels', pairs=[])
        checks = dict(protocol=payload['protocol'], status='preparing', read_only=True,
            scope='Local read-only replay of previously reviewed validation policies, not live inference or confirmation',
            evidence_sha256=EVIDENCE_SHA256, source_sha256=source_hashes, rows_per_pair=len(rows), pairs=[], **dict.fromkeys(FALSE_FLAGS, False))
        for pair, old in zip(previous['prepared_pairs'], evidence['pairs']):
            seed, lane = pair['seed'], pair['lane']
            with ExitStack() as cache:
                probs = np.load(root/f'seed{seed}-{lane}-probabilities.npy', mmap_mode='r', allow_pickle=False)
                cache.callback(probs._mmap.close)
                require(probs.shape == (manifest['val_rows'], len(view.CLASSES)) and probs.dtype == np.float32, 'Wrong probability cache')
                selected = np.array(probs[rows], copy=True)
                predictions = policy.prior.get_predictions(root, seed, lane, rows)
                scores = dict(confidence=1-selected.max(1),
                    learned=policy.prior.score_rows(policy.frozen.read(root/pair['gate']), x, probs, rows),
                    random=policy.prior.random_scores(len(retained), previous['identity']['plan']['random_seed']+seed)[indices])
                shown, checked = replay_pair(seed, lane, rows, predictions, selected, scores, old)
                payload['pairs'].append(shown)
                checks['pairs'].append(checked)
                print(f'PASS: {lane} seed={seed}; nine frozen review policies exact; labels unchanged', flush=True)
    # Verify immutable inputs again before writing the new private artifact.
    policy.verify_private(root, previous)
    load_evidence()
    require(all(digest(ROOT/n) == h for n, h in source_hashes.items())
            and digest(packed/'val_x.npy') == manifest['files']['val_x.npy']
            and digest(packed/'manifest.json') == parent['data_identity']['packed_manifest_sha256']
            and digest(panel/'receipt.json') == info['receipt_sha256']
            and all(digest(panel/n) == h for n, h in info['files'].items()), 'Inputs/source changed during replay')
    html = view.render(payload)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('preview.json', payload), ('index.html', html)]:
        with (output/name).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False)+'\n')
    checks.update(status='complete', policies_verified=sum(len(p['scenarios']) for p in checks['pairs']),
                  files={n: digest(output/n) for n in ('preview.json', 'index.html')})
    with (output/'checks.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(checks, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--router-run', type=Path, default=ROOT/'outputs/official39-routing-gate-v2')
    parser.add_argument('--packed', type=Path, default=ROOT/'outputs/official39-packed-2m-v1')
    parser.add_argument('--panel', type=Path, default=ROOT/'outputs/official39-ablation-panel-v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.router_run, args.packed, args.panel, args.output)
    print(f'COMPLETE: {result["policies_verified"]} policies. Open {(args.output/"index.html").resolve()}')


if __name__ == '__main__':
    main()
