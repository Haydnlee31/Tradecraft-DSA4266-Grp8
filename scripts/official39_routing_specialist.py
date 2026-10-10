"""A frozen specialist comparison, not another training or threshold search.

Every base model, validation row, confidence rank, random selection and budget
is shared with stage one. Only the fallback changes. Always-fallback and
label-informed oracle results are kept separate from actual budgeted routing.
The final test, paid APIs and training routines are not part of this runner.
"""
import argparse
from dataclasses import replace
import hashlib
import io
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts import official39_routing_headroom as headroom
from src.data.label_map import CLASSES
from src.eval.official_evidence import describe, digest, require
from src.eval.official_training_fit import state_hash
from src.eval.official_panel_validation import validation_batches
from src.models.official_streaming import evaluate

frozen = headroom.frozen
PLAN = ROOT / 'reports/risk_aware_routing/specialist-plan.json'
PREVIOUS = ROOT / 'reports/risk_aware_routing/headroom-results.json'
FALLBACKS = ('centralized_full_cohort', 'balanced_panel')
CONFIG = dict(name='centralized_light', hidden_dims=[64, 32], dropout=.2, normalization='layer')


def validate_plan(plan):
    expected = dict(protocol='official39-routing-specialist-v1', seeds=[7, 17, 27],
                    specialist_condition='small_dropout', specialist_epochs=300,
                    specialist_historical_batch_size=8192, base_lanes=['iid', 'dirichlet'],
                    fallbacks=list(FALLBACKS), budgets_percent=[1, 5, 10],
                    controls=['lowest_base_confidence', 'random_same_budget'], device='cpu', batch_size=512)
    require(all(plan.get(k) == v for k, v in expected.items()), 'Unreviewed specialist scope')
    require(all(plan.get(k) is False for k in headroom.FLAGS), 'No training, test or external API permission')
    require(plan['headroom_plan_sha256'] == digest(headroom.PLAN)
            and plan['headroom_results_sha256'] == digest(PREVIOUS)
            and plan['headroom_runner_sha256'] == digest(headroom.__file__), 'Stage-one evidence changed')
    require([s['seed'] for s in plan['specialists']] == plan['seeds']
            and all(s['checkpoint'] == f'small_dropout_seed{s["seed"]}.pt' for s in plan['specialists']),
            'Require three named specialist checkpoints')
    require(plan['specialist_receipt_sha256'] ==
            'b5767472daeb19218e2a704c1c3d6edf54a129c96ac8da0a88f8d20698c6df99', 'Unreviewed specialist receipt')


def specialist_reference(root, plan, manifest):
    """Pin the original training/evaluation record before loading any weights."""
    path = Path(root) / 'receipt.json'
    require(digest(path) == plan['specialist_receipt_sha256'], 'Specialist receipt checksum changed')
    r = frozen.read(path)
    require(r['status'] == 'complete' and r['all_training_bridges_exact'] is True
            and r['test_opened'] is False and r['model_promoted'] is False,
            'Incomplete or wrong-scope specialist reference')
    require(r['packed_manifest_sha256'] == plan['packed_manifest_sha256']
            and r['panel_sha256'] == plan['specialist_training_panel_sha256']
            and r['validation_file_sha256'] == {n: manifest['files'][n] for n in ('val_x.npy', 'val_y.npy')},
            'Specialist preprocessing/data cohort differs')
    verify_specialist_sources(r, plan)
    return r


def verify_specialist_sources(reference, plan):
    # The training wrapper gained opt-in ablation support after the specialist
    # run. Its evaluate function is byte-identical. Allow only this reviewed,
    # pinned bridge; do not silently waive whole-file provenance failures.
    bridge = plan['reviewed_evaluator_bridge']
    for name in ('models/architectures.py', 'eval/official_training_fit.py',
                 'eval/official_panel_validation.py', 'models/official_streaming.py'):
        if name == 'models/official_streaming.py':
            require(bridge['file'] == name and reference['source_sha256'][name] == bridge['historical_sha256']
                    and digest(ROOT/'src'/name) == bridge['current_sha256']
                    and hashlib.sha256(inspect.getsource(evaluate).encode()).hexdigest() == bridge['evaluate_source_sha256'],
                    'Unreviewed specialist evaluator bridge')
        else:
            require(digest(ROOT/'src'/name) == reference['source_sha256'][name], 'Specialist source identity changed')


def load_specialist(root, spec, reference, features):
    """Restricted loading of one small, hash-pinned final checkpoint.

    Checkpoint configuration alone is not trusted: compare it to the pinned
    receipt, schema, epoch, seed, architecture and final tensor digest too.
    """
    require(spec['checkpoint'] == f'small_dropout_seed{spec["seed"]}.pt', 'Unsafe or wrong checkpoint name')
    entries = [f for f in reference['neural_fits']
               if f['condition'] == 'small_dropout' and f['training_fit']['seed'] == spec['seed']]
    require(len(entries) == 1, 'Missing or duplicated specialist seed')
    entry, path = entries[0], Path(root)/spec['checkpoint']
    require(entry['checkpoint'] == spec['checkpoint'] and entry['checkpoint_sha256'] == spec['sha256'],
            'Checkpoint differs from the pinned receipt')
    raw = path.read_bytes()
    require(len(raw) < 1_000_000 and hashlib.sha256(raw).hexdigest() == spec['sha256'], 'Checkpoint checksum changed')
    checkpoint = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    config = dict(checkpoint['model_config'])
    config['hidden_dims'] = list(config['hidden_dims'])
    fit = entry['training_fit']
    require(config == CONFIG == fit['model_config'] and checkpoint['classes'] == CLASSES
            and checkpoint['features'] == features and checkpoint['epochs'] == 300
            and checkpoint['seed'] == spec['seed'] and checkpoint['panel_sha256'] == reference['panel_sha256']
            and fit['num_parameters'] == 5096 and fit['examples_processed'] == 3045600
            and fit['optimizer_steps'] == 6000 and fit['observations'][-1]['epoch'] == 300,
            'Specialist metadata/budget changed')
    model = frozen.MLPClassifier(39, len(CLASSES), replace(frozen.config_for_variant('light'), normalization='layer'))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    require(model.num_parameters() == 5096 and state_hash(model) == fit['final_state_sha256'],
            'Specialist tensor state changed')
    old = [v for v in reference['validation_results'] if v['name'] == entry['name']]
    require(len(old) == 1 and old[0]['seed'] == spec['seed'], 'Missing historical specialist evaluation')
    return model, old[0]


def compare_pairs(control, treatment):
    """Prove the same gate called a different expert on exactly the same rows."""
    for key in ('seed', 'base_lane', 'rows', 'base'):
        require(control[key] == treatment[key], 'Base cohort/seed/predictions changed')
    require(len(control['controls']) == len(treatment['controls']) == 6, 'Missing budgeted controls')
    result = []
    for left, right in zip(control['controls'], treatment['controls']):
        for key in ('rule', 'budget_percent', 'escalated_rows', 'selected_rows_sha256'):
            require(left[key] == right[key], 'Routing selections differ between fallbacks')
        for c in CLASSES:
            require(left['per_class_routing'][c]['escalated'] == right['per_class_routing'][c]['escalated'],
                    'Per-class routing population changed')
        result.append(dict(rule=left['rule'], budget_percent=left['budget_percent'],
            macro_f1=right['metrics']['macro_f1']-left['metrics']['macro_f1'],
            benign_false_alert_rate=right['metrics']['benign_false_alert_rate']-left['metrics']['benign_false_alert_rate'],
            per_class_recall={c: right['metrics']['per_class'][c]['recall']-left['metrics']['per_class'][c]['recall']
                              for c in CLASSES},
            corrected_rows=right['corrected_rows']-left['corrected_rows'],
            damaged_rows=right['damaged_rows']-left['damaged_rows']))
    return dict(seed=control['seed'], base_lane=control['base_lane'], controls=result)


def summarize_comparison(pairs, previous):
    require(len(pairs) == 12, 'Require both fallbacks for six seed/lane pairs')
    grouped = {key: [p for p in pairs if p['fallback'] == key] for key in FALLBACKS}
    summaries = {key: headroom.summarize(value) for key, value in grouped.items()}
    require(summaries[FALLBACKS[0]] == previous['summary'], 'Stage-one aggregate replay differs')
    # Also compare the finer per-seed complementarity counts, not just means.
    for pair in grouped[FALLBACKS[0]]:
        old = next(r for r in previous['pairs'] if (r['seed'], r['base_lane']) == (pair['seed'], pair['base_lane']))
        require(pair['per_class'] == old['per_class'] and pair['budget_correctness_bounds'] == old['budget_correctness_bounds'],
                'Stage-one paired counts differ')
    paired = [compare_pairs(a, b) for a, b in zip(grouped[FALLBACKS[0]], grouped[FALLBACKS[1]])]
    deltas = {}
    for lane in ('iid', 'dirichlet'):
        rows = [p for p in paired if p['base_lane'] == lane]
        deltas[lane] = []
        for i, control in enumerate(rows[0]['controls']):
            deltas[lane].append(dict(rule=control['rule'], budget_percent=control['budget_percent'],
                **{k: describe([r['controls'][i][k] for r in rows])
                   for k in ('macro_f1', 'benign_false_alert_rate', 'corrected_rows', 'damaged_rows')},
                per_class_recall={c: describe([r['controls'][i]['per_class_recall'][c] for r in rows]) for c in CLASSES}))
    return dict(fallbacks=summaries, balanced_minus_centralized=deltas,
                stage_one_replay_exact=True, identical_routing_selections=True)


def run(archives, packed, panel, specialists, output, plan_path=PLAN):
    require(not Path(output).exists(), 'Output exists; preserve evidence')
    plan = frozen.read(plan_path)
    validate_plan(plan)
    parent, previous, runtime = frozen.load_plan(), frozen.read(PREVIOUS), frozen.configure('cpu')
    require(runtime == previous['identity']['runtime'], 'Use the recorded local CPU runtime for the paired replay')
    manifest = frozen.verify_pack(packed, parent)
    plan_with_cohort = dict(plan, packed_manifest_sha256=parent['data_identity']['packed_manifest_sha256'])
    old_specialists = specialist_reference(specialists, plan_with_cohort, manifest)
    candidates = headroom.select_candidates(parent)
    sources = dict(previous['identity']['source_sha256'])
    require(all(digest(ROOT/k) == v for k, v in sources.items()), 'Stage-one shared source changed')
    for name in ('scripts/official39_routing_specialist.py', 'src/eval/official_training_fit.py',
                 'src/eval/official_panel_validation.py', 'src/models/official_streaming.py'):
        sources[name] = digest(ROOT/name)
    identity = dict(protocol=plan['protocol'], plan=plan, plan_sha256=digest(plan_path),
                    source_sha256=sources, runtime=runtime, **dict.fromkeys(headroom.FLAGS, False))
    started = time.perf_counter()
    with frozen.session(output, identity, resume=False) as folder:
        receipt = dict(status='incomplete', scope='exploratory_reused_validation_not_independent_test',
                       identity=identity, pairs=[], model_checks=[], **dict.fromkeys(headroom.FLAGS, False))
        frozen.save(folder/'receipt.json', receipt)
        with headroom.validation_only(packed, manifest) as (x, y):
            keep, info = headroom.load_panel(panel, SimpleNamespace(root=Path(packed), arrays={'val': (x, y)}, manifest=manifest))
            require(info == previous['identity']['validation_panel'], 'Stage-one validation population changed')
            receipt['validation_panel'] = info
            py = np.array(y[keep], copy=True)
            store = frozen.Artifacts(archives, candidates)
            try:
                for seed in plan['seeds']:
                    predictions, confidences = {}, {}
                    for lane in ('light', 'iid', 'dirichlet', 'balanced_panel'):
                        if lane == 'balanced_panel':
                            spec = next(s for s in plan['specialists'] if s['seed'] == seed)
                            model, historical = load_specialist(specialists, spec, old_specialists, manifest['features'])
                        else:
                            candidate = next(c for c in candidates if c['seed'] == seed and c['lane'] == lane)
                            model, env, _ = store.load(candidate, parent, 'cpu')
                            require(env['input_transform']['panel'] == info, 'Model validation population changed')
                        pred, conf = headroom.scan(model, x, y, manifest['features'])
                        reference = frozen.score(model, frozen.batches(x, y, manifest['features'], 'full39'), 'cpu', keep)
                        for yy, pp, population in ((y, pred, 'all'), (py, pred[keep], 'panel')):
                            require(headroom.metrics(yy, pp)['metrics']['confusion_matrix'] ==
                                    reference[population]['metrics']['confusion_matrix'], 'Independent CPU scan mismatch')
                        check = dict(lane=lane, seed=seed, model_state_sha256=reference['model_state_sha256'],
                                     prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),
                                     same_runtime_reference_exact=True, weights_unchanged=True)
                        if lane == 'balanced_panel':
                            # Reproduce the earlier all-validation endpoint at its
                            # original batch size before using the shared panel.
                            _, old_metrics = evaluate(model, validation_batches(x, y, 8192), 'cpu')
                            require(old_metrics['confusion_matrix'] == historical['validation_metrics']['confusion_matrix'],
                                    'Historical specialist endpoint did not reproduce')
                            require(state_hash(model) == next(f['training_fit']['final_state_sha256'] for f in
                                    old_specialists['neural_fits'] if f['name'] == f'small_dropout_seed{seed}'),
                                    'Historical reference pass changed weights')
                            check.update(checkpoint_sha256=spec['sha256'], historical_batch_replay_exact=True,
                                historical_to_current_batch_confusion_l1=int(np.abs(np.asarray(old_metrics['confusion_matrix'])-
                                    np.asarray(reference['all']['metrics']['confusion_matrix'])).sum()))
                        else:
                            old = next(c for c in previous['model_checks'] if c['candidate']['id'] == candidate['id'])
                            require(check['model_state_sha256'] == old['model_state_sha256'] and
                                    check['prediction_sha256'] == old['validation_prediction_sha256'], 'Stage-one model replay differs')
                            check['stage_one_replay_exact'] = True
                        receipt['model_checks'].append(check)
                        predictions[lane], confidences[lane] = pred[keep], conf[keep]
                        print(f'PASS: seed={seed} {lane}, frozen weights and validation replay', flush=True)
                        del model, pred, conf
                    for base in plan['base_lanes']:
                        for fallback, key in zip(FALLBACKS, ('light', 'balanced_panel')):
                            pair = headroom.pair_analysis(py, predictions[base], predictions[key], confidences[base], seed, plan['budgets_percent'])
                            pair.update(seed=seed, base_lane=base, expert_lane=key, fallback=fallback)
                            receipt['pairs'].append(pair)
                    frozen.save(folder/'receipt.json', receipt)
            finally:
                store.close()
        require(all(digest(ROOT/k) == v for k, v in sources.items()) and digest(plan_path) == identity['plan_sha256'],
                'Source or plan changed during run')
        require(digest(PREVIOUS) == plan['headroom_results_sha256']
                and digest(Path(specialists)/'receipt.json') == plan['specialist_receipt_sha256'], 'Reference changed')
        require(digest(Path(packed)/'manifest.json') == parent['data_identity']['packed_manifest_sha256']
                and digest(Path(packed)/'scaler.json') == parent['data_identity']['scaler_sha256']
                and digest(Path(panel)/'receipt.json') == info['receipt_sha256'], 'Input metadata changed')
        for name, value in info['files'].items():
            require(digest(Path(panel)/name) == value, 'Panel membership changed')
        for spec in plan['specialists']:
            require(digest(Path(specialists)/spec['checkpoint']) == spec['sha256'], 'Specialist weights changed')
        for name, value in {(c['artifact']['archive'], c['artifact']['archive_sha256']) for c in candidates}:
            require(digest(Path(archives)/name) == value, 'Archive changed')
        receipt.update(status='complete', summary=summarize_comparison(receipt['pairs'], previous),
                       elapsed_seconds=time.perf_counter()-started)
        frozen.save(folder/'receipt.json', receipt)
    return receipt


def publish(receipt, receipt_path, destination):
    """Aggregate-only handoff, with a hash of the private joint-count receipt."""
    require(receipt == frozen.read(receipt_path) and receipt['status'] == 'complete'
            and all(receipt[k] is False for k in headroom.FLAGS), 'Require complete development-only evidence')
    require(receipt['summary'] == summarize_comparison(receipt['pairs'], frozen.read(PREVIOUS)), 'Summary mismatch')
    export = {k: receipt[k] for k in ('status', 'scope', 'identity', 'validation_panel', 'model_checks', 'summary', *headroom.FLAGS)}
    export['source_receipt_sha256'] = digest(receipt_path)
    export['pairs'] = [dict(seed=p['seed'], base_lane=p['base_lane'], fallback=p['fallback'], rows=p['rows'],
        per_class=p['per_class'], controls=[{k: c[k] for k in ('rule', 'budget_percent', 'escalated_rows',
            'corrected_rows', 'damaged_rows', 'per_class_routing')} for c in p['controls']]) for p in receipt['pairs']]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(export, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archives', 'packed', 'panel', 'specialists', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--publish-summary', type=Path)
    args = parser.parse_args()
    require(args.publish_summary is None or not args.publish_summary.exists(), 'Summary exists; preserve evidence')
    result = run(args.archives, args.packed, args.panel, args.specialists, args.output)
    if args.publish_summary:
        publish(result, args.output/'receipt.json', args.publish_summary)
    print('COMPLETE: both fallbacks, all paired seeds; no training, final test or paid API')


if __name__ == '__main__':
    main()
