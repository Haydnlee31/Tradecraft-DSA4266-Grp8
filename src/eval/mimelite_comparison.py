"""Exact-work, development-only comparison including full-anchor MimeLite.

Historical kernels and records stay unchanged. A shorter final local block is
explicit in the schedule; anchors are never hidden, truncated, or discarded.
Resume accepts only the same inputs, settings, code and strict CPU runtime.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.eval import methodology_comparison as parent
from src.federated import mime_lite
from src.models.research import atomic_json, atomic_save

old = parent.old
PLAN = old.ROOT/'reports/mimelite_fair_comparison/plan.json'
LABELS = {'central_adam':'Central Adam', 'shared_adam':'Shared Adam',
          'mimelite5':'MimeLite (5 steps)', 'mimelite20':'MimeLite (20 steps)',
          'mime':'Full MIME (20 steps)', 'fedavg_adam':'FedAvg-Adam (20 steps)',
          'central_mime_base':'Central MIME base'}


def read_plan():
    p = json.loads(PLAN.read_text())
    old.require(set(p['arms']) == set(LABELS) and p['seeds'] == [7,17,27], 'Incomplete comparison')
    old.require(p['total_training_backward_examples'] == len(p['arms'])*len(p['seeds'])*p['backward_examples_per_run'], 'Budget differs')
    return p


def contract():
    return {'plan_sha256':old.sha256(PLAN), 'runner_sha256':old.sha256(__file__),
            'mimelite_sha256':old.sha256(mime_lite.__file__), 'parent':parent.contract(),
            'references_sha256':old.sha256(old.ROOT/'reports/methodology_comparison/results.json')}


def is_lite(arm):
    return arm in ('mimelite5','mimelite20')


def schedule(base, arm, settings, budget):
    """Return local-step counts per unit; refuse budgets requiring fake padding."""
    old.require(type(budget) is int and budget > 0 and arm in LABELS, 'Invalid schedule')
    b, c = base['batch_size'], base['clients']
    if arm.startswith('central_') or arm == 'shared_adam':
        n, rem = divmod(budget, b if arm.startswith('central_') else b*c)
        old.require(rem == 0, 'Nonintegral minibatch budget')
        return [1]*n
    k = settings['local_steps']
    old.require(type(k) is int and k > 0, 'Invalid local block')
    anchor = base['fit_rows'] if arm == 'mime' or is_lite(arm) else 0
    per_step = b*c*(2 if arm == 'mime' else 1)
    n, rem = divmod(budget, anchor+k*per_step)
    steps = [k]*n
    if rem:
        final, extra = divmod(rem-anchor, per_step)
        old.require(extra == 0 and 0 < final < k, 'Budget cannot fund a valid final round')
        steps.append(final)
    old.require(bool(steps), 'Empty schedule')
    return steps


def make_state(base, arm, lr):
    return old.training.make_state(base, 'mime' if is_lite(arm) else arm, lr)


def restore(saved, base, arm, lr):
    return old.training.restore(saved, base, 'mime' if is_lite(arm) else arm, lr)


def advance(state, data, base, arm, settings, steps, criterion):
    lr = settings['lr']
    # Reuse the exact historical path wherever its fixed local block applies.
    if not is_lite(arm) and (arm.startswith('central_') or arm == 'shared_adam' or
                            steps == base['mime']['local_steps']):
        old.training.advance(state, data, base, arm, lr, criterion)
        return
    model, u, w = state['model'], state['unit'], state['work']
    model.train()
    counts = list(map(len, data.parts))
    c, b = len(counts), base['batch_size']

    def local_batch(client, step):
        rows = old.training.sampled_rows(data, base, u, client, step)
        old.training.record_rows(state, rows)
        return (*data.batch(rows), base['seed'])

    def anchors(client):
        # FitData verifies disjoint exact client ownership. Stream each assigned
        # row once, including the short final batch, at the starting global model.
        owned = data.parts[client]
        for start in range(0, len(owned), base['anchor_batch_size']):
            rows = owned[start:start+base['anchor_batch_size']]
            old.training.record_rows(state, rows)
            yield (*data.batch(rows), base['seed'])

    if arm == 'fedavg_adam':
        deltas = []
        for client in range(c):
            local = copy.deepcopy(model)
            opt = old.training.shared.make_optimizer(local, {**base['adam'], 'lr':lr})
            for step in range(steps):
                x, y, seed = local_batch(client, step)
                g = old.training.shared.client_gradient(local, (x,y), criterion, seed)
                old.training.shared.apply_gradient(local, opt, g)
            deltas.append(tuple(p.detach()-q.detach() for p,q in zip(local.parameters(), model.parameters())))
        with torch.no_grad():
            for p, delta in zip(model.parameters(), old.training.shared.aggregate_gradients(deltas, counts)):
                p.add_(delta)
        w['backward_examples'] += c*steps*b
        w['local_examples'] += c*steps*b
        w['backward_batches'] += c*steps
        w['optimizer_steps'] += c*steps
        w['estimated_payload_bytes'] += 2*c*model.parameter_bytes()
        w['logical_barriers'] += 1
    else:
        kernel = mime_lite.lite_round if is_lite(arm) else old.training.mime.full_round
        model, moments, receipt = kernel(model, state['moments'], counts, anchors, local_batch,
                                        criterion, {**base['mime'], 'lr':lr}, steps)
        state.update(model=model, moments=moments)
        for key in ('backward_examples','backward_batches','anchor_examples','local_examples'):
            w[key] += receipt['work'][key]
        w['optimizer_steps'] += receipt['work']['local_updates']
        w['estimated_payload_bytes'] += receipt['estimated_payload_bytes']
        w['logical_barriers'] += 1 if is_lite(arm) else 2
    w['synchronization_rounds'] += 1
    state['unit'] += 1
    old.require(all(bool(torch.isfinite(p).all()) for p in state['model'].parameters()), 'Nonfinite model')


def check_work(state, base, arm, data, steps):
    """Independently derive cumulative work, including shortened final blocks."""
    u, c, b = state['unit'], base['clients'], base['batch_size']
    old.require(0 <= u <= len(steps), 'Round outside schedule')
    local_steps = sum(steps[:u])
    central = arm.startswith('central_')
    anchored = arm == 'mime' or is_lite(arm)
    anchors = u*base['fit_rows'] if anchored else 0
    local = 0 if central or arm == 'shared_adam' else local_steps*c*b
    backwards = u*b if central else u*c*b if arm == 'shared_adam' else anchors+local*(2 if arm == 'mime' else 1)
    batches = u if central else u*c if arm == 'shared_adam' else local_steps*c*(2 if arm == 'mime' else 1)
    if anchored:
        batches += u*sum((len(part)+base['anchor_batch_size']-1)//base['anchor_batch_size'] for part in data.parts)
    vectors = 6 if arm == 'mime' else 5 if is_lite(arm) else 2
    expected = {'backward_examples':backwards, 'backward_batches':batches,
                'optimizer_steps':u if central or arm == 'shared_adam' else local_steps*c,
                'anchor_examples':anchors, 'local_examples':local,
                'synchronization_rounds':0 if central else u,
                'logical_barriers':0 if central else u*(2 if arm == 'mime' else 1),
                'estimated_payload_bytes':0 if central else vectors*c*u*state['model'].parameter_bytes()}
    old.require(state['work'] == expected, f'Work counter mismatch: {arm}')


def score(model, x, y, calibration, alarm, path):
    logits = old.rep.infer(model, x, alarm['inference_batch_size'])
    np.save(path, logits, allow_pickle=False)
    result = old.alarms.evaluate_scores(logits, y, calibration, [100,500,1000,2000])
    old.replication.independent_points(logits, y, calibration, result['operating_points'])
    return result


def run_one(root, base, arm, settings, steps, data, criterion, identity, evaluation, references):
    path = root/f"seed{base['seed']}-{arm}"
    path.mkdir(exist_ok=True)
    job = {'contract':identity, 'seed':base['seed'], 'arm':arm, 'settings':settings, 'schedule':steps}
    if (path/'result.json').exists():
        r = json.loads((path/'result.json').read_text())
        old.require(r['job'] == job and r['checkpoint_sha256'] == old.sha256(path/'last.pt') and
                    r['logits_sha256'] == old.sha256(path/'logits.npy') and r['recovery_exact'], 'Completed output changed')
        print('REUSE verified completed run', path.name, flush=True)
        return r
    state, seconds = make_state(base, arm, settings['lr']), 0.
    if (path/'resume.pt').exists():
        saved = torch.load(path/'resume.pt', weights_only=True)
        old.require(saved['job'] == job, 'Resume contract changed')
        state = restore(saved['state'], base, arm, settings['lr'])
        seconds = saved['training_seconds']
    check_work(state, base, arm, data, steps)
    x, y, mask, alarm = evaluation
    for index in range(state['unit'], len(steps)):
        if index == len(steps)-2:
            atomic_save({'job':job, 'state':old.training.snapshot(state)}, path/'recovery.pt')
        start = time.perf_counter()
        advance(state, data, base, arm, settings, steps[index], criterion)
        seconds += time.perf_counter()-start
        # Preserve exact older endpoints as a numerical bridge, not a candidate
        # for model selection. Every new primary result uses the final budget.
        if not is_lite(arm) and state['work']['backward_examples'] == 21600000:
            metrics = score(state['model'], x, y, mask, alarm, path/'bridge-logits.npy')
            ref = next(r for r in references if r['seed'] == base['seed'] and r['arm'] == arm)
            checks = {'model_exact':old.state_digest(state['model']) == ref['model_sha256'],
                      'logits_exact':old.sha256(path/'bridge-logits.npy') == ref['development_logits_sha256'],
                      'metrics_exact':metrics == ref['development'], 'work_exact':state['work'] == ref['work'],
                      'samples_exact':state['sample_chain_sha256'] == ref['sample_chain_sha256']}
            old.require(all(checks.values()), f'Historical bridge failed: {checks}')
            atomic_json(checks, path/'bridge.json')
        if (index+1) % max(1,len(steps)//12) == 0 or index+1 == len(steps):
            check_work(state, base, arm, data, steps)
            atomic_save({'job':job, 'state':old.training.snapshot(state), 'training_seconds':seconds}, path/'resume.pt')
            print(f"{path.name} unit={index+1}/{len(steps)} work={state['work']['backward_examples']}", flush=True)
    check_work(state, base, arm, data, steps)
    saved = torch.load(path/'recovery.pt', weights_only=True)
    old.require(saved['job'] == job, 'Recovery identity differs')
    recovered = restore(saved['state'], base, arm, settings['lr'])
    before = recovered['work']['backward_examples']
    for index in range(recovered['unit'], len(steps)):
        advance(recovered, data, base, arm, settings, steps[index], criterion)
    old.require(old.block.nested_equal(old.training.snapshot(state), old.training.snapshot(recovered)), 'Recovery differs')
    metrics = score(state['model'], x, y, mask, alarm, path/'logits.npy')
    bridge = None if is_lite(arm) else json.loads((path/'bridge.json').read_text())
    old.require(bridge is None or all(bridge.values()), 'Historical bridge missing')
    atomic_save({'job':job, 'state':old.training.snapshot(state)}, path/'last.pt')
    result = {'job':job, 'arm':arm, 'seed':base['seed'], 'development':metrics,
              'work':state['work'], 'model_sha256':old.state_digest(state['model']),
              'initial_model_sha256':old.state_digest(old.training.initial_model(base)),
              'sample_chain_sha256':state['sample_chain_sha256'],
              'checkpoint_sha256':old.sha256(path/'last.pt'), 'logits_sha256':old.sha256(path/'logits.npy'),
              'recovery_exact':True, 'recovery_backward_examples':state['work']['backward_examples']-before,
              'historical_bridge':bridge, 'training_seconds':seconds}
    atomic_json(result, path/'result.json')
    primary = next(p['audit_metrics'] for p in metrics['operating_points'] if p['calibration']['target_basis_points']==500)
    print(f"DONE {path.name} audit_macro_f1={primary['macro_f1']:.6f} FAR={primary['benign_false_alert_rate']:.6f}", flush=True)
    return result


def train(data_root, output, resume=False):
    _, base, data, alarm, x, y, mask = parent.preflight(data_root)
    plan, identity = read_plan(), contract()
    schedules = {a:schedule(base,a,s,plan['backward_examples_per_run']) for a,s in plan['arms'].items()}
    root = Path(output)
    receipt = {'contract':identity, 'plan':plan, 'data_manifest_sha256':alarm['data_manifest_sha256'],
               'runtime':parent.portable.runtime_record()}
    if resume:
        old.require((root/'contract.json').exists(), 'No matching resumable output')
        previous = json.loads((root/'contract.json').read_text())
        old.require(all(previous[k] == receipt[k] for k in ('contract','plan','data_manifest_sha256')), 'Resume inputs/code changed')
        old.require({k:v for k,v in previous['runtime'].items() if k != 'git_commit'} ==
                    {k:v for k,v in receipt['runtime'].items() if k != 'git_commit'}, 'Resume runtime changed')
    else:
        old.require_fresh_output(root)
        atomic_json(receipt, root/'contract.json')
    references = json.loads((old.ROOT/'reports/methodology_comparison/results.json').read_text())['summary']['results']
    criterion = old.build_official_criterion('sqrt_weighted_ce',data.manifest['counts']['fit'],'fixed_train_mean')
    results = []
    for seed in plan['seeds']:
        for arm, settings in plan['arms'].items():
            atomic_json({'status':'training','seed':seed,'arm':arm,'completed':len(results)},root/'status.json')
            results.append(run_one(root,{**base,'seed':seed},arm,settings,schedules[arm],data,criterion,
                                   identity,(x,y,mask,alarm),references))
    old.require(contract() == identity, 'Source changed during training')
    old.require(sum(r['work']['backward_examples'] for r in results) == plan['total_training_backward_examples'], 'Total work differs')
    for seed in plan['seeds']:
        old.require(len({r['initial_model_sha256'] for r in results if r['seed']==seed}) == 1, 'Unpaired initialization')
    summary = {**receipt,'status':'complete','results':results, 'confirmation_evaluated':False,
               'original_validation_evaluated':False,'test_evaluated':False,
               'fit_rows':len(data.arrays['fit'][1]),'development_rows':len(y),
               'calibration_rows':int(mask.sum()),'audit_rows':int((~mask).sum())}
    atomic_json(summary,root/'summary.json')
    atomic_json({'status':'complete','runs':len(results),'test_evaluated':False},root/'status.json')
    print('PASS: all 21 exact-work runs complete; confirmation/test untouched',flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','train'))
    parser.add_argument('--data',type=Path,default=old.DATA)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.action == 'check':
        _,base,*_ = parent.preflight(args.data)
        plan = read_plan()
        print(json.dumps({a:{'units':len(ks),'final_local_steps':ks[-1]} for a,s in plan['arms'].items()
                          for ks in [schedule(base,a,s,plan['backward_examples_per_run'])]},indent=2))
    else:
        old.require(args.output is not None,'Output directory required')
        train(args.data,args.output,args.resume)


if __name__ == '__main__':
    main()
