"""Resource-bound replay subset without changing Laya's classification screen.

Freeze this subset before opening the first inference scores. Full numerical
training replay remains mandatory. Never describe this as full Laya replay.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import official39_representation_pilot as pilot
from src.eval.official_evidence import digest, require

PLAN = ROOT/'reports/risk_aware_routing/laya-repeatability-plan.json'
read, save = pilot.frozen.read, pilot.frozen.save


def prepare(prepared, output):
    require(not output.exists(), 'Preserve frozen repeat subset')
    plan, p = read(PLAN), read(prepared/'preparation.json')
    require(digest(pilot.PLAN) == plan['original_plan_sha256']
            and p['identity'] == pilot.identity(), 'Original experiment changed')
    require(all(digest(prepared/n) == h for n, h in p['files'].items()), 'Prepared inputs changed')
    labels = np.load(prepared/'labels.npy', allow_pickle=False)
    ids = pilot.rep.stratified_rows(np.arange(len(labels)), labels,
                                   plan['repeat_rows_per_class'], plan['repeat_sample_seed'])
    request = read(prepared/'inputs.json')
    request['states'] = {k: [v[i] for i in ids] for k, v in request['states'].items()}
    request['plan']['laya']['rows_per_class'] = plan['repeat_rows_per_class']
    output.mkdir()
    save(output/'inputs.json', request)
    np.save(output/'indices.npy', ids, allow_pickle=False)
    save(output/'frozen-subset.json', dict(plan=plan, plan_sha256=digest(PLAN),
        source_sha256=digest(__file__), indices_sha256=digest(output/'indices.npy'),
        inputs_sha256=digest(output/'inputs.json'), original_inputs_sha256=digest(prepared/'inputs.json')))
    print('PASS: 128-row repeat subset frozen before first scores are reviewed', flush=True)


def review(prepared, repeat, first, replay, destination):
    require(not destination.exists() and first.resolve() != replay.resolve(), 'Fresh result required')
    p, frozen = read(prepared/'preparation.json'), read(repeat/'frozen-subset.json')
    require(p['identity'] == pilot.identity() and frozen['plan'] == read(PLAN)
            and frozen['plan_sha256'] == digest(PLAN) and frozen['source_sha256'] == digest(__file__)
            and all(digest(prepared/n) == h for n, h in p['files'].items())
            and frozen['indices_sha256'] == digest(repeat/'indices.npy')
            and frozen['inputs_sha256'] == digest(repeat/'inputs.json')
            and frozen['original_inputs_sha256'] == digest(prepared/'inputs.json'), 'Frozen inputs changed')
    ids = np.load(repeat/'indices.npy', allow_pickle=False)
    labels = np.load(prepared/'labels.npy', allow_pickle=False)
    expected = pilot.rep.stratified_rows(np.arange(len(labels)), labels, 16, 20261013)
    require(np.array_equal(ids, expected), 'Repeat membership changed')
    a, b = read(first), read(replay)
    original_request = read(prepared/'inputs.json')
    for r, source in [(a, prepared/'inputs.json'), (b, repeat/'inputs.json')]:
        require(r['status'] == 'complete' and not r['test_opened'] and not r['fine_tuned']
                and r['input_sha256'] == digest(source) and r['cpu_fallback_count'] == 0
                and r['checkpoint_files'] == original_request['checkpoint_files']
                and r['source_sha256'] == digest(ROOT/'scripts/official39_laya_inference.py')
                and set(r['arms']) == set(pilot.rep.SERIALIZERS), 'Wrong inference receipt')
    require(a['runtime'] == b['runtime'] and a['device'] == b['device'] == 'mps'
            and a['inference_dtype'] == b['inference_dtype'] == 'torch.float32', 'Runtime changed')
    scores, passing = {}, []
    for name in pilot.rep.SERIALIZERS:
        x, y = a['arms'][name], b['arms'][name]
        require(x['rows'] == 1024 and y['rows'] == 128, 'Screen or replay size changed')
        for key in ['predictions', 'rounded_probabilities']:
            require([x[key][i] for i in ids] == y[key], 'Repeat subset predictions changed')
        pred = np.asarray(x['predictions'], np.uint8)
        scores[name] = pilot.old.prior.prior.scored(labels, pred, p['calibration_population_counts'])
        if pilot.rep.gate(scores[name]['prior_standardized'], p['baseline']['prior_standardized']):
            passing.append(name)
    result = dict(status='complete', identity=pilot.identity(), preparation=p, inference=a,
        repeatability=frozen, repeat_subset_exact=True, full_replay_performed=False,
        first_sha256=digest(first), replay_sha256=digest(replay), scores=scores,
        passing_arms=passing, pivot_authorized=not passing, test_opened=False, audit_scored=False,
        interpretation='1024-row balanced calibration screen reweighted to calibration prior; 128-row repeat subset per arm; exploratory only')
    save(destination, result)
    print({'passing_arms': passing, 'pivot_authorized': not passing, 'repeat_subset_exact': True}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    for name in ['prepared', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('review')
    for name in ['prepared', 'repeat', 'first', 'replay', 'destination']:
        p.add_argument('--'+name, type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    {'prepare': prepare, 'review': review}[command](**args)


if __name__ == '__main__':
    main()
