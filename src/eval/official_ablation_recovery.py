"""Two-step mask/panel compatibility gate; never launch the 20-step experiment.

The explicit reviewed-panel checksum is a handoff guard, not a deployment gate.
CPU success does not establish CUDA recovery. Both views remain validation,
and no score from this short pilot may select a mask or hyperparameter.
"""
import argparse
import json
from pathlib import Path

import torch

from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.data.official_ablation_panel import load_panel, write_json
from src.federated.official_frozen import load_frozen
from src.models import official_streaming as streaming
from src.models.official_ablation import ARMS
from src.eval.official_streaming_recovery import same


def compare(left, right, bridge=False):
    for key in ('step', 'model', 'optimizer', 'best', 'best_score', 'best_step', 'rng'):
        if not same(left[key], right[key]):
            raise ValueError(f'Ablation bridge/recovery mismatch: {key}')
    omit = {'elapsed_seconds'}
    if bridge:
        omit |= {'panel_validation_metrics', 'panel_val_loss'}
    elif left['input_transform'] != right['input_transform']:
        raise ValueError('Input transform changed on recovery')
    a, b = [[{k: v for k, v in row.items() if k not in omit} for row in checkpoint['history']]
            for checkpoint in (left, right)]
    if a != b:
        raise ValueError('Ablation bridge/recovery history mismatch')


def pilot(data, panel, partition_root, reviewed_panel_sha256, output, device='cpu',
          clients=20, batch_size=512, reference_run=None):
    data, panel, partition_root, output = map(Path, (data, panel, partition_root, output))
    if sha256(panel/'receipt.json') != reviewed_panel_sha256:
        raise ValueError('Panel differs from the explicitly reviewed receipt')
    if output.exists():
        raise FileExistsError('Use a new pilot output directory')
    packed = PackedData(data)
    try:
        _, panel_info = load_panel(panel, packed)
        _, partition_info = load_frozen(partition_root, packed, clients)
        train_rows = packed.manifest['train_rows']
    finally:
        for pair in packed.arrays.values():
            for array in pair:
                array._mmap.close()
    # Dependency injection is only for an independently archived local baseline;
    # the CLI always compares against the unchanged default path in this code.
    reference = streaming.run if reference_run is None else reference_run
    reference_source = Path(reference.__code__.co_filename)
    output.mkdir(parents=True, exist_ok=False)
    checks = dict(status='incomplete', purpose='two-step compatibility and recovery only; no mask selection',
                  device=device, test_evaluated=False, comparison_training_launched=False,
                  packed_manifest_sha256=sha256(data/'manifest.json'), panel=panel_info,
                  partition_control=partition_info, pilot_sha256=sha256(__file__),
                  runner_sha256=sha256(streaming.__file__), reference_runner_sha256=sha256(reference_source),
                  expected_total_examples_processed=42*train_rows, lanes={})
    write_json(output/'checks.json', checks)
    total_examples = 0
    for lane in ('light', 'iid', 'dirichlet'):
        lane_root = output/lane
        lane_root.mkdir()
        options = dict(data_root=data, lane=lane, epochs=2, seed=7, clients=clients, batch_size=batch_size,
                       device=device, threads=2, normalization='layer', partition_seed=7,
                       partition_root=partition_root if lane == 'dirichlet' else None)
        baseline = lane_root/'default-full'
        ref_result = reference(output=baseline, **options)
        ref_state = torch.load(baseline/'last.pt', map_location='cpu', weights_only=True)
        total_examples += ref_result['examples_processed']
        initial, assignments, arms = None, None, {}
        for arm in ARMS:
            full_dir, resume_dir = lane_root/(arm+'-full'), lane_root/(arm+'-resumed')
            arm_options = dict(options, ablation_arm=arm, validation_panel=panel)
            full = streaming.run(output=full_dir, **arm_options)
            streaming.run(output=resume_dir, stop_after=1, **arm_options)
            recovered = streaming.run(output=resume_dir, resume=True, **arm_options)
            a, b = [torch.load(p/'last.pt', map_location='cpu', weights_only=True) for p in (full_dir, resume_dir)]
            compare(a, b)
            if arm == 'full39':
                compare(ref_state, a, bridge=True)
            init = json.loads((full_dir/'initialization.json').read_text())
            if init != json.loads((resume_dir/'initialization.json').read_text()) or (initial is not None and initial != init):
                raise ValueError('Paired initial model states differ')
            initial = init
            for result in (full, recovered):
                if (result['test_metrics'] is not None or result['examples_processed'] != 2*train_rows
                        or result['optimizer_steps'] != ref_result['optimizer_steps']
                        or result['primary_validation_metrics'] != result['history'][-1]['panel_validation_metrics']):
                    raise ValueError('Ablation work/endpoint guard failed')
                total_examples += result['examples_processed']
            if lane != 'light':
                hashes = [sha256(p/'assignments.npz') for p in (baseline, full_dir, resume_dir)]
                if len(set(hashes)) != 1 or (assignments is not None and assignments != hashes[0]):
                    raise ValueError('Paired client assignments differ')
                assignments = hashes[0]
            arms[arm] = dict(recovery_exact=True, examples_processed=full['examples_processed'],
                             optimizer_steps=full['optimizer_steps'], primary_rows=panel_info['retained_rows'])
        checks['lanes'][lane] = dict(default_path_bridge_exact=True, paired_initial_state=initial,
                                     assignment_sha256=assignments, arms=arms)
        write_json(output/'checks.json', checks)
    if total_examples != checks['expected_total_examples_processed']:
        raise ValueError('Pilot total work differs from the declared budget')
    checks.update(status='complete', total_examples_processed=total_examples,
                  next='Review this pilot; historical 20-step bridges and six ablations have NOT been launched')
    write_json(output/'checks.json', checks)
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'panel', 'partition-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--reviewed-panel-sha256', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    print(json.dumps(pilot(**vars(p.parse_args())), indent=2))


if __name__ == '__main__':
    main()
