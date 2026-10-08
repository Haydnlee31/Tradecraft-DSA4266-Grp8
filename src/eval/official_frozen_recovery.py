"""Two-round frozen-partition bridge/recovery gate, not a scaling experiment.

The 500k arm also compares the native seed-7 partition with the imported anchor.
The 2M arm must NOT compare with a newly drawn partition: that would change the
client mixtures. CUDA checks remain necessary even after a local CPU pass.
"""
import argparse
import json
from pathlib import Path

import torch

from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.eval.official_streaming_recovery import same
from src.federated.official_frozen import load_frozen
from src.models.official_streaming import run, partitions
from src.models.research import atomic_json


def compare(left, right):
    for key in ('step', 'model', 'optimizer', 'best', 'best_score', 'best_step', 'rng'):
        if not same(left[key], right[key]):
            raise ValueError(f'Frozen recovery/bridge mismatch: {key}')
    if len(left['history']) != len(right['history']):
        raise ValueError('Frozen history length mismatch')
    for a, b in zip(left['history'], right['history']):
        if {k:v for k,v in a.items() if k!='elapsed_seconds'} != {k:v for k,v in b.items() if k!='elapsed_seconds'}:
            raise ValueError('Frozen history mismatch')


def pilot(data, partition_root, output, device='cpu', clients=20, batch_size=512):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    packed = PackedData(data)
    parts, control = load_frozen(partition_root, packed, clients)
    if control['cohort'] == '500k':
        native, _ = partitions(packed.arrays['train'][1], 'dirichlet', clients, 7, .5)
        import numpy as np
        if not all(np.array_equal(a,b) for a,b in zip(parts,native)):
            raise ValueError('Native 500k partition does not match the frozen anchor; no bridge launched')
    output.mkdir(parents=True)
    record = {'status': 'incomplete', 'device': device, 'test_evaluated': False,
              'purpose': 'Two-round recovery/bridge only; not model selection or scaling evidence',
              'pilot_sha256': sha256(__file__), 'partition_control': control,
              'packed_manifest_sha256': sha256(Path(data)/'manifest.json')}
    atomic_json(record, output/'checks.json')
    options = dict(data_root=data, lane='dirichlet', epochs=2, device=device, clients=clients,
                   batch_size=batch_size, normalization='layer', seed=7, partition_seed=7)
    run(output=output/'frozen-full', partition_root=partition_root, **options)
    run(output=output/'frozen-resumed', partition_root=partition_root, stop_after=1, **options)
    run(output=output/'frozen-resumed', partition_root=partition_root, resume=True, **options)
    def read(name):
        return torch.load(output/name/'last.pt', map_location='cpu', weights_only=True)
    full = read('frozen-full')
    compare(full, read('frozen-resumed'))
    record['recovery_exact'] = True
    if control['cohort'] == '500k':
        run(output=output/'native-full', **options)
        compare(full, read('native-full'))
        record['same_code_anchor_bridge_exact'] = True
    else:
        record['same_code_anchor_bridge_exact'] = None  # Not applicable to the expanded partition.
    record['examples_processed'] = sum(h['examples_processed'] for h in full['history'])
    record['optimizer_steps'] = sum(h['optimizer_steps'] for h in full['history'])
    record['status'] = 'complete'
    atomic_json(record, output/'checks.json')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'partition-root', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    print(json.dumps(pilot(**vars(p.parse_args())), indent=2))


if __name__ == '__main__':
    main()
