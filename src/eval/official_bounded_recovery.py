"""Two-round capped FedAvg recovery check; never launches the experiment grid."""
import argparse
from pathlib import Path

import torch

from src.data.official_inventory import sha256
from src.eval.official_streaming_recovery import same
from src.models.official_streaming import run
from src.models.research import atomic_json


def pilot(data, output, device='cpu', clients=20, batch_size=512):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'incomplete', 'device': device, 'test_evaluated': False,
              'purpose': 'Capped full-participation recovery including a shorter final round; not model selection',
              'packed_manifest_sha256': sha256(Path(data)/'manifest.json'),
              'pilot_sha256': sha256(__file__), 'lanes': {}}
    atomic_json(result, output/'checks.json')
    for lane in ('iid', 'dirichlet'):
        options = dict(data_root=data, lane=lane, epochs=2, clients=clients,
                       batch_size=batch_size, normalization='layer', device=device,
                       local_max_batches=2, total_update_budget=clients*3)
        full_path, resumed_path = output/f'{lane}-full', output/f'{lane}-resumed'
        run(output=full_path, **options)
        run(output=resumed_path, stop_after=1, **options)
        run(output=resumed_path, resume=True, **options)
        full, resumed = [torch.load(p/'last.pt', map_location='cpu', weights_only=True)
                         for p in (full_path, resumed_path)]
        for key in ('model', 'optimizer', 'best', 'best_score', 'best_step', 'rng'):
            if not same(full[key], resumed[key]):
                raise ValueError(f'{lane}: recovery mismatch in {key}')
        clean = lambda h: [{k:v for k,v in row.items() if k != 'elapsed_seconds'} for row in h]
        if clean(full['history']) != clean(resumed['history']):
            raise ValueError(f'{lane}: recovery history mismatch')
        if [h['local_batch_cap'] for h in full['history']] != [2, 1]:
            raise ValueError('Final-round schedule mismatch')
        updates = sum(h['optimizer_steps'] for h in full['history'])
        if updates != clients*3:
            raise ValueError('Exact budget mismatch')
        result['lanes'][lane] = {'recovery_exact': True, 'optimizer_steps': updates,
                                 'final_round_cap': 1, 'full_participation': True}
        atomic_json(result, output/'checks.json')
    result['status'] = 'complete'
    atomic_json(result, output/'checks.json')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    pilot(args.data, args.output, args.device)


if __name__ == '__main__':
    main()
