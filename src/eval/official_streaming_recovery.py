"""Two-step compatibility/recovery pilot, not a hyperparameter sweep."""
import argparse
import json
from pathlib import Path

import torch

from src.data.official_inventory import sha256
from src.data.official_shards import save_json
from src.models.official_streaming import run


def same(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def pilot(data, output, device='cpu', lanes=('light', 'dirichlet'), clients=20, batch_size=512,
          normalization='batch', loss_reduction='batch_weight_sum'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checks = {'status': 'incomplete', 'device': device, 'test_evaluated': False,
              'purpose': 'compatibility and recovery, not model selection or data-scale conclusions',
              'packed_manifest_sha256': sha256(Path(data)/'manifest.json'),
              'pilot_sha256': sha256(__file__), 'normalization': normalization,
              'loss_reduction': loss_reduction, 'lanes': {}}
    save_json(output/'checks.json', checks)
    for lane in lanes:
        options = dict(data_root=data, lane=lane, epochs=2, device=device, clients=clients, batch_size=batch_size,
                       normalization=normalization, loss_reduction=loss_reduction)
        run(output=output/f'{lane}-full', **options)
        run(output=output/f'{lane}-resumed', stop_after=1, **options)
        run(output=output/f'{lane}-resumed', resume=True, **options)
        full, resumed = [torch.load(output/f'{lane}-{kind}'/'last.pt', map_location='cpu', weights_only=True)
                         for kind in ('full', 'resumed')]
        for key in ('model', 'optimizer', 'best', 'best_score', 'best_step', 'rng'):
            if not same(full[key], resumed[key]):
                raise ValueError(f'{lane}: recovery mismatch in {key}')
        for a, b in zip(full['history'], resumed['history']):
            if {k: v for k, v in a.items() if k != 'elapsed_seconds'} != {k: v for k, v in b.items() if k != 'elapsed_seconds'}:
                raise ValueError(f'{lane}: history mismatch')
        checks['lanes'][lane] = {'recovery_exact': True, 'steps': 2,
                                'examples_processed': sum(h['examples_processed'] for h in full['history']),
                                'optimizer_steps': sum(h['optimizer_steps'] for h in full['history'])}
        save_json(output/'checks.json', checks)
    checks['status'] = 'complete'
    save_json(output/'checks.json', checks)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--normalization', choices=['batch', 'layer'], default='batch')
    parser.add_argument('--loss-reduction', choices=['batch_weight_sum', 'fixed_train_mean'], default='batch_weight_sum')
    parser.add_argument('--lanes', nargs='+', choices=['light', 'heavy', 'iid', 'dirichlet'], default=['light', 'dirichlet'])
    args = parser.parse_args()
    print(json.dumps(pilot(args.data, args.output, args.device, lanes=args.lanes,
                           normalization=args.normalization, loss_reduction=args.loss_reduction), indent=2))


if __name__ == '__main__':
    main()
