"""Prepare a nested non-IID scaling control anchored to observed 500k clients.

Do not redraw Dirichlet proportions at 2M. Preserve every existing row's owner,
and expand each class according to the original clients' shares, with integer
largest-remainder rounding. Zero class shares stay zero. This freezes one
simulated heterogeneity scenario; it is not a new sample of physical devices.
No training runner is changed or launched by this builder.
"""
import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_packed import PackedData
from src.federated.partition import check_exact_cover
from src.models.research import atomic_json
from src.eval.official_scale_plan import budgets


def nested_offsets(shards, small_tier=0, large_tier=1):
    """Packed source order is manifest shard order, then each shard's row order.

    Shared tier-0 shards need no floating-point feature matching after scalers
    change: their exact source records identify where old rows sit in the new pack.
    """
    if not 0 <= small_tier < large_tier:
        raise ValueError('Increasing subset tiers required')
    mapping, large_offset, seen = [], 0, set()
    for shard in shards:
        if shard['path'] in seen or shard['rows'] <= 0 or shard['tier'] < 0:
            raise ValueError('Invalid or repeated source shard')
        seen.add(shard['path'])
        if shard['tier'] > large_tier:
            continue
        if shard['tier'] <= small_tier:
            mapping.append(np.arange(large_offset, large_offset+shard['rows'], dtype=np.int64))
        large_offset += shard['rows']
    if not mapping:
        raise ValueError('No shared rows')
    return np.concatenate(mapping), large_offset


def scaled_counts(anchor, totals):
    """Integer apportionment: each cell is within one row of its target share."""
    anchor, totals = np.asarray(anchor), np.asarray(totals)
    if (anchor.ndim != 2 or totals.shape != (anchor.shape[1],)
            or not np.issubdtype(anchor.dtype, np.integer)
            or not np.issubdtype(totals.dtype, np.integer)
            or np.any(anchor < 0) or np.any(totals < 0)):
        raise ValueError('Nonnegative integer class counts required')
    small_totals = anchor.sum(axis=0)
    if np.any(small_totals == 0) or np.any(totals < small_totals):
        raise ValueError('Every class must exist and grow or remain constant')
    result = np.empty_like(anchor, dtype=np.int64)
    for c, total in enumerate(totals):
        products = anchor[:, c].astype(np.int64)*int(total)
        result[:, c] = products//small_totals[c]
        remaining = int(total-result[:, c].sum())
        order = sorted(range(len(anchor)), key=lambda i: (-(int(products[i]) % int(small_totals[c])), i))
        result[order[:remaining], c] += 1
    if np.any(result < anchor) or np.any(result[anchor == 0] != 0):
        raise ValueError('Allocation violated nesting or zero-share support')
    return result


def expand(small_y, large_y, small_parts, mapping, seed=426639):
    """Keep old ownership; shuffle only NEW rows with an independent class RNG."""
    small_y, large_y, mapping = map(np.asarray, (small_y, large_y, mapping))
    check_exact_cover(small_parts, len(small_y))
    if (mapping.shape != small_y.shape or not np.issubdtype(mapping.dtype, np.integer)
            or len(np.unique(mapping)) != len(mapping) or np.any(mapping < 0)
            or np.any(mapping >= len(large_y))):
        raise ValueError('Invalid nested row mapping')
    for labels in (small_y, large_y):
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= len(CLASSES)):
            raise ValueError('Invalid class codes')
    if not np.array_equal(small_y, large_y[mapping]):
        raise ValueError('Nested row labels differ')
    anchor = np.array([np.bincount(small_y[p], minlength=len(CLASSES)) for p in small_parts])
    target = scaled_counts(anchor, np.bincount(large_y, minlength=len(CLASSES)))
    owners = np.full(len(large_y), -1, dtype=np.int32)
    for i, p in enumerate(small_parts):
        owners[mapping[p]] = i
    for c in range(len(CLASSES)):
        added = np.flatnonzero((large_y == c) & (owners == -1))
        # Independent RNG streams avoid the old size-dependent Dirichlet draw.
        np.random.default_rng(np.random.SeedSequence([seed, c])).shuffle(added)
        offset = 0
        for i in range(len(small_parts)):
            n = int(target[i, c]-anchor[i, c])
            owners[added[offset:offset+n]] = i
            offset += n
        if offset != len(added):
            raise ValueError('Unallocated added rows')
    parts = [np.flatnonzero(owners == i) for i in range(len(small_parts))]
    check_exact_cover(parts, len(large_y))
    actual = np.array([np.bincount(large_y[p], minlength=len(CLASSES)) for p in parts])
    if not np.array_equal(actual, target):
        raise ValueError('Actual client counts differ from frozen targets')
    return parts, anchor, target


def prepare(small, large, subsets, archive, member_root, output):
    small, large, subsets, output = map(Path, (small, large, subsets, output))
    if output.exists():
        raise FileExistsError(output)
    a, b = PackedData(small), PackedData(large)
    if (a.manifest['train_rows'], b.manifest['train_rows']) != (500000, 2000000):
        raise ValueError('Only the reviewed 500k to 2M expansion is supported')
    for d in (a, b):
        if (d.manifest['source_sha256']['subset_manifest'] != sha256(subsets/'manifest.json')
                or d.manifest['source_sha256']['subset_recipe'] != sha256(subsets/'recipe.json')
                or d.manifest['row_order'] != 'source view shard order, then record order; shuffle indices at training time'):
            raise ValueError('Packed source order/provenance mismatch')
    if a.manifest['source_sha256'] != b.manifest['source_sha256']:
        raise ValueError('Different source recipes')
    subset_manifest = json.loads((subsets/'manifest.json').read_text())
    recipe = json.loads((subsets/'recipe.json').read_text())
    if recipe['sizes'][:2] != [500000, 2000000] or subset_manifest['status'] != 'complete':
        raise ValueError('Unexpected subset tiers')
    mapping, total = nested_offsets(subset_manifest['shards'])
    if total != b.manifest['train_rows'] or len(mapping) != a.manifest['train_rows']:
        raise ValueError('Source offset count mismatch')
    with tarfile.open(archive) as t:
        def raw(name):
            return t.extractfile(member_root.rstrip('/')+'/'+name).read()
        env = json.loads(raw('environment.json'))
        info = json.loads(raw('partition.json'))
        packed = raw('assignments.npz')
        import hashlib
        if (env['packed_manifest_sha256'] != sha256(small/'manifest.json')
                or env['settings']['lane'] != 'dirichlet'
                or env['settings']['partition_seed'] != 7
                or hashlib.sha256(packed).hexdigest() != info['sha256']):
            raise ValueError('Anchor archive provenance mismatch')
        with np.load(io.BytesIO(packed), allow_pickle=False) as saved:
            small_parts = [saved[f'client_{i}'].copy() for i in range(env['settings']['clients'])]
    if len(small_parts) != 20:
        raise ValueError('Twenty anchor clients required')
    parts, anchor, target = expand(a.arrays['train'][1], b.arrays['train'][1], small_parts, mapping)
    if anchor.tolist() != info['counts']:
        raise ValueError('Anchor class counts mismatch')
    output.mkdir(parents=True)
    np.save(output/'small_to_large.npy', mapping)
    for name, assignments in [('500k', small_parts), ('2m', parts)]:
        np.savez_compressed(output/f'{name}-assignments.npz', **{f'client_{i}': p for i, p in enumerate(assignments)})
    share_a, share_b = anchor/anchor.sum(axis=0), target/target.sum(axis=0)
    record = {'status': 'prepared_not_trained', 'test_opened': False, 'builder_sha256': sha256(__file__),
              'numpy': np.__version__, 'classes': CLASSES,
              'source_sha256': a.manifest['source_sha256'],
              'anchor_archive_sha256': sha256(archive), 'anchor_member_root': member_root,
              'packed_manifest_sha256': {'500k': sha256(small/'manifest.json'), '2m': sha256(large/'manifest.json')},
              'files': {p.name: sha256(p) for p in output.iterdir()},
              'anchor_counts': anchor.tolist(), 'expanded_counts': target.tolist(),
              'old_rows_retaining_client': len(mapping), 'class_support_preserved': True,
              'max_class_share_absolute_change': float(np.abs(share_a-share_b).max()),
              'class_allocation_total_variation': dict(zip(CLASSES, map(float, .5*np.abs(share_a-share_b).sum(axis=0)))),
              'client_row_ranges': {'500k': [int(v) for v in (anchor.sum(axis=1).min(), anchor.sum(axis=1).max())],
                                    '2m': [int(v) for v in (target.sum(axis=1).min(), target.sum(axis=1).max())]},
              'budgets': {'500k_step20': budgets(list(map(len, small_parts)), 20),
                          '2m_step5': budgets(list(map(len, parts)), 5),
                          '2m_step20': budgets(list(map(len, parts)), 20)},
              'runner_ready': False,
              'limitations': 'One anchored client scenario; class proportions approximate to integer rounding; cohort scalers, class weights and local epoch lengths still change'}
    atomic_json(record, output/'plan.json')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('small', 'large', 'subsets', 'archive', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--member-root', required=True)
    prepare(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
