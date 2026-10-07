"""Freeze exact, nested, class-proportional training subsets with bounded memory.

Rank each training vector independently of file order using a fixed SHA256 domain.
A histogram locates the quota boundaries, then only boundary buckets are sorted.
This avoids sorting or loading all 16M training vectors in memory. No validation
or test features enter selection. Tier 0 belongs to every subset; tier 1 adds rows
to the second and larger subsets, etc. Store each selected vector only once.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_loader import OfficialBatches
from src.data.official_quality import fingerprints
from src.data.official_shards import save_json

DOMAIN = b'tradecraft-official39-learning-curve-v1:426639:'
BUCKETS = 4096


def rank_key(digest):
    # Append the feature digest as a deterministic tie-break for equal rank hashes.
    return hashlib.sha256(DOMAIN + digest).digest() + digest


def bucket(key):
    return int.from_bytes(key[:2], 'big') >> 4


def quotas(counts, sizes):
    """Largest-remainder allocation preserves class proportions up to rounding."""
    total = sum(counts)
    if not sizes or sizes != sorted(set(sizes)) or sizes[0] < len(CLASSES) or sizes[-1] > total:
        raise ValueError('Sizes must be increasing, unique and within training availability')
    allocated = []
    for size in sizes:
        q = [size*n // total for n in counts]
        order = sorted(range(len(counts)), key=lambda i: (-(size*counts[i] % total), i))
        for i in order[:size-sum(q)]:
            q[i] += 1
        if min(q) < 1:
            raise ValueError('Requested subset omits a class; choose a larger size')
        if allocated and any(a > b for a, b in zip(allocated[-1], q)):
            # Largest-remainder allocations are not monotone for arbitrary sizes.
            # Fail rather than quietly violating nesting or changing the design.
            raise ValueError('Rounded class quotas are not nested for these sizes')
        allocated.append(q)
    return allocated


def metadata_batches(data):
    for shard in data.shards:
        path = data.root / shard['path']
        if sha256(path) != shard['sha256']:
            raise ValueError('Parent shard checksum changed')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=['_digest', '_class_id']):
            yield zip(batch.column('_digest').to_pylist(), batch.column('_class_id').to_pylist())


def choose_recipe(data, sizes, boundary_cap=131072):
    counts = [data.manifest['counts_by_split_and_class']['train'][name] for name in CLASSES]
    allocations = quotas(counts, sizes)
    hist = np.zeros((len(CLASSES), BUCKETS), dtype=np.int64)
    print('Subset pass 1: training rank histogram', flush=True)
    for rows in metadata_batches(data):
        for digest, target in rows:
            hist[target, bucket(rank_key(digest))] += 1
    if hist.sum(axis=1).tolist() != counts:
        raise ValueError('Training counts disagree with manifest')
    cumulative = hist.cumsum(axis=1)
    boundaries = [[int(np.searchsorted(cumulative[c], q[c])) for c in range(len(CLASSES))] for q in allocations]
    needed = {(c, bins[c]) for bins in boundaries for c in range(len(CLASSES))}
    required = sum(int(hist[c, b]) for c, b in needed)
    if required > boundary_cap:
        raise ValueError('Boundary-bucket cap exceeded; review memory budget before retrying')
    keys = {pair: [] for pair in needed}
    print(f'Subset pass 2: sorting only {required:,} boundary keys', flush=True)
    for rows in metadata_batches(data):
        for digest, target in rows:
            key = rank_key(digest)
            pair = (target, bucket(key))
            if pair in keys:
                keys[pair].append(key)
    for pair, values in keys.items():
        if len(values) != int(hist[pair]):
            raise ValueError('Boundary count changed between passes')
        values.sort()
    thresholds = []
    for q, bins in zip(allocations, boundaries):
        row = []
        for c, b in enumerate(bins):
            before = int(cumulative[c, b-1]) if b else 0
            row.append(keys[c, b][q[c]-before-1].hex())
        thresholds.append(row)
    return {'version': 'official39-nested-proportional-v1', 'sizes': sizes,
            'domain_utf8': DOMAIN.decode(), 'ranking': 'SHA256(domain||feature_digest)||feature_digest; ascending bytes',
            'quota_rule': 'proportional largest remainder, class-order tie-break; monotonicity checked',
            'quotas': [dict(zip(CLASSES, q)) for q in allocations], 'thresholds_hex': thresholds,
            'parent_manifest_sha256': sha256(data.root/'manifest.json'),
            'parent_protocol_sha256': sha256(data.root/'protocol.json'),
            'features': data.features, 'classes': CLASSES, 'builder_sha256': sha256(__file__),
            'boundary_keys': required, 'boundary_cap': boundary_cap,
            'test_opened': False, 'validation_used_for_selection': False, 'model_selection_performed': False,
            'validation_policy': 'same complete parent validation split for every size',
            'scaler_policy': 'fit separately on each selected training subset',
            'scope': 'within-collection unique vectors; not independent capture sessions'}


def tier_for(key, target, thresholds):
    return next((tier for tier, row in enumerate(thresholds) if key <= row[target]), None)


def build(parent, output, sizes=(500000, 2000000, 5000000), boundary_cap=131072):
    data = OfficialBatches(parent, 'train')
    output = Path(output).resolve()
    if output.is_relative_to(data.root) or data.root.is_relative_to(output):
        raise ValueError('Subset output must be separate from parent data')
    if output.exists():
        raise FileExistsError('Use a new subset output; incomplete runs fail closed')
    if shutil.disk_usage(output.parent).free < 10*1024**3:
        raise RuntimeError('Less than 10 GiB free')
    recipe = choose_recipe(data, list(sizes), boundary_cap)
    output.mkdir(parents=True)
    save_json(output/'recipe.json', recipe)
    shutil.copyfile(__file__, output/'builder_source.py')
    thresholds = [[bytes.fromhex(value) for value in row] for row in recipe['thresholds_hex']]
    manifest = {'status': 'incomplete', 'recipe_sha256': sha256(output/'recipe.json'),
                'parent_manifest_sha256': recipe['parent_manifest_sha256'], 'shards': [],
                'training_ready': False, 'test_opened': False}
    save_json(output/'manifest.json', manifest)
    tier_counts = [Counter({name: 0 for name in CLASSES}) for _ in sizes]
    print('Subset pass 3: writing disjoint tiers and checking read-back', flush=True)
    for shard_id, shard in enumerate(data.shards):
        path = data.root/shard['path']
        if sha256(path) != shard['sha256']:
            raise ValueError('Parent shard checksum changed')
        for batch_id, batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=8192)):
            digests = batch.column('_digest').to_pylist()
            targets = batch.column('_class_id').to_pylist()
            selected = [[] for _ in sizes]
            for i, (digest, target) in enumerate(zip(digests, targets)):
                tier = tier_for(rank_key(digest), target, thresholds)
                if tier is not None:
                    selected[tier].append(i)
            for tier, indices in enumerate(selected):
                if not indices:
                    continue
                table = pa.Table.from_batches([batch]).take(pa.array(indices, type=pa.int64()))
                relative = f'tier-{tier}/shard-{shard_id:04d}-batch-{batch_id:04d}.parquet'
                destination = output/relative
                destination.parent.mkdir(exist_ok=True)
                pq.write_table(table, destination, compression='zstd', row_group_size=8192)
                # Compare complete columns, including source-row provenance, not
                # just the counts. Each input row visits exactly one output tier.
                saved = pq.read_table(destination)
                if not saved.equals(table):
                    raise ValueError('Subset read-back differs from selected parent rows')
                values = np.column_stack([saved[name].to_numpy() for name in data.features])
                if fingerprints(values) != saved['_digest'].to_pylist():
                    raise ValueError('Subset feature fingerprint mismatch')
                tier_counts[tier].update(CLASSES[targets[i]] for i in indices)
                manifest['shards'].append({'path': relative, 'tier': tier, 'rows': len(indices),
                                           'sha256': sha256(destination), 'parent_shard': shard['path']})
        if (shard_id+1) % 100 == 0:
            print(f'{shard_id+1}/{len(data.shards)} parent training shards checked', flush=True)
    counts = [{name: sum(tier_counts[t][name] for t in range(i+1)) for name in CLASSES} for i in range(len(sizes))]
    if counts != recipe['quotas']:
        raise ValueError('Materialized subset quotas disagree with frozen recipe')
    if sha256(data.root/'manifest.json') != recipe['parent_manifest_sha256']:
        raise ValueError('Parent manifest changed during subset build')
    manifest.update(status='complete', counts_by_size=dict(zip(map(str, sizes), counts)),
                    tier_counts=tier_counts, exact_nesting_verified=True,
                    next_gate='fit subset-specific scalers; matched streaming trainers and recovery tests')
    save_json(output/'manifest.json', manifest)
    return manifest


class SubsetBatches(OfficialBatches):
    """Training-only view of a nested size, using the tested streaming primitives."""
    def __init__(self, parent, subset_root, size, batch_size=8192):
        super().__init__(parent, 'train', batch_size=batch_size)
        parent_hash = sha256(self.root/'manifest.json')
        self.root = Path(subset_root).resolve()
        self.subset_manifest = json.loads((self.root/'manifest.json').read_text())
        self.recipe = json.loads((self.root/'recipe.json').read_text())
        m, r = self.subset_manifest, self.recipe
        if (m['status'] != 'complete' or m['recipe_sha256'] != sha256(self.root/'recipe.json')
                or m['parent_manifest_sha256'] != parent_hash or r['parent_manifest_sha256'] != parent_hash
                or r['features'] != self.features or r['classes'] != CLASSES):
            raise ValueError('Subset recipe, parent or completed manifest mismatch')
        if size not in r['sizes']:
            raise ValueError('Size is not frozen in this recipe')
        tier = r['sizes'].index(size)
        self.shards = [shard for shard in m['shards'] if shard['tier'] <= tier]
        if any(not (self.root/s['path']).resolve().is_relative_to(self.root) for s in self.shards):
            raise ValueError('Subset shard path escapes output')
        self.rows = sum(shard['rows'] for shard in self.shards)
        self.class_counts = m['counts_by_size'][str(size)]
        if self.rows != size or self.class_counts != r['quotas'][tier]:
            raise ValueError('Subset row count or quotas mismatch')
        # Do not let downstream weighting accidentally use full-training counts.
        self.parent_manifest = self.manifest
        self.manifest = {'status': 'complete', 'features': self.features, 'classes': CLASSES,
                         'counts_by_split_and_class': {'train': self.class_counts},
                         'subset_size': size, 'parent_manifest_sha256': parent_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[500000, 2000000, 5000000])
    args = parser.parse_args()
    result = build(args.parent, args.output, args.sizes)
    print(json.dumps(result['counts_by_size'], indent=2))


if __name__ == '__main__':
    main()
