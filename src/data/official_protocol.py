"""Freeze a within-collection protocol from completed official-39 audits.

Assignments depend on feature fingerprints, never labels or model performance.
This deliberately does not claim session independence or an unseen mirror holdout.
It writes a recipe and audited counts, not training data or model evaluations.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256


def split_for_digest(digest):
    # Fixed once before training. Integer cutoffs avoid platform float effects.
    value = int.from_bytes(hashlib.sha256(b'tradecraft-official39-v1:426639:' + digest).digest()[:8], 'big')
    bucket = value % 10000
    return 'train' if bucket < 8000 else 'val' if bucket < 9000 else 'test'


def freeze(audit_dir, output):
    audit_dir, output = Path(audit_dir), Path(output)
    if output.exists():
        raise FileExistsError('Protocol is immutable; use a new study version for changes')
    summary_path, database = audit_dir / 'summary.json', audit_dir / 'vectors.sqlite'
    summary = json.loads(summary_path.read_text())
    if summary['status'] != 'complete' or len(summary['features']) != 39:
        raise ValueError('Completed official-39 quality audit required')
    if summary['totals']['float32_overflow_rows']:
        raise ValueError('Float32 overflow needs a separate preprocessing decision')
    # Open read-only: the completed duplicate audit remains unchanged.
    connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    counts = {split: Counter({name: 0 for name in CLASSES}) for split in ('train', 'val', 'test')}
    fingerprint_counts = {split: 0 for split in counts}
    unambiguous = Counter()
    try:
        for digest, mask in connection.execute('SELECT digest,class_mask FROM vectors WHERE (class_mask & (class_mask-1))=0'):
            if mask < 1 or mask >= (1 << len(CLASSES)):
                raise ValueError('Invalid class mask')
            name = CLASSES[mask.bit_length()-1]
            split = split_for_digest(digest)
            counts[split][name] += 1
            fingerprint_counts[split] += 1
            unambiguous[name] += 1
    finally:
        connection.close()
    expected = summary['duplicates']['unambiguous_unique_per_class']
    if any(unambiguous[name] != expected.get(name, 0) for name in CLASSES):
        raise ValueError('Duplicate index disagrees with audit summary')
    if any(n == 0 for group in counts.values() for n in group.values()):
        raise ValueError('A class is missing from a split; do not hunt seeds to hide it')
    protocol = {
        'version': 'official39-within-collection-v1', 'status': 'frozen_recipe',
        'training_ready': False, 'data_materialized': False, 'test_evaluated': False,
        'scope': 'within-collection; not capture/session independent',
        'mirror_overlap_excluded': False, 'near_duplicate_independence_claimed': False,
        'audit_summary_sha256': sha256(summary_path), 'duplicate_index_sha256': sha256(database),
        'protocol_builder_sha256': sha256(__file__), 'features': summary['features'], 'classes': CLASSES,
        'split_algorithm': 'SHA256(domain||feature_digest), first 8 bytes unsigned big-endian modulo 10000',
        'domain_utf8': 'tradecraft-official39-v1:426639:',
        'cutoffs': {'train': [0, 8000], 'val': [8000, 9000], 'test': [9000, 10000]},
        'ratios_are_approximate': True, 'seed_search_performed': False,
        'sampling_unit': 'one unique finite float64 39-feature vector, not repeated flow frequency',
        'malformed_policy': 'exclude from derived data; retain untouched raw sources and audit counts',
        'nonfinite_policy': 'exclude any row with a missing/infinite feature; no imputation',
        'duplicate_policy': 'retain one representative per feature digest; all copies share the same split',
        'class_conflict_policy': 'exclude entire digest group when eight-class targets disagree',
        'raw_conflict_within_class_policy': 'retain one vector with eight-class target; preserve raw-label mask',
        'scaler_policy': 'fit new scaler on selected training subset only; never reuse legacy scaler',
        'test_policy': 'sealed for model selection; only provenance and coverage audited before final evaluation',
        'counts_by_split_and_class': counts, 'unique_vectors_by_split': fingerprint_counts,
        'legacy_46_feature_data_modified': False,
        'next_gate': 'materialize derived shards and verify hashes, exclusion counts, split disjointness, then loader parity',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(protocol, stream, indent=2)
        stream.write('\n')
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = freeze(args.audit, args.output)
    print(json.dumps(protocol['counts_by_split_and_class'], indent=2))


if __name__ == '__main__':
    main()
