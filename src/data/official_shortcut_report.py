"""Publish two identical shortcut-audit receipts; never open research arrays."""
import argparse
import hashlib
import json
from pathlib import Path

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256
from src.data.official_shards import save_json
from src.data import official_shortcut_audit as audit


def validate(receipt, plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    audit.validate_plan(plan)
    reference = json.loads((plan_path.parent/'explanation-cpu-plan.json').read_text())
    if (receipt.get('status') != 'complete' or any(receipt.get(k) is not False for k in audit.FLAGS)
            or receipt.get('plan_sha256') != sha256(plan_path)
            or receipt.get('auditor_sha256') != sha256(audit.__file__)
            or receipt.get('packed_manifest_sha256') != plan['packed_manifest_sha256']):
        raise ValueError('Incomplete or mismatched shortcut receipt')
    for split, count_key in [('train', 'train_class_counts'), ('val', 'validation_class_counts')]:
        for denominator in ('AVG', 'Tot size'):
            stats = receipt['proxy'][split][denominator]
            if set(stats) != set(CLASSES):
                raise ValueError('Missing proxy class')
            for name, r in stats.items():
                if (r['rows'] != reference[count_key][name]
                        or not 0 <= r['within_tolerance'] <= r['valid_denominator'] <= r['rows']
                        or not 0 <= r['window_group_agreement'] <= r['near_10_or_100'] <= r['rows']):
                    raise ValueError('Invalid proxy counts')
    for table in audit.PROJECTIONS:
        r = receipt['collisions'][table]
        for split in ('train', 'val'):
            if (r[f'{split}_rows'] != plan['rows'][split]
                    or r[f'{split}_duplicate_excess_rows'] != r[f'{split}_rows']-r[f'{split}_unique_vectors']
                    or sum(v[split] for v in r['cross_split_rows_by_class'].values()) != r[f'cross_split_{split}_rows']
                    or not 0 <= r['cross_split_groups'] <= r[f'cross_split_{split}_rows'] <= r[f'{split}_rows']):
                raise ValueError('Invalid collision counts')
        if r['unique_vectors'] != r['train_unique_vectors']+r['val_unique_vectors']-r['cross_split_groups']:
            raise ValueError('Invalid set union counts')
    c = receipt['collisions']
    for split in ('train', 'val'):
        if c['new_overlap'][f'{split}_rows'] != c['without_number'][f'cross_split_{split}_rows']-c['full39'][f'cross_split_{split}_rows']:
            raise ValueError('Invalid new overlap counts')
    if 'source_receipt_sha256' in receipt:
        additions = ('source_receipt_sha256', 'independent_replay_receipt_sha256',
                     'publication_scope', 'source_newline')
        raw = {k: v for k, v in receipt.items() if k not in additions}
        # save_json uses the platform text newline. Preserve the original
        # receipt's byte hash even when auditing it on a different OS.
        newline = receipt.get('source_newline')
        if newline not in ('\n', '\r\n'):
            raise ValueError('Unknown source receipt serialization')
        serialized = (json.dumps(raw, indent=2)+'\n').replace('\n', newline)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        if digest != receipt['source_receipt_sha256'] or digest != receipt['independent_replay_receipt_sha256']:
            raise ValueError('Published content differs from pinned source receipt')


def publish(first, replay, plan, output):
    first, replay, output = map(Path, (first, replay, output))
    if first.resolve() == replay.resolve() or first.read_bytes() != replay.read_bytes():
        raise ValueError('Require separate byte-identical completed replay receipts')
    if output.exists():
        raise FileExistsError('Do not overwrite published evidence')
    receipt = json.loads(first.read_text())
    validate(receipt, plan)
    receipt['source_receipt_sha256'] = sha256(first)
    receipt['independent_replay_receipt_sha256'] = sha256(replay)
    receipt['source_newline'] = '\r\n' if b'\r\n' in first.read_bytes() else '\n'
    receipt['publication_scope'] = 'Verified receipt replay; no database, per-row data or model included.'
    validate(receipt, plan)
    save_json(output, receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    publish(args.first, args.replay, args.plan, args.output)
    print('PASS: identical completed audit receipts published; no test access')


if __name__ == '__main__':
    main()
