"""Produce a small, shareable audit receipt without copying raw data or SQLite."""
import argparse
from collections import Counter
import json
import shutil
from pathlib import Path

from src.data.label_map import CLASSES
from src.data.official_inventory import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.audit.read_text())
    protocol = json.loads(args.protocol.read_text())
    if audit['status'] != 'complete' or protocol['audit_summary_sha256'] != sha256(args.audit):
        raise ValueError('Audit/protocol mismatch or incomplete audit')
    rows, invalid = Counter(), Counter()
    for file in audit['files']:
        rows[file['class']] += file['rows']
        invalid[file['class']] += file['nonfinite_rows']
    evidence = {'audit_sha256': sha256(args.audit), 'protocol_sha256': sha256(args.protocol),
                'totals': audit['totals'], 'duplicates': audit['duplicates'],
                'parseable_rows_by_class': dict(rows), 'nonfinite_rows_by_class': dict(invalid),
                'single_file_raw_labels': audit['single_file_raw_labels'],
                'counts_by_split_and_class': protocol['counts_by_split_and_class'],
                'elapsed_reported_seconds': audit['elapsed_seconds'],
                'timing_scope': audit.get('timing_scope', 'entire audit'), 'training_ready': False,
                'test_evaluated': False, 'model_performance_measured': False}
    d = audit['duplicates']
    lines = ['# Official 39 feature data audit and split protocol', '',
             'The full source collection was audited locally without training or changing the raw files. '
             'The frozen recipe is a duplicate-grouped within-collection comparison, not an independent-session benchmark.', '',
             f"Readable records: {audit['totals']['rows']:,}. Malformed records: {audit['totals']['malformed']:,}. "
             f"Readable records containing nonfinite features: {audit['totals']['nonfinite_rows']:,}.", '',
             f"Finite feature vectors: {d['finite_rows']:,} occurrences, {d['unique_vectors']:,} SHA256-distinct groups. "
             f"Repeated occurrences beyond the first: {d['duplicate_excess_rows']:,}. "
             f"Groups with conflicting eight-class targets: {d['class_conflict_vectors']:,}.", '',
             '| Class | Readable rows | Nonfinite rows | Unique eligible train | Unique eligible validation | Unique eligible test |',
             '|---|---:|---:|---:|---:|---:|']
    for name in CLASSES:
        counts = [protocol['counts_by_split_and_class'][split][name] for split in ('train', 'val', 'test')]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{n:,}' for n in [rows[name], invalid[name], *counts]) + ' |')
    lines += ['', 'Exclude malformed/nonfinite records and whole groups with contradictory eight-class labels. '
              'Keep one representative per other feature vector. Raw labels that differ within the same eight-class '
              'target remain documented by the duplicate index. The table therefore describes unique vectors, '
              'not original repeated flow frequency; differences between readable and eligible counts are not all invalid data.', '',
              'Removing contradictory-target groups also removes ambiguity from the task. Future metrics will describe '
              'this cleaned subset, not performance on every original record; keep that selection effect visible.', '',
              'Several attack types have only one source file. Grouping exact duplicates across fixed 80/10/10 '
              'hash assignments prevents exact feature-copy leakage, but not near-duplicate or session correlation. '
              'The proportions are approximate; no favourable split seed was searched. Historical mirror overlap '
              'has not been excluded. The test partition is sealed for model selection; only coverage has been audited.', '',
              'The earlier 46-feature study is unchanged. Both the feature representation and evaluation protocol '
              'differ, so cross-study score differences cannot be attributed solely to training-set size.', '',
              'This frozen recipe alone does not certify training readiness. See the extension guide for subsequent '
              'materialization, loader checks and the remaining training gates. No new model quality or speedup is claimed.', '',
              '- [Machine-readable evidence](evidence.json)',
              '- [Frozen protocol recipe](protocol.json)',
              '- [Extension guide](../README.md)', '']
    args.output.mkdir(parents=True)
    shutil.copyfile(args.protocol, args.output / 'protocol.json')
    (args.output / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    (args.output / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
