# Official 39 feature data audit and split protocol

The full source collection was audited locally without training or changing the raw files. The frozen recipe is a duplicate-grouped within-collection comparison, not an independent-session benchmark.

Readable records: 46,776,697. Malformed records: 3. Readable records containing nonfinite features: 1,037.

Finite feature vectors: 46,775,660 occurrences, 21,055,707 SHA256-distinct groups. Repeated occurrences beyond the first: 25,719,953. Groups with conflicting eight-class targets: 461,347.

| Class | Readable rows | Nonfinite rows | Unique eligible train | Unique eligible validation | Unique eligible test |
|---|---:|---:|---:|---:|---:|
| Benign | 1,098,191 | 65 | 874,352 | 109,482 | 109,137 |
| DDoS | 33,984,450 | 528 | 9,707,930 | 1,213,890 | 1,213,933 |
| DoS | 7,845,117 | 223 | 2,985,317 | 373,093 | 373,680 |
| Recon | 690,534 | 13 | 545,933 | 68,008 | 68,364 |
| Web-based | 24,829 | 1 | 19,699 | 2,475 | 2,437 |
| Brute Force | 13,064 | 0 | 10,450 | 1,284 | 1,310 |
| Spoofing | 486,458 | 23 | 363,217 | 44,949 | 45,485 |
| Mirai | 2,634,054 | 184 | 1,967,314 | 246,103 | 246,518 |

Exclude malformed/nonfinite records and whole groups with contradictory eight-class labels. Keep one representative per other feature vector. Raw labels that differ within the same eight-class target remain documented by the duplicate index. The table therefore describes unique vectors, not original repeated flow frequency; differences between readable and eligible counts are not all invalid data.

Removing contradictory-target groups also removes ambiguity from the task. Future metrics will describe this cleaned subset, not performance on every original record; keep that selection effect visible.

Several attack types have only one source file. Grouping exact duplicates across fixed 80/10/10 hash assignments prevents exact feature-copy leakage, but not near-duplicate or session correlation. The proportions are approximate; no favourable split seed was searched. Historical mirror overlap has not been excluded. The test partition is sealed for model selection; only coverage has been audited.

The earlier 46-feature study is unchanged. Both the feature representation and evaluation protocol differ, so cross-study score differences cannot be attributed solely to training-set size.

This frozen recipe alone does not certify training readiness. See the extension guide for subsequent materialization, loader checks and the remaining training gates. No new model quality or speedup is claimed.

- [Machine-readable evidence](evidence.json)
- [Frozen protocol recipe](protocol.json)
- [Extension guide](../README.md)
