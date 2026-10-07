"""Aggregate the local diagnostic checks without publishing row-level artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.explain.research import sha


def fractions(split,source,c):
    records=[r for r in split[source].values() if r['class']==c]
    n=sum(r['rows'] for r in records)
    return n,sum(r['iat_bins'][0] for r in records)/n,sum(r['iat_bins'][-1] for r in records)/n


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    d=json.loads((a.input/'distributions/summary.json').read_text())
    s=json.loads((a.input/'sensitivity/summary.json').read_text())
    assert s.get('complete') and len(s['models'])==4
    assert not s['test_evaluated'] and not d['test_evaluated']
    assert d['source_sha256']==sha(Path(__file__).with_name('distributions.py'))
    assert s['source_sha256']==sha(Path(__file__).with_name('sensitivity.py'))
    lines=['# Local distribution and SHAP sensitivity checks', '', '## Decision', '',
        'No new cloud training is justified by these diagnostics alone. The suspicious median jumps '
        'do not establish a local preprocessing bug. Higher SHAP integration budgets improve the numerical '
        'approximation overall, but changing the background changes the question and some feature rankings. '
        'Keep precise feature thresholds and automated blocking recommendations provisional.', '',
        '## The median warning is a mixture effect', '',
        'Earlier analysis flagged huge train/validation IAT median changes. The new streaming raw-CSV '
        'check shows two widely separated regimes, with approximately half the rows in each for these classes. '
        'Moving a few percent of rows across the 50% point can move the median by orders of magnitude. '
        'This is not a corresponding orders-of-magnitude change in every row.', '',
        '| Split | Source | Class | Rows | IAT below 1 | IAT at least 100 million |',
        '|---|---|---|---:|---:|---:|']
    for split,v in d['splits'].items():
        for source in ('raw','sampled'):
            for c in ('Benign','Web-based','Brute Force'):
                n,low,high=fractions(v,source,c)
                lines.append(f'| {split} | {source} | {c} | {n} | {low:.2%} | {high:.2%} |')
    lines += ['', 'Bins use recorded feature units for description only, not time-unit claims or security thresholds. '
              'For sampled Brute Force, the low-IAT proportion changes from 47.31% to 52.71%; the enormous '
              'median jump should not be described as proof of a severe domain shift. Smaller mixture differences '
              'may still affect predictions; no causal effect was estimated.', '',
              '## Raw data and preprocessing checks', '',
              '- Streamed only train and validation raw CSVs in bounded record batches. No test data was opened and no new training was performed.',
              '- Raw label counts match the legacy sampling manifest. Rare training classes retain all raw rows; their three-feature value multisets match at 1e-12 relative/absolute tolerance.',
              '- The legacy manifest is pre-cleaning: validation has 2,882 fewer rows than its recorded sample count. This is flagged, not silently rewritten.',
              '- For all seven rare raw labels, a read-only reconstruction of the existing train-overlap anti-join reproduces retained validation counts and the IAT/Number/Weight multisets. This checks the rare labels, not the full historical sampling process.',
              '- Saved scaler means and variances match sampled training values for all three inspected features. Current split hashes match the cloud model manifests.',
              '- The legacy manifest lacks a sampling-seed field. Current sampler code records seeds, but that does not retroactively prove the historical sampling seed. Preserve the hashed sampled files.', '',
              'The scope is three inspected features, not a universal guarantee against all data issues. '
              'No data was regenerated, repaired, pooled or re-split.', '',
              '## What the original dataset documentation supports', '',
              'The [dataset paper, Table 4](https://mdpi-res.com/d_attachment/sensors/sensors-23-05941/article_deploy/sensors-23-05941.pdf) '
              'describes IAT as the gap from the previous packet, Number as a packet count, and Weight as the product of incoming and outgoing packet counts. '
              'The [official UNB statistics](https://www.unb.ca/cic/datasets/iotdataset-2023.html) already report large IAT values '
              'and fractional Number/Weight summaries (including medians 9.5 and 141.55). Thus fractional values and large IAT magnitudes '
              'are not, by themselves, evidence of a local pipeline bug. The precise units and aggregation implementation were not '
              'independently reconstructed from the original packet extraction code. Do not invent a unit conversion.', '',
              '## Frozen sensitivity screen', '',
              f"Four seed-7 checkpoints (heavy, central light, IID, non-IID); {s['core_count']} balanced core rows plus {s['supplement_count']} "
              'fixed error examples shared by every model and variant. These rows are a subset of the previous frozen selection, not chosen after seeing these results.', '',
              '- Original natural-prevalence 128-row background at 256 and 1,024 integration samples.',
              '- An independent seeded natural-prevalence 128-row background at 1,024 samples.',
              '- A 128-row balanced background (16 training rows per class) at 1,024 samples.',
              '- Two independent Monte Carlo repeats per condition. No architecture, weights, loss, validation rows or predictions were changed.', '',
              'The balanced background changes the reference population. It is a sensitivity analysis, not a more accurate substitute for natural prevalence.', '',
              '| Model | Variant | Relative mean absolute residual | Mean repeat top-5 overlap | Error residual at least margin |',
              '|---|---|---:|---:|---:|']
    for lane,r in s['models'].items():
        for name,v in r['variants'].items():
            repeat=np.mean([c['mc_repeat_top5_overlap'] for c in v['classes'].values()])
            lines.append(f"| {lane} | {name} | {v['relative_mean_abs_residual']:.2%} | {repeat:.1%} | {v['error_residual_exceeds_margin']} / {v['error_count']} |")
    lines += ['', 'Residual means the error in reconstructing logit differences from attributions, not classification error. '
              'The reported ratio is aggregated; inspect individual margins before using a signed explanation. '
              'Tiny class cohorts (two core rows per class) make rankings exploratory.', '',
              '## Background dependence of rare class features', '',
              '| Model | Comparison | Web top-5 overlap | Brute Force overlap | DoS overlap |', '|---|---|---:|---:|---:|']
    for lane,r in s['models'].items():
        for name,c in r['comparison_top5_overlap'].items():
            lines.append(f"| {lane} | {name} | {c['Web-based']:.0%} | {c['Brute Force']:.0%} | {c['DoS']:.0%} |")
    lines += ['', '## Next decision', '',
              'Retain the fixed-budget cloud comparison and the failed FedProx screen. Use only qualified, '
              'feature-level explanation statements with background and approximation caveats. Do not convert '
              'feature importance into packet thresholds or claims about device vulnerabilities. '
              'A future capacity or feature-ablation experiment would need its own frozen hypothesis and fair '
              'centralized/federated comparison; it is not launched or approved by this report.', '',
              '## Reproduce', '',
              'Run from the repository root with the existing private cloud-analysis checkpoint staging directory:', '',
              '```bash',
              'python -m reports.local_diagnostics_2026_10_07.distributions --output outputs/diagnostic-recheck/distributions',
              'python -m reports.local_diagnostics_2026_10_07.sensitivity --output outputs/diagnostic-recheck/sensitivity',
              'python -m reports.local_diagnostics_2026_10_07.summarize --input outputs/diagnostic-recheck --output outputs/diagnostic-recheck/report',
              '```', '',
              'Source and package hashes, full distributions, per-class validation metrics and all attribution quality statistics '
              'are in [evidence.json](evidence.json). Row selections and raw attribution tensors remain in ignored outputs/. '
              'No new dependencies were added. Historical reports remain dated records; this report refines their median warning.', '']
    a.output.mkdir(parents=True)
    (a.output/'README.md').write_text('\n'.join(lines))
    (a.output/'evidence.json').write_text(json.dumps({'distributions':d,'sensitivity':s},indent=2)+'\n')


if __name__=='__main__': main()
