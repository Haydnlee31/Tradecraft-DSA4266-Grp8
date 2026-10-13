"""Post-hoc fitting diagnostics only; no training, selection or new test score.

Raw fitting F1 is distorted by the capped class sample. The receipt retains
raw counts and also reweights both fitting and calibration confusion matrices
to the same original FIT class prior. Per-class recall needs no reweighting.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts import official39_contrastive as screen
from src.data.label_map import WEB_BASED, BRUTE_FORCE


def run(source, destination):
    screen.require(not destination.exists(), 'Preserve existing diagnostic')
    receipt = screen.verify(source)
    scores = {}
    with screen.prior.data() as d:
        raw = np.array(d.x[d.fit][:, screen.rep.retained(d.features)])
        x = screen.rep.encode(raw, screen.frozen.read(source/'edges.json'))
        tree = screen.prior.tree_model(d, 7)
        predictions = {'tree': screen.prior.ablation.predict_rows(tree, d.x, d.fit, d.features, screen.neural.ARM)}
        for m in receipt['models']:
            model = screen.core.FlowModel(x.shape[1])
            screen.require(screen.digest(source/m['file']) == m['sha256'], 'Checkpoint changed')
            model.load_state_dict(torch.load(source/m['file'], weights_only=True, map_location='cpu'), strict=True)
            screen.require(screen.neural.state_hash(model) == m['final_state_sha256'], 'Tensor state changed')
            predictions[m['arm']] = screen.core.predict(model, x)
        for name, pred in predictions.items():
            # Recompute the calibration score from the actual fixed labels as
            # well, rather than accepting only a saved confusion matrix.
            calibration = np.load(source/(name+'-calibration.npy'), allow_pickle=False)
            observed = screen.prior.old.prior.prior.scored(d.y[d.calibration], calibration, d.pop)
            screen.require(observed == receipt['scores'][name], 'Calibration labels/predictions changed')
            fit = screen.prior.old.prior.prior.scored(d.y[d.fit], pred, d.pop)
            scores[name] = dict(fitting=fit, calibration=observed)
    screen.require(screen.verify(source) == receipt, 'Source evidence changed during diagnostic')
    result = dict(status='complete', purpose='post_hoc_fit_vs_calibration_diagnostic_not_model_selection',
        source_receipt_sha256=screen.digest(source/'receipt.json'),
        diagnostic_source_sha256=screen.digest(Path(__file__)), source_identity=screen.identity(), scores=scores,
        model_trained=False, test_opened=False, test_evaluated=False, audit_scored=False,
        fit_population_counts=receipt['recipe']['fit_population_counts'])
    screen.frozen.save(destination, result)
    for name, s in scores.items():
        print(name, flush=True)
        for c in [WEB_BASED, BRUTE_FORCE]:
            print(c, 'fit recall', s['fitting']['metrics']['per_class'][c]['recall'],
                  'calibration recall', s['calibration']['metrics']['per_class'][c]['recall'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--destination', required=True, type=Path)
    args = vars(ap.parse_args())
    run(**args)
