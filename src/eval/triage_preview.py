"""Read-only triage presentation; neither ground truth nor an expert is an input.

The classifier's prediction and the review decision are two separate fields.
An unflagged flow is not certified safe, and a flagged flow is not a repaired
classification. This module does not train, call services, or modify models.
"""
from pathlib import Path
import json

import numpy as np

from src.data.label_map import CLASSES
from src.eval.official_evidence import require
from src.eval.routing_gate import decisions

TEMPLATE = Path(__file__).with_name('triage_preview.html')
METHODS = ('learned', 'confidence', 'random')


def apply_review(predictions, scores, threshold):
    """Keep class IDs unchanged while independently applying a frozen threshold."""
    require(predictions.ndim == 1 and len(predictions) > 0 and scores.shape == predictions.shape
            and np.issubdtype(predictions.dtype, np.integer)
            and np.all((predictions >= 0) & (predictions < len(CLASSES))), 'Invalid prediction alignment/schema')
    # decisions handles finite scores, whole ties and the explicit no-call
    # threshold. Never sort or trim on this later cohort to hide an overrun.
    return predictions.copy(), decisions(scores, threshold)


def preview_rows(rows, predictions, confidence, scores, threshold, method, limit=8):
    """First N flagged and N unflagged rows in source order, without labels.

    This small display sample is not a prevalence sample, top-risk ranking,
    or class-balanced showcase. Population workload uses ALL audit rows.
    """
    require(method in METHODS and type(limit) is int and 1 <= limit <= 20, 'Invalid preview settings')
    require(rows.shape == predictions.shape and np.issubdtype(rows.dtype, np.integer)
            and np.all(rows >= 0) and np.all(rows[1:] > rows[:-1]), 'Invalid source row identifiers')
    require(confidence.shape == predictions.shape and np.isfinite(confidence).all()
            and np.all((confidence >= 0) & (confidence <= 1)), 'Invalid uncalibrated confidence')
    retained, flags = apply_review(predictions, scores, threshold)
    sample = np.sort(np.concatenate([np.flatnonzero(flags)[:limit], np.flatnonzero(~flags)[:limit]]))
    result = []
    for i in sample:
        flagged = bool(flags[i])
        result.append(dict(row_ref=f'validation:{int(rows[i])}', prediction=CLASSES[int(retained[i])],
            max_softmax_uncalibrated=float(confidence[i]), review_score=float(scores[i]), needs_review=flagged,
            status='Needs review' if flagged else 'Not flagged',
            reason=('Frozen threshold met; review unresolved' if flagged else
                    'No-review threshold' if threshold is None else 'Below frozen threshold; not a safety guarantee')))
    return result, flags


def workload(flags, budget):
    require(flags.ndim == 1 and flags.dtype == np.bool_ and len(flags) > 0
            and type(budget) is int and budget in (1, 5, 10), 'Invalid workload inputs')
    count, rows = int(flags.sum()), len(flags)
    cap = rows*budget//100
    return dict(rows=rows, referrals=count, referral_rate=count/rows, calibration_budget_percent=budget,
        illustrative_cohort_cap=cap, over_cap=max(0, count-cap), budget_respected=count <= cap,
        policy_approved=False, automatic_blocking=False, classification_labels_changed=False)


def render(payload):
    """A self-contained local file: no server, network requests or CDN assets.

    Escape script delimiters even for unexpected metadata. The template uses
    textContent for values; no data-derived HTML or event handlers are built.
    """
    require(payload['protocol'] == 'official39-triage-preview-v1' and payload['read_only'] is True
            and payload['ground_truth_rows_opened'] is False and payload['operational_budget_approved'] is False,
            'Wrong preview scope')
    encoded = json.dumps(payload, allow_nan=False, separators=(',', ':'))
    for character, escaped in [('&', '\\u0026'), ('<', '\\u003c'), ('>', '\\u003e')]:
        encoded = encoded.replace(character, escaped)
    template = TEMPLATE.read_text(encoding='utf-8')
    require(template.count('__TRIAGE_DATA__') == 1, 'Invalid preview template')
    return template.replace('__TRIAGE_DATA__', encoded)
