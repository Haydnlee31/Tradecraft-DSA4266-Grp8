"""Label-free flow representations for a bounded exploratory comparison.

Text is a serialization, not new evidence: it cannot recover missing packet
payloads or device identities. Numerical bins are fitted on FIT inputs only.
This implements a clipped-tail PLE adaptation, not the paper's full search.
"""
import numpy as np

from src.data.label_map import CLASSES, WEB_BASED, BRUTE_FORCE
from src.eval.official_evidence import require

MASKED = ('Number', 'Tot sum')
SERIALIZERS = ('values', 'values_with_fit_bins')


def retained(features):
    require(len(features) == 39 and len(set(features)) == 39
            and all(n in features for n in MASKED), 'Wrong official feature schema')
    return [i for i, n in enumerate(features) if n not in MASKED]


def fit_edges(x, bins=16):
    """Duplicate quantiles collapse; a constant gets one zero output column."""
    require(x.ndim == 2 and len(x) > 0 and np.isfinite(x).all()
            and type(bins) is int and bins >= 2, 'Invalid fitting inputs')
    return [np.unique(np.quantile(x[:, j].astype(np.float64),
                                 np.linspace(0, 1, bins+1))).tolist()
            for j in range(x.shape[1])]


def encode(x, edges):
    """Cumulative linear ramps: [0,0] -> [1,0] -> [1,1] for two bins.

The fixed basis does not learn labels; the subsequent dense layer learns
which combinations distinguish attacks. Clipping tails is explicit so we
do not claim to preserve unseen extremes outside the fitting range.
"""
    require(x.ndim == 2 and x.shape[1] == len(edges) and np.isfinite(x).all(), 'Invalid encoder input')
    columns = []
    for j, values in enumerate(edges):
        e = np.asarray(values, dtype=np.float64)
        require(e.ndim == 1 and len(e) > 0 and np.isfinite(e).all()
                and (np.diff(e) > 0).all(), 'Invalid bin edges')
        if len(e) == 1:
            columns.append(np.zeros((len(x), 1), np.float32))
        else:
            z = (x[:, j:j+1].astype(np.float64)-e[:-1]) / np.diff(e)
            columns.append(np.clip(z, 0, 1).astype(np.float32))
    return np.concatenate(columns, axis=1)


def serialize(x, features, edges, mode):
    """Nine significant digits round-trip float32; no inferred attack stories.

The optional bin index only uses fitting quantiles. It is not a class label,
nor an exact percentile when repeated feature values collapse quantiles.
"""
    require(mode in SERIALIZERS and x.ndim == 2 and x.shape[1] == 39
            and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid serializer input')
    ids = retained(features)
    require(len(edges) == len(ids), 'Wrong bin schema')
    prefix = 'IoT flow. Values are standardized numeric statistics, not physical units.'
    if mode == SERIALIZERS[1]:
        prefix += ' b is the fitting-quantile bin, low to high; c means constant in fitting data.'
    states = []
    for row in x:
        fields = []
        for j, column in enumerate(ids):
            value = f'{features[column]}={float(row[column]):.9g}'
            if mode == SERIALIZERS[1]:
                e = edges[j]
                b = np.searchsorted(e[1:-1], row[column], side='right')
                value += f' b{b+1}/{len(e)-1}' if len(e) > 1 else ' c'
            fields.append(value)
        states.append(prefix+'\n'+';'.join(fields))
    return states


def stratified_rows(rows, labels, per_class=128, seed=20261012):
    require(rows.ndim == labels.ndim == 1 and len(rows) == len(labels)
            and len(np.unique(rows)) == len(rows)
            and np.isin(labels, np.arange(len(CLASSES))).all(), 'Invalid sampling frame')
    rng = np.random.default_rng(seed)
    chosen = []
    for i in range(len(CLASSES)):
        eligible = rows[labels == i]
        require(len(eligible) >= per_class > 0, 'Insufficient class support')
        chosen.extend(rng.choice(eligible, per_class, replace=False))
    return np.sort(np.array(chosen, dtype=np.int64))


def gate(candidate, baseline):
    """Fixed practical continuation screen, not a significance test."""
    return (candidate['macro_f1'] > baseline['macro_f1']
            and candidate['benign_false_alert_rate'] <= baseline['benign_false_alert_rate']
            and all(candidate['per_class'][c]['f1'] > baseline['per_class'][c]['f1']
                    for c in (WEB_BASED, BRUTE_FORCE)))
