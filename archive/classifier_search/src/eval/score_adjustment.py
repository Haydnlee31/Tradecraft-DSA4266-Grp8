"""A bounded, automatic decision layer on already fitted eight-class trees.

These are class scores, not calibrated probabilities of a real-world attack.
The exponent is selected on development calibration rows only. Increasing it
favours classes common in the fitting population; this may reduce the balanced
tree's rare-class overprediction, but must earn its place against the baseline.
"""
import numpy as np

from src.data.label_map import CLASSES, WEB_BASED, BRUTE_FORCE
from src.eval.official_evidence import describe, require

SEEDS = [7, 17, 27]
RARE = (WEB_BASED, BRUTE_FORCE)
METHODS = ('sqrt_population', 'equal_class')
EXPONENTS = (0., .25, .5, .75, 1.)


def candidates():
    # Listed order is the final tie-breaker. The first entry is the unchanged
    # baseline, not an eligible "improvement" candidate.
    return [dict(id=f'{method}_a{i}', method=method, exponent=alpha)
            for method in METHODS for i, alpha in enumerate(EXPONENTS)]


BASELINE = candidates()[0]['id']


def probabilities_ok(p):
    require(isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] == 8
            and len(p) > 0 and np.issubdtype(p.dtype, np.floating)
            and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            and np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-12),
            'Invalid canonical eight-class probabilities')


def predict(probabilities, population_counts, exponent):
    """No labels or stateful fitting enter the deployment-time calculation.

    Use the full fitting-partition prior, not the capped training sample or
    calibration/audit frequencies. Alpha zero uses raw argmax exactly (including
    ties). Other rules add alpha*log(prior) to log probabilities, with a tiny
    positive floor only for numerical safety. Equal scores favour the first
    class in the repository's canonical order. Inputs are never changed.
    """
    probabilities_ok(probabilities)
    pop = np.asarray(population_counts)
    require(pop.shape == (8,) and np.issubdtype(pop.dtype, np.integer)
            and np.all(pop > 0) and type(exponent) in (int, float)
            and not isinstance(exponent, bool) and exponent in EXPONENTS,
            'Invalid fitting prior or unplanned exponent')
    if exponent == 0:
        return probabilities.argmax(axis=1).astype(np.uint8)
    prior = pop.astype(np.float64)/float(pop.sum())
    adjusted = np.log(np.maximum(probabilities, np.finfo(np.float64).tiny)) + exponent*np.log(prior)
    return adjusted.argmax(axis=1).astype(np.uint8)


def screen(base, other):
    return dict(macro_f1_higher=other['macro_f1'] > base['macro_f1'],
        benign_false_alerts_not_higher=other['benign_false_alert_rate'] <= base['benign_false_alert_rate'],
        rare_f1_higher={c: other['per_class'][c]['f1'] > base['per_class'][c]['f1'] for c in RARE})


def passes(checks):
    return checks['macro_f1_higher'] and checks['benign_false_alerts_not_higher'] and all(checks['rare_f1_higher'].values())


def select(calibration):
    """One shared rule for all seeds. This function cannot see audit results.

    Require every safety/benefit check at EVERY paired seed. Among eligible
    alternatives, maximize mean rare-class F1, then mean macro-F1, then prefer
    the earlier declared rule. No eligible alternative means baseline fallback,
    never a claimed improvement or a reason to enlarge the candidate set.
    """
    defs = candidates()
    require([v['seed'] for v in calibration] == SEEDS
            and all(set(v['scores']) == {c['id'] for c in defs} for v in calibration),
            'Need every fixed rule and paired seed')
    table = []
    for rule in defs:
        scores = [v['scores'][rule['id']]['metrics'] for v in calibration]
        checks = [dict(seed=v['seed'], **screen(v['scores'][BASELINE]['metrics'], other))
                  for v, other in zip(calibration, scores)]
        table.append(dict(rule=rule, checks=checks,
            eligible=rule['id'] != BASELINE and all(passes(c) for c in checks),
            mean_rare_f1=float(np.mean([m['per_class'][c]['f1'] for m in scores for c in RARE])),
            mean_macro_f1=float(np.mean([m['macro_f1'] for m in scores]))))
    eligible = [v for v in table if v['eligible']]
    chosen = max(eligible, key=lambda v: (v['mean_rare_f1'], v['mean_macro_f1'])) if eligible else table[0]
    return dict(selected_rule=chosen['rule'], eligible_alternatives=len(eligible),
                fallback_to_baseline=not bool(eligible), table=table)


def summarize(cases, selection):
    require([c['seed'] for c in cases] == SEEDS, 'Missing audit seed')
    methods = {}
    for method in ('baseline', 'equal_raw', 'selected'):
        values = [v['scores'][method]['metrics'] for v in cases]
        methods[method] = dict(
            **{k: describe([m[k] for m in values]) for k in ('macro_f1', 'benign_false_alert_rate')},
            per_class={c: {k: describe([m['per_class'][c][k] for m in values])
                          for k in ('precision', 'recall', 'f1')} for c in CLASSES})
    checks = [dict(seed=v['seed'], **screen(v['scores']['baseline']['metrics'], v['scores']['selected']['metrics']))
              for v in cases]
    return dict(methods=methods, screens=checks,
                continuation_screen_passed=not selection['fallback_to_baseline'] and all(passes(c) for c in checks))
