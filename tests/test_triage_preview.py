"""The preview must preserve predictions and never fabricate a review outcome."""
import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import official39_triage as runner
from src.eval import triage_preview as view


def fixture():
    y = np.tile(np.arange(8, dtype=np.int64), 100)
    predictions = y.copy()
    predictions[::3] = (predictions[::3]+1) % 8
    probabilities = np.full((len(y), 8), .1, dtype=np.float32)
    probabilities[np.arange(len(y)), predictions] = .3
    scores = {m: np.linspace(-1, 1, len(y)) for m in view.METHODS}
    historical = dict(seed=7, lane='iid', base=runner.policy.headroom.metrics(y, predictions), reviews=[])
    for m in view.METHODS:
        for b in (1, 5, 10):
            threshold = 1-b/100*2
            old = runner.policy.review_counts(y, predictions, scores[m], threshold, b)
            historical['reviews'].append(dict(rule=m, budget_percent=b, threshold=threshold, **old))
    return np.arange(len(y)), predictions, probabilities, scores, historical


class TriagePreviewTests(unittest.TestCase):
    def test_review_keeps_every_class_and_does_not_mutate(self):
        _, p, _, scores, _ = fixture()
        original = p.copy()
        retained, flags = view.apply_review(p, scores['learned'], .5)
        np.testing.assert_array_equal(p, original)
        np.testing.assert_array_equal(retained, p)
        self.assertFalse(np.shares_memory(retained, p))
        np.testing.assert_array_equal(flags, scores['learned'] >= .5)

    def test_whole_ties_and_null_threshold(self):
        p = np.arange(8)
        _, flags = view.apply_review(p, np.ones(8), 1.)
        self.assertTrue(flags.all())
        _, flags = view.apply_review(p, np.ones(8)*1e10, None)
        self.assertFalse(flags.any())

    def test_prediction_and_score_validation(self):
        for p, s, t in [(np.array([8]), np.ones(1), 0), (np.array([1.]), np.ones(1), 0),
                         (np.array([1]), np.array([np.nan]), 0), (np.array([1]), np.ones(1), np.inf),
                         (np.array([], dtype=int), np.ones(0), 0), (np.array([1]), np.ones(2), 0)]:
            with self.assertRaises(ValueError):
                view.apply_review(p, s, t)

    def test_preview_is_bounded_source_order_and_contains_no_true_class(self):
        rows, p, probs, scores, _ = fixture()
        shown, flags = view.preview_rows(rows, p, probs.max(1), scores['random'], 0, 'random')
        expected = np.sort(np.r_[np.flatnonzero(flags)[:8], np.flatnonzero(~flags)[:8]])
        self.assertEqual([r['row_ref'] for r in shown], [f'validation:{i}' for i in expected])
        self.assertEqual(len(shown), 16)
        for r in shown:
            self.assertEqual(set(r), {'row_ref', 'prediction', 'max_softmax_uncalibrated', 'review_score', 'needs_review', 'status', 'reason'})
            self.assertEqual(r['status'], 'Needs review' if r['needs_review'] else 'Not flagged')
            self.assertNotIn('safe', r['status'].lower())

    def test_preview_handles_no_flags_and_invalid_rows(self):
        rows, p, probs, scores, _ = fixture()
        shown, flags = view.preview_rows(rows, p, probs.max(1), scores['learned'], None, 'learned')
        self.assertEqual(len(shown), 8)
        self.assertFalse(flags.any())
        for bad_rows in (rows[::-1], np.zeros_like(rows), rows.astype(float)):
            with self.assertRaises(ValueError):
                view.preview_rows(bad_rows, p, probs.max(1), scores['learned'], 0, 'learned')

    def test_overruns_are_exposed_not_trimmed(self):
        flags = np.ones(800, dtype=bool)
        result = view.workload(flags, 5)
        self.assertEqual(result['referrals'], 800)
        self.assertEqual(result['illustrative_cohort_cap'], 40)
        self.assertEqual(result['over_cap'], 760)
        self.assertFalse(result['budget_respected'])
        self.assertFalse(result['policy_approved'])
        self.assertFalse(result['classification_labels_changed'])
        self.assertTrue(flags.all())

    def test_workload_requires_boolean_flags_and_reviewed_budget(self):
        for flags, budget in [(np.ones(8), 5), (np.ones(8, dtype=bool), 50),
                              (np.ones(0, dtype=bool), 1), (np.ones((2, 4), dtype=bool), 5)]:
            with self.assertRaises(ValueError):
                view.workload(flags, budget)

    def test_replay_matches_all_historical_review_counts(self):
        rows, p, probs, scores, evidence = fixture()
        result, checks = runner.replay_pair(7, 'iid', rows, p, probs, scores, evidence)
        self.assertEqual(len(result['scenarios']), 9)
        self.assertTrue(checks['predictions_unchanged'])
        self.assertTrue(all(s['historical_count_exact'] for s in checks['scenarios']))

    def test_aggregate_outcomes_do_not_select_preview_examples(self):
        rows, p, probs, scores, evidence = fixture()
        changed = copy.deepcopy(evidence)
        changed['base']['metrics']['macro_f1'] = .999
        for scenario in changed['reviews']:
            scenario['per_class'] = {'not_a_routing_input': 'not used'}
        one, _ = runner.replay_pair(7, 'iid', rows, p, probs, scores, evidence)
        two, _ = runner.replay_pair(7, 'iid', rows, p, probs, scores, changed)
        self.assertEqual([s['preview'] for s in one['scenarios']], [s['preview'] for s in two['scenarios']])

    def test_misaligned_counts_or_missing_policies_fail_closed(self):
        rows, p, probs, scores, evidence = fixture()
        changed = copy.deepcopy(evidence)
        changed['reviews'][0]['calls'] += 1
        with self.assertRaisesRegex(ValueError, 'count'):
            runner.replay_pair(7, 'iid', rows, p, probs, scores, changed)
        changed['reviews'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            runner.replay_pair(7, 'iid', rows, p, probs, scores, changed)

    def test_invalid_probabilities_fail_closed(self):
        rows, p, probs, scores, evidence = fixture()
        for bad in (probs[:, :7], probs*0, probs*np.nan, probs.astype(np.float64)):
            with self.assertRaisesRegex(ValueError, 'probabilities'):
                runner.replay_pair(7, 'iid', rows, p, bad, scores, evidence)

    def test_private_output_cannot_be_in_reports(self):
        with patch.object(runner, 'load_evidence', side_effect=AssertionError('No input reads')):
            with self.assertRaisesRegex(ValueError, 'outputs/'):
                runner.run('absent', 'absent', 'absent', runner.ROOT/'reports/triage-not-allowed')

    def test_existing_output_preserved_before_any_reads(self):
        with patch.object(runner, 'load_evidence', side_effect=AssertionError('No input reads')):
            with tempfile.TemporaryDirectory() as tmp, patch.object(runner, 'ROOT', Path(tmp)):
                out = Path(tmp)/'outputs/existing'
                out.mkdir(parents=True)
                with self.assertRaisesRegex(ValueError, 'Preserve'):
                    runner.run('absent', 'absent', 'absent', out)

    def test_published_evidence_digest_and_aggregate_audit(self):
        self.assertEqual(runner.load_evidence()['status'], 'complete')
        with patch.object(runner, 'EVIDENCE_SHA256', 'wrong'):
            with self.assertRaisesRegex(ValueError, 'changed'):
                runner.load_evidence()

    def test_html_embeds_valid_json_without_script_injection(self):
        payload = dict(protocol='official39-triage-preview-v1', read_only=True,
            ground_truth_rows_opened=False, operational_budget_approved=False,
            note='</script><script>alert(1)</script> & <img>')
        html = view.render(payload)
        self.assertNotIn(payload['note'], html)
        class Scripts(HTMLParser):
            def __init__(self):
                super().__init__(); self.scripts = []; self.current = None
            def handle_starttag(self, tag, attrs):
                if tag == 'script': self.current = ''; self.scripts.append(dict(attrs))
            def handle_data(self, data):
                if self.current is not None: self.current += data
            def handle_endtag(self, tag):
                if tag == 'script': self.scripts[-1]['body'] = self.current; self.current = None
        parser = Scripts(); parser.feed(html)
        self.assertEqual(len(parser.scripts), 2)
        self.assertEqual(json.loads(parser.scripts[0]['body']), payload)
        self.assertIn("connect-src 'none'", html)
        self.assertNotIn('src=', html)
        self.assertNotIn('innerHTML', html)

    def test_template_has_all_named_accessible_controls_and_no_remote_assets(self):
        class Elements(HTMLParser):
            def __init__(self):
                super().__init__(); self.ids = []; self.options = []; self.active = None
            def handle_starttag(self, tag, attrs):
                a = dict(attrs)
                if 'id' in a: self.ids.append(a['id'])
                if tag == 'select': self.active = a['id']
                if tag == 'option' and 'value' in a: self.options.append((self.active, a['value']))
        template = view.TEMPLATE.read_text()
        parser = Elements(); parser.feed(template)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        self.assertTrue({'lane', 'seed', 'rule', 'budget', 'preview', 'classes', 'triage-data'}.issubset(parser.ids))
        self.assertEqual({v for k, v in parser.options if k == 'rule'}, set(view.METHODS))
        self.assertEqual({v for k, v in parser.options if k == 'budget'}, {'1', '5', '10'})
        self.assertNotIn('https://', template)
        self.assertNotIn('http://', template)
        self.assertIn('<noscript>', template)
        self.assertIn('scope="col"', template)


if __name__ == '__main__':
    unittest.main()
