"""Synthetic specialist tests: no private checkpoints or official arrays."""
import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import official39_routing_specialist as runner
from tests.test_official_routing_headroom import fixture


def checkpoint_fixture(root, change=None):
    torch.manual_seed(7)
    model = runner.frozen.MLPClassifier(39, 8, replace(runner.frozen.config_for_variant('light'), normalization='layer'))
    features = runner.frozen.read(runner.frozen.PLAN)['data_identity']['features']
    checkpoint = dict(state_dict=model.state_dict(), model_config=dict(runner.CONFIG),
                      panel_sha256='synthetic-panel', features=features, classes=runner.CLASSES,
                      seed=7, epochs=300)
    if change:
        change(checkpoint)
    path = root/'small_dropout_seed7.pt'
    torch.save(checkpoint, path)
    spec = dict(seed=7, checkpoint=path.name, sha256=runner.digest(path))
    entry = dict(condition='small_dropout', name='small_dropout_seed7', checkpoint=path.name,
        checkpoint_sha256=spec['sha256'], training_fit=dict(seed=7, model_config=dict(runner.CONFIG),
            num_parameters=5096, examples_processed=3045600, optimizer_steps=6000,
            final_state_sha256=runner.state_hash(model), observations=[dict(epoch=300)]))
    reference = dict(panel_sha256='synthetic-panel', neural_fits=[entry],
                     validation_results=[dict(name=entry['name'], seed=7)])
    return model, features, spec, reference


def paired_fixture():
    pairs = []
    for seed in (7, 17, 27):
        for lane in ('iid', 'dirichlet'):
            for fallback in runner.FALLBACKS:
                y, base, expert, conf = fixture()
                # The alternative expert recovers an extra attack but also
                # damages an additional benign prediction. Same routing order.
                if fallback == 'balanced_panel':
                    expert[12], expert[8] = 4, 2
                p = runner.headroom.pair_analysis(y, base, expert, conf, seed, [1, 5, 10])
                pairs.append(dict(p, seed=seed, base_lane=lane, fallback=fallback))
    control = [p for p in pairs if p['fallback'] == runner.FALLBACKS[0]]
    previous = dict(summary=runner.headroom.summarize(control), pairs=control)
    return pairs, previous


class RoutingSpecialistTests(unittest.TestCase):
    def test_plan_fixes_model_seeds_budgets_and_scope(self):
        plan = runner.frozen.read(runner.PLAN)
        runner.validate_plan(plan)
        for key, value in [('specialist_condition', 'wide_no_dropout'), ('seeds', [7]),
                           ('specialist_epochs', 100), ('device', 'cuda'), ('budgets_percent', [50])]:
            bad = copy.deepcopy(plan)
            bad[key] = value
            with self.assertRaisesRegex(ValueError, 'scope'):
                runner.validate_plan(bad)
        for flag in runner.headroom.FLAGS:
            bad = copy.deepcopy(plan)
            bad[flag] = True
            with self.assertRaisesRegex(ValueError, 'permission'):
                runner.validate_plan(bad)

    def test_plan_pins_prior_stage_and_all_specialists(self):
        for field in ('headroom_plan_sha256', 'headroom_runner_sha256', 'headroom_results_sha256',
                      'specialist_receipt_sha256'):
            plan = runner.frozen.read(runner.PLAN)
            plan[field] = 'changed'
            with self.assertRaises(ValueError):
                runner.validate_plan(plan)
        plan = runner.frozen.read(runner.PLAN)
        plan['specialists'].pop()
        with self.assertRaisesRegex(ValueError, 'three named'):
            runner.validate_plan(plan)

    def test_restricted_checkpoint_load_preserves_exact_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, features, spec, reference = checkpoint_fixture(root)
            with patch.object(runner.torch, 'load', wraps=torch.load) as load:
                actual, old = runner.load_specialist(root, spec, reference, features)
            self.assertTrue(load.call_args.kwargs['weights_only'])
            self.assertEqual(runner.state_hash(actual), runner.state_hash(model))
            self.assertEqual(old['seed'], 7)

    def test_checkpoint_hash_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, features, spec, reference = checkpoint_fixture(root)
            (root/spec['checkpoint']).write_bytes(b'changed')
            with patch.object(runner.torch, 'load', side_effect=AssertionError('Must not load')):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    runner.load_specialist(root, spec, reference, features)

    def test_checkpoint_rejects_wrong_epoch_seed_features_or_classes(self):
        for key, value in [('epochs', 100), ('seed', 17), ('features', ['wrong']*39),
                           ('classes', list(reversed(runner.CLASSES))), ('panel_sha256', 'wrong')]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, features, spec, reference = checkpoint_fixture(root, lambda c: c.update({key: value}))
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    runner.load_specialist(root, spec, reference, features)

    def test_checkpoint_rejects_different_architecture_and_tensor_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, features, spec, reference = checkpoint_fixture(root)
            reference['neural_fits'][0]['training_fit']['final_state_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'tensor'):
                runner.load_specialist(root, spec, reference, features)
            _, features, spec, reference = checkpoint_fixture(root, lambda c: c['model_config'].update(dropout=0))
            with self.assertRaisesRegex(ValueError, 'metadata'):
                runner.load_specialist(root, spec, reference, features)

    def test_checkpoint_rejects_paths_duplicates_and_missing_historical_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, features, spec, reference = checkpoint_fixture(root)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                runner.load_specialist(root, dict(spec, checkpoint='../outside.pt'), reference, features)
            duplicate = copy.deepcopy(reference)
            duplicate['neural_fits'] *= 2
            with self.assertRaisesRegex(ValueError, 'duplicated'):
                runner.load_specialist(root, spec, duplicate, features)
            reference['validation_results'] = []
            with self.assertRaisesRegex(ValueError, 'historical'):
                runner.load_specialist(root, spec, reference, features)

    def test_specialist_receipt_checksum_checked_before_read(self):
        with patch.object(runner, 'digest', return_value='bad'):
            with patch.object(runner.frozen, 'read', side_effect=AssertionError('Must not read')):
                with self.assertRaisesRegex(ValueError, 'receipt checksum'):
                    runner.specialist_reference('absent', dict(specialist_receipt_sha256='expected'), {})

    def test_only_reviewed_evaluator_bridge_is_allowed(self):
        plan = runner.frozen.read(runner.PLAN)
        sources = {name: runner.digest(runner.ROOT/'src'/name) for name in
                   ('models/architectures.py', 'eval/official_training_fit.py',
                    'eval/official_panel_validation.py', 'models/official_streaming.py')}
        sources['models/official_streaming.py'] = plan['reviewed_evaluator_bridge']['historical_sha256']
        reference = dict(source_sha256=sources)
        runner.verify_specialist_sources(reference, plan)
        for key in ('historical_sha256', 'current_sha256', 'evaluate_source_sha256'):
            bad = copy.deepcopy(plan)
            bad['reviewed_evaluator_bridge'][key] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'evaluator bridge'):
                runner.verify_specialist_sources(reference, bad)
        reference['source_sha256']['models/architectures.py'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'source identity'):
            runner.verify_specialist_sources(reference, plan)

    def test_pairing_requires_identical_predictions_and_selected_rows(self):
        pairs, _ = paired_fixture()
        control, treatment = pairs[:2]
        delta = runner.compare_pairs(control, treatment)
        self.assertEqual(len(delta['controls']), 6)
        for field, value in [('selected_rows_sha256', 'wrong'), ('escalated_rows', 99), ('budget_percent', 50)]:
            bad = copy.deepcopy(treatment)
            bad['controls'][0][field] = value
            with self.assertRaisesRegex(ValueError, 'selections'):
                runner.compare_pairs(control, bad)
        bad = copy.deepcopy(treatment)
        bad['base']['metrics']['macro_f1'] = 0
        with self.assertRaisesRegex(ValueError, 'Base cohort'):
            runner.compare_pairs(control, bad)

    def test_summary_keeps_all_classes_seeds_and_paired_deltas(self):
        pairs, old = paired_fixture()
        summary = runner.summarize_comparison(pairs, old)
        self.assertTrue(summary['stage_one_replay_exact'])
        self.assertTrue(summary['identical_routing_selections'])
        entry = summary['balanced_minus_centralized']['iid'][0]
        self.assertEqual(len(entry['macro_f1']['values']), 3)
        self.assertEqual(set(entry['per_class_recall']), set(runner.CLASSES))
        for lane in ('iid', 'dirichlet'):
            self.assertEqual(summary['fallbacks']['centralized_full_cohort'][lane]['base'],
                             summary['fallbacks']['balanced_panel'][lane]['base'])

    def test_summary_rejects_missing_pairs_or_changed_prior_evidence(self):
        pairs, old = paired_fixture()
        with self.assertRaisesRegex(ValueError, 'six seed'):
            runner.summarize_comparison(pairs[:-1], old)
        bad = copy.deepcopy(old)
        bad['summary']['iid']['base']['macro_f1']['mean'] = 0
        with self.assertRaisesRegex(ValueError, 'aggregate replay'):
            runner.summarize_comparison(pairs, bad)
        bad = copy.deepcopy(old)
        bad['pairs'][0]['per_class']['Benign']['both_correct'] = 0
        with self.assertRaisesRegex(ValueError, 'paired counts'):
            runner.summarize_comparison(pairs, bad)

    def test_existing_output_blocks_all_input_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner.frozen, 'read', side_effect=AssertionError('Must not read')):
                with self.assertRaisesRegex(ValueError, 'preserve'):
                    runner.run('absent', 'absent', 'absent', 'absent', tmp)

    def test_publication_rejects_incomplete_or_wrong_scope_receipt(self):
        receipt = dict(status='complete', **dict.fromkeys(runner.headroom.FLAGS, False))
        for key, value in [('status', 'incomplete'), *[(k, True) for k in runner.headroom.FLAGS]]:
            bad = dict(receipt, **{key: value})
            with patch.object(runner.frozen, 'read', return_value=bad):
                with self.assertRaisesRegex(ValueError, 'development-only'):
                    runner.publish(bad, 'synthetic.json', 'must-not-create.json')

    def test_publication_is_aggregate_only_and_never_overwrites(self):
        pairs, previous = paired_fixture()
        receipt = dict(status='complete', scope='synthetic', identity={}, validation_panel={}, model_checks=[],
                       pairs=pairs, summary=runner.summarize_comparison(pairs, previous),
                       **dict.fromkeys(runner.headroom.FLAGS, False))
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = Path(tmp)/'receipt.json', Path(tmp)/'summary.json'
            runner.frozen.save(source, receipt)
            actual_read = runner.frozen.read
            def read(path):
                return previous if path == runner.PREVIOUS else actual_read(path)
            with patch.object(runner.frozen, 'read', side_effect=read):
                runner.publish(receipt, source, destination)
                before = destination.read_bytes()
                with self.assertRaises(FileExistsError):
                    runner.publish(receipt, source, destination)
                self.assertEqual(before, destination.read_bytes())
            published = actual_read(destination)
            self.assertEqual(published['source_receipt_sha256'], runner.digest(source))
            self.assertNotIn('joint_counts', published['pairs'][0])
            self.assertEqual(len(published['pairs']), 12)


if __name__ == '__main__':
    unittest.main()
