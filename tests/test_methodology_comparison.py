"""Matched adapters must preserve optimizer semantics, pairing and recovery."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from src.eval import methodology_comparison as study
from tests import test_shared_optimizer_compression as fixtures


class MethodologyComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        base, data, criterion = fixtures.SharedOptimizerCompressionTests().fixture()
        base['backward_examples_per_arm'] = 144
        base['mime']['rounds'] = 2
        return base, data, criterion

    def test_frozen_plan_has_all_controls_and_three_seeds(self):
        plan = study.read_plan()
        self.assertEqual(set(plan['arms']), {'central_adam', 'shared_adam', 'fedavg_adam',
            'central_mime_base', 'mime', 'central_sgd', 'fedavg_sgd', 'scaffold',
            'fedprox001', 'fedprox01', 'block128'})
        self.assertEqual(plan['total_training_backward_examples'], 712800000)
        self.assertFalse(plan['cloud_used'])

    def test_zero_proximal_coefficient_matches_historical_fedavg_exactly(self):
        base, data, criterion = self.fixture()
        baseline = study.make_state(base, 'fedavg_adam', .003)
        prox = study.make_state(base, 'fedprox001', .003)
        positive = study.make_state(base, 'fedprox001', .003)
        for _ in range(3):
            study.advance(baseline, data, base, 'fedavg_adam', {'lr': .003}, criterion)
            study.advance(prox, data, base, 'fedprox001', {'lr': .003, 'mu': 0.}, criterion)
            study.advance(positive, data, base, 'fedprox001', {'lr': .003, 'mu': .1}, criterion)
            self.assertTrue(study.old.block.nested_equal(study.snapshot(baseline, 'fedavg_adam'), study.snapshot(prox, 'fedprox001')))
        self.assertNotEqual(study.old.state_digest(positive['model']), study.old.state_digest(prox['model']))
        self.assertEqual(positive['sample_chain_sha256'], baseline['sample_chain_sha256'])

    def test_zero_controls_match_fedavg_sgd_first_round(self):
        base, data, criterion = self.fixture()
        fed = study.make_state(base, 'fedavg_sgd', .01)
        controlled = study.make_state(base, 'scaffold', .01)
        for arm, state in [('fedavg_sgd', fed), ('scaffold', controlled)]:
            study.advance(state, data, base, arm, {'lr': .01}, criterion)
            study.check_work(state, base, arm, data)
        self.assertEqual(study.old.state_digest(fed['model']), study.old.state_digest(controlled['model']))
        self.assertEqual(fed['sample_chain_sha256'], controlled['sample_chain_sha256'])
        self.assertEqual(controlled['work']['estimated_payload_bytes'], 2*fed['work']['estimated_payload_bytes'])
        study.scaffold.verify_controls(controlled['model'], controlled['controls'], list(map(len, data.parts)))

    def test_all_arms_recover_and_charge_exact_work(self):
        base, data, criterion = self.fixture()
        for arm, settings in study.read_plan()['arms'].items():
            with self.subTest(arm=arm):
                state = study.make_state(base, arm, settings['lr'])
                study.advance(state, data, base, arm, settings, criterion)
                saved = study.snapshot(state, arm)
                recovered = study.restore(saved, base, arm, settings['lr'])
                study.advance(state, data, base, arm, settings, criterion)
                study.advance(recovered, data, base, arm, settings, criterion)
                self.assertTrue(study.old.block.nested_equal(study.snapshot(state, arm), study.snapshot(recovered, arm)))
                study.check_work(state, base, arm, data)
                state['work']['backward_examples'] += 1
                with self.assertRaises(ValueError):
                    study.check_work(state, base, arm, data)

    def test_existing_training_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                study.old.require_fresh_output(Path(tmp))

    def test_evaluation_requires_all_frozen_runs_before_loading_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study.atomic_json({'contract': study.contract(), 'plan': study.read_plan(), 'results': []}, root/'training-summary.json')
            with patch.object(study.old.cohort, 'load_reconstructed') as load:
                with self.assertRaisesRegex(ValueError, 'Incomplete training'):
                    study.evaluate(root/'missing', root)
                load.assert_not_called()

    def test_saved_recovery_replays_actual_serialized_controls(self):
        base, data, criterion = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            state, seconds, replay = study.train_one(data, base, 'scaffold', {'lr': .01},
                                                     Path(tmp)/'run', {'test': True}, criterion)
            study.check_work(state, base, 'scaffold', data)
            self.assertEqual(state['work']['backward_examples'], 144)
            self.assertEqual(replay, 32)
            self.assertGreater(seconds, 0)


if __name__ == '__main__':
    unittest.main()
