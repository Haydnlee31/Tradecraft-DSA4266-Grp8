"""The focused entry point must compare every arm and never hide differences."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import study


class StudyEntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(study.EVIDENCE.read_text(encoding='utf-8'))['summary']

    def test_all_primary_endpoints_are_checked(self):
        result = study.compare(self.reference, self.reference)
        self.assertEqual(result['endpoints'], 33)
        self.assertTrue(all(all(r['exact'].values()) for r in result['checks']))

    def test_registry_and_frozen_sources_remain_consistent(self):
        registry = json.loads((study.ROOT/'studies/primary/study.json').read_text())
        plan = json.loads((study.ROOT/registry['plan']).read_text())
        self.assertEqual(registry['seeds'], plan['seeds'])
        self.assertEqual(registry['configurations'], len(plan['arms']))
        self.assertEqual(registry['backward_examples_per_run'], plan['backward_examples_per_run'])
        sources = json.loads((study.ROOT/'studies/primary/preserved-sources.json').read_text())
        for name, digest in sources['files'].items():
            self.assertEqual(hashlib.sha256((study.ROOT/name).read_bytes()).hexdigest(), digest, name)

    def test_changed_supporting_method_cannot_pass(self):
        # Historical bridge tests covered four controls. The new comparison must
        # also catch changes in a supporting arm's weights/settings/sample stream.
        for field in ('model_sha256', 'sample_chain_sha256', 'settings'):
            candidate = copy.deepcopy(self.reference)
            row = next(r for r in candidate['results'] if r['arm'] == 'fedprox001')
            row[field] = {'lr': 123} if field == 'settings' else 'changed'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Endpoint differs'):
                study.compare(candidate, self.reference)

    def test_missing_or_duplicate_run_is_rejected(self):
        for duplicate in (False, True):
            candidate = copy.deepcopy(self.reference)
            candidate['results'].pop()
            if duplicate:
                candidate['results'].append(candidate['results'][0])
            with self.assertRaisesRegex(ValueError, 'Missing/duplicate'):
                study.compare(candidate, self.reference)

    def test_setup_reuses_interpreter_without_training(self):
        with patch('sys.argv', ['study.py', 'setup']), patch.object(study, 'call') as call:
            study.main()
            call.assert_called_once_with('scripts/reproduce_shared_adam.py', 'setup',
                                         '--python', study.sys.executable)

    def test_invalid_paths_stop_before_subprocess(self):
        with tempfile.TemporaryDirectory() as root:
            for argv in (['reproduce', '--output', root], ['train'], ['evaluate'],
                         ['check', '--archive', 'file.tgz']):
                with patch('sys.argv', ['study.py', *argv]), patch.object(study, 'call') as call:
                    with self.assertRaises(SystemExit):
                        study.main()
                    call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
