"""The cleanup must preserve evidence and separate retired imports safely."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import archive_tools as archive


class ArchiveLayoutTests(unittest.TestCase):
    def fixture(self, root):
        original = 'tests/test_old.py'
        saved = 'archive/classifier_search/' + original
        data = b'# Keep the original explanatory comment.\n'
        path = root / saved
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
        manifest = {'schema': 1, 'files': [{'original': original, 'archived': saved,
            'group': 'classifier_search', 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}]}
        (root / 'archive/manifest.json').write_text(json.dumps(manifest))
        return manifest

    def test_all_relocated_bytes_and_expected_scope(self):
        manifest = archive.verify()
        self.assertEqual(len(manifest['files']), 172)
        self.assertEqual(len(archive.archive_tests(manifest)), 22)
        self.assertEqual(sum(e['bytes'] for e in manifest['files']), 10823316)
        for entry in manifest['files']:
            self.assertEqual((archive.ROOT / entry['original']).is_file(),
                             entry.get('active_replacement', False), entry['original'])
        # Hash-pinned active reports and their figure sources were not moved.
        for name in ('thesis/thesis.md', 'federated_shared_optimizer/confirmation-results.json',
                     'full_data_extension/final-closeout.md',
                     'confirmation_2026_10_05/results/summary.json'):
            self.assertTrue((archive.ROOT / 'reports' / name).is_file())

    def test_active_imports_do_not_depend_on_retired_modules(self):
        modules = {e['original'][:-3].replace('/', '.') for e in archive.verify()['files']
                   if e['original'].endswith('.py')}
        for folder in ('src', 'scripts', 'tests'):
            for path in (archive.ROOT / folder).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                    imported = set()
                    if isinstance(node, ast.Import):
                        imported.update(a.name for a in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        imported.add(node.module or '')
                        imported.update((node.module or '') + '.' + a.name for a in node.names)
                    self.assertFalse(imported & modules, (str(path), imported & modules))

    def test_unsafe_paths_are_rejected_on_all_platforms(self):
        for value in ('', '.', '/tmp/file', '../file', 'a/../b', 'a//b',
                      'C:/file', 'a\\b', './file', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                archive.relative_path(value)

    def test_modified_archived_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            archive.verify(root)
            (root / manifest['files'][0]['archived']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                archive.verify(root)

    def test_duplicate_and_misdirected_mappings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self.fixture(root)
            bad = copy.deepcopy(baseline)
            bad['files'].append(copy.deepcopy(bad['files'][0]))
            (root / 'archive/manifest.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                archive.verify(root)
            bad = copy.deepcopy(baseline)
            bad['files'][0]['archived'] = 'archive/proposals/tests/test_old.py'
            (root / 'archive/manifest.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'mapping changed'):
                archive.verify(root)

    def test_restore_copies_only_listed_files_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repo'
            root.mkdir()
            manifest = self.fixture(root)
            (root / 'README.md').write_text('Active guide\n')
            (root / 'private.npy').write_bytes(b'do not copy')
            destination = Path(directory) / 'review'
            with patch.object(archive, 'tracked_files', return_value=[
                    'README.md', 'tests/test_old.py', 'archive/manifest.json']):
                archive.restore(destination, root)
                self.assertEqual((destination / 'README.md').read_text(), 'Active guide\n')
                self.assertFalse((destination / 'private.npy').exists())
                self.assertFalse((destination / 'archive').exists())
                entry = manifest['files'][0]
                self.assertEqual((destination / entry['original']).read_bytes(),
                                 (root / entry['archived']).read_bytes())
                with self.assertRaisesRegex(ValueError, 'Preserve existing'):
                    archive.restore(destination, root)

    def test_missing_active_file_fails_before_creating_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repo'
            root.mkdir()
            self.fixture(root)
            destination = Path(directory) / 'review'
            with patch.object(archive, 'tracked_files', return_value=['missing.py']):
                with self.assertRaisesRegex(ValueError, 'missing or unsafe'):
                    archive.restore(destination, root)
            self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
