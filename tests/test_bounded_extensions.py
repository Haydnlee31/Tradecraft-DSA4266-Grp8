"""Fail-closed replay checks; historical evidence and input cohorts are immutable."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import bounded_extensions as study


class ExtensionContractTests(unittest.TestCase):
    def test_refuses_existing_run_and_unknown_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Preserve existing'):
                study.new_folder(root)
            (root/'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Confirmation manifest'):
                study.validate_confirmation(root)

    def test_confirmation_requires_every_manifest_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'manifest.json').write_text(json.dumps({'files':{'confirmation_y.npy':'expected'}}))
            with patch.object(study,'digest',side_effect=['1f54dffeb692fc4fa020f5d6abd3d2f51192a6f66cabde47df06a258ab31d536','tampered']):
                with self.assertRaisesRegex(ValueError, 'Confirmation file differs'):
                    study.validate_confirmation(root)

    def test_historical_source_bytes_are_unchanged(self):
        saved = study.read(study.ROOT/'studies/primary/preserved-sources.json')
        for path, expected in saved['files'].items():
            self.assertEqual(study.digest(study.ROOT/path),expected,path)


if __name__=='__main__':
    unittest.main()
