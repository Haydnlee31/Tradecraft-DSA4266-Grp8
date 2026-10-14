"""Integrated thesis checks keep distinct studies and saved evidence intact."""
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import study_thesis as thesis


class IntegratedThesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources=thesis.sources()
        cls.history=thesis.read(thesis.ROOT/thesis.HISTORY)
        cls.derived=thesis.derive(cls.sources,cls.history)

    def test_saved_history_identity_and_complete_roster(self):
        self.assertEqual(thesis.digest(thesis.ROOT/thesis.HISTORY),thesis.HISTORY_SHA256)
        self.assertEqual(sum(len(r['history']) for r in self.history['records']),720)
        thesis.validate_histories(self.history,self.sources[thesis.LEGACY])
        self.assertFalse(self.history['model_trained'])
        self.assertFalse(self.history['checkpoint_opened'])

    def test_missing_history_or_changed_selected_checkpoint_is_rejected(self):
        bad=copy.deepcopy(self.history); bad['records'].pop()
        with self.assertRaisesRegex(ValueError,'roster'): thesis.validate_histories(bad,self.sources[thesis.LEGACY])
        bad=copy.deepcopy(self.history); bad['records'][0]['best_step']=1
        with self.assertRaisesRegex(ValueError,'selection'): thesis.validate_histories(bad,self.sources[thesis.LEGACY])

    def test_original_means_counts_and_gap_are_not_optimizer_results(self):
        d=self.derived
        np.testing.assert_allclose(np.mean(d['macro_f1'],axis=1),[.6746654900706801,.658334,.6328,.5557],atol=.00006)
        self.assertEqual(d['num_parameters'],[198984,5544,5544,5544])
        self.assertEqual(sum(d['support']),327257)
        self.assertEqual(np.asarray(d['recall']).shape,(4,3,8))
        self.assertAlmostEqual(sum(d['official_gap'].values()),.25179415788078116)
        self.assertEqual(d['recall'][3][1][4],0)
        self.assertEqual(d['recall'][3][2][4],0)

    def test_explanation_checkpoints_are_same_as_original_predictions(self):
        bad=copy.deepcopy(self.sources)
        bad[thesis.EXPLAIN]['results']['heavy-seed-0']['checkpoint_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'checkpoint'): thesis.derive(bad,self.history)
        v=np.asarray(self.derived['relative_attribution_residual'])
        self.assertTrue(np.all((v>.05)&(v<.09)))

    def test_figure_pack_all_hashes_and_flags(self):
        root=thesis.BASE/'figures'; receipt=thesis.read(root/'manifest.json')
        self.assertEqual(receipt['generator_sha256'],thesis.digest(thesis.__file__))
        self.assertEqual(len(receipt['figures']),6)
        self.assertEqual(len(receipt['files']),13)
        for name,value in receipt['files'].items(): self.assertEqual(thesis.digest(root/name),value)
        self.assertEqual(thesis.read(root/'figure-data.json'),self.derived)
        for key in ('model_trained','new_model_inference','dataset_arrays_opened','cloud_used'):
            self.assertFalse(receipt[key])

    def test_figure_rebuild_uses_no_private_arrays(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(np,'load',side_effect=AssertionError('No traffic arrays')):
            path=Path(temp)/'figures'; receipt=thesis.figures(path)
            self.assertEqual(len(receipt['figures']),6)
            with self.assertRaisesRegex(ValueError,'Preserve'): thesis.figures(path)

    def test_manuscript_sections_figures_and_links(self):
        text=(thesis.BASE/'thesis.md').read_text()
        self.assertEqual([int(v) for v in re.findall(r'^## (\d+)\. ',text,re.M)],list(range(1,23)))
        self.assertEqual([int(v) for v in re.findall(r'^Figure (\d+)\.',text,re.M)],list(range(1,16)))
        self.assertEqual(len(re.findall(r'^!\[',text,re.M)),15)
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if not target.startswith(('https://','http://')):
                self.assertTrue((thesis.BASE/target.split('#')[0]).exists(),target)
        for phrase in ('not a common leaderboard','no matched official39 heavy arm','not estimated causal effects','not a deployment-ready detector'):
            self.assertIn(phrase,text)

    def test_assembly_is_reproducible_and_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'thesis.md'
            receipt=thesis.assemble(output)
            self.assertEqual(output.read_bytes(),(thesis.BASE/'thesis.md').read_bytes())
            self.assertEqual(receipt,thesis.read(thesis.BASE/'thesis.manifest.json'))
            with self.assertRaisesRegex(ValueError,'Preserve'): thesis.assemble(output)

    def test_manuscript_bytes_survive_windows_text_newline_translation(self):
        # Emulate Windows' default text writer even when CI runs on Unix. JSON
        # receipts are parsed as JSON; the manuscript itself must be byte-exact.
        original_write_text = Path.write_text
        def windows_write(path, text, *args, **kwargs):
            if path.suffix == '.md' and kwargs.get('newline') != '\n':
                return path.write_bytes(text.replace('\n', '\r\n').encode('utf-8'))
            return original_write_text(path, text, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp, patch.object(Path, 'write_text', windows_write):
            output = Path(temp) / 'thesis.md'
            thesis.assemble(output)
            self.assertEqual(output.read_bytes(), (thesis.BASE / 'thesis.md').read_bytes())


if __name__=='__main__': unittest.main()
