"""Thesis visuals must use frozen aggregate evidence, never new predictions."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import shared_optimizer_thesis_figures as figures
from scripts.render_thesis_pdf import inline


class ThesisFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.history, cls.data = figures.evidence()
        cls.derived = figures.extract(cls.summary, cls.history, cls.data)

    def test_primary_roster_and_published_means(self):
        d = self.derived
        self.assertEqual(d['seeds'], [7,17,27])
        self.assertEqual(set(d['points']), {'100','500','1000','2000'})
        f1 = np.asarray(d['points']['500']['macro_f1'])
        self.assertEqual(f1.shape, (4,3))
        np.testing.assert_allclose(f1.mean(axis=1),
            [.6535212148280509,.6360958362979091,.6325367323327683,.5186688260677309])
        self.assertAlmostEqual(100*(f1[2].mean()-f1[3].mean()), 11.386790626503739)

    def test_confusion_normalization_recall_and_rare_outcomes(self):
        d = self.derived
        cm = np.asarray(d['mean_confusion_row_percent'])
        np.testing.assert_allclose(cm.sum(axis=-1), 100)
        np.testing.assert_allclose(np.diagonal(cm,axis1=1,axis2=2).T, d['mean_recall_percent'])
        outcomes = np.asarray(d['rare_error_percent'])
        np.testing.assert_allclose(outcomes.sum(axis=-1), 100)
        self.assertTrue(np.all(outcomes >= 0))
        for k, idx in enumerate(d['rare_class_indices']):
            np.testing.assert_allclose(outcomes[k,:,0], cm[:,idx,idx])
        self.assertEqual([d['class_order'][k] for k in d['rare_class_indices']], ['Web-based','Brute Force'])

    def test_cohort_sizes_and_communication_not_conflated(self):
        d = self.derived
        self.assertEqual(sum(d['fit_support']), 500000)
        self.assertEqual(sum(d['confirmation_support']), 398502)
        self.assertEqual(d['development_population'], 399552)
        self.assertEqual(d['historical_population'], 2040729)
        self.assertEqual(d['synchronizations'], [0,2160,2160,108])
        self.assertEqual(d['payload_bytes'], [0,1761177600,240255840,88058880])
        self.assertAlmostEqual(d['payload_bytes'][2]/d['payload_bytes'][3], 2.7283544828187685)

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in figures.SOURCES:
                dest = root/relative; dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(figures.ROOT/relative,dest)
            first = root/next(iter(figures.SOURCES))
            first.write_bytes(first.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'Evidence identity differs'): figures.evidence(root)

    def test_wrong_support_is_rejected(self):
        s = copy.deepcopy(self.summary)
        s['data']['counts']['Benign'] += 1
        with self.assertRaisesRegex(ValueError,'class supports'):
            figures.extract(s,self.history,self.data)

    def test_committed_pack_and_every_asset_have_exact_provenance(self):
        root = figures.ROOT/'reports/federated_shared_optimizer/thesis/figures'
        receipt = json.loads((root/'manifest.json').read_text())
        self.assertEqual(receipt['generator_sha256'], figures.digest(figures.__file__))
        self.assertEqual(receipt['sources'], figures.SOURCES)
        self.assertEqual(receipt['figure_count'], 9)
        self.assertEqual(len(receipt['files']), 19)  # Nine PNG/SVG pairs plus data.
        for name, digest in receipt['files'].items(): self.assertEqual(figures.digest(root/name), digest)
        self.assertEqual(json.loads((root/'figure-data.json').read_text()), self.derived)
        for flag in ('model_trained','new_model_inference','dataset_arrays_opened','thresholds_refitted','cloud_used'):
            self.assertFalse(receipt[flag])

    def test_rebuild_without_dataset_arrays_and_preserve_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp)/'figures'
            with patch.object(np,'load',side_effect=AssertionError('No dataset arrays')):
                receipt = figures.build(dest)
            self.assertEqual(receipt['figure_count'], 9)
            self.assertTrue(all((dest/(name+'.png')).stat().st_size > 10000 for name,_,_ in figures.FIGURES))
            with self.assertRaisesRegex(ValueError,'Preserve existing'): figures.build(dest)

    def test_document_inline_escaping_and_links(self):
        self.assertEqual(inline('A < B & C'), 'A &lt; B &amp; C')
        self.assertIn('<link href="https://example.org"', inline('[Paper](https://example.org)'))
        self.assertEqual(inline('[Evidence](../evidence.json)'), 'Evidence')


if __name__ == '__main__': unittest.main()
