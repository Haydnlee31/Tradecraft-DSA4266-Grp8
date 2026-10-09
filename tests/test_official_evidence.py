"""Teammates can audit the ledger without data, checkpoints or an accelerator."""
import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from src.data.label_map import BENIGN, CLASSES
from src.eval.official_evidence import (
    close_tree, describe, digest, export, metrics_from_counts, read_json, safe_path, summarize,
)


def fixture():
    supports = dict.fromkeys(CLASSES, 10)
    cm = [[10 if i == j else 0 for j in range(8)] for i in range(8)]
    # One missed attack, and two benign false alerts are different error types.
    cm[1][1], cm[1][0] = 9, 1
    cm[0][0], cm[0][2] = 8, 2
    metrics, errors = metrics_from_counts(cm, supports)
    return dict(scope='official39_validation_only', classes=CLASSES, validation_class_counts=supports,
                groups={'control': {'scope': 'fixture', 'conditions': {'a': [7], 'b': [7]}}},
                comparisons=[dict(id='pair', group='control', left='b', right='a', scope='fixture')],
                dataset_opened=False, checkpoint_opened=False, model_trained=False, test_evaluated=False,
                deployment_authorized=False, model_promoted=False,
                records=[dict(id=c, group='control', condition=c, seed=7, metrics=copy.deepcopy(metrics),
                              errors=copy.deepcopy(errors)) for c in ('a', 'b')])


class OfficialEvidenceTests(unittest.TestCase):
    def test_metrics_and_error_categories(self):
        evidence = fixture()
        r = evidence['records'][0]
        self.assertEqual(r['errors'][BENIGN]['false_alerts'], 2)
        self.assertEqual(r['errors'][CLASSES[1]]['predicted_benign'], 1)
        self.assertEqual(r['errors'][CLASSES[1]]['wrong_attack_category'], 0)
        summary = summarize(evidence)
        self.assertIsNone(summary['groups']['control']['a']['macro_f1']['sample_sd'])
        self.assertEqual(summary['paired_differences']['pair']['macro_f1']['values'], [0.])

    def test_sample_sd_and_all_seed_values(self):
        self.assertEqual(describe([1., 2., 3.]), dict(values=[1., 2., 3.], mean=2., sample_sd=1.))

    def test_reject_corrupt_counts_and_population(self):
        for value in (-1, .5, True, float('nan')):
            e = fixture()
            e['records'][0]['metrics']['confusion_matrix'][0][0] = value
            with self.assertRaises(ValueError):
                summarize(e)
        e = fixture()
        e['validation_class_counts'][BENIGN] += 1
        with self.assertRaisesRegex(ValueError, 'population'):
            summarize(e)

    def test_reject_stale_metrics_and_error_counts(self):
        for key in ('metrics', 'errors'):
            e = fixture()
            if key == 'metrics':
                e['records'][0]['metrics']['macro_f1'] += .01
            else:
                e['records'][0]['errors'][BENIGN]['false_alerts'] += 1
            with self.assertRaises(ValueError):
                summarize(e)

    def test_reject_missing_duplicate_and_mismatched_seeds(self):
        e = fixture()
        e['records'].pop()
        with self.assertRaises(ValueError):
            summarize(e)
        e = fixture()
        e['records'].append(copy.deepcopy(e['records'][0]))
        with self.assertRaises(ValueError):
            summarize(e)
        e = fixture()
        e['groups']['control']['conditions']['b'] = [17]
        e['records'][1]['seed'] = 17
        with self.assertRaisesRegex(ValueError, 'Paired seeds'):
            summarize(e)

    def test_scope_class_order_and_action_guards(self):
        for key, value in [('scope', 'legacy46'), ('classes', list(reversed(CLASSES))),
                           ('test_evaluated', True), ('model_trained', True),
                           ('deployment_authorized', True), ('model_promoted', True)]:
            e = fixture()
            e[key] = value
            with self.assertRaises(ValueError):
                summarize(e)

    def test_reject_json_nan_and_duplicate_keys(self):
        for raw in ('{"x": NaN}', '{"x": Infinity}', '{"x": 1, "x": 2}'):
            with self.assertRaises(ValueError):
                read_json(raw)

    def test_source_containment(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('../outside.json', '/absolute.json'):
                with self.assertRaises(ValueError):
                    safe_path(folder, name)

    def test_hash_pinned_export_and_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            e = fixture()
            source = root/'source.json'
            source.write_text(json.dumps({'test_metrics': None, 'metrics': e['records'][0]['metrics']}))
            plan = {k: e[k] for k in ('scope', 'classes', 'validation_class_counts', 'groups', 'comparisons')}
            plan['sources'] = {'fixture': dict(path='source.json', sha256=digest(source), guards={'/test_metrics': None})}
            plan['records'] = [{k: v for k, v in r.items() if k not in ('metrics', 'errors')} | dict(
                source='fixture', pointer='/metrics', format='metrics') for r in e['records']]
            path = root/'plan.json'
            path.write_text(json.dumps(plan))
            first = export(path, root, None)
            self.assertEqual(first, export(path, root, None))
            self.assertEqual(first['summary'], summarize(e))
            source.write_text(source.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                export(path, root, None)
            plan['sources']['fixture']['sha256'] = digest(source)
            plan['sources']['fixture']['guards']['/test_metrics'] = 'evaluated'
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                export(path, root, None)

    def test_archive_members_and_history_budgets(self):
        with tempfile.TemporaryDirectory() as folder:
            root, e = Path(folder), fixture()
            metrics = e['records'][0]['metrics']
            receipt = dict(history=[dict(step=1, validation_metrics=metrics,
                                        examples_processed=80, optimizer_steps=2)],
                           examples_processed=80, optimizer_steps=2, test_metrics=None)
            raw = json.dumps(receipt).encode()
            archive = root/'receipt.tgz'

            def write_archive(copies):
                with tarfile.open(archive, 'w:gz') as stream:
                    for _ in range(copies):
                        info = tarfile.TarInfo('run/result.json')
                        info.size = len(raw)
                        stream.addfile(info, io.BytesIO(raw))

            write_archive(1)
            plan = {k: e[k] for k in ('scope', 'classes', 'validation_class_counts', 'groups', 'comparisons')}
            plan['sources'] = {'fixture': dict(path=archive.name, member='run/result.json',
                                               sha256=digest(archive), guards={'/test_metrics': None})}
            plan['records'] = [{k: v for k, v in r.items() if k not in ('metrics', 'errors')} | dict(
                source='fixture', pointer='/history/0/validation_metrics', format='metrics', endpoint=1,
                examples_processed=80, optimizer_steps=2) for r in e['records']]
            path = root/'plan.json'
            path.write_text(json.dumps(plan))
            self.assertEqual(export(path, root, root)['summary'], summarize(e))
            self.assertFalse((root/'run').exists())  # Members were never extracted.
            plan['records'][0]['examples_processed'] = 81
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'work mismatch'):
                export(path, root, root)
            write_archive(2)
            plan['sources']['fixture']['sha256'] = digest(archive)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                export(path, root, root)

    def test_published_evidence_recomputes_without_local_artifacts(self):
        root = Path(__file__).resolve().parents[1]/'reports/full_data_extension'
        e = read_json((root/'decision-evidence.json').read_bytes())
        self.assertEqual(e['plan_sha256'], digest(root/'decision-evidence-plan.json'))
        close_tree(e['summary'], summarize(e))
        self.assertEqual(len(e['records']), 72)  # Endpoints, NOT 72 independent runs.
        groups = e['summary']['groups']
        self.assertAlmostEqual(groups['scaling_light']['2m_step20']['macro_f1']['mean'], .65572, places=5)
        self.assertIsNone(groups['controlled_noniid']['2m_step20']['macro_f1']['sample_sd'])
        self.assertEqual(groups['controlled_noniid']['2m_step20']['per_class'][CLASSES[4]]['recall']['mean'], 0.)
        # Published source JSON still matches the pinned inputs; ignored/local
        # receipts and archives are deliberately not required by this CI test.
        repository = Path(__file__).resolve().parents[1]
        for spec in e['source_evidence'].values():
            if spec['path'].startswith('reports/'):
                self.assertEqual(digest(repository/spec['path']), spec['sha256'])


if __name__ == '__main__':
    unittest.main()
