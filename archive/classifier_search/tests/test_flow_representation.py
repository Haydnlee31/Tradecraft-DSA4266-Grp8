"""Synthetic guardrails; no real dataset, Laya weights or GPU needed in CI."""
import copy
import unittest
import numpy as np
import torch

from src.eval import flow_representation as rep
from src.eval import numerical_embedding as num
from src.eval import matched_neural as old
from src.data.label_map import CLASSES


class RepresentationTests(unittest.TestCase):
    def test_piecewise_basis_and_constants(self):
        x = np.array([[-1, 0], [.5, 2], [1.5, 9], [3, 4]], np.float32)
        out = rep.encode(x, [[0, 1, 2], [4]])
        np.testing.assert_array_equal(out, [[0, 0, 0], [.5, 0, 0], [1, .5, 0], [1, 1, 0]])
        self.assertEqual(out.dtype, np.float32)
        with self.assertRaises(ValueError):
            rep.encode(x, [[0, 0], [4]])

    def test_edges_use_only_supplied_fit_inputs(self):
        x = np.array([[0, 1], [1, 1], [2, 1]], np.float32)
        edges = rep.fit_edges(x)
        before = copy.deepcopy(edges)
        rep.encode(np.array([[100, -10]], np.float32), edges)
        self.assertEqual(edges, before)
        self.assertEqual(edges[1], [1.])
        with self.assertRaises(ValueError):
            rep.fit_edges(np.array([[np.nan]]))

    def test_serializer_omits_shortcuts_and_roundtrips_float32(self):
        names = [f'f{i}' for i in range(37)]+list(rep.MASKED)
        x = np.random.default_rng(3).normal(size=(2, 39)).astype(np.float32)
        edges = rep.fit_edges(x[:, :37])
        for mode in rep.SERIALIZERS:
            before = x.copy()
            states = rep.serialize(x, names, edges, mode)
            self.assertTrue(all(n not in states[0] for n in rep.MASKED))
            np.testing.assert_array_equal(x, before)
        fields = rep.serialize(x, names, edges, 'values')[0].split('\n')[1].split(';')
        np.testing.assert_array_equal(np.array([float(f.split('=')[1]) for f in fields], np.float32), x[0, :37])

    def test_sampling_is_fixed_disjoint_and_balanced(self):
        rows, labels = np.arange(80), np.repeat(np.arange(8), 10)
        a = rep.stratified_rows(rows, labels, 3, 7)
        np.testing.assert_array_equal(a, rep.stratified_rows(rows, labels, 3, 7))
        np.testing.assert_array_equal(np.bincount(labels[a], minlength=8), [3]*8)
        self.assertEqual(len(np.unique(a)), 24)
        with self.assertRaises(ValueError):
            rep.stratified_rows(rows, labels, 11)

    def test_gate_requires_both_rare_f1_and_no_extra_false_alerts(self):
        base = dict(macro_f1=.5, benign_false_alert_rate=.1,
                    per_class={c: dict(f1=.2) for c in CLASSES})
        good = copy.deepcopy(base)
        good['macro_f1'] = .6
        for item in good['per_class'].values():
            item['f1'] = .3
        self.assertTrue(rep.gate(good, base))
        good['benign_false_alert_rate'] = .1001
        self.assertFalse(rep.gate(good, base))
        self.assertFalse(rep.gate(base, base))

    def test_capacity_and_exact_fit_replay(self):
        torch.set_num_threads(2)
        x = np.random.default_rng(8).normal(size=(32, 12)).astype(np.float32)
        y = np.tile(np.arange(8), 4)
        pop = np.array([10]*8)
        a, ra = num.fit(x, y, pop, 7, 16, epochs=2, batch=9)
        b, rb = num.fit(x, y, pop, 7, 16, epochs=2, batch=9)
        self.assertEqual(ra, rb)
        np.testing.assert_array_equal(num.predict(a, x), num.predict(b, x))
        self.assertEqual(ra['examples_processed'], 64)
        self.assertEqual(ra['optimizer_steps'], 8)
        self.assertEqual(ra['parameters'], num.parameter_count(12, 16))
        target = num.parameter_count(120, 64)
        width = num.raw_width(120)
        self.assertLessEqual(abs(num.parameter_count(39, width)-target), 37)
        self.assertEqual(old.state_hash(a), old.state_hash(b))


if __name__ == '__main__':
    unittest.main()
