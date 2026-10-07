"""Bounded diagnostic helpers must be deterministic and preserve class scope."""
from collections import defaultdict
import unittest

import numpy as np
import polars as pl

from reports.local_diagnostics_2026_10_07.distributions import accumulate
from reports.local_diagnostics_2026_10_07.sensitivity import backgrounds
from src.data.label_map import CLASSES


class LocalDiagnosticTests(unittest.TestCase):
    def test_backgrounds_are_seeded_bounded_and_balanced(self):
        y=np.repeat(np.arange(len(CLASSES)),20)
        original=np.arange(128)
        a=backgrounds(y,original);b=backgrounds(y,original)
        for name in a:
            np.testing.assert_array_equal(a[name],b[name])
            self.assertEqual(len(a[name]),128)
            self.assertEqual(len(np.unique(a[name])),128)
        np.testing.assert_array_equal(a['original'],original)
        np.testing.assert_array_equal(np.bincount(y[a['balanced']]),[16]*len(CLASSES))
        self.assertFalse(np.array_equal(a['original'],a['independent']))

    def test_streaming_counts_equal_whole_small_frame(self):
        frame=pl.DataFrame({'label':['DictionaryBruteForce']*3+['BenignTraffic'],
            'IAT':[.1,1.,1e8,.2],'Number':[5.5,9.5,13.5,5.5],'Weight':[38.5,141.55,244.6,38.5]})
        full={};parts={};rare_full=defaultdict(list);rare_parts=defaultdict(list)
        accumulate(frame,full,rare_full)
        accumulate(frame[:2],parts,rare_parts);accumulate(frame[2:],parts,rare_parts)
        self.assertEqual(full,parts)
        self.assertEqual(full['DictionaryBruteForce']['iat_bins'],[1,1,0,0,1])
        self.assertEqual(full['DictionaryBruteForce']['invalid'],[0,0,0])
        np.testing.assert_array_equal(np.concatenate(rare_full['DictionaryBruteForce']),np.concatenate(rare_parts['DictionaryBruteForce']))
        self.assertNotIn('BenignTraffic',rare_full)


if __name__=='__main__': unittest.main()
