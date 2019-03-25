#!/usr/bin/env python

import unittest
import sys

from testutils import DpkTestCase, slowtest
from ..metric import BrillLindquistMetric
from .initialguess import InitHelper


class TestInitialguess(DpkTestCase):
    @slowtest
    def test_brill_lindquist(self):
        g = BrillLindquistMetric(d=0.6, m1=0.2, m2=0.8)
        h = InitHelper(metric=g, verbose=False)
        curves = h.find_four_MOTSs(m1=g.m1, m2=g.m2, d=g.d, plot=False)
        self.assertEqual(len(curves), 4)
        self.assertTrue(all(curves))


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
