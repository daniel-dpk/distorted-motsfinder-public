#!/usr/bin/env python3

import unittest
import sys

import numpy as np

from testutils import DpkTestCase, slowtest
from ..metric import BrillLindquistMetric
from .curve import StarShapedCurve
from .newton import newton_kantorovich, InsufficientResolutionOrRoundoff
from .newton import StepLimitExceeded


class TestNewton(DpkTestCase):
    def setUp(self):
        metric = BrillLindquistMetric(d=1.4)
        self.__c0 = StarShapedCurve.create_sphere(radius=2.0, num=1, metric=metric)

    def assertExpansionSmall(self, curve, num, tol):
        space = np.linspace(0, np.pi, num+1, endpoint=False)[1:]
        self.assertListAlmostEqual(curve.expansions(space),
                                   np.zeros(num).tolist(),
                                   delta=tol)

    def test_AH(self):
        c0 = self.__c0.resample(30)
        c = newton_kantorovich(c0, atol=1e-6, auto_resolution=False,
                               accurate_test_res=None)
        self.assertExpansionSmall(c, 100, 1e-6)

    @slowtest
    def test_silentWithoutDisp(self):
        c0 = self.__c0.resample(20)
        c = newton_kantorovich(c0, atol=1e-14, auto_resolution=False,
                               accurate_test_res=100, disp=False)
        # It can't have converged. Check that we're still at res=20.
        self.assertEqual(c.h.N, 20)

    @slowtest
    def test_maxSteps(self):
        c0 = self.__c0.resample(80)
        with self.assertRaises(StepLimitExceeded):
            newton_kantorovich(c0, atol=1e-14, auto_resolution=False,
                               accurate_test_res=None, steps=5)

    @slowtest
    def test_maxStepsNoDisp(self):
        c0 = self.__c0.resample(80)
        # No convergence expected. But we shouldn't raise NoConvergence here.
        newton_kantorovich(c0, atol=1e-14, auto_resolution=False,
                           accurate_test_res=None, steps=5, disp=False)

    @slowtest
    def test_AH_precise(self):
        c0 = self.__c0.resample(80)
        c = newton_kantorovich(c0, atol=1e-14, auto_resolution=False,
                               accurate_test_res=None)
        self.assertExpansionSmall(c, 100, 1e-14)

    @slowtest
    def test_raisesWhenNotConverging(self):
        c0 = self.__c0.resample(20)
        with self.assertRaises(InsufficientResolutionOrRoundoff):
            newton_kantorovich(c0, atol=1e-14, auto_resolution=False)

    @slowtest
    def test_autoResolution(self):
        c0 = self.__c0.resample(20)
        c = newton_kantorovich(c0, atol=1e-14, auto_resolution=True,
                               max_resolution=100)
        self.assertGreater(c.h.N, 40)
        self.assertExpansionSmall(c, 100, 1e-14)

    @slowtest
    def test_raisesWhenMaxRes(self):
        c0 = self.__c0.resample(20)
        with self.assertRaises(InsufficientResolutionOrRoundoff):
            newton_kantorovich(c0, atol=1e-14, auto_resolution=True,
                               max_resolution=30)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
