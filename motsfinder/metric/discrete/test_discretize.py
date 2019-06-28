#!/usr/bin/env python3

import unittest
import sys

from testutils import DpkTestCase, slowtest
from ...axisym.initialguess import InitHelper
from ..analytical import BrillLindquistMetric
from .discretize import DiscretizedMetric


class TestDiscretize(DpkTestCase):
    @slowtest
    def test_discretizedmetric(self):
        m1 = 0.2; m2 = 0.8; d = 0.6
        gBL = BrillLindquistMetric(d=d, m1=m1, m2=m2)
        hBL = InitHelper(metric=gBL, verbose=False)
        c1 = hBL.find_AH(m1=m1, m2=m2, d=d)
        # c1 is the AH found with the analytical metric.
        # Now, create a discretized version and find the same curve.
        g = DiscretizedMetric(
            patch=DiscretizedMetric.construct_patch(res=64, radius=1),
            metric=gBL,
            curv=gBL.get_curv(),
        )
        h = InitHelper(metric=g, verbose=False)
        c2 = h.find_AH(m1=m1, m2=m2, d=d)
        c_norm = c1.inf_norm(c2)[1]
        # We can't expect high accuracy with res=64.
        self.assertAlmostEqual(c_norm, 0.0, delta=1e-4)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
