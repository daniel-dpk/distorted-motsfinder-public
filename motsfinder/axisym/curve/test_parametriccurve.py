#!/usr/bin/env python3

import unittest
import sys

import numpy as np

from testutils import DpkTestCase
from ...metric import BrillLindquistMetric
from ...exprs.trig import SineSeries, CosineSeries
from .parametriccurve import ParametricCurve


class TestParametriccurve(DpkTestCase):
    def test_line_segment(self):
        pi = np.pi
        a = np.array([0.1, 0.0, 1.5])
        b = np.array([0.5, 0.0, 2.0])
        c1 = ParametricCurve.create_line_segment(a, b)
        a = np.array([a[0], a[2]])
        b = np.array([b[0], b[2]])
        c2 = ParametricCurve.create_line_segment(a, b)
        np.testing.assert_allclose(c1(0), a, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(c1(pi), b, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(c2(0), a, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(c2(pi), b, rtol=1e-14, atol=1e-14)
        self.assertAlmostEqual(c1.arc_length(), np.linalg.norm(b-a))

    def test_arc_length(self):
        a = np.array([0.1, 1.5])
        b = np.array([0.5, 2.0])
        c = ParametricCurve.create_line_segment(a, b)
        metric = BrillLindquistMetric(d=1.8)
        self.assertAlmostEqual(
            c.arc_length_using_metric(metric), 2.00071133164626,
            places=12
        )

    def test_finding(self):
        c = ParametricCurve(
            SineSeries([0.23480309, -0.15654891, 0.04624369, 0.00278076]),
            CosineSeries([-0.04993168, 0.21305378, -0.16156875, 0.04760843])
        )
        x0 = c.find_line_intersection([-0.1, -0.05], [0.5, 0.05])
        self.assertAlmostEqual(x0, 2.046922008999949)
        x0 = c.find_max_x()
        self.assertAlmostEqual(x0, 2.2817278350992027)
        # Cases where we intersect multiple times and we're on the same side
        # of the line at t=0 and t=pi. Since it's not defined *which* of the
        # intersections is found, we just test that we don't have an error.
        x0 = c.find_line_intersection([-0.1, 0.05], [0.5, 0.05])
        x0 = c.find_line_intersection([-0.1, 0.0385], [0.5, 0.05])
        x0 = c.find_line_intersection([-0.1, 0.09], [0.5, 0.05])
        with self.assertRaises(RuntimeError):
            # line does not intersect
            x0 = c.find_line_intersection([-0.1, 0.1], [0.5, 0.05])


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
