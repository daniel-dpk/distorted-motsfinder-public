#!/usr/bin/env python3

import unittest
import sys

import numpy as np

from testutils import DpkTestCase
from ...metric import SchwarzschildSliceMetric, BrillLindquistMetric
from ...exprs.trig import CosineSeries
from .starshapedcurve import StarShapedCurve


class TestStarshapedcurve(DpkTestCase):
    def test_general(self):
        r"""Test creation of a sphere and Schwarzschild expansion computation."""
        metric = SchwarzschildSliceMetric(m=2)
        c = StarShapedCurve.create_sphere(radius=4, num=20, metric=metric)
        params = np.linspace(0, np.pi, 5)
        v = c.expansions(params)
        np.testing.assert_allclose(v, [0.192] * 5)
        c.h.a_n[0] = 1 # change radius
        v = c.expansions(params)
        np.testing.assert_allclose(v, [0.] * 5, atol=1e-12)
        c.h.a_n[0] = 90
        v = c.expansions(params)
        np.testing.assert_allclose(v, [0.021258779862813194] * 5)
        self.assertAlmostEqual(c.expansion(0.2), 0.02125877986281319)

    def test_diff(self):
        r"""Test functional derivatives of expansion."""
        metric = SchwarzschildSliceMetric(m=2)
        c = StarShapedCurve.create_sphere(radius=4, num=20, metric=metric)
        rs = np.linspace(.5, 20, 5)
        def f(r, hdiff=None):
            c.h.a_n[0] = r
            c.horizon_function_changed()
            return c.expansion(np.pi/5., hdiff=hdiff)
        v = [f(r) for r in rs]
        np.testing.assert_allclose(
            v,
            [-0.148148,  0.181529,  0.13318,  0.101909,  0.082065],
            atol=1e-6
        )
        v = [f(r, hdiff=0) for r in rs]
        np.testing.assert_allclose(
            v,
            [0.296296, -0.01016, -0.008124, -0.005007, -0.003301],
            atol=1e-6
        )
        v = [f(r, hdiff=1) for r in rs]
        np.testing.assert_allclose(
            v,
            [-0.611725, -0.033867, -0.010875, -0.005293, -0.003121],
            atol=1e-6
        )
        v = [f(r, hdiff=2) for r in rs]
        np.testing.assert_allclose(
            v,
            [-0.444444, -0.024606, -0.007901, -0.003846, -0.002268],
            atol=1e-6
        )

    def test_linearized_equation(self):
        r"""Test linearized equation for Brill-Lindquist data."""
        metric = BrillLindquistMetric(m1=1.4, m2=1.1, d=1.4, axis='z')
        h = CosineSeries(a_n=[6.0, 0.07, 0., 0., -0.03, -0.01])
        c = StarShapedCurve(h, metric=metric)
        params = np.linspace(0.01, np.pi-0.01, 10)
        (dhH, dhpH, dhppH), inhom = c.linearized_equation(params)
        H = -inhom
        np.testing.assert_allclose(
            H,
            [0.11832, 0.137624, 0.15999, 0.153356, 0.140799, 0.148376,
             0.157814, 0.151442, 0.140906, 0.13725],
            atol=1e-6
        )
        np.testing.assert_allclose(
            dhH,
            [0.004278, -0.001364, -0.008039, -0.00666, -0.003401, -0.00557,
             -0.007997, -0.005725, -0.002187, -0.000874],
            atol=1e-6
        )
        np.testing.assert_allclose(
            dhpH,
            [-1.868867, -0.049159, -0.021413, -0.010754, -0.00343,  0.003559,
             0.010911,  0.022319,  0.05171,  1.950668],
            atol=1e-6
        )
        np.testing.assert_allclose(
            dhppH,
            [-0.01869, -0.018528, -0.018459, -0.018729, -0.019138, -0.019226,
             -0.019162, -0.019235, -0.019414, -0.019507],
            atol=1e-6
        )

    def test_area_mass_radius(self):
        metric = SchwarzschildSliceMetric(m=2)
        # In isotropic coordinates, the horizon is located at r = m/2
        r = metric.horizon_coord_radius()
        c = StarShapedCurve.create_sphere(radius=r, num=1, metric=metric)
        self.assertAlmostEqual(c.area(), metric.horizon_area())
        self.assertAlmostEqual(c.irreducible_mass(), metric.m)
        # The Schwarzschild radius is 2m.
        self.assertAlmostEqual(c.horizon_radius(), 2*metric.m)

    def test_z_distance(self):
        metric = BrillLindquistMetric(m1=2.4, m2=0.8, d=1.6, axis='z')
        d = metric.d
        r1, r2 = 0.4, 1.1
        c1 = StarShapedCurve.create_sphere(radius=r1, num=1, metric=metric,
                                           origin=(0, d/2))
        c2 = StarShapedCurve.create_sphere(radius=r2, num=1, metric=metric,
                                           origin=(0, -d/2))
        fl = None
        self.assertAlmostEqual(c1.z_distance_using_metric(fl), d/2 - r1)
        self.assertAlmostEqual(c2.z_distance_using_metric(fl), 0.0)
        self.assertAlmostEqual(c1.z_distance(), 4.9137593596979325)
        self.assertAlmostEqual(c2.z_distance(), 0.0)
        self.assertAlmostEqual(c1.z_distance_using_metric(fl, c2), d - r1 - r2)
        self.assertAlmostEqual(c2.z_distance_using_metric(fl, c1), d - r1 - r2)
        self.assertAlmostEqual(c1.z_distance(c2), 1.6233677938493223)
        self.assertAlmostEqual(c2.z_distance(c1), 1.6233677938493223)
        c3 = StarShapedCurve.create_sphere(radius=d, num=1, metric=metric)
        self.assertAlmostEqual(c1.z_distance_using_metric(fl, c3), 0.0)
        self.assertAlmostEqual(c2.z_distance_using_metric(fl, c3), 0.0)
        self.assertAlmostEqual(c1.z_distance(c3), 0.0)
        self.assertAlmostEqual(c2.z_distance(c3), 0.0)
        self.assertAlmostEqual(c3.z_distance_using_metric(fl, c1), 0.0)
        self.assertAlmostEqual(c3.z_distance_using_metric(fl, c2), 0.0)
        self.assertAlmostEqual(c3.z_distance(c1), 0.0)
        self.assertAlmostEqual(c3.z_distance(c2), 0.0)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
