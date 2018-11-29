#!/usr/bin/env python3

import unittest
import sys

import numpy as np

from testutils import DpkTestCase
from .metric import SchwarzschildSliceMetric


class TestSchwarzschildMetric(DpkTestCase):
    def test_general(self):
        pi = np.pi
        Id = np.identity(3)
        metric = SchwarzschildSliceMetric(m=2)
        self.assertAlmostEqual(metric.horizon_area(), 16*pi*metric.m**2)
        self.assertAlmostEqual(metric.horizon_coord_radius(), metric.m/2)
        g = metric.at([1, 0, 0])
        self.assertTrue(np.allclose(g.mat, 16*Id))
        x = np.linspace(1, 10, 5)
        y = z = np.zeros_like(x)
        points = np.array([x, y, z]).T
        for p in points:
            g = metric.at(p)
            self.assertEqual(g.mat.shape, (3, 3))
            np.testing.assert_allclose(g.inv, metric.psi(p)**(-4) * Id)
        g = metric.at([5., 4., 3.])
        v = [1.2, 3.4, -5.6]
        v_up = g.raise_idx(v)
        w = g.inv.dot(v)
        np.testing.assert_allclose(v_up, w)
        v_down = g.lower_idx(v_up)
        np.testing.assert_allclose(v, v_down)

    def test_diff(self):
        metric = SchwarzschildSliceMetric(m=2)
        p = [1.2, 3.4, 5.6]
        for inverse in (False, True):
            metric.force_fd = True
            dg_fd = metric.diff(p, inverse=inverse)
            metric.force_fd = False
            dg_sd = metric.diff(p, inverse=inverse)
            np.testing.assert_allclose(dg_fd, dg_sd, atol=1e-7)

    def test_diff_lnsqrtg(self):
        metric = SchwarzschildSliceMetric(m=2)
        p = [1.2, 3.4, 5.6]
        metric.force_fd = True
        v1 = metric.diff_lnsqrtg(p)
        metric.force_fd = False
        v2 = metric.diff_lnsqrtg(p)
        np.testing.assert_allclose(v1, v2, atol=1e-7)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
