#!/usr/bin/env python3
r"""@package motsfinder.exprs.test_trig

Cosine series expression test suite.
"""

from __future__ import print_function

import unittest
import sys
import math

import numpy as np
from mpmath import mp

from testutils import DpkTestCase, slowtest
from ..utils import lmap
from ..numutils import inf_norm1d
from .trig import CosineSeries, SineSeries, FourierSeries
from .trig import evaluate_trig_mat, evaluate_trig_series


class TestSineSeries(DpkTestCase):
    r"""Test the SineSeries class."""
    def test_derivatives(self):
        pi = math.pi
        a, b = 0, pi
        space = np.linspace(a, b, 10)
        sin, cos = math.sin, math.cos
        f = lambda x: 2*sin(1*x) - 1*sin(2*x) + 3*sin(3*x)
        df = lambda x: 2*cos(1*x) - 2*cos(2*x) + 9*cos(3*x)
        ddf = lambda x: -2*sin(1*x) + 4*sin(2*x) - 27*sin(3*x)
        s = SineSeries([2, -1, 3], domain=(a, b))
        e = s.evaluator()
        self.assertListAlmostEqual(lmap(e, space), lmap(f, space), delta=1e-14)
        self.assertListAlmostEqual(lmap(e.function(1), space), lmap(df, space), delta=1e-14)
        self.assertListAlmostEqual(lmap(e.function(2), space), lmap(ddf, space), delta=1e-14)
        f = lambda x: 2*sin(2*x) - 1*sin(4*x) + 3*sin(6*x)
        df = lambda x: 4*cos(2*x) - 4*cos(4*x) + 18*cos(6*x)
        ddf = lambda x: -8*sin(2*x) + 16*sin(4*x) - 108*sin(6*x)
        a, b = 0, 10*pi
        space = np.linspace(a, b, 10)
        f = lambda x: 2 * sin(3*x/10.)
        df = lambda x: 6/10. * cos(3*x/10.)
        ddf = lambda x: -18/100. * sin(3*x/10.)
        s = SineSeries([0, 0, 2], domain=(a, b))
        e = s.evaluator()
        self.assertListAlmostEqual(lmap(e, space), lmap(f, space), delta=1e-14)
        self.assertListAlmostEqual(lmap(e.function(1), space), lmap(df, space), delta=1e-14)
        self.assertListAlmostEqual(lmap(e.function(2), space), lmap(ddf, space), delta=1e-14)

    def test_collocation_points(self):
        pi = math.pi
        num = 4
        pts = SineSeries.create_collocation_points(num, lobatto=True)
        self.assertEqual(len(pts), num)
        self.assertListAlmostEqual([p/pi for p in pts], [0.2, 0.4, 0.6, 0.8])
        num = 5
        pts = SineSeries.create_collocation_points(num, lobatto=False)
        self.assertEqual(len(pts), num)
        self.assertListAlmostEqual([p/pi for p in pts], [0.1, 0.3, 0.5, 0.7, 0.9])
        num = 4
        s = SineSeries([0]*num, domain=(0, 1))
        pts = s.collocation_points()
        self.assertEqual(len(pts), num)
        self.assertListAlmostEqual(pts, [0.2, 0.4, 0.6, 0.8])

    def test_from_function(self):
        pi = math.pi
        sin = math.sin
        a, b = 0, 10*pi
        space = np.linspace(a, b, 10)
        f = lambda x: 2 * sin(3*x/10.)
        s = SineSeries.from_function(f, 4, domain=(a, b))
        e = s.evaluator()
        self.assertListAlmostEqual(lmap(e, space), lmap(f, space), delta=1e-14)
        s = SineSeries.from_function(f, 4, (a, b), use_mp=True)
        e = s.evaluator()
        self.assertListAlmostEqual(lmap(e, space), lmap(f, space), delta=1e-14)

    def _check_convergence(self, f, num, lobatto, use_mp, tol):
        s = SineSeries.from_function(f, num, lobatto=lobatto, use_mp=use_mp)
        norm = inf_norm1d(s.evaluator(use_mp), f)[1]
        self.assertLessEqual(norm, tol)

    def _test_accuracy(self, use_mp):
        ctx = mp if use_mp else math
        convert = mp.mpf if use_mp else float
        pi = ctx.pi
        sin, cos, exp = ctx.sin, ctx.cos, ctx.exp
        with mp.workdps(50):
            five = convert(5)
            # 2pi periodic, antisym w.r.t. 0, not sym w.r.t. pi/2
            f = lambda x: exp(cos(x)) * sin(x)**2 * sin(five*x)
            f.domain = (0, pi)
            tol = 1e-30 if use_mp else 4e-15
            self._check_convergence(f, num=30, lobatto=True, use_mp=use_mp, tol=tol)
            self._check_convergence(f, num=30, lobatto=False, use_mp=use_mp, tol=tol)

    def test_accuracy_float(self):
        self._test_accuracy(False)

    @slowtest
    def test_accuracy_mpmath(self):
        self._test_accuracy(True)

    def test_copy(self):
        s1 = SineSeries([0, 0, 2], domain=(4, 8))
        s2 = s1.copy()
        self.assertEqual(s2.domain[0], 4)
        self.assertEqual(s2.domain[1], 8)
        self.assertEqual(s1.a_n, s2.a_n)
        self.assertIsNot(s1.a_n, s2.a_n)


class TestCosineSeries(DpkTestCase):
    r"""Test the CosineSeries class."""
    def test_derivative(self):
        r"""Derivatives of simple cosine series."""
        pi = math.pi
        a, b = 0, pi
        space = np.linspace(a, b, 10)
        sin, cos = math.sin, math.cos
        f = lambda x: -1 + 2*cos(1*x) - 1*cos(2*x) + 3*cos(3*x)
        df = lambda x: -2*sin(1*x) + 2*sin(2*x) - 9*sin(3*x)
        ddf = lambda x: -2*cos(1*x) + 4*cos(2*x) - 27*cos(3*x)
        s = CosineSeries([-1, 2, -1, 3], (a, b))
        e = s.evaluator()
        self.assertListAlmostEqual(lmap(e, space), lmap(f, space), delta=1e-14)
        self.assertListAlmostEqual(lmap(e.function(1), space), lmap(df, space), delta=1e-14)
        self.assertListAlmostEqual(lmap(e.function(2), space), lmap(ddf, space), delta=1e-14)
        a, b = 0, 10*pi
        space = np.linspace(a, b, 10)
        f = lambda x: 2 * cos(3*x/10.)
        df = lambda x: -6/10. * sin(3*x/10.)
        ddf = lambda x: -18/100. * cos(3*x/10.)
        s = CosineSeries([0, 0, 0, 2], (a, b))
        e = s.evaluator()
        self.assertListAlmostEqual(lmap(e, space), lmap(f, space), delta=1e-14)
        self.assertListAlmostEqual(lmap(e.function(1), space), lmap(df, space), delta=1e-14)
        self.assertListAlmostEqual(lmap(e.function(2), space), lmap(ddf, space), delta=1e-14)

    def _check_convergence(self, f, num, lobatto, use_mp, tol):
        s = CosineSeries.from_function(f, num, lobatto=lobatto, use_mp=use_mp)
        norm = inf_norm1d(s.evaluator(use_mp), f)[1]
        self.assertLessEqual(norm, tol)

    def _test_accuracy(self, use_mp):
        ctx = mp if use_mp else math
        fl = mp.mpf if use_mp else float
        pi = ctx.pi
        sin, cos, exp = ctx.sin, ctx.cos, ctx.exp
        with mp.workdps(50):
            five = fl(5)
            # 2pi periodic, sym w.r.t. 0, not sym w.r.t. pi/2
            f = lambda x: exp(cos(x)) * sin(x) * sin(five*x)
            f.domain = (0, pi)
            tol = 1e-30 if use_mp else 4e-15
            self._check_convergence(f, num=30, lobatto=True, use_mp=use_mp, tol=tol)
            self._check_convergence(f, num=30, lobatto=False, use_mp=use_mp, tol=tol)

    def test_accuracy_float(self):
        r"""Test that functions are approximated accurately (float)."""
        self._test_accuracy(False)

    @slowtest
    def test_accuracy_mpmath(self):
        r"""Test that functions are approximated accurately (mpmath)."""
        self._test_accuracy(True)

    def test_copy(self):
        r"""Test that copies of series are independent."""
        s1 = CosineSeries([0, 0, 2], domain=(4, 8))
        s2 = s1.copy()
        self.assertIs(type(s2), CosineSeries)
        self.assertEqual(s2.domain[0], 4)
        self.assertEqual(s2.domain[1], 8)
        self.assertEqual(s1.a_n, s2.a_n)
        self.assertIsNot(s1.a_n, s2.a_n)


class TestFourierSeries(DpkTestCase):
    r"""Test the FourierSeries class."""
    def test_derivative(self):
        r"""Derivatives of simple Fourier series."""
        pi = math.pi
        a, b = 0, 2*pi
        space = np.linspace(a, b, 10)
        sin, cos = math.sin, math.cos
        f = lambda x: -1 + 2*sin(1*x) - 1*cos(2*x) + 3*cos(3*x)
        df = lambda x: 2*cos(1*x) + 2*sin(2*x) - 9*sin(3*x)
        ddf = lambda x: -2*sin(1*x) + 4*cos(2*x) - 27*cos(3*x)
        s = FourierSeries([-1, 0, 2, -1, 0, 3], (a, b))
        e = s.evaluator()
        self.assertListAlmostEqual(lmap(e, space), lmap(f, space), delta=1e-14)
        self.assertListAlmostEqual(lmap(e.function(1), space), lmap(df, space), delta=1e-14)
        self.assertListAlmostEqual(lmap(e.function(2), space), lmap(ddf, space), delta=1e-14)
        a, b = 0, 20*pi
        space = np.linspace(a, b, 10)
        f = lambda x: 2 * cos(3*x/10.) - sin(4*x/10.)
        df = lambda x: -6/10. * sin(3*x/10.) - 4/10. * cos(4*x/10.)
        ddf = lambda x: -18/100. * cos(3*x/10.) + 16/100. * sin(4*x/10.)
        s = FourierSeries([0, 0, 0, 0, 0, 2, 0, 0, -1, 0], (a, b))
        e = s.evaluator()
        self.assertListAlmostEqual(lmap(e, space), lmap(f, space), delta=1e-14)
        self.assertListAlmostEqual(lmap(e.function(1), space), lmap(df, space), delta=1e-14)
        self.assertListAlmostEqual(lmap(e.function(2), space), lmap(ddf, space), delta=1e-14)

    def test_copy(self):
        r"""Test that copies of series are independent."""
        s1 = FourierSeries([0, 0, 2, 0], domain=(4, 8))
        s2 = s1.copy()
        self.assertIs(type(s2), FourierSeries)
        self.assertEqual(s2.domain[0], 4)
        self.assertEqual(s2.domain[1], 8)
        self.assertEqual(s1.a_n, s2.a_n)
        self.assertIsNot(s1.a_n, s2.a_n)

    def test_from_function(self):
        pi, sin, cos, exp = np.pi, math.sin, math.cos, math.exp
        a, b = 0, 2*pi
        f1 = lambda x: -1 + 2*sin(1*x) - 1*cos(2*x) + 3*cos(3*(x-0.2))
        space = np.linspace(a, b, 10)
        s1 = FourierSeries.from_function(f1, num=10, domain=(a, b))
        e1 = s1.evaluator()
        self.assertListAlmostEqual(lmap(e1, space), lmap(f1, space), delta=1e-14)
        f2 = lambda x: exp(f1(x))
        s2 = FourierSeries.from_function(f2, num=120, domain=(a, b))
        e2 = s2.evaluator()
        self.assertListAlmostEqual(lmap(e2, space), lmap(f2, space), delta=1e-12)


class TestEvaluateTrig(DpkTestCase):
    def _assert_mat_almost_equal(self, mat1, mat2, tol):
        if isinstance(mat1, (np.ndarray, mp.matrix)):
            mat1 = mat1.tolist()
        if isinstance(mat2, (np.ndarray, mp.matrix)):
            mat2 = mat2.tolist()
        for row_a, row_b in zip(mat1, mat2):
            self.assertListAlmostEqual(row_a, row_b, delta=tol)

    def _mat(self, use_mp, tol):
        xs = CosineSeries.create_collocation_points(5)
        jn = list(range(len(xs)))
        ctx = mp if use_mp else np
        sin, cos = ctx.sin, ctx.cos
        mat1 = evaluate_trig_mat("cos", xs, jn, use_mp=use_mp)
        mat2 = [[cos(j*x) for j in jn] for x in xs]
        self._assert_mat_almost_equal(mat1, mat2, tol)
        mat1 = evaluate_trig_mat("sin", xs, jn, use_mp=use_mp, scale=2)
        mat2 = [[sin(j*x) for j in jn] for x in xs]
        self._assert_mat_almost_equal(mat1, mat2, tol)
        mat1 = evaluate_trig_mat("sin", xs, jn, use_mp=use_mp, diff=1,
                                 scale=2)
        mat2 = [[2*j*cos(j*x) for j in jn] for x in xs]
        self._assert_mat_almost_equal(mat1, mat2, tol)
        mat1 = evaluate_trig_mat("cos", xs, jn, use_mp=use_mp, diff=1,
                                 scale=2)
        mat2 = [[-2*j*sin(j*x) for j in jn] for x in xs]
        self._assert_mat_almost_equal(mat1, mat2, tol)
        mat1 = evaluate_trig_mat("cos", xs, jn, use_mp=use_mp, diff=2,
                                 scale=2)
        mat2 = [[-4*j**2*cos(j*x) for j in jn] for x in xs]
        self._assert_mat_almost_equal(mat1, mat2, tol)
        mat1 = evaluate_trig_mat("sin", xs, jn, use_mp=use_mp, diff=2,
                                 scale=2)
        mat2 = [[-4*j**2*sin(j*x) for j in jn] for x in xs]
        self._assert_mat_almost_equal(mat1, mat2, tol)

    def test_mat(self):
        self._mat(use_mp=False, tol=1e-12)
        with mp.workdps(30):
            self._mat(use_mp=True, tol=1e-25)

    def _trig_series(self, use_mp, tol):
        x = 0.1234
        jn = list(range(5))
        ctx = mp if use_mp else np
        sin, cos = ctx.sin, ctx.cos
        vals1 = evaluate_trig_series("sin", x, jn, use_mp=use_mp, diff=0,
                                     scale=2)
        vals2 = [sin(j*x) for j in jn]
        self.assertListAlmostEqual(vals1, vals2, delta=tol)
        vals1 = evaluate_trig_series("sin", x, jn, use_mp=use_mp, diff=1,
                                     scale=2)
        vals2 = [2*j*cos(j*x) for j in jn]
        self.assertListAlmostEqual(vals1, vals2, delta=tol)
        vals1 = evaluate_trig_series("sin", x, jn, use_mp=use_mp, diff=2,
                                     scale=2)
        vals2 = [-4*j**2*sin(j*x) for j in jn]
        self.assertListAlmostEqual(vals1, vals2, delta=tol)
        vals1 = evaluate_trig_series("cos", x, jn, use_mp=use_mp, diff=1,
                                     scale=2)
        vals2 = [-2*j*sin(j*x) for j in jn]
        self.assertListAlmostEqual(vals1, vals2, delta=tol)
        vals1 = evaluate_trig_series("cos", x, jn, use_mp=use_mp, diff=2,
                                     scale=2)
        vals2 = [-4*j**2*cos(j*x) for j in jn]
        self.assertListAlmostEqual(vals1, vals2, delta=tol)

    def test_trig_series(self):
        self._trig_series(use_mp=False, tol=1e-12)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
