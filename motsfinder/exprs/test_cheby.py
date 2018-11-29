#!/usr/bin/env python3

from __future__ import print_function
from builtins import range

import unittest
import sys
import math
import numpy as np
from mpmath import mp

from testutils import DpkTestCase
from ..utils import lmap
from .cheby import Cheby, evaluate_Tn, force_pure_python


class TestEvaluateTn(DpkTestCase):
    def test_dtype(self):
        num = 5
        Tn = evaluate_Tn(mp.mpf('0.2'), [0]*num, use_mp=True)
        self.assertIs(type(Tn), list)
        self.assertEqual(len(Tn), num)
        for i in range(num):
            self.assertIs(type(Tn[i]), mp.mpf,
                          msg="type(Tn[%d]) == %s is not an 'mpf'" % (i, type(Tn[i])))
        Tn = evaluate_Tn(0.2, [0]*num, use_mp=True)
        self.assertIs(type(Tn), list)
        self.assertEqual(len(Tn), num)
        for i in range(num):
            self.assertIs(type(Tn[i]), mp.mpf,
                          msg="type(Tn[%d]) == %s is not an 'mpf'" % (i, type(Tn[i])))
        Tn = evaluate_Tn(mp.mpf('0.2'), [0]*num, use_mp=False)
        self.assertIs(type(Tn), np.ndarray)
        self.assertEqual(len(Tn), num)
        self.assertEqual(Tn.dtype, np.float64)
        Tn = evaluate_Tn(0.2, [0]*num, use_mp=False)
        self.assertIs(type(Tn), np.ndarray)
        self.assertEqual(len(Tn), num)
        self.assertEqual(Tn.dtype, np.float64)


class TestCheby(DpkTestCase):
    def test_modify_an(self):
        a, b = 0, math.pi
        an1 = [-0.01555564, 0.01952286, 0.03171914, -0.13432793,
               0.14499185, 0.2404123, -0.53171914, -0.12560723, 0.37056379]
        an2 = [-0.0083028, -0.09043416, 0.18818946, 0.11192333,
               -0.38831467, 0.02780892, 0.31181054, -0.04929809, -0.10338253]
        cheby = Cheby(an1, domain=(a, b))
        space = np.linspace(a, b, 10)
        self.assertListAlmostEqual(
            lmap(cheby.evaluator(), space),
            [0.0, 0.58215566, -0.70661809, -1.03383155, 0.47327958,
             0.99632402, -0.14152985, -0.53644362, 0.25732658, 0.0]
        )
        cheby.a_n = an2
        self.assertListAlmostEqual(
            lmap(cheby.evaluator(), space),
            [0.0, 0.28192779, 0.67791116, 0.06267216, -0.85467607,
             -0.84947417, -0.06502122, 0.2867633, -0.01803222, 0.0]
        )
        cheby.a_n = [.5, 0, -.5]
        self.assertListAlmostEqual(
            lmap(cheby.evaluator(), space),
            [0.0, 0.39506173, 0.69135802, 0.88888889, 0.98765432, 0.98765432,
             0.88888889, 0.69135802, 0.39506173, 0.0]
        )
        with self.assertRaises(ValueError):
            Cheby(1.2, domain=(a, b))
        with self.assertRaises(ValueError):
            cheby.a_n = 1.2
            cheby.evaluator()
        cheby.a_n = []
        self.assertListAlmostEqual(lmap(cheby.evaluator(), space), [0]*10)
        self.assertListAlmostEqual(lmap(cheby.evaluator(use_mp=True), space), [0]*10)

    def test_derivative(self):
        a, b = 0, 2
        cheby = Cheby([.5, .5], domain=(a, b))
        space = np.linspace(a, b, 10)
        self.assertListAlmostEqual(lmap(cheby.evaluator().function(1), space),
                                   [.5]*10)
        self.assertListAlmostEqual(lmap(cheby.evaluator().function(2), space),
                                   [0]*10)
        a, b = 0, 1
        cheby.domain = (a, b)
        space = np.linspace(a, b, 10)
        cheby.a_n = [.5, 0, .5]
        self.assertListAlmostEqual(lmap(cheby.evaluator().function(1), space),
                                   [-4. + 8.*x for x in space])
        self.assertListAlmostEqual(lmap(cheby.evaluator().function(2), space),
                                   [8. for x in space])
        cheby.a_n = [1]
        ev = cheby.evaluator(use_mp=True)
        self.assertEqual(ev(0), 1.0)
        self.assertEqual(ev.diff(0, 1), 0.0)
        self.assertEqual(ev.diff(0, 2), 0.0)

    def test_values(self):
        a, b = 10., 20.
        # pylint: disable=bad-whitespace
        cheby = Cheby([-3.249, -3.743,  4.731,  7.597,  1.126, -1.15 , -0.909,
                       -2.088,  0.002,  1.229,  0.353,  0.234, -0.35 , -0.745,
                       -0.612, -0.612, -0.482, -0.347, -0.323, -0.215, -0.197,
                       -0.136, -0.089, -0.051],
                      domain=(a, b))
        e_fp = cheby.evaluator(False)
        e_mp = cheby.evaluator(True)
        e_fp.set_x(12.3456789)
        e_mp.set_x(12.3456789)
        self.assertListAlmostEqual(
            e_fp._values(n=0, apply_coeffs=False), e_mp._values(n=0, apply_coeffs=False)
        )
        self.assertListAlmostEqual(
            e_fp._values(n=1, apply_coeffs=False), e_mp._values(n=1, apply_coeffs=False)
        )
        self.assertListAlmostEqual(
            e_fp._values(n=2, apply_coeffs=False), e_mp._values(n=2, apply_coeffs=False)
        )
        self.assertListAlmostEqual(
            e_fp._values(n=0, apply_coeffs=True), e_mp._values(n=0, apply_coeffs=True)
        )
        self.assertListAlmostEqual(
            e_fp._values(n=1, apply_coeffs=True), e_mp._values(n=1, apply_coeffs=True)
        )
        self.assertListAlmostEqual(
            e_fp._values(n=2, apply_coeffs=True), e_mp._values(n=2, apply_coeffs=True)
        )

    def _eval_basis_at(self, e, n, reference, x):
        e.set_x(x)
        self.assertListAlmostEqual(
            e.evaluate_basis(n=n),
            [f(x) for f in reference],
            delta=1e-12
        )

    def _eval_basis(self, e, n, reference, space):
        for x in space:
            self._eval_basis_at(e, n, reference, x)

    def test_evaluate_basis(self):
        cheby = Cheby([0]*8)
        Tn = [np.polynomial.Chebyshev.basis(n) for n in range(len(cheby.a_n))]
        dTn = [f.deriv(m=1) for f in Tn]
        ddTn = [f.deriv(m=2) for f in Tn]
        space = np.linspace(-1, 1, 10)
        e = cheby.evaluator()
        self._eval_basis(e, 0, Tn, space)
        self._eval_basis(e, 1, dTn, space)
        self._eval_basis(e, 2, ddTn, space)
        e = cheby.evaluator(True)
        self._eval_basis(e, 0, Tn, space)
        self._eval_basis(e, 1, dTn, space)
        self._eval_basis(e, 2, ddTn, space)
        cheby = Cheby([0]*4, symmetry='even')
        e = cheby.evaluator()
        self._eval_basis(e, 0, Tn[0::2], space)
        self._eval_basis(e, 1, dTn[0::2], space)
        self._eval_basis(e, 2, ddTn[0::2], space)
        e = cheby.evaluator(True)
        self._eval_basis(e, 0, Tn[0::2], space)
        self._eval_basis(e, 1, dTn[0::2], space)
        self._eval_basis(e, 2, ddTn[0::2], space)
        cheby = Cheby([0]*4, symmetry='odd')
        e = cheby.evaluator()
        self._eval_basis(e, 0, Tn[1::2], space)
        self._eval_basis(e, 1, dTn[1::2], space)
        self._eval_basis(e, 2, ddTn[1::2], space)
        e = cheby.evaluator(True)
        self._eval_basis(e, 0, Tn[1::2], space)
        self._eval_basis(e, 1, dTn[1::2], space)
        self._eval_basis(e, 2, ddTn[1::2], space)

    def test_complex(self):
        cheby = Cheby([0]*4, domain=(10, 20))
        ev = cheby.evaluator(use_mp=True)
        with mp.workdps(20):
            self.assertIsNot(type(ev.diff(10, 1)), mp.mpc)
            self.assertIsNot(type(ev.diff(20, 1)), mp.mpc)
            self.assertIsNot(type(ev.diff(10, 2)), mp.mpc)
            self.assertIsNot(type(ev.diff(20, 2)), mp.mpc)

    def test_collocation_points(self):
        cheby = Cheby([], domain=(10, 20))
        self.assertListAlmostEqual(cheby.collocation_points(), [])
        cheby.a_n = [1, 2, 3]
        self.assertListAlmostEqual(cheby.collocation_points(), [20, 15, 10])
        self.assertListAlmostEqual(
            cheby.collocation_points(5, internal_domain=True, use_mp=False),
            [1, 0.707106781187, 0, -0.707106781187, -1]
        )
        self.assertListAlmostEqual(
            cheby.collocation_points(5, internal_domain=False, use_mp=True, dps=30),
            [20, 18.5355339059327, 15, 11.4644660940673, 10]
        )
        self.assertListAlmostEqual(
            cheby.collocation_points(5, internal_domain=True, use_mp=True, dps=30),
            [1, 0.707106781187, 0, -0.707106781187, -1]
        )
        self.assertListAlmostEqual(
            Cheby.create_collocation_points(5, use_mp=False),
            [1, 0.707106781187, 0, -0.707106781187, -1]
        )

    def test_set_coefficients(self):
        a, b = 10., 20.
        cheby = Cheby([], domain=(a, b))
        with self.assertRaises(ValueError):
            cheby.set_coefficients(5)
        space = np.linspace(a, b, 10)
        cheby.set_coefficients([.5, 0, -.5])
        self.assertListAlmostEqual(
            lmap(cheby.evaluator(), space),
            [0.0, 0.39506173, 0.69135802, 0.88888889, 0.98765432, 0.98765432,
             0.88888889, 0.69135802, 0.39506173, 0.0]
        )
        def f(x):
            x = 2*(x-a)/(b-a) - 1
            return 1-x**2 + .2*math.exp(-x)*math.cos(10*x**2) + .9
        cheby.set_coefficients([f(x) for x in cheby.collocation_points(25)],
                               physical_space=True)
        self.assertListAlmostEqual(lmap(cheby.evaluator(), space), lmap(f, space), delta=1e-4)

    def test_roundtrip(self):
        a, b = 10., 20.
        # pylint: disable=bad-whitespace
        cheby1 = Cheby([-3.249, -3.743,  4.731,  7.597,  1.126, -1.15 , -0.909,
                        -2.088,  0.002,  1.229,  0.353,  0.234, -0.35 , -0.745,
                        -0.612, -0.612, -0.482, -0.347, -0.323, -0.215, -0.197,
                        -0.136, -0.089, -0.051],
                       domain=(a, b))
        pts = cheby1.collocation_points()
        fvals = lmap(cheby1.evaluator(), pts)
        cheby2 = Cheby([], domain=(a, b))
        cheby2.set_coefficients(fvals, physical_space=True)
        self.assertListAlmostEqual(lmap(cheby2.evaluator(), pts), fvals, delta=1e-12)
        fvals_gauss = lmap(cheby1.evaluator(), cheby1.collocation_points(lobatto=False))
        cheby3 = Cheby([], domain=(a, b))
        cheby3.set_coefficients(fvals_gauss, physical_space=True, lobatto=False)
        self.assertListAlmostEqual(lmap(cheby3.evaluator(), pts), fvals, delta=1e-12)

    def test_approximate(self):
        try:
            a, b = domain = 10., 20.
            space = np.linspace(a, b, 10)
            cheby = Cheby([], domain=domain)
            cheby.approximate(1, 1)
            self.assertListAlmostEqual(cheby.a_n, [1])
            with self.assertRaises(ValueError):
                cheby.approximate(1, 0)
            cheby.approximate(1, 10)
            self.assertListAlmostEqual(cheby.a_n, [1] + [0]*9)
            self.assertListAlmostEqual(lmap(cheby.evaluator(), space), [1]*10, delta=1e-16)
            cheby.approximate(0)
            self.assertEqual(cheby.a_n, [0]*10)
            def f(x):
                x = 2*(x-a)/(b-a) - 1
                return 1-x**2 + .2*math.exp(-x)*math.cos(10*x**2) + .9
            f_values = lmap(f, space)
            force_pure_python(True)
            cheby.approximate(f, 25)
            self.assertListAlmostEqual(lmap(cheby.evaluator(), space), f_values, delta=1e-4)
            force_pure_python(False)
            cheby.approximate(f, 25)
            self.assertListAlmostEqual(lmap(cheby.evaluator(), space), f_values, delta=1e-4)
            cheby.approximate(f, use_mp=True, dps=30)
            self.assertListAlmostEqual(lmap(cheby.evaluator(), space), f_values, delta=1e-4)
            cheby.approximate(f, 25, lobatto=False)
            self.assertListAlmostEqual(lmap(cheby.evaluator(), space), f_values, delta=1e-4)
        finally:
            force_pure_python(False)

    def test_from_function(self):
        num = 25
        a, b = 10., 20.
        space = np.linspace(a, b, 10)
        def f(x):
            x = 2*(x-a)/(b-a) - 1
            return 1-x**2 + .2*math.exp(-x)*math.cos(10*x**2) + .9
        f_values = lmap(f, space)
        with self.assertRaises(AttributeError):
            cheby = Cheby.from_function(f, num)
        cheby = Cheby.from_function(f, num, domain=(a, b))
        self.assertListAlmostEqual(lmap(cheby.evaluator(), space), f_values, delta=1e-4)
        f.domain = (a, b)
        cheby = Cheby.from_function(f, num)
        self.assertListAlmostEqual(lmap(cheby.evaluator(), space), f_values, delta=1e-4)

    def test_interpolate(self):
        a, b = 10., 20.
        cheby = Cheby([], domain=(a, b))
        y = [1, 1.5, 1.1, 0.2, 0.6, 0.9, 0.8, 0.6, 0.6]
        x = np.linspace(a, b, len(y))
        cheby.interpolate(y=y, num=20)
        vals = lmap(cheby.evaluator(), x)
        self.assertListAlmostEqual(vals, y, delta=1e-2)
        cheby.interpolate(y=y, num=45)
        vals = lmap(cheby.evaluator(), x)
        self.assertListAlmostEqual(vals, y, delta=1e-3)

    def test_copy(self):
        pi = math.pi
        a, b = 0, pi
        space = np.linspace(a, b, 10)
        cheby = Cheby([.5, .5], domain=(a, b))
        copy = cheby.copy()
        cheby.a_n = [.5, 0, -.5]
        self.assertListAlmostEqual(
            lmap(cheby.evaluator(), space),
            [0.0, 0.39506173, 0.69135802, 0.88888889, 0.98765432, 0.98765432,
             0.88888889, 0.69135802, 0.39506173, 0.0]
        )
        self.assertListAlmostEqual(lmap(copy.evaluator(), space),
                                   [x/pi for x in space])

    def test_resample(self):
        a, b = 0, 10
        space = np.linspace(a, b, 10)
        f = math.cos
        cheby = Cheby([], domain=(a, b))
        cheby.approximate(f, 20)
        self.assertEqual(cheby.get_dof(), 20)
        self.assertListAlmostEqual(lmap(cheby.evaluator(), space),
                                   lmap(f, space))
        cheby.resample(30)
        self.assertEqual(cheby.get_dof(), 30)
        self.assertListAlmostEqual(lmap(cheby.evaluator(), space),
                                   lmap(f, space))
        cheby.resample(20)
        self.assertEqual(cheby.get_dof(), 20)
        self.assertListAlmostEqual(lmap(cheby.evaluator(), space),
                                   lmap(f, space))


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
