#!/usr/bin/env python3

from __future__ import print_function
from builtins import range, map

import unittest
import sys
import math

import numpy as np
from mpmath import mp

from testutils import DpkTestCase
from .solver import NDSolver, ndsolve
from .bases.cheby import ChebyBasis
from .bases.trig import SineBasis, CosineBasis
from .bcs import DirichletCondition


def lmap(func, *iterables):
    return list(map(func, *iterables))


class TestNdsolve(DpkTestCase):
    def _expand_function(self, num, atol, use_mp=False, dps=None):
        f = lambda x: math.exp(math.cos(4*x)**3)
        basis = ChebyBasis(domain=(2, 4), num=num)
        sol = ndsolve(([1], f), basis, use_mp=use_mp, dps=dps)
        f2 = sol.evaluator(use_mp=use_mp)
        pts = np.linspace(sol.domain[0], sol.domain[1], 20)
        max_err = max(map(lambda x: abs(f(x) - f2(x)), pts))
        self.assertLessEqual(max_err, atol)

    def test_chebybasis(self):
        self._expand_function(num=8, use_mp=False, atol=0.82)
        self._expand_function(num=32, use_mp=False, atol=3e-3)
        self._expand_function(num=64, use_mp=False, atol=1e-6)
        self._expand_function(num=128, use_mp=False, atol=1e-14)
        self._expand_function(num=32, use_mp=True, dps=30, atol=3e-3)

    def _ode1(self, num, atol, use_mp=False, dps=None,
              mat_solver='scipy.solve', eq_generator=False):
        # Solve the ODE
        #   f'' - a b sin(b x) f' = a b^2 cos(b x) f
        # with an exact solution
        #   f(x) = exp(-a cos(b x)).
        # The domain is chosen to avoid symmetries and to test non-trivial
        # boundary conditions. This test is designed to be able to perform
        # fast tests with floating point precision or slow tests with mpmath
        # arbitrary precision arithmetics. All constants are coded such that
        # arbitrary precision tests can succeed with an accuracy of up to (for
        # example):
        #   atol = 2e-39    with    num=160, dps=50
        fl = mp.mpf if use_mp else float
        ctx = mp if use_mp else np
        sin, cos, exp = ctx.sin, ctx.cos, ctx.exp
        with mp.workdps(dps or mp.dps):
            a, b = map(fl, (2, 6))
            domain = lmap(fl, (0.5, 2))
            exact = lambda x: exp(-a*cos(b*x))
            v1 = fl(mp.mpf('7.2426342955875784959555289115078253477587230131548'))
            v2 = fl(mp.mpf('0.18494294304163188136560483509304192452611801175781'))
            if eq_generator:
                eq = lambda pts: ((-a*b**2*np.cos(b*pts), -a*b*np.sin(b*pts), 1), 0)
            else:
                eq = ((lambda x: -a*b**2*cos(b*x), lambda x: -a*b*sin(b*x), 1), 0)
            sol = ndsolve(
                eq=eq,
                basis=ChebyBasis(domain=domain, num=num),
                boundary_conditions=(
                    DirichletCondition(x=domain[0], value=v1),
                    DirichletCondition(x=domain[1], value=v2),
                ),
                use_mp=use_mp,
                mat_solver=mat_solver
            )
            f = sol.evaluator(use_mp)
            pts = np.linspace(float(sol.domain[0]), float(sol.domain[1]), 50)
            max_err = max(map(lambda x: abs(exact(x) - f(x)), pts))
            self.assertLessEqual(max_err, atol)

    def test_ode1(self):
        self._ode1(num=20, atol=0.16)
        self._ode1(num=50, atol=4e-8)
        self._ode1(num=80, atol=1e-11)
        # Just to test for errors, don't expect good accuracy here:
        self._ode1(num=10, atol=10, use_mp=True, mat_solver='scipy.solve')
        self._ode1(num=10, atol=10, use_mp=True, mat_solver='mp.lu_solve')
        #self._ode1(num=160, atol=2e-39, dps=50, use_mp=True, mat_solver='mp.lu_solve')

    def test_eqgen(self):
        self._ode1(num=50, atol=4e-8, eq_generator=True)

    def _cos_basis(self, lobatto):
        # The exact solution in this test is 2*pi-periodic and symmetric about
        # x = 0. It has no symmetry w.r.t. pi/2. This means a Fourier cosine
        # basis is suitable for this problem.
        sin, cos, exp = mp.sin, mp.cos, mp.exp
        inhom = lambda x: exp(cos(x)) * (
            sin(x) * ((sin(x)**2-sin(x)-25)*sin(5*x)+(5-10*sin(x))*cos(5*x))
            + cos(x) * ((1-3*sin(x))*sin(5*x)+10*cos(5*x))
        )
        f = ndsolve(
            eq=([1,1,1], inhom),
            basis=CosineBasis(domain=(0, mp.pi), num=21, lobatto=lobatto),
            boundary_conditions=DirichletCondition(0),
            mat_solver='scipy.solve'
        )
        f_exact = lambda x: exp(cos(x)) * sin(x) * sin(mp.mpf(5)*x)
        space = np.linspace(0, np.pi, 20)
        self.assertListAlmostEqual(
            lmap(f.evaluator(), space),
            lmap(f_exact, space),
            delta=1e-13
        )

    def test_exp_cos_general(self):
        self._cos_basis(lobatto=True)
        self._cos_basis(lobatto=False)

    def _sin_basis(self, lobatto):
        # The exact solution in this test is 2*pi-periodic and antisymmetric
        # about x = 0. It has no symmetry w.r.t. pi/2. This means a Fourier
        # sine basis is suitable for this problem.
        sin, cos, exp = mp.sin, mp.cos, mp.exp
        inhom = lambda x: exp(cos(x)) * (
            sin(5*x) * (sin(x)**4-sin(x)**3-26*sin(x)**2+sin(2*x))
            + 2*sin(5*x)*cos(x)**2-5*sin(x)**2*(2*sin(x)-1)*cos(5*x)
            - 5*sin(x)*cos(x)*(sin(x)*sin(5*x)-4*cos(5*x))
        )
        f = ndsolve(
            eq=([1,1,1], inhom),
            basis=SineBasis(domain=(0, mp.pi), num=20, lobatto=lobatto),
            mat_solver='scipy.solve'
        )
        f_exact = lambda x: exp(cos(x)) * sin(x)**2 * sin(5*x)
        space = np.linspace(0, np.pi, 20)
        self.assertListAlmostEqual(
            lmap(f.evaluator(), space),
            lmap(f_exact, space),
            delta=1e-13
        )

    def test_exp_sin_general(self):
        self._sin_basis(lobatto=True)
        self._sin_basis(lobatto=False)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
