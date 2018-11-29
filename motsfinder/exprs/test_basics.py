#!/usr/bin/env python3

from __future__ import print_function
from builtins import range, map

import unittest
import sys
import pickle

import numpy as np
from mpmath import mp

from testutils import DpkTestCase
from .numexpr import NumericExpression
from .numexpr import isclose


class _TestExpr1(NumericExpression):
    def __init__(self, a=1, **kw):
        super(_TestExpr1, self).__init__(**kw)
        self.a = a
    def _expr_str(self): return "a x**2, where a=%r" % self.a
    def _evaluator(self, use_mp):
        a = self.a
        return (lambda x: a*x**2, lambda x: 2*a*x, lambda x: 2*a, self.zero)

class _TestExpr2(NumericExpression):
    def __init__(self, expr, a=1):
        super(_TestExpr2, self).__init__(x=expr)
        self.a = a
    def _expr_str(self):
        return "a/x, where a=%r, x=%s" % (self.a, self.x.str())
    def _evaluator(self, use_mp):
        a = self.a
        x = self.x.evaluator(use_mp)
        def f(t):
            return a/x(t)
        def df(t):
            return -x.diff(t)*a/x(t)**2
        def ddf(t):
            xt = x(t)
            dxt = x.diff(t, 1)
            ddxt = x.diff(t, 2)
            return a*(-ddxt/xt**2 + 2*dxt**2/xt**3)
        return (f, df, ddf)

class _TestExpr3(NumericExpression):
    def __init__(self, expr1, expr2, a=1, b=1):
        super(_TestExpr3, self).__init__(x1=expr1, x2=expr2)
        self.a = a
        self.b = b
    def _expr_str(self):
        return ("a x1 + b x2, where a=%r, b=%r, x1=%s, x2=%s"
                % (self.a, self.b, self.x1.str(), self.x2.str()))
    def _evaluator(self, use_mp):
        a, b = self.a, self.b
        x1, x2 = self.x1.evaluator(use_mp), self.x2.evaluator(use_mp)
        return (lambda t: a*x1(t) + b*x2(t), lambda t: a*x1.diff(t) + b*x2.diff(t))

class _TestExprDomain(NumericExpression):
    def __init__(self, domain):
        super(_TestExprDomain, self).__init__(domain=domain)
        self.__domain = domain
    def get_domain(self): return self.__domain
    def _expr_str(self): return "id"
    def _evaluator(self, use_mp): return (lambda x: x, lambda x: 1, self.zero)


class TestIsclose(DpkTestCase):
    def test_float(self):
        self.assertTrue(isclose(1e7+1, 1e7+1, rel_tol=0, abs_tol=0))
        self.assertTrue(isclose(1e7+1, 1e7, rel_tol=1e-6))
        self.assertFalse(isclose(1e7+1, 1e7, rel_tol=1e-8))
        self.assertTrue(isclose(1e7+1, 1e7, rel_tol=0, abs_tol=2.0))
        self.assertFalse(isclose(1e7+1, 1e7, rel_tol=0, abs_tol=0.5))

    def test_mpmath(self):
        with mp.workdps(30):
            a = mp.mpf('1e7') + mp.mpf('1e-20')
            b = mp.mpf('1e7')
            self.assertTrue(isclose(a, a, rel_tol=0, abs_tol=0, use_mp=True))
            self.assertFalse(isclose(a, b, use_mp=True))
            with mp.workdps(26):
                self.assertTrue(isclose(a, b, use_mp=True))
            self.assertTrue(isclose(a, b, rel_tol=1e-26, abs_tol=0, use_mp=True))
            self.assertFalse(isclose(a, b, rel_tol=1e-28, abs_tol=0, use_mp=True))
            self.assertTrue(isclose(a, b, rel_tol=0, abs_tol=1e-19, use_mp=True))
            self.assertFalse(isclose(a, b, rel_tol=0, abs_tol=1e-21, use_mp=True))


class TestNumexpr(DpkTestCase):
    def test_expressions(self):
        expr = _TestExpr2(_TestExpr1())
        self.assertEqual(repr(expr), "<_TestExpr2(a/x, where a=1, x=(a x**2, where a=1))>")
        self.assertEqual(expr.a, 1)
        expr.a = 5
        self.assertEqual(expr.a, 5)

    def test_name(self):
        expr = _TestExpr1()
        self.assertEqual(expr.name, "_TestExpr1")
        expr.name = "foo"
        self.assertEqual(expr.name, "foo")

    def test_pickle(self):
        a = 1.5
        expr = _TestExpr2(_TestExpr1(-1), a=a)
        expr.name = "foo"
        s = pickle.dumps(expr)
        expr = pickle.loads(s)
        self.assertIs(type(expr), _TestExpr2)
        self.assertEqual(expr.a, 1.5)
        self.assertIs(type(expr.x), _TestExpr1)
        self.assertEqual(expr.x.a, -1)
        self.assertEqual(expr.name, "foo")

    def test_pickle_domain(self):
        expr = _TestExpr1(domain=(0, 1))
        s = pickle.dumps(expr)
        expr = pickle.loads(s)
        self.assertEqual(expr.domain, (0, 1))
        expr = _TestExpr1(domain=(0, mp.pi))
        s = pickle.dumps(expr)
        expr = pickle.loads(s)
        self.assertEqual(expr.domain, (0, mp.pi))

    def test_evaluators(self):
        a = 1.5
        expr = _TestExpr2(_TestExpr1(), a=a)
        f = expr.evaluator()
        for t in np.linspace(0.1, 2, 4):
            self.assertAlmostEqual(f(t), a/t**2)
        for t in np.linspace(0.1, 2, 4):
            self.assertAlmostEqual(f.diff(t), -2*a/t**3)
        for t in np.linspace(0.1, 2, 4):
            self.assertAlmostEqual(f.diff(t, 2), 6*a/t**4)
        with self.assertRaises(NotImplementedError):
            f.diff(0, 3)

    def test_string_clashing(self):
        expr1 = _TestExpr1(a=1)
        expr2 = _TestExpr2(2, a=3)
        comp1 = _TestExpr3(expr1, expr2)

        expr2 = _TestExpr2(2, a=1)
        expr1 = _TestExpr1(a=3)
        comp2 = _TestExpr3(expr1, expr2)

        e1 = comp1.evaluator()
        e2 = comp2.evaluator()

        # The expressions are different:
        self.assertNotEqual(e1(.5), e2(.5))

        # Their string are different too:
        self.assertNotEqual(repr(comp1), repr(comp2))

    def test_domain(self):
        expr = _TestExprDomain([-1, 1])
        e = expr.evaluator()
        self.assertTrue(hasattr(e, 'domain'))
        self.assertFalse(hasattr(e, 'domainX'))
        self.assertEqual(e.domain[0], -1)
        self.assertEqual(e.domain[1], 1)

        expr = _TestExprDomain(([-1, 1], [0, 10]))
        e = expr.evaluator()
        self.assertTrue(hasattr(e, 'domain'))
        self.assertEqual(e.domainX[0], -1)
        self.assertEqual(e.domainX[1], 1)
        self.assertEqual(e.domainY[0], 0)
        self.assertEqual(e.domainY[1], 10)
        f = e.function()
        self.assertTrue(hasattr(f, 'domain'))
        self.assertTrue(hasattr(f, 'domainX'))
        self.assertTrue(hasattr(f, 'domainY'))
        self.assertFalse(hasattr(f, 'domainZ'))


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
