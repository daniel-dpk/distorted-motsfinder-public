#!/usr/bin/env python3

import unittest
import sys
import pickle

import numpy as np
from scipy.misc import derivative

from testutils import DpkTestCase
from .basics import SimpleSinhExpression, ScaleExpression
from .inverse import InverseExpression


class TestInverse(DpkTestCase):
    def test_increasing(self):
        f = SimpleSinhExpression(domain=(-4, 4))
        self.__check_inverse(f, samples=10, N=5)

    def test_decreasing(self):
        f = SimpleSinhExpression(domain=(-4, 4))
        f = ScaleExpression(f, a=-1)
        self.__check_inverse(f, samples=10, N=5)

    def __check_inverse(self, f, samples, N, tol=1e-12, fi=None):
        a, b = f.domain
        ev = f.evaluator()
        fa, fb = ev(a), ev(b)
        increasing = fa < fb
        if fi is None:
            fi = InverseExpression(f, samples=samples)
            self.assertEqual(fi.samples, samples)
        self.assertEqual(fi.expr_domain, f.domain)
        evi = fi.evaluator()
        self.assertAlmostEqual(evi.domain[increasing-1], fa)
        self.assertAlmostEqual(evi.domain[increasing], fb)
        self.__check_inverse_values(ev, evi, N=N, tol=tol)

    def __check_inverse_values(self, ev, evi, N, tol=1e-12):
        for x in np.linspace(*ev.domain, N):
            self.assertAlmostEqual(x, evi(ev(x)), delta=tol)
        for x in np.linspace(*evi.domain, N):
            self.assertAlmostEqual(x, ev(evi(x)), delta=tol)

    def test_callable(self):
        # Test using a callable instead of an expression.
        f = lambda x: np.cos(x)
        f.domain = (0.0, np.pi)
        fi = InverseExpression(f, samples=10, domain=(-1, 1))
        evi = fi.evaluator()
        self.__check_inverse_values(f, evi, 10)

    def test_derivatives(self):
        f = SimpleSinhExpression(domain=(-4, 4))
        fi = InverseExpression(f, samples=10)
        evi = fi.evaluator()
        N = 10
        for n in (1, 2):
            for x in np.linspace(*evi.domain, N+1, endpoint=False)[1:]:
                self.assertAlmostEqual(
                    evi.diff(x, n),
                    derivative(evi, x0=x, n=n, dx=1e-3, order=5),
                    delta=1e-8,
                    msg="x0=%s, n=%s" % (x, n),
                )

    def test_pickle(self):
        f = SimpleSinhExpression(domain=(-4, 4))
        fi = InverseExpression(f, samples=10)
        fi_str = pickle.dumps(fi)
        fi_loaded = pickle.loads(fi_str)
        self.__check_inverse(f, samples=10, N=5, fi=fi_loaded)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    return len(unittest.TextTestRunner(verbosity=2).run(suite).failures)


if __name__ == '__main__':
    unittest.main()
