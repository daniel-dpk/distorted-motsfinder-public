r"""@package motsfinder.ndsolve.bases.cheby

Chebyshev polynomial basis for the pseudospectral solver.
"""

from __future__ import print_function
import math

from builtins import range
import numpy as np
from numpy.polynomial import Chebyshev as ChebyshevT
from mpmath import mp

from ...numutils import clip
from ...exprs import cheby
from .base import _SpectralSeriesBasis


__all__ = [
    "ChebyBasis",
]


class ChebyBasis(_SpectralSeriesBasis):
    r"""Pseudospectral basis set of Chebyshev polynomials."""
    def __init__(self, domain, num, lobatto=True):
        super(ChebyBasis, self).__init__(domain=domain, num=num,
                                         lobatto=lobatto)
        self._Tderiv = dict()

    @classmethod
    def get_series_cls(cls):
        return cheby.Cheby

    def evaluate_all_at(self, x, n=0):
        num = self.num
        if n == 0:
            row = mp.zeros(1, num) if self._use_mp else np.zeros(num)
            row = cheby.evaluate_Tn(x, row, use_mp=self._use_mp)
            return row
        dnT = self.Tderiv(n)
        s = self._scl**n
        fl = mp.mpf if self._use_mp else float
        return [s * fl(dnT[i](x)) for i in range(num)]

    def _compute_deriv_mat(self, n):
        matrix = mp.matrix if self._use_mp else np.array
        pts = self.pts_internal
        if n > 0 and not self._use_mp:
            dnT = self.Tderiv(n)
            pts = np.array(pts)
            s = self._scl**n
            return s * np.array([dnT[k](pts) for k in range(self.num)]).T
        return matrix([self.evaluate_all_at(pts[i], n=n) for i in range(self.num)])

    def Tderiv(self, n):
        r"""Return the n'th derivative of the first `num` Chebyshev polynomials.

        The result is a list of callables.
        """
        if n not in self._Tderiv:
            self._Tderiv[n] = [self._create_Tderiv(k, n) for k in range(self._num)]
        return self._Tderiv[n]

    def _create_Tderiv(self, k, n):
        r"""Create the n'th derivative of the k'th Chebyshev polynomial."""
        if not self._use_mp:
            return ChebyshevT.basis(k).deriv(n)
        else:
            if n == 0:
                return lambda x: mp.chebyt(k, x)
            # Since the Chebyshev derivatives attain high values near the
            # borders +-1 and usually are subtracted to obtain small values,
            # minimizing their relative error (i.e. fixing the number of
            # correct decimal places (dps)) is not enough to get precise
            # results. Hence, the below `moredps` variable is set to higher
            # values close to the border.
            extradps = 5
            pos = mp.fprod([mp.mpf(k**2-p**2)/(2*p+1) for p in range(n)])
            neg = (-1)**(k+n) * pos
            if n == 1:
                def dTk(x):
                    with mp.extradps(extradps):
                        x = clip(x, -1.0, 1.0)
                        if mp.almosteq(x, mp.one): return pos
                        if mp.almosteq(x, -mp.one): return neg
                        moredps = max(0, int(-math.log10(min(abs(x-mp.one),abs(x+mp.one)))/2))
                        moredps = min(moredps, 100)
                        with mp.extradps(moredps):
                            t = mp.acos(x)
                            return k * mp.sin(k*t)/mp.sin(t)
                return dTk
            if n == 2:
                def ddTk(x):
                    with mp.extradps(extradps):
                        x = clip(x, -1.0, 1.0)
                        if mp.almosteq(x, mp.one): return pos
                        if mp.almosteq(x, -mp.one): return neg
                        moredps = max(0, int(-math.log10(min(abs(x-mp.one),abs(x+mp.one)))*1.5) + 2)
                        moredps = min(moredps, 100)
                        with mp.extradps(moredps):
                            t = mp.acos(x)
                            s = mp.sin(t)
                            return - k**2 * mp.cos(k*t)/s**2 + k * mp.cos(t) * mp.sin(k*t)/s**3
                return ddTk
            raise NotImplementedError('Derivatives of order > 2 not implemented.')
