r"""@package motsfinder.ndsolve.bases.trig

Fourier sine and cosine bases for the pseudospectral solver.
"""

from __future__ import print_function

from builtins import range

from ...exprs.trig import SineSeries, CosineSeries, evaluate_trig_mat
from ...exprs.trig import evaluate_trig_series
from .base import _SpectralSeriesBasis


__all__ = [
    "SineBasis",
    "CosineBasis",
]


class SineBasis(_SpectralSeriesBasis):
    r"""Pseudospectral basis set of sines."""

    def get_series_cls(self):
        return SineSeries

    def evaluate_all_at(self, x, n=0):
        jn = list(range(1, self.num+1))
        return evaluate_trig_series("sin", x, jn=jn, use_mp=self._use_mp,
                                    diff=n, scale=self._scl)

    def _compute_deriv_mat(self, n):
        pts = self.pts_internal
        jn = list(range(1, self.num+1))
        return evaluate_trig_mat("sin", xs=pts, jn=jn, use_mp=self._use_mp,
                                 diff=n, scale=self._scl)


class CosineBasis(_SpectralSeriesBasis):
    r"""Pseudospectral basis set of cosines.

    This class stores the last computed derivative matrix statically for
    re-use in case the exact same settings are requested again.
    """

    # TODO: Generalize to parent class and make optional.
    _DERIV_MATRICES = dict()

    def get_series_cls(self):
        return CosineSeries

    def evaluate_all_at(self, x, n=0):
        jn = list(range(self.num))
        return evaluate_trig_series("cos", x, jn=jn, use_mp=self._use_mp,
                                    diff=n, scale=self._scl)

    def _compute_deriv_mat(self, n):
        if self._use_mp:
            return self._compute_deriv_mat_doit(n)
        cache = self._DERIV_MATRICES.get(n, dict())
        key = (self.num, self._lobatto, self._scl)
        if key not in cache:
            cache = dict()
            cache[key] = self._compute_deriv_mat_doit(n)
        self._DERIV_MATRICES[n] = cache
        return cache[key]

    def _compute_deriv_mat_doit(self, n):
        pts = self.pts_internal
        jn = list(range(self.num))
        return evaluate_trig_mat("cos", xs=pts, jn=jn, use_mp=self._use_mp,
                                 diff=n, scale=self._scl)
