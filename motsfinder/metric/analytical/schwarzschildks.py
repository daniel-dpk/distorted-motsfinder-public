r"""@package motsfinder.metric.analytical.schwarzschildks

Schwarzschild slice in Kerr-Schild coordinates.

The class defined here can be used to produce the data for a slice of
Schwarzschild spacetime in Kerr-Schild form.
"""

import numpy as np
from scipy import linalg

from ..base import _ThreeMetric


__all__ = [
    "SchwarzschildKSSlice",
]


class SchwarzschildKSSlice(_ThreeMetric):
    r"""Data for a slice of Schwarzschild spacetime in Kerr-Schild coordinates.

    The formulas implemented here are based on [1].

    @b References

    [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker. "Initial
        data and coordinates for multiple black hole systems." Physical Review
        D 59.2 (1998): 024015.
    """

    def __init__(self, M=1):
        r"""Create a Schwarzschild slice in Kerr-Schild form.

        @param M
            ADM mass of the spacetime.
        """
        super().__init__()
        self._M = float(M)

    @property
    def M(self):
        r"""ADM mass of the spacetime (\ie of the black hole)."""
        return self._M

    def H(self, point):
        r"""Scalar H function of [1]."""
        r = linalg.norm(point)
        return self._M/r

    def _mat_at(self, point):
        r = linalg.norm(point)
        M = self._M
        x = point
        return np.identity(3) + [
            [2*M * x[i]*x[j] / r**3 for j in range(3)]
            for i in range(3)
        ]

    def diff(self, point, inverse=False, diff=1):
        if inverse:
            return self._compute_inverse_diff(point, diff=diff)
        if diff == 0:
            return self._mat_at(point)
        x = point
        M = self._M
        r = linalg.norm(point)
        if diff == 1:
            ra = list(range(3))
            def _partial_k_h_ij(k, i, j):
                dik = i == k
                djk = j == k
                return 2*M*(r**2*(dik*x[j] + djk*x[i]) - 3*x[k]*x[i]*x[j])/r**5
            return np.asarray([[[_partial_k_h_ij(k, i, j)
                                 for i in ra] for j in ra] for k in ra])
        if diff == 2:
            ra = list(range(3))
            def _partial_l_k_h_ij(l, k, i, j):
                dik = i == k
                dil = i == l
                djk = j == k
                djl = j == l
                dkl = k == l
                return 2*M * (
                    (dil*djk + dik*djl) / r**3
                    - 3 * (
                        dkl*x[i]*x[j] + dil*x[k]*x[j] + djl*x[k]*x[i]
                        + dik*x[l]*x[j] + djk*x[l]*x[i]
                    ) / r**5
                    + 15 * x[k]*x[l]*x[i]*x[j] / r**7
                )
            return np.asarray([[[[_partial_l_k_h_ij(l, k, i, j)
                                  for i in ra]
                                 for j in ra]
                                for k in ra]
                               for l in ra])
        raise NotImplementedError

    def get_curv(self):
        return SchwarzschildKSSliceCurv(self)

    def get_lapse(self):
        return SchwarzschildKSSliceLapse(self)

    def get_shift(self):
        return SchwarzschildKSSliceShift(self)


class SchwarzschildKSSliceLapse():
    r"""Lapse function of the Schwarzschild slice in Kerr-Schild form."""
    def __init__(self, metric):
        self._g = metric

    def __call__(self, point, diff=0):
        if diff != 0:
            raise NotImplementedError
        M = self._g.M
        r = linalg.norm(point)
        return 1.0 / np.sqrt(1 + 2*M/r)


class SchwarzschildKSSliceShift():
    r"""Shift vector field of the Schwarzschild slice in Kerr-Schild form."""
    def __init__(self, metric):
        self._g = metric

    def __call__(self, point, diff=0):
        if diff != 0:
            raise NotImplementedError
        x = point
        M = self._g.M
        r = linalg.norm(point)
        betal = np.asarray([2*M/r**2 * x[i] for i in range(3)])
        g_inv = self._g.diff(point, inverse=True, diff=0)
        beta = np.einsum('ij,j->i', g_inv, betal)
        return beta


class SchwarzschildKSSliceCurv():
    r"""Extrinsic curvature of the Schwarzschild slice in Kerr-Schild form."""
    def __init__(self, metric):
        self._g = metric

    def __call__(self, point, diff=0):
        sqrt = np.sqrt
        x = point
        M = self._g.M
        r = linalg.norm(point)
        if diff == 0:
            return 2*M / (r**4 * sqrt(1+2*M/r)) * (
                r**2 * np.identity(3)
                - np.asarray([
                    [(2 + M/r) * x[i]*x[j] for j in range(3)]
                    for i in range(3)
                ])
            )
        if diff == 1:
            ra = list(range(3))
            def _partial_k_K_ij(k, i, j):
                dij = i == j
                dik = i == k
                djk = j == k
                return 2*M * (
                    M*(r**2*dij - (M/r + 2)*x[i]*x[j])*x[k] / (r**7*(2*M/r + 1)**(3/2))
                    + (
                        + M*x[i]*x[j]*x[k]/r**3
                        - (M/r + 2)*dik*x[j]
                        - (M/r + 2)*djk*x[i]
                        + 2*x[k]*dij
                    ) / (r**4*sqrt(2*M/r + 1))
                    - 4 * (r**2*dij - (M/r + 2)*x[i]*x[j])*x[k] / (r**6*sqrt(2*M/r + 1))
                )
            return np.asarray([[[_partial_k_K_ij(k, i, j)
                                 for i in ra] for j in ra] for k in ra])
        raise NotImplementedError
