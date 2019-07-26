r"""@package motsfinder.metric.analytical.schwarzschildks

Schwarzschild slice in Kerr-Schild coordinates.

The class defined here can be used to produce the data for a slice of
Schwarzschild spacetime in Kerr-Schild form.
"""

import numpy as np
from scipy import linalg

from ..base import _ThreeMetric, trivial_dtlapse, trivial_dtshift


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

    def get_dtlapse(self):
        return trivial_dtlapse

    def get_shift(self):
        return SchwarzschildKSSliceShift(self)

    def get_dtshift(self):
        return trivial_dtshift


class SchwarzschildKSSliceLapse():
    r"""Lapse function of the Schwarzschild slice in Kerr-Schild form."""
    def __init__(self, metric):
        self._g = metric

    def __call__(self, point, diff=0):
        ra = range(3)
        M = self._g.M
        r = linalg.norm(point)
        if diff == 0:
            return 1.0 / np.sqrt(1 + 2*M/r)
        x = point
        if diff == 1:
            return np.asarray([M * x[i] / (r**3 * (2*M/r + 1)**(3./2))
                               for i in ra])
        if diff == 2:
            def _partial(i, j):
                dij = i == j
                return (
                    3*M**2*x[i]*x[j] / (r**6*(2*M/r + 1)**(5./2))
                    + dij * M / (r**3 * (2*M/r + 1)**(3./2))
                    - 3*M*x[i]*x[j] / (r**5 * (2*M/r + 1)**(3./2))
                )
            return np.asarray([[_partial(i, j) for j in ra] for i in ra])
        raise NotImplementedError("Higher derivatives of lapse.")


class SchwarzschildKSSliceShift():
    r"""Shift vector field of the Schwarzschild slice in Kerr-Schild form."""
    def __init__(self, metric):
        self._g = metric

    def __call__(self, point, diff=0):
        ra = range(3)
        x = point
        M = self._g.M
        r = linalg.norm(point)
        betal = np.asarray([2*M/r**2 * x[i] for i in ra])
        g_inv = self._g.diff(point, inverse=True, diff=0)
        beta = np.einsum('ij,j->i', g_inv, betal)
        if diff == 0:
            return beta
        if diff == 1:
            def _partial(i, j):
                dij = i == j
                return 2*M/(r**3 * (2*M+r)**2) * (
                    r**2 * (2*M + r) * dij - 2 * (M+r) * x[i]*x[j]
                )
            return np.asarray([[_partial(i, j) for j in ra] for i in ra])
        if diff == 2:
            def _partial(i, j, k):
                dij = i == j
                dik = i == k
                djk = j == k
                return (
                    -4*M*(3*M+2*r)/(r**5*(2*M+r)**2) * x[i] * (
                        r**2 * djk - x[j]*x[k] - r*x[j]*x[k]/(2*M+r)
                    )
                    + 2*M/(r**3*(2*M+r)) * (
                        2*x[i] * djk - x[k] * dij - x[j] * dik
                        - 1/(r*(2*M+r)) * (
                            r**2*(dij*x[k] + dik*x[j]) + x[i]*x[j]*x[k]
                            - r * x[i]*x[j]*x[k] / (2*M + r)
                        )
                    )
                )
            return np.asarray([[[_partial(i, j, k) for j in ra] for i in ra]
                               for k in ra])
        raise NotImplementedError("Higher derivatives of shift.")


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
