r"""@package motsfinder.metric.analytical.kerrks

Kerr slice in Kerr-Schild coordinates.

The class defined here can be used to produce the data for a slice of
Kerr spacetime in Kerr-Schild form.
"""

import math

import numpy as np
from scipy import linalg

from ..base import _ThreeMetric, trivial_dtlapse, trivial_dtshift


__all__ = [
    "KerrKSSlice",
]


class KerrKSSlice(_ThreeMetric):
    r"""Data for a slice of Kerr spacetime in Kerr-Schild coordinates.

    The formulas implemented here are based on [1] and [2].

    @b References

    [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker. "Initial
        data and coordinates for multiple black hole systems." Physical Review
        D 59.2 (1998): 024015.

    [2] Moreno, Claudia, Dar\'{i}o N\'{u}\~{n}ez, and Olivier Sarbach. "Kerr-
        Schild type initial data for black holes with angular momenta". Class.
        Quant. Grav. 19 (2002) 6059-6073.
    """

    def __init__(self, M=1, a=0):
        r"""Create a Kerr slice in Kerr-Schild form.

        @param M
            ADM mass of the spacetime.
        @param a
            Angular momentum parameter of the black hole.
        """
        super().__init__()
        self._M = float(M)
        self._a = float(a)

    @property
    def M(self):
        r"""ADM mass of the spacetime (\ie of the black hole)."""
        return self._M

    @property
    def a(self):
        r"""Angular momentum parameter of the black hole."""
        return self._a

    def rho(self, point):
        r"""The variable \f$\rho\f$ used in [1].

        @b References

        [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker.
            "Initial data and coordinates for multiple black hole systems."
            Physical Review D 59.2 (1998): 024015.
        """
        return linalg.norm(point)

    def r(self, point):
        r"""The variable \f$r\f$ used in [1].

        This is eq. (11) of [1].

        @b References

        [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker.
            "Initial data and coordinates for multiple black hole systems."
            Physical Review D 59.2 (1998): 024015.
        """
        rho = self.rho(point)
        a = self._a
        x = point
        return np.sqrt(0.5*(math.pow(rho,2.0) - math.pow(a,2.0)) + np.sqrt(0.25*math.pow(math.pow(rho,2.0) - math.pow(a,2.0),2.0) + math.pow(a*x[2],2.0)))

    def norm(self,point):
        r"""The denominator of the spatial components of the ingoing null
            vector \f$\ell_{\mu}\f$.

        The ingoing null vector is eq. (9) of [1].

        @b References

        [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker.
            "Initial data and coordinates for multiple black hole systems."
            Physical Review D 59.2 (1998): 024015.
        """

        r = self.r(point)
        a = self._a
        return math.pow(r,2.0) + math.pow(a,2.0)

    def llower(self,point):
        r"""The ingoing null vector \f$\ell_{\mu}\f$.

        This is eq. (9) of [1].

        @b References

        [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker.
            "Initial data and coordinates for multiple black hole systems."
            Physical Review D 59.2 (1998): 024015.
        """

        r = self.r(point)
        x = point
        a = self._a
        norm = self.norm(point)
        return [(r*x[0] + a*x[1])/norm, (r*x[1] - a*x[0])/norm, x[2]/r]

    def H(self, point):
        r"""Scalar H function.

        This is eq. (8) of [1].

        @b References

        [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker.
            "Initial data and coordinates for multiple black hole systems."
            Physical Review D 59.2 (1998): 024015.
        """
        r = self.r(point)
        a = self._a
        x = point
        return self._M*math.pow(r,3.0)/(math.pow(r,4.0) + math.pow(a*x[2],2.0))

    def _mat_at(self, point):
        r"""Three metric.

        This is eq. (15) of [1], using same notation.

        @b References

        [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker.
            "Initial data and coordinates for multiple black hole systems."
            Physical Review D 59.2 (1998): 024015.
        """
        H = self.H(point)
        llower = self.llower(point)

        return np.identity(3) + np.array([
            [2.0 * H * llower[i] * llower[j] for j in range(3)]
            for i in range(3)
        ])

    def diff(self, point, inverse=False, diff=1):
        if inverse:
            return self._compute_inverse_diff(point, diff=diff)
        if diff == 0:
            return self._mat_at(point)

        raise NotImplementedError

    def get_curv(self):
        return KerrKSSliceCurv(self)

    def get_lapse(self):
        return KerrKSSliceLapse(self)

    def get_shift(self):
        return KerrKSSliceShift(self)

    def get_dtlapse(self):
        return trivial_dtlapse

    def get_dtshift(self):
        return trivial_dtshift


class KerrKSSliceLapse():
    r"""Lapse function of the Kerr slice in Kerr-Schild form.

    This is eq. (13) of [1].

    @b References

    [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker. "Initial
        data and coordinates for multiple black hole systems." Physical Review
        D 59.2 (1998): 024015.
    """
    def __init__(self, metric):
        self._g = metric

    def __call__(self, point, diff=0):
        if diff != 0:
            raise NotImplementedError

        H = self._g.H(point)

        return 1.0 / np.sqrt(1.0 + 2.0*H)


class KerrKSSliceShift():
    r"""Shift vector field of the Kerr slice in Kerr-Schild form.

    This is the raised version of eq. (14) of [1].

    @b References

    [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker. "Initial
        data and coordinates for multiple black hole systems." Physical Review
        D 59.2 (1998): 024015.
    """
    def __init__(self, metric):
        self._g = metric

    def __call__(self, point, diff=0):
        if diff != 0:
            raise NotImplementedError

        H = self._g.H(point)
        llower = self._g.llower(point)

        betal = np.asarray([2.0 * H * llower[i] for i in range(3)]) # eq. (14) of [1]
        g_inv = self._g.diff(point, inverse=True, diff=0)
        beta = np.einsum('ij,j->i', g_inv, betal)
        return beta


class KerrKSSliceCurv():
    r"""Extrinsic curvature of the Kerr slice in Kerr-Schild form.

    This is eq. (3) of [2], with difference in overall sign convention taken
    into account here, obtained by comparison of a = 0 case with Schwarschild.

    @b References

    [1] Matzner, Richard A., Mijan F. Huq, and Deirdre Shoemaker. "Initial
        data and coordinates for multiple black hole systems." Physical Review
        D 59.2 (1998): 024015.

    [2] Moreno, Claudia, Dar\'{i}o N\'{u}\~{n}ez, and Olivier Sarbach. "Kerr-
        Schild type initial data for black holes with angular momenta". Class.
        Quant. Grav. 19 (2002) 6059-6073.
    """
    def __init__(self, metric):
        self._g = metric

    def __call__(self, point, diff=0):

        M = self._g.M
        rho = self._g.rho(point)
        a = self._g.a
        x = point # point is (x,y,z) components

        if diff == 0: # 0th derivative

            rInnerRoot = np.sqrt(4.0*math.pow(a*x[2],2.0) + math.pow(math.pow(rho,2.0) - math.pow(a,2.0),2.0))

            r = self._g.r(point)

            H = self._g.H(point)

            norm = self._g.norm(point)

            llower = [1.0, (r*x[0] + a*x[1])/norm, (r*x[1] - a*x[0])/norm, x[2]/r]

            lupper = [1.0, -(r*x[0] + a*x[1])/norm, -(r*x[1] - a*x[0])/norm, -x[2]/r]

            alpha = 1.0 / np.sqrt(1.0 + 2.0*H)

            # In below, using the K_{ij} formula (eq. 3) in [2], with
            # difference in overall sign convention taken into account here,
            # obtained by comparison of a = 0 case with Schwarschild
            # Note that in eq. (3) of [2], the first term with the time
            # derivative is 0 in our case for Kerr
            # Note that K_{ij} is symmetric

            # Derivative of H with respect to i for i = 1,2,3
            # (1 = x, 2 = y, 3 = z); note that derivatives wrt to (time)
            # are 0, but are not used, and are present only so that
            # indexing works out (so i = 0 = t).
            # For example, dHd[1] is the derivative of H (eq. 8 of [1])
            # with respect to x, etc.
            dHd = [0 ,

                   -M * x[0] * math.pow(r,3.0) * (4.0 * math.pow(r,4.0)/(math.pow(a*x[2],2.0) + math.pow(r,4.0)) - 3) / (rInnerRoot * (math.pow(a*x[2],2.0) + math.pow(r,4.0))) ,

                   -M * x[1] * math.pow(r,3.0) * (4.0 * math.pow(r,4.0)/(math.pow(a*x[2],2.0) + math.pow(r,4.0)) - 3) / (rInnerRoot * (math.pow(a*x[2],2.0) + math.pow(r,4.0))) ,

                   -M * x[2] * r / (math.pow(a*x[2],2.0) + math.pow(r,4.0)) * (2.0 * math.pow(r,2.0) * (math.pow(a,2.0) + 2.0 * math.pow(r,2.0) * (math.pow(r,2.0) + math.pow(a,2.0)) / rInnerRoot) / (math.pow(a*x[2],2.0) + math.pow(r, 4.0)) - 3.0 * (math.pow(r,2.0) + math.pow(a,2.0)) / rInnerRoot) ]

            # Derivative of x component of llower (eq. 9 in [1]) with
            # respect to i for i = 1,2,3 (1 = x, 2 = y, 3 = z).
            # Note that derivatives wrt to (time) are 0, but are not used,
            # and are present only so that indexing works out (so i = 0 = t).
            # For example, dllower1d[1] is the derivative of the x component
            # of llower (eq. 9 in [1]) with respect to x, etc.
            dllower1d = [0,

                         -r / (math.pow(a,2.0) + math.pow(r,2.0)) * (2.0 * r * (a * x[1] + x[0] * r) * x[0] / (rInnerRoot * (math.pow(a,2.0) + math.pow(r,2.0))) - math.pow(x[0],2.0) / rInnerRoot - 1) ,

                         -1.0 / (math.pow(a,2.0) + math.pow(r,2.0)) * ((a * x[1] + x[0] * r) * x[1] * 2.0 * math.pow(r,2.0) / (rInnerRoot * (math.pow(a,2.0) + math.pow(r,2.0))) - a - x[0] * x[1] * r / rInnerRoot) ,

                         -2.0 * x[2] / rInnerRoot * ((a * x[1] + x[0] * r) / (math.pow(a,2.0) + math.pow(r,2.0)) - x[0] / (2.0 * r)) ]

            # Derivative of y component of llower (eq. 9 in [1]) with respect
            # to i for i = 1,2,3 (1 = x, 2 = y, 3 = z); note that derivatives
            # wrt to (time) are 0, but are not used, and are present only so
            # that indexing works out (so i = 0 = t).
            # For example, dllower2d[1] is the derivative of the y component
            # of llower (eq. 9 in [1]) with respect to x, et.c
            dllower2d = [0,

                         -1.0 / (math.pow(a,2.0) + math.pow(r,2.0)) * ((-a * x[0] + r * x[1]) * x[0] * 2.0 * math.pow(r,2.0) / (rInnerRoot * (math.pow(a,2.0) + math.pow(r,2.0))) + a - x[0] * x[1] * r / rInnerRoot) ,

                         -r / (math.pow(a,2.0) + math.pow(r,2.0)) * (2.0 * x[1] * r * (r * x[1] - a * x[0]) / (rInnerRoot * (math.pow(a,2.0) + math.pow(r,2.0))) - math.pow(x[1],2.0) / rInnerRoot - 1) ,

                         -2.0 * x[2] / rInnerRoot * ((r * x[1] - a * x[0]) / (math.pow(a,2.0) + math.pow(r,2.0)) - x[1] / (2.0 * r)) ]

            # Derivative of z component of llower (eq. 9 in [1]) with respect
            # to i for i = 1,2,3 (1 = x, 2 = y, 3 = z); note that derivatives
            # wrt to (time) are 0, but are not used, and are present only so
            # that indexing works out (so i = 0 = t).
            # For example, dllower3d[1] is the derivative of the z component
            # of llower (eq. 9 in [1]) with respect to x, etc.
            dllower3d = [0,

                         -x[0] * x[2] / (r * rInnerRoot) ,

                         -x[1] * x[2] / (r * rInnerRoot) ,

                         -math.pow(x[2],2.0) * (math.pow(a,2.0) + math.pow(r,2.0)) / (math.pow(r,3.0) * rInnerRoot) + 1.0 / r ]

            # dllowerArr array is made so I can access the derivatives of
            # llower components using indexing in sumInCurv
            # For example, dllowerArr[1][2] is the derivative of the
            # x component of llower (eq. 9 in [1]) with respect to y, etc.
            dllowerArr = [0, dllower1d, dllower2d, dllower3d]

            # calculates the contraction $$l^c(Vl_il_j)_{,c}$$ in [2].
            # Note that $$_{,c}$$ in eq. (3) of [2] means partial
            # derivative with respect to c.
            def sumInCurv(i,j):
                val = 0
                for c in range(1,4):
                    val = val + lupper[c] * (dHd[c] * llower[i] * llower[j] + H * dllowerArr[i][c] * llower[j] + H * llower[i] * dllowerArr[j][c])
                return val

            # Note: global negative sign for all K_{ij} introduced
            # after comparison of a = 0 with Schwarzschild case.
            K_11 = -alpha * 2.0 * (H * sumInCurv(1,1) - dHd[1] * llower[1] - H * dllowerArr[1][1])

            K_12 = -alpha * (2.0 * H * sumInCurv(1,2) - dHd[2] * llower[1] - H * dllowerArr[1][2] - dHd[1] * llower[2] - H * dllowerArr[2][1])

            K_13 = -alpha * (2.0 * H * sumInCurv(1,3) - dHd[3] * llower[1] - H * dllowerArr[1][3] - dHd[1] * llower[3] - H * dllowerArr[3][1])

            K_22 = -alpha * 2.0 * (H * sumInCurv(2,2) - dHd[2] * llower[2] - H * dllowerArr[2][2])

            K_23 = -alpha * (2.0 * H * sumInCurv(2,3) - dHd[3] * llower[2] - H * dllowerArr[2][3] - dHd[2] * llower[3] - H * dllowerArr[3][2])

            K_33 = -alpha * 2.0 * (H * sumInCurv(3,3) - dHd[3] * llower[3] - H * dllowerArr[3][3])

            return np.array([[K_11, K_12, K_13],
                             [K_12, K_22, K_23],
                             [K_13, K_23, K_33]])

        raise NotImplementedError
