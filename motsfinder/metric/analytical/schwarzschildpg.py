r"""@package motsfinder.metric.analytical.schwarzschildpg

Schwarzschild slice in Painleve-Gullstrand coordinates.

Represents a slice of the Schwarzschild spacetime in Painleve-Gullstrand
coordinates based on [1].

@b References

[1] Booth, Ivan, Robie A. Hennigar, and Saikat Mondal. "Marginally outer
    trapped surfaces in the Schwarzschild spacetime: Multiple
    self-intersections and extreme mass ratio mergers." Physical Review D
    102.4 (2020): 044031.
"""

import math

import numpy as np

from ..base import (_ThreeMetric, trivial_lapse, trivial_dtlapse,
                    trivial_shift, trivial_dtshift)


__all__ = [
    "SchwarzschildPGSlice",
]


class SchwarzschildPGSlice(_ThreeMetric):
    r"""3-metric of a slice of Schwarzschild spacetime in Painleve-Gullstrand coordinates.

    Implementation based on the formulas in [1].

    @b References

    [1] Booth, Ivan, Robie A. Hennigar, and Saikat Mondal. "Marginally outer
        trapped surfaces in the Schwarzschild spacetime: Multiple
        self-intersections and extreme mass ratio mergers." Physical Review D
        102.4 (2020): 044031.
    """

    def __init__(self, M=1):
        r"""Create a metric object for a slice of Schwarzschild spacetime in Painleve-Gullstrand coordinates.

        @param M
            Mass parameter. Default is 1.
        """
        super().__init__()
        self._M = float(M)

    @property
    def M(self):
        r"""ADM mass of the Schwarzschild spacetime."""
        return self._M

    def _mat_at(self, point):
        r"""Three metric at a given point in Cartesian (x,y,z) coordinates."""
        return np.identity(3)

    def diff(self, point, inverse=False, diff=1):
        if inverse:
            return self._compute_inverse_diff(point, diff=diff)
        if diff == 0:
            return self._mat_at(point)
        return np.zeros([3] * (diff+2))

    def diff_lnsqrtg(self, point):
        return np.zeros(3)

    def get_curv(self):
        return SchwarzschildPGSliceCurv(self)

    def get_lapse(self):
        return trivial_lapse

    def get_dtlapse(self):
        return trivial_dtlapse

    def get_shift(self):
        return trivial_shift

    def get_dtshift(self):
        return trivial_dtshift


class SchwarzschildPGSliceCurv():
    r"""Extrinsic curvature of a ``tau=const`` slice of the Schwarzschild
    spacetime in Painleve-Gullstrand coordinates."""

    def __init__(self, metric):
        self._g = metric

    def __call__(self, point, diff=0):
        x, y, z = point
        M = self._g.M
        r = np.sqrt(x**2 + y**2 + z**2)
        ta = np.arccos(z/r)
        ph = np.arctan2(y, x)
        r_x = x/r
        r_y = y/r
        r_z = z/r
        ta_x = x*z / (r**3 * np.sin(ta))
        ta_y = y*z / (r**3 * np.sin(ta))
        ta_z = z**2/(r**3*np.sin(ta)) - 1/(r * np.sin(ta))
        ph_x = - y / (x**2 * np.cos(ph)**2)
        ph_y = np.cos(ph)**2 / x
        ph_z = 0.0
        Krr = np.sqrt(M/(2*r**3))
        Ktt = -np.sqrt(2*M*r)
        Kpp = -np.sqrt(2*M*r) * np.sin(ta)**2
        Kxx = Krr * r_x**2 + Kpp * ph_x**2 + Ktt * ta_x**2
        Kxy = Krr * r_x*r_y + Kpp * ph_x*ph_y + Ktt * ta_x*ta_y
        Kxz = Krr * r_x*r_z + Ktt * ta_x*ta_z
        Kyy = Krr * r_y**2 + Kpp * ph_y**2 + Ktt * ta_y**2
        Kyz = Krr * r_y*r_z + Ktt * ta_y*ta_z
        Kzz = Krr * r_z**2 + Ktt * ta_z**2
        if diff == 0:
            return -np.array([
                [Kxx, Kxy, Kxz],
                [Kxy, Kyy, Kyz],
                [Kxz, Kyz, Kzz],
            ])
        raise NotImplementedError
