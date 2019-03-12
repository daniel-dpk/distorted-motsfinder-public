r"""@package motsfinder.metric.analytical.simple

Simple metrics in conformally flat form.


@b Examples

```
    # Create a Schwarzschild slice metric.
    metric = SchwarzschildSliceMetric(m=2)
    print("Schwarzschild radius: %s" % (2*metric.m))
    print("Horizon radial coordinate: %s" % metric.horizon_coord_radius())
```
"""

from abc import abstractmethod

import numpy as np
from scipy import linalg

from ..base import _ThreeMetric


__all__ = [
    "FlatThreeMetric",
    "SchwarzschildSliceMetric",
    "BrillLindquistMetric",
]


class FlatThreeMetric(_ThreeMetric):
    r"""Flat 3-metric in Cartesian coordinates."""
    def __init__(self):
        super(FlatThreeMetric, self).__init__()
        self._g = np.identity(3)

    def _mat_at(self, point):
        return self._g

    def diff(self, point, inverse=False, diff=1):
        if diff == 0:
            return self.at(point).inv if inverse else self.at(point).mat
        return np.zeros([3] * (diff+2))

    def diff_lnsqrtg(self, point):
        return np.zeros(3)

    def christoffel(self, point):
        return np.zeros((3, 3, 3))

    def christoffel_deriv(self, point):
        return np.zeros((3, 3, 3, 3))

    def ricci_tensor(self, point):
        return np.zeros((3, 3))

    def ricci_scalar(self, point):
        return 0.0


class _ConformallyFlatMetric(_ThreeMetric):
    r"""Base class for conformally flat 3-metrics.

    The metrics represented here are conformally flat and represented as
    \f[
        g = \psi^4 \delta ,
    \f]
    where \f$ \delta \f$ is the flat 3-metric.

    Subclasses need to provide a method for computing `psi`.
    If derivatives of `psi` can also be computed, we gain analytic
    differentiation of the metric and its inverse.
    """

    def _mat_at(self, point):
        return self.conformal_factor(point) * np.identity(3)

    def conformal_factor(self, point):
        r"""Compute the conformal factor at the given point."""
        return self.psi(point)**4

    def diff(self, point, inverse=False, diff=1):
        r"""Analytically compute derivatives \wrt x, y, z.

        @return Multidimensional list with indices `i1, i2, ..., k, l`
            corresponding to \f$\partial_{i_1}\partial_{i_2}\ldots g_{kl}\f$
            (or the inverse components if ``inverse==True``).

        @b Notes

        Conformally flat 3-metrics are represented here via
        \f[
            g = \psi^4 \delta,
            \qquad g^{-1} = \psi^{-4} \delta.
        \f]
        The derivatives at `x` are hence simply:
        \f{eqnarray*}{
            \partial_i g(x) &=&
                4 \psi^3 \partial_i\psi\ \mathrm{Id} \\
            \partial_i \partial_j g(x) &=&
                (12 \psi^2 \partial_i\psi \partial_j\psi
                 + 4\psi^3 \partial_i\partial_j\psi)\ \mathrm{Id} \\
            \partial_i g^{-1}(x) &=&
                -4 \psi^{-5} \partial_i\psi\ \mathrm{Id} \\
            \partial_i \partial_j g^{-1}(x) &=&
                (20 \psi^{-6} \partial_i\psi \partial_j\psi
                 - 4 \psi^{-5} \partial_i\partial_j\psi)\ \mathrm{Id}
        \f}
        """
        if diff == 0:
            return self.at(point).inv if inverse else self.at(point).mat
        Id = np.identity(3)
        if diff == 1:
            # indices mean: [i,j,k] <=> \partial_i g_jk
            psi, dpsi = self.psi(point, derivatives=True, max_deriv=1)
            if inverse:
                return np.array([-4 * dpsii * psi**(-5) * Id for dpsii in dpsi])
            return np.array([4 * dpsii * psi**3 * Id for dpsii in dpsi])
        if diff == 2:
            # indices mean: [i,j,k,l] <=> \partial_i \partial_j g_kl
            psi, dpsi, ddpsi = self.psi(point, derivatives=True, max_deriv=2)
            if inverse:
                return np.outer(
                    20*psi**(-6) * np.outer(dpsi, dpsi) - 4*psi**(-5) * ddpsi,
                    Id
                ).reshape(3, 3, 3, 3)
            return np.outer(
                12*psi**2 * np.outer(dpsi, dpsi) + 4*psi**3 * ddpsi,
                Id
            ).reshape(3, 3, 3, 3)
        raise NotImplementedError

    def diff_lnsqrtg(self, point):
        r"""Return x,y,z derivatives of ln(sqrt(det(g))).

        This computes the terms \f$ \partial_i \ln(\sqrt{g}) \f$
        and returns the results as a list (i.e. one element per value of `i`).

        For conformally flat metrics, the derivatives simply become
        \f[
            \partial_i \ln(\sqrt{g})
                = \partial_i \ln(\sqrt{\psi^{12}})
                = \partial_i \ln(\psi^6)
                = \frac{1}{\psi^6} 6 \psi^5 \partial_i \psi
                = 6 \frac{\partial_i\psi}{\psi}.
        \f]
        """
        psi, dpsi = self.psi(point, derivatives=True)
        return [6 * dpsii / psi for dpsii in dpsi]

    @abstractmethod
    def psi(self, point, derivatives=False, max_deriv=1):
        r"""Compute the value of psi at a given point.

        @param point
            3D point at which to evaluate psi.
        @param derivatives
            Whether to compute the derivatives in addition to the value of psi
            itself. Default is `False`, i.e. compute only psi itself.
        @param max_deriv
            Maximum derivative order to return.

        @return If ``derivatives==False``, returns just the value
            ``psi(point)``. Otherwise, returns a tuple of derivatives of
            `psi`, starting at 0 up to `max_deriv` (inclusive). Each
            derivative of order `> 0` is a nested sequence with indices
            `i1, i2, ...` meaning \f$\partial_{i_1}\partial_{i_2}\psi\f$.
        """
        pass


class SchwarzschildSliceMetric(_ConformallyFlatMetric):
    r"""Spatial slice of Schwarzschild metric in isotropic coordinates.

    In isotropic coordinates, the Schwarzschild spacetime metric becomes
    \f[
        {}^{(4)}g = - \left(\frac{1-m/(2r)}{1+m/(2r)}\right)^2 dt^2
                    + \psi^4 \delta ,
    \f]
    where \f$ \delta \f$ is the flat 3-metric and the conformal factor is
    \f[
        \psi = 1 + \frac{m}{2r}.
    \f]
    See e.g. Equation (10.89) in \ref schutz2011 "[1]" or Equation (3.16) in
    \ref straumann2004 "[2]".

    For a spatial slice, we just take the spatial part of this metric for
    constant \f$ t \f$.

    @b Notes

    Note that this is not the form the metric is usually written and as such,
    the physical quantities expressed in values of the "radial" parameter have
    different expressions. Specifically, the value of this parameter at which
    the horizon lies is given by (cf. \ref schutz2011 "[1]")
    \f$ r = m/2 \f$. This is not to be confused with the Schwarzschild radius
    \f$ \tilde r_s = 2m \f$, which has a geometric meaning related to the
    surface area of the horizon and at which the horizon lies in the
    *traditional* Schwarzschild coordinates.

    Derivatives of the conformal factor are computed in psi() using
    \f[
        \partial_i\psi = - \frac{m x^i}{2r^3},
        \qquad
        \partial_i\partial_j\psi
            = - \frac{m \delta_{ij}}{2r^3}
              + \frac{3}{2} \frac{m x^i x^j}{r^5}.
    \f]

    @b References

    \anchor schutz2011 [1] Schutz, Bernard. A first course in general
    relativity. Cambridge university press, 2009.

    \anchor straumann2004 [2] Straumann, Norbert. General relativity. Springer
    Science & Business Media, 2004.
    """
    def __init__(self, m):
        r"""Create a Schwarzschild slice metric of a given mass.

        @param m (float)
            ADM Mass of the metric.
        """
        super(SchwarzschildSliceMetric, self).__init__()
        self.m = float(m)

    def horizon_area(self):
        r"""Area of the Schwarzschild horizon.

        The area is given by \f$4 \pi \tilde r_s^2\f$, where
        \f$\tilde r_s = 2m\f$ is the *Schwarzschild radius*.
        Note that in isotropic coordinates, the value of this radius cannot be
        taken as being a meaningful value of our `r` coordinate.
        """
        return 16 * np.pi * self.m**2

    def horizon_coord_radius(self):
        r"""Schwarzschild radius \wrt isotropic coordinates.

        In the typical Schwarzschild coordinates, the horizon is at a "radial"
        coordinate ``2m``. This class implements the slice in isotropic
        coordinates, where the "radius" parameter value at which the horizon
        is located is ``m/2`` instead.
        """
        return self.m / 2.0

    def psi(self, point, derivatives=False, max_deriv=1):
        # see class docstring for the formulas
        r = linalg.norm(point)
        m = self.m
        psi = 1.0 if m == 0.0 else 1.0 + m/(2*r)
        if not derivatives:
            return psi
        dpsi = [- m * point[i]/(2*r**3) for i in range(3)]
        if max_deriv == 1:
            return psi, dpsi
        Id = np.identity(3)
        x = np.asarray(point)
        ddpsi = -Id * m/(2*r**3) + 3./2 * m * np.outer(x, x) / r**5
        if max_deriv == 2:
            return psi, dpsi, ddpsi
        raise NotImplementedError


class BrillLindquistMetric(_ConformallyFlatMetric):
    r"""Brill-Lindquist 3-metric of two non-rotating, non-spinning black holes.

    This metric due to Brill and Lindquist \ref brill1963 "[1]" describes, as
    implemented here, two non-spinning black holes at a moment of
    time-symmetry before evolving to a head-on collision.

    The metric itself is conformally flat, i.e.
    \f[
        g = \psi^4\ \delta ,
    \f]
    where \f$ \delta \f$ is the flat 3-metric and the conformal factor is
    obtained from
    \f[
        \psi = 1 + \frac{m_1}{2 r_1} + \frac{m_2}{2 r_2}.
    \f]
    Here, \f$ m_i \f$ are mass parameters (*not* the ADM masses of the two
    asymptotic ends) for the individual black holes and \f$ r_i \f$ the
    coordinate distances to the two punctures, i.e.
    \f[
        r_i = \Vert \vec x_i \Vert,
        \quad\mbox{where}\quad
            \vec x_i := \vec x - \vec c_i.
    \f]

    In this class, we choose to put the two punctures along one of the axes at
    equal coordinate distances from the coordinate origin. For example for the
    `z`-axis, we have \f$\vec c_1 = -\vec c_2 = (0, 0, d/2)\f$, where `d` can
    be specified upon construction.

    @b Notes

    Derivatives of the conformal factor are computed using
    \f{eqnarray*}{
        \partial_i\psi
            &=& - \frac{m_1 x_1^i}{2r_1^3}
                - \frac{m_2 x_2^i}{2r_2^3}
        \\
        \partial_i\partial_j\psi
            &=& - \frac{m_1 \delta_{ij}}{2r_1^3}
                - \frac{m_2 \delta_{ij}}{2r_2^3}
                + \frac{3}{2} \frac{m_1 x_1^i x_1^j}{r_1^5}
                + \frac{3}{2} \frac{m_2 x_2^i x_2^j}{r_2^5}.
    \f}

    @b References

    \anchor brill1963 [1] Brill, Dieter R., and Richard W. Lindquist.
        "Interaction energy in geometrostatics." Physical Review 131.1 (1963):
        471.
    """
    def __init__(self, d, m1=1, m2=1, axis='z'):
        r"""Construct a Brill-Lindquist 3-metric.

        The ADM mass of the metric (specifically, the ADM mass of the
        asymptotic end \f$ \Vert \vec x \Vert \to \infty\f$) will be `m1+m2`.

        @param d (float)
            Distance parameter. This represents the coordinate distance of the
            two punctures. If it is chosen low enough, a common horizon forms
            which is usually interpreted as the two black holes having merged
            into a single black hole. Separate inner MOTSs may still exist.
            The value at which a common horizon forms for equal mass black
            holes has been determined to low accuracy in \ref brill1963 "[1]"
            and with higher accuracy in \ref lages2010 "[2]", see Table 2.2
            therein. It reads approximately \f$d/m \approx 1.532395\f$.
        @param m1 (float)
            Mass parameter of first black hole. Default is `1`.
        @param m2 (float)
            Mass parameter of second black hole. Default is `1`.
        @param axis (``{'x', 'y', 'z'}``, optional)
            Symmetry axis on which to place the two punctures. Default is
            `'z'`.

        @b References

        \anchor lages2010 [2] Lages, Norbert. Apparent Horizons and Marginally
            Trapped Surfaces in Numerical General Relativity. Diss. PhD thesis,
            University of Jena, 2010.
        """
        super(BrillLindquistMetric, self).__init__()
        self._m1 = float(m1)
        self._m2 = float(m2)
        self._d = d
        axis = dict(x=0, y=1, z=2).get(axis, axis)
        self._p1 = np.zeros(3)
        self._p2 = np.zeros(3)
        self._p1[axis] = d/2.0
        self._p2[axis] = -d/2.0

    @property
    def m1(self):
        r"""Mass parameter of the first black hole."""
        return self._m1

    @property
    def m2(self):
        r"""Mass parameter of the second black hole."""
        return self._m2

    @property
    def d(self):
        r"""Distance parameter of the black holes."""
        return self._d

    def psi(self, point, derivatives=False, max_deriv=1):
        # see class docstring for the formulas
        m1, m2 = self._m1, self._m2
        X1 = np.asarray(point) - self._p1
        X2 = np.asarray(point) - self._p2
        x1, y1, z1 = X1
        x2, y2, z2 = X2
        r1 = np.sqrt(x1**2 + y1**2 + z1**2)
        r2 = np.sqrt(x2**2 + y2**2 + z2**2)
        psi = 1.0 + m1 / (2 * r1) + m2 / (2 * r2)
        if not derivatives:
            return psi
        dpsi = - m1 * X1/(2*r1**3) - m2 * X2/(2*r2**3)
        if max_deriv == 1:
            return psi, dpsi
        Id = np.identity(3)
        ddpsi = (
            -Id * (m1/(2*r1**3) + m2/(2*r2**3))
            + 3 * m1 * np.outer(X1, X1) / (2*r1**5)
            + 3 * m2 * np.outer(X2, X2) / (2*r2**5)
        )
        if max_deriv == 2:
            return psi, dpsi, ddpsi
        raise NotImplementedError
