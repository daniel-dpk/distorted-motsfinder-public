r"""@package motsfinder.axisym.curve.starshapedcurve

Represents surfaces in star-shaped parameterization relative to some origin.


@b Examples

```
    metric = FlatThreeMetric()
    c1 = StarShapedCurve.create_sphere(.5, 20, origin=(0, 0.9),
                                       metric=metric)
    c2 = StarShapedCurve.create_sphere(.5, 20, origin=(0, -0.5),
                                       metric=metric)
    c3 = StarShapedCurve(CosineSeries([1.2, 0.2, 0.4, 0.1]),
                         metric=metric)
    c1.plot_curves(c1, c2, c3)
```
"""

import numbers

import numpy as np

from ...numutils import binomial_coeffs
from ...exprs.trig import CosineSeries
from .expcalc import ExpansionCalc
from .expcurve import ExpansionCurve


__all__ = [
    "StarShapedCurve",
]


class StarShapedCurve(ExpansionCurve):
    r"""Represent as angle-dependent distance from some origin.

    This class can represent star-shaped curves with respect to some arbitrary
    origin representing axisymmetric smooth closed surfaces in x, y, z
    coordinate space. The curve is represented as
    \f[
        \gamma(\lambda) = x_0 + h(\lambda) (\sin\lambda, 0, \cos\lambda),
        \qquad \lambda \in [0,\pi],
    \f]
    where \f$ x_0 \f$ is the `origin` and `h` the *horizon function*.

    This class can compute geometric quantities like the expansion along the
    surface represented by this curve. The geometry of the spatial slice is
    specified by supplying a Riemannian 3-metric and optionally an extrinsic
    curvature when constructing this curve.

    See the examples in the package description of curve.starshapedcurve.
    """

    def __init__(self, h, metric, origin=(0, 0, 0), name=''):
        r"""Create a star-shaped parameterized curve.

        @param h (exprs.numexpr.NumericExpression)
            Horizon function given as function of the angle `theta` w.r.t. the
            `z`-axis.
        @param metric
            The Riemannian 3-metric defining the geometry of the surrounding
            space.
        @param origin (float or 2- or 3-tuple/list)
            Origin with respect to which this curve is parameterized.
            A float is interpreted as the `z`-axis coordinate value.
            Two elements are taken as `x` and `z` component, although for
            axisymmetry the `x` component must always be zero...
            Three elements are interpreted as `x, y, z` components, where the
            `x` and `y` components should be zero.
        @param name
            Name of this curve. This may be used when printing information
            about this curve or as label in plots.
        """
        super(StarShapedCurve, self).__init__(h, metric, name=name)
        self._origin = None
        self.set_origin(origin)

    @staticmethod
    def create_sphere(radius, num, metric, origin=(0, 0, 0)):
        r"""Create a circle of given radius representing a sphere in 3D.

        @param radius
            Radius (in coordinate space) of the created circle.
        @param num
            Resolution of the horizon function. This only affects subsequent
            modification, since one coefficient is enough to represent a
            circle.
        @param metric
            The Riemannian 3-metric defining the geometry of the surrounding
            space.
        @param origin
            Origin with respect to which this curve is parameterized. See
            #__init__() for details.
        """
        h = CosineSeries([radius], domain=(0, np.pi))
        curve = StarShapedCurve(h=h, metric=metric, origin=origin)
        return curve.resample(num)

    def h_diff(self, param):
        return np.array([np.sin(param), 0.0, np.cos(param)])

    @property
    def origin(self):
        r"""Origin with respect to which the curve is represented."""
        return self._origin

    def set_origin(self, origin):
        r"""Change the origin to the given point.

        The horizon function is not modified, i.e. it is *moved* to the new
        place.
        """
        if isinstance(origin, numbers.Number):
            origin = np.array([0., origin])
        else:
            origin = np.asarray(origin, dtype=float)
            if origin.size == 3:
                origin = np.array([origin[0], origin[2]])
        self._origin = origin

    def _create_calc_obj(self, param):
        return _StarShapedExpansionCalc(
            curve=self, h_fun=self._get_evaluators(), param=param,
            metric=self.metric,
        )

    def copy(self):
        return type(self)(h=self.h.copy(), metric=self.metric,
                          origin=self.origin, name=self.name)

    def __call__(self, param, xyz=False):
        r = self._get_evaluators()(param)
        x = r * np.sin(param) + self._origin[0]
        z = r * np.cos(param) + self._origin[1]
        return self._prepare_result(x, z, xyz)

    def _diff(self, param, diff):
        r"""Compute derivatives of the component functions.

        Recall that the component functions are
        \f[
            x(\lambda) = h(\lambda) \sin(\lambda), \qquad
            z(\lambda) = h(\lambda) \cos(\lambda).
        \f]
        The derivatives are hence given by
        \f[
            \partial_\lambda^n \gamma(\lambda) =
                \sum_{k=0}^n
                    {n \choose k} h^{(n-k)}(\lambda)
                    \ \partial_\lambda^k
                    \left(\begin{array}{@{}c@{}}
                        \sin(\lambda) \\
                        \cos(\lambda)
                    \end{array}\right).
        \f]
        """
        h_ev = self._get_evaluators()
        s = np.sin(param)
        c = np.cos(param)
        # The first two derivatives are hard-coded to gain some speed.
        # The general formula below also works for these cases (diff == 1, 2).
        if diff == 1:
            h = h_ev(param)
            hp = h_ev.diff(param)
            return hp*s + h*c, hp*c - h*s
        if diff == 2:
            h = h_ev(param)
            hp = h_ev.diff(param)
            hpp = h_ev.diff(param, 2)
            return hpp*s + 2*hp*c - h*s, hpp*c - 2*hp*s - h*c
        circ_diffs = np.array([[s, c], [c, -s], [-s, -c], [-c, s]])
        coeffs = binomial_coeffs(diff)
        return np.sum(
            [coeffs[k] * h_ev.diff(param, diff-k) * circ_diffs[k % 4]
             for k in range(diff+1)],
            axis=0
        )


class _StarShapedExpansionCalc(ExpansionCalc):
    r"""Expansion cache/calculator for StarShapedCurve objects.

    This computes and caches interim results of the expansion and its
    functional derivatives w.r.t. the horizon function for curves in
    star-shaped parameterization.

    This object is usually only created by the StarShapedCurve object.
    """

    def _compute_s_ds_X_Y(self):
        r"""Compute the terms we need to compute the expansion.

        To compute the covariant normal \f$ s_i \f$ on the curve represented
        by the StarShapedCurve, we define the function
        \f[
            F(\vec x) = r - h(\lambda(\vec x)),
        \f]
        This means that on the surface defined by `h` we will have `F = 0`.
        The covariant normal is therefore obtained via
        \f[
            s_i = \nabla_i F = \partial_i F = \frac{x^i}{r} - X_i h',
        \f]
        where \f$ X_i = \partial_i\lambda \f$, compare Thornburg [1].
        We furthermore need the derivatives
        \f[
            \partial_i s_j
                = \frac{\delta_{ij}}{r} - \frac{x^i x^j}{r^3}
                    - X_{ij}h' - X_i X_j h'',
        \f]
        where \f$ \delta_{ij} \f$ is the Kronecker delta and we set
        \f$ X_{ij} = \partial_i\partial_j\lambda \f$. Since this will always
        be evaluated on the curve (trial surface), we replace
        \f$ r \to h \f$ in the following.

        Note that since we cannot differentiate between variables based on how
        many indices they have, we use `Y` for the components \f$ X_{ij} \f$.

        Since we know the coordinate transformations analytically,
        \f[
            \cos\lambda = \frac{z}{r},
        \f]
        we can easily compute all the required terms and obtain
        \f[
            X_1 = \frac{\cos\lambda}{h}, \qquad
            X_2 = 0, \qquad
            X_3 = -\frac{\sin\lambda}{h},
        \f]
        \f[
            X_{11} = -\frac{\sin(2\lambda)}{h^2}, \qquad
            X_{22} = \frac{\cot\lambda}{h^2}, \qquad
            X_{33} = \frac{\sin(2\lambda)}{h^2}, \qquad
            X_{13} = -\frac{\cos(2\lambda)}{h^2} = X_{31},
        \f]
        \f[
            s_1 = \sin\lambda-\frac{h'}{h} \cos\lambda, \qquad
            s_2 = 0, \qquad
            s_3 = \cos\lambda+\frac{h'}{h} \sin\lambda,
        \f]
        \f{eqnarray*}{
            \partial_1 s_1 &=& \frac{1}{h^2} \left(
                    (h-h'')\cos^2\lambda + h'\sin(2\lambda)
                \right),
            \\
            \partial_2 s_2 &=& \frac{1}{h^2} \left(
                    h-h' \frac{\cos\lambda}{\sin\lambda}
                \right),
            \\
            \partial_3 s_3 &=& \frac{1}{h^2} \left(
                    (h-h'')\sin^2\lambda - h'\sin(2\lambda)
                \right),
            \\
            \partial_1 s_3 &=& \frac{1}{2h^2} \left(
                    (h''-h)\sin(2\lambda) + 2h'\cos(2\lambda)
                \right)
            = \partial_3 s_1.
        \f}

        @b References

        [1] Thornburg, Jonathan. "A fast apparent horizon finder for
            three-dimensional Cartesian grids in numerical relativity."
            Classical and quantum gravity 21.2 (2003): 743.
        """
        ta = self.param
        h, dh, ddh = self.h, self.dh, self.ddh
        sin = np.sin(ta)
        cos = np.cos(ta)
        tan = np.tan(ta)
        sin2 = sin**2
        cos2 = cos**2
        sin2x = np.sin(2*ta)
        cos2x = np.cos(2*ta)
        h2 = h**2
        s1 = sin - dh/h * cos
        s3 = cos + dh/h * sin
        s = np.array([s1, 0.0, s3])
        ds = np.zeros((3, 3))
        ds[0,0] = 1/h2 * ((h-ddh) * cos2 + dh * sin2x)
        ds[0,2] = 1/(2*h2) * ((ddh-h) * sin2x + 2*dh*cos2x)
        if tan == 0.0:
            ds[1,1] = 1/h
        else:
            ds[1,1] = 1/h2 * (h - dh/tan)
        ds[2,0] = ds[0,2]
        ds[2,2] = 1/h2 * ((h-ddh) * sin2 - dh*sin2x)
        X = np.array([cos/h, 0.0, - sin/h])
        Y = np.zeros((3, 3))
        Y[0,0] = - sin2x/h2
        Y[1,1] = 1/(h2*tan) if tan != 0.0 else float('nan')
        Y[2,2] = sin2x/h2
        Y[0,2] = Y[2,0] = -cos2x/h2
        return s, ds, X, Y
