r"""@package motsfinder.axisym.curve.parametriccurve

General parametric curve in the x-z-plane.

This curve is not aware of the geometry it is embedded in. It purely lives in
coordinate space. This makes it suitable for use as reference curve for
refparamcurve.RefParamCurve objects.


@b Examples

```
    curve = ParametricCurve(SineSeries([1, -.4, 0, .3]),
                            CosineSeries([0, .6, -.6, -.3]))
    curve.plot()
```

Parametric curves can also be created directly from other curves:

```
    # This is a simple curve in star-shaped parameterization.
    c0 = StarShapedCurve(CosineSeries([1.2, 0.2, 0.4, 0.1]),
                         metric=FlatThreeMetric)

    # Define an "offset function".
    from math import sin, cos, exp, pi
    func = lambda x: .2 * exp(cos(x)) * sin(x) * sin(5*x) - .2

    # Convert the offset function to a "horizon function" expression.
    f = CosineSeries.from_function(func, 30, domain=(0, pi))

    # Apply the offset using the reference parametrisation.
    c1 = RefParamCurve(f, c0)

    # Convert it to a parametric curve and plot it.
    curve = ParametricCurve.from_curve(c1, num=20)
    curve.plot()
```

Continuing the previous example, we can plot the normal vectors too:

```
    ax = curve.plot(label='curve', show=False)
    space = np.linspace(0, np.pi, 50)
    for i, t in enumerate(space):
        p = curve(t)
        t = curve.normal(t)
        ax.plot(*np.array([p, p+0.2*t]).T, ".-g",
                label=None if i else 'normal')
        p[0] *= -1; t[0] *= -1
        ax.plot(*np.array([p, p+0.2*t]).T, ".-g")
    ax.legend(loc=0)
    ax.set_ylim(-1.5, 2)
```
"""

import numpy as np
from scipy.linalg import norm
from scipy.integrate import solve_ivp, quad

from ...exprs.trig import SineSeries, CosineSeries
from ...exprs.cheby import Cheby
from .baseparamcurve import BaseParametricCurve


__all__ = [
    "ParametricCurve",
]


class ParametricCurve(BaseParametricCurve):
    r"""Parametric curve in the x-z-plane.

    This curve can take any numeric expression for its `x` and `z` component
    functions to represent a parametric curve, i.e. it is represented as
    \f[
        \gamma(\lambda) = (x(\lambda), 0, z(\lambda)),
        \qquad \lambda \in [0,\pi].
    \f]

    See the examples in the package documentation of curve.parametriccurve.
    """

    def __init__(self, s_fun=None, t_fun=None, freeze=True, name='', **kw):
        r"""Create a parametric curve from two component functions.

        @param s_fun (exprs.numexpr.NumericExpression)
            Component function for the `x`-component of the curve. For curves
            starting and ending on the `z`-axis, a suitable choice is a
            function expanded into a pure sine series (see
            exprs.trig.SineSeries).
        @param t_fun (exprs.numexpr.NumericExpression)
            Component function for the `z`-component of the curve. For curves
            starting and ending on the `z`-axis, a suitable choice is a
            function expanded into a pure cosine series (see
            exprs.trig.CosineSeries).
        @param freeze (boolean)
            Since these curves often will not change during their life time,
            their evaluators can be frozen upon construction of the curve.
            This makes subsequent evaluation of the curve, its tangent, and
            its normal (and derivatives thereof) very efficient.
        @param name
            Name of this curve. This may be used when printing information
            about this curve or as label in plots.
        """
        kw['s_fun'] = s_fun
        kw['t_fun'] = t_fun
        if 'x_fun' in kw:
            if s_fun is not None and kw['x_fun'] is not None:
                raise TypeError(
                    "Parameters `s_fun` and `x_fun` are mutually exclusive."
                )
            kw['s_fun'] = kw.pop('x_fun')
        if 'z_fun' in kw:
            if t_fun is not None and kw['z_fun'] is not None:
                raise TypeError(
                    "Parameters `t_fun` and `z_fun` are mutually exclusive."
                )
            kw['t_fun'] = kw.pop('z_fun')
        if kw['s_fun'] is None or kw['t_fun'] is None:
            raise TypeError("Missing required argument `s_fun` and/or `t_fun`.")
        super().__init__(**kw)

    @classmethod
    def from_curve(cls, curve, num, x_cls=SineSeries, z_cls=CosineSeries):
        r"""Static method to sample any curve to convert it to a parametric curve.

        See the package description for curve.parametriccurve for an example.

        @param curve (basecurve.BaseCurve)
            The curve to sample and represent.
        @param num (int)
            Resolution for sampling the curve. Higher values will lead to a
            more accurate match to the original `curve`.
        @param x_cls (exprs.series.SeriesExpression)
            The series class to expand the x-component function into. Default
            is exprs.trig.SineSeries.
        @param z_cls (exprs.series.SeriesExpression)
            The series class to expand the z-component function into. Default
            is exprs.trig.CosineSeries.
        """
        with curve.fix_evaluator():
            fx = lambda t: curve(t)[0]
            fz = lambda t: curve(t)[1]
            x_fun = x_cls.from_function(fx, num=num, domain=curve.domain)
            z_fun = z_cls.from_function(fz, num=num, domain=curve.domain)
            return cls(x_fun, z_fun)

    @classmethod
    def create_line_segment(cls, point_a, point_b):
        r"""Create a straight line segment from one point to another.

        For simplicity, all curves will be defined on the domain `[0,pi]`.

        @param point_a
            Start point of the curve. Should be a array-like with 2 or 3
            elements. Two elements are interpreted as `x` and `z` coordinates.
            Three elements as `x`, `y`, `z`. Note that the `y` component will
            be ignored in this case.
        @param point_b
            End point of the curve. Same type as `point_a`.
        """
        if len(point_a) == 2:
            ax, az = point_a
            bx, bz = point_b
        else:
            ax, _, az = point_a
            bx, _, bz = point_b
        pi = np.pi
        fx = lambda t: t/pi * (bx - ax) + ax
        fz = lambda t: t/pi * (bz - az) + az
        x_fun = Cheby.from_function(fx, num=2, domain=(0, np.pi))
        z_fun = Cheby.from_function(fz, num=2, domain=(0, np.pi))
        return cls(x_fun, z_fun)

    @property
    def x_fun(self):
        r"""X-component function."""
        return self.s_fun
    @x_fun.setter
    def x_fun(self, func):
        r"""Set the x-component function."""
        self.s_fun = func

    @property
    def z_fun(self):
        r"""Z-component function."""
        return self.t_fun
    @z_fun.setter
    def z_fun(self, func):
        r"""Set the z-component function."""
        self.t_fun = func

    def _add_z_offset(self, z_offset):
        self.z_fun.add_constant(z_offset)

    def __call__(self, param, xyz=False):
        x_ev, z_ev = self._get_evaluators()
        x = x_ev(param)
        z = z_ev(param)
        return self._prepare_result(x, z, xyz)

    def _diff(self, param, diff):
        x_ev, z_ev = self._get_evaluators()
        dx = x_ev.diff(param, n=diff)
        dz = z_ev.diff(param, n=diff)
        return dx, dz

    def __setstate__(self, state):
        r"""Restore this object from the given unpickled state."""
        if 'x_fun' in state:
            # compatibility with previous version of this class
            state['s_fun'] = state.pop('x_fun')
            state['t_fun'] = state.pop('z_fun')
        super().__setstate__(state)
