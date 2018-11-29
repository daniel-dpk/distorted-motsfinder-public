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
from .basecurve import BaseCurve


__all__ = [
    "ParametricCurve",
]


class ParametricCurve(BaseCurve):
    r"""Parametric curve in the x-z-plane.

    This curve can take any numeric expression for its `x` and `z` component
    functions to represent a parametric curve, i.e. it is represented as
    \f[
        \gamma(\lambda) = (x(\lambda), 0, z(\lambda)),
        \qquad \lambda \in [0,\pi].
    \f]

    See the examples in the package documentation of curve.parametriccurve.
    """

    def __init__(self, x_fun, z_fun, freeze=True, name=''):
        r"""Create a parametric curve from two component functions.

        @param x_fun (exprs.numexpr.NumericExpression)
            Component function for the `x`-component of the curve. For curves
            starting and ending on the `z`-axis, a suitable choice is a
            function expanded into a pure sine series (see
            exprs.trig.SineSeries).
        @param z_fun (exprs.numexpr.NumericExpression)
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
        super(ParametricCurve, self).__init__(name=name)
        self.x_fun = x_fun
        self.z_fun = z_fun
        if freeze:
            self.freeze_evaluator()

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

    def copy(self):
        return type(self)(x_fun=self.x_fun.copy(), z_fun=self.z_fun.copy(),
                          name=self.name)

    def _create_evaluators(self):
        x_ev = self.x_fun.evaluator()
        z_ev = self.z_fun.evaluator()
        return x_ev, z_ev

    def resample(self, new_num):
        r"""Change the resolution of the two component functions."""
        self.x_fun.resample(new_num)
        self.z_fun.resample(new_num)
        self.force_evaluator_update()
        return self

    @property
    def num(self):
        r"""Resolution (number of DOFs) of the component functions.

        A `ValueError` is raised if the two component functions have different
        DOFs.
        """
        if self.x_fun.N != self.z_fun.N:
            raise ValueError("Component functions have different DOFs.")
        return self.x_fun.N

    def reparameterize(self, metric=None, rtol=1e-12, atol=1e-14, num=None,
                       blend=1.0):
        r"""Parameterize the curve by its arc length.

        After calling this method, all tangent vectors will have (roughly)
        equal lengths. The parameter will be scaled such that the domain
        remains ``[0,pi]``. An optional metric can be given to parameterize
        the curve by its length in curved space.

        @param metric
            Optional metric for computing tangent norms. If not given, the
            curve will have constant "speed" in coordinate space. If given, it
            will have constant "speed" in the curved space described by the
            metric.
        @param rtol
            Relative tolerance for solving the initial value problem (IVP),
            see below. Default is `1e-12`.
        @param atol
            Absolute tolerance for solving the IVP. Default is `1e-14`. Both
            `rtol` and `atol` are passed to `scipy.integrate.solve_ivp` and
            control the adaptive step size such that the local error estimates
            are less than ``atol + rtol * abs(s)``, where `s` is the
            reparameterization function (see below).
        @param num
            Optional new resolution for the `x` and `z` component functions.
            The default is to keep the current resolution.
        @param blend
            How much to consider metric versus flat coordinate space. Default
            is `1.0`, i.e. use provided metric.

        @b Notes

        Consider the reparameterized curve
        \f[
            \tilde\gamma(\lambda) := \gamma(s(\lambda)),
        \f]
        where \f$\gamma\f$ is the current curve.
        The reparameterization is performed by numerically integrating the
        differential equation we get by demanding that
        \f[
            \Vert\tilde\gamma'(\lambda)\Vert = \frac{\ell(\gamma)}{\pi},
        \f]
        where \f$\ell(\gamma)\f$ is the length of the curve. As a result, the
        domain remains ``[0,pi]`` as before. We hence get the following
        nonlinear ODE for \f$s\f$
        \f[
            s'(\lambda)=\frac{\ell(\gamma)/\pi}{\Vert\gamma'(s(\lambda))\Vert}
                =: f(s(\lambda)).
        \f]
        Starting with \f$s(0) = 0\f$, this equation is integrated numerically
        using a Runge-Kutta [1] integration method provided by
        `scipy.integrate.solve_ivp`.
        Afterwards, the `x` and `z` component functions are re-computed to
        match the new parameterization.

        In practice, we do not need to compute \f$\ell(\gamma)\f$ and instead
        integrate the above ODE with an arbitrary constant (`1.0`) until we
        have `s == pi`.

        @b References

        [1] Press, William H., et al. Numerical recipes 3rd edition: The art
            of scientific computing. Cambridge university press, 2007.
        """
        if metric is None:
            def f(_, s):
                return 1.0 / norm(self.tangent(s[0]))
        else:
            def f(_, s):
                g = metric.at(self(s, xyz=True))
                tangent = self.tangent(s[0], xyz=True)
                g_norm = g.norm(tangent)
                if blend != 1.0:
                    g_norm = blend * g_norm + (1-blend) * norm(tangent)
                return 1.0 / g_norm
        self._reparameterize(f, rtol=rtol, atol=atol, num=num)

    def reparameterize_by_curvature(self, metric=None, alpha=.1, beta=.5,
                                    rtol=1e-12, atol=1e-14, num=None,
                                    blend=1.0):
        r"""Experimental reparameterization focussing on parts with high curvature.

        This is similar to reparameterize(), but instead of requiring
        \f[
            \Vert\tilde\gamma'(\lambda)\Vert_g = \mathrm{const},
        \f]
        we implement an experimental formula which slows the parameterization
        at places of high curvature in coordinate space. Specifically,
        \f[
            \Vert\tilde\gamma'(\lambda)\Vert_g = c\ \exp\left(-\alpha\,\kappa^\beta\right),
        \f]
        where `alpha` and `beta` are constants and \f$\kappa(\lambda)\f$ is
        the curvature of the curve in coordinate space. The constant `c` is
        chosen such that the reparameterized curve keeps its domain `[0,pi]`.

        As in reparameterize(), the norm on the L.H.S. above is the flat-space
        norm by default, but may be chosen to be any norm by supplying a
        metric to this function.

        @param metric
            Optional metric for computing tangent norms (see above).
        @param alpha
            Parameter \f$\alpha\f$ for the above formula. Default is `0.1`.
        @param beta
            Parameter \f$\beta\f$ for the above formula. Default is `0.5`.
        @param rtol
            Relative tolerance for solving the initial value problem (IVP),
            see below. Default is `1e-12`.
        @param atol
            Absolute tolerance for solving the IVP. Default is `1e-14`. Both
            `rtol` and `atol` are passed to `scipy.integrate.solve_ivp` and
            control the adaptive step size such that the local error estimates
            are less than ``atol + rtol * abs(s)``, where `s` is the
            reparameterization function (see reparameterize()).
        @param num
            Optional new resolution for the `x` and `z` component functions.
            The default is to keep the current resolution.
        @param blend
            How much to consider metric versus flat coordinate space. Default
            is `1.0`, i.e. use provided metric.
        """
        if metric is None:
            def _norm(s):
                return norm(self.tangent(s))
        else:
            def _norm(s):
                g = metric.at(self(s, xyz=True))
                tangent = self.tangent(s, xyz=True)
                g_norm = g.norm(tangent)
                if blend != 1.0:
                    g_norm = blend * g_norm + (1-blend) * norm(tangent)
                return g_norm
        def f(_, s):
            n = _norm(s)
            k = self.curvature_in_coord_space(s)
            return 1.0 / (n * np.exp(alpha * k**beta))
        self._reparameterize(f, rtol=rtol, atol=atol, num=num)

    def _reparameterize(self, f, rtol=1e-12, atol=1e-14, num=None):
        r"""Perform the reparameterization using an arbitrary speed function."""
        if num is None:
            num = self.num
        with self.fix_evaluator():
            def pi_reached(_, s):
                return s[0] - np.pi
            pi_reached.terminal = True
            pi_reached.direction = 1
            # solve until pi_reached() returns 0
            s = solve_ivp(f, [0.0, 1e10], [0.0], rtol=rtol, atol=atol,
                          events=pi_reached, dense_output=True)
            bound = s.t_events[0][0] # the "event" where pi is reached
            series = type(self.z_fun)([], domain=[0.0, bound])
            pts = series.collocation_points(num=num, lobatto=False)
            x, z = self._get_evaluators()
            sol = s.sol
            x_vals = [x(sol(t)[0]) for t in pts]
            z_vals = [z(sol(t)[0]) for t in pts]
        # These two calls transform the function values back to coefficients.
        self.x_fun.set_coefficients(x_vals, physical_space=True,
                                    lobatto=False)
        self.z_fun.set_coefficients(z_vals, physical_space=True,
                                    lobatto=False)
        self.force_evaluator_update()

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
