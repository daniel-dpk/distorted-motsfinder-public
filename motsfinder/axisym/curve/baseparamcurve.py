r"""@package motsfinder.axisym.curve.baseparamcurve

Baseclass for parametric curves in the x-z-plane.

This curves are not aware of the geometry it is embedded in. They purely live
in coordinate space. This makes them suitable for use as reference curves for
refparamcurve.RefParamCurve objects.

See examples in .curve.parametriccurve and .curve.bipolarcurve.
"""

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import norm
from scipy.integrate import solve_ivp, quad

from ...exprs.trig import SineSeries, CosineSeries
from ...exprs.cheby import Cheby
from ...metric import FlatThreeMetric
from .basecurve import BaseCurve


__all__ = [
    "BaseParametricCurve",
]


# not using six.add_metaclass() since it breaks the super() calls
class BaseParametricCurve(BaseCurve, metaclass=ABCMeta):
    r"""Base class for parametric curves in the x-z-plane.

    This base class does the management of component functions
    (freezing/unfreezing/resampling/...). Interpreting the component functions
    and transforming to Cartesian coordinates is the responsibility of child
    classes. These have to implement:
        * from_curve()
        * #__call__()
        * _diff()
    """

    def __init__(self, s_fun, t_fun, freeze=True, name=''):
        super().__init__(name=name)
        self.s_fun = s_fun
        self.t_fun = t_fun
        if freeze:
            self.freeze_evaluator()

    @classmethod
    @abstractmethod
    def from_curve(cls, curve, num, **kw):
        r"""Static method to sample any curve to convert it to a parametric curve.

        @param curve (basecurve.BaseCurve)
            The curve to sample and represent.
        @param num (int)
            Resolution for sampling the curve. Higher values will lead to a
            more accurate match to the original `curve`.
        @param **kw
            Arguments specific to the subclass.
        """
        pass

    @abstractmethod
    def _add_z_offset(self, z_offset):
        r"""Move the curve in z-direction."""
        pass

    def add_z_offset(self, z_offset):
        r"""Move the curve in z-direction."""
        self._add_z_offset(z_offset)
        self.force_evaluator_update()

    def copy(self):
        return type(self)(s_fun=self.s_fun.copy(), t_fun=self.t_fun.copy(),
                          name=self.name)

    def collocation_points(self, lobatto=False, **kw):
        return self.t_fun.collocation_points(lobatto=lobatto, **kw)

    def _create_evaluators(self):
        s_ev = self.s_fun.evaluator()
        t_ev = self.t_fun.evaluator()
        return s_ev, t_ev

    def resample(self, new_num):
        r"""Change the resolution of the two component functions."""
        self.s_fun.resample(new_num)
        self.t_fun.resample(new_num)
        self.force_evaluator_update()
        return self

    @property
    def num(self):
        r"""Resolution (number of DOFs) of the component functions.

        A `ValueError` is raised if the two component functions have different
        DOFs.
        """
        if self.s_fun.N != self.t_fun.N:
            raise ValueError("Component functions have different DOFs.")
        return self.s_fun.N

    def reparameterize(self, strategy="arc_length", metric=None,
                       coord_space=False, **kw):
        r"""Reparameterize the curve using a given strategy.

        Supported strategies are:
            * `arc_length` - see reparameterize_by_arc_length()
            * `curv2` - see reparameterize_by_extr_curvature2()
            * `experimental` - see reparameterize_by_curvature()

        Default is `"arc_length"`.

        @param strategy
            Strategy for reparameterization (see above).
        @param metric
            Optional metric to use for computing curve velocities. Default is
            to use the flat (Euclidean) metric.
        @param coord_space
            Whether to reparameterize in internal coordinate space. Only
            relevant for non-trivial coordinate representations (i.e. makes no
            difference for .curve.ParametricCurve). Cannot be used when
            specifying a metric. Default is `False`.
        @param **kw
            Options specific to the respective strategy.
        """
        self._check_reparam_args(metric, coord_space)
        kw['metric'] = metric
        kw['coord_space'] = coord_space
        if strategy == "arc_length":
            return self.reparameterize_by_arc_length(**kw)
        if strategy == "curv2":
            return self.reparameterize_by_extr_curvature2(**kw)
        if strategy == "experimental":
            return self.reparameterize_by_curvature(**kw)
        raise ValueError("Unknown reparameterization strategy: %s"
                         % strategy)

    def reparameterize_by_arc_length(self, metric=None, coord_space=False,
                                     rtol=1e-12, atol=1e-14, num=None,
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
        @param coord_space
            Whether to reparameterize in internal coordinate space. Default is
            `False`.
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
        self._check_reparam_args(metric, coord_space)
        if metric is None or isinstance(metric, FlatThreeMetric):
            if coord_space:
                s_ev, t_ev = self._get_evaluators()
                def f(_, s):
                    param = s[0]
                    return 1.0 / np.sqrt(s_ev.diff(param)**2 + t_ev.diff(param)**2)
            else:
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

    def reparameterize_by_extr_curvature2(self, metric=None,
                                          coord_space=False, rtol=1e-12,
                                          atol=1e-14, num=None,
                                          res_factor=1.0, max_res=6000,
                                          smoothing=0.05, exponent=1/2.):
        r"""Reparameterize the curve using the square of its extrinsic curvature.

        The steps of this strategy are:
            * curve is parameterized by its arc length (and possibly upsampled
              in this step to keep as much of its shape as possible)
            * the square \f$k^{AB}k_{AB}\f$ of the extrinsic curvature of the
              curve is sampled as a function and represented by a cosine
              series
            * this series is "smoothed" by damping the coefficients of the
              series representation exponentially
            * the resulting (smoothed) function is taken as a "speed function"
              for the reparameterization, which also resamples the curve back
              to its original (or desired target) resolution `num`

        By starting with arc length parameterization, we ensure that repeated
        reparameterizations are well behaved (except for eventual precision
        loss by multiple changes of representation, of course).

        @param metric
            Optional metric to use for the arc length parameterization and
            also for computing the extrinsic curvature. Since the curves are
            represented and manipulated in coordinate space, however, it is
            recommended to not use the spacetime but the flat (default) metric
            here.
        @param coord_space
            Whether to reparameterize in internal coordinate space. Default is
            `False`.
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
        @param res_factor
            Factor by which to increase the resolution for the initial arc
            length parameterization. Default is `1.0`.
        @param max_res
            Resolution limit when applying `res_factor`. Default is `6000`.
        @param smoothing
            Smoothing factor to apply. A value of `0` skips any smoothing.
            Note that due to the finite sampling resolution, the resulting
            speed function may become negative (i.e. reparameterization will
            fail). Smoothing will typically help in these cases. Default is
            `0.05`.
        @param exponent
            Taking the square of the extrinsic curvature results in high
            density variations. These can be further reduced by taking a
            fractional power of \f$k^{AB}k_{AB}\f$ as the speed function.
            Default is `0.5`, i.e. taking the square root.
        """
        if coord_space:
            from .parametriccurve import ParametricCurve
            c = ParametricCurve(self.s_fun, self.t_fun)
            c.reparameterize_by_extr_curvature2(
                metric=metric, coord_space=False, rtol=rtol, atol=atol,
                num=num, res_factor=res_factor, max_res=max_res,
                smoothing=smoothing, exponent=exponent,
            )
            self.s_fun = c.x_fun
            self.t_fun = c.z_fun
            self.force_evaluator_update()
            return
        from .refparamcurve import RefParamCurve
        self._check_reparam_args(metric, coord_space)
        if metric is None:
            metric = FlatThreeMetric()
        if num is None:
            num = self.num
        init_num = min(max_res, int(res_factor*self.num))
        if init_num < self.num:
            init_num = self.num
        self.reparameterize(strategy='arc_length', metric=metric, num=init_num)
        c_flat = RefParamCurve.from_curve(self, metric=metric)
        with c_flat.fix_evaluator():
            f = CosineSeries.from_function(
                lambda x: c_flat.extrinsic_surface_curvature(x, square=True),
                num=c_flat.num, domain=(0.0, np.pi), lobatto=False,
            )
        f.a_n = [a*np.exp(-smoothing*n) for n, a in enumerate(f.a_n)]
        ev = f.evaluator()
        func = lambda _, x: 1/ev(x)**exponent
        self._reparameterize(func, rtol=rtol, atol=atol, num=num)

    def reparameterize_by_curvature(self, metric=None, coord_space=False,
                                    alpha=.1, beta=.5, rtol=1e-12, atol=1e-14,
                                    num=None, blend=1.0):
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
        @param coord_space
            Whether to reparameterize in internal coordinate space. Default is
            `False`.
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
        self._check_reparam_args(metric, coord_space)
        if metric is None or isinstance(metric, FlatThreeMetric):
            if coord_space:
                s_ev, t_ev = self._get_evaluators()
                def _norm(s):
                    param = s
                    return 1.0 / np.sqrt(s_ev.diff(param)**2 + t_ev.diff(param)**2)
            else:
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
            series = type(self.t_fun)([], domain=[0.0, bound])
            pts = series.collocation_points(num=num, lobatto=False)
            x, z = self._get_evaluators()
            sol = s.sol
            x_vals = [x(sol(t)[0]) for t in pts]
            z_vals = [z(sol(t)[0]) for t in pts]
        # These two calls transform the function values back to coefficients.
        self.s_fun.set_coefficients(x_vals, physical_space=True,
                                    lobatto=False)
        self.t_fun.set_coefficients(z_vals, physical_space=True,
                                    lobatto=False)
        self.force_evaluator_update()

    def _check_reparam_args(self, metric, coord_space):
        if metric and not isinstance(metric, FlatThreeMetric) and coord_space:
            raise ValueError(
                "Intrinsic coordinate space reparameterization not "
                "compatible with curved space reparameterization."
            )
