r"""@package motsfinder.axisym.curve.expcurve

Base class for curves that can calculate their expansion in the slice.


@b Examples

See the implemented subclasses starshapedcurve.StarShapedCurve and
refparamcurve.RefParamCurve for examples.
"""

from abc import abstractmethod
from contextlib import contextmanager
import math
import sys
import warnings

import numpy as np
from scipy import linalg
from scipy.integrate import quad, fixed_quad, IntegrationWarning
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar, bracket, brentq, root, brute
from scipy.special import sph_harm # pylint: disable=no-name-in-module
from mpmath import mp

from ...utils import insert_missing, isiterable, parallel_compute
from ...numutils import clip, IntegrationResult, IntegrationResults
from ...numutils import find_all_roots
from ...exprs.inverse import InverseExpression
from ...exprs.trig import CosineSeries
from ...metric import FlatThreeMetric, FourMetric
from ...ndsolve import ndsolve, NDSolver, CosineBasis, ChebyBasis
from ...ndsolve import DirichletCondition
from ..utils import detect_coeff_knee
from .basecurve import BaseCurve
from .parametriccurve import ParametricCurve
from .stabcalc import StabilityCalc, StabilitySpectrum


__all__ = [
    "TimeVectorData",
    "alm_matrix",
    "evaluate_al0",
]


# It is customary to denote indices of tensors without spaces, e.g.:
#   T_{ijk}  =>  T[i,j,k]
# We disable the respective pylint warning for this file.
# pylint: disable=bad-whitespace


class ExpansionCurve(BaseCurve):
    r"""Base class for curves that can calculate their expansion.

    This base class implements the functionality common to all concrete curve
    classes that are aware of the geometry they are living in. These take the
    Riemannian 3-metric and extrinsic curvature of the slice to compute
    quantities such as the expansion.

    Most of the actual calculation of these quantities is done in so called
    "calculation" objects that exist for a specific point on the curve and
    cache and reuse interim results.

    In addition to the abstract functions in basecurve.BaseCurve not
    implemented here, subclasses need to implement the following additional
    functions:
        * copy() to create an independent copy of the curve
        * _create_calc_obj() to create the cache/calculator object
    """

    def __init__(self, h, metric, name=''):
        r"""Base class constructor taking a horizon function and metric.

        @param h (exprs.numexpr.NumericExpression)
            The "horizon function" defining this curve. How this function is
            interpreted is up to the subclass and its calculator object.
        @param metric
            The Riemannian 3-metric defining the geometry of the surrounding
            space.
        @param name
            Name of this curve. This may be used when printing information
            about this curve or as label in plots.
        """
        super(ExpansionCurve, self).__init__(name=name)
        self.h = h
        self.metric = metric
        self.extr_curvature = metric.get_curv() if metric else None
        self._calc = None

    def __getstate__(self):
        with self.suspend_calc_obj(), self.suspend_curv():
            return super(ExpansionCurve, self).__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.metric:
            self.extr_curvature = self.metric.get_curv()

    @abstractmethod
    def h_diff(self, param):
        r"""Compute derivative of this curve \wrt horizon function `h`."""
        pass

    def collocation_points(self, lobatto=False, **kw):
        return self.h.collocation_points(lobatto=lobatto, **kw)

    def arc_length(self, a=0, b=np.pi, atol=1e-12, rtol=1e-12,
                   full_output=False):
        return self.arc_length_using_metric(metric=self.metric, a=a, b=b,
                                            atol=atol, rtol=rtol,
                                            full_output=full_output)

    def proper_length_map(self, num=None, **kw):
        metric = kw.pop('metric', self.metric)
        return super().proper_length_map(num=num, metric=metric, **kw)

    def z_distance(self, other_curve=None, atol=1e-12, rtol=1e-12, limit=100,
                   allow_intersection=False, full_output=False):
        return self.z_distance_using_metric(
            metric=self.metric, other_curve=other_curve, atol=atol, rtol=rtol,
            limit=limit, allow_intersection=allow_intersection,
            full_output=full_output
        )

    def inner_z_distance(self, other_curve, where='top', full_output=False,
                         **kw):
        r"""Compute the z-distance of two points of this and another curve.

        In contrast to z_distance(), this method does *not* compute how close
        two surfaces approach each other. Instead, it computes one possible
        measure for how close the two surfaces are from being identical, i.e.
        it computes the distance of two corresponding points.

        In this function, we take either the top or bottom points of both
        surfaces on the z-axis and compute their distance. The distance will
        have negative sign if this curve is in the *interior* of the
        `other_curve`.

        @param other_curve
            Curve to which to compute the distance.
        @param where
            One of ``'top', 'bottom'``. Default is ``'top'``. Where to take
            the points on both surfaces.
        @param metric
            Which metric to use for integrating along the connecting line of
            the two points. By default, takes the current metric stored in
            this curve. Explicitely specify `None` to get the coordinate
            distance.
        @param full_output
            If `True`, return the computed result and an estimation of the
            error. Otherwise (default), just return the result.
        @param **kw
            Further keyword arguments are passed to arc_length_using_metric().
        """
        metric = kw.pop('metric', self.metric)
        if where not in ('top', 'bottom'):
            raise ValueError('Unknown location: %s' % where)
        t = 0 if where == 'top' else np.pi
        line = ParametricCurve.create_line_segment(self(t), other_curve(t))
        dist, dist_err = line.arc_length_using_metric(metric=metric,
                                                      full_output=True, **kw)
        upwards = line(0)[1] < line(np.pi)[1]
        if (where == 'top' and upwards) or (where == 'bottom' and not upwards):
            dist *= -1
        if full_output:
            return dist, dist_err
        return dist

    def inner_x_distance(self, other_curve, where='zero', **kw):
        r"""Compute the x-distance of two points of this and another curve.

        Similar to inner_z_distance(), this method does *not* compute how
        close two surfaces approach each other (which would return zero for
        intersecting surfaces). Instead, it computes one possible measure for
        how close the two surfaces are from being identical, i.e. it computes
        the distance of two corresponding points.

        In this function, we measure the distance on a straight coordinate
        line in x-direction connecting either two points on the x-axis itself
        (i.e. for ``x==0``) or on a line in x-direction starting at the point
        of largest coordinate distance to the z-axis (i.e. for largest x
        value).

        @param other_curve
            Curve to which to compute the distance.
        @param where
            One of ``'zero', 'max'``. Default is ``'zero'``. Where to take the
            points on both surfaces. Here, ``'max'`` means to take the point
            of largest distance to the z-axis of this curve. Hence, the result
            will not be symmetric w.r.t. switching the two curves.
        @param metric
            Which metric to use for integrating along the connecting line of
            the two points. By default, takes the current metric stored in
            this curve. Explicitely specify `None` to get the coordinate
            distance.
        @param **kw
            Further keyword arguments are passed to arc_length_using_metric().
        """
        metric = kw.pop('metric', self.metric)
        if where not in ('zero', 'max'):
            raise ValueError('Unknown location: %s' % where)
        if where == 'zero':
            t1 = self.find_line_intersection(point=[0, 0], vector=[1, 0])
        else:
            t1 = self.find_max_x()
        t2 = other_curve.find_line_intersection(point=self(t1), vector=[1, 0])
        line = ParametricCurve.create_line_segment(self(t1), other_curve(t2))
        return line.arc_length_using_metric(metric=metric, **kw)

    def get_calc_obj(self, param):
        r"""Create or reuse a calculator object for a given parameter value.

        In case the current cache/calculator already belongs to the given
        parameter value, it is reused and returned instead of a new one being
        created each time.

        @param param (float)
            Parameter of the curve for which to construct the
            cache/calculator object.
        """
        if self._calc is None or self._calc.param != param:
            self._calc = self._create_calc_obj(param)
        return self._calc

    def horizon_function_changed(self):
        r"""Notify this curve that the underlying horizon function has changed.

        Since we cache the calculation object for efficiency, changes to the
        horizon function will not affect results if evaluating the expansion
        at the exact same point. This method invalidates any cached evaluators
        and calculation objects to produce newly computed results in the next
        computation call.
        """
        self._calc = None
        self.force_evaluator_update()

    def _create_evaluators(self):
        return self.h.evaluator()

    @property
    def num(self):
        r"""Resolution (number of DOFs) of the horizon function."""
        return self.h.N

    def resample(self, new_num):
        r"""Change the number of DOFs of the horizon function.

        Other algorithms may use this number to implement modification of the
        horizon function for a search for e.g. MOTSs.

        This function returns the curve object itself to allow for chaining.
        """
        self.h.resample(new_num)
        self.force_evaluator_update()
        return self

    @abstractmethod
    def _create_calc_obj(self, param):
        r"""Create a cache/calculator for a specific parameter value.

        This is only called when a new calculator object needs to be
        constructed (i.e. this is a calculator factory). Subclasses should
        create the respective calculator type they need.
        """
        pass

    @contextmanager
    def suspend_calc_obj(self):
        r"""Context manager to temporarily ignore the current calculator.

        When leaving the with-scope, the current calculator is restored. This
        may be useful during evaluation of expansions for perturbed curves in
        finite difference approximations of functional derivatives of the
        expansion.
        """
        calc_orig = self._calc
        try:
            self._calc = None
            yield
        finally:
            self._calc = calc_orig

    @contextmanager
    def suspend_curv(self):
        r"""Context manager to temporarily ignore the extrinsic curvature."""
        curv = self.extr_curvature
        try:
            self.extr_curvature = None
            yield
        finally:
            self.extr_curvature = curv

    @contextmanager
    def temp_metric(self, metric=None):
        r"""Context manager to temporarily replace the used metric.

        When leaving the with-scope, the previous metric is restored.
        """
        metric_orig = self.metric
        try:
            if metric_orig is metric:
                yield
            else:
                self.metric = metric or FlatThreeMetric()
                with self.suspend_calc_obj():
                    yield
        finally:
            self.metric = metric_orig

    def expansion(self, param, hdiff=None, ingoing=False):
        r"""Compute the expansion at one parameter value.

        This function can also compute derivatives of the expansion w.r.t. the
        horizon function `h` or one of its derivatives.

        @param param
            Parameter (i.e. \f$\lambda\f$ value) specifying the point on the
            curve at which to compute the expansion of the surface.
        @param hdiff
            If given, compute the functional derivative of the expansion
            w.r.t. a derivative of the horizon function. For example, if
            ``hdiff==0``, compute \f$\partial_h\Theta\f$ and for ``hdiff==1``,
            compute \f$\partial_{h'}\Theta\f$. If `None` (default), compute
            the expansion.
        @param ingoing
            By default, the outgoing null geodesics' expansion is returned. If
            ``ingoing==True``, the expansion of the ingoing null geodesics is
            computed instead. Functional derivatives of the ingoing expansion
            are not currently implemented.
        """
        calc = self.get_calc_obj(param)
        if hdiff is None:
            return calc.expansion(ingoing=ingoing)
        if ingoing:
            raise NotImplementedError
        return calc.diff(hdiff=hdiff)

    def expansions(self, params, hdiff=None, ingoing=False):
        r"""As expansion(), but on a whole set of parameter values."""
        with self.fix_evaluator():
            return [self.expansion(p, hdiff=hdiff, ingoing=ingoing)
                    for p in params]

    def average_expansion(self, ingoing=False, area=None, full_output=False,
                          **kw):
        r"""Compute the average expansion across the surface.

        This computes the average expansion
        \f[
            \overline\Theta = \frac{1}{A} \int \Theta \sqrt{q}\ d^2x.
        \f]

        @param ingoing
            Whether to average over the ingoing (`True`) or outgoing (`False`)
            expansion. Default is `False`.
        @param area
            Optionally re-use an already computed area value.
        @param full_output
            If `True`, return the average expansion and the estimated error.
            Otherwise return just the average expansion. Default is `False`.
        @param **kw
            Additional keyword arguments are passed to `scipy.integrate.quad`.
        """
        with self.fix_evaluator():
            if area is None:
                area = self.area()
            det_q = self.get_det_q_func()
            def integrand(la):
                Th = self.expansion(la, ingoing=ingoing)
                return Th * np.sqrt(det_q(la))
            res, err = quad(integrand, a=0, b=np.pi, **kw)
            ex = 2*np.pi/area * res
            err = 2*np.pi/area * err
            if full_output:
                return ex, err
            return ex

    def collect_close_in_time(self, curves, steps=3):
        r"""Return a list of curves close (in time) to this one.

        Given a list of `curves`, which may or may not contain the current one
        (`self`), we collect at most `steps` curves before and `steps` curves
        after this one. The returned list will contain the current curve.
        """
        if steps is None:
            return curves
        if not any(c is self for c in curves):
            tmp = [self]
            tmp.extend(curves)
            curves = tmp
        curves = sorted(curves, key=lambda c: c.metric.time)
        idx = next((i for i, c in enumerate(curves) if self is c), None)
        return curves[max(0, idx-steps):idx+steps+1]

    def time_interpolators(self, curves, points=None, proper=True, steps=3,
                           eps=0.0):
        r"""Construct a set of functions interpolating along a MOTT.

        @return A list of callables with signature ``f(x, nu=0)``, where `x`
            is the time `t` at which to evaluate and `nu` the time derivative.
            There will be one such function per point in `points` (see below).
            Each function will return a 3-element array with the coordinates
            of the intersection of the tube with the ``t=const`` slice. The
            coordinates will correspond to the same point on the respective
            curve for each `t`.

        @param curves
            List of curves that build up the tube. The current object (`self`)
            may or may not be in that list. Each curve needs to have a metric
            with a `time` attribute. They do not need to be sorted.
        @param points
            List of points each being a number in ``[0, pi]``. Alternatively,
            this can be an integer specifying how many of such points should
            be created. They will be equally spaced and exclude the boundary.
            Default is to take the current curve's resolution ``self.num``.
        @param proper
            Whether to take the points at the pi-normalized proper length or
            at the curves current parameterization. It is strongly recommended
            to take the proper length, which is also the default. Otherwise,
            the worldlines are most likely not differentiable (i.e. the
            interpolation may be very poor).
        @param steps
            How many curves before and after this current curve to consider
            for interpolation. Default is `3`. Set to `None` to use all
            curves as-is. The current curve then has to be in `curves`.
        @param eps
            Parameterized deviation from the "equal-parameter" rule. If
            nonzero, the surrounding curves are not evaluated at the same
            (possibly proper-length) parameter `u` as this curve but instead
            at ``u + eps*(t-t0)``, where `t0` is the time of the slice this
            curve lives in. Default is `0.0`.
        """
        if steps is not None:
            curves = self.collect_close_in_time(curves, steps=steps)
        if len(curves) < 2:
            raise ValueError("Cannot interpolate with only one curve.")
        if points is None:
            points = self.num
        if isiterable(points):
            pts = np.asarray(points)
        else:
            pts = CosineSeries.create_collocation_points(
                num=points, lobatto=False,
            )
        times = np.array([c.metric.time for c in curves])
        data = np.zeros((len(curves), len(pts), 3))
        t0 = self.metric.time
        for i, (c, t) in enumerate(zip(curves, times)):
            with c.fix_evaluator():
                if proper:
                    inv_map = c.cached_length_maps()[1]
                else:
                    inv_map = lambda x: x
                data[i] = [
                    c(inv_map(clip(x + eps*(t-t0), 0.0, np.pi)), xyz=True)
                    for x in pts
                ]
        domain = curves[0].metric.time, curves[-1].metric.time
        interps = []
        for i, pt in enumerate(pts):
            interp = CubicSpline(times, data[:,i])
            interp.domain = domain
            interp.param = pt
            interps.append(interp)
        return interps

    def time_shift_params(self, params, tevs, next_curve):
        r"""Transport a list of parameters in time.

        This takes a set of parameter values (i.e. points along the MOTS) and
        computes the parameters of points on a future (or past) MOTS obtained
        by transporting the points along the evolution vector field on the
        dynamical horizon.

        @param params
            Sequence of parameter values specifying the points on the MOTS to
            transport.
        @param tevs
            List of time evolution vectors along the current MOTS. These will
            be interpolated to the given `params` to obtain the direction into
            which to transport the points.
        @param next_curve
            The MOTS to transport the points to.
        """
        t0 = self.metric.time
        lm = self.cached_length_maps()[0]
        proper_params = [lm(param) for param in params]
        pts = [lm(v.param) for v in tevs]
        epsilons = [v.eps for v in tevs]
        eps_interp = CubicSpline(pts, epsilons)
        epsilons = eps_interp(proper_params)
        _next_im = next_curve.cached_length_maps()[1]
        def next_im(proper_param):
            if max(abs(proper_param), abs(np.pi-proper_param)) < 1e-6:
                return proper_param
            return _next_im(proper_param)
        t = next_curve.metric.time
        next_params = [
            next_im(clip(proper_param + eps*(t-t0), 0.0, np.pi))
            for proper_param, eps in zip(proper_params, epsilons)
        ]
        return next_params

    def signature_quantities(self, pts, curves, proper=True, steps=3):
        r"""Compute the MOTS tube signature along points of this MOTS.

        This method takes a list of points and evaluates the signature of the
        MOTT, i.e. the hypersurface consisting of stacking the MOTSs at
        different times to build a world-tube. This surface may be space-like
        (the usual case for a dynamical horizon) or null (in the static case,
        e.g. Schwarzschild), timelike, or even of mixed signature.

        The signature is computed using a list of curves representing this
        MOTS at different time steps. We collect a given number of such curves
        before and after the time slice of this current MOTS and interpolate
        along the time direction in order to evaluate the coordinate
        derivative vector.

        @return A list of SignatureQuantities objects, one for each point in
            `pts`.

        @param pts
            The points along the MOTS at which to evaluate the signature.
            Should be a sequence of values in the open interval ``(0, pi)``.
            Interpretation of these values is subject to `proper`.
        @param curves
            Additional curves for different times in order to interpolate and
            compute the time derivative. The current MOTS (i.e. this object)
            may or may not be part of this list and their order does not
            matter. This list needs to have at least one other MOTS that is
            not the current object.
        @param proper
            Whether to interpret the values in `pts` as fraction of the proper
            length (scaled by `pi`) or as simple curve parameter. Default is
            `True`. If the parameterization of the other curves in `curves`
            does not vary slowly in time, taking the 'simple' parameter values
            produces non-differentiable world lines of constant curve
            parameter and hence very inaccurate signature evaluations. Since
            smooth parameterization can not be guaranteed in most cases, it is
            highly recommended to take the proper parameterizations. Note that
            in both cases, the values of `pts` should lie in ``(0, pi)``.
            **NOTE:** All curves (the current one and those in `curves`)
            should have ``'length_maps'`` data in `user_data`. This can be
            obtained using the methods in ..trackmots.props.
        @param steps
            How many curves in the future and past of this curve to consider
            from `curves`. Default is `3`.

        @b Notes

        Let \f$\mathcal{H}\f$ be the MOTT foliated by the MOTSs given in the
        `curves` parameter and \f$N\f$ a normal of \f$\mathcal{H}\f$. Then, at
        a point \f$p \in \mathcal{H}\f$ we have
            * \f$\mathcal{H}\f$ spacelike at \f$p\f$ iff \f$g(N_p, N_p) < 0\f$,
            * \f$\mathcal{H}\f$ timelike at \f$p\f$ iff \f$g(N_p, N_p) > 0\f$,
            * \f$\mathcal{H}\f$ null at \f$p\f$ iff \f$g(N_p, N_p) = 0\f$.

        Let \f$k^\mu\f$ and \f$\ell^\mu\f$ span the (2D) space of normals to
        the current MOTS in spacetime \f$M\f$, with the scaling fixed such
        that \f$k^\mu \ell_\mu = -2\f$. Then, \f$N\f$ will be of the form
        \f$N = c_1 \ell + c_2 k\f$ and hence \f$g(N,N) = -4c_1c_2\f$.
        This follows from the fact that a normal to the MOTT is also a normal
        to the MOTS. The condition for \f$N\f$ to be normal to
        \f$\mathcal{H}\f$ now becomes
        \f[
            g(N, \partial_\tau) = 0
            \qquad \Leftrightarrow \qquad
            \frac{c_1}{c_2} = -\frac{k\cdot\partial_\tau}{\ell\cdot\partial_\tau}\,,
        \f]
        where \f$\tau\f$ is a coordinate along \f$\mathcal{H}\f$ and
        \f$\partial_\tau\f$ a tangent to \f$\mathcal{H}\f$ (which is not
        necessarily normal to the MOTS). We choose \f$\tau(p) = t\f$ iff
        \f$p \in \mathcal{S}_t\f$, where \f$\mathcal{S}_t\f$ is the MOTS in
        the slice with coordinate time \f$t\f$. We get the coordinate line of
        \f$\tau\f$ by fixing the other coordinates on the MOTSs. As explained
        above, we do not fix the parameter \f$\lambda\f$ of our MOTS
        representation. Instead, we use a fraction of the proper length as
        coordinate, which ensures a smooth coordinate line when varying just
        \f$\tau\f$.

        The sign of \f$g(N,N) = -4c_1c_2\f$ is the same as that of
        \f$-\frac{c_1}{c_2}\f$, which is the same as that of
        \f$f_{sig}:=(k\cdot\partial_\tau)(\ell\cdot\partial_\tau)\f$.
        We then have
            * \f$\mathcal{H}\f$ spacelike iff \f$f_{sig} < 0\f$,
            * \f$\mathcal{H}\f$ timelike iff \f$f_{sig} > 0\f$,
            * \f$\mathcal{H}\f$ null iff \f$f_{sig} = 0\f$.
        """
        interps = self.time_interpolators(curves=curves, points=pts,
                                          proper=proper, steps=steps)
        with self.fix_evaluator():
            if proper:
                inv_map = self.cached_length_maps()[1]
            else:
                inv_map = lambda x: x
            metric4 = FourMetric(self.metric)
            results = []
            for interp in interps:
                pt = inv_map(interp.param)
                stab_calc = StabilityCalc(curve=self, param=pt, metric4=metric4)
                xdot = interp(self.metric.time, nu=1)
                tau = np.ones((4,))
                tau[1:] = xdot
                results.append(SignatureQuantities(
                    pt, stab_calc.g4, stab_calc.l, stab_calc.k, tau,
                ))
            return results

    def time_evolution_vector(self, pts, curves, steps=3, search_step=1e-3,
                              search_dist_increase=10, max_increases=4,
                              Ns=400, tol=1e-12, full_output=False,
                              verbose=False):
        r"""Compute the time evolution 4D vector.

        Stacking the MOTSs at different times produces a 3-dimensional
        manifold, the marginally outer trapped tube (MOTT) foliated by the
        MOTSs. At any given point of a MOTS, we can find a normal in spacetime
        pointing tangent to the MOTT. This will be a linear combination of the
        ingoing and outgoing null normals \f$k\f$ and \f$\ell\f$, since these
        span the 2D space of normals to the MOTS.

        **NOTE 1:** This method is currently limited to MOTSs with vanishing
        spin (i.e. where the ingoing and outgoing null normals to the MOTS
        have no `y` component).

        **NOTE 2:** All curves (the current one and those in `curves`) should
        have ``'length_maps'`` data in `user_data`. This can be obtained using
        the methods in ..trackmots.props.

        @return For each value in `pts`, a 4-vector `V` pointing along the
            MOTT. If ``full_output==True``, return a TimeVectorData object for
            each such value.

        @param pts
            The points along the MOTS at which to evaluate the signature.
            Should be a sequence of values in the open interval ``(0, pi)``.
            Note that these are not the intrinsic parameters (called `lambda`
            or `param` in this code) but the `pi`-normalized proper length
            parameters. For example, a value ``pi/2`` is the point splitting
            the curve into two parts of equal proper length.
        @param curves
            Additional curves for different times in order to interpolate and
            compute the time derivative. The current MOTS (i.e. this object)
            may or may not be part of this list and their order does not
            matter. This list needs to have at least one other MOTS that is
            not the current object.
        @param steps
            How many curves in the future and past of this curve to consider
            from `curves`. Default is `3`.
        @param search_step
            Initial step for optimizing the tilting parameter (see below).
            Default is `1e-3`.
        @param search_dist_increase
            Factor by which the searched region for the tilting parameter is
            increased in each trial. Default is `10`.
        @param max_increases
            Maximum number of times the search region for the tilting
            parameter is increased by `search_dist_increase`. Default is `4`.
        @param Ns
            Number of sample points for the `scipy.optimize.brute()` searches
            done after the search region is increased.
        @param tol
            Tolerance for the error. If the error drops below `tol`, we define
            the current tilting parameter as converged and return the
            resulting time evolution vector.
        @param full_output
            Wether to output a list of TimeVectorData object (if `True`) or a
            list of 4D numpy vectors (default).

        @b Notes

        The time vector is found using a numerical minimization strategy as
        follows. For each MOTS we take one point at the same relative proper
        length (say `25%` of the curve's total length) and interpolate a curve
        through these points. The curve parameter will be the simulation time.
        The tangent to this curve is a candidate for the 4D time evolution
        vector of the MOTT. However, in general it will be slightly off the
        plane of normals from the MOTS (it points along the MOTT, but not
        orthogonal to the MOTS). We project this vector onto the plane spanned
        by \f$k\f$ and \f$\ell\f$ and compute the sum of the squares of
        coordinate changes this induces. This is the error we wish to
        minimize. We then tilt the vector by building a new time interpolation
        curve. The tilting is achieved by not taking the same relative proper
        length parameter for each MOTS. Instead, we move away from this
        parameter linearly in coordinate time. The "speed" for this moving is
        then numerically optimized to minimize the error. We call this speed
        the tilting parameter.

        The optimization is first performed using Brent's method, starting
        from the same tilting parameter that was successful for the previous
        point. Near the poles, the tilting is guaranteed to be small. Having a
        large number of points in `pts` hence leads to faster convergence per
        point, given the points are ordered and somewhat evenly distributed.
        If the search converges in a local minimum (given tolerance is not
        reached), we do a brute search on a grid of points in a larger region
        around the previously successful tilting parameter. The best match is
        then used as starting point for another search using Brent's method.
        """
        curves = self.collect_close_in_time(curves, steps=steps)
        with self.fix_evaluator():
            inv_map = self.cached_length_maps()[1]
            metric4 = FourMetric(self.metric)
            def approx_time_vector(eps, u, k, l, g4, param):
                interp, = self.time_interpolators(
                    curves=curves, points=[u], proper=True, eps=eps,
                    steps=None # `curves` are already prepared
                )
                xdot = interp(self.metric.time, nu=1)
                tau = np.ones((4,))
                tau[1:] = xdot
                return TimeVectorData(l=l, k=k, tau=tau, eps=eps, g4=g4,
                                      param=param)
            prev_eps = 0
            results = []
            sqrt2 = np.sqrt(2)
            for u in pts:
                stab_calc = StabilityCalc(curve=self, param=inv_map(u),
                                          metric4=metric4)
                k = stab_calc.k / sqrt2 # different normalization here: l*k = -1
                l = stab_calc.l / sqrt2
                def _approx_time_vector(eps):
                    return approx_time_vector(
                        eps, u, k, l, stab_calc.g4,
                        stab_calc.calc.param
                    )
                def _error(eps):
                    return _approx_time_vector(eps).error
                # Use the `eps` of the previous point as starting point for
                # the next search.
                x0 = prev_eps
                for increases in range(max_increases+1):
                    if increases:
                        delta = search_step * search_dist_increase**increases
                        if verbose:
                            print("Searching for eps in %s +- %s" % (prev_eps, delta))
                        x0 = brute(_error, [[prev_eps-delta, prev_eps+delta]],
                                   Ns=Ns, finish=None)
                    res = minimize_scalar(
                        _error, bracket=(x0, x0+search_step)
                    )
                    time_vec = _approx_time_vector(res.x)
                    if time_vec.error <= tol:
                        break
                    # We probably ended up in a local minimum. Try looking
                    # around at larger distances in next iteration.
                if time_vec.error > tol:
                    # We've not converged. If `u` is very close to the domain
                    # boundary 0 or pi, then the search sometimes doesn't see
                    # `eps=0.0` as possibility (i.e. the minimum is too
                    # narrow). Try looking around `eps=0.0` now.
                    res = minimize_scalar(
                        _error, bracket=(0.0, max(abs(prev_eps), 1e-6))
                    )
                    time_vec = _approx_time_vector(res.x)
                if time_vec.error > tol:
                    if abs(l[2]) > tol or abs(k[2]) > tol:
                        raise TimeEvolutionVectorSearchError(
                            "Horizon appears to spin. Finding time evolution "
                            "vector not implemented for this case."
                        )
                    raise TimeEvolutionVectorSearchError(
                        "Could not find a time evolution vector normal to S."
                    )
                results.append(time_vec)
                prev_eps = time_vec.eps
            if full_output:
                return results
            return [res.V for res in results]

    def integrate_tev_divergence(self, curves, tev_args=None, n="auto",
                                 unity_b=False, full_output=False):
        r"""Integrate the divergence of the time evolution vector (TEV).

        This computes the integral of the divergence of the time evolution
        vector (TEV) over the MOTS. We compute the integrand via
        \f[
            q^{\mu\nu} \nabla_\mu V_\nu
                = q^{\mu\nu} \nabla_\mu (b \ell_\nu + c k_\nu)
                = c q^{\mu\nu} \nabla_\mu k_\nu
                = c \Theta_{(k)} \,.
        \f]

        Note that the ingoing expansion is computed with the convention
        \f$\ell^\mu k_\mu = -2\f$. To convert the result to the other often
        used convention \f$\ell^\mu k_\mu = -1\f$, simply divide the
        integration result by \f$\sqrt{2}\f$.

        @param curves
            List of curves that build up the tube. The current object (`self`)
            may or may not be in that list. Each curve needs to have a metric
            with a `time` attribute. They do not need to be sorted.
        @param tev_args
            Optional further arguments for the time_evolution_vector() call.
        @param n
            Order of the fixed quadrature integration (roughly equal to the
            number of points at which the integrand is evaluated). The default
            ``"auto"`` will use twice the current curve's resolution, but at
            least 30 points.
        @param unity_b
            If `True`, expand the TEV as \f$V = \ell + c k\f$, effectively
            transforming `c` to `c/b`. If `False` (default), use the usual
            scaling (with unity time component of the TEV).
        @param full_output
            If `True`, return the computed value of the integral and a list of
            `n` TimeVectorData objects created while evaluating the integrand.
            Default is `False`, i.e. to only return the value.
        """
        if n == "auto":
            n = max(30, 2 * self.num)
        tev_args = (tev_args or {}).copy()
        tev_args['full_output'] = True
        with self.fix_evaluator():
            if unity_b:
                value, vectors = self._integrate_tev_divergence_unity_b(
                    curves, tev_args=tev_args, n=n
                )
            else:
                value, vectors = self._integrate_tev_divergence_unity_time(
                    curves, tev_args=tev_args, n=n
                )
        if full_output:
            return value, vectors
        return value

    def _integrate_tev_divergence_unity_time(self, curves, tev_args, n):
        r"""Implement integrate_tev_divergence() for unity time component."""
        area_element = self.get_area_integrand()
        length_map = self.cached_length_maps()[0]
        params = []
        vectors = []
        def integrand(xs):
            proper_params = [length_map(param) for param in xs]
            tevs = self.time_evolution_vector(
                pts=proper_params, curves=curves, **tev_args
            )
            c = np.asarray([tev.c_B for tev in tevs])
            # This is a factor of sqrt(2) too large if our scaling is l*k = -1
            ingoing_exp = np.asarray(self.expansions(xs, ingoing=True))
            measure = np.asarray([area_element(x) for x in xs])
            values = c * ingoing_exp * measure
            # Store data we'd like to return.
            params[:] = xs
            vectors[:] = tevs
            return values
        value = 2*np.pi * _fixed_quad(
            integrand, a=0.0, b=np.pi, n=n
        )
        return value, vectors

    def _integrate_tev_divergence_unity_b(self, curves, tev_args, n):
        r"""Implement integrate_tev_divergence() for unity `b` coefficient."""
        # `n` is adapted to the complexity of the shape (i.e. larger for more
        # distorted shapes). This means we can compute `chi` at `n` points and
        # determine all sign-changes of `cos(chi)` just by looking at these
        # points.
        length_map = self.cached_length_maps()[0]
        params = _get_fixed_quad_abscissas(a=0.0, b=np.pi, n=n)
        area_element = self.get_area_integrand()
        def _chis(xs, full_output=False):
            # We convert the points to proper length parameters since that's
            # what time_evolution_vector() expects.
            proper_params = [length_map(x) for x in xs]
            vectors = self.time_evolution_vector(
                pts=proper_params, curves=curves, **tev_args
            )
            chis = np.asarray([V.chi for V in vectors])
            return (chis, vectors) if full_output else chis
        def _c_from_chis(chis):
            return np.asarray([TimeVectorData.c_from_chi(chi) for chi in chis])
        def _c(xs):
            return _c_from_chis(_chis(xs))
        def _theta_k(xs):
            return np.asarray(self.expansions(xs, ingoing=True))
        def _measure(xs):
            return np.asarray([area_element(x) for x in xs])
        chis, vectors = _chis(params, full_output=True)
        interval_borders = find_all_roots(
            xs=params, ys=np.cos(chis),
            func=lambda param: np.cos(_chis([param])[0])
        )
        interval_borders = [0.0] + interval_borders + [np.pi]
        intervals = [(interval_borders[i], interval_borders[i+1])
                     for i in range(len(interval_borders)-1)]
        # Optimization: In case of only one root, use the already computed chi
        #               values.
        if len(intervals) == 1:
            total = 2*np.pi * _fixed_quad(
                lambda xs: _c_from_chis(chis) * _theta_k(xs) * _measure(xs),
                a=0.0, b=np.pi, n=n,
            )
        else:
            # We do have poles to deal with. First integrate between the
            # poles, keeping well away from them.
            def integrand(xs):
                return _c(xs) * _theta_k(xs) * _measure(xs)
            total = 0.0
            pole_dists = [0.0]
            for i, interval in enumerate(intervals):
                a, b = interval
                w = b - a
                # Keep away from the poles. We'll integrate over these in a
                # separate step.
                if i+1 < len(intervals):
                    post_interval = intervals[i+1]
                    post_w = post_interval[1] - post_interval[0]
                    post_dist = min(w/2, post_w/2)
                    pole_dists.append(post_dist)
                else:
                    pole_dists.append(0.0)
                a = a + pole_dists[i]
                b = b - pole_dists[i+1]
                val = 2*np.pi * _fixed_quad(
                    integrand, a=a, b=b, n=n, full_domain=(0.0, np.pi),
                )
                total += val
            # Now, integrate over the poles, symmetrically away, by adding the
            # values to the left and right of it (these cancel).
            for i in range(1, len(intervals)):
                pole = intervals[i][0]
                val = 2*np.pi * _fixed_quad(
                    lambda xs: integrand(pole-xs) + integrand(pole+xs),
                    a=0.0, b=pole_dists[i], n=n,
                    full_domain=(0.0, np.pi),
                )
                total += val
        return total, vectors

    def dt_normals(self, pts, curves, steps=3, use_tev=True, tevs=None,
                   unload=True, full_output=False):
        r"""Compute the time derivative of the MOTS normal within the slice.

        The outward pointing normal to the MOTS within the spatial slice is
        traced along the MOTT section (defined by the `curves`) and its
        derivative w.r.t. coordinate time is computed.

        @param pts
            List of points (in terms of pi-normalized proper length) to
            compute the derivative of the normal at. It is much more efficient
            to compute multiple normals at once instead of calling this method
            multiple times. This holds especially when the metric is given as
            data on a grid loaded from a file and not available in memory.
        @param curves
            List of curves that build up the tube. The current object (`self`)
            may or may not be in that list. Each curve needs to have a metric
            with a `time` attribute. They do not need to be sorted.
        @param steps
            How many curves in the future and past of this curve to consider
            from `curves`. Default is `3`.
        @param use_tev
            Which `tau` coordinate to use (see below). If `True`, use the
            integral curve of the time evolution vector (i.e. with tangents
            orthogonal to the MOTSs). This is much more expensive. If `False`,
            just use a curve constructed from taking the same fraction of the
            proper length of the curves on each curve. The two variants should
            produce the same results (within numerical errors). If the time
            evolution vectors are needed afterwords, it makes sense to set
            ``use_tev=True`` and ``full_output=True`` to get the vectors
            together with the time computed derivative.
        @param tevs
            Optional list of time evolution vector objects. Only used in case
            ``use_tev==True``. Each vector in this sequence has to correspond
            to the respective element in `pts`.
        @param unload
            Whether to free memory by unloading slice data from all MOTSs not
            in the same slice as the current MOTS. Default is `True`. If
            `False`, slice data of up to ``2*steps + 1`` will reside
            simultaneously in memory and not be removed after calling this
            method.
        @param full_output
            Whether to also return the tangent vectors to the MOTT that were
            used to compute the time derivative of the normal. In case
            ``use_tev=True``, this will be a list of TimeVectorData objects,
            while for ``use_tev=False``, it will be a list of
            SignatureQuantities objects.

        @b Notes

        This computes \f$\partial_t \nu^i\f$, where \f$\nu^i\f$ are the
        spatial components of the normal vector in the slice. It does **not**
        simply compute how the normal changes along the MOTT. If we define a
        "time" parameter along the MOTT via \f$\tau(p) = t\f$ iff
        \f$p\in\mathcal{S}_t\subset\Sigma_t\f$, then we can choose some
        coordinates on the MOTSs and follow the lines of varying \f$\tau\f$.
        There are two options to do that: One is to use the fraction of each
        curve's proper length as curve parameter and the other is to choose
        the parameter such that the \f$\tau\f$ coordinate produces a vector
        field \f$\partial_\tau\f$ that is orthogonal to the MOTSs. The latter
        is used if ``use_tev=True``. Note that
        \f[
            \partial_\tau = \frac{\partial x^\mu}{\partial\tau} \partial_\mu
                = \partial_t + \dot x^i \partial_i \,,
        \f]
        where the dot is defined as derivative w.r.t. \f$\tau\f$. As a result,
        we compute the derivative of the normal w.r.t. time via
        \f[
            \partial_t \nu^i = \dot\nu^i - \dot x^j \partial_j \nu^i \,.
        \f]
        """
        with self.fix_evaluator():
            return self._dt_normals(
                pts, curves, steps, use_tev=use_tev, tevs=tevs, unload=unload,
                full_output=full_output,
            )

    def _dt_normals(self, pts, curves, steps, use_tev, tevs, unload,
                    full_output):
        r"""Implements dt_normals()."""
        curves = self.collect_close_in_time(curves, steps=steps)
        t0 = self.metric.time
        im = self.cached_length_maps()[1]
        if use_tev:
            if tevs is None:
                tevs = self.time_evolution_vector(
                    pts=pts, curves=curves, steps=steps, full_output=True,
                )
            else:
                # Check that given TEVs match given points.
                if len(tevs) != len(pts):
                    raise ValueError("Incompatible shapes: %s != %s" %
                                     (len(tevs), len(pts)))
                for proper, tev in zip(pts, tevs):
                    param = im(proper)
                    if abs(param-tev.param) > 1e-10:
                        print("WARNING: Given time evolution vectors don't "
                              "seem to correspond to given points.")
                        break
            vectors = tevs
            epsilons = [v.eps for v in tevs]
        else:
            sigs = self.signature_quantities(pts=pts, curves=curves,
                                             steps=steps)
            vectors = sigs
            epsilons = np.zeros(len(pts))
        def _nu(c):
            r"""Return shape=(len(pts), 3) array with normals at each point."""
            try:
                im = c.cached_length_maps()[1]
                t = c.metric.time
                results = []
                for proper_param, eps in zip(pts, epsilons):
                    param = im(clip(proper_param + eps*(t-t0), 0.0, np.pi))
                    calc = c.get_calc_obj(param)
                    nu3_cov = calc.covariant_normal(diff=0)
                    nu3 = calc.g.inv.dot(nu3_cov)
                    results.append(nu3)
                return np.asarray(results)
            finally:
                if c is not self and unload:
                    c.metric.unload_data()
        # shape=(len(curves), len(pts), 3)
        normals_along_tube = np.asarray([_nu(c) for c in curves])
        dt_normals = []
        for i, (proper_param, vec) in enumerate(zip(pts, vectors)):
            tau = vec.tau
            nu_interp = CubicSpline(
                [c.metric.time for c in curves],
                [normals_along_tube[j,i,:] for j in range(len(curves))],
            )
            # \partial_\tau nu^i
            dtau_normal = nu_interp(t0, 1)
            param = im(proper_param)
            stab_calc = self.get_stability_calc_obj(param)
            # \partial_t nu^i
            dt_normal = (
                dtau_normal - np.einsum('j,ji->i', tau[1:], stab_calc.dnu3)
            )
            dt_normals.append(dt_normal)
        dt_normals = np.asarray(dt_normals)
        if full_output:
            return dt_normals, vectors
        return dt_normals

    def extremality_parameter(self, curves, steps=3, n="auto",
                              full_output=False):
        r"""Compute the extremality parameter.

        This computes the extremality parameter given in \[1\] (the correct
        formula can be found as eq. (14) in [2]) as
        \f[
            e = 1 - \frac{1}{8\pi} \int_{\mathcal{S}}
                \left(\frac{v_\bot + 1}{v_\bot - 1}\right)
                \Vert \sqrt{2}\sigma_{(\ell)} \Vert^2\ d^2V
              = 1 - \frac{1}{2} \int_0^\pi \sqrt{q}
                \left(\frac{v_\bot + 1}{v_\bot - 1}\right)
                \Vert \sigma_{(\ell)} \Vert^2\ d\lambda
                \,,
        \f]
        where \f$\lambda\f$ is our parameter along the curve and \f$v_\bot\f$
        is defined via
        \f[
            V^\mu = \alpha\;( n^\mu + v_\bot \nu^\mu ) \,.
        \f]
        Here \f$\alpha\f$ is the lapse function, \f$n^\mu\f$ the future
        pointing timelike normal to the spatial slice, \f$\nu^\mu\f$ the
        normal of the MOTS within the slice, and \f$V^\mu\f$ the time
        evolution vector (tangent to the MOTT and normal to the MOTS).

        The factor of \f$\sqrt{2}\f$ results from a different scaling
        convention for the null normals used to define the shear.

        @b References

        [1] Booth, Ivan, and Stephen Fairhurst. "Extremality conditions for
            isolated and dynamical horizons." Physical review D 77.8 (2008):
            084005.

        [2] Booth, Ivan. "Two physical characteristics of numerical apparent
            horizons." Canadian Journal of Physics 86.4 (2008): 669-673.
        """
        if n == "auto":
            n = max(30, 2 * self.num)
        with self.fix_evaluator():
            area_element = self.get_area_integrand()
            length_map = self.cached_length_maps()[0]
            v_bots = []
            params = []
            def integrand(xs):
                values = []
                params[:] = xs
                proper_params = [length_map(param) for param in params]
                tevs = self.time_evolution_vector(
                    pts=proper_params, curves=curves, steps=steps,
                    full_output=True
                )
                for param, tev in zip(params, tevs):
                    stab_calc = self.get_stability_calc_obj(param)
                    shear2 = stab_calc.compute_shear_squared()
                    V = tev.V_tilde
                    g4 = stab_calc.g4
                    n = stab_calc.n
                    nu = stab_calc.nu
                    v_bot = - g4.dot(V).dot(nu) / g4.dot(V).dot(n)
                    values.append(
                        area_element(param) * shear2 * (v_bot + 1)/(v_bot - 1)
                    )
                    v_bots.append(v_bot)
                return np.asarray(values)
            e = 1.0 - 0.5 * _fixed_quad(
                integrand, a=0.0, b=np.pi, n=n
            )
            if full_output:
                return e, np.asarray(v_bots), np.asarray(params)
            return e

    def surface_gravity(self, pts, curves, steps=3, wrt="ell", tevs=None):
        r"""Compute a (slicing dependent) surface gravity.

        This computes the surface gravity according to eq. (19) in Ref. [1],
        i.e.
        \f[
            \kappa = - k_\mu X^\nu \nabla_\nu \ell^\mu \,,
        \f]
        where \f$X^\mu = \ell^\mu\f$ by default.
        However, we use a different (slicing dependent) scaling of the null
        normals. In particular, we choose
        \f[
            \ell^\mu = \frac{1}{\sqrt{2}} (n^\mu + v^\mu) \,,
            \qquad
            k^\mu = \frac{1}{\sqrt{2}} (n^\mu - v^\mu) \,,
        \f]
        where \f$\ell^\mu\f$ and \f$k^\mu\f$ are the future pointing outgoing
        and ingoing null normals, respectively. Here, \f$n^\mu\f$ is the
        timelike future pointing normal on the spatial slice and \f$v^\mu\f$
        the outward normal of the MOTS within the slice. The above choice
        leads to a cross-normalization of \f$\ell^\mu k_\mu = -1\f$.

        Note that this cross-normalization does not fix the scaling of the
        null normals. If \f$f\f$ is an arbitrary positive function, then
        \f$\tilde\ell^\mu := f\ell^\mu\f$ and \f$\tilde k^\mu := fk^\mu\f$
        satisfy the same condition but will result in different values for the
        surface gravity.

        @param pts
            Points at which to compute the surface gravity. Values should be
            the pi-normalized proper lengths, i.e. `0` is at the north pole,
            `pi` at the south pole, and `0.5` at `50%` proper length.
        @param curves
            List of curves that build up the tube. The current object (`self`)
            may or may not be in that list. Each curve needs to have a metric
            with a `time` attribute. They do not need to be sorted. This
            parameter is needed to numerically estimate the time derivative of
            the normal to the MOTS in the slice.
        @param steps
            How many curves in the future and past of this curve to consider
            from `curves`. Default is `3`.
        @param wrt
            Vector with respect to which to compute the surface gravity.
            Possible values are ``"ell"`` (default) and ``"tev"``.
        @param tevs
            In case you use the time evolution vector (``"tev"``) option for
            the `wrt` parameter, you may optionally supply a list of these
            vectors (objects) here, in case they are already precomputed.
            These should correspond to the points in `pts`.

        @b References

        [1] Pielahn, Mathias, Gabor Kunstatter, and Alex B. Nielsen.
            "Dynamical surface gravity in spherically symmetric black hole
            formation." Physical Review D 84.10 (2011): 104008.
        """
        if wrt not in ("ell", "tev"):
            raise ValueError("Unknown vector: %s" % (wrt,))
        with self.fix_evaluator():
            return self._surface_gravity(
                pts, curves, steps, wrt=wrt, tevs=tevs,
            )

    def _surface_gravity(self, pts, curves, steps, wrt, tevs):
        r"""Implements surface_gravity()."""
        dt_normals, vectors = self.dt_normals(
            pts, curves, steps=steps, use_tev=wrt == "tev", tevs=tevs,
            full_output=True,
        )
        kappa = []
        im = self.cached_length_maps()[1]
        for proper_param, dt_normal, vec in zip(pts, dt_normals, vectors):
            param = im(proper_param)
            stab_calc = self.get_stability_calc_obj(param)
            X = vec.V_B if wrt == "tev" else None
            kappa.append(stab_calc.surface_gravity(dt_normal, X=X))
        return np.asarray(kappa)

    def timescale_tau2(self, tevs, kappas_tev, proper_pts, curves, steps=3, option=0):
        r"""Compute the square of the timescale `tau`.

        This computes
        \f[
            \tau^2 := \left(
                \frac{1}{A} \int_{\mathcal{S}} \kappa^{(V)} \Theta_{(V)}\ dA
            \right)^{-1} \,,
        \f]
        where `A` is the MOTS's surface area,
        \f$\kappa^{(V)} = - k^\mu V^\nu \nabla_\nu \ell_\mu\f$ is the surface
        gravity with respect to the time evolution vector `V` (with unit time
        component) and \f$\ell\f$ and \f$k\f$ are the outgoing and ingoing
        null normals with scaling \f$\ell \cdot k = -1\f$.

        @param tevs
            List of `n` time evolution vector objects at the Gaussian
            quadrature points for an integral of order `n`.
        @param kappas_tev
            List of values of the surface gravity w.r.t. the time evolution
            vectors.
        @param proper_pts
            List of proper parameter values (in ``[0, pi]``) at which the
            `kappas_tev` are computed. Only used to verify that all quantities
            are evaluated at the correct points.
        @param curves
            List of curves that build up the tube. The current object (`self`)
            may or may not be in that list. Each curve needs to have a metric
            with a `time` attribute. They do not need to be sorted. This
            parameter is needed to numerically estimate the time derivative of
            the `c` coefficient of the time evolution vectors (used only in
            option 4, see below).
        @param steps
            How many curves in the future and past of this curve to consider
            from `curves`. Default is `3`.
        @param option
            For debugging/exploration purposes. Use different definitions of
            \f$\tau\f$ to experiment with. The options are (**note** that for
            option 2 we return \f$\tau\f$ instead of \f$\tau^2\f$):
                * option 0: default, the one given in the above equation
                * option 1: use \f$1/\tau^2 = \frac{1}{A} \int_{\mathcal{S}} \Theta_{(V)}^2\ dA\f$
                * option 2: use \f$1/\tau = \frac{1}{A} \int_{\mathcal{S}} \Theta_{(V)}\ dA\f$
                * option 3: use \f$1/\tau^2 = \frac{1}{A} \int_{\mathcal{S}} [\kappa^{(V)} \Theta_{(V)} - \frac12 \Theta_{(V)}^2] \ dA\f$
                * option 4: use \f$1/\tau^2 = \frac{1}{A} \int_{\mathcal{S}} [\kappa^{(V)} \Theta_{(V)} - \frac12 \Theta_{(V)}^2 - \Theta_{(V)} \frac{d}{dt} \ln c] \ dA\f$
        """
        if len(kappas_tev) != len(proper_pts):
            raise ValueError("Surface gravity sampled on incorrect number of points.")
        n = len(kappas_tev)
        kappas_tev = np.asarray(kappas_tev)
        with self.fix_evaluator():
            area_element = self.get_area_integrand()
            length_map = self.cached_length_maps()[0]
            def _integrand(xs):
                proper_params = [length_map(param) for param in xs]
                cmp_opts = dict(rtol=1e-10, atol=0.0)
                if (not np.allclose(proper_params, proper_pts, **cmp_opts) or
                        not np.allclose(xs, [tev.param for tev in tevs], **cmp_opts)):
                    print("WARNING: Objects not sampled at quadrature points. "
                          "Integral value might be incorrect.")
                measure = np.asarray([area_element(x) for x in xs])
                area = 2*np.pi * _fixed_quad(lambda _: measure, a=0.0, b=np.pi, n=n)
                ingoing_exp = np.asarray(self.expansions(xs, ingoing=True))
                ingoing_exp = ingoing_exp / np.sqrt(2) # convert to l*k = -1
                c = np.asarray([tev.c_B for tev in tevs])
                theta_V = c * ingoing_exp
                if option == 0:
                    integrand = kappas_tev * theta_V
                elif option == 1:
                    integrand = theta_V**2
                elif option == 2:
                    integrand = theta_V # The result will be tau not tau^2 here!
                elif option == 3:
                    integrand = kappas_tev * theta_V - 0.5 * theta_V**2
                elif option == 4:
                    dt_ln_c = self._compute_dtau_c(curves=curves, steps=steps, tevs=tevs) / c
                    integrand = (
                        kappas_tev * theta_V - 0.5 * theta_V**2
                        - theta_V * dt_ln_c
                    )
                return 1/area * integrand * measure
            integral = 2*np.pi * _fixed_quad(_integrand, a=0.0, b=np.pi, n=n)
            return 1/integral

    def _compute_dtau_c(self, curves, steps, tevs):
        curves = self.collect_close_in_time(curves, steps=steps)
        t0 = self.metric.time
        lm = self.cached_length_maps()[0]
        pts = [lm(v.param) for v in tevs]
        epsilons = [v.eps for v in tevs]
        def _c(curve):
            r"""Return shape=(len(pts),) array with `c` parameters at each point."""
            c_tevs = curve.user_data["tev_divergence"]["vectors"]
            c_c_interp = CubicSpline(
                [v.param for v in c_tevs], [v.c_B for v in c_tevs],
            )
            c_im = curve.cached_length_maps()[1]
            t = curve.metric.time
            results = []
            for proper_param, eps in zip(pts, epsilons):
                param = c_im(clip(proper_param + eps*(t-t0), 0.0, np.pi))
                results.append(c_c_interp(param))
            return np.asarray(results)
        # shape=(len(curves), len(pts))
        c_along_tube = np.asarray([_c(curve) for curve in curves])
        dtau_c = []
        for i, (proper_param, vec) in enumerate(zip(pts, tevs)):
            tau = vec.tau
            c_interp = CubicSpline(
                [c.metric.time for c in curves],
                [c_along_tube[j,i] for j in range(len(curves))],
            )
            # \partial_\tau c
            dtau_c.append(c_interp(t0, 1))
        dtau_c = np.asarray(dtau_c)
        return dtau_c

    def timescale_T2(self, tevs):
        r"""Compute the square of the timescale `T`.

        This computes
        \f[
            T^2 := \left(
                \frac{1}{A} \int_{\mathcal{S}} \sigma^{(V)}_{AB}\sigma^{(\tau)\,AB}\ dA
            \right)^{-1} \,,
        \f]
        where `A` is the MOTS's surface area, `V` is the time evolution vector
        (scaled to have unity time component) with
        \f[
            V^\mu = b \ell^\mu + c k^\mu \,.
        \f]
        Then, \f$\tau^\mu = b \ell^\mu - c k^\mu\f$. \f$\ell\f$ and \f$k\f$
        are the outgoing and ingoing null normals with scaling
        \f$\ell \cdot k = -1\f$.

        @param tevs
            List of `n` time evolution vector objects at the Gaussian
            quadrature points for an integral of order `n`.
        """
        n = len(tevs)
        with self.fix_evaluator():
            area_element = self.get_area_integrand()
            length_map = self.cached_length_maps()[0]
            def integrand(xs):
                proper_params = [length_map(param) for param in xs]
                cmp_opts = dict(rtol=1e-10, atol=0.0)
                if not np.allclose(xs, [tev.param for tev in tevs], **cmp_opts):
                    print("WARNING: Objects not sampled at quadrature points. "
                          "Integral value might be incorrect.")
                sigma_V_sigma_tau = []
                for param, tev in zip(xs, tevs):
                    stab_calc = self.get_stability_calc_obj(param)
                    sigma_ell2 = stab_calc.compute_shear_squared()
                    sigma_k2 = stab_calc.compute_shear_k(full_output=True)[1]
                    sigma_V_sigma_tau.append(
                        tev.b_B**2 * sigma_ell2 - tev.c_B**2 * sigma_k2
                    )
                sigma_V_sigma_tau = np.asarray(sigma_V_sigma_tau)
                measure = np.asarray([area_element(x) for x in xs])
                area = 2*np.pi * _fixed_quad(lambda _: measure, a=0.0, b=np.pi, n=n)
                return 1/area * sigma_V_sigma_tau * measure
            integral = 2*np.pi * _fixed_quad(integrand, a=0.0, b=np.pi, n=n)
            return 1/integral

    def xi_vector(self, pts, curves, steps=3, full_output=False):
        r"""Compute the xi vector.

        The quantity computed here is called \f$\zeta^A\f$ in eq. (3.10) in
        Ref. [1], see also eq. (3.19) in Ref. [2]. We compute
        \f[
            \xi^A := q^{AB} \hat r^\mu \nabla_\mu \ell_B \,,
        \f]
        where \f$\hat r^\mu\f$ is the normalized time evolution vector (see
        time_evolution_vector()), \f$q^{AB}\f$ the inverse 2-metric on the
        MOTS, and \f$\ell^\mu\f$ the future pointing outgoing null normal to
        the MOTS. Note that we use the scaling \f$k^\mu \ell_\mu = -1\f$ here,
        where \f$k^\mu\f$ is the future pointing ingoing null normal.

        **Note:** This quantity depends on the scaling of the null normals. We
        construct the null normals in a slicing dependent way using
        \f[
            \ell^\mu = \frac{1}{\sqrt{2}} (n^\mu + v^\mu) \,,
            \qquad
            k^\mu = \frac{1}{\sqrt{2}} (n^\mu - v^\mu) \,,
        \f]
        where \f$n^\mu\f$ is the future pointing normal on the slice and
        \f$v^\mu\f$ the outward normal of the MOTS within the slice. This
        choice leads to \f$\ell^\mu k_\mu = -1\f$. However, you can scale both
        null normals by an arbitrary function in a way to keep the condition
        \f$\ell^\mu k_\mu = -1\f$. In Ref. [2], a different choice is made,
        namely
        \f[
            \ell^\mu = \hat\tau^\mu + \hat r^\mu \,,
        \f]
        where \f$\hat\tau^\mu\f$ is the normalized outward normal to the MOTT
        (i.e. a timelike vector for spacelike surfaces like dynamical
        horizons) and \f$\hat r^\mu\f$ the time evolution vector, i.e. the
        tangent to the MOTT which is also a normal to the MOTS. Hence,
        quantitative results presented in [2] will not hold for the vector
        computed here.

        @return Components of \f$\xi^A\f$ as a 2-element array. If
            ``full_output=True``, return \f$\xi^A\f$, \f$\xi_A\f$,
            \f$|\xi|^2\f$, \f$\xi_{(\ell)}\f$, where
            \f$\xi_{(\ell)} := \overline{m}^A \xi_A\f$. Here
            \f$\overline{m}^\mu\f$ is one of the components of the complex
            null tetrad \f$(\ell,k,m,\overline{m})\f$.

        @param pts
            Proper length parameters at which to compute the results. Each
            point should be in the range ``(0, pi)``, where a value of e.g.
            ``pi/2`` refers to a point on the MOTS dividing the curve into two
            parts of equal proper length.
        @param curves
            List of curves that build up the tube. The current object (`self`)
            may or may not be in that list. Each curve needs to have a metric
            with a `time` attribute. They do not need to be sorted. This
            parameter is needed to numerically estimate the time derivative of
            the normal to the MOTS in the slice.
        @param steps
            How many curves in the future and past of this curve to consider
            from `curves`. Default is `3`.
        @param full_output
            Whether to return just the contravariant components of the xi
            vector or also the covariant, the square, and the complex scalar.
            See above for details. Default is `False`.

        @b References

        [1] Ashtekar, Abhay, Miguel Campiglia, and Samir Shah. "Dynamical
            black holes: Approach to the final state." Physical Review D 88.6
            (2013): 064045.

        [2] Ashtekar, Abhay, and Badri Krishnan. "Dynamical horizons and their
            properties." Physical Review D 68.10 (2003): 104030.
        """
        with self.fix_evaluator():
            return self._xi_vector(pts, curves, steps, full_output)

    def _xi_vector(self, pts, curves, steps, full_output):
        r"""Implements xi_vector()."""
        dt_normals, tevs = self.dt_normals(
            pts, curves, steps=steps, use_tev=True, full_output=True
        )
        xi_A_up = []
        xi_A = []
        xi2 = []
        xi_scalar = []
        im = self.cached_length_maps()[1]
        for proper_param, dt_normal, tev in zip(pts, dt_normals, tevs):
            param = im(proper_param)
            stab_calc = self.get_stability_calc_obj(param)
            xi_A_up.append(stab_calc.xi_vector(dt_normal, tev, up=True))
            if full_output:
                xi_A.append(stab_calc.xi_vector(dt_normal, tev, up=False))
                xi2.append(stab_calc.xi_squared(dt_normal, tev))
                xi_scalar.append(stab_calc.xi_scalar(dt_normal, tev))
        xi_A_up = np.asarray(xi_A_up)
        if full_output:
            xi_A = np.asarray(xi_A)
            xi2 = np.asarray(xi2)
            xi_scalar = np.asarray(xi_scalar)
            return xi_A_up, xi_A, xi2, xi_scalar
        return xi_A_up

    def xi_square_integral(self, curves, steps=3, n="auto"):
        r"""Compute the integral of the square of the xi vector.

        We compute \f$\int_\mathcal{S} \xi^A \xi_A d^2V\f$.

        @param curves
            List of curves that build up the tube. The current object (`self`)
            may or may not be in that list. Each curve needs to have a metric
            with a `time` attribute. They do not need to be sorted.
        @param steps
            How many curves in the future and past of this curve to consider
            from `curves`. Default is `3`.
        @param n
            Order of the fixed quadrature integration (equal to the number of
            points at which the integrand is evaluated). The default
            ``"auto"`` will use twice the current curve's resolution, but at
            least 30 points.
        """
        if n == "auto":
            n = max(30, 2 * self.num)
        with self.fix_evaluator():
            area_element = self.get_area_integrand()
            def integrand(xs):
                measure = np.asarray([area_element(x) for x in xs])
                xi2 = self.xi_vector(pts=xs, curves=curves, steps=steps,
                                     full_output=True)[2]
                return measure * xi2
            return 2*np.pi * _fixed_quad(
                integrand, a=0.0, b=np.pi, n=n
            )

    def expand_xi_scalar(self, curves, steps=3, lmax="auto", zeta=None,
                         min_lmax=64, max_lmax=512, coeff_threshold=None,
                         compress=True, full_output=False):
        r"""Expand the xi vector into spin weighted spherical harmonics.

        We expand the quantity
        \f[
            \xi_{(\ell)} := \overline{m}^A \xi_A
        \f]
        into spin weighted spherical harmonics of spin weight -1. The complex
        null tetrad \f$(\ell, k, m, \overline{m})\f$ is constructed using
        \f$m = \frac{1}{\sqrt{2}} (e_1 + i e_2)\f$, where
        \f$e_1 = \frac{\partial_\lambda}{\sqrt{q_{\lambda\lambda}}}\f$ and
        \f$e_2 = \frac{\partial_\varphi}{\sqrt{q_{\varphi\varphi}}}\f$.

        @param curves
            List of curves that build up the tube. The current object (`self`)
            may or may not be in that list. Each curve needs to have a metric
            with a `time` attribute. They do not need to be sorted.
        @param steps
            How many curves in the future and past of this curve to consider
            from `curves`. Default is `3`.
        @param lmax
            Largest `ell` value in the expansion into \f${}_{-1}Y_{\ell m}\f$.
            Default is ``"auto"``, which tries to find a maximum between
            `min_lmax` and `max_lmax` such that the coefficients fall off
            sufficiently up to `lmax`.
        @param min_lmax,max_lmax
            If ``lmax="auto"``, start expanding with ``lmax=min_lmax``. If no
            "knee" is detected in the resulting coefficients, the resolution
            is doubled until a knee is detected or we have reached `max_lmax`.
        @param coeff_threshold
            If the final few coefficients are larger than `coeff_threshold`
            times the largest coefficient, then do not try to detect a knee in
            the coefficients but immediately increase the resolution. By
            default, knee detection is always run.
        @param compress
            Since the scalar is purely real for axisymmetric MOTSs, the only
            non-zero coefficients in the expansion will be the ones with
            `m=0`. Setting ``compress=True`` (default) will extract this one
            row of possibly non-vanishing coefficients and return it.
        @param full_output
            If `True`, return the coefficients, the (complex) shear scalar
            values at the grid points in \f$\theta=\arccos\zeta\f$ and the
            grid points themselves. Otherwise (default), return just the
            coefficients.
        """
        curves = self.collect_close_in_time(curves, steps=steps)
        lm = self.cached_length_maps()[0]
        def _xi_scalar(params):
            pts = [lm(param) for param in params]
            xi_scalar = self.xi_vector(
                pts, curves=curves, steps=None, full_output=True
            )[3]
            return xi_scalar
        return self._expand_into_SWSPH(
            func=_xi_scalar, spinweight=-1, lmax=lmax, min_lmax=min_lmax,
            max_lmax=max_lmax, coeff_threshold=coeff_threshold,
            compress=compress, zeta=zeta, full_output=full_output,
        )

    def get_stability_calc_obj(self, param):
        r"""Return a StabilityCalc object at the given point of the curve."""
        metric4 = FourMetric(self.metric)
        stab_calc = StabilityCalc(curve=self, param=param, metric4=metric4)
        return stab_calc

    def linearized_equation_at(self, param, target_expansion=0.0):
        r"""Return the terms of the linearized version of H=0 at one point."""
        calc = self.get_calc_obj(param)
        H = calc.expansion()
        dhH = calc.diff(hdiff=0)
        dhpH = calc.diff(hdiff=1)
        dhppH = calc.diff(hdiff=2)
        return dhH, dhpH, dhppH, -H + target_expansion

    def linearized_equation(self, params, target_expansion=0.0,
                            parallel=False, pool=None):
        r"""Return linearized equation as operator and inhomogeneity on a grid.

        This form is suitable to be processed by the pseudospectral solver
        ndsolve.solver.ndsolve().

        @param params
            Points at which to evaluate the equation terms. Should be the
            collocation points for a pseudospectral solver.
        @param target_expansion
            Optional float indicating the desired expansion of the surface to
            find. Default is `0.0` (i.e. a MOTS).
        @param parallel
            Whether to evaluate the equation using multiple processes in parallel.
            If `True`, uses all available threads. If an integer, uses that many
            threads. Default is `False`, i.e. don't compute in parallel.
        @param pool
            Optional processing pool to re-use. If not given, a new pool is
            created and then destroyed after the computation.
        """
        with self.fix_evaluator():
            c = target_expansion
            if parallel:
                if parallel is True:
                    processes = None
                else:
                    processes = parallel if isinstance(parallel, int) else None
                values = parallel_compute(
                    func=self.linearized_equation_at,
                    arg_list=params,
                    args=(c,),
                    processes=processes,
                    callstyle='plain',
                    pool=pool,
                )
            else:
                values = [self.linearized_equation_at(p, c) for p in params]
            op0, op1, op2, inhom = np.array(values).T
            op = (op0, op1, op2)
            return op, inhom

    def ricci_scalar(self, param):
        r"""Compute the Ricci scalar of the surface at the given parameter.

        See expcalc.ExpansionCalc.ricci_scalar() for details.
        """
        with self.fix_evaluator():
            calc = self.get_calc_obj(param)
            return calc.ricci_scalar()

    def extrinsic_surface_curvature(self, param, trace=False, square=False):
        r"""Compute the extrinsic curvature tensor of the surface.

        This computes the components
        \f[
            k_{AB} = -\nabla_A \nu_B,
                \qquad A,B = \lambda,\varphi
        \f]
        where \f$\lambda,\varphi\f$ are coordinates on the surface represented
        by this curve, \f$\nu\f$ is the outward pointing normal of the surface
        in the slice, and \f$\nabla\f$ is the spacetime covariant derivative.

        @param param
            Parameter (i.e. \f$\lambda\f$ value) specifying the point on the
            curve at which to compute the extrinsic curvature of the surface.
        @param trace
            If `True`, returns the trace of the extrinsic curvature. Default
            is `False`. May not be used together with `square`.
        @param square
            If `True`, returns the square \f$k_{AB} k^{AB}\f$. Default is
            `False`. May not be used together with `trace`.

        @return A NumPy 2x2 array containing the components of `k_AB`. If
            either `trace` or `square` is `True`, returns a float.
        """
        with self.fix_evaluator():
            calc = self.get_calc_obj(param)
            return calc.extrinsic_curvature(trace=trace, square=square)

    def area(self, full_output=False, disp=False, domain=(0, np.pi), **kw):
        r"""Compute the area of the surface represented by this curve.

        Note that no warning will be generated in case the integral does not
        converge properly. To obtain information about such warnings, either
        set ``disp=True`` to raise an error in these cases, or use
        ``full_output=True`` and check if the last returned element is `None`
        (signaling no problem was found).

        @param full_output
            If `True`, return a 4-tuple containing the computed area, an
            estimated error, the `infodict` result of `scipy.integrate.quad`
            and any generated warning message. This fourth element will be
            `None` in case no warning occurred. Default is `False`.
        @param disp
            If `Ture`, raise any warnings generated during integration as an
            `IntegrationError`.
        @param domain
            Parameter range to integrate over. Default is the full surface,
            i.e. ``(0, pi)``.
        @param **kw
            Additional keyword arguments are passed to `scipy.integrate.quad`.

        @return The computed area. See also `full_output` above.

        @b Notes

        The area of the surface \f$\sigma\f$ is defined as
        \f[
            A = \int_\sigma \sqrt{\det q}\ d^2x,
        \f]
        where
        \f[
            q = \Pi_\sigma g = g\big|_\sigma - \underline{\nu} \otimes \underline{\nu},
            \qquad
            q_{ab} = g_{ab} - \nu_a \nu_b
        \f]
        is the induced metric on \f$\sigma\f$ (compare equation (2) in [1]).
        Here, \f$\nu\f$ is the outward pointing normal of \f$\sigma\f$ and
        \f$\underline{\nu} = g(\nu,\,\cdot\,)\f$.

        @b References

        [1] Gundlach, Carsten. "Pseudospectral apparent horizon finders: An
            efficient new algorithm." Physical Review D 57.2 (1998): 863.
        """
        with self.fix_evaluator():
            a, b = domain
            result = quad(
                self.get_area_integrand(), a=a, b=b, full_output=True,
                **kw
            )
            if len(result) == 4:
                res, err, info, warning = result
                if disp:
                    raise IntegrationError(warning)
            else:
                res, err, info = result
                warning = None
            area = 2*np.pi * res
            err = 2*np.pi * err
            if full_output:
                return area, err, info, warning
            return area

    def get_ricci_scalar_func(self, cached=False):
        r"""Create an (optionally cached) function to evaluate the Ricci scalar."""
        if cached:
            return _cached(self.ricci_scalar)
        return self.ricci_scalar

    def get_det_q_func(self, cached=False):
        r"""Create a function to evaluate det(q).

        @param cached
            If `True`, cache every evaluation of the returned callable.
            Default is `False`. Note that a new cache is created for each
            created function.
        """
        def det_q(param):
            calc = self.get_calc_obj(param)
            q = calc.induced_metric()
            return linalg.det(q)
        if cached:
            return _cached(det_q)
        return det_q

    def get_area_integrand(self):
        r"""Return a callable evaluating the sqrt(det(q)), i.e. the area element."""
        det_q = self.get_det_q_func()
        integrand = lambda t: np.sqrt(det_q(t))
        return integrand

    def euler_char(self, a=0, b=np.pi, full_output=False, disp=False, **kw):
        r"""Compute the Euler characteristic of the surface.

        This assumes the curve to represent a closed (compact) smooth surface
        without boundary. As such, we get by the Gauss-Bonnet theorem
        \f[
            2\pi\chi(\mathcal{S}) = \frac{1}{2} \int_\mathcal{S} \mathcal{R}\ dA,
        \f]
        where \f$\chi\f$ is the Euler characteristic of the surface
        \f$\mathcal{S}\f$ which is represented by this curve and
        \f$\mathcal{R}\f$ its Ricci (scalar) curvature.

        In axisymmetry and corresponding coordinates, we can carry out the
        integral over the angle \f$\phi\f$, which gives \f$2\pi\f$ and end up
        with
        \f[
            \chi(\mathcal{S}) = \frac{1}{2} \int_0^\pi \mathcal{R}(\lambda)\sqrt{q}\ d\lambda,
        \f]
        where \f$\sqrt{q}\f$ is the square root of the determinant of the
        induced metric `q`.

        From the Euler characteristic, you can compute the genus `G` of the
        surface via \f$G = -\chi/2 + 1\f$.

        @param a,b
            Curve parameters between which to integrate. Defaults are `0` and
            `pi`, respectively.
        @param full_output
            Whether to return information about the integration in addition to
            the value. If `True`, return a tuple of ``chi, chi_err, info,
            warning``. Otherwise (default), return just ``chi``. If
            ``warning`` is not `None`, the integral did not converge as
            expected.
        @param disp
            Whether to raise an error in case the integral did not converge as
            expected.
        @param **kw
            Further keyword arguments are passed to `scipy.integrate.quad()`.
            Use these to set e.g. `epsrel` to the desired tolerance.
        """
        with self.fix_evaluator():
            det_q = self.get_det_q_func()
            integrand = lambda t: np.sqrt(det_q(t)) * 0.5 * self.ricci_scalar(t)
            result = quad(
                integrand, a=a, b=b, full_output=True,
                **kw
            )
            if len(result) == 4:
                res, err, info, warning = result
                if disp:
                    raise IntegrationError(warning)
            else:
                res, err, info = result
                warning = None
            # surface integral is 2*pi times res
            # chi = integral / 2*pi, so we're fine
            chi = res
            chi_err = err
            if full_output:
                return chi, chi_err, info, warning
            return chi

    def irreducible_mass(self, area=None):
        r"""Compute the irreducible mass.

        The irreducible mass of a horizon is (see e.g. [1])
        \f[
            M := \sqrt{A/16\pi},
        \f]
        where `A` is the horizon's area.

        @param area
            Optional area for which to compute the mass. By default, the area
            of the axisymmetric surface represented by this curve is computed.

        @b References

        [1] Chu, Tony, Harald P. Pfeiffer, and Michael I. Cohen. "Horizon
            dynamics of distorted rotating black holes." Physical Review D
            83.10 (2011): 104018.
        """
        if area is None:
            area = self.area()
        return np.sqrt(area/(16*np.pi))

    def horizon_radius(self, area=None):
        r"""Compute the horizon radius.

        The horizon radius is defined as (cf. [1])
        \f[
            R := \sqrt{A/4\pi},
        \f]
        where `A` is the horizon's area.

        @param area
            Optional area for which to compute the horizon radius. By default,
            the area of the axisymmetric surface represented by this curve is
            computed.

        @b References

        [1] Ashtekar, Abhay, and Badri Krishnan. "Isolated and dynamical
            horizons and their applications." Living Reviews in Relativity 7.1
            (2004): 10.
        """
        if area is None:
            area = self.area()
        return np.sqrt(area/(4*np.pi))

    def stability_parameter(self, num=None, m_max=None, m_terminate_index=30,
                            rtol=1e-12, compute_eigenfunctions=False,
                            slice_normal=True, transform_torsion=False,
                            zeta=None, full_output=False):
        r"""Compute the stability parameter.

        The stability parameter is defined in [1] as the principal eigenvalue
        of the stability operator. We use the normal vector in the slice and
        use the axisymmetry to simplify the analytical task.

        According to Proposition 5.1 in [1], the MOTS represented by this
        curve is stably outermost iff the stability parameter is greater or
        equal to zero and strictly stably outermost iff it is greater than
        zero.

        @param num
            Pseudospectral resolution of the discretization of the operator.
            By default, uses the resolution of the curve representation
            itself.
        @param m_max
            Maximum angular mode to consider. Default is to use `num-1`.
        @param m_terminate_index
            Index of the eigenvalue of the `m=0` mode to use to as stopping
            criterion for the angular mode. If the real part of the thus
            specified `m=0` eigenvalue is less or equal to the (real part of
            the) smallest eigenvalue of the current `m`, then no higher `m`
            modes are considered. This effectively gives the number of `m=0`
            eigenvalues of which we want to determine the multiplicity.
            Default is `30`. Set explicitly to `None` to not stop based on
            this criterion.
        @param rtol
            Tolerance for reality check. The principal eigenvalue is shown in
            [1] to be real. If it has a non-zero imaginary part greater than
            ``rtol * |real_part|``, then a `RuntimeError` is raised. Default
            is `1e-12`.
        @param compute_eigenfunctions
            Whether to compute the eigenfunction for each eigenvalue. Default
            is `False`. The eigenfunctions will be accessible from the
            StabilitySpectrum object via the ``spectrum.get_eigenfunction()``
            method. Note that the method for computing the eigenvalues is a
            different one in this case and hence the numerical values (and
            even the accuracy) may differ from the case
            ``compute_eigenfunctions=False``.
        @param slice_normal
            Whether to consider the stability operator w.r.t. the outward
            normal in the spatial slice (default). If `False`, consider the
            operator w.r.t. the past-pointing outward null normal
            \f$-k^\mu\f$.
        @param transform_torsion
            Apply a transformation to the lightlike null vectors `k` and `l`
            such that the rotation 1-form \f$s_A\f$ (torsion of `l`) becomes
            divergence free. Default is `False`.
        @param zeta
            Optional invariant angle zeta as function of curve parameter. If
            not supplied, will generate it when needed (i.e. when
            eigenfunctions are requested).
        @param full_output
            If `True`, return all eigenvalues in addition to the principal
            eigenvalue.

        @return The principal eigenvalue as a float (i.e. the real part,
            ignoring any spurious imaginary part). If `full_output==True`,
            returns a StabilitySpectrum object as second element containing
            all found eigenvalues.

        @b Notes

        If the MOTS \f$\mathcal{S}\f$ is contained in a time-symmetric slice
        \f$\Sigma\f$ of spacetime, and we compute the stability w.r.t. the
        outward pointing normal \f$\nu\f$ of \f$\mathcal{S}\f$ in
        \f$\Sigma\f$, then the stability operator simplifies greatly to
        \f[
            L_\nu \zeta = -\Delta_\mathcal{S} \zeta
                - (R_{ij}\nu^i\nu^j + \mathcal{K}_{AB}\mathcal{K}^{AB}) \zeta,
        \f]
        where \f$R_{ij}\f$ is the Ricci tensor of \f$(\Sigma, g)\f$,
        \f$\mathcal{K}_{AB}\f$ is the extrinsic curvature of
        \f$\mathcal{S}\f$, \f$\Delta_\mathcal{S} = q^{AB}\,D_A\,D_B\f$
        is the Laplacian on \f$(\mathcal{S}, q)\f$, `g` is the 3-metric on the
        slice and `q` the induced 2-metric on the MOTS.

        Without time symmetry, the full operator (in vacuum) reads
        \f[
            L_\nu \zeta = -\Delta_\mathcal{S} \zeta + 2 s^A D_A \zeta
                + \big(
                    \frac{1}{2} \mathcal{R} - s_A s^A + D_A s^A
                    - \frac{1}{2} q^{AC} q^{BD} K^\mu_{AB} K^\nu_{CD} \ell_\mu \ell_\nu
                \big).
        \f]
        Here, \f$\mathcal{R}\f$ is the Ricci scalar of \f$\mathcal{S}\f$,
        \f$s_A = -\frac{1}{2} k_\mu \nabla_A \ell^\mu\f$ and
        \f$K^\mu_{AB}\ell_\mu = -\nabla_A \ell_B\f$. Note that \f$\nabla\f$
        refers to the covariant derivative compatible with the spacetime
        4-metric and \f$k^\mu\f$ and \f$\ell^\mu\f$ are the future pointing
        ingoing and outgoing null normals to \f$\mathcal{S}\f$, respectively.

        To find the principal eigenvalue of \f$L_\nu\f$, first note that due
        to the axisymmetry, the \f$\varphi\f$-dependence of \f$\zeta\f$ can be
        expressed as
        \f[
            \zeta(\lambda, \varphi) = \sum_{m=-\infty}^\infty \zeta_m(\lambda) e^{im\varphi}.
        \f]
        Since the \f$e^{im\varphi}\f$ are linearly independent, the spectrum
        will consist of the union of spectra of the \f$\zeta_m\f$. When taking
        the union, we label the eigenvalues by their `m`-mode to be able to
        count their multiplicity.

        The effect of applying \f$L\f$ to \f$\zeta_m e^{im\varphi}\f$ can in
        axisymmetry be reduced to a 1D problem of acting on just \f$\zeta_m\f$
        via (no summation here, \f$m\f$ is the angular mode, not an index)
        \f[
            L^m \zeta_m = (L + 2 i m s^\varphi + m^2 q^{\varphi\varphi}) \zeta_m.
        \f]

        The eigenvalues themselves are found using
        motsfinder.ndsolve.solver.NDSolver.eigenvalues(), which means we just
        need to provide the coefficient functions defining the one-dimensional
        operator \f$L^m\f$ acting on \f$\zeta_m\f$. The Laplacian acting on
        \f$\zeta_m\f$ reads
        \f{eqnarray*}{
            \Delta_\sigma \zeta_m
                &=& \frac{1}{\sqrt{q}} \partial_A (\sqrt{q} q^{AB} \partial_B \zeta_m)
                = \left(
                    \frac{1}{2} q^{AB} q^{C\lambda} \partial_C q_{AB}
                    + \partial_A q^{A\lambda}
                \right) \zeta_m' + q^{\lambda\lambda} \zeta_m''
                \\
                &=& \left(
                    \frac{1}{2} q^{AB} q^{C\lambda} - q^{AC} q^{B\lambda}
                \right) (\partial_C q_{AB}) \zeta_m'
                + q^{\lambda\lambda} \zeta_m''.
        \f}

        @b References

        [1] Andersson, Lars, Marc Mars, and Walter Simon. "Stability of
            marginally outer trapped surfaces and existence of marginally
            outer trapped tubes." arXiv preprint arXiv:0704.2889 (2007).
        """
        if compute_eigenfunctions and zeta is None:
            zeta = self.compute_zeta(num=num)
        if num is None:
            num = self.num
        m_max = min(num-1 if m_max is None else m_max, num-1)
        if m_terminate_index is not None:
            m_terminate_index = min(num-1, m_terminate_index)
        spectrum = StabilitySpectrum(rtol=rtol, zeta=zeta)
        with self.fix_evaluator():
            if self.extr_curvature is None:
                func = self._stability_eigenvalue_equation_timesym
            else:
                func = self._stability_eigenvalue_equation_general
            cache = dict()
            def _eig(m):
                solver = NDSolver(
                    eq=lambda pts: self._cached_stability_op(
                        func, pts, m=m, cache=cache,
                        slice_normal=slice_normal,
                        transform_torsion=transform_torsion,
                    ),
                    basis=ChebyBasis(domain=(0, np.pi), num=num, lobatto=False),
                )
                eigenfunctions = ()
                if compute_eigenfunctions:
                    eigenfunctions = solver.eigenfunctions()
                    eigenvals = [f.eigenvalue for f in eigenfunctions]
                else:
                    eigenvals, _ = solver.eigenvalues()
                return eigenvals, eigenfunctions
            for m in range(m_max+1):
                spectrum.add(m, *_eig(m))
                if not full_output: # we only want the principal eigenvalue
                    break
                if m > 0:
                    eigenvals = spectrum.get(l='all', m=m)
                    if all(spectrum.is_value_real(v) for v in eigenvals):
                        eigenfunctions = ()
                        if compute_eigenfunctions:
                            eigenfunctions = [
                                spectrum.get_eigenfunction(
                                    l=l, m=m, evaluator=False
                                )
                                for l in range(m, m+num)
                            ]
                        spectrum.add(-m, eigenvals, eigenfunctions)
                    else:
                        spectrum.add(-m, *_eig(-m))
                if (m_terminate_index is not None
                        and spectrum.get(l=m, m=m) > spectrum.get(l=m_terminate_index, m=0)):
                    break
        if not spectrum.is_real(l=0, m=0):
            raise RuntimeError("Non-real principal eigenvalue detected.")
        if full_output:
            return spectrum.principal.real, spectrum
        return spectrum.principal.real

    def _cached_stability_op(self, func, pts, m, cache=None, **kw):
        if cache and 'pts' in cache:
            op_id = cache['op_id']
            op_partial1 = cache['op_partial1']
            op_partial2 = cache['op_partial2']
            q_inv11 = cache['q_inv11']
            s_phi = cache['s^phi']
        else:
            (op_id, op_partial1, op_partial2), q_inv11, s_phi = func(pts, **kw)
            if cache is not None:
                cache['pts'] = pts
                cache['op_id'] = op_id
                cache['op_partial1'] = op_partial1
                cache['op_partial2'] = op_partial2
                cache['q_inv11'] = q_inv11
                cache['s^phi'] = s_phi
        op_id = [v + qi * m**2 for v, qi in zip(op_id, q_inv11)]
        if m != 0 and any(s_phi):
            op_id = [v + 2j * m * s_ph for v, s_ph in zip(op_id, s_phi)]
        return [op_id, op_partial1, op_partial2], 0.0

    def _stability_eigenvalue_equation_timesym(self, pts, slice_normal,
                                               transform_torsion):
        r"""Eigenvalue equation evaluator for time-symmetric case."""
        if not slice_normal:
            raise NotImplementedError(
                "Time-symmetric case only implemented for slice normal"
            )
        # Values for 3 derivative orders (0,1,2) per point.
        operator_values = np.zeros((len(pts), 3))
        q_inv11 = []
        for param, op in zip(pts, operator_values):
            stab_calc = StabilityCalc(
                curve=self, param=param, transform_torsion=transform_torsion
            )
            op += stab_calc.compute_op_timesym()
            q_inv11.append(stab_calc.q_inv[1,1])
        s_phi = [0.] * len(pts)
        return operator_values.T, q_inv11, s_phi

    def _stability_eigenvalue_equation_general(self, pts, slice_normal,
                                               transform_torsion):
        r"""Eigenvalue equation evaluator for the general case."""
        metric4 = FourMetric(self.metric)
        # Values for 3 derivative orders (0,1,2) per point.
        operator_values = np.zeros((len(pts), 3))
        q_inv11 = []
        s_phi = []
        for param, op in zip(pts, operator_values):
            stab_calc = StabilityCalc(
                curve=self, param=param, metric4=metric4,
                transform_torsion=transform_torsion
            )
            op += stab_calc.compute_op_general(slice_normal=slice_normal)
            q_inv11.append(stab_calc.q_inv[1,1])
            s_phi.append(stab_calc.torsion_vector[1])
        return operator_values.T, q_inv11, s_phi

    def shear(self, param, full_output=False):
        r"""Compute the shear of the MOTS at a point.

        See ..stabcalc.StabilityCalc.compute_shear() for more information.

        @param param
            Point on the MOTS to compute the shear at.
        @param full_output
            If `True`, return the shear (2,2)-tensor and its square
            \f$\sigma_{AB}\sigma^{AB}\f$. Default is `False`, i.e. only return
            the shear tensor.
        """
        metric4 = FourMetric(self.metric)
        stab_calc = StabilityCalc(curve=self, param=param, metric4=metric4)
        sigma = stab_calc.compute_shear()
        if full_output:
            sigma2 = stab_calc.compute_shear_squared()
            return sigma, sigma2
        return sigma

    def shear_square_integral(self, n="auto"):
        r"""Compute the integral of the shear squared.

        This computes the integral
        \f[
            \int_\mathcal{S} \sigma_{AB}\sigma^{AB}\ dV
                = 2\pi \int_0^\pi \sigma_{AB}\sigma^{AB} \sqrt{q}\ d\lambda\,.
        \f]

        @param n
            Order of the fixed quadrature integration (which is equal to the
            number of points at which the integrand is evaluated). The default
            ``"auto"`` will use twice the current curve's resolution, but at
            least 30 points.
        """
        if n == "auto":
            n = max(30, 2 * self.num)
        with self.fix_evaluator():
            area_element = self.get_area_integrand()
            def integrand(xs):
                return np.asarray([
                    area_element(x) * self.shear(x, full_output=True)[1]
                    for x in xs
                ])
            return 2*np.pi * _fixed_quad(
                integrand, a=0.0, b=np.pi, n=n
            )

    def shear_scalar(self, param, full_output=False):
        r"""Compute the complex shear scalar.

        @param param
            Point on the MOTS to compute the shear scalar at.
        @param full_output
            If `True`, return the shear scalar, the shear tensor, and the
            square of the shear. Default is `False`.

        @b Notes

        Given two orthonormal spacelike tangents \f$e_1\f$, \f$e_2\f$ to the
        MOTS, we define a null tetrad \f$(\ell, k, m, \bar m)\f$, where
        \f$m^\mu = \frac{1}{\sqrt{2}}(e_1 + i e_2)\f$.
        Here, \f$\ell^\mu, k^\mu\f$ are the outgoing and ingoing future
        pointing null normals to the MOTS, respectively, with the scaling
        condition \f$\ell^\mu k_\mu = -1\f$ (notice that this is different
        than our usual scaling condition, see also
        ..stabcalc.StabilityCalc.compute_shear()).
        Then, the shear scalar is defined as
        \f[
            \sigma_{(\ell)}
                = m^\mu m^\nu \nabla_\mu \ell_\nu
                = \frac{1}{2} \left(
                    \frac{\sigma_{\lambda\lambda}}{q_{\lambda\lambda}}
                    - \frac{\sigma_{\varphi\varphi}}{q_{\varphi\varphi}}
                \right)
                + i \frac{\sigma_{\lambda\varphi}}
                         {\sqrt{q_{\lambda\lambda}q_{\varphi\varphi}}}
                \,,
        \f]
        where we have used
        \f$e_1 = \frac{\partial_\lambda}{\sqrt{q_{\lambda\lambda}}}\f$
        and
        \f$e_2 = \frac{\partial_\varphi}{\sqrt{q_{\varphi\varphi}}}\f$.
        """
        metric4 = FourMetric(self.metric)
        stab_calc = StabilityCalc(curve=self, param=param, metric4=metric4)
        sigma_l = stab_calc.compute_shear_scalar()
        if full_output:
            sigma = stab_calc.compute_shear()
            sigma2 = stab_calc.compute_shear_squared()
            return sigma_l, sigma, sigma2
        return sigma_l

    def expand_shear_scalar(self, lmax="auto", zeta=None, min_lmax=64,
                            max_lmax=4096, coeff_threshold=None,
                            compress=True, full_output=False):
        r"""Expand the shear scalar into spin 2 weighted spherical harmonics.

        This uses the `spinsfast` module to expand the shear scalar (computed
        using shear_scalar()) into spherical harmonics of spin weight 2. To do
        this, the shear scalar is constructed as a function of \f$\theta\f$,
        where \f$\cos\theta=\zeta\f$ is the invariant angle defining a
        geometrically invariant parameterization of axisymmetric MOTSs.

        @param lmax
            Largest `ell` value in the expansion into \f${}_2Y_{\ell m}\f$.
            The default, ``"auto"`` will start with a low resolution and
            successively double this resolution until a "knee" is detected in
            the expansion coefficients. Such a knee indicates convergence and
            increasing the resolution further does not increase accuracy of
            results.
        @param zeta
            Expression evaluating the invariant angle as function of the
            intrinsic curve parameter \f$\lambda\f$. If not given, this
            function is numerically computed using compute_zeta().
        @param min_lmax,max_lmax
            If ``lmax="auto"``, start expanding with ``lmax=min_lmax``. If no
            "knee" is detected in the resulting coefficients, the resolution
            is doubled until a knee is detected or we have reached `max_lmax`.
        @param coeff_threshold
            If the final few coefficients are larger than `coeff_threshold`
            times the largest coefficient, then do not try to detect a knee in
            the coefficients but immediately increase the resolution. By
            default, knee detection is always run.
        @param compress
            Since the shear scalar is purely real for axisymmetric MOTSs, the
            only non-zero coefficients in the expansion will be the ones with
            `m=0`. Setting ``compress=True`` (default) will extract this one
            row of possibly non-vanishing coefficients and return it.
        @param full_output
            If `True`, return the coefficients, the (complex) shear scalar
            values at the grid points in \f$\theta=\arccos\zeta\f$ and the
            grid points themselves. Otherwise (default), return just the
            coefficients.
        """
        def _shear_scalar(params):
            with self.fix_evaluator():
                return [self.shear_scalar(param) for param in params]
        return self._expand_into_SWSPH(
            func=_shear_scalar, spinweight=2, lmax=lmax, min_lmax=min_lmax,
            max_lmax=max_lmax, coeff_threshold=coeff_threshold,
            compress=compress, zeta=zeta, full_output=full_output,
        )

    @classmethod
    def _cache_vectorized(cls, func, boundary_tol=0.0, boundary_val=0.0,
                          domain=(0.0, np.pi)):
        cache = dict()
        a, b = domain
        def cached_func(xs):
            to_compute = []
            for x in xs:
                if min(abs(x-a), abs(x-b)) < boundary_tol:
                    cache[x] = boundary_val
                elif x not in cache:
                    to_compute.append(x)
            if to_compute:
                values = func(to_compute)
                for x, val in zip(to_compute, values):
                    cache[x] = val
            return [cache[x] for x in xs]
        return cached_func

    def _expand_into_SWSPH(self, func, spinweight, lmax, min_lmax, max_lmax,
                           coeff_threshold, compress, zeta=None,
                           full_output=False):
        r"""Expand some quantity into SWSPH.

        The `func` has to be vectorized, i.e. take a sequence of parameter
        values and return a sequence of complex values of the same length.
        """
        # import on-demand to make dependency optional
        import spinsfast
        if zeta is None:
            zeta = self.compute_zeta()
        zetai = InverseExpression(zeta, domain=(-1.0, 1.0))
        zetai_ev = zetai.evaluator()
        def _func(ta_values):
            params = [zetai_ev(np.cos(ta)) for ta in ta_values]
            return func(params)
        cached_func = self._cache_vectorized(
            _func, boundary_tol=1e-14, boundary_val=0.0,
        )
        def _expand(lmax):
            Nta = 2*lmax+1
            Nph = 2*lmax+1
            ta_space = np.linspace(0.0, np.pi, num=Nta, endpoint=True)
            ta_data = cached_func(ta_space)
            data = np.zeros((Nta, Nph), dtype=np.complex)
            for i in range(len(ta_space)):
                data[i, :] = ta_data[i]
            alm = spinsfast.map2salm(data, spinweight, lmax)
            return alm, data[:, 0], ta_space
        if lmax == "auto":
            lmax = min_lmax
            while True:
                alm, values, ta_space = _expand(lmax=lmax)
                if lmax == max_lmax:
                    break
                al0r = _extract_al0(alm).real
                knee = detect_coeff_knee(al0r[2:], n_min=1, min_window=3)
                ok = True
                if coeff_threshold is not None:
                    ok = max(np.absolute(al0r[-5:])) < max(al0r)*coeff_threshold
                if ok and knee is not None:
                    break
                lmax = min(2*lmax, max_lmax)
        else:
            alm, values, ta_space = _expand(lmax=lmax)
        result = alm
        if compress:
            result = _extract_al0(alm)
        if full_output:
            return result, values, ta_space
        return result

    def compute_zeta(self, num=None, det_q_function=None):
        r"""Numerically compute the invariant angle zeta.

        For a "round" 2-sphere we have \f$\zeta = \cos\theta\f$. In general,
        it is defined in [1] via
        \f[
            \partial_A\zeta = \frac{1}{R^2} \epsilon_{BA} \phi^B, \qquad
            \int_{\mathcal{S}} \zeta\, d^2V = 0 \,,
        \f]
        where \f$\epsilon_{AB}\f$ is the volume 2-form on the surface
        \f$\mathcal{S}\f$ represented by this curve.
        Since our coordinates \f$(\lambda, \varphi)\f$ are already adapted to
        the axisymmetry, this simplifies to an ODE for \f$\zeta\f$, namely
        \f[
            \partial_\lambda = - \frac{\sqrt{\det q}}{R^2} \,,
        \f]
        where \f$R\f$ is the areal radius of \f$\mathcal{S}\f$.

        @return A exprs.numexpr.NumericExpression representing \f$\zeta\f$.

        @param num
            Spectral resolution of the numerical solution. The default is to
            take the curve's current resolution and at least `100`.
        @param det_q_function
            Optional callable evaluating \f$\det q\f$. Can be used to supply a
            cached version of `det_q`. If not supplied, uses an uncached
            version.

        @b References

        [1] Ashtekar, Abhay, et al. "Multipole moments of isolated horizons."
            Classical and Quantum Gravity 21.11 (2004): 2549.
        """
        if num is None:
            num = max(self.num, 100)
        with self.fix_evaluator():
            det_q = det_q_function
            if det_q is None:
                det_q = self.get_det_q_func()
            R2 = self.horizon_radius()**2
            def zeta_eq(pts):
                return (0, 1), [-np.sqrt(det_q(pt))/R2 for pt in pts]
            zeta = ndsolve(
                eq=zeta_eq,
                boundary_conditions=(
                    DirichletCondition(x=0, value=1),
                ),
                basis=CosineBasis(domain=(0., np.pi), num=num, lobatto=False),
            )
        return zeta

    def multipoles(self, min_n=0, max_n=10, num=None, det_q_function=None,
                   ricci_scalar_function=None, zeta=None, full_output=False,
                   disp=False, **kw):
        r"""Compute the multipoles I_n of the horizon.

        This is based on the considerations in [1].

        @param min_n,max_n
            Integers specifying the minimum and maximum number `n` (inclusive)
            up to which the moments \f$I_n\f$ should be computed. By default,
            we compute for ``n=0,...,10``.
        @param num
            Resolution for the pseudospectral solution of the invariant
            coordinate \f$\zeta\f$. Default is to use the current curve's
            resolution, but at least 100.
        @param det_q_function,ricci_scalar_function,zeta
            Optional callables evaluating \f$\det q\f$, the Ricci scalar, and
            the invariant angle \f$\zeta(\lambda)\f$, respectively. Can be
            used to supply cached or previously generated versions of these
            functions.
        @param full_output
            Whether to return an numutils.IntegrationResults object containing
            all values with their errors and any warnings. Default is `False`.
        @param disp
            Raise an error in case of an integration warning.
        @param **kw
            Additional keyword arguments are supplied to the
            `scipy.integrate.quad` call.

        @return A list of computed moments \f$I_n\f$ for `n=0,1,..,max_n`.

        @b References

        [1] Ashtekar, Abhay, et al. "Multipole moments of isolated horizons."
            Classical and Quantum Gravity 21.11 (2004): 2549.
        """
        get_zeta = kw.pop('get_zeta', False)
        if get_zeta:
            warnings.warn(
                "The `get_zeta` argument is deprecated. Use the "
                "`compute_zeta()` method instead.",
            )
        if num is None:
            num = max(self.num, 100)
        with self.fix_evaluator():
            det_q = det_q_function
            if det_q is None:
                det_q = self.get_det_q_func(cached=True)
            if zeta is None:
                zeta = self.compute_zeta(num=num, det_q_function=det_q)
            if get_zeta:
                return zeta
            zeta = zeta.evaluator()
            ricci_scal = ricci_scalar_function
            if ricci_scal is None:
                ricci_scal = self.get_ricci_scalar_func(cached=True)
            def integrand(n):
                def f(la):
                    zt = clip(zeta(la), -1, 1)
                    Scal = ricci_scal(la)
                    Yn = sph_harm(0, n, 0., np.arccos(zt)).real
                    return (2 * np.pi * np.sqrt(det_q(la)) * Scal * Yn)
                return f
            I_n = []
            for n in range(min_n, max_n+1):
                res = IntegrationResult(
                    *quad(integrand(n), a=0, b=np.pi, full_output=True, **kw),
                    mult=0.25
                )
                if disp and not res.is_ok():
                    raise IntegrationWarning(res.warning)
                I_n.append(res)
            if full_output:
                return IntegrationResults(*I_n, sum_makes_sense=False)
            return [res.value for res in I_n]

    def circumference(self, param):
        r"""Return the circumference of the surface at the given parameter."""
        pt = self(param, xyz=True)
        gyy = self.metric.at(pt).mat[1,1]
        return 2*np.pi * pt[0] * np.sqrt(gyy)

    def x_distance(self, param, **kw):
        r"""Return the proper length of a coordinate line from the curve to the z-axis."""
        x, z = self(param)
        def integrand(t):
            gxx = self.metric.at([t*x, 0.0, z]).mat[0,0]
            return np.sqrt(gxx)
        return x * quad(integrand, a=0, b=1, **kw)[0]

    def find_neck(self, algo='coord', xtol=1e-8, **kw):
        r"""Try to locate a 'neck' of this curve.

        The neck can be defined by various means. One working definition is to
        take a locally minimal proper circumference of the surface defined by
        this curve by rotating a point around the z-axis. The two poles lying
        on the z-axis are excluded, of course.

        This function implements the following algorithm to find the neck:
        Starting from the north-pole (``param=0``), we first find the point
        where the quantity (e.g. circumference) has a local *maximum*. From
        there, we do small steps forward to bracket the first local minimum,
        which is then located precisely.

        @return A tuple of ``(param, value)``, where ``param`` is the curve
            parameter of the neck and ``value`` the value of the quantity used
            to define the neck.

        @param algo
            The algorithm/definition of the neck. Understood values currently
            are ``'coord'`` (neck has minimal x-coordinate value),
            ``'circumference'`` (neck has minimal proper circumference),
            ``'proper_x_dist'`` (neck has minimal proper distance to z-axis,
            measured along a straight coordinate line in x-direction).
            Default is ``'coord'``.
        @param xtol
            Tolerance in curve parameter value for the search. Default is
            `1e-8`.
        @param **kw
            Further keyword arguments are used in case of
            ``algo==proper_x_dist`` and are supplied to the x_distance()
            method calls.
        """
        if algo == 'circumference':
            f = self.circumference
        elif algo in ('coord', 'proper_x_dist'):
            f = lambda param: self(param)[0]
        else:
            raise ValueError("Unknown algorithm: %s" % algo)
        last_param = [None]
        def func(param):
            param = clip(param, 0, np.pi)
            value = f(param)
            last_param[0] = param
            return value
        func_neg = lambda param: -f(clip(param, 0, np.pi))
        with self.fix_evaluator():
            xa, xb, xc = bracket(func_neg, 0.0, 0.1, grow_limit=2)[:3]
            res = minimize_scalar(func_neg, bracket=(xa, xb, xc), options=dict(xtol=1e-1))
            x_max = res.x
            try:
                xa, xb, xc = bracket(func, x_max, x_max+0.1, grow_limit=2)[:3]
                if xb < 0 or xc < 0:
                    # Something went wrong in bracketing. Do it half-manually now.
                    # This case occurs if the MOTS is *too little* deformed
                    # such that the "neck" is not very pronounced.
                    params = self.collocation_points()
                    xs = [f(x) for x in params]
                    max1 = next(params[i] for i, x in enumerate(xs)
                                if xs[i+1] < x)
                    max2 = next(params[i] for i, x in reversed(list(enumerate(xs)))
                                if xs[i-1] < x)
                    xa, xb, xc = bracket(func, max1, max1+0.1*(max2-max1), grow_limit=2)[:3]
            except RuntimeError:
                # This happens in bipolar coordinates in extremely distorted
                # cases where the "neck" is streched over a large parameter
                # interval.
                x0 = last_param[0]
                xa, xb, xc = bracket(func, x0, x0+0.1, grow_limit=5, maxiter=10000)[:3]
            if algo == 'proper_x_dist':
                func = lambda param: self.x_distance(param, **kw)
                xa, xb, xc = bracket(func, xb, xb+1e-2, grow_limit=2)[:3]
                res = minimize_scalar(func, bracket=(xa, xb, xc), options=dict(xtol=xtol))
            else:
                res = minimize_scalar(func, bracket=(xa, xb, xc), options=dict(xtol=xtol))
            return res.x, res.fun

    def locate_intersection(self, other_curve, xtol=1e-8, domain1=(0, np.pi),
                            domain2=(0, np.pi), strict1=True, strict2=True,
                            N1=20, N2=20):
        r"""Locate one point at which this curve intersects another curve.

        @return Tuple ``(param1, param2)``, where ``param1`` is the parameter
            value of this curve and ``param2`` the parameter value of the
            `other_curve` at which the two curves have the same location. If
            no intersection is found, returns ``(None, None)``.

        @param other_curve
            Curve to find the intersection with.
        @param xtol
            Tolerance in curve parameter values for the search. Default is
            `1e-8`.
        @param domain1,domain2
            Optional interval of the curves to consider. Default is the full
            curve, i.e. ``(0, pi)`` for both.
        @param strict1,strict2
            Whether to only allow solutions in the given domains `domain1`,
            `domain2`, respectively (default). If either is `False`, the
            respective domain is used just for the initial coarse check to
            find a starting point. Setting e.g. ``N1=1`` and ``strict1=False``
            allows specifying a starting point on this (first) curve.
        @param N1,N2
            Number of equally spaced rough samples to check for a good
            starting point. To avoid running into some local minimal distance
            (e.g. at the curve ends), this number should be high enough.
            Alternatively (or additionally), one may specify a smaller
            domain if there is prior knowledge about the curve shapes.
        """
        z_dist = self.z_distance_using_metric(
            metric=None, other_curve=other_curve, allow_intersection=True,
        )
        if z_dist >= 0:
            return (None, None)
        c1 = self
        c2 = other_curve
        with c1.fix_evaluator(), c2.fix_evaluator():
            space1 = np.linspace(*domain1, N1, endpoint=False)
            space1 += (space1[1] - space1[0]) / 2.0
            space2 = np.linspace(*domain2, N2, endpoint=False)
            space2 += (space2[1] - space2[0]) / 2.0
            pts1 = [[c1(la), la] for la in space1]
            pts2 = [[c2(la), la] for la in space2]
            dists = [[np.linalg.norm(np.asarray(p1)-p2), l1, l2]
                     for p1, l1 in pts1 for p2, l2 in pts2]
            _, la1, la2 = min(dists, key=lambda d: d[0])
            dyn_domain1 = domain1 if strict1 else (0.0, np.pi)
            dyn_domain2 = domain2 if strict2 else (0.0, np.pi)
            def func(x):
                la1, la2 = x
                p1 = np.asarray(c1(clip(la1, *dyn_domain1)))
                p2 = np.asarray(c2(clip(la2, *dyn_domain2)))
                return p2 - p1
            sol = root(func, x0=[la1, la2], tol=xtol)
            if not sol.success:
                return (None, None)
            la1, la2 = sol.x
            if strict1:
                la1 = clip(la1, *domain1)
            if strict2:
                la2 = clip(la2, *domain2)
            if np.linalg.norm(func((la1, la2))) > np.sqrt(xtol):
                # points are too far apart to be an intersection
                return (None, None)
            return (la1, la2)

    def locate_self_intersection(self, neck=None, xtol=1e-8):
        r"""Locate a *loop* in the MOTS around its neck.

        @return Two parameter values ``(param1, param2)`` where the curve has
            the same location in the x-z-plane. If not loop is found, returns
            ``(None, None)``.

        @param neck
            Parameter where the neck is located. If not given, finds the neck
            using default arguments of find_neck().
        @param xtol
            Tolerance in curve parameter values for the search. Default is
            `1e-8`.
        """
        try:
            return self._locate_self_intersection(neck=neck, xtol=xtol)
        except _IntersectionDetectionError:
            raise
            pass
        return None, None

    def _locate_self_intersection(self, neck, xtol):
        r"""Implements locate_self_intersection()."""
        if neck is None:
            neck = self.find_neck()[0]
        with self.fix_evaluator():
            match_cache = dict()
            def find_matching_param(la1, max_step=None, _recurse=2):
                try:
                    return match_cache[la1]
                except KeyError:
                    pass
                if abs(la1-neck) < xtol:
                    return la1
                x1 = self(la1)[0]
                def f(la): # x-coord difference
                    return self(la)[0] - x1
                dl = neck - la1
                if max_step:
                    dl = min(dl, max_step)
                a = neck + dl
                while f(a) > 0:
                    dl = dl/2.0
                    a = neck + dl
                    if abs(dl) < xtol:
                        return la1
                step = min(dl/2.0, max_step) if max_step else (dl/2.0)
                b = a + step
                while f(b) < 0:
                    a, b = b, b+step
                    if b >= np.pi:
                        if _recurse > 0:
                            max_step = (max_step/10.0) if max_step else 0.05
                            return find_matching_param(la1, max_step=max_step,
                                                       _recurse=_recurse-1)
                        # Last resort: seems the x-coordinate is not reached
                        # on this branch at all. Returning pi guarantees a
                        # large `delta_z` (albeit with a jump that might be
                        # problematic). In practice, however, this often works.
                        return np.pi
                # We now have a <= la2 <= b (la2 == matching param)
                la2 = brentq(f, a=a, b=b, xtol=xtol)
                match_cache[la1] = la2
                return la2
            def delta_z(la):
                la2 = find_matching_param(la)
                return self(la)[1] - self(la2)[1]
            for step in np.linspace(0.01, np.pi-neck, 100):
                # Note that `step>0`, but the minimum lies at `a<neck` the way
                # we have defined `delta_z()`. The minimize call will however
                # happily turn around...
                res = minimize_scalar(delta_z, bracket=(neck, neck+step),
                                      options=dict(xtol=1e-2))
                a = res.x
                if delta_z(a) < 0:
                    break
            if delta_z(a) >= 0:
                raise _IntersectionDetectionError("probably no intersection")
            # We start with `shrink=2.0` to make this fully compatible with
            # the original strategy (i.e. so that results are reproducible).
            # In practice, however, this `step` might be larger than the
            # distance to the domain boundaries, making the following
            # `brent()` call stop immediately at the wrong point. The cause is
            # that, first of all we assume here `a>neck`, but our definition
            # of `delta_z()` leads to a minimum at `a<neck`. Secondly, with
            # extreme parameterizations (e.g. using bipolar coordinates plus
            # `curv2` coordinate space reparameterization), the `neck` region
            # could house the vast majority of affine parameters.
            step_shrinks = [2.0]
            step_shrinks.extend(np.linspace(
                (neck-a)/((1-0.9)*a), max(10.0, (neck-a)/((1-0.1)*a)), 10
            ))
            # For the above:
            #            a + step = x*a,                 where 0<x<1, step<0
            # <=> (a-neck)/shrink = (x-1)*a
            # <=>          shrink = (a-neck)/(a*(x-1))
            #                     = (neck-a)/(a*(1-x))   > 0
            for shrink in step_shrinks:
                step = (a-neck)/shrink
                b = a + step
                while delta_z(b) < 0:
                    a, b = b, b+step
                    if not 0 <= b <= np.pi:
                        raise _IntersectionDetectionError("curve upside down...")
                la1 = brentq(delta_z, a=a, b=b, xtol=xtol)
                if la1 > 0.0:
                    break
            la2 = find_matching_param(la1)
            return la1, la2

    def get_distance_function(self, other_curve, Ns=None, xatol=1e-12,
                              mp_finish=True, dps=50, minima_to_check=1):
        r"""Return a callable computing the distance between this and another curve.

        The computed distance is the coordinate distance between a point of
        this curve to the closest point on another curve.

        The returned callable will take one mandatory argument: the parameter
        at which to evaluate the current curve. The resulting point is then
        taken and the given `other_curve` searched for the point closest to
        that point. The distance to this function is then returned.

        A second optional parameter of the returned function determines
        whether only the distance (`False`, default) or the distance and the
        parameter on the other curve is returned (if `True`).

        @param other_curve
            The curve to which the distance should be computed in the returned
            function.
        @param Ns
            Number of points to take on `other_curve` for finding the initial
            guess for the minimum search. Default is to take the resolution of
            `other_curve`.
        @param xatol
            In case ``mp_finish==False``, use this tolerance in the SciPy
            `minimize_scalar()` call. Default is `1e-12`.
        @param mp_finish
            If `True` (default), evaluate some terms using arbitrary precision
            mpmath arithmetics and find the minimum with a robust but slow
            golden section search. This allows finding minima *much* closer to
            the actual minimum. In general, with ``mp_finish==False``, we are
            usually limited by approximately `1e-8`, while with
            ``mp_finish==True``, we can get to around `1e-15` (e.g. if
            `other_curve` is this curve).
        @param dps
            Decimal places for mpmath computations. Default is `50`.
        @param minima_to_check
            Number of local minima to check for a global minimum. Default is
            `1`, which means we search just around the smallest distance from
            the initial grid of points. Higher values may help locate the
            correct minimum especially in cases of self-intersecting curves.
        """
        if Ns is None:
            Ns = other_curve.num
        params = other_curve.collocation_points(num=Ns)
        with other_curve.fix_evaluator():
            pts_other = np.array([other_curve(la) for la in params])
        fl = mp.mpf if mp_finish else float
        pi = mp.pi if mp_finish else np.pi
        def _distance(param, full_output=False):
            with mp.workdps(dps), other_curve.fix_evaluator():
                pt = np.asarray(self(param))
                dists = np.linalg.norm(pts_other - pt, axis=1)
                indices = _discrete_local_minima(dists, minima_to_check)
                if mp_finish:
                    x1, z1 = fl(pt[0]), fl(pt[1])
                    def _f(la):
                        pt_other = other_curve(float(la))
                        x2, z2 = fl(pt_other[0]), fl(pt_other[1])
                        return float(mp.sqrt((x2-x1)**2 + (z2-z1)**2))
                else:
                    def _f(la):
                        return np.linalg.norm(pt-other_curve(la))
                results = [
                    _search_local_minimum(
                        _f, use_mp=mp_finish,
                        a=fl(params[idx-1]) if idx > 0 else fl(0),
                        b=fl(params[idx+1]) if idx+1 < Ns else pi,
                        xatol=xatol, full_output=True,
                    )
                    for idx in indices
                ]
                results.sort(key=lambda res: res[0])
                return results[0] if full_output else results[0][0]
        _distance.domain = 0, np.pi
        return _distance

    def plot_expansion(self, c=0, points=500, name=r"\gamma", figsize=(5, 3),
                       verbose=True, ingoing=False, **kw):
        r"""Convenience function to plot the expansion along the curve.

        This may help in quickly judging convergence. The parameter `c` is
        subtracted from the values, which is useful to analyse constant
        expansion surfaces.
        """
        from ...ipyutils import plot_data
        if isiterable(points):
            pts = points
        else:
            pts = np.linspace(0, np.pi, points+1, endpoint=False)[1:]
        values = np.asarray(self.expansions(pts, ingoing=ingoing)) - c
        if verbose:
            max_vio = np.max(np.absolute(values))
            print("Max violation at given points: %s" % max_vio)
        plot_data(
            pts, values, figsize=figsize,
            **insert_missing(
                kw, ylog=True, absolute=True, xlabel=r"$\lambda$",
                ylabel=r"$|\Theta|$",
                title=(r"Expansion%s along $%s$"
                       % (" (- $%s$)" % c if c else "", name))
            )
        )

    def plot_coeffs(self, name=r"\gamma", figsize=(5, 3), **kw):
        r"""Convenience function to plot the horizon function's spectral coefficients.

        This may help in quickly judging convergence based on exponential
        decay.
        """
        from ...ipyutils import plot_data
        return plot_data(
            range(self.num), self.h.a_n, figsize=figsize,
            **insert_missing(
                kw, absolute=True, ylog=True, xlabel='$n$', ylabel='$|a_n|$',
                title=r"Coefficients of horizon function of $%s$" % name
            )
        )


class TimeVectorData():
    r"""MOTT time evolution vector.

    The time evolution vector `V` represented as linear combination of the
    future pointing ingoing and outgoing null normals, `k` and `l`, to the
    MOTS. We currently support two versions of this vector,
    \f[
        V^\mu = \ell^\mu + c k^\mu
    \f]
    and
    \f[
        \tilde V^\mu = \cos(\chi)\,\ell^\mu  + \sin(\chi)\,k^\mu \,.
    \f]
    If `V` is parallel to `k`, the coefficient `c` diverges. Hence, the second
    version, \f$\tilde V\f$ (`V_tilde` in code) is better behaved near these
    points. Since both versions can be computed from the underlying data we
    store, you can freely switch between these using a single object.

    Note that if you intend to interpolate `c` across a MOTS, it is better to
    interpolate the better behaved `chi` and convert the interpolated values
    to the corresponding `c` values using the class method c_from_chi().

    Internally, the time evolution vector is built using a vector `tau` that
    is tangent to the MOTT but not necessarily orthogonal to the MOTS. Its
    projection onto the plane of normals (spanned by `k` and `l`) is used to
    compute an approximation of `V`. You can use the error() method to
    determine how close `tau` is to `V` in terms of its coordinate
    representation. If the error is large, then `V` might not be tangent to
    the MOTT. In our representation, we have
    \f[
        \tau^\mu = a V^\mu = \tilde a \tilde V^\mu
    \f]
    and hence \f$a = - \tau \cdot k\f$,
    \f$\tilde a = \sqrt{(\tau\cdot\ell)^2 + (\tau\cdot k)^2}\f$,
    \f$\tan\chi = \tau\cdot\ell/\tau \cdot k = c\f$.

    Another representation of the time evolution vector is given in [1]. It
    reads
    \f[
        \mathcal{V}^\mu = \alpha\ (n^\mu + v_\bot \nu^\mu)
            = \frac{\alpha}{\sqrt{2}}(1 + v_\bot) \ell^\mu
                + \frac{\alpha}{\sqrt{2}}(1 - v_\bot) k^\mu
            =: b_B \ell^\mu + c_B k^\mu
        \,,
    \f]
    where \f$\alpha\f$ is the lapse function, \f$n^\mu\f$ the timelike normal
    on the Cauchy slice, \f$\nu^\mu\f$ the normal of the MOTS within the slice
    and \f$v_\bot\f$ the velocity of the horizon relative to the foliation.
    In the code, \f$\mathcal{V}^\mu\f$ is accessed through the `V_B` attribute
    and \f$b_B\f$ and \f$c_B\f$ through the `b_B` and `c_B` attributes,
    respectively.

    Note that by its definition, `V_B` has a time component of 1 and is hence
    equal to `tau`, except for any residual numerical error.

    @b References

    [1] Booth, Ivan, and Stephen Fairhurst. "Extremality conditions for
        isolated and dynamical horizons." Physical review D 77.8 (2008):
        084005.
    """

    __slots__ = ("l", "k", "tau", "tau_cov", "eps", "g4", "param")

    def __init__(self, l, k, tau, eps, g4, param):
        r"""Create a new time vector object.

        @param l,k
            The future pointing outgoing and ingoing null normals to the MOTS,
            respectively.
        @param tau
            Candidate for a time evolution vector. This should be tangent to
            the MOTT but not necessarily orthogonal to the MOTS. It is used to
            compute the error and its projection onto the (`l`,`k`) plane is
            used to construct the actual time evolution vector.
        @param eps
            Tilting parameter producing the given `tau` vector. Used by
            ExpansionCurve.time_evolution_vector().
        @param g4
            4x4 matrix representing the spacetime 4-metric at the point of the
            curve.
        @param param
            Parameter along the curve representing the point at which to
            evaluate the vector.
        """
        ## Outgoing null normal scaled to `l*k = -1`.
        self.l = l
        ## Ingoing null normal scaled to `l*k = -1`.
        self.k = k
        ## 4-vector close to the time evolution vector.\ Scaled to have unit
        ## time component.
        self.tau = tau
        ## Tilting parameter producing the given `tau` vector.
        self.eps = eps
        ## Spacetime 4-metric at the current point.
        self.g4 = g4
        ## Curve parameter (w.r.t.\ intrinsic parametrization).
        self.param = param
        ## Covariant components of `tau`.
        self.tau_cov = g4.dot(tau)

    @classmethod
    def c_from_chi(cls, chi):
        r"""Static function to convert a mixing angle `chi` to `c`."""
        return np.tan(chi)

    @property
    def c(self):
        r"""Value of `c` (see class docstring)."""
        return np.tan(self.chi)

    @property
    def a(self):
        r"""Value of `a` (see class docstring)."""
        return -self.tau_k

    @property
    def chi(self):
        r"""Value of `chi` (see class docstring)."""
        return np.arctan2(-self.tau_l, -self.tau_k)

    @property
    def tau_l(self):
        r"""Value of \f$\tau^\mu\ell_\mu\f$."""
        return self.tau_cov.dot(self.l)

    @property
    def tau_k(self):
        r"""Value of \f$\tau^\mu k_\mu\f$."""
        return self.tau_cov.dot(self.k)

    @property
    def l_cov(self):
        r"""Covariant components of the outgoing null normal."""
        return self.g4.dot(self.l)

    @property
    def k_cov(self):
        r"""Covariant components of the ingoing null normal."""
        return self.g4.dot(self.k)

    @property
    def V(self):
        r"""Time evolution vector `V` (see class docstring)."""
        return self.l + self.c * self.k

    @property
    def V_tilde(self):
        r"""Time evolution vector \f$\tilde V\f$ (see class docstring)."""
        chi = self.chi
        return np.cos(chi) * self.l + np.sin(chi) * self.k

    @property
    def a_tilde(self):
        r"""Value of \f$\tilde a\f$ (see class docstring)."""
        return np.sqrt(self.tau_l**2 + self.tau_k**2)

    @property
    def V_B(self):
        r"""Value of \f$\mathcal{V}\f$ (see class docstring)."""
        return self.b_B * self.l + self.c_B * self.k

    @property
    def b_B(self):
        r"""Value of the coefficient `b_B` of `V_B`."""
        return - self.tau_cov.dot(self.k)

    @property
    def c_B(self):
        r"""Value of the coefficient `c_B` of `V_B`."""
        return - self.tau_cov.dot(self.l)

    @property
    def error(self):
        r"""Difference between `V_tilde` and the time evolution vector not
        perpendicular to the MOTS.

        The time evolution vector `tau` is tangent to the MOTT but not
        necessarily normal on the MOTS. The vector `V_tilde` is `tau`
        projected onto the plane of normals to the MOTS and scaled by some
        constant. The `error` is the sum of the squares of coordinate
        difference introduced by this projection. Ideally, `tau` is already
        normal to the MOTS making this projection the identity transformation
        (and this error zero).
        """
        a_tilde = self.a_tilde
        V_tilde = self.V_tilde
        tau = self.tau
        return math.fsum(x**2 for x in a_tilde*V_tilde-tau)

    def __str__(self):
        r"""Detailed string representation of this vector."""
        c = self.c
        chi = self.chi
        s = ("time evolution vector at param = {param}:\n"
             "  V       = ell {op} {c} k\n"
             "          = {V}\n"
             "  V_tilde = cos({chi:.2f}) ell + sin({chi:.2f}) k\n"
             "          = {V_tilde}\n"
             "  error = {err:.1e}; eps = {eps}").format(
                 op="-" if c < 0 else "+", c=abs(c), chi=chi,
                 V=self.V, V_tilde=self.V_tilde, err=self.error,
                 param=self.param, eps=self.eps
             )
        return s


class SignatureQuantities():
    r"""Class storing quantities allowing visualization of signature."""

    __slots__ = ("param", "l", "k", "tau", "l_cov", "k_cov", "tau_cov",
                 "tau_l", "tau_k", "future_it", "past_it")

    DEPRECATION_WARNING_PRINTED = True # turn off printing the warning

    def __init__(self, param, g4, l, k, tau):
        r"""Set up an object holding signature data at a point of a MOTS.

        @param param
            The parameter along the MOTS at which the signature is computed.
        @param g4
            The 4x4 matrix representing the 4-metric at the point of the MOTS.
        @param l,k
            The outgoing and ingoing null normals to the MOTS, respectively.
        @param tau
            Any timelike tangent vector along the MOTS tube.
        """
        self.param = param
        self.l = l
        self.k = k
        self.tau = tau
        self.k_cov = g4.dot(k)
        self.l_cov = g4.dot(l)
        self.tau_cov = g4.dot(tau)
        self.tau_l = self.tau_cov.dot(l)
        self.tau_k = self.tau_cov.dot(k)
        self.future_it = None
        self.past_it = None

    def signature(self):
        r"""Return one of ``{-1,0,1}`` representing the signature.

        The values `-1, 0, 1` indicate timelike, null, and spacelike
        signature, respectively.
        """
        f_sig = self.f_sig()
        if f_sig > 0.0:
            # normal is spacelike => 3-surface is timelike
            return -1
        if f_sig < 0.0:
            # normal is timelike => 3-surface is spacelike
            return 1
        # normal is null => 3-surface is null
        return 0

    def f_sig(self):
        r"""Return the value of `f_sig`.

        See the documentation of ExpansionCurve.signature_quantities() for
        more information.
        """
        return self.tau_l * self.tau_k

    def __setstate__(self, state):
        r"""Restore this object from the given unpickled state."""
        self.future_it = self.past_it = None
        state = state[1]
        for slot in state:
            setattr(self, slot, state[slot])
        if self.future_it is not None or self.past_it is not None:
            self._warn_deprecated()

    @classmethod
    def _warn_deprecated(cls):
        r"""Warn that depcretated data has been loaded into this object."""
        if not cls.DEPRECATION_WARNING_PRINTED:
            print("WARNING: Loaded deprecated signature data from an old "
                  "version.")
            cls.DEPRECATION_WARNING_PRINTED = True


def _golden_section_search(f, a, b, tol, use_mp=False):
    r"""Trivial golden section search for locating a local minimum.

    The function `f` should have a local minimum between `a` and `b`.
    """
    if tol <= 0.0:
        raise ValueError("Tolerance must be > 0.")
    fl = mp.mpf if use_mp else float
    sqrt = mp.sqrt if use_mp else np.sqrt
    golden = (1 + sqrt(5)) / 2
    a = fl(a)
    b = fl(b)
    c = b - (b - a) / golden
    d = a + (b - a) / golden
    while abs(c-d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / golden
        d = a + (b - a) / golden
    return (b + a) / 2


def _discrete_local_minima(values, max_count=1):
    r"""Find the indices of N non-consecutive smallest values.

    @b Examples

    ```
        f = lambda x: np.cos(10*x) * np.exp(-x/2) + x/3
        dists = list(map(f, np.linspace(0, np.pi, 100)))
        indices = _discrete_local_minima(dists, 3)
        print("Minima at indices: %s" % (indices,))
        ax = plot_data(dists, l='.', show=False)
        ax.plot(indices, [dists[i] for i in indices], 'or')
    ```
    """
    if max_count <= 0:
        raise ValueError("Cannot find less than one local minimum.")
    indices = sorted([i for i in range(len(values))], key=lambda i: values[i])
    islands = []
    for i in indices:
        new = True
        for island in islands:
            if i+1 in island or i-1 in island:
                new = False
                island.append(i)
        if new:
            islands.append([i])
        if len(islands) >= max_count:
            break
    return [island[0] for island in islands]


def _search_local_minimum(func, a, b, use_mp=False, full_output=False,
                          xatol=1e-12):
    r"""Search for a local minimum of `func` between `a` and `b`."""
    if use_mp:
        x0 = _golden_section_search(
            f=func, a=a, b=b, use_mp=True,
            tol=sys.float_info.epsilon
        )
        return (func(x0), x0) if full_output else func(x0)
    res = minimize_scalar(
        func, bounds=[a, b], method='bounded',
        options=dict(xatol=xatol),
    )
    if not res.success:
        raise DistanceSearchError(res.message)
    return (res.fun, res.x) if full_output else res.fun


def _cached(fun):
    r"""Simple wrapper caching function evaluations using a dictionary.

    @param fun
        Function whose evaluations should be cached. Must take one parameter.
    """
    cache = dict()
    def wrapper(x):
        try:
            return cache[x]
        except KeyError:
            val = fun(x)
            cache[x] = val
            return val
    return wrapper


def _lm_ind(l, m):
    r"""Return the 1-D index corresponding to the given (l,m) indices."""
    return l*(l+1) + m


def _lmax_from_Nlm(Nlm):
    r"""Return the largest ell index from the total number of coefficients."""
    return int(round(np.sqrt(Nlm) - 1))


def _lmax_from_al0(al0):
    r"""Return the largest ell index for a given `a_l0` array."""
    return len(al0) - 1


def _extract_al0(alm):
    r"""Extract the `a_l0` coefficients of a given expansion."""
    lmax = _lmax_from_Nlm(len(alm))
    al0 = np.asarray([alm[_lm_ind(ell, 0)] for ell in range(lmax+1)])
    return al0


def _construct_alm_from_al0(al0):
    r"""Reconstruct the full `a_lm` coefficients from the `a_l0` ones."""
    lmax = _lmax_from_al0(al0)
    Nlm = lmax*(lmax+2) + 1
    alm = np.zeros(Nlm, dtype=np.complex)
    for ell, value in enumerate(al0):
        alm[_lm_ind(ell, 0)] = value
    return alm


def alm_matrix(alm):
    r"""Convert a sequence of a_lm coefficients into matrix form.

    The `alm` sequence returned by e.g. ExpansionCurve.expand_shear_scalar()
    (when ``compress=False`` is set) is a flattened (1D) sequence of values
    that represent all the values \f$a_{lm}\f$, where `l = 0, ..., l_max` and
    `m = -l, ..., +l`. This function takes the flattened sequence `alm` and
    returns the complex valued matrix of `a[l,m]` coefficients.

    Note that this is a pure Python implementation not suitable for large
    matrices.
    """
    lmax = _lmax_from_Nlm(len(alm))
    alm_mat = np.zeros((lmax+1, 2*lmax+1), dtype=np.complex)
    for ell in range(lmax+1):
        for m in range(-ell, ell+1):
            alm_mat[ell, lmax+m] = alm[_lm_ind(ell, m)]
    return alm_mat


def evaluate_al0(al0, num=None, spinweight=2, real=True):
    r"""Evaluate the expanded function at a set of points.

    Given a list of coefficients `a_l0` of a function expanded into spin
    weighted spherical harmonics, synthesize the function on a grid of `num`
    points.

    @return ``xs, ys``, where `xs` is an array of theta values at which the
        function has been evaluated and `ys` the corresponding function
        values.

    @param al0
        Coefficients `a_l0`, where `l = 0, ..., lmax`.
    @param num
        Number of theta-values at which to evaluate the function. The default
        is to use the points that can be used to expand the function into
        exactly the same `al0` coefficients.
    @param spinweight
        The spin weight of the functions expanded into. Default is `2`.
    @param real
        Whether to drop any imaginary part and return an array of real values.
        Default is `True`, which is suitable if the expanded function had
        vanishing imaginary part in the first place.
    """
    import spinsfast
    lmax = len(al0) - 1
    Nta = 2*lmax+1
    Nph = 2*lmax+1
    alm = _construct_alm_from_al0(al0)
    if num is None:
        num = Nta
    xs = np.linspace(0.0, np.pi, num=Nta, endpoint=True)
    ys = spinsfast.salm2map(alm, spinweight, lmax, num, Nph)
    ys = ys[:, 0]
    if real:
        ys = ys.real
    return xs, ys


def _get_fixed_quad_abscissas(a, b, n):
    r"""Return the abscissas of the Gaussian quadrature of order `n`.

    These are the exact points at which an integrand will be evaluated by
    `scipy.integrate.fixed_quad(..., a, b, n)`. Note that this will hold for
    _fixed_quad() *only* if no `full_domain` is given (or it is equal to the
    integration interval).
    """
    xs_list = [None]
    def f(x):
        xs_list[0] = x
        return np.zeros_like(x)
    fixed_quad(f, a=a, b=b, n=n)
    return xs_list[0]


def _fixed_quad(func, a, b, n, full_domain=None, min_n=30):
    r"""Integrate a function using fixed order Gaussian quadrature.

    This is a wrapper for `scipy.integrate.fixed_quad()`. Given a full domain
    to integrate, it uses the information of the current sub-domain to choose
    less than the full set of `n` points such that integrating the full domain
    in several intervals, the total number of evaluations is approximately
    `n`.

    @param func
        Function taking a list of values and returning a list of results.
    @param a,b
        Interval to integrate over.
    @param n
        Order of the Gaussian quadrature for the `full_domain`.
    @param full_domain
        Full domain to eventually integrate over. If not given, `a,b` is taken
        to be the full domain, i.e. no reduction of quadrature order is done.
    @param min_n
        Minimum quadrature order for very small sub-domains.
    """
    if full_domain is not None:
        w = full_domain[1] - full_domain[0]
        if abs(b-a) < 1e-14*w:
            return 0.0
        n = max(min_n, int(round(n * abs(b-a)/w)))
    value, _ = fixed_quad(func, a=a, b=b, n=n)
    return value


class DistanceSearchError(Exception):
    r"""Raised when the closest distance between curves could not be found."""
    pass


class TimeEvolutionVectorSearchError(Exception):
    r"""Raised when the time evolution vector could not be represented via k and l."""
    pass


class _IntersectionDetectionError(Exception):
    r"""Raised when self-intersection cannot be located."""
    pass


class IntegrationError(Exception):
    r"""Raised for non-converging integrals or if accuracy cannot be reached."""
    pass
