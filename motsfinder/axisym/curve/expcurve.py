r"""@package motsfinder.axisym.curve.expcurve

Base class for curves that can calculate their expansion in the slice.


@b Examples

See the implemented subclasses starshapedcurve.StarShapedCurve and
refparamcurve.RefParamCurve for examples.
"""

from abc import abstractmethod
from contextlib import contextmanager
from math import fsum

import numpy as np
from scipy import linalg
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, bracket
from scipy.special import sph_harm

from ...utils import insert_missing, isiterable, parallel_compute
from ...numutils import clip
from ...metric import FlatThreeMetric, FourMetric
from ...ndsolve import ndsolve, NDSolver, CosineBasis
from ...ndsolve import DirichletCondition
from .basecurve import BaseCurve
from .parametriccurve import ParametricCurve


__all__ = []


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

    def z_distance(self, other_curve=None, atol=1e-12, rtol=1e-12, limit=100,
                   allow_intersection=False, full_output=False):
        return self.z_distance_using_metric(
            metric=self.metric, other_curve=other_curve, atol=atol, rtol=rtol,
            limit=limit, allow_intersection=allow_intersection,
            full_output=full_output
        )

    def inner_z_distance(self, other_curve, where='top', **kw):
        r"""Compute the z-distance of two points of this and another curve.

        In contrast to z_distance(), this method does *not* compute how close
        two surfaces approach each other (which would return zero for
        intersecting surfaces). Instead, it computes one possible measure for
        how close the two surfaces are from being identical, i.e. it computes
        the distance of two corresponding points.

        In this function, we take either the top or bottom points of both
        surfaces on the z-axis and compute their distance.

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
        @param **kw
            Further keyword arguments are passed to arc_length_using_metric().
        """
        metric = kw.pop('metric', self.metric)
        if where not in ('top', 'bottom'):
            raise ValueError('Unknown location: %s' % where)
        t = 0 if where == 'top' else np.pi
        line = ParametricCurve.create_line_segment(self(t), other_curve(t))
        return line.arc_length_using_metric(metric=metric, **kw)

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
    def copy(self):
        r"""Create an independent copy of this curve."""
        pass

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

    def signature_quantities(self, pts, past_curve=None, future_curve=None):
        if past_curve is None and future_curve is None:
            raise TypeError("Either a past or future curve (or both) must "
                            "be specified.")
        delta_time = None
        if (future_curve and past_curve
                and future_curve.metric.time < past_curve.metric.time):
            future_curve, past_curve = past_curve, future_curve
        if future_curve:
            delta_time = future_curve.metric.time - self.metric.time
        if past_curve:
            dt = self.metric.time - past_curve.metric.time
            if delta_time and abs(dt-delta_time) > 1e-8:
                raise ValueError("Future and past curves different time "
                                 "distances away")
            delta_time = dt
        if delta_time <= 0.0:
            raise ValueError("Curves not in increasing coordinate time.")
        future_it = future_curve.metric.iteration if future_curve else None
        past_it = past_curve.metric.iteration if past_curve else None
        with self.fix_evaluator(), \
                (past_curve or self).fix_evaluator(), \
                (future_curve or self).fix_evaluator():
            metric4 = FourMetric(self.metric)
            results = []
            for pt in pts:
                calc = self.get_calc_obj(pt)
                point = calc.point
                g4 = metric4.at(point).mat
                n = metric4.normal(point)
                g3_inv = calc.g.inv
                nu3_cov = calc.covariant_normal(diff=0)
                nu3 = g3_inv.dot(nu3_cov)
                nu = np.zeros((4,))
                nu[1:] = nu3
                k = n - nu
                l = n + nu
                if past_curve and future_curve:
                    xfuture = future_curve(pt, xyz=True)
                    xpast = past_curve(pt, xyz=True)
                    xdot = (xfuture - xpast) / (2*delta_time)
                elif future_curve:
                    xfuture = future_curve(pt, xyz=True)
                    xdot = (xfuture - point) / delta_time
                else:
                    xpast = past_curve(pt, xyz=True)
                    xdot = (point - xpast) / delta_time
                tau = np.ones((4,))
                tau[1:] = xdot
                results.append(SignatureQuantities(pt, g4, l, k, tau,
                                                   future_it=future_it,
                                                   past_it=past_it))
            return results

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

    def area(self, full_output=False, disp=False, **kw):
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
            result = quad(
                self.get_area_integrand(), a=0, b=np.pi, full_output=True,
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

    def get_det_q_func(self):
        r"""Create a function to evaluate det(q)."""
        def det_q(param):
            calc = self.get_calc_obj(param)
            q = calc.induced_metric()
            return linalg.det(q)
        return det_q

    def get_area_integrand(self):
        det_q = self.get_det_q_func()
        integrand = lambda t: np.sqrt(det_q(t))
        return integrand

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

    def stability_parameter(self, num=None, rtol=1e-12, full_output=False):
        r"""Compute the stability parameter.

        The stability parameter is defined in [1] as the principle eigenvalue
        of the stability operator. We use the normal vector in the slice and
        use the axisymmetry to greatly simplify the analytical task.

        According to Proposition 5.1 in [1], the MOTS represented by this
        curve is stably outermost iff the stability parameter is greater or
        equal to zero and strictly stably outermost iff it is greater than
        zero.

        @param num
            Pseudospectral resolution of the discretization of the operator.
            By default, uses the resolution of the curve representation
            itself.
        @param rtol
            Tolerance for reality check. The principle eigenvalue is shown in
            [1] to be real. If it has a non-zero imaginary part greater than
            ``rtol * |real_part|``, then a `RuntimeError` is raised. Default
            is `1e-12`.
        @param full_output
            If `True`, return all eigenvalues in addition to the principle
            eigenvalue.

        @return The principle eigenvalue as a float (i.e. the real part,
            ignoring any spurious imaginary part). If `full_output==True`,
            returns an array of all eigenvalues as second element. Note that
            these will be left as-is, i.e. they will be complex numbers which
            should have vanishing imaginary part.

        @b Notes

        If the MOTS \f$\sigma\f$ is contained in a time-symmetric slice
        \f$\Sigma\f$ of spacetime, and we compute the stability w.r.t. the
        outward pointing normal \f$\nu\f$ of \f$\sigma\f$ in \f$\Sigma\f$,
        then the stability operator simplifies greatly to
        \f[
            L_\nu \zeta = -\Delta_\sigma \zeta
                - (R_{ij}\nu^i\nu^j + k_{AB}k^{AB}) \zeta,
        \f]
        where \f$R_{ij}\f$ is the Ricci tensor of \f$(\Sigma, g)\f$,
        \f$k_{AB}\f$ is the extrinsic curvature of \f$\sigma\f$,
        \f$\Delta_\sigma = q^{AB}\,{}^{(2)}\nabla_A\,{}^{(2)}\nabla_B\f$
        is the Laplacian on \f$(\sigma, q)\f$, `g` is the 3-metric on the
        slice and `q` the induced 2-metric on the MOTS.

        To find the principle eigenvalue of \f$L_\nu\f$, first note that due
        to the axisymmetry, \f$\zeta\f$ will be a function of \f$\lambda\f$
        only, i.e. it will not depend on \f$\varphi\f$.
        The eigenvalues themselves are found using
        motsfinder.ndsolve.solver.NDSolver.eigenvalues(), which means we just
        need to provide the coefficient functions defining the one-dimensional
        operator \f$L_\nu\f$ acting on \f$\zeta\f$. The only part not already
        derived and computed in other methods is the Laplacian term, which
        reads
        \f{eqnarray*}{
            \Delta_\sigma \zeta
                &=& \frac{1}{\sqrt{q}} \partial_A (\sqrt{q} q^{AB} \partial_B \zeta)
                = \left(
                    \frac{1}{2} q^{AB} q^{C\lambda} \partial_C q_{AB}
                    + \partial_A q^{A\lambda}
                \right) \zeta' + q^{\lambda\lambda} \zeta''
                \\
                &=& \left(
                    \frac{1}{2} q^{AB} q^{C\lambda} - q^{AC} q^{B\lambda}
                \right) (\partial_C q_{AB}) \zeta'
                + q^{\lambda\lambda} \zeta''.
        \f}

        @b References

        [1] Andersson, Lars, Marc Mars, and Walter Simon. "Stability of
            marginally outer trapped surfaces and existence of marginally
            outer trapped tubes." arXiv preprint arXiv:0704.2889 (2007).
        """
        if num is None:
            num = self.num
        with self.fix_evaluator():
            if self.extr_curvature is None:
                eq = self.stability_eigenvalue_equation_timesym
            else:
                eq = self.stability_eigenvalue_equation_general
        solver = NDSolver(
            eq=eq,
            basis=CosineBasis(domain=(0, np.pi), num=num, lobatto=False),
        )
        eigenvals, principle = solver.eigenvalues()
        if abs(principle.imag) > rtol * abs(principle.real):
            raise RuntimeError("Non-real principle eigenvalue detected.")
        if full_output:
            return principle.real, eigenvals
        return principle.real

    def stability_eigenvalue_equation_timesym(self, pts):
        r"""Eigenvalue equation evaluator for time-symmetric case.

        This function is suitable to be passed as `eq` parameter to
        motsfinder.ndsolve.solver.NDSolver.
        """
        # Values for 3 derivative orders (0,1,2) per point.
        operator_values = np.zeros((len(pts), 3))
        for pt, op in zip(pts, operator_values):
            calc = self.get_calc_obj(pt)
            dq = calc.induced_metric(diff=1)
            q_inv = calc.induced_metric(inverse=True)
            # Laplacian:
            op[1] += -(
                0.5 * np.einsum('kl,i,ikl', q_inv, q_inv[:,0], dq)
                - np.einsum('ik,l,ikl', q_inv, q_inv[0,:], dq)
            )
            op[2] += -q_inv[0,0]
            # Remaining terms:
            k2 = calc.extrinsic_curvature(square=True)
            nu_cov = calc.covariant_normal(diff=0)
            nu = calc.g.inv.dot(nu_cov)
            Ric = self.metric.ricci_tensor(calc.point)
            Rnn = Ric.dot(nu).dot(nu)
            op[0] += -Rnn - k2
        return operator_values.T, 0.0

    def stability_eigenvalue_equation_general(self, pts):
        r"""Eigenvalue equation evaluator for the general case.

        This function is suitable to be passed as `eq` parameter to
        motsfinder.ndsolve.solver.NDSolver.
        """
        metric4 = FourMetric(self.metric)
        # Values for 3 derivative orders (0,1,2) per point.
        operator_values = np.zeros((len(pts), 3))
        for pt, op in zip(pts, operator_values):
            calc = self.get_calc_obj(pt)
            point = calc.point
            g3_inv = calc.g.inv
            dg3_inv = self.metric.diff(point, diff=1, inverse=True)
            g4 = metric4.at(point).mat
            n = metric4.normal(point)
            dn = metric4.normal(point, diff=1) # shape=(4,4), a,b -> del_a n^b
            nu3_cov = calc.covariant_normal(diff=0)
            nu3 = g3_inv.dot(nu3_cov)
            nu = np.zeros((4,))
            nu[1:] = nu3
            dnu3_cov = calc.covariant_normal(diff=1)
            dnu3 = (
                np.einsum('ijk,k->ij', dg3_inv, nu3_cov)
                + np.einsum('jk,ik->ij', g3_inv, dnu3_cov)
            )
            dnu = np.zeros((4, 4)) # a,b -> del_a nu^b (normal of MOTS)
            dnu[0,1:] = np.nan # don't know time derivatives of MOTS's normal
            dnu[1:,1:] = dnu3
            k = n - nu # ingoing future-pointing null normal (shape=4) k^a
            l = n + nu # outgoing future-pointing null normal (shape=4) l^a
            k_cov = g4.dot(k) # k_a
            l_cov = g4.dot(l) # l_a
            dux3 = calc.diff_xyz_wrt_laph(diff=1) # shape=(2,3), A,i -> del_A x^i
            dl = dn + dnu # shape=(4,4), a,b -> del_a l^b
            G4 = metric4.christoffel(point)
            # shape=(4,4), a,b -> nabla_a l^b  (NaN for a=0)
            cov_a_l = dl + np.einsum('min,n', G4, l)
            cov_A_l = np.einsum('Ai,im->Am', dux3, cov_a_l[1:]) # nabla_A l^mu
            s_A = -0.5 * np.einsum('m,Am->A', k_cov, cov_A_l) # torsion of l^mu
            q_inv = calc.induced_metric(inverse=True) # q^AB
            dg4 = metric4.diff(point, diff=1) # shape=(4,4,4), c,a,b -> del_c g_ab
            dl_cov = (
                np.einsum('ija,a->ij', dg4, l)
                + np.einsum('ja,ia->ij', g4, dl)
            )
            K_AB_l = -( # shape=(2,2), A,B -> K^mu_AB l_mu
                np.einsum('Ai,Bj,ij->AB', dux3, dux3, dl_cov[1:,1:])
                - np.einsum('Ai,Bj,aij,a->AB', dux3, dux3, G4[:,1:,1:], l_cov)
            )
            dq = calc.induced_metric(diff=1) # shape=(2,2,2), C,A,B -> del_C q_AB
            # TODO: remove duplication
            # Laplacian:
            op[1] += -(
                0.5 * np.einsum('AB,C,CAB', q_inv, q_inv[:,0], dq)
                - np.einsum('CA,B,CAB', q_inv, q_inv[0,:], dq)
            )
            op[2] += -q_inv[0,0]
            # 2nd term: 2 s^A D_A zeta = 2 s^lambda partial_lambda zeta
            s_A_up = q_inv.dot(s_A) # s^A (contravariant torsion)
            op[1] += 2 * s_A_up[0]
            # Terms with no differentiation: (1/2 R_S - Y - s^2 + Ds) zeta
            R_S = calc.ricci_scalar()
            Y = 0.5 * np.einsum('AB,AC,BD,CD', K_AB_l, q_inv, q_inv, K_AB_l)
            s2 = s_A_up.dot(s_A) # s^A s_A
            # Now the most involved one: Ds = 2nabla_A s^A  (2nabla: cov.der. on MOTS)
            G2 = calc.christoffel() # A,B,C -> Gamma^A_BC (Christoffel on MOTS)
            ddn = metric4.normal(point, diff=2) # shape=(4,4,4), a,b,c -> del_a del_b n^c
            ddg3_inv = np.asarray(self.metric.diff(point, diff=2, inverse=True))
            ddnu3_cov = calc.covariant_normal(diff=2) # shape=(3,3,3), i,j,k -> del_i del_j nu_k
            ddnu3 = (
                np.einsum('ijkl,l->ijk', ddg3_inv, nu3_cov)
                + np.einsum('ikl,jl->ijk', dg3_inv, dnu3_cov)
                + np.einsum('jkl,il->ijk', dg3_inv, dnu3_cov)
                + np.einsum('kl,ijl->ijk', g3_inv, ddnu3_cov)
            )
            ddnu = np.zeros((4, 4, 4)) # a,b,c -> del_a del_b nu^c
            ddnu[1:,1:,1:] = ddnu3
            ddnu[0,:,1:] = ddnu[:,0,1:] = np.nan
            ddl = ddn + ddnu # shape=(4,4,4), a,b,c -> del_a del_b l^c
            ddux3 = calc.diff_xyz_wrt_laph(diff=2) # shape=(2,2,3), A,B,i -> del_A del_B x^i
            dG4 = metric4.christoffel_deriv(point)
            # du_cov_i_l = del_A nabla_i l^a
            #   = del_A x^j (del_i del_j l^a + del_j G^a_ib l^b + G^a_ib del_j l^b)
            #   -> shape=(2,3,4)
            du_cov_i_l = (
                np.einsum('Aj,ija->Aia', dux3, ddl[1:,1:,:])
                + np.einsum('Aj,jaib,b->Aia', dux3, dG4[1:,:,1:,:], l)
                + np.einsum('Aj,aib,jb->Aia', dux3, G4[:,1:,:], dl[1:,:])
            )
            dk = dn - dnu # shape=(4,4), a,b -> del_a k^b
            duk_cov = ( # shape=(2,4), A,a -> del_A k_a
                np.einsum('Ai,iab,b->Aa', dux3, dg4[1:,:,:], k)
                + np.einsum('ab,Ai,ib->Aa', g4, dux3, dk[1:,:])
            )
            ds_cov = -0.5 * ( # shape=(2,2), A,B -> del_A s_B
                np.einsum('Aa,Ba->AB', duk_cov, cov_A_l)
                + np.einsum('a,ABi,ia->AB', k_cov, ddux3, cov_a_l[1:,:])
                + np.einsum('a,Bi,Aia->AB', k_cov, dux3, du_cov_i_l)
            )
            Ds = (np.einsum('AB,AB', q_inv, ds_cov)
                  - np.einsum('AB,CAB,C', q_inv, G2, s_A))
            op[0] += 0.5 * R_S - Y - s2 + Ds
        return operator_values.T, 0.0

    def multipoles(self, max_n=10, num=None, **kw):
        r"""Compute the mass multipoles I_n of the horizon.

        This is based on the considerations in [1].

        @param max_n
            Integer specifying the maximum number `n` (inclusive) up to which
            the moments \f$I_n\f$ should be computed.
        @param num
            Resolution for the pseudospectral solution of the invariant
            coordinate \f$\zeta\f$. Default is to use the current curve's
            resolution, but at least 100.
        @param get_zeta
            Optional argument. If specified and `True`, don't retuen the
            multipole moments but the solution \f$\zeta\f$.

        @return A list of computed moments \f$I_n\f$ for `n=0,1,..,max_n`.

        @b References

        [1] Ashtekar, Abhay, et al. "Multipole moments of isolated horizons."
            Classical and Quantum Gravity 21.11 (2004): 2549.
        """
        get_zeta = kw.pop('get_zeta', False)
        if kw:
            raise ValueError('Unexpected arguments: %s' % (kw,))
        if num is None:
            num = max(self.num, 100)
        def _cached(fun):
            cache = dict()
            def wrapper(la):
                try:
                    return cache[la]
                except KeyError:
                    val = fun(la)
                    cache[la] = val
                    return val
            return wrapper
        with self.fix_evaluator():
            det_q = _cached(self.get_det_q_func())
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
            if get_zeta:
                return zeta
            zeta = zeta.evaluator()
            ricci_scal = _cached(self.ricci_scalar)
            def integrand(n):
                def f(la):
                    zt = clip(zeta(la), -1, 1)
                    Scal = ricci_scal(la)
                    Yn = sph_harm(0, n, 0., np.arccos(zt)).real
                    return (2 * np.pi * np.sqrt(det_q(la)) * Scal * Yn)
                return f
            I_n = []
            for n in range(max_n+1):
                I_n.append(0.25 * quad(integrand(n), a=0, b=np.pi)[0])
            return I_n

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
        if algo == 'circumference':
            f = self.circumference
        elif algo in ('coord', 'proper_x_dist'):
            f = lambda param: self(param)[0]
        else:
            raise ValueError("Unknown algorithm: %s" % algo)
        func = lambda param: f(clip(param, 0, np.pi))
        func_neg = lambda param: -f(clip(param, 0, np.pi))
        with self.fix_evaluator():
            xa, xb, xc = bracket(func_neg, 0.0, 0.1, grow_limit=2)[:3]
            res = minimize_scalar(func_neg, bracket=(xa, xb, xc), options=dict(xtol=1e-1))
            x_max = res.x
            xa, xb, xc = bracket(func, x_max, x_max+0.1, grow_limit=2)[:3]
            if algo == 'proper_x_dist':
                func = lambda param: self.x_distance(param, **kw)
                xa, xb, xc = bracket(func, xb, xb+1e-2, grow_limit=2)[:3]
                res = minimize_scalar(func, bracket=(xa, xb, xc), options=dict(xtol=xtol))
            else:
                res = minimize_scalar(func, bracket=(xa, xb, xc), options=dict(xtol=xtol))
            return res.x, res.fun

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


class SignatureQuantities():
    __slots__ = ("param", "l", "k", "tau", "l_cov", "k_cov", "tau_cov",
                 "tau_l", "tau_k", "future_it", "past_it")

    def __init__(self, param, g4, l, k, tau, future_it, past_it):
        self.param = param
        self.l = l
        self.k = k
        self.tau = tau
        self.k_cov = g4.dot(k)
        self.l_cov = g4.dot(l)
        self.tau_cov = g4.dot(tau)
        self.tau_l = self.tau_cov.dot(l)
        self.tau_k = self.tau_cov.dot(k)
        self.future_it = future_it
        self.past_it = past_it

    def signature(self):
        if self.tau_l * self.tau_k > 0.0:
            # normal is spacelike => 3-surface is timelike
            return -1
        if self.tau_l * self.tau_k < 0.0:
            # normal is timelike => 3-surface is spacelike
            return 1
        # normal is null => 3-surface is null
        return 0


class IntegrationError(Exception):
    r"""Raised for non-converging integrals or if accuracy cannot be reached."""
    pass
