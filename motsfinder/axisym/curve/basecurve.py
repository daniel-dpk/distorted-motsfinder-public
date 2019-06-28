r"""@package motsfinder.axisym.curve.basecurve

Base class for general curves in the x-z-plane.

These curves will be taken to represent surfaces in axisymmetric geometry. As
such, they are assumed to begin and end on the symmetry axis, which is assumed
to be the `z`-axis. Without loss of generality, the `y` component is fixed to
zero.

Child classes may add additional meaning by taking a metric and extrinsic
curvature to define the geometry and embedding of the spatial slice in which
the surface represented by the curve lives.

@b Examples

See concrete implementations in e.g. starshapedcurve.StarShapedCurve or
parametriccurve.ParametricCurve.
"""

from contextlib import contextmanager
from abc import ABCMeta, abstractmethod

from six import add_metaclass
from scipy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar, brute, brentq
import numpy as np

from ...utils import save_to_file, load_from_file
from ...numutils import inf_norm1d, clip
from ...pickle_helpers import prepare_dict, restore_dict
from ...utils import insert_missing, update_dict, isiterable
from ...exprs.cheby import Cheby


__all__ = [
    "BaseCurve",
]


@add_metaclass(ABCMeta)
class BaseCurve(object):
    r"""Base class for curves in the x-z-plane

    Curves of this type are supposed to represent surfaces in axisymmetric 3D
    space. As such, they are (without loss of generality) taken to live on the
    x-z-plane for positive `x` values.

    The default domain is set to `(0,pi)`.

    Curves are expected to require one or more numeric expressions for their
    internal representation, which themselves need to be converted to
    evaluators before being usable. The extra step of converting expressions
    to evaluators can (and should) be avoided in case the underlying
    expression does not change. This is easily accomplished with the
    mechanisms for "freezing" evaluators provided by this base class. To
    utilize this mechanism, make sure your subclass provides all required
    evaluators in its _create_evaluators() implementation and also uses
    evaluators only by calling _get_evaluators().

    To temporarily freeze evaluators to their current state, use the
    fix_evaluator() context, e.g.:

    @code
    with curve.fix_evaluator():
        for p in np.linspace(0, np.pi, 1000):
            do_sth(curve(p))
    @endcode

    Subclasses need to implement the following functions:
        * copy() to create an independent copy of the curve
        * _create_evaluators() to create one or more evaluators for any
          internally stored numeric expressions. This is part of the evaluator
          caching mechanism.
        * #__call__() to evaluate the curve at a given parameter value. This
          returns the position (in x-z-coordinates) of the point on the curve.
        * _diff() returning the derivative of the two component functions.
        * collocation_points() returning the collocation or discretization
          points of the current curve representation.

    This class will then automatically compute tangent and normal vectors and
    any (supported) derivatives.

    Note: Tangent and normal vectors (and their derivatives) are not
          normalized.
    """

    def __init__(self, name=''):
        r"""Baseclass init for curves.

        @param name
            Name of this curve. This may be used when printing information
            about this curve or as label in plots.
        """
        self.domain = (0, np.pi)
        self._evaluator_overrides = None
        self._name = name
        self._user_data = dict()

    def save(self, filename, overwrite=False, verbose=True, msg=''):
        r"""Save the curve to disk.

        @param filename
            The file to store the data in. The extension ``'.npy'`` will be
            added if not already there.
        @param overwrite
            Whether to overwrite an existing file with the same name. If
            `False` (default) and such a file exists, a `RuntimeError` is
            raised.
        @param verbose
            Whether to print a message upon success.
        @param msg
            Optional additional text shown upon successful save in case
            `verbose==True`.
        """
        if msg:
            msg = ' "%s"' % msg
        save_to_file(
            filename, self, overwrite=overwrite, verbose=verbose,
            showname='curve%s' % msg
        )

    @staticmethod
    def load(filename):
        r"""Static function to load an expression object from disk."""
        return load_from_file(filename)

    def __getstate__(self):
        r"""Return a picklable state object representing the whole curve."""
        with self.override_evaluator(None):
            return prepare_dict(self.__dict__)

    def __setstate__(self, state):
        r"""Restore a complete curve from the given unpickled state."""
        self.__dict__.update(restore_dict(state))
        # compatibility with data from previous versions
        self.__dict__['_name'] = self.__dict__.get('_name', '')
        self.__dict__['_user_data'] = self.__dict__.get('_user_data', dict())

    @property
    def user_data(self):
        r"""Dictionary of custom data stored in the curve.

        This may be used to store search settings used to find the curve, or
        physical properties like the area. Note that all stored data must be
        picklable in order for saving and loading of the curve to work.
        """
        return self._user_data

    @property
    def name(self):
        r"""Name of this curve (used \eg as label for plotting)."""
        return self._name
    @name.setter
    def name(self, value):
        self._name = value

    def _get_evaluators(self):
        r"""Provide new or cached evaluators for any used expression object.

        Subclasses should use this function in conjunction with
        _create_evaluators() to make sure efficient evaluator caching is used.
        """
        if self._evaluator_overrides is not None:
            return self._evaluator_overrides
        return self._create_evaluators()

    @contextmanager
    def override_evaluator(self, evaluator):
        r"""Context for temporarily overriding the evaluator(s) used.

        This may be used to inject stand-in evaluators to temporarily alter
        the state of the curve. The primary use case is to supply the actual
        evaluator(s) of the expressions to fix them to their current state. An
        alternative use case is to modify the curve for (e.g.) implementing
        functional derivatives of operators depending on this curve
        ultralocally.
        """
        prev = self._evaluator_overrides
        try:
            self._evaluator_overrides = evaluator
            yield
        finally:
            self._evaluator_overrides = prev

    @contextmanager
    def fix_evaluator(self):
        r"""Context to temporarily fix the evaluator(s) of used expressions."""
        with self.override_evaluator(self._get_evaluators()):
            yield

    def freeze_evaluator(self):
        r"""Indefinitely freeze all evaluators of this curve.

        This should only be used if the underlying expressions are guaranteed
        not to change during the life time of this object, or at least until
        unfreeze_evaluator() is called.
        """
        self._evaluator_overrides = self._get_evaluators()

    def unfreeze_evaluator(self):
        r"""Undo a freeze_evaluator() call."""
        self._evaluator_overrides = None

    def force_evaluator_update(self):
        r"""Force all cached evaluators to be refreshed."""
        if self._evaluator_overrides is not None:
            self._evaluator_overrides = self._create_evaluators()

    def _prepare_result(self, x, z, xyz):
        r"""Convenience method to construct a 2D or 3D result.

        Functions such as normal() or tangent() have an option to either
        return a 2D array representing a vector or point in x-z-coordinates or
        a full 3D array including a value for the `y`-axis. This convenience
        method interprets the `xyz` boolean to construct the respective result
        from the supplied `x` and `z` values.
        """
        if xyz:
            return np.array([x, 0.0, z])
        return np.array([x, z])

    @abstractmethod
    def copy(self):
        r"""Create an independent copy of this curve."""
        pass

    @abstractmethod
    def _create_evaluators(self):
        r"""Return evaluator(s) of any internal numeric expressions.

        Subclasses may have multiple expressions representing the curve. Since
        it is inefficient to repeatedly create evaluators when the underlying
        expressions don't change, the BaseCurve class can handle freezing
        (i.e. reusing) evaluators. Any evaluators returned from this function
        will participate in this mechanism.
        """
        pass

    @abstractmethod
    def __call__(self, param, xyz=False):
        r"""Evaluate the curve at a particular parameter value.

        @param param (float)
            Parameter value at which to evaluate the curve.
        @param xyz (boolean)
            Whether to return 3D (x,y,z) coordinates or not. Default is
            `False`, i.e. return 2D coordinate values in the x-z-plane.

        @return The coordinates of the corresponding point on the curve.
        """
        pass

    @abstractmethod
    def collocation_points(self, **kw):
        r"""Return the collocation points of the current curve representation."""
        pass

    @abstractmethod
    def _diff(self, param, diff):
        r"""Compute the derivatives of the x and z component functions.

        If a subclass implements this function, tangent and normal vectors can
        be computed by the base class.
        """
        pass

    def diff(self, param, diff=1, xyz=False):
        r"""Compute the derivative of this curve \wrt its parameter.

        This computes \f$ \partial_\lambda^n\gamma(\lambda) \f$, where
        `diff=n` and `param=lambda`.
        """
        if diff == 0:
            return self(param, xyz=xyz)
        x, z = self._diff(param, diff=diff)
        return self._prepare_result(x, z, xyz)

    def tangent(self, param, diff=0, xyz=False):
        r"""Compute the (Euclidean) tangent vector or one of its derivatives.

        Note that the tangent vector is not normalized. The quantitiy computed
        here is simply \f$ \gamma'(\lambda) \f$ (for `diff==0`).

        @param param (float)
            Parameter value at which to evaluate the curve.
        @param diff (int)
            Derivative order of the tangent vector w.r.t. the parameter.
        @param xyz (boolean)
            Whether to return 3D (x,y,z) coordinates or not. Default is
            `False`, i.e. return 2D coordinate values in the x-z-plane.
        """
        return self.diff(param, diff=diff+1, xyz=xyz)

    def normal(self, param, diff=0, xyz=False):
        r"""Compute the (Euclidean) normal vector or a derivative thereof.

        Note: The returned vector will not be normalized. Its length will
              depend on the *speed* of the curve's parameterization.

        This function simply rotates the tangent vector (or a derivative) by
        90 degrees to obtain a vector normal to the curve in flat coordinate
        space.

        @param param (float)
            Parameter value at which to evaluate the curve.
        @param diff (int)
            Derivative order of the normal vector w.r.t. the parameter.
        @param xyz (boolean)
            Whether to return 3D (x,y,z) coordinates or not. Default is
            `False`, i.e. return 2D coordinate values in the x-z-plane.
        """
        tx, tz = self.tangent(param, diff=diff)
        return self._prepare_result(-tz, tx, xyz)

    def curvature_in_coord_space(self, param):
        r"""Compute the curvature of this curve in flat coordinate space.

        Consider this curve parameterized by arc length (in flat space), i.e.
        \f[
            \tilde\gamma(t) = \gamma(s(t)),
                \qquad\mathrm{s.t.\ }
            \Vert\tilde\gamma'(t)\Vert = 1.
        \f]
        Then, the curvature is defined as
        \f[
            \kappa = \Vert\tilde\gamma''\Vert
                = \frac{|x'z''-x''z'|}{\Vert\gamma'\Vert^3},
        \f]
        where `x` and `z` are the component functions of this curve.
        """
        with self.fix_evaluator():
            xp, zp = self.diff(param, diff=1)
            xpp, zpp = self.diff(param, diff=2)
            n = np.sqrt(xp**2 + zp**2)
            return abs(xp*zpp - xpp*zp) / n**3

    def arc_length(self, a=0, b=np.pi, atol=1e-12, rtol=1e-12, limit=100,
                   full_output=False):
        r"""Compute the length of the curve.

        We use numerical integration to compute
        \f[
            \int_a^b \Vert\gamma'(t)\Vert_g\ dt,
        \f]
        where \f$g\f$ is the current metric of this curve. Curves that only
        live in coordinate space (e.g. parametriccurve.ParametricCurve) use
        the flat Euclidean metric instead.

        @param a
            Start value of parameter interval to integrate over. Default is
            `0`.
        @param b
            End value of the parameter interval to integrate over. Default is
            `pi`.
        @param atol
            Absolute tolerance used in the `scipy.integrate.quad` call.
            Default is `1e-12`.
        @param rtol
            Relative tolerance used in the `scipy.integrate.quad` call.
            Default is `1e-12`.
        @param limit
            Maximum number of subdivisions in the adaptive integration
            routine. Default is `100`.
        @param full_output
            If `True`, return the computed result and an estimation of the
            error. Otherwise (default), just return the result.

        @return Computed arc length. If `full_output==True`, returns a pair
            ``(length, err)``, where `length` is the computed arc length and
            `err` the estimated error.

        @b Notes

        Child classes that know their metric should override this method and
        supply the metric to arc_length_using_metric().
        """
        return self.arc_length_using_metric(metric=None, a=a, b=b, atol=atol,
                                            rtol=rtol, limit=limit,
                                            full_output=full_output)

    def arc_length_using_metric(self, metric, a=0, b=np.pi, atol=1e-12,
                                rtol=1e-12, limit=100, full_output=False):
        r"""Compute the length of this curve \wrt a given metric.

        Similar to arc_length(), but this method allows you to specify an
        arbitrary metric for computing the norms of the tangent vectors.

        @param metric
            Metric to use for computing norms. Explicitly specify ``None`` to
            use the flat metric.
        @param a
            Start value of parameter interval to integrate over. Default is
            `0`.
        @param b
            End value of the parameter interval to integrate over. Default is
            `pi`.
        @param atol
            Absolute tolerance used in the `scipy.integrate.quad` call.
            Default is `1e-12`.
        @param rtol
            Relative tolerance used in the `scipy.integrate.quad` call.
            Default is `1e-12`.
        @param limit
            Maximum number of subdivisions in the adaptive integration
            routine. Default is `100`.
        @param full_output
            If `True`, return the computed result and an estimation of the
            error. Otherwise (default), just return the result.

        @return The computed arc length. If `full_output==True`, returns a
            pair ``(length, err)``, where `length` is the computed arc length
            and `err` the estimated error.
        """
        if metric is None:
            def tangent_length(s):
                return norm(self.tangent(s))
        else:
            def tangent_length(s):
                pt = self(s, xyz=True)
                return metric.at(pt).norm(self.tangent(s, xyz=True))
        with self.fix_evaluator():
            val, err = quad(tangent_length, a=a, b=b, limit=limit,
                            epsabs=atol, epsrel=rtol)
            if full_output:
                return val, err
            return val

    def proper_length_map(self, num=None, evaluators=True, full_output=False,
                          **kw):
        r"""Generate functions to map between curve parameter and proper length.

        The resulting callables can be used to e.g. plot properties along the
        curve in terms of the curve's proper length instead of the current
        (seemingly) arbitrary parameterization.

        The first returned function `length_map` maps the parameter values to
        the normalized relative proper length values (i.e. scaled to
        `[0,pi]`). This can be used to plot quantities already computed on a
        grid of parameter values, e.g.:
        @code
            length_map, inv_map, l0 = c.proper_length_map(full_output=True)
            params = np.linspace(0, np.pi, 100, endpoint=False)[1:]
            riccis = [c.ricci_scalar(la) for la in params]
            # plotting w.r.t. normalized proper length
            plot_data(
                [length_map(la) for la in params], riccis,
                xlabel=r"$\bar \lambda$", ylabel=r"$\mathcal{R}$",
            )
            # plotting w.r.t. actual proper length
            plot_data(
                [length_map(la)*l0 for la in params], riccis,
                xlabel=r"proper length", ylabel=r"$\mathcal{R}$",
            )
        @endcode

        The second function converts normalized proper lengths to parameter
        values, i.e. it can be used to plot curve properties not already
        computed:
        @code
            length_map, inv_map = c.proper_length_map()
            plot_1d(
                f=lambda x: c.ricci_scalar(inv_map(x)),
                domain=(0, np.pi), value_pad=1e-6, points=100,
                xlabel=r"$\bar \lambda$", ylabel=r"$\mathcal{R}$",
            )
        @endcode

        @return Two callables `length_map` and `inv_map`. If
            ``full_output==True``, also returns the full (i.e. not normalized)
            proper length.

        @param num
            Number of points to sample the proper length at for building an
            approximation function. Defaults to the current curve resolution.
        @param evaluators
            If `True` (default), return the callables instead of the
            interpolant objects.
        @param full_output
            If `True`, return the proper length as third return value. Default
            is `False`.
        @param metric
            (Extracted from ``**kw``) The metric to use for computation. If
            set to `None`, uses coordinate space arc length instead of actual
            proper length. Default is to use the current curve's metric. The
            flat metric is used in case the curve is a pure coordinate curve
            (e.g. parametriccurve.ParametricCurve).
        @param **kw
            Further arguments are passed to arc_length_using_metric().
        """
        # Child classes having a metric should use:
        #   metric = kw.pop('metric', self.metric)
        #   return super().proper_length_map(num=num, metric=metric, **kw)
        metric = kw.pop('metric', None)
        if num is None:
            num = self.num
        xs = Cheby([], domain=(0, np.pi)).collocation_points(num=num, lobatto=True)
        xs = np.asarray(list(reversed(xs)))
        intervals = list(zip(xs, xs[1:]))
        ys = [0.] + [self.arc_length_using_metric(metric=metric, a=a, b=b, **kw)
                     for a, b in intervals]
        for i in range(1, len(ys)):
            ys[i] += ys[i-1]
        proper_length = ys[-1]
        ys = np.pi * np.asarray(ys) / proper_length
        length_map = Cheby([], domain=(0, np.pi))
        length_map.set_coefficients(
            list(reversed(ys)), # Chebyshev points are reversed
            physical_space=True, lobatto=True
        )
        interp = PchipInterpolator(ys, xs) # guarantees monotonicity of interpolant
        inv_map = Cheby.from_function(interp, num=num, domain=(0, np.pi))
        if evaluators:
            result = length_map.evaluator(), inv_map.evaluator()
        else:
            result = length_map, inv_map
        if full_output:
            return result + (proper_length,)
        return result

    def z_distance(self, other_curve=None, atol=1e-12, rtol=1e-12, limit=100,
                   allow_intersection=False, full_output=False):
        r"""Compute the z-distance to another curve or the origin.

        Curves in axisymmetry represent surfaces. These, by construction, have
        two intersections with the `z`-axis, which itself is a geodesic for
        any axisymmetric metric. This function computes the length of the
        geodesic along the `z`-axis connecting this curve with the given other
        curve. If no other curve is specified, computes the distance to the
        origin.

        If the surfaces intersect or one is enclosed in the other, we define
        their distance as zero.
        Otherwise, the distance is computed via
        \f[
            \mathrm{dist} = \left| \int_a^b \sqrt{g_{zz}}\ dz \right|,
        \f]
        where `a` and `b` are the respective intersections of the two curves
        with the `z`-axis lying closest to each other.

        Curves that only live in coordinate space (for example
        parametriccurve.ParametricCurve) use the flat Euclidean metric
        instead (i.e. they return `|z|`).

        @param other_curve
            Curve to which to compute the distance. By default, computes the
            distance to the origin.
        @param atol
            Absolute tolerance used in the `scipy.integrate.quad` call.
            Default is `1e-12`.
        @param rtol
            Relative tolerance used in the `scipy.integrate.quad` call.
            Default is `1e-12`.
        @param limit
            Maximum number of subdivisions in the adaptive integration
            routine. Default is `100`.
        @param allow_intersection
            If `False` (default), surfaces that intersect are defined to have
            zero distance. If `True`, the distance of the points intersecting
            the z-axis is still computed and returned as negative number.
        @param full_output
            If `True`, return the computed result and an estimation of the
            error. Otherwise (default), just return the result.

        @return Computed distance. If `full_output==True`, returns a pair
            ``(dist, err)``, where `dist` is the computed distance and `err`
            the estimated error.

        @b Notes

        Child classes that know their metric should override this method and
        supply the metric to z_distance_using_metric().
        """
        return self.z_distance_using_metric(
            metric=None, other_curve=other_curve, atol=atol, rtol=rtol,
            limit=limit, allow_intersection=allow_intersection,
            full_output=full_output
        )

    def z_distance_using_metric(self, metric, other_curve=None, atol=1e-12,
                                rtol=1e-12, limit=100,
                                allow_intersection=False, full_output=False):
        r"""Compute z-distance to another curve \wrt a given metric.

        Similar to z_distance(), but allows you to specify an arbitrary
        metric for the computation.

        @param metric
            Metric to use for computing the distance. Explicitly specify
            ``None`` to use the flat metric.
        @param other_curve
            Curve to which to compute the distance. By default, computes the
            distance to the origin.
        @param atol
            Absolute tolerance used in the `scipy.integrate.quad` call.
            Default is `1e-12`.
        @param rtol
            Relative tolerance used in the `scipy.integrate.quad` call.
            Default is `1e-12`.
        @param limit
            Maximum number of subdivisions in the adaptive integration
            routine. Default is `100`.
        @param allow_intersection
            If `False` (default), surfaces that intersect are defined to have
            zero distance. If `True`, the distance of the points intersecting
            the z-axis is still computed and returned as negative number.
        @param full_output
            If `True`, return the computed result and an estimation of the
            error. Otherwise (default), just return the result.

        @return Computed distance. If `full_output==True`, returns a pair
            ``(dist, err)``, where `dist` is the computed distance and `err`
            the estimated error.
        """
        # This curve's z-components:   a1 >= b1
        # Other curve's z-compoonents: a2 >= b2
        a1 = self(0)[1]
        b1 = self(np.pi)[1]
        if a1 < b1:
            a1, b1 = b1, a1
        if other_curve is None:
            a2 = b2 = 0.0
        else:
            a2 = other_curve(0)[1]
            b2 = other_curve(np.pi)[1]
            if a2 < b2:
                a2, b2 = b2, a2
        # We now have a1 >= b1 and a2 >= b2.
        if a1 < a2:
            # This curve is "above" the other one. Swap them.
            a1, b1, a2, b2 = a2, b2, a1, b1
        # We now have two intervals [b1, a1] and [b2, a2] with a1 >= a2.
        # Are the two intervals disjoint?
        if not allow_intersection and b1 <= a2:
            # No, distance is zero by definition if we don't allow intersection.
            return (0.0, 0.0) if full_output else 0.0
        c1 = (a1 + b1) / 2.0
        c2 = (a2 + b2) / 2.0
        if c2 < c1:
            a, b = a2, b1
        else:
            a, b = a1, b2
        if metric is None:
            dist = b - a
            return (dist, 0.0) if full_output else dist
        def ds(z):
            g = metric.at([0, 0, z])
            g_zz = g.mat[2,2]
            return np.sqrt(g_zz)
        val, err = quad(ds, a=a, b=b, limit=limit, epsabs=atol, epsrel=rtol)
        if full_output:
            return val, err
        return val

    def rotate_to_xz_plane(self, point):
        r"""Rotate a 3D point onto the x-z-plane with positive x.

        This is basically a transformation to cylindrical `(rho,z)`
        coordinates. In case a 2D point is given, the second component is
        taken to be the z-component.
        """
        if len(point) == 2:
            x, z = point
        else:
            x, y, z = point
            if y != 0.0:
                x = np.sqrt(x**2 + y**2)
        return abs(x), z

    def closest_point(self, point, start_param=None, Ns=25):
        r"""Find the parameter at which the curve is closest to a given point.

        The point is projected onto the x-z-plane prior to starting the
        search. Therefore, the parameter is gauranteed to lie in the domain
        `(0,pi)`. The search itself is a numerical minimum search for the
        distance. The (coordinate) line connecting the `point` to the curve at
        the returned parameter value will be orthogonal to the curve at the
        intersection (up to numerical errors due to the search, of course).

        @param point
            2D or 3D coordinates of the point to use. Will be projected onto
            the x-z-plane with positive x-component prior to the search.
        @param start_param
            Optional starting parameter value for the search. If not given, a
            `scipy.optimize.brute` search is performed to increase the chance
            of finding the global minimum instead of a local one.
        @param Ns
            Number of samples for the `scipy.optimize.brute` search in case no
            `start_param` is given.

        @return The parameter along the curve at which the curve is closest to
            the point.
        """
        x, z = self.rotate_to_xz_plane(point)
        la = self._closest_point(x, z, start_param, Ns)
        return la

    def _closest_point(self, x, z, start_param, Ns):
        r"""Implements closest_point().

        Performs the actual search assuming `x>0`. The result will lie in the
        range `(0,pi)`.
        """
        pi = np.pi
        def f(t):
            px, pz = self(t)
            return np.sqrt((x-px)**2 + (z-pz)**2)
        if start_param is None:
            x0 = brute(lambda x: f(x[0]), [[0, pi]], Ns=Ns, finish=None)
            step = np.pi/(Ns-1)
            res = minimize_scalar(
                f, bounds=[max(0, x0-step), min(np.pi, x0+step)], method='bounded',
                options=dict(xatol=1e-12),
            )
        else:
            res = minimize_scalar(f, bracket=(start_param, pi/Ns),
                                  options=dict(xtol=1e-12))
        la = res.x
        return la

    def find_line_intersection(self, point, vector, Ns=50):
        r"""Find the point at which this curve intersects a given line.

        The line is specified by giving a `point` and a direction `vector`.
        The search is carried out numerically by finding the root of
        \f[
            f(\lambda) := \nu \cdot (\gamma(\lambda) - p_0),
        \f]
        where \f$p_0\f$ is the given `point` and \f$\nu\f$ a normal on the
        line. Note that the computation only considers coordinate space, i.e.
        we don't integrate geodesics from the `point`.

        In case of multiple intersections, one of them is found (which one is
        not defined).

        @param point
            A point on the line.
        @param vector
            Direction vector of the line.
        @param Ns
            Number of points on the curve to check for a sign change of the
            above function. Increasing may help in cases where the line barely
            intersects the curve.

        @return Parameter of the curve where the line intersects the curve.
        """
        point = np.asarray(point, dtype=float)
        vector = np.asarray(vector, dtype=float)
        if point.size == 3:
            point = np.array([point[0], point[2]])
        if vector.size == 3:
            vector = np.array([vector[0], vector[2]])
        normal = np.array([-vector[1], vector[0]])
        normal /= norm(normal)
        with self.fix_evaluator():
            def f(t):
                t = clip(t, 0, np.pi)
                rel_vec = self(t) - point
                return normal.dot(rel_vec)
            f0 = f(0)
            if f0 == 0.0:
                return 0.0
            step = np.pi/Ns
            a = 0
            while f(a+step)*f0 > 0:
                if a == np.pi:
                    raise RuntimeError("Line seems to not intersect curve.")
                a = min(np.pi, a+step)
            return brentq(f, a=a, b=a+step)

    def find_max_x(self, Ns=50):
        r"""Find the point of the curve with maximum distance to the z-axis.

        The search is carried out numerically. Note that the computation only
        considers coordinate space.

        @param Ns
            Number of points in the initial `scipy.optimize.brute` search to
            get close to the global extremum.

        @return Parameter of the curve where the line has maximum x-value.
        """
        with self.fix_evaluator():
            x0 = brute(lambda x: -self(x[0])[0], [[0, np.pi]], Ns=Ns,
                       finish=None)
            res = minimize_scalar(
                lambda x: -self(x)[0],
                bracket=(x0, np.pi/Ns), bounds=(0, np.pi), method='bounded',
                options=dict(xatol=1e-12)
            )
            return res.x

    def inf_norm(self, other_curve, **kw):
        r"""Compute L_inf norm of this curve minus another curve.

        The quantity computed here is
        \f[
            L^\infty(\gamma_1, \gamma_2)
                := \max_\lambda(
                    \Vert\gamma_1(\lambda) - \gamma_2(\lambda)\Vert
                ),
        \f]
        where the norm is the flat (Euclidean) norm in coordinate space.

        Note that this quantity depends on how the two curves are
        parameterized.
        """
        with self.fix_evaluator(), other_curve.fix_evaluator():
            def f(t):
                return norm(self(t)-other_curve(t))
            return inf_norm1d(f, domain=self.domain, **kw)

    def point_distances(self, params=None):
        r"""Return the coordinate distances of points along the curve.

        For `N` parameter values given as `params`, the result will be `N-1`
        distances.
        """
        if params is None:
            params = self.collocation_points()
        with self.fix_evaluator():
            pts = np.array([self(la) for la in params])
        deltas = np.diff(pts, axis=0)
        distances = norm(deltas, axis=1)
        return distances

    def plot_distances(self, params=None, **kw):
        r"""Convenience method for quickly plotting coordinate distances of collocation points."""
        from ...ipyutils import plot_data
        if params is None:
            params = self.collocation_points()
        if 'distances' in kw:
            distances = kw.pop('distances')
        else:
            distances = self.point_distances(params=params)
        return plot_data(
            params[:-1], distances,
            **insert_missing(
                kw,
                figsize=(4, 2.4), dpi=110,
                ylog=True,
                xlabel=r"curve parameter $\lambda$",
                ylabel=r"coordinate distance",
            )
        )

    def plot(self, lw=2, equal_lengths=True, copy_x=True, **kw):
        r"""Plot the curve represented by this object.

        The optional keyword arguments are passed to the called
        ipyutils.plotting.plot_curve() function. Additional modifications of
        the plot (e.g. adding more curves) is thus supported in the usual way
        by setting ``show=False`` and working with the returned `ax` object.

        @param lw (float)
            Linewidth for the curve. Default is `2`, which is slightly wider
            than the default linewidth of the plot() function.
        @param equal_lengths (boolean)
            Whether to force an aspect ratio of 1. Default is `True`.
        @param copy_x (boolean)
            Curves in axisymmetry are defined only for positive `x` values.
            Setting this to `False` prevents the curve to be completed on the
            negative `x`-side.
        @param **kw
            Additional keyword arguments passed to
            ipyutils.plotting.plot_curve().
        """
        from ...ipyutils import plot_curve
        return plot_curve(self, lw=lw, equal_lengths=equal_lengths,
                          copy_x=copy_x,
                          **insert_missing(kw, xlabel='$x$', ylabel='$z$',
                                           label=self.name or None))

    @staticmethod
    def plot_curves(*curves, cmap=None, last_callback=None, **kw):
        r"""Static method for plotting multiple curves in the same plot.

        Without the optional format string, colors are cycled in their default
        order or through the given colormap in the `cmap` argument.

        @param *curves
            Positional arguments are interpreted as curves to plot. Each curve
            can have one of three forms:
            @code
                curve
                (curve, label)
                (curve, label, fmt)
                (curve, dict(...))
                (curve, dict(...), fmt)
            @endcode
            Here, `label` is a string and `fmt` a format string such as
            ``'.-k'``. The dictionary can be used to pass arbitrary options to
            the individual BaseCurve.plot() calls. If the `label` is not set
            and not the empty string ``''``, then the curve's stored name is
            used as label.
        @param cmap
            Colormap through which to cycle for curve colors not set
            otherwise.
        @param last_callback
            Optional callback function called in the last plotting command.
            Has the signature ``last_callback(ax)``.
        @param **kw
            Optional keyword arguments supplied to BaseCurve.plot().
            Individual options specified via ``*curves`` take precedence.

        @b Examples
        @code
        c1 = StarShapedCurve.create_sphere(2.0, 20, FlatThreeMetric())
        c2 = StarShapedCurve.create_sphere(1.0, 20, c1.metric)
        c1.plot_curves((c1, '$c_1$'), (c2, '$c_2$', '--g'))
        @endcode
        """
        ax = kw.get('ax', None)
        eql = kw.get('equal_lengths', True)
        kw['equal_lengths'] = False
        show = False
        close = False
        save = None
        for i, curve in enumerate(curves):
            fmt = '-'
            label = None
            opts = dict()
            if isiterable(curve):
                if len(curve) == 3:
                    curve, label, fmt = curve
                else:
                    curve, label = curve
                if isinstance(label, dict):
                    opts = label.copy()
                    label = opts.pop('label', None)
                    fmt = opts.pop('l', fmt)
            if cmap is not None:
                c = cmap(float(i) / (len(curves)-1))
                opts['color'] = opts.get('color', c)
            if label is None:
                label = curve.name
            if label == '':
                label = None
            if i+1 == len(curves):
                show = kw.get('show', True)
                close = kw.get('close', False)
                save = kw.get('save', None)
                kw['equal_lengths'] = eql
                if last_callback:
                    kw['cfg_callback'] = last_callback
            ax = curve.plot(
                **update_dict(kw, l=fmt, label=label, show=show, close=close,
                              ax=ax, save=save, **opts)
            )
        return ax
