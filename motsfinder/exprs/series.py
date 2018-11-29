r"""@package motsfinder.exprs.series

Base class for general truncated series expressions.
"""

from __future__ import print_function
from six import add_metaclass, iteritems

from abc import abstractmethod
from scipy.interpolate import interp1d
import numpy as np
from mpmath import mp

from ..utils import isiterable
from .numexpr import NumericExpression


__all__ = [
    "SeriesExpression",
]


def resize_coeffs(coeffs, old_shape, new_shape):
    r"""Resize 2D coefficient matrix to a new shape.

    For orthogonal bases, this is equivalent to *resampling* the series
    expansion to a different resolution. Increasing the resolution and
    shrinking it afterwards is lossless.

    This function can be used to resize 1D coefficients by specifying the
    second dimension to have one element, i.e.:

        coeffs = resize_coeffs(coeffs, (old_num, 1), (new_num, 1))

    Args:
        coeffs: Flattened list of current coefficients, which should be
            interpreted via `old_shape`.
        old_shape: 2-tuple/list of the coefficient matrix size.
        new_shape: 2-tuple/list of the desired new matrix size.

    Returns:
        Flattened coefficient list corresponding to the new shape.
    """
    arr = np.array(coeffs, dtype=object)
    mat = arr.reshape(*old_shape)
    N1, N2 = new_shape
    mat = mat[:N1, :N2]
    mat = np.pad(mat, ((0, max(0, N1-mat.shape[0])), (0, max(0, N2-mat.shape[1]))),
                 mode='constant', constant_values=0)
    return list(mat.flatten())


def transform_domains(points, from_domain, to_domain, use_mp=False, dps=None):
    r"""Linearly transform a set of points between two intervals.

    The points are transformed bijectively, linearly, and orientation
    preserving. The values are guaranteed to lie in the target domain, i.e.
    any numerical noise will not lead to points lying slightly outside after
    the transformation.

    Args:
        points: Iterable of all points to transform. Points must be plain
                values, i.e. no vectors.
        from_domain: 2-tuple/list indicating the interval the points are
                currently living in.
        to_domain: 2-tuple/list indicating the target interval the returned
                points should live in.
        use_mp: Whether to use arbitrary precision math operations (`True`) or
                faster floating point precision operations (`False`, default).
        dps:    Number of decimal places to use in case of `use_mp==True`.

    Returns:
        A `list` of the transformed points.
    """
    with mp.workdps(dps or mp.dps):
        fl = mp.mpf if use_mp else float
        a, b = map(fl, to_domain)
        c, d = map(fl, from_domain)
        scale = (b-a)/(d-c)
        trans = a - (b-a)*c/(d-c)
        return [min(b, max(a, scale*p + trans)) for p in points]


class SeriesExpression(NumericExpression):
    r"""Base class for truncated series expressions.

    This base class is responsible for managing the coefficient list, domain
    transformations and many of the sampling and resampling tasks. To perform
    its task, sub-classes need to implement the following functions:

        _from_physical_space()      # compute coefficients from function values
        internal_domain()           # the domain of the basis
        create_collocation_points() # collocation points of the basis

    ... plus the following for numexpr.NumericExpression:

        _expr_str()  # represent the expression
        _evaluator() # create an evaluator object
    """

    def __init__(self, a_n, domain, **kw):
        r"""Parent init for series expressions.

        Args:
            a_n:    Iterable of coefficient values. May also be empty.
            domain: Domain on which this expansion should be defined. This is
                    independent of the domain the basis is defined on.

        Further keyword arguments are passed to the parent init, that is to
        numexpr.NumericExpression.__init__().
        """
        super(SeriesExpression, self).__init__(domain=domain, **kw)
        self._check_a_n(a_n)
        self.a_n = a_n
        self.a = domain[0]
        self.b = domain[1]

    def _check_a_n(self, a_n):
        r"""Make sure the coefficients are iterable."""
        if not isiterable(a_n):
            raise ValueError('Coefficient list must be an iterable.')

    def get_dof(self):
        r"""Return the degrees of freedom, \ie the current number of coefficients."""
        return len(self.a_n)

    @property
    def N(self):
        r"""Degrees of freedom, \ie current number of coefficients."""
        return self.get_dof()

    def copy(self):
        r"""Create a copy of the series expansion expression.

        The coefficients will be copied, i.e. the new expression can be
        modified independently.

        Sub classes are responsible for transferring additional state over to
        the copy and should override this method in case they need to.
        """
        obj = type(self)(self.a_n[:], domain=self.domain, name=self.name)
        if isinstance(obj.a_n, np.ndarray):
            obj.a_n = obj.a_n.copy()
        return obj

    def resample(self, num, use_mp=False, dps=None):
        r"""Change resolution of the expansion.

        For orthogonal bases, upsampling is lossless. Also in that case,
        increasing and then reducing the resolution to its original value is
        lossless and introduces no numerical noise.

        For non-orthogonal bases, the function currently represented by the
        expansion will be sampled at the new requested number of collocation
        points to perform a lossy resampling.
        """
        if self.get_dof() == num:
            return
        if self.is_orthogonal():
            self.a_n = resize_coeffs(self.a_n, (self.N, 1), (num, 1))
        else:
            with self.context(use_mp, dps):
                self.approximate(self.evaluator(use_mp), num=num,
                                 use_mp=use_mp)

    def interpolate(self, y, x=None, kind='cubic', num=None, lobatto=True,
                    use_mp=False, dps=None):
        r"""Interpolate data and model the result as a series.

        This is a convenience function that interpolates the given data using
        `scipy.interpolate.interp1d` via cubic splines (by default) and then
        approximates the result as a series with `num` terms.

        If no x-values are present, the values `y` are assumed to lie
        equidistant in the domain `[a,b]` (borders included).

        Since the resulting interpolation function is approximated as a
        series, there is no guarantee that \f[
            f(x_i) = y_i
        \f]
        holds. In general, the higher the value of `num`, the better the above
        equation is satisfied.

        Args:
            y:  Data y-values.
            x:  Positions of the data values on the x-axis. By default, an
                equidistant grid on the domain `[a,b]` is used.
            kind: (string or int, optional)
                `kind` parameter for the `interp1d` call. Default is `'cubic'`.
            num: Number of series terms to use for approximation. Default is
                to use the current length of the coefficient list.
            use_mp: Whether to use `mpmath` computations. Default is `False`.
                Only affects the approximation step as the interpolation is
                done by SciPy.
            dps: Number of decimal places to use when `use_mp==True`.
                Default is to use the current global setting.

        """
        if x is None:
            x = np.linspace(float(self.a), float(self.b), len(y))
        f = interp1d(x, y, kind=kind)
        self.approximate(lambda x: float(f(float(x))),
                         num=num, lobatto=lobatto, use_mp=use_mp, dps=dps)

    def approximate(self, f, num=None, lobatto=True, use_mp=False, dps=None):
        r"""Approximate a function.

        This samples the function `f` at `num` points in the current interval
        `[a,b]` to create a series approximation using `num` terms.

        After calling this, the series represented by this object is
        guaranteed to have `num` elements, corresponding to `N = num-1` in
        most textbooks for most bases.

        Args:
            f:  The function to approximate. May also be a constant, in which
                case we still create `num` elements (most of which will be
                zero for most bases).
            num: Number of terms to use for approximation. This is also the
                number of times `f` is evaluated. Default is to use the
                current length of the coefficient list.
            lobatto: Whether to use the Gauss-Lobatto (default) points for
                sampling or just the Gauss points. May not be used by all
                child classes.
            use_mp: Whether to use `mpmath` computations. Default is `False`.
            dps: Number of decimal places to use when `use_mp==True`.
                Default is to use the current global setting.
        """
        with self.context(use_mp, dps):
            if num is None:
                num = len(self.a_n)
            if num == 0:
                raise ValueError("Cannot sample function with zero points.")
            if not callable(f):
                try:
                    f = f.evaluator(use_mp)
                except AttributeError:
                    pass
            if not callable(f):
                self._approximate_constant(f, num, lobatto, use_mp, dps)
            else:
                self._approximate(f, num, lobatto, use_mp, dps)

    def _approximate(self, f, num, lobatto, use_mp, dps):
        r"""Perform the approximation of a function.

        This samples the function at the desired number of collocation points
        and then triggers these samples to be converted to coefficients. The
        actual conversion will be handled by set_coefficients().
        """
        with self.context(use_mp, dps):
            pts = self.collocation_points(num, lobatto=lobatto, use_mp=use_mp, dps=dps)
            self.set_coefficients(
                [f(x) for x in pts],
                physical_space=True, lobatto=lobatto, use_mp=use_mp, dps=dps
            )

    def _approximate_constant(self, value, num, lobatto, use_mp, dps):
        r"""Approximate a constant value.

        Subclasses may implement a more efficient algorithm here. This default
        implementation simply creates a constant function and approximates
        this function.
        """
        self._approximate(lambda x: value, num, lobatto, use_mp, dps)

    def set_coefficients(self, a_n, physical_space=False, lobatto=True,
                         use_mp=False, dps=None):
        r"""Store a new list of coefficients after optional transformation.

        Takes a new coefficient list and either stores it directly. If the
        list actually represents function values at the collocation points
        (i.e. in physical space), they are transformed to spectral space
        before being stored.

        The actual transformation must be implemented by sub classes in the
        _from_physical_space() method.
        """
        self._check_a_n(a_n)
        if physical_space:
            self.a_n = self._from_physical_space(a_n, lobatto=lobatto,
                                                 use_mp=use_mp, dps=dps)
        else:
            self.a_n = a_n

    @abstractmethod
    def _from_physical_space(self, a_n, lobatto, use_mp, dps):
        r"""Convert values in physical space to spectral space."""
        pass

    def collocation_points(self, num=None, lobatto=True, internal_domain=False,
                           use_mp=False, dps=None, **kw):
        r"""Return the Gauss-Lobatto or Gauss points for this series.

        The returned list will have `num` elements. If you know the value of
        `N`, you can find the corresponding value of `num` by checking the
        textbooks for the number of degrees of freedom. The exact use of `N`
        seems to be inconsistent across different books for many bases.

        The ``**kw`` argument is passed to the subclass method for creating
        the collocation points. These may be used to further specify options
        such as symmetry, etc.

        Args:
            num: (int, optional)
                Number of collocation points to create. Note that in most printed
                formulas, the points have indices `0, 1, ..., N`, i.e. there are
                `N+1` points. This function, however, returns `num` points. To get
                the points that belong to a certain value of `N` in these
                formulas, call this function with `num=N+1`.
                The default is the current number of coefficients.
            lobatto: (boolean, optional)
                If `True` (default), return the Gauss-Lobatto points, which
                include the endpoints of the interval. Otherwise, return the
                interior (Gauss) points.
            internal_domain: (boolean, optional)
                If `False` (the default), return the collocation points mapped to
                the range `[a,b]`. If `True`, use the internal domain (depends on
                subclass).
            use_mp: (boolean, optional)
                Whether to compute the points using `mpmath` arbitrary precision
                calculations. The result will then be a list of `mpmath.mpf`
                floats. Default is `False`.
            dps: (int, optional)
                Number of decimal places to use when `use_mp==True`.

        Returns:
            List of collocation points either in the domain of the basis (if
            `internal_domain==True`) or the physical domain (if
            `internal_domain==False`).
        """
        if num is None:
            num = len(self.a_n)
        x = self.create_collocation_points(num, lobatto=lobatto,
                                           use_mp=use_mp, dps=dps, **kw)
        if not internal_domain:
            x = transform_domains(x, self.internal_domain(), self.domain,
                                  use_mp=use_mp, dps=dps)
        return x

    @abstractmethod
    def internal_domain(self):
        pass

    @abstractmethod
    def create_collocation_points(cls, num, lobatto=True, use_mp=False, dps=None, **kw):
        """Subclasses should implement this as @classmethod."""
        # pylint: disable=no-self-argument
        pass

    @abstractmethod
    def is_orthogonal(self):
        r"""Whether this series consists of orthogonal functions."""
        pass

    @classmethod
    def from_function(cls, f, num, domain=None, lobatto=True, use_mp=False,
                      dps=None, **kw):
        if domain is None:
            domain = f.domain
        expr = cls([], domain=domain, **kw)
        expr.approximate(f, num=num, lobatto=lobatto, use_mp=use_mp, dps=dps)
        return expr
