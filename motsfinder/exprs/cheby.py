r"""@package motsfinder.exprs.cheby

Expression representing a truncated Chebyshev polynomial series.

The main items in this module are the Cheby class for representing said series
and a few helper functions, like evaluate_Tn() for efficiently evaluating a
potentially large number of Chebyshev polynomials \f$ T_n \f$ at the same
point using a recusion relation.

The function diff() in this module provides a fast implementation for
computing the coefficients of derivatives of truncated Chebyshev series.

A significant part of the useful methods is inherited from
series.SeriesExpression and numexpr.NumericExpression.

@b Examples

```
    # Create a function using some coefficients.
    # The default interval is (-1,1).
    f = Cheby([1, -1, 0.5])
    plot_1d(f, title="f(x)")

    # Plot the first and second derivatives
    ev = f.evaluator()
    plot_1d(ev.function(1), title="f'(x)")
    plot_1d(ev.function(2), title="f''(x)")
```

Create a series expansion from a known function.
```
    from math import cos, exp
    a, b = 10, 20
    def func(x):
        x = 2*(x-a)/(b-a) - 1
        return 1 - x**2 + .2*exp(-x) * cos(10*x**2) + .9
    f = Cheby.from_function(func, 70, domain=(a, b))

    # Plot the function and the error (i.e. difference)
    plot_1d(f, title="f(x)")
    plot_1d(f, func, difference=True, title="difference")

    # Plot how the coefficients converge
    plot_data(range(len(f.a_n)), f.a_n, title="coefficients",
              absolute=True, ylog=True)
```
"""

from __future__ import print_function
from builtins import range

from scipy import linalg
import numpy as np
from mpmath import mp

from .evaluators import EvaluatorBase
from .series import SeriesExpression

try:
    from ._cheby import evaluate_Tn_double
except ImportError:
    evaluate_Tn_double = None


__all__ = [
    "Cheby",
]


def _evaluate_Tn_mpmath_py(x, Tn):
    r"""mpmath implementation of evaluate_Tn()."""
    num = len(Tn)
    one = mp.one
    two = mp.mpf(2)
    if x == 1:
        return [one] * num
    if x == -1:
        return [(-one)**k for k in range(num)]
    x = mp.mpf(x)
    Tn[0] = one
    if num > 1:
        Tn[1] = x
    for n in range(2, num):
        Tn[n] = (two * x) * Tn[n-1] - Tn[n-2]
    return Tn

# No Cython implementation yet
evaluate_Tn_mpmath = _evaluate_Tn_mpmath_py


def _evaluate_Tn_double_py(x, Tn):
    r"""Python (\ie non-Cython) implementation of evaluate_Tn()."""
    x = float(x)
    num = len(Tn)
    if x == 1.0:
        return np.ones(num)
    if x == -1.0:
        return np.array([(-1.0)**k for k in range(num)], dtype=float)
    Tn[0] = 1.0
    if num > 1:
        Tn[1] = x
    for n in range(2, num):
        Tn[n] = (2.0 * x) * Tn[n-1] - Tn[n-2]
    if not isinstance(Tn, np.ndarray):
        Tn = np.array(Tn, dtype=float)
    return Tn


if evaluate_Tn_double is None:
    # Seems the Cython implementation was not imported.
    evaluate_Tn_double = _evaluate_Tn_double_py


def force_pure_python(pure_python=True):
    r"""Toggle during runtime which implementation of evaluate_Tn() to use.

    @param pure_python
        If `True`, switch to the pure Python implementation. If `False`, try
        to use the Cython implementation but don't fail if it is not
        available.

    Returns:
        `True` if the requested implementation could be selected, `False` otherwise.
    """
    global evaluate_Tn_double
    if pure_python:
        evaluate_Tn_double = _evaluate_Tn_double_py
        return True
    try:
        from ._cheby import evaluate_Tn_double
        return True
    except ImportError:
        evaluate_Tn_double = _evaluate_Tn_double_py
        return False


def evaluate_Tn(x, Tn, use_mp):
    r"""Efficiently evaluate multiple Chebyshev polynomials at one point.

    This method evaluates `T_n` at the given position `x`. How many of the
    polynomials are evaluated depends on the number of elements in `Tn`.
    The computed values are returned as a list. In this process, `Tn` may
    or may not be modified, so it is best to assign the returned value to
    `Tn` afterwards.

    @param x (float or mp.mpf)
        Argument at which to evaluate the `T_n`. Must lie in the interval
        `[-1,1]`.
    @param Tn (list or numpy array)
        The list to possibly modify. The number of elements in this list
        determines the number of Chebyshev polynomials evaluated.
    @param use_mp (boolean, optional)
        Whether to use `mpmath` computations.

    Returns:
        The Chebyshev polynomials evaluated at `x` as `list` or numpy array.
        This may be the `Tn` supplied as argument or a new list (in case this
        is more efficient than modifying the existing list).\n
        In case a new list is created, it will be a numpy array in case
        `use_mp=False`.
    """
    if use_mp:
        return evaluate_Tn_mpmath(x, Tn)
    if not isinstance(Tn, np.ndarray):
        Tn = np.array(Tn, dtype=np.double)
    return evaluate_Tn_double(x, Tn)


def diff(a_n, use_mp, scale):
    r"""Compute coefficients of the Chebyshev interpolation derivative.

    Given the coefficients `a_n` of a Chebyshev interpolation \f[
        u(x) = \sum_n a_n T_n(x),
    \f]
    this function computes the coefficients `b_n` of the derivative \f[
        u'(x) = \sum_n b_n T_n(x).
    \f]

    @param a_n (iterable)
        Coefficients of the original function.
    @param use_mp (boolean, optional)
        Whether to use `mpmath` computations.
    @param scale (float or mp.mpf)
        Each coefficient will be multiplied with this scaling factor. This
        can be used to account for a physical domain `[a,b]` which is
        different from `[-1,1]`.

    Returns:
        Coefficients of the derivative of the function as a `list`.
    """
    num = len(a_n)
    N = num-1
    if N <= 0:
        return []
    zero = mp.zero if use_mp else 0.0
    one = mp.one if use_mp else 1.0
    two = mp.mpf(2) if use_mp else 2.0
    c = lambda k: (2 if k == 0 else 1)
    b_n = [zero] * N
    for k in reversed(range(N)): # k = N-1, N-2, ..., 0
        if k <= N-3:
            b_n[k] = one/c(k) * (b_n[k+2] + (two*(k+1)) * a_n[k+1])
        else:
            b_n[k] = one/c(k) * ((two*(k+1)) * a_n[k+1])
    if scale != 1.0:
        b_n = [scale * v for v in b_n]
    return b_n


def _create_jn(sym, num):
    r"""Create the indices for Chebyshev polynomials with the desired symmetry.

    For no symmetry, this is simply `0, ..., num-1`, while for even symmetry,
    we return `0, 2, 4, ..., 2*num-1` and for odd symmetry
    `1, 3, 5, ..., 2*num`.
    """
    if sym is None:
        return range(0, num)
    elif sym == 'even':
        return range(0, 2*num, 2)
    elif sym == 'odd':
        return range(1, 2*num+1, 2)
    else:
        raise TypeError("Invalid symmetry: %s" % sym)


class Cheby(SeriesExpression):
    r"""Truncated Chebyshev series expression.

    This class implements a seris.SeriesExpression for Chebyshev polynomials.
    Note that most of the functionality lives in the parent classes.
    """
    def __init__(self, a_n, domain=(-1, 1), symmetry=None, name=None):
        r"""Create a truncated Chebyshev series expression.

        @param a_n (iterable)
            The coefficients of the polynomials. May be empty to indicate a
            zero function.
        @param domain (2-tuple/list)
            The domain of this expression. The Chebyshev polynomials are
            defined on `[-1,1]`. Specifying a different domain will map it to
            this domain, while also ensuring that derivatives take into
            account this coordinate change.
        @param symmetry
            Optional symmetry setting to select only the even or odd
            polynomials having even or odd symmetry w.r.t. zero, respectively.
            This translates to the respective symmetry w.r.t. the domain
            center.
        @param name
            Name of the expression (e.g. for print_tree()).
        """
        super(Cheby, self).__init__(a_n=a_n, domain=domain, name=name)
        if symmetry not in (None, 'even', 'odd'):
            raise TypeError("Invalid symmetry: %s" % symmetry)
        self.sym = symmetry

    def internal_domain(self):
        return (-mp.one, mp.one)

    def is_orthogonal(self):
        return True

    def copy(self):
        obj = super(Cheby, self).copy()
        obj.sym = self.sym
        return obj

    def _expr_str(self):
        where = ", where a_n=%r" % (self.a_n,)
        if self.sym is None:
            return "sum a_n T_n(x)" + where
        elif self.sym == 'even':
            return "sum a_n T_(2n)(x)" + where
        elif self.sym == 'odd':
            return "sum a_n T_(2n+1)(x)" + where

    def _evaluator(self, use_mp):
        self._check_a_n(self.a_n)
        if len(self.a_n) == 0:
            return self.zero
        return _ChebyEval(self, use_mp)

    def _approximate_constant(self, value, num, lobatto, use_mp, dps):
        with self.context(use_mp, dps) as ctx:
            if self.sym == 'odd' and value != 0.0:
                raise ValueError("Cannot approximate non-zero constant with "
                                 "odd-symmetry series.")
            self.a_n = [ctx.mpf(value)] + [ctx.zero] * (num-1)

    def _from_physical_space(self, a_n, lobatto, use_mp, dps):
        with self.context(use_mp, dps) as ctx:
            if self.sym is None:
                a_n = self._direct_transform(a_n, lobatto, use_mp, ctx)
            else:
                a_n = self._indirect_transform(a_n, lobatto, use_mp)
        return a_n

    def _indirect_transform(self, f_n, lobatto, use_mp):
        r"""Solve a system of equations to obtain the coefficients.

        Given a set of values `f_i` of the function to approximate at the
        collocation points `x_i`, this evaluates and solves a matrix equation
        \f[
            \sum_{j=0}^{\mathrm{num-1}} a_j T_{n(j)}(x_i) = f_i
        \f]
        to get the coefficients `a_n` of the basis functions. The notation
        `n(j)` indicates that this works even if only a subset of basis
        functions is considered.
        """
        num = len(f_n)
        jn = _create_jn(self.sym, num)
        Tn = [mp.one] * (2*num) if use_mp else np.ones(2*num)
        def row(xi):
            values = evaluate_Tn(xi, Tn, use_mp)
            if use_mp:
                return [values[j] for j in jn]
            return values[jn]
        x = self.collocation_points(num, internal_domain=True,
                                    lobatto=lobatto, use_mp=use_mp)
        Minv = map(row, x)
        if use_mp:
            a_n = mp.lu_solve(Minv, f_n)
        else:
            a_n = linalg.solve(np.array(Minv), np.array(f_n, dtype=float))
        return a_n

    def _direct_transform(self, a_n, lobatto, use_mp, ctx):
        r"""Use a more efficient formula to transform from physical to spectral space.

        The formula implemented here is taken from [1]. This one can only be
        used if all Chebyshev polynomials up to the truncation point are used.

        @b References

        [1] Canuto, Claudio, et al. Spectral Methods: Fundamentals in Single
            Domains. Springer Science & Business Media, 2007.
        """
        num = len(a_n)
        N = num-1
        pi = ctx.pi
        x = self.collocation_points(num, internal_domain=True,
                                    lobatto=lobatto, use_mp=use_mp)
        if lobatto:
            w = [pi/N] * num
            w[0] = w[-1] = pi/(2*N)
            row_norms = [pi/2] * num
            row_norms[0] = row_norms[-1] = pi
        else:
            w = [pi/(N+1)] * num
            row_norms = [pi/2] * num
            row_norms[0] = pi
        Tn = [mp.one] * num if use_mp else np.ones(num)
        def col(j):
            vals = evaluate_Tn(x[j], Tn, use_mp)
            return [vals[i]/row_norms[i] * w[j] for i in range(num)]
        M_T = [col(j) for j in range(num)]
        if use_mp:
            a_n = mp.matrix(M_T).T * mp.matrix(a_n)
        else:
            a_n = np.array(M_T).T.dot(np.array(a_n))
        return a_n

    def collocation_points(self, num=None, lobatto=True, internal_domain=False,
                           use_mp=False, dps=None, symmetry='current', **kw):
        r"""As series.SeriesExpression.collocation_points(), but with `symmetry` parameter.

        @param symmetry
            The symmetry the collocation points should respect. For example,
            for even symmetry it makes no sense to have any collocation point
            which has a corresponding collocation point mirrored across `x=0`.
            The default, `'current'`, means that the current symmetry should
            be used.
        """
        if symmetry == 'current':
            symmetry = self.sym
        return super(Cheby, self).collocation_points(
            num, lobatto, internal_domain, use_mp, dps, symmetry=symmetry, **kw
        )

    @classmethod
    def create_collocation_points(cls, num, lobatto=True, use_mp=False, dps=None, symmetry=None):
        r"""As series.SeriesExpression.create_collocation_points(), but with `symmetry` parameter.

        @param symmetry
            The symmetry the collocation points should respect.
        """
        with cls.context(use_mp, dps) as ctx:
            pi = ctx.pi
            if lobatto:
                N = num - 1
                if symmetry is None:
                    return [ctx.cos(i*pi/N) for i in range(0, N+1)]
                elif symmetry == 'even':
                    return [ctx.cos(i*pi/(2*N)) for i in range(0, N+1)]
                elif symmetry == 'odd':
                    return [ctx.cos(i*pi/(2*N+1)) for i in range(0, N+1)]
                else:
                    raise TypeError("Invalid symmetry: %s" % symmetry)
            else:
                N = num
                if symmetry is None:
                    return [ctx.cos((2*i-1)*pi/(2*N)) for i in range(1, N+1)]
                elif symmetry == 'even':
                    return [ctx.cos((2*i-1)*pi/(4*N)) for i in range(1, N+1)]
                elif symmetry == 'odd':
                    return [ctx.cos((2*i-1)*pi/(4*N+2)) for i in range(1, N+1)]
                else:
                    raise TypeError("Invalid symmetry: %s" % symmetry)

class _ChebyEval(EvaluatorBase):
    r"""Evaluator for the Cheby class."""
    def __init__(self, expr, use_mp):
        super(_ChebyEval, self).__init__(expr, use_mp)
        self._sym = expr.sym
        an = self._expand(expr.a_n, expr.sym)
        ctx = self.ctx
        a, b = map(ctx.mpf, expr.domain)
        self._scale = 2/(b-a)
        self._trans = -1 - 2*a/(b-a)
        bn = diff(an, use_mp, self._scale)
        cn = diff(bn, use_mp, self._scale)
        self._coeffs = [self._vector(coeffs) for coeffs in (an, bn, cn)]
        self._xrel = None
        num = len(an)
        self._num = num
        self._Tk = [ctx.zero] * num if use_mp else np.zeros(num)
        self._dirty = True
        self._basis_derivs = dict()
        self._jn = None

    def _expand(self, an, sym):
        r"""Convert coefficients \wrt a symmetry to general coefficients.

        In case the Cheby expression is configured to be symmetric or
        antisymmetric w.r.t. zero, its coefficients will consist of only those
        of the basis functions with the correct symmetry. For the algorithms
        used in the evaluator class, the simplest solution is to use
        *expanded*, i.e. normal coefficients for all basis functions,
        previously missing ones being zero, of course.

        @param an
            Coefficients as iterable.
        @param sym
            Symmetry with respect to which the `an` are given.

        @return Coefficients as list, `mp.matrix` or NumPy array for all basis
            functions up to the point of truncation.
        """
        if sym is None:
            return self._vector(an)
        jn = _create_jn(sym, len(an))
        num = 2*len(an)
        if self.use_mp:
            result = [mp.zero] * num
            for i, j in enumerate(jn):
                result[j] = an[i]
        else:
            result = np.zeros(num)
            result[jn] = an
        return result

    def _vector(self, elements):
        if len(elements) == 0:
            return []
        return mp.matrix(elements) if self.use_mp else np.array(elements, dtype=float)

    def _x_changed(self, x):
        # Small rounding errors might push the value slightly outside the
        # range (-1,1).
        self._xrel = min(1, max(-1, self._scale * x + self._trans))
        self._dirty = True

    def _get_coeffs(self, n):
        r"""Return cached coefficients for a certain derivative order.

        If coefficients for the requested derivative `n` are not yet cached,
        they are computed and then stored.

        @param n
            Derivative order to get coefficients for.

        @return Coefficients as a NumPy array or `mpmath` matrix. An empty
            coefficient list will be returned as just an empty list `[]`.
        """
        if n >= len(self._coeffs):
            for i in range(len(self._coeffs), n+1):
                coeffs = diff(self._coeffs[i-1], self.use_mp, self._scale)
                self._coeffs.append(self._vector(coeffs))
        return self._coeffs[n]

    def _values(self, n=0, apply_coeffs=False):
        r"""Return the basis function values optionally multiplied with the coefficients.

        Basis function values are only recomputed if required. This means that
        if you evaluate the series and its derivatives at the same point, the
        basis has to be evaluated only once.

        @param n
            Derivative order of coefficients to use. Only has an effect if
            `apply_coeffs==True`.
        @param apply_coeffs
            Whether to multiply the basis functions with the selected
            coefficients (`True`) or return just the values of the basis
            functions (`False`, default).
        """
        if self._dirty:
            self._Tk = evaluate_Tn(self._xrel, self._Tk, self.use_mp)
            self._dirty = False
        values = self._Tk
        if apply_coeffs:
            an = self._get_coeffs(n)
            if self.use_mp:
                values = [an[n]*values[n] for n in range(len(an))]
            else:
                if n == 0:
                    values = an * values
                else:
                    values = an * values[:len(an)]
        return values

    def _get_basis_derivs(self, n):
        r"""Return the coefficients for evaluating derivatives of basis functions.

        The n'th derivative of an individual Chebyshev polynomial \f$T_k\f$ can
        itself be expressed exactly as Chebyshev series, i.e. \f[
            T_k^{(n)}(x) = \sum_{i=0}^{k-1} a^{k,n}_i T_i(x).
        \f]
        This function computes the coefficients \f$ a^{k,n}_i \f$ for all
        basis functions `k = 1, ..., num` of a requested derivative order `n`,
        where `num` is the number of basis functions in the current expansion.

        The results are independent of the point `x` at which to evaluate the
        basis and can hence efficiently be cached for the lifetime of this
        object.

        @param n
            Derivative order of coefficients to return.

        @return List of coefficients. The i'th element of the returned list
            corresponds to the coefficients of \f$ T_i^{(n)} \f$.
        """
        if n in self._basis_derivs:
            return self._basis_derivs[n]
        if not isinstance(n, int) or n <= 0:
            raise TypeError("Not a valid derivative order: %s" % n)
        ctx = self.ctx
        use_mp = self.use_mp
        scale = self._scale
        if n == 1:
            derivs = ([[]] + [diff([ctx.zero]*(i-1) + [1], use_mp, scale)
                              for i in range(2, self._num+1)])
        else:
            derivs = [diff(an, use_mp, scale) for an in self._get_basis_derivs(n-1)]
        self._basis_derivs[n] = derivs
        return derivs

    def evaluate_basis(self, n=0):
        r"""Evaluate all basis functions at the current point.

        This evaluates the `n`'th derivative of all basis functions at the
        current point and returns the resulting list/array of values.
        """
        if self._sym is not None and self._jn is None:
            self._jn = _create_jn(self._sym, int(self._num/2))
        jn = self._jn
        Tk = self._values()
        if n == 0:
            if jn is None:
                return Tk
            return [Tk[j] for j in jn] if self.use_mp else Tk[jn]
        derivs = self._get_basis_derivs(n)
        if self.use_mp:
            ctx = self.ctx
            results = [ctx.fsum(an[n]*Tk[n] for n in range(len(an))) for an in derivs]
            if jn is None:
                return results
            return [results[j] for j in jn]
        else:
            results = [Tk[:len(an)].dot(an) for an in derivs]
            if jn is None:
                return results
            return np.array(results)[jn]

    def _eval(self, n=0):
        ctx = self.ctx
        an = self._get_coeffs(n)
        if len(an) == 0:
            return ctx.zero
        if self.use_mp:
            return ctx.fsum(self._values(n=n, apply_coeffs=True))
        values = self._values(n=n, apply_coeffs=False)
        if n == 0:
            return values.dot(an)
        else:
            return values[:len(an)].dot(an)
