r"""@package motsfinder.exprs.trig

Expression representing a truncated sine or cosine series.


@b Examples

```
    # Create and plot a function f(x) = 1 - cos(x) + cos(2x)/2
    f = CosineSeries([1, -1, 0.5])
    plot_1d(f, title="f(x)")

    # Plot the first and second derivatives
    ev = f.evaluator()
    plot_1d(ev.function(1), title="f'(x)")
    plot_1d(ev.function(2), title="f''(x)")
```

We can also create a series expansion from a known function.
```
    from math import sin, cos, exp, pi
    func = lambda x: exp(cos(x)) * sin(x) * sin(5*x)
    f = CosineSeries.from_function(func, 30, domain=(0, pi))

    # Plot the function and the error (i.e. difference)
    plot_1d(f, title="f(x)")
    plot_1d(f, func, difference=True, title="difference")
```

Now the same as above, but using mpmath to get an approximation of the
function correct up to about `1e-30`.
```
    sin, cos, exp, pi = mp.sin, mp.cos, mp.exp, mp.pi
    with mp.workdps(30):
        func = lambda x: exp(cos(x)) * sin(x) * sin(mp.mpf(5)*x)
        f = CosineSeries.from_function(func, 40, domain=(0, pi), use_mp=True)

        # Plot the function and the error (i.e. difference)
        ev = f.evaluator(use_mp=True)
        plot_1d(ev, title="f(x)")
        plot_1d(ev, func, difference=True, title="difference")

        # Plot how the coefficients converge
        plot_data(range(len(f.a_n)), f.a_n, title="coefficients",
                  absolute=True, ylog=True, ylim=(1e-32, 1))
```
"""

from abc import abstractmethod

from six import add_metaclass
from builtins import range
from scipy import linalg
import numpy as np
from mpmath import mp

from .evaluators import EvaluatorBase
from .series import SeriesExpression


__all__ = [
    "SineSeries",
    "CosineSeries",
]


class _TrigSeries(SeriesExpression):
    r"""Base class for trigonometric series expressions.

    This unifies all common aspects of truncated sine and cosine series
    expressions.
    """
    def __init__(self, a_n, domain=(0, mp.pi), name=None):
        r"""Construct a trigonometric series from a set of coefficients.

        @param a_n
            Coefficients to initialize the series with. May be modified later
            using the public `a_n` property.
        @param domain
            Domain of the function to represent. This is independent of the
            domain `(0, pi)` on which the basis functions are evaluated.
            Default is the native domain `(0, pi)`.
        @param name
            Name of the expression (e.g. for print_tree()).
        """
        super(_TrigSeries, self).__init__(a_n=a_n, domain=domain, name=name)

    def internal_domain(self):
        return (0, mp.pi)

    def is_orthogonal(self):
        return True

    def _from_physical_space(self, a_n, lobatto, use_mp, dps):
        num = len(a_n)
        with self.context(use_mp, dps):
            x = self.collocation_points(num, internal_domain=True,
                                        lobatto=lobatto, use_mp=use_mp)
            jn = self.create_frequencies(num)
            Minv = self.evaluate_trig_mat(x, jn, use_mp)
            if use_mp:
                a_n = mp.lu_solve(Minv, a_n)
            else:
                a_n = linalg.solve(Minv, np.array(a_n, dtype=float))
        return a_n

    @abstractmethod
    def evaluate_trig_mat(self, xs, jn, use_mp=False):
        r"""Evaluate the basis functions on all points, returning a matrix."""
        pass

    @abstractmethod
    def create_frequencies(self, num):
        r"""Create all frequencies for the used basis.

        A cosine series consists of all cos-frequencies, starting from 0,
        while for a sine series, we should start from 1.
        """
        pass


class _TrigEval(EvaluatorBase):
    r"""Common base evaluator for trigonometric expressions.

    This implements everything a trigonometric expression evaluator needs to
    do that is not specific for it being a sine or cosine series. The
    differentiating aspects have been moved to very light-weight functions the
    child classes need to implement.
    """
    def __init__(self, expr, use_mp):
        super(_TrigEval, self).__init__(expr, use_mp)
        ## Whether to clamp the arguments to evaluate at to the domain.
        ## Default is `True`.
        self.clamp = True
        ## Constant `pi` with correct accuracy (float or mpmath).
        self._pi = pi = mp.pi if use_mp else np.pi
        an = expr.a_n
        ## Number of coefficients.
        self._num = num = len(an)
        ## Indices of active basis functions.
        self._jn = expr.create_frequencies(num)
        ctx = self.ctx
        a, b = map(ctx.mpf, expr.domain)
        ## Scale factor for transforming to native domain.
        self._scale = pi/(b-a)
        ## Translation for transforming to native domain.
        self._trans = -pi*a/(b-a)
        ## Coefficients of the function and any derivatives.
        self._deriv_coeffs = [self._vector(an)]
        ## Current point to evaluate at in native domain.
        self._xrel = None
        ## All used sine frequencies evaluated at current point.
        self._sin = None
        ## All used cosine frequencies evaluated at current point.
        self._cos = None

    def _vector(self, elements):
        # pylint: disable=len-as-condition
        if len(elements) == 0:
            return []
        return mp.matrix(elements) if self.use_mp else np.array(elements, dtype=float)

    def _x_changed(self, x):
        if self.clamp:
            self._xrel = min(self._pi, max(0, self._scale * x + self._trans))
        else:
            self._xrel = self._scale * x + self._trans
        self._sin = None
        self._cos = None

    @property
    def cos_values(self):
        r"""All cosine basis function values at the current point (cached)."""
        if self._cos is None:
            self._cos = evaluate_trig_series("cos", self._xrel, self._jn,
                                             use_mp=self.use_mp)
        return self._cos

    @property
    def sin_values(self):
        r"""All sine basis function values at the current point (cached)."""
        if self._sin is None:
            self._sin = evaluate_trig_series("sin", self._xrel, self._jn,
                                             use_mp=self.use_mp)
        return self._sin

    def _get_deriv_coeffs(self, n=0):
        r"""Return the coefficients of the represented function or its derivatives.

        The coefficients are computed on-demand and then cached, so that
        subsequent evaluation of derivatives of this function are as efficient
        as evaluating the function itself.
        """
        while len(self._deriv_coeffs) <= n:
            prev_coeffs = self._deriv_coeffs[-1]
            order = len(self._deriv_coeffs)
            gen_fn = self._get_diff_coeff_gen(order)
            coeffs = gen_fn(prev_coeffs, self._jn, self.use_mp, self._scale)
            self._deriv_coeffs.append(coeffs)
        return self._deriv_coeffs[n]

    def evaluate_basis(self, n=0):
        r"""Evaluate all basis functions at the current point.

        This evaluates the `n`'th derivative of all basis functions at the
        current point and returns the resulting list/array of values.
        """
        values = self.basis_values(n)
        an = self._get_deriv_coeffs(n)
        if self.use_mp:
            values = [an[n]*values[n] for n in range(len(an))]
        else:
            values = an * values
        return values

    def _eval(self, n=0):
        ctx = self.ctx
        an = self._get_deriv_coeffs(n)
        # pylint: disable=len-as-condition
        if len(an) == 0:
            return ctx.zero
        if self.use_mp:
            return ctx.fsum(self.evaluate_basis(n=n))
        return self.basis_values(n).dot(an)

    @abstractmethod
    def _get_diff_coeff_gen(self, diff_order):
        r"""Return the function used to differentiate coefficients of this
        series at a certain derivative order."""
        pass

    @abstractmethod
    def basis_values(self, diff_order):
        r"""Compute all used basis functions for even or odd derivatives (including 0).

        This evaluates the basis functions (sine/cosine frequencies) usable
        for the even or odd derivatives. Note that no derivatives are actually
        computed. However, these can easily be obtained from the results
        returned here.

        The result is cached such that it can be reused in subsequent
        evaluations at the same point. A use case is that the function and its
        second derivative need to be evaluated at the same point `x`. The
        values will be computed only once in this case.
        """
        pass


class SineSeries(_TrigSeries):
    r"""Truncated sine series expression.

    This series is suitable to approximate functions vanishint at both ends of
    their domain, which are additionally antisymmetric about both of those
    ends.
    """

    def _expr_str(self):
        where = ", where a_n=%r" % (self.a_n,)
        return "sum a_n sin((n+1) x)" + where

    def evaluate_trig_mat(self, xs, jn, use_mp=False):
        return evaluate_trig_mat("sin", xs, jn, use_mp)

    def create_frequencies(self, num):
        return list(range(1, num+1))

    def _evaluator(self, use_mp):
        self._check_a_n(self.a_n)
        # pylint: disable=len-as-condition
        if len(self.a_n) == 0: # handle list and numpy array case
            return self.zero
        return _SineSeriesEval(self, use_mp)

    def _approximate_constant(self, value, num, lobatto, use_mp, dps):
        if value != 0.0:
            raise ValueError("Cannot approximate non-zero constant with "
                             "a sine series.")
        with self.context(use_mp, dps) as ctx:
            self.a_n = [ctx.zero] * num

    @classmethod
    def create_collocation_points(cls, num, lobatto=True, use_mp=False,
                                  dps=None):
        with cls.context(use_mp, dps):
            pi = mp.pi if use_mp else np.pi
            fl = mp.mpf if use_mp else float
            if lobatto:
                N = fl(num + 1)
                return [i*pi/N for i in range(1, num+1)]
            N = fl(num)
            return [(2*i-1)*pi/(2*N) for i in range(1, num+1)]


class _SineSeriesEval(_TrigEval):
    r"""Evaluator class for SineSeries.

    This evaluator can efficiently evaluate the function represented by the
    series expression and its derivatives up to arbitrary order.
    """

    def basis_values(self, diff_order):
        if diff_order % 2:
            return self.cos_values
        return self.sin_values

    def _get_diff_coeff_gen(self, diff_order):
        if diff_order % 2:
            return diff_sin_coeffs
        return diff_cos_coeffs


class CosineSeries(_TrigSeries):
    r"""Truncated cosine series to represent a function.

    This series is suitable to approximate functions symmetric about both ends
    of their domain. As such, the represented function will be guaranteed to
    have a vanishing first derivative at both domain boundaries.

    For examples, see the module docstring.
    """

    def _expr_str(self):
        where = ", where a_n=%r" % (self.a_n,)
        return "sum a_n cos(n x)" + where

    def evaluate_trig_mat(self, xs, jn, use_mp=False):
        return evaluate_trig_mat("cos", xs, jn, use_mp)

    def create_frequencies(self, num):
        return list(range(num))

    def _evaluator(self, use_mp):
        self._check_a_n(self.a_n)
        # pylint: disable=len-as-condition
        if len(self.a_n) == 0: # handle list and numpy array case
            return self.zero
        return _CosineSeriesEval(self, use_mp)

    def _approximate_constant(self, value, num, lobatto, use_mp, dps):
        with self.context(use_mp, dps) as ctx:
            self.a_n = [ctx.mpf(value)] + [ctx.zero] * (num-1)

    @classmethod
    def create_collocation_points(cls, num, lobatto=True, use_mp=False,
                                  dps=None):
        with cls.context(use_mp, dps):
            pi = mp.pi if use_mp else np.pi
            fl = mp.mpf if use_mp else float
            if lobatto:
                N = fl(num - 1)
                return [i*pi/N for i in range(0, num)]
            N = fl(num)
            return [(2*i-1)*pi/(2*N) for i in range(1, num+1)]


class _CosineSeriesEval(_TrigEval):
    r"""Evaluator class for CosineSeries.

    This evaluator can efficiently evaluate the function represented by the
    series expression and its derivatives up to arbitrary order.
    """

    def basis_values(self, diff_order):
        if diff_order % 2:
            return self.sin_values
        return self.cos_values

    def _get_diff_coeff_gen(self, diff_order):
        if diff_order % 2:
            return diff_cos_coeffs
        return diff_sin_coeffs


def evaluate_trig_mat(trig_func, xs, jn, use_mp, diff=0, scale=1.0):
    r"""Evaluate sin or cos at multiple frequencies and points to produce a matrix.

    This evaluates \f$\sin(j x_i)\f$ or \f$\cos(j x_i)\f$ for the given values
    of `j` and `x`. The result is a matrix with columns for the `j` values and
    rows for the `x` values.

    Evaluation is performed very efficiently using the vectorized
    trigonometric functions provided by numpy (in case ``use_mp==False``).

    Optionally, derivatives of these basis functions can be evaluated.

    @param trig_func
        String indicating which function to evaluate. Must be either ``"sin"``
        or ``"cos"``.
    @param xs
        Sequence of `x` values to evaluate the basis functions at.
    @param jn
        Sequence of frequencies defining which basis functions to evaluate.
        Must have the same length as `xs`.
    @param use_mp
        Whether to use mpmath computations.
    @param diff
        How often to differentiate the basis functions. Default is `0`.
    @param scale
        Scaling to apply due to derivatives being taken in the native domain
        instead of the physical one. Higher derivatives will be taken into
        account, i.e. the result is multiplied by ``scale**diff``.

    @return An (N x N) numpy matrix if `use_mp==False` or an (N x N) mpmath
        matrix for `use_mp==True`.
    """
    trig_func, sgn = _get_trig_diff_sign(trig_func, diff)
    f = _get_trig_func(trig_func, use_mp)
    if not use_mp:
        jn = np.asarray(jn)
        mat = f(np.outer(xs, jn))
        if diff:
            mat = sgn * float(scale)**diff * (mat * jn**diff)
        return mat
    if diff == 0:
        mat = [[f(j*x) for j in jn] for x in map(mp.mpf, xs)]
    else:
        s = sgn * mp.mpf(scale)**diff
        mat = [[s * j**diff * f(j*x) for j in jn] for x in map(mp.mpf, xs)]
    return mp.matrix(mat)


def evaluate_trig_series(trig_func, x, jn, use_mp, diff=0, scale=1.0):
    r"""Evaluate multiple sin or cos frequencies at one point.

    Same as evaluate_trig_mat() but evaluates all basis functions at the given
    single point `x`, resulting in a sequence of values rather than a matrix.

    All remaining arguments have the same meaning as in evaluate_trig_mat().

    @return An (N x 1) numpy array if `use_mp==False` or a list of `N` mpmath
        floats in case `use_mp==True`.
    """
    trig_func, sgn = _get_trig_diff_sign(trig_func, diff)
    f = _get_trig_func(trig_func, use_mp)
    if not use_mp:
        jn = np.asarray(jn)
        x = float(x)
        vec = f(x*jn)
        if diff:
            vec = sgn * float(scale)**diff * (vec * jn**diff)
        return vec
    x = mp.mpf(x)
    if diff == 0:
        vec = [f(j*x) for j in jn]
    else:
        s = sgn * mp.mpf(scale)**diff
        vec = [s * j**diff * f(j*x) for j in jn]
    return vec


def _get_trig_func(trig_func, use_mp):
    r"""Intepret a string and return the correct trig function.

    The mpmath version is returned in case ``use_mp==True``.
    """
    if trig_func == "sin":
        return mp.sin if use_mp else np.sin
    if trig_func == "cos":
        return mp.cos if use_mp else np.cos
    raise ValueError("`trig_func` must be either 'sin' or 'cos'.")


def _get_trig_diff_sign(trig_func, diff):
    r"""Determine the correct sign for a given trig derivative order.

    For odd derivative orders, sines and cosines are swapped.

    @return A 2-tuple of a string with the trigonometric function to use and a
        float indicating the sign of the derivative.

    @b Examples
    ```
        >>> _get_trig_diff_sign("sin", 0)
        "sin", 1.0

        >>> _get_trig_diff_sign("sin", 1)
        "cos", 1.0

        >>> _get_trig_diff_sign("sin", 2)
        "sin", -1.0

        >>> _get_trig_diff_sign("cos", 1)
        "sin", -1.0
    ```
    """
    if diff == 0:
        return trig_func, 1.0
    if diff % 2 == 0: # even derivative orders
        # switch sign every two derivative orders
        sgn = (-1)**(diff/2)
    else: # odd derivative orders (i.e. switch cos <-> sin)
        if trig_func == "sin":
            sgn = (-1)**((diff+1)/2 + 1)
            trig_func = "cos"
        else:
            sgn = (-1)**((diff+1)/2)
            trig_func = "sin"
    return trig_func, sgn


def diff_sin_coeffs(an, jn, use_mp, scale):
    r"""Compute coefficients of the derivative of a sine series.

    Given coefficients of a sine series, this computes the coefficients of a
    cosine series representing the derivative of the represented function.

    @param an
        Coefficients of the sine series.
    @param jn
        Indices of the active basis functions. Non-trivial only in case only
        the even or odd frequencies are used.
    @param use_mp
        Whether to use mpmath computations.
    @param scale
        Scaling to apply due to derivatives being taken in the native domain
        instead of the physical one. Each coefficient will be multiplied by
        this factor.
    """
    if use_mp:
        scale = mp.mpf(scale) # forces mpmath computation below
    return [(scale * j) * a for j, a in zip(jn, an)]


def diff_cos_coeffs(an, jn, use_mp, scale):
    r"""Compute coefficients of the derivative of a cosine series.

    Similar as diff_sin_coeffs(), but for a function represented as cosine
    series and with coefficients of the derivative as a sine series as result.
    """
    if use_mp:
        scale = mp.mpf(scale) # forces mpmath computation below
    return [(-scale * j) * a for j, a in zip(jn, an)]
