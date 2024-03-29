r"""@package motsfinder.numutils

Miscellaneous numerical utilities and helpers.


@b Examples

```
    >>> binomial(5, 3)
    10
```
"""

from contextlib import contextmanager
import warnings

from scipy.linalg import LinAlgWarning
from scipy.integrate import fixed_quad, IntegrationWarning
from scipy.interpolate import interp1d
from scipy import optimize
import numpy as np
import sympy as sp


__all__ = [
    "nan_mat",
    "clip",
    "linear_interp",
    "binomial",
    "binomial_coeffs",
    "inf_norm1d",
    "raise_all_warnings",
    "try_quad_tolerances",
    "bracket_root",
    "find_root",
    "find_all_roots",
    "interpolate_root",
    "extrapolate_root",
    "IntegrationResult",
    "IntegrationResults",
    "NumericalError",
]


_golden = 1.61803398874989 # (1+sqrt(5))/2, the "golden ratio"


class NumericalError(Exception):
    r"""Exception raised for problems with numerical evaluation.

    For example, a tensor field class based on numerical data may raise this
    (or a subclass) if evaluation outside the numerical domain is requested.
    """
    pass


def nan_mat(shape):
    r"""Create a matrix of NaN values."""
    T = np.empty(shape)
    T[:] = np.nan
    return T


def clip(x, x_min, x_max):
    r"""Confine a value to an interval."""
    return max(x_min, min(x_max, x))


def linear_interp(x, x1, x2, y1, y2, extrapolate=True):
    r"""Linearly interpolate between two numbers.

    @param x
        Abscissa to interpolate to.
    @param x1,x2
        Abscissas of the two data points to interpolate between.
    @param y1,y2
        Ordinates of the two data points to interpolate between.
    """
    if not extrapolate:
        x = clip(x, x1, x2)
    return (y2-y1) * (x-x1)/(x2-x1) + y1


def binomial(n, k):
    r"""Compute the binomial coefficient n choose k."""
    return int(sp.binomial(n, k))


def binomial_coeffs(n):
    r"""Compute all binomial coefficients n choose k for 0 <= k <= n.

    The result is a list of integers
    \f[
        {n \choose 0}, {n \choose 1}, \ldots, {n \choose n}.
    \f]
    """
    return _BinomialCoeffs.all_coeffs(n)


class _BinomialCoeffs():
    r"""Helper class to simply cache the coefficient lists.

    This is used by binomial_coeffs() to re-use once computed lists.
    """

    __binomial_coeffs = []

    @classmethod
    def all_coeffs(cls, n):
        r"""Generate and cache the results for binomial_coeffs()."""
        while len(cls.__binomial_coeffs) <= n:
            nn = len(cls.__binomial_coeffs)
            coeffs = [binomial(nn, k) for k in range(n+1)]
            cls.__binomial_coeffs.append(coeffs)
        return cls.__binomial_coeffs[n]


def inf_norm1d(f1, f2=None, domain=None, Ns=50, xatol=1e-12):
    r"""Compute the L^inf norm of f1-f2.

    The `scipy.optimize.brute` method is used to find a candidate close to the
    global maximum difference. This is then taken as starting point for a
    search for the local maximum difference. Setting the number of samples
    `Ns` high enough should lead to the global maximum difference being found.

    @param f1
        First function. May also be a `NumericExpression`.
    @param f2
        Second function. May also be a `NumericExpression`. If not given,
        simply finds the maximum absolute value of `f1`.
    @param domain
        Domain ``[a, b]`` inside which to search for the maximum difference.
        By default, `f1` is queried for the domain.
    @param Ns
        Number of initial samples for the `scipy.optimize.brute` call. In case
        ``Ns <= 2``, the `brute()` step is skipped an a local extremum is
        found inside the given `domain`. Default is `50`.

    @return A pair ``(x, delta)``, where `x` is the point at which the maximum
        difference was found and `delta` is the difference at that point.
    """
    if domain is None:
        domain = f1.domain
    if not callable(f1):
        f1 = f1.evaluator()
    if f2 is None:
        f2 = lambda x: 0.0
    if not callable(f2):
        f2 = f2.evaluator()
    domain = list(map(float, domain))
    a, b = domain
    func = lambda x: (
        -float(abs(f1(float(x))-f2(float(x)))) if a <= x <= b else 0.
    )
    if Ns <= 2:
        bounds = [a, b]
    else:
        x0 = optimize.brute(func, [domain], Ns=Ns, finish=None)
        step = (b-a)/(Ns-1)
        bounds = [x0-step, x0+step]
    res = optimize.minimize_scalar(
        func, bounds=bounds, method='bounded',
        options=dict(xatol=xatol),
    )
    return res.x, -res.fun


def try_quad_tolerances(func, args=(), kwargs=None, tol_min=1e-11,
                        tol_max=1e-2, tol_steps=None, verbose=False):
    r"""Try to run a given function with increasing tolerance until integration succeeds.

    @param func
        Callable performing the integration. This should issue or raise an
        `IntegrationWarning` for too low tolerances. It is called as
        ``func(tol, *args, **kwargs)``.
    @param args
        Optional additional positional arguments for `func`.
    @param kwargs
        Optional additional keyword arguments for `func`.
    @param tol_min
        Minimal tolerance to try first. Default is `1e-11`.
    @param tol_max
        Maximum tolerance to allow. If `func` fails for this tolerance, no
        more trials are done and the `IntegrationWarning` warning is raised.
        Default is `1e-2`.
    @param tol_steps
        How many steps to try when going from `tol_min` to `tol_max`. Should
        be at least two. Default is to go roughly through each order of
        magnitude.
    @param verbose
        If `True`, print the tolerances as they are tried out. Default is
        `False`.
    """
    if tol_min > tol_max:
        raise ValueError("minimal tolerance greater than maximum tolerance")
    tol_min = np.log10(tol_min)
    tol_max = np.log10(tol_max)
    if tol_steps is None:
        tol_steps = max(2, int(round(tol_max-tol_min) + 1))
    tols = np.logspace(tol_min, tol_max, tol_steps)
    with raise_all_warnings():
        for tol in tols:
            if verbose:
                print("Trying with tol=%s" % tol)
            try:
                return func(tol, *args, **(kwargs or dict()))
            except IntegrationWarning:
                if verbose:
                    print("... failed with tol=%s" % tol)
                if tol == tols[-1]:
                    raise


class IntegrationResults():
    r"""Represents a sequence of multiple integration results.

    This class presents convenience methods to sum individual results and
    errors and check whether all results were computed without any warnings.

    @b Examples

    ```
        #res = ... # obtained via some means
        res[0].value     # value of first result
        res.sum()        # sum of all values
        res.sum(0, 2)    # sum first and third result
        res.sum(full_output=True).error # access errors, infos and warnings
        print("\n".join(str(r.error) for r in res)) # print individual errors
    ```
    """

    def __init__(self, *results, info=None, sum_makes_sense=True):
        r"""Create a results object from given results.

        Positional arguments can be either the ``full_output=True`` results of
        `scipy.integrate.quad()` calls or IntegrationResult objects.

        @param *results
            Results to collect.
        @param info
            Arbitrary info object to be stored with the results.
        @param sum_makes_sense
            Whether the sum of all results is a meaningful number. This
            controls if the total is printed in case of string conversion.
        """
        def _to_result(res):
            if not isinstance(res, IntegrationResult):
                res = IntegrationResult(*res)
            return res
        self._results = [_to_result(r) for r in results]
        ## Additional info supplied to the constructor.
        self.info = info
        ## Whether the sum of all results is a meaningful number.
        self.sum_makes_sense = sum_makes_sense

    @property
    def value(self):
        r"""Total value (sum of all results)."""
        return self.sum()

    @property
    def error(self):
        r"""Total error (sum of all errors)."""
        return self.sum(full_output=True).error

    def __len__(self):
        r"""Number of stored results."""
        return len(self._results)

    def __getitem__(self, key):
        r"""Access individual results."""
        return self._results[key]

    def __iter__(self):
        r"""Iterate through individual results."""
        return iter(self._results)

    def sum(self, *indices, full_output=False):
        r"""Combine results (sum values and optionally errors)."""
        if indices:
            results = [self._results[i] for i in indices]
        else:
            results = self._results
        result = sum([r.value for r in results])
        if not full_output:
            return result
        err = sum([r.error for r in results])
        infos = [r.info for r in results]
        if all([r.is_ok() for r in results]):
            warning = None
        else:
            warning = "\n".join([r.warning for r in results if r.warning])
        return IntegrationResult(result, err, infos, warning)

    def __repr__(self):
        result = "\n".join("[%s] %s" % (i, r) for i, r in enumerate(self._results))
        if self.sum_makes_sense and len(self._results) > 1:
            total = self.sum(full_output=True)
            result += "\nTotal: %s +- %s" % (total.value, total.error)
        if self.info:
            result += "\nInfo:\n%s" % (self.info,)
        return result

    def all_ok(self):
        r"""Return whether none of the results produced a warning."""
        return all([r.is_ok() for r in self._results])


class IntegrationResult():
    r"""Wrapper of the `full_output` of a `quad()` call."""

    def __init__(self, value, error, info, warning=None, mult=None):
        r"""Create a result object from the output of `quad()`.

        @param value
            Main result, i.e. the computed value.
        @param error
            The estimate of the error of the value.
        @param info
            Integration info object.
        @param warning
            Any warnings produced during integration.
        @param mult
            Factor by which to multiply the result and error.
        """
        if mult is not None:
            value = mult * value
            error = mult * error
        ## Computed value.
        self.value = value
        ## Estimated error.
        self.error = error
        ## Info object of the integration `quad()` call.
        self.info = info
        ## Warnings produced while integrating (`None` in case of no warnings).
        self.warning = warning

    def is_ok(self):
        r"""Return whether the result is OK and produced no warning."""
        return self.warning is None

    def __repr__(self):
        txt = "%s +- %s" % (self.value, self.error)
        if self.warning is not None:
            w = str(self.warning).split("\n")
            w = "\n       ".join(w)
            txt += "\nWarning: %s" % w
        return txt


def inverse_2x2_matrix_derivative(A, dA=None, ddA=None, diff=1):
    r"""Compute derivatives of the inverse of a 2x2 matrix.

    Given an invertible 2x2 matrix `A` with elements \f$a_{ij}\f$ and any
    needed derivatives w.r.t. two different variables, this returns the
    derivatives of the inverse of `A`.

    @param A
        The original matrix to compute the inverse of.
    @param dA
        Nested list or NumPy array with three indices where `dA[i][j][k]`
        contains the value of \f$\partial_i a_{jk}\f$.
    @param ddA
        Nested list or NumPy array with four indices where `dA[i][j][k][l]`
        contains the value of \f$\partial_i \partial_j a_{kl}\f$.
    @param diff
        Derivative order of the inverse matrix. If ``diff==0``, the inverse of
        `A` is returned and `dA` and `ddA` are not needed. `dA` is needed if
        ``diff > 0`` and `ddA` for ``diff > 1``. Default is `1`.

    @return NumPy array with two, three, or four axes depending on `diff`. The
        meaning of the indices such that `result[i1,i2,...,k,l]` contains the
        value \f$\partial_{i_1} \partial_{i_2} \ldots (B)_{kl}\f$, where `B`
        is the inverse \f$B = A^{-1}\f$.

    @b Notes

    Consider the matrix
    \f[
        A = \left(\begin{array}{@{}cc@{}}
                a & b\\
                c & d
            \end{array}\right).
    \f]
    The inverse is then given by
    \f[
        B := A^{-1} = \frac{1}{\det A} \left(\begin{array}{@{}cc@{}}
                b & -b\\
                -c & a
            \end{array}\right),
    \f]
    where \f$\det A = ad-bc\f$.
    The derivatives are easily computed using the chain and Leibniz' rule,
    which result in (using the shorthand notation \f$a_i := \partial_i a\f$
    and \f$a_{ij} := \partial_i \partial_j a\f$)
    \f[
        \partial_i B =
            - \frac{\partial_i \det A}{(\det A)^2}
                \left(\begin{array}{@{}cc@{}}
                    d & -b\\
                    -c & a
                \end{array}\right)
            + \frac{1}{\det A}
                \left(\begin{array}{@{}cc@{}}
                    d_i & -b_i\\
                    -c_i & a_i
                \end{array}\right)
    \f]
    and
    \f{eqnarray*}{
        \partial_i \partial_j B &=&
            \left(
                -\frac{\partial_i\partial_j\det A}{(\det A)^2}
                + 2 \frac{(\partial_i\det A)(\partial_j\det A)}{(\det A)^3}
            \right)
            \left(\begin{array}{@{}cc@{}}
                d & -b\\
                -c & a
            \end{array}\right)
        \\&&
            - \frac{\partial_i\det A}{(\det A)^2}
                \left(\begin{array}{@{}cc@{}}
                    d_j & -b_j\\
                    -c_j & a_j
                \end{array}\right)
            - \frac{\partial_j\det A}{(\det A)^2}
                \left(\begin{array}{@{}cc@{}}
                    d_i & -b_i\\
                    -c_i & a_i
                \end{array}\right)
        \\&&
            + \frac{1}{\det A}
                \left(\begin{array}{@{}cc@{}}
                    d_{ij} & -b_{ij}\\
                    -c_{ij} & a_{ij}
                \end{array}\right),
    \f}
    where
    \f{eqnarray*}{
        \partial_i \det A &=&
            a_i d + a d_i - b_i c - b c_i,
        \\
        \partial_i \partial_j \det A &=&
            a_{ij} d + a_i d_j + a_j d_i + a d_{ij}
            - b_{ij} c - b_i c_j - b_j c_i - b c_{ij}.
    \f}
    """
    if diff not in (0, 1, 2):
        raise NotImplementedError
    ra = range(2)
    A = np.asarray(A)
    dA = np.asarray(dA)
    a, b, c, d = A.flatten()
    det = a*d - b*c
    B = np.array([[d, -b], [-c, a]])
    if diff == 0:
        return 1/det * B
    # dA has axes [partial_i, row, col]
    # we want e.g.: da = [partial_x a, partial_y a]
    da, db, dc, dd = [dA[:,i,j] for i in ra for j in ra]
    ddet = np.array([da[i]*d + a*dd[i] - db[i]*c - b*dc[i] for i in ra])
    dB = np.array([[[dd[i], -db[i]], [-dc[i], da[i]]] for i in ra])
    if diff == 1:
        return np.array([-ddet[i]/det**2 * B + 1/det * dB[i] for i in ra])
    ddA = np.asarray(ddA)
    # these are the second derivatives dda[i,j] := partial_i partial_j a
    dda, ddb, ddc, ddd = [ddA[:,:,i, j] for i in ra for j in ra]
    # ddB has axes [partial_i, partial_j, row, col]
    ddB = np.array([[[[ddd[i,j], -ddb[i,j]], [-ddc[i,j], dda[i,j]]]
                     for j in ra] for i in ra])
    dddet = np.array(
        [[dda[i,j]*d + da[i]*dd[j] + da[j]*dd[i] + a*ddd[i,j]
          -ddb[i,j]*c - db[i]*dc[j] - db[j]*dc[i] - b*ddc[i,j]
          for j in ra] for i in ra]
    )
    return np.array(
        [[(-dddet[i,j]/det**2 + 2*ddet[i]*ddet[j]/det**3) * B
          - ddet[i]/det**2 * dB[j]
          - ddet[j]/det**2 * dB[i]
          + 1/det * ddB[i,j]
          for j in ra] for i in ra]
    )


def bracket_root(f, x0, step, domain=(float("-inf"), float("+inf")),
                 max_steps=10000, full_output=False, disp=False):
    r"""Simple (naive) sign change finder to bracket a root.

    Starting from an initial position (`x0`), this takes repeated steps until
    the function changes sign. The two x-values before and after the sign
    change are then returned (in undefined order) as a bracket. For any step
    taken, the step size is increased by the "golden ratio".

    If the function's absolute value increases, we continue to walk in the
    other direction. The step size is still increased, so that there is a
    chance to escape a local minimum.

    @param f
        Function to bracket a root of.
    @param x0
        Where to start walking.
    @param step
        Initial step length.
    @param domain
        Where to look for a sign change. If the search hits one of these
        boundaries, we either return the current (boundary) position or raise
        an error, depending on the `disp` parameter. Default is to not impose
        a boundary.
    @param max_steps
        Maximum number of steps to try. Default is `10000`.
    @param full_output
        If `True`, return the bracket and the two corresponding function
        values as ``a, b, f(a), f(b)``. Default is `False`, i.e. only return
        the bracket.
    @param disp
        If `True` and we get stuck at a domain boundary, raise a
        NumericalError. Default is `False`, i.e. return the boundary position
        as ``a, a, f(a), f(a)``.
    """
    a = x0
    fa = f(a)
    turned = False
    for _ in range(max_steps):
        b = clip(a + step, *domain)
        if a == b and turned: # we ran against the domain boundary
            if disp:
                break
            return (a, a, fa, fa) if full_output else (a, a)
        fb = f(b)
        if fa*fb < 0.0:
            return (a, b, fa, fb) if full_output else (a, b)
        step *= _golden
        if abs(fa) <= abs(fb):
            step = -step
            turned = True
        else:
            a = b
            fa = fb
    raise NumericalError(
        "Could not bracket a root within %s steps." % max_steps
    )


def find_root(f, x0, step, max_steps=10000, **kw):
    r"""Find a root of a function using Brent's method.

    This is a simple wrapper around `scipy.optimize.brentq()`, which first
    tries to automatically find a bracket required for the `brentq()` call.

    @param f
        Function (callable) to find a root of.
    @param x0
        Point to start searching for a sign change.
    @param step
        First step for the sign change search.
    @param max_steps
        Number of steps to take before givin up. Default is `10000`.
    @param **kw
        Further keyword arguments are passed to the `brentq()` call.
    """
    a, b = bracket_root(f, x0=x0, step=step, max_steps=max_steps, disp=True)
    if b < a:
        a, b = b, a
    return optimize.brentq(f, a, b, **kw)


def find_all_roots(xs, ys, func=None, full_output=False):
    r"""Find all roots of a sampled function.

    This is used for functions that are already sampled at a grid of points
    `xs` with function values `ys`. It is assumed that this grid is dense
    enough such that all roots appear as sign changes in consecutive values in
    `ys` (and hence that the roots all have odd multiplicity).

    @return List of roots. Empty list if no sign change occurs in `ys`.

    @param xs,ys
        Precomputed abscissas and ordinates of `func`.
    @param func
        Callable for fine-tuning the root search. If not specified, an
        interpolating function will be generated from the `xs` and `ys` data.
    @param full_output
        Whether to output the roots only (`False`, default) or the roots and
        interpolating function (`True`).
    """
    if len(xs) == 0:
        return ([], None) if full_output else []
    if func is None:
        kind = 'linear' if len(xs) < 4 else 'cubic'
        func = interp1d(xs, ys, kind=kind, fill_value="extrapolate")
    roots = []
    for i in range(1, len(ys)):
        if ys[i]*ys[i-1] < 0: # sign change => root here
            x0 = optimize.brentq(func, a=xs[i-1], b=xs[i])
            roots.append(x0)
    return (roots, func) if full_output else roots


def interpolate_root(xs, ys, guess, step, kind="cubic", full_output=False):
    r"""Interpolate data to estimate a root.

    @param xs,ys
        The data points (x- and y-values). The y-values need to change sign.
    @param guess
        The point to start searching for a root.
    @param step
        The first step size for searching for a root.
    @param kind
        Interpolation kind. Default is ``"cubic"``.
    @param full_output
        Whether to output the root only (`False`, default) or the root and
        interpolating function (`True`).
    """
    if not len(xs):
        return (None, None) if full_output else None
    f = interp1d(xs, ys, kind=kind, fill_value="extrapolate")
    r = find_root(f, x0=guess, step=step)
    return (r, f) if full_output else r


def extrapolate_root(xs, ys, guess=None, at_end=True, kind="cubic",
                     full_output=False):
    r"""Extrapolate data to estimate a root.

    @param xs,ys
        The data points (x- and y-values). These need not have a sign change.
    @param guess
        The point to start searching for a root in the interpolant. If not
        given, take the last x-interval as first guess.
    @param at_end
        If `True` (default), try to find a root *after* the data. Else, try to
        find it *before* the data.
    @param kind
        Interpolation kind. Default is ``"cubic"``.
    @param full_output
        Whether to output the root only (`False`, default) or the root and
        interpolating function (`True`).
    """
    if not len(xs):
        return (None, None) if full_output else None
    if not at_end:
        xs = list(reversed(xs))
        ys = list(reversed(ys))
    f = interp1d(xs, ys, kind=kind, fill_value="extrapolate")
    if guess is None:
        guess = 2*xs[-1] - xs[-2]
    r = find_root(f, x0=xs[-1], step=guess - xs[-1])
    return (r, f) if full_output else r


def _fixed_quad_abscissas(a, b, n):
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


@contextmanager
def raise_all_warnings():
    r"""Context manager for turning numpy and native warnings into exceptions.

    For example:
    ```
        with raise_all_warnings():
            np.pi / np.linspace(0, 1, 10)
    ```
    Without the `raise_all_warnings()` context, the above code would just
    issue a warning but otherwise run fine. This allows catching the exception
    to act upon it, e.g.
    ```
        with raise_all_warnings():
            try:
                np.pi / np.linspace(0, 1, 10)
            except FloatingPointError:
                print("Could not compute.")
    ```
    """
    old_settings = np.seterr(divide='raise', over='raise', invalid='raise')
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=IntegrationWarning)
            warnings.filterwarnings('error', category=LinAlgWarning)
            yield
    finally:
        np.seterr(**old_settings)
