r"""@package motsfinder.exprs.inverse

Expression for the inverse of a strictly monotonous function.


@b Examples

See the InverseExpression class.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator

from ..numutils import bracket_root, clip
from .numexpr import NumericExpression, SimpleExpression
from .evaluators import TrivialEvaluator


__all__ = [
    "InverseExpression",
]


_rtol = 4 * np.finfo(float).eps


class InverseExpression(NumericExpression):
    r"""Inverse function expression.

    @b Examples

    ```
        f = SimpleSinhExpression(domain=(-4, 4))
        finv = InverseExpression(f)
        ev = f.evaluator()
        evi = finv.evaluator()
        plot_1d(lambda x: x - evi(ev(x)), domain=f.domain)
    ```
    """
    def __init__(self, expr, samples=100, domain=None, expr_domain=None,
                 rtol=_rtol, atol=1e-15, name='inverse'):
        r"""Create a new inverse function expression from a function.

        @param expr
            Expression or callable to invert. If a callable is given, some
            features may not be available (like saving into a file or
            computing derivatives).
        @param samples
            Number of initial samples to create the estimated inverse with.
            Default is `100`.
        @param domain
            Domain of this expression. This needs to be equal to the range of
            the given `expr`, which is also its default value. Specify this to
            force the endpoints to evaluate to precise values, e.g. in case
            `expr` has some numerical noise.
        @param expr_domain
            Domain of the given expression. This is the domain to invert the
            expression on, which may be used to invert a non-invertable
            function (e.g. a sine) locally. Default is to take the domain from
            the expression.
        @param rtol
            Relative tolerance for the numerical inverse. Default is a few
            times machine epsilon.
        @param atol
            Absolute tolerance for the numerical inverse. Default is `1e-15`.
        @param name
            Name of the expression (e.g. for print_tree()).
        """
        if expr_domain is None:
            expr_domain = expr.domain
        super().__init__(domain=domain, name=name)
        ## Expression or function to invert.
        if hasattr(expr, "evaluator"):
            self.e = expr
        else:
            # This is not picklable
            def _mp_version(x): # pylint: disable=unused-argument
                raise NotImplementedError("mpmath version not implemented")
            self.e = SimpleExpression(
                domain=expr_domain,
                fp_terms=[expr],
                mp_terms=[_mp_version],
                desc="callable", name="callable",
            )
        ## Domain of function to invert (i.e. range of this inverse).
        self.expr_domain = expr_domain
        ## Number of samples for estimated inverse.
        self.samples = samples
        ## Relative tolerance for the computed inverse.
        self.rtol = rtol
        ## Absolute tolerance for the computed inverse.
        self.atol = atol

    def _expr_str(self):
        return "inverse of f(x), where f(x)=%s" % (self.e.str())

    def __get_endpoints(self, f=None):
        r"""Return the endpoint values of the function to invert.

        If a domain for this inverse function is specified, it will be used
        instead (with the order corrected such that we have approximately
        ``f(self.domain[0]) == fa``).
        """
        if f is None:
            f = _make_callable(self.e)
        a, b = self.expr_domain
        fa, fb = f(a), f(b)
        if self.domain is not None:
            if (fb-fa) * (self.domain[1]-self.domain[0]) < 0:
                # order of values in domain is reversed w.r.t. function slope
                fa, fb = reversed(self.domain)
            else:
                fa, fb = self.domain
        return fa, fb

    def _construct_inverse_function(self, f):
        rtol = self.rtol
        atol = self.atol
        a, b = self.expr_domain
        fa, fb = self.__get_endpoints(f=f)
        fi_rough = _estimate_inverse_function(
            f, samples=self.samples, f_domain=(a, b), endpoints=(fa, fb),
        )
        fw = max(abs(fb - fa), abs(fa), abs(fb))
        def fi(y):
            if abs(y - fa) < fw*rtol + atol:
                return a
            if abs(y - fb) < fw*rtol + atol:
                return b
            x0 = clip(fi_rough(y), a, b)
            err = lambda eps: y - f(x0+eps)
            err0 = err(0)
            if abs(err0) < fw*rtol + atol:
                return x0
            xa, xb = bracket_root(
                err, x0=0.0, step=max(abs(err0), 1e-4), domain=(a-x0, b-x0),
            )
            if xa == xb:
                return x0 + xa
            sol = brentq(err, xa, xb, xtol=atol)
            return x0 + sol
        return fi, list(sorted([fa, fb]))

    def _evaluator(self, use_mp):
        if use_mp:
            raise NotImplementedError("mpmath version not implemented")
        f = _make_callable(self.e)
        fi, domain = self._construct_inverse_function(f)
        def d1fi(y):
            x = fi(y)
            df = f.diff(x)
            return 1/df
        def d2fi(y):
            x = fi(y)
            df = [0.0] + [f.diff(x, n) for n in range(1, 3)]
            return -df[2]/df[1]**3
        def d3fi(y):
            x = fi(y)
            df = [0.0] + [f.diff(x, n) for n in range(1, 4)]
            return -df[3]/df[1]**4 + 3*df[2]**2/df[1]**5
        def d4fi(y):
            x = fi(y)
            df = [0.0] + [f.diff(x, n) for n in range(1, 5)]
            return -df[4]/df[1]**5 + 10*df[2]*df[3]/df[1]**6 - 15*df[2]**3/df[1]**7
        def d5fi(y):
            x = fi(y)
            df = [0.0] + [f.diff(x, n) for n in range(1, 6)]
            return (-df[5]/df[1]**6 + 15*df[2]*df[4]/df[1]**7 + 10*df[3]**2/df[1]**7
                    - 105*df[2]**2*df[3]/df[1]**8 + 105*df[2]**4/df[1]**9)
        def d6fi(y):
            x = fi(y)
            df = [0.0] + [f.diff(x, n) for n in range(1, 7)]
            return (-df[6]/df[1]**7 + 21*df[2]*df[5]/df[1]**8 + 35*df[3]*df[4]/df[1]**8
                    - 210*df[2]**2*df[4]/df[1]**9 - 280*df[2]*df[3]**2/df[1]**9
                    + 1260*df[2]**3*df[3]/df[1]**10 - 945*df[2]**5/df[1]**11)
        e = TrivialEvaluator(self, [fi, d1fi, d2fi, d3fi, d4fi, d5fi, d6fi])
        e.domain = domain
        return e


def _make_callable(func):
    r"""Return the function itself or create an evaluator for it."""
    try:
        return func.evaluator()
    except AttributeError:
        return func


def _filter_nonmonotone_data_points(xs, ys):
    r"""Remove data points that kill the monotonicity.

    In the first half of points, points that appear first have precedence over
    later points. In the second half, later ones override earlier ones. As a
    result, we prefer points closer to the boundaries to those in the middle.
    """
    diffs = np.diff(xs)
    xs_new = [xs[0]]
    ys_new = [ys[0]]
    middle = int(len(xs)/2)
    sgn = 1 if xs[0] < xs[-1] else -1
    for i, dx in enumerate(diffs):
        if sgn*dx > 0.0:
            xs_new.append(xs[i+1])
            ys_new.append(ys[i+1])
        elif i > middle:
            xs_new[-1] = xs[i+1]
            ys_new[-1] = ys[i+1]
    return xs_new, ys_new


def _estimate_inverse_function(f, samples, f_domain=None, endpoints=None):
    r"""Create a rough estimate for the inverse function."""
    if f_domain is None:
        f_domain = f.domain
    xs = np.linspace(*f_domain, samples)
    ys = [f(x) for x in xs]
    if endpoints is not None:
        fa, fb = endpoints
        if fa is not None:
            ys[0] = fa
        if fb is not None:
            ys[-1] = fb
    ys, xs = _filter_nonmonotone_data_points(ys, xs)
    if ys[0] > ys[-1]: # reversed order
        ys = list(reversed(ys))
        xs = list(reversed(xs))
    # PchipInterpolator guarantees monotonicity of interpolant
    interp = PchipInterpolator(ys, xs, extrapolate=True)
    return interp
