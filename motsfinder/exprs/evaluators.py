r"""@package motsfinder.exprs.evaluators

Base classes for simple or custom evaluators of numexpr.NumericExpression sub
classes.
"""

from __future__ import print_function
from abc import ABCMeta, abstractmethod
from builtins import range

from six import add_metaclass
import numpy as np
from mpmath import mp

from .common import _update_domains, _zero_function, is_zero_function


__all__ = [
    "TrivialEvaluator",
    "EvaluatorBase",
    "EvaluatorFactory",
]


class _Evaluator(object):
    r"""Base class for all evaluator classes.

    Users of the expression system who don't need to implement their own new
    expression don't need to deal with how evaluators are implemented at all.

    Basically, these are very light-weight objects that can store interim
    computations to efficiently compute values once requested.

    Evaluators are expected to be callable, which should evaluate the
    expression and return a numeric value. They should also have a
    `diff(x, n=1)` function evaluating the n'th derivative at the point `x`.
    Furthermore, a function `function(n=0)` should return a callable for the
    n'th derivative. Evaluators are free to expect a scalar `x` or some other
    input, like a vector or list/tuple.

    Each evaluator will have a `domain` attribute, which is populated with the
    original expressions domain at initialization time.
    """
    def __init__(self, expr, sub_evaluators=None):
        r"""Base class init for evaluators.

        @param expr
            The expression object for which this evaluator is created.
        @param sub_evaluators
            List of further evaluators required to evaluate this one.
            Currently this list is stored but not used.
        """
        self._domain = None
        self.domain = expr.domain  # calls the domain.setter
        self._sub_evaluators = [] if sub_evaluators is None else sub_evaluators

    @property
    def domain(self):
        r"""Domain of this evaluator."""
        return self._domain
    @domain.setter
    def domain(self, domain):
        self._domain = domain
        _update_domains(self)

    def is_zero_function(self, n=0):
        r"""Return whether the n'th derivative of this evaluator is identically zero."""
        # pylint: disable=unused-argument
        return False

    def store_domain(self, obj):
        r"""Store the domain of this evaluator on the given object."""
        obj.domain = self.domain
        _update_domains(obj)


class TrivialEvaluator(_Evaluator):
    r"""Convenience class to create simple evaluators.

    If your expression and the needed derivatives can be represented by simple
    lambda functions, this convenience class can be used to bypass the need to
    implement a full evaluator child class.
    """
    def __init__(self, expr, f, sub_evaluators=None):
        r"""Create an evaluator for a given function.

        @param expr
            The expression object for which this evaluator is created.
        @param f
            Either a simple callable representing the function to evaluate or
            a tuple/list of functions representing the function and a few of
            its derivatives. This function/these functions should respect the
            `use_mp` setting supplied to the
            numexpr.NumericExpression._evaluator() call which usually creates
            these evaluators.
        @param sub_evaluators
            List of further evaluators required to evaluate this one.
        """
        super(TrivialEvaluator, self).__init__(expr, sub_evaluators=sub_evaluators)
        if callable(f):
            f = (f,)
        self._f = f[0]
        self._derivs = f

    def is_zero_function(self, n=0):
        return is_zero_function(self._derivs[n])

    def __call__(self, x):
        r"""Evaluate the expression at a point x."""
        return self._f(x)

    def diff(self, x, n=1):
        r"""Evaluate the n'th derivative of the expression at a point x."""
        try:
            return self._derivs[n](x)
        except IndexError:
            if self.is_zero_function(-1):
                return 0
            raise NotImplementedError('Derivative for n = %s not implemented.' % n)

    def function(self, n=0):
        r"""Return a callable for the n'th derivative."""
        try:
            fn = self._derivs[n]
            if is_zero_function(fn):
                f = fn
            else:
                f = lambda x: fn(x) # pylint: disable=unnecessary-lambda
                self.store_domain(f)
            return f
        except IndexError:
            if self.is_zero_function(-1):
                return _zero_function
            raise NotImplementedError('Derivative for n = %s not implemented.' % n)


@add_metaclass(ABCMeta)
class EvaluatorBase(_Evaluator):
    r"""Base class for custom evaluator classes.

    Sub classes need to implement only _x_changed() and _eval(). The idea is
    that for more complex expressions, evaluation at a certain point `x`
    requires interim results that may be usable for computing e.g. derivatives
    of the function too. Often, we need the function and its derivatives at
    the same point for computing other functions (e.g. when computing
    \f$ \partial_x(f(x)g(x)) \f$ we need both functions and their first
    derivatives), which is why this class intercepts the evaluation
    (#__call__()) and diff() calls to check whether the point `x` has changed.
    If it has, your sub class can clear its cache and perform its interim
    computations at `x` in _x_changed(), which is skipped if `x` has not
    changed since the previous call. Then, you compute the requested value in
    \ref _eval() "_eval(n=0)", where you can use the protected member
    EvaluatorBase._x to evaluate the n'th derivative of the expression.
    """
    def __init__(self, expr, use_mp, sub_evaluators=None):
        super(EvaluatorBase, self).__init__(expr, sub_evaluators=sub_evaluators)
        ## Boolean indicating if computation should use `mpmath` (if `True`)
        ## or floating point operations.
        self.use_mp = use_mp
        ## Either `mpmath.mp` or `mpmath.fp`, depending on `use_mp`.
        self.ctx = expr.mpmath_context(use_mp)
        ## Convenience function that converts scalar values to floats or
        ## `mp.mpf`, depending on the `use_mp` setting.
        self.converter = mp.mpf if use_mp else float
        ## Parameter at which the next evaluation(s) should compute their values.
        self._x = None

    def __call__(self, x):
        r"""Compute the result of this evaluator at a given point x."""
        return self.diff(x, 0)

    def diff(self, x, n=1):
        r"""Evaluate the n'th derivative of the expression at a point x.

        The last point the expression was evaluated at is stored such that
        interim computations can be cached if desired.

        """
        self.set_x(x)
        return self._eval(n)

    def set_x(self, x):
        r"""Check if x has changed since the last call and trigger an update."""
        if isinstance(x, np.ndarray) or isinstance(self._x, np.ndarray):
            if np.array_equal(x, self._x):
                return False
        elif self._x == x:
            return False
        self._x = x
        self._x_changed(x)
        return True

    @abstractmethod
    def _x_changed(self, x):
        r"""Triggered to signal evaluation at x is about to happen.

        This is skipped if the previous evaluation was at `x` too, so that sub
        classes can recompute reusable interim results here.
        """
        pass

    @abstractmethod
    def _eval(self, n=0):
        r"""Compute the n'th derivative using previously cached interim results.

        This is only called after _x_changed() has been called for the
        protected variable EvaluatorBase._x, so you can use interim results
        computed in _x_changed() here.
        """
        pass

    def function(self, n=0):
        r"""Return a callable for the n'th derivative."""
        fn = lambda x: self.diff(x, n)
        self.store_domain(fn)
        return fn


class EvaluatorFactory(_Evaluator):
    r"""Convenience class to create slightly more complicated evaluators.

    If lambda functions can be generated on-the-fly for your expression and
    the needed derivatives, this convenience class can be used to avoid having
    to implement a full evaluator child class.

    As an example, see the implementation of basics.SimpleSinExpression.
    """
    def __init__(self, expr, factory, sub_evaluators=None):
        r"""Create an evaluator from a factory function.

        @param expr
            The expression object for which this evaluator is created.
        @param factory
            A function called with one parameter `n` to return a callable that
            computes the value of the n'th derivative.
        @param sub_evaluators
            List of further evaluators required to evaluate this one.
        """
        super(EvaluatorFactory, self).__init__(expr, sub_evaluators=sub_evaluators)
        if not callable(factory):
            raise TypeError("`factory` argument must be callable.")
        ## The given factory function.
        self._factory = factory
        ## Cached callables created by the factory.
        self._funcs = []
        self.function()

    def function(self, n=0):
        r"""Return a callable for the n'th derivative."""
        # Use a lambda to avoid side-effects of setting attributes on a
        # possibly user-supplied function.
        fn = self._get_func(n)
        f = lambda x: fn(x) # pylint: disable=unnecessary-lambda
        self.store_domain(f)
        return f

    def _get_func(self, n):
        r"""Cached call to the factory to create the n'th derivative."""
        for i in range(len(self._funcs), n+1):
            self._funcs.append(self._create_function(i))
        return self._funcs[n]

    def _create_function(self, n):
        r"""Non-cached call to the factory to create the n'th derivative."""
        f = self._factory(n)
        return f

    def __call__(self, x):
        r"""Compute the result of this evaluator at a given point x."""
        return self._funcs[0](x)

    def diff(self, x, n=1):
        r"""Evaluate the n'th derivative of the expression at a point x."""
        return self._get_func(n)(x)
