r"""@package motsfinder.exprs.basics

Collection of basic numexpr.NumericExpression subclasses.
"""

from builtins import range

import math

import sympy as sp
from mpmath import mp

from ..numutils import binomial_coeffs
from .numexpr import NumericExpression, SimpleExpression
from .evaluators import EvaluatorBase, EvaluatorFactory


__all__ = [
    "ConstantExpression",
    "IdentityExpression",
    "ScaleExpression",
    "ProductExpression",
    "ProductExpression2D",
    "DivisionExpression2D",
    "SumExpression",
    "BlendExpression2D",
    "EmbedExpression2D",
    "SimpleSinExpression",
    "SimpleCosExpression",
    "SinSquaredExpression",
]


class ConstantExpression(NumericExpression):
    """Represent an expression that is a constant.

    Represents an expression of the form \f$ f(x) = c = \mathrm{const} \f$.

    The value of the constant can be accessed through the `c` property.
    """

    def __init__(self, value=0, name='const'):
        r"""Init function.

        Args:
            value:  The constant value.
            name:   Name of the expression (e.g. for print_tree()).
        """
        super(ConstantExpression, self).__init__(name=name)
        ## The constant value this expression represents.
        self.c = value

    def _expr_str(self):
        return "%r" % self.c

    @property
    def nice_name(self):
        return "%s (%r)" % (self.name, self.c)

    def is_zero_expression(self):
        return self.c == 0

    def _evaluator(self, use_mp):
        c = self.c
        if c == 0:
            return (self.zero,)
        return (lambda x: c, self.zero)


class IdentityExpression(NumericExpression):
    r"""Identity expression with an optional multiplication factor.

    Represents an expression of the form \f$ f(x) = a x \f$.

    To multiply another expression use the ScaleExpression instead.
    """
    def __init__(self, a=1.0, name='Id'):
        r"""Init function.

        Args:
            a:      Factor to multiply the argument with.
            name:   Name of the expression (e.g. for print_tree()).
        """
        super(IdentityExpression, self).__init__(name=name)
        ## Factor to multiply the argument with.
        self.a = a

    def _expr_str(self):
        return ("a x, where a=%r" % self.a) if self.a != 1.0 else "x"

    @property
    def nice_name(self):
        if self.a != 1.0:
            return "%r * %s" % (self.a, self.name)
        return self.name

    def is_zero_expression(self):
        return self.a == 0

    def _evaluator(self, use_mp):
        a = self.a
        if a == 0:
            return (self.zero,)
        return (lambda x: a * x, lambda x: a, self.zero)


class ScaleExpression(NumericExpression):
    r"""Scale another expression by a factor.

    Represents an expression of the form \f$ f(x) = a g(x) \f$.
    """
    def __init__(self, expr, a, name='scale'):
        r"""Init function.

        Args:
            expr:   The expression to scale.
            a:      Factor to multiply the `expr` with.
            name:   Name of the expression (e.g. for print_tree()).
        """
        super(ScaleExpression, self).__init__(e=expr, domain=expr.domain,
                                              verbosity=expr.verbosity,
                                              name=name)
        ## Factor to scale the expression by.
        self.a = a

    def _expr_str(self):
        if self.a == 0.0:
            return "0"
        return "a f(x), where a=%r, f(x)=%s" % (self.a, self.e.str())

    @property
    def nice_name(self):
        return "%s (%r)" % (self.name, self.a)

    def is_zero_expression(self):
        return self.a == 0

    def _evaluator(self, use_mp):
        a = self.a
        if a == 0:
            return (self.zero,)
        e = self.e.evaluator(use_mp)
        def factory(n):
            return lambda x: a * e.diff(x, n)
        return EvaluatorFactory(self, factory, [e])


class ProductExpression(NumericExpression):
    r"""Multiply two expressions of one variable.

    Represents an expression of the form \f$ f(x) = g(x) h(x) \f$.
    """
    def __init__(self, expr1, expr2, name='mult'):
        r"""Init function.

        Args:
            expr1:  First expression.
            expr2:  Second expression.
            name:   Name of the expression (e.g. for print_tree()).
        """
        super(ProductExpression, self).__init__(e1=expr1, e2=expr2,
                                                domain=expr1.domain,
                                                name=name)

    def _expr_str(self):
        return "e1 * e2, where e1 = %s, e2 = %s" % (self.e1.str(), self.e2.str())

    def _evaluator(self, use_mp):
        e1 = self.e1.evaluator(use_mp)
        e2 = self.e2.evaluator(use_mp)
        fsum = mp.fsum if use_mp else math.fsum
        def f(x):
            return e1(x) * e2(x)
        def df(x):
            return e1.diff(x, 1) * e2(x) + e1(x) * e2.diff(x, 1)
        def ddf(x):
            e1x = e1(x)
            de1x = e1.diff(x, 1)
            dde1x = e1.diff(x, 2)
            e2x = e2(x)
            de2x = e2.diff(x, 1)
            dde2x = e2.diff(x, 2)
            return dde1x * e2x + 2 * de1x * de2x + e1x * dde2x
        def factory(n):
            if n == 0: return f
            if n == 1: return df
            if n == 2: return ddf
            coeffs = binomial_coeffs(n)
            return lambda x: fsum(coeffs[k] * e1.diff(x, n-k) * e2.diff(x, k)
                                  for k in range(0, n+1))
        return EvaluatorFactory(self, factory, [e1, e2])


class ProductExpression2D(NumericExpression):
    r"""Multiply two expressions of one or two variables each.

    Represents an expression of the form (e.g.)
    \f$ f(\mathbf{x}) = g(x) h(y) \f$ or
    \f$ f(\mathbf{x}) = g(y) h(\mathbf{x}) \f$, etc.
    """
    def __init__(self, expr1, expr2, variables=('both', 'both'), name='mult'):
        r"""Init function.

        Args:
            expr1:  First expression.
            expr2:  Second expression.
            variables: tuple/list of two elements each bein either `'both'`,
                    `0`, or `1`. The elements correspond to `expr1` and
                    `expr2`, respectively. `'both'` means that the expression
                    is a function of two variables, while `0` or `1` specify
                    the functions to depend only on `x` or `y`, respectively.
            name:   Name of the expression (e.g. for print_tree()).
        """
        super(ProductExpression2D, self).__init__(e1=expr1, e2=expr2, name=name)
        self._v1, self._v2 = variables
        if self._v1 not in ('both', 0, 1) or self._v2 not in ('both', 0, 1):
            raise TypeError("Expressions can only depend on element 0, 1 or both.")
        self.domain = self._get_domain()

    @property
    def v1(self):
        r"""Variable(s) of first expression (one of 'both', 0, 1)."""
        return self._v1

    @property
    def v2(self):
        r"""Variable(s) of second expression (one of 'both', 0, 1)."""
        return self._v2

    def _get_domain(self):
        if self._v1 == 'both':
            return self.e1.domain
        if self._v2 == 'both':
            return self.e2.domain
        result = [None] * (max(self._v1, self._v2) + 1)
        result[self._v1] = self.e1.domain
        result[self._v2] = self.e2.domain
        return result

    def _expr_str(self):
        v1 = 'x' if self._v1 == 'both' else 'x%d' % (self._v1 + 1)
        v2 = 'x' if self._v2 == 'both' else 'x%d' % (self._v2 + 1)
        return ("e1(%s) * e2(%s), where e1=%s, e2=%s"
                % (v1, v2, self.e1.str(), self.e2.str()))

    def _evaluator(self, use_mp):
        return _ProductExpression2DEval(self, use_mp)

class _ProductExpression2DEval(EvaluatorBase):
    r"""Evaluator for ProductExpression2D.

    Derivatives are implemented up to second order in both variables
    (separately) if supported by the sub-expressions.
    """
    def __init__(self, expr, use_mp):
        e1 = expr.e1.evaluator(use_mp)
        e2 = expr.e2.evaluator(use_mp)
        super(_ProductExpression2DEval, self).__init__(expr, use_mp, [e1, e2])
        self._v1 = None if expr.v1 == 'both' else expr.v1
        self._v2 = None if expr.v2 == 'both' else expr.v2
        self.e1 = e1
        self.e2 = e2

    def prepare_evaluation_at(self, pts, orders=(0,)):
        ordersXY = self.orders_for_2d_axes(orders)
        if self._v1 is None:
            self.e1.prepare_evaluation_at(pts, orders=orders)
        else:
            pts1d = self.unique_for_axis(self._v1, pts)
            self.e1.prepare_evaluation_at(pts1d, orders=ordersXY[self._v1])
        if self._v2 is None:
            self.e2.prepare_evaluation_at(pts, orders=orders)
        else:
            pts1d = self.unique_for_axis(self._v2, pts)
            self.e2.prepare_evaluation_at(pts1d, orders=ordersXY[self._v2])

    def _x_changed(self, x):
        pass

    def _evaluate_evaluator(self, e, x, v, n):
        if v is None:
            return e.diff(x, n)
        x = x[v]
        if n == 0:
            return e(x)
        nX, nY = self.unpack_2d_diff_order(n)
        if v == 0:
            return 0. if nY != 0 else e.diff(x, nX)
        if v == 1:
            return 0. if nX != 0 else e.diff(x, nY)

    def _eval(self, n=0):
        e1, e2 = self.e1, self.e2
        v1, v2 = self._v1, self._v2
        x = self._x
        e1x = self._evaluate_evaluator(e1, x, v1, n=0)
        e2x = self._evaluate_evaluator(e2, x, v2, n=0)
        if n == 0:
            return e1x * e2x
        if n == 1:
            dx1 = self._evaluate_evaluator(e1, x, v1, n=1)
            dx2 = self._evaluate_evaluator(e2, x, v2, n=1)
            return dx1 * e2x + e1x * dx2
        if n == 2:
            dy1 = self._evaluate_evaluator(e1, x, v1, n=2)
            dy2 = self._evaluate_evaluator(e2, x, v2, n=2)
            return dy1 * e2x + e1x * dy2
        if n == 3:
            dx1 = self._evaluate_evaluator(e1, x, v1, n=1)
            dx2 = self._evaluate_evaluator(e2, x, v2, n=1)
            dxx1 = self._evaluate_evaluator(e1, x, v1, n=3)
            dxx2 = self._evaluate_evaluator(e2, x, v2, n=3)
            return dxx1 * e2x + 2 * dx1 * dx2 + e1x * dxx2
        if n == 4:
            dx1 = self._evaluate_evaluator(e1, x, v1, n=1)
            dx2 = self._evaluate_evaluator(e2, x, v2, n=1)
            dy1 = self._evaluate_evaluator(e1, x, v1, n=2)
            dy2 = self._evaluate_evaluator(e2, x, v2, n=2)
            dxy1 = self._evaluate_evaluator(e1, x, v1, n=4)
            dxy2 = self._evaluate_evaluator(e2, x, v2, n=4)
            return dxy1*e2x + dy1*dx2 + dx1*dy2 + e1x*dxy2
        if n == 5:
            dy1 = self._evaluate_evaluator(e1, x, v1, n=2)
            dy2 = self._evaluate_evaluator(e2, x, v2, n=2)
            dyy1 = self._evaluate_evaluator(e1, x, v1, n=5)
            dyy2 = self._evaluate_evaluator(e2, x, v2, n=5)
            return dyy1 * e2x + 2 * dy1 * dy2 + e1x * dyy2
        raise NotImplementedError


class DivisionExpression2D(NumericExpression):
    r"""Divide one expression by another.

    Represents an expression of the form (e.g.)
    \f$ f(\mathbf{x}) = g(x)/h(y) \f$ or
    \f$ f(\mathbf{x}) = g(y)/h(\mathbf{x}) \f$, etc.
    """
    def __init__(self, expr1, expr2, variables=('both', 'both'),
                 singularity_handling='raise', eps=None, name='divide'):
        r"""Init function.

        Args:
            expr1:  First expression.
            expr2:  Second expression.
            variables: tuple/list of two elements each bein either `'both'`,
                    `0`, or `1`. The elements correspond to `expr1` and
                    `expr2`, respectively. `'both'` means that the expression
                    is a function of two variables, while `0` or `1` specify
                    the functions to depend only on `x` or `y`, respectively.
            singularity_handling: How to deal with the case when `expr2`
                    vanishes at a certain point. Possible values are:
                        * `"raise"` (default) raises a `ZeroDivisionError`
                        * `"zero"` returns `0.0`, which may be useful if you
                          (analytically) can determine that the respective
                          limit exists and is zero
                        * `"one"` returns `1.0` (rarely useful)
                        * `"+inf"` positive infinity
                        * `"-inf"` negative infinity
            eps:    Small value below which to treat `expr2` to be zero. By
                    default, only exactly zero is considered zero.
            name:   Name of the expression (e.g. for print_tree()).
        """
        super(DivisionExpression2D, self).__init__(e1=expr1, e2=expr2, name=name)
        self._v1, self._v2 = variables
        if self._v1 not in ('both', 0, 1) or self._v2 not in ('both', 0, 1):
            raise TypeError("Expressions can only depend on element 0, 1 or both.")
        if singularity_handling not in ("raise", "zero", "one", "+inf", "-inf"):
            raise TypeError("Singularity handling must be one of "
                            "'raise', 'zero', 'one', '+inf', '-inf'.")
        self._sing_handling = singularity_handling
        self._eps = eps
        self.domain = self._get_domain()

    def _get_domain(self):
        if self._v1 == 'both':
            return self.e1.domain
        if self._v2 == 'both':
            return self.e2.domain
        result = [None] * (max(self._v1, self._v2) + 1)
        result[self._v1] = self.e1.domain
        result[self._v2] = self.e2.domain
        return result

    def _expr_str(self):
        v1 = 'x' if self._v1 == 'both' else 'x%d' % (self._v1 + 1)
        v2 = 'x' if self._v2 == 'both' else 'x%d' % (self._v2 + 1)
        sing_handling = self._sing_handling
        if self._eps is not None:
            sing_handling += " (eps=%r)" % self._eps
        return ("e1(%s) / e2(%s), where e1=%s, e2=%s, singularity handling = %s"
                % (v1, v2, self.e1.str(), self.e2.str(), sing_handling))

    @property
    def v1(self):
        r"""Variable(s) of first expression (one of 'both', 0, 1)."""
        return self._v1

    @property
    def v2(self):
        r"""Variable(s) of second expression (one of 'both', 0, 1)."""
        return self._v2

    @property
    def singularity_handling(self):
        r"""Specified singularity handling (see #__init__())."""
        return self._sing_handling

    @property
    def eps(self):
        r"""Specified epsilon (see #__init__())."""
        return self._eps

    def _evaluator(self, use_mp):
        return _DivisionExpression2DEval(self, use_mp)

class _DivisionExpression2DEval(EvaluatorBase):
    r"""Evaluator for DivisionExpression2D.

    Derivatives are implemented up to second order in both variables
    (separately) if supported by the sub-expressions.
    """
    def __init__(self, expr, use_mp):
        e1 = expr.e1.evaluator(use_mp)
        e2 = expr.e2.evaluator(use_mp)
        super(_DivisionExpression2DEval, self).__init__(expr, use_mp, [e1, e2])
        self._v1 = None if expr.v1 == 'both' else expr.v1
        self._v2 = None if expr.v2 == 'both' else expr.v2
        if expr.singularity_handling == 'zero':
            self._sing_handling = mp.zero if use_mp else 0.
        elif expr.singularity_handling == 'one':
            self._sing_handling = mp.one if use_mp else 1.
        elif expr.singularity_handling == '+inf':
            self._sing_handling = mp.inf if use_mp else float('+inf')
        elif expr.singularity_handling == '-inf':
            self._sing_handling = -mp.inf if use_mp else float('-inf')
        else:
            self._sing_handling = None
        self._eps = expr.eps
        if self._eps is None:
            self._eps = 0.0
        self.e1 = e1
        self.e2 = e2

    def prepare_evaluation_at(self, pts, orders=(0,)):
        ordersXY = self.orders_for_2d_axes(orders)
        if self._v1 is None:
            self.e1.prepare_evaluation_at(pts, orders=orders)
        else:
            pts1d = self.unique_for_axis(self._v1, pts)
            self.e1.prepare_evaluation_at(pts1d, orders=ordersXY[self._v1])
        if self._v2 is None:
            self.e2.prepare_evaluation_at(pts, orders=orders)
        else:
            pts1d = self.unique_for_axis(self._v2, pts)
            self.e2.prepare_evaluation_at(pts1d, orders=ordersXY[self._v2])

    def _x_changed(self, x):
        pass

    def _evaluate_evaluator(self, e, x, v, n):
        if v is None:
            return e.diff(x, n)
        x = x[v]
        if n == 0:
            return e(x)
        nX, nY = self.unpack_2d_diff_order(n)
        if v == 0:
            return 0. if nY != 0 else e.diff(x, nX)
        if v == 1:
            return 0. if nX != 0 else e.diff(x, nY)

    def _eval(self, n=0):
        e1, e2 = self.e1, self.e2
        v1, v2 = self._v1, self._v2
        eps = self._eps
        x = self._x
        e1x = self._evaluate_evaluator(e1, x, v1, n=0)
        e2x = self._evaluate_evaluator(e2, x, v2, n=0)
        if e2x <= eps:
            if self._sing_handling is None:
                raise ZeroDivisionError
            return self._sing_handling
        if n == 0:
            return e1x/e2x
        if n == 1: # del_x
            dx1 = self._evaluate_evaluator(e1, x, v1, n=1)
            dx2 = self._evaluate_evaluator(e2, x, v2, n=1)
            return dx1/e2x - dx2*e1x/e2x**2
        if n == 2: # del_y
            dy1 = self._evaluate_evaluator(e1, x, v1, n=2)
            dy2 = self._evaluate_evaluator(e2, x, v2, n=2)
            return dy1/e2x - dy2*e1x/e2x**2
        if n == 3: # del_x del_x
            dx1 = self._evaluate_evaluator(e1, x, v1, n=1)
            dx2 = self._evaluate_evaluator(e2, x, v2, n=1)
            dxx1 = self._evaluate_evaluator(e1, x, v1, n=3)
            dxx2 = self._evaluate_evaluator(e2, x, v2, n=3)
            return (-2*dx1*dx2/e2x + 2*dx2**2*e1x/e2x**2 + dxx1 - dxx2*e1x/e2x)/e2x
        if n == 4: # del_x del_y
            dx1 = self._evaluate_evaluator(e1, x, v1, n=1)
            dx2 = self._evaluate_evaluator(e2, x, v2, n=1)
            dy1 = self._evaluate_evaluator(e1, x, v1, n=2)
            dy2 = self._evaluate_evaluator(e2, x, v2, n=2)
            dxy1 = self._evaluate_evaluator(e1, x, v1, n=4)
            dxy2 = self._evaluate_evaluator(e2, x, v2, n=4)
            return (-dx1*dy2/e2x - dx2*dy1/e2x + 2*dx2*dy2*e1x/e2x**2 + dxy1 - dxy2*e1x/e2x)/e2x
        if n == 5: # del_y del_y
            dy1 = self._evaluate_evaluator(e1, x, v1, n=2)
            dy2 = self._evaluate_evaluator(e2, x, v2, n=2)
            dyy1 = self._evaluate_evaluator(e1, x, v1, n=5)
            dyy2 = self._evaluate_evaluator(e2, x, v2, n=5)
            return (-2*dy1*dy2/e2x + 2*dy2**2*e1x/e2x**2 + dyy1 - dyy2*e1x/e2x)/e2x
        raise NotImplementedError


class SumExpression(NumericExpression):
    r"""Sum of two expressions is an optional coefficient for the second term.

    Represents an expression of the form \f$ f(x) = g(x) + a h(x) \f$.

    The coefficient \f$ a \f$ can be set/retrieved using the `coeff` property.
    """
    def __init__(self, expr1, expr2, coeff=1.0, name='add'):
        r"""Init function.

        Args:
            expr1:  First expression.
            expr2:  Second expression.
            coeff:  Coefficient for the second expression. Default is `1.0`.
            name:   Name of the expression (e.g. for print_tree()).
        """
        super(SumExpression, self).__init__(e1=expr1, e2=expr2, name=name)
        self._coeff = coeff
        self.domain = self.e1.domain

    @property
    def coeff(self):
        r"""Coefficient of the second term in the sum."""
        return self._coeff
    @coeff.setter
    def coeff(self, value):
        self._coeff = value

    def _expr_str(self):
        where = "e1=%s, e2=%s" % (self.e1.str(), self.e2.str())
        if self._coeff == 1.0:
            op = "+"
        elif self._coeff == -1.0:
            op = "-"
        else:
            op = "+ c"
            where += ", c=%r" % self._coeff
        return "e1 %s e2, where %s" % (op, where)

    @property
    def nice_name(self):
        if self._coeff == 1.0:
            return "%s (e1 + e2)" % self.name
        elif self._coeff == -1.0:
            return "%s (e1 - e2)" % self.name
        op = "+" if self._coeff >= 0 else "-"
        return "%s (e1 %s %r * e2)" % (self.name, op, abs(self._coeff))

    def _evaluator(self, use_mp):
        e1 = self.e1.evaluator(use_mp)
        e2 = self.e2.evaluator(use_mp)
        c = self._coeff
        if c == 1.0:
            def factory(n):
                return lambda x: e1.diff(x, n) + e2.diff(x, n)
        else:
            def factory(n):
                return lambda x: e1.diff(x, n) + c * e2.diff(x, n)
        return EvaluatorFactory(self, factory, [e1, e2])


class BlendExpression2D(SumExpression):
    r"""Blend between two expressions based on a third.

    Represents an expression of the form \f[
        f(\mathbf{x}) = f_1(\mathbf{x}) (1 - \beta(\mathbf{x}))
                        + \beta(\mathbf{x}) f_2(\mathbf{x}).
    \f]
    """
    def __init__(self, expr1, expr2, blend_expr, variables=('both', 'both', 'both'),
                 name='blend'):
        r"""Init function.

        Args:
            expr1:  First expression (\f$ f_1 \f$).
            expr2:  Second expression (\f$ f_2 \f$).
            blend_expr: Blending expression (\f$ \beta \f$).
            variables:  As in ProductExpression2D, but with a third element
                        for the `blend_expr`.
            name:   Name of the expression (e.g. for print_tree()).
        """
        v1, v2, v3 = variables
        first = ProductExpression2D(expr1,
                                    SumExpression(1, blend_expr, coeff=-1.0),
                                    variables=(v1, v3))
        second = ProductExpression2D(expr2, blend_expr, variables=(v2, v3))
        super(BlendExpression2D, self).__init__(expr1=first, expr2=second, name=name)


class EmbedExpression2D(NumericExpression):
    r"""Embed a 1D function into 2D along the x- or y-axis.

    Represents the expression
    \f$ \mathbf{x} \mapsto f(x) \f$ or
    \f$ \mathbf{x} \mapsto f(y) \f$.
    """
    def __init__(self, expr, axis=0, name='embedding'):
        r"""Init function.

        Args:
            expr:   Expression to embed.
            axis:   `0` to embed along the x-axis, `1` for the y-axis.
            name:   Name of the expression (e.g. for print_tree()).
        """
        super(EmbedExpression2D, self).__init__(e=expr, name=name)
        self._axis = axis
        self.domain = self._get_domain()

    def _get_domain(self):
        domain = [None, None]
        domain[self._axis] = self.e.domain
        return domain

    def _expr_str(self):
        return "f(x%d), where f(x)=%s" % (self._axis, self.e.str())

    def is_zero_expression(self):
        return self.e.is_zero_expression()

    def _evaluator(self, use_mp):
        e = self.e.evaluator(use_mp)
        axis = self._axis
        def factory(n):
            nX, nY = self.unpack_2d_diff_order(n)
            if (axis == 0 and nY != 0) or (axis == 1 and nX != 0):
                return self.zero
            n_axis = (nX, nY)[axis]
            return lambda x: e.diff(x[axis], n_axis)
        return EvaluatorFactory(self, factory, [e])


class SimpleSinExpression(SimpleExpression):
    r"""Sine expression with arbitrary derivatives."""
    def __init__(self, domain=None, name='sin'):
        def mp_factory(n):
            return (mp.sin, mp.cos, lambda x: -mp.sin(x), lambda x: -mp.cos(x))[n % 4]
        def fp_factory(n):
            return (math.sin, math.cos, lambda x: -math.sin(x), lambda x: -math.cos(x))[n % 4]
        super(SimpleSinExpression, self).__init__(
            mp_terms=mp_factory,
            fp_terms=fp_factory,
            desc="sin(x)",
            domain=domain,
            name=name
        )


class SimpleCosExpression(SimpleExpression):
    r"""Cosine expression with arbitrary derivatives."""
    def __init__(self, domain=None, name='cos'):
        def mp_factory(n):
            return (mp.cos, lambda x: -mp.sin(x), lambda x: -mp.cos(x), mp.sin)[n % 4]
        def fp_factory(n):
            return (math.cos, lambda x: -math.sin(x), lambda x: -math.cos(x), math.sin)[n % 4]
        super(SimpleCosExpression, self).__init__(
            mp_terms=mp_factory,
            fp_terms=fp_factory,
            desc="cos(x)",
            domain=domain,
            name=name
        )


class SinSquaredExpression(SimpleExpression):
    r"""Sine squared (`sin(x)**2`) expression with first two derivatives."""
    def __init__(self, domain=None, name='sin^2'):
        super(SinSquaredExpression, self).__init__(
            mp_terms=[lambda x: mp.sin(x)**2,
                      lambda x: mp.sin(2*x),
                      lambda x: 2*mp.cos(2*x)],
            fp_terms=[lambda x: math.sin(x)**2,
                      lambda x: math.sin(2*x),
                      lambda x: 2*math.cos(2*x)],
            desc="sin^2(x)",
            domain=domain,
            name=name
        )
