r"""@package motsfinder.exprs

Expression system for composing functions and efficiently evaluating them and
their derivatives.

The idea is to have each expression represent either a function (like
\f$ \sin(x) \f$ or a series \f$ \sum_{n=0}^N a_n \phi_n(x) \f$)
or a composite expression of one or more functions (like
\f$ f_1(x) + f_2(x) \f$, where \f$ f_i \f$ are other numeric expressions).

By implementing derivatives of each of the expressions, possibly in terms of
derivatives of composing sub-expressions, efficient evaluation of derivatives
of arbitrary expression trees can be achieved.

NOTE: Expression objects themselves cannot be evaluated. Instead, you take a
      *snapshot* of the current state and turn it into a callable object, here
      called an *evaluator* and subclasses of numexpr._Evaluator.

Upon creation of an evaluator, evaluators of all composing sub-expressions are
created. Also, at creation time, evaluators can be configured to either
evaluate using fast floating point operations or slower `mpmath` arbitrary
precision operations.

The distinction between expressions and their evaluators may resemble that of
`SymPy` symbolic expressions and their `lambdify`'ed counterparts, but here we
have, among other things, much more control of how derivatives are computed.

All expressions are supposed to be *picklable*, which means they can easily be
stored to disk and retrieved later, possibly even on a different machine.
The numexpr.NumericExpression and all child classes have a convenience method
numexpr.NumericExpression.save() for this purpose.
Loading such an expression using the class method
numexpr.NumericExpression.load() will restore the expression even if the used
classes have since been updated (provided they account for any required data
member changes in their `__setstate__()` implementation).
"""

from .cheby import Cheby
from .trig import SineSeries, CosineSeries
