r"""@package motsfinder.ndsolve.common

Utils used by multiple modules in motsfinder.ndsolve.
"""

__all__ = []


def _make_callable(func, use_mp):
    r"""Make sure a given object is callable.

    If `func` is a `NumericExpression`, an evaluator is returned. In this
    case, `use_mp` specifies whether the evaluator uses mpmath arbitrary
    precision arithmetics (if `True`) or faster floating point operations.

    `func` may also be a single numeric value, in which case a dummy function
    is created always evaluating to this value. If `func==None`, that value is
    set to zero.
    """
    try:
        func = func.evaluator(use_mp=use_mp)
    except AttributeError:
        pass
    if func is None:
        func = lambda x: 0.0
    if not callable(func):
        value = func
        func = lambda x: value
    return func
