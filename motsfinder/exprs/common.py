r"""@package motsfinder.exprs.common

Utils used by multiple modules in motsfinder.exprs.
"""

from ..utils import isiterable


__all__ = [
    "is_zero_function",
]


def _zero_function(x):
    """Constant function 0.

    This is used in the expression system to indicate the zero function, such
    that we know, e.g., that derivatives vanish too.
    """
    # pylint: disable=unused-argument
    return 0


def is_zero_function(func):
    r"""Check whether a given function is the zero function.

    This checks the identity of the given function with a particular zero
    function. This can be useful if expressions/evaluators actually
    use/return this _zero_function() function object when they know this is
    correct.
    """
    return func is _zero_function


def _del_attr(obj, attr):
    r"""Delete an attribute if it exists on an object."""
    try:
        delattr(obj, attr)
    except AttributeError:
        pass


def _update_domains(obj):
    r"""(Re-)Generate `domainX/Y/Z` attributes from the current domain.

    This is a convenience for higher (up to 3) dimensional functions where a
    stored domain of the form (e.g.)

        obj.domain = ((a1, b1), (a2, b2))

    is used to create/update the attributes

        obj.domainX = (a1, b1)
        obj.domainY = (a2, b2)
    """
    domain = obj.domain
    if domain is None or not isiterable(domain[0]):
        for x in 'XYZ':
            _del_attr(obj, "domain%s" % x)
        return
    for d, x in zip(domain, 'XYZ'):
        setattr(obj, "domain%s" % x, d)
