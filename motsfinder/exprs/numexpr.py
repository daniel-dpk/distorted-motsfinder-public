r"""@package motsfinder.exprs.numexpr

Base of the NumericExpression system.

The idea is to have a notion of a numeric expression which is 'self aware' and
can e.g. produce efficient derivatives of itself. Furthermore, we want to be
able to build composite expressions out of other, more basic expressions.

In order to achieve this and to allow using the same expression in a fast,
purely floating point context and easily switch to a slower arbitrary
precision context, the definition and configuration of an expression is
decoupled from the so-called 'evaluator' objects.

Something similar could have been achieved using SymPy symbolic expressions,
but then we have no control over how the analytic derivatives of expressions
are actually computed (and they would need to be computed at runtime).

As a simple example, let's create a Chebyshev polynomial expression and
multiply it with a \f$ \sin^2(x) \f$ function:

~~~.py
cheby = Cheby([1., -.2, .3, .1], domain=(0, np.pi))
sin2 = SinSquaredExpression(domain=(0, np.pi))
expr = ProductExpression(cheby, sin2)
ev = expr.evaluator()
print("f(.5) =", ev(.5))
~~~

Note that the `plot_1d` command knows about `NumericExpression` objects, so
you could plot the expression with either of the following two lines:

~~~.py
plot_1d(ev)
plot_1d(expr)
~~~
"""

from __future__ import print_function

from contextlib import contextmanager
from abc import ABCMeta, abstractmethod
import os
import os.path as op

from six import add_metaclass, iteritems
import numpy as np
from mpmath import mp, fp

from ..pickle_helpers import prepare_dict, restore_dict
from .common import _update_domains, _zero_function
from .evaluators import TrivialEvaluator, EvaluatorFactory


__all__ = [
    "NumericExpression",
    "SimpleExpression",
]


def save_to_file(filename, data, overwrite=False, verbose=True,
                 showname='data', mkpath=True):
    r"""Save an object to disk.

    This uses `numpy.save()` to store an object in a file. Use
    load_from_file() to restore the data afterwards.

    @param filename
        The file name to store the data in. An extension ``'.npy'`` will be
        added if not already there.
    @param overwrite
        Whether to overwrite an existing file with the same name. If `False`
        (default) and such a file exists, a `RuntimeError` is raised.
    @param verbose
        Whether to print when the file was written. Default is `True`.
    @param showname
        Name to print in the confirmation message in case `verbose==True`.

    @b Notes

    The data will be put into a 1-element list to avoid creating 0-dimensional
    numpy arrays.
    """
    filename = op.expanduser(filename)
    if not filename.endswith('.npy'):
        filename += '.npy'
    if mkpath:
        os.makedirs(op.normpath(op.dirname(filename)), exist_ok=True)
    if op.exists(filename) and not overwrite:
        raise RuntimeError("File already exists.")
    np.save(filename, [data])
    if verbose:
        print("%s saved to: %s" % (showname, filename))


def load_from_file(filename):
    r"""Load an object from disk.

    If the object had been stored using save_to_file(), the result should be a
    perfect copy of the object.

    @b Notes

    This assumes the object is the only element of a list stored in the file,
    which will be the case if the file was created using save_to_file(). If
    the data is not a single-element list, it is returned as is.
    """
    filename = op.expanduser(filename)
    result = np.load(filename)
    if result.shape == (1,):
        return result[0]
    # Not a single value. Return as is.
    return result


class ExpressionWarning(UserWarning):
    """Warning issued when expressions might not evaluate as expected."""
    pass


def isclose(a, b, rel_tol=None, abs_tol=None, use_mp=False):
    r"""Test if two numbers agree within an absolute/relative tolerance.

    For floating point comparison (i.e. if `use_mp==False`), the default
    relative tolerance is `1e-9` and the absolute one `0.0`.
    """
    if use_mp:
        return mp.almosteq(a, b, rel_eps=rel_tol, abs_eps=abs_tol)
    if rel_tol is None:
        rel_tol = 1e-9
    if abs_tol is None:
        abs_tol = 0.0
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


@contextmanager
def _noop_context(*_args, **_kwargs):
    r"""Empty context manager used as placeholder."""
    yield


# Fix unpickling objects stored prior to the function being moved.
_zero = _zero_function


@add_metaclass(ABCMeta)
class NumericExpression(object):
    """Parent class for numeric expressions.

    The expression objects cannot be numerically evaluated by themselves.
    Instead, they can be used to create 'evaluators' for the current parameter
    values.

    They also support building a string representation of the complete
    expression, including any sub-expressions, and can be stored to disk and
    loaded back from disk.

    The methods a child has to override are:
        * _expr_str() returning a representation of the expression and its
          settings
        * _evaluator() creating callable evaluator objects
    """
    # pylint: disable=too-many-public-methods

    def __init__(self, domain=None, name=None, verbosity=1, **sub_exprs):
        r"""Base class init for numeric expressions.

        The ``**sub_exprs`` sub expressions given as keyword arguments here
        are stored in this object and can be accessed with the keys used here.
        They are used when traversing through a complete expression hierarchy
        in e.g. print_tree() or traverse_tree().

        Args:
            domain: (various, optional)
                Domain to store for this expression.
            name: (string, optional)
                Name for the expression. Can be useful to label expressions in
                a more complex expression tree to indicate their role/meaning.
                By default, the current class name is used as name.
            verbosity: (int, optional)
                Verbosity to be used by child classes through the `verbosity`
                attribute. Default is `1`.

        """
        self.__verbosity = verbosity
        ## Current domain stored for this expression.
        self._domain = None
        ## Domain getter/setter automatically updating attributes such as
        ## `domainX`, `domainY` for 2D expressions.
        self.domain = domain
        self.__sub_expressions = dict()
        self.__name = name if name else self.__class__.__name__
        self.set_sub_exprs(**sub_exprs)
        ## Pointer to the common._zero_function() function.
        self.zero = _zero_function
        ## Whether evaluators should adhere to the requested evaluation mode
        ## or override it.
        self._force_evaluation_mode = None

    @property
    def domain(self):
        r"""Domain on which this expression is defined."""
        return self._domain
    @domain.setter
    def domain(self, domain):
        self._domain = domain
        _update_domains(self)

    @property
    def name(self):
        r"""Name given to this instance of the expression."""
        return self.__name
    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def verbosity(self):
        r"""Verbosity setting, which may be used during complex computations."""
        return self.__verbosity
    @verbosity.setter
    def verbosity(self, verbosity):
        self.__verbosity = verbosity

    @property
    def nice_name(self):
        r"""More descriptive name, which may be overridden by sub classes."""
        return self.__name

    def traverse_tree(self, include_root=False, skip_zeros=False, parents=None):
        r"""Generator that walks through a complete expression tree.

        In each iteration, the returned values represent the current node's
        parents (as a list from root to immediate parent), its key under which
        it is stored in its parent, and the node itself.

        Args:
            include_root: Whether to include the root as first item. Default
                is `False`.
            skip_zeros: Whether to skip zero (i.e. unused) sub expressions.
                Default is `False`.
            parents: Optional list of parents of the root. Normally only used
                internally for the recursion.

        @b Examples
        \code
            for parents, name, expr in root_expr.traverse_tree():
                print("-"*len(parents), name)
        \endcode

        """
        if parents is None:
            parents = []
        if include_root:
            yield parents, "", self
        parents = parents + [self]
        for name, expr in iteritems(self.__sub_expressions):
            if not (skip_zeros and expr.is_zero_expression()):
                yield parents, name, expr
            for node in expr.traverse_tree(include_root=False,
                                           skip_zeros=skip_zeros,
                                           parents=parents):
                yield node

    def print_tree(self, root_name='root', nice_names=True, skip_zeros=False):
        r"""Print the whole expression tree.

        Each expression's key under which it is stored as sub expression will
        be shown as well as its actual name and the class name.

        Args:
            root_name: Key name to print for the root expression.
            nice_names: Whether to use the nice more descriptive name (when
                implemented) or the usually shorter abstract names.
            skip_zeros: Whether to skip zero (i.e. unused) sub expressions.
        """
        def _p(expr, name, parents=()):
            n = expr.nice_name if nice_names else expr.name
            print("%s%s [%s] <%s>" % (
                ". " * len(parents), name, n, type(expr).__name__
            ))
        _p(self, root_name)
        for parents, name, expr in self.traverse_tree(skip_zeros=skip_zeros):
            _p(expr, name, parents)

    def save(self, filename, overwrite=False, verbose=True):
        r"""Save the expression object to disk.

        Args:
            filename: The file name to store the data in. An extension
                ``'.npy'`` will be added if not already there.
            overwrite: Whether to overwrite an existing file with the same
                name. If `False` (default) and such a file exists, a
                `RuntimeError` is raised.
            verbose: Whether to print when the file was written. Default is
                `True`.
        """
        save_to_file(
            filename, self, overwrite=overwrite, verbose=verbose,
            showname="%s [%s]" % (self.nice_name, type(self).__name__)
        )

    @classmethod
    def load(cls, filename):
        r"""Static function to load an expression object from disk."""
        return load_from_file(filename)

    def __getstate__(self):
        r"""Return a picklable state object representing the whole expression.

        Some care is taken to ensure even `mpmath` constants and matrices
        successfully pickle/unpickle.
        """
        return prepare_dict(self.__dict__)

    def __setstate__(self, state):
        r"""Restore a complete expression from the given unpickled state."""
        self.__dict__.update(restore_dict(state))
        self.zero = _zero_function

    def __repr__(self):
        r"""Return a string representing the whole expression tree.

        This string may become relatively large for expressions containing
        much data (like series expansions with many coefficients).
        """
        cls = self.__class__.__name__
        return "<%s%s>" % (cls, self.str())

    def force_evaluation_mode(self, use_mp):
        r"""Override the evaluation mode of future evaluators.

        This might be useful in a complicated expression tree when you want
        e.g. all expressions to be computed using mpmath arbitrary precision
        arithmetics, except for one very expensive expression which should use
        floating point arithmetics.

        Args:
            use_mp: (``{None, True, False}``)
                If `None`, don't override the evaluation mode and create
                evaluators with the requested mode. If `True` or `False`,
                ignore the requested evaluation mode when creating an
                evaluator and use the value set here.
        """
        self._force_evaluation_mode = use_mp

    def evaluator(self, use_mp=False):
        r"""Create an evaluator for the expression in the current state.

        Use `use_mp` to control whether the evaluator will use floating point
        arithmetics (for `False`) or arbitrary precision mpmath computations
        (for `True`). Default is `False`.

        Args:
            use_mp: Boolean indicating whether the evaluator should use
                `mpmath` math operations or standard (and faster) floating
                point operations. This may be ignored if
                force_evaluation_mode() has been used to override this setting.
        """
        if self._force_evaluation_mode is not None:
            use_mp = self._force_evaluation_mode
        e = self._evaluator(use_mp=use_mp)
        if callable(e) and not hasattr(e, 'diff'):
            e = (e,)
        if isinstance(e, (list, tuple)):
            e = TrivialEvaluator(self, e)
        return e

    def store_domain(self, obj):
        r"""Store the domain of this expression on the given object."""
        obj.domain = self.domain

    def str(self):
        """Return the expression and any values of local parameters as a string."""
        return "(%s)" % self._expr_str()

    @abstractmethod
    def _expr_str(self):
        """String representing the expression with any parameter values.

        For example, if the expression is ``a + b x``, with ``a`` and ``b``
        parameters and ``x`` the variable, then this method should return for
        example:

            "a + b x, where a=0.3, b=0.5"

        If ``x`` is a sub-expressions, be sure to use its `str` method and
        not the `_expr_str`. For example:

            def _expr_str(self):
                return ("a + b x, where a=%r, b=%r, x=%s"
                        % (self.a, self.b, self.x.str()))
        """
        pass

    @abstractmethod
    def _evaluator(self, use_mp):
        r"""Child classes need to implement this and create their evaluator here."""
        pass

    def is_zero_expression(self):
        r"""Return whether this expression is zero and constant.

        Child classes should override this if they can determine whether
        they're zero. By default, all expressions will deny being zero.
        """
        return False

    @classmethod
    def mpmath_context(cls, use_mp):
        r"""Return the `mpmath.mp` or `mpmath.fp` contexts.

        Some of the features of the `mp` context are missing on the `fp`
        context. To make these two truly drop-in replacements in the code, the
        `fp` context will be endowed with the missing context handlers like
        `extradps()` etc.
        """
        # TODO: Changing the global fp object may not be too good of an idea...
        ctx = mp if use_mp else cls.__ensure_mp_contexts(fp)
        return ctx

    @classmethod
    def __ensure_mp_contexts(cls, ctx):
        r"""Ensure the given context can do everything we need.

        This may be useful to create an object that is interchangeable with
        `mpmath.mp` in most situations.
        """
        if not hasattr(ctx, 'extradps'):
            setattr(ctx, 'extradps', _noop_context)
        if not hasattr(ctx, 'extrarpec'):
            setattr(ctx, 'extraprec', _noop_context)
        if not hasattr(ctx, 'workdps'):
            setattr(ctx, 'workdps', _noop_context)
        if not hasattr(ctx, 'workrpec'):
            setattr(ctx, 'workprec', _noop_context)
        return ctx

    @classmethod
    @contextmanager
    def context(cls, use_mp, dps):
        r"""Convenience function to be used as context manager.

        This will automatically choose the correct context (`mp` or `fp`)
        based on the choice of `use_mp` and configure the desired decimal
        places.

        Args:
            use_mp: Whether to use `mp` (if `True`) or `fp`.
            dps:    Decimal places to use in `mp` computations.
        """
        ctx = cls.mpmath_context(use_mp)
        if not use_mp or dps is None:
            dps = mp.dps
        with ctx.workdps(dps):
            yield ctx

    def set_sub_exprs(self, **sub_exprs):
        r"""Set/replace sub expressions under public attributes of this object.

        Each of the ``**sub_exprs`` keyword arguments will be stored on this
        object such that it is accessible via the key name used here as
        attribute name on the object.

        Numeric values given here will be converted to ConstantExpression
        objects.
        """
        sub_exprs = dict((k, self.__ensure_expr(e)) for k, e in iteritems(sub_exprs))
        for k, e in iteritems(sub_exprs):
            setattr(self, k, e)
        self.__sub_expressions.update(sub_exprs)

    def __ensure_expr(self, expr):
        """Ensure an object is an expression, converting it if necessary.

        If `expr` is not an expression object, it is converted to a
        `ConstantExpression`.
        """
        if isinstance(expr, NumericExpression):
            return expr
        from .basics import ConstantExpression
        return ConstantExpression(expr)


class SimpleExpression(NumericExpression):
    r"""Base class for simple functions.

    Subclassing this class and providing a list of callables (e.g. lambda
    functions) for the 0'th to n'th derivative order in both floating point
    and `mpmath` versions allows for very simple creation of expressions based
    on simple functions.

    An example of such an expression is basics.SinSquaredExpression.

    As an alternative to supplying a fixed list of derivatives, you may
    instead supply a function that creates such callables for a given
    derivative order. This will be turned into an evaluators.EvaluatorFactory
    and can be used to easily implement arbitrarily high derivatives, as is
    done in basics.SimpleSinExpression.
    """
    def __init__(self, fp_terms, mp_terms, desc, domain=None, name=None):
        r"""Init function.

        Args:
            fp_terms: List or callable for the floating point implementations
                    of the various supported derivative orders. See the class
                    description or the examples basics.SinSquaredExpression
                    and basics.SimpleSinExpression for details.
            mp_terms: Same as `fp_terms`, but for the `mpmath` implementations.
            desc:   Abstract description/name of the kind of expression. This
                    is used when printing the expression.
            domain: Optional domain of the expression.
            name:   Name of the expression (e.g. for print_tree()).
        """
        super(SimpleExpression, self).__init__(domain=domain, name=name)
        self._desc = desc
        if callable(fp_terms) and callable(mp_terms):
            fp_terms = EvaluatorFactory(self, fp_terms)
            mp_terms = EvaluatorFactory(self, mp_terms)
        elif len(fp_terms) != len(mp_terms):
            raise TypeError("Floating and Mpmath terms should match. "
                            "Got different lengths.")
        self._fp = fp_terms
        self._mp = mp_terms

    def __getstate__(self):
        fp = self._fp
        mp = self._mp
        self._fp = self._mp = None
        state = super(SimpleExpression, self).__getstate__()
        self._fp = fp
        self._mp = mp
        return state

    def __setstate__(self, state):
        super(SimpleExpression, self).__setstate__(state)
        tmp = type(self)()
        self._fp = tmp._fp
        self._mp = tmp._mp

    def _expr_str(self):
        return self._desc

    def _evaluator(self, use_mp):
        return self._mp if use_mp else self._fp
