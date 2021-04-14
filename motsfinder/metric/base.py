r"""@package motsfinder.metric.base

Base classes for implementations of different metrics.

These base classes implement some of the functionality any metric needs to
provide such as computing Christoffel symbols. Child classes may override some
of these for better accuracy/perfomance.

Good examples of implementations are the analytical metrics in
motsfinder.metric.analytical.
"""

from __future__ import print_function
from abc import ABCMeta, abstractmethod
from math import fsum
from six import add_metaclass

import numpy as np
from scipy import linalg

from .helpers import christoffel_symbols, christoffel_deriv
from .helpers import riemann_components


__all__ = []


class _MetricTensorCache():
    r"""Stub to ensure forwards compatibility of saved results."""
    pass


class MetricTensor(object):
    r"""Metric tensor class at a given point.

    The metric tensor is represented as a matrix, which means that some basis
    has been chosen. Operations with vectors are only well-defined if they are
    given in components w.r.t. the same basis.

    There are convenience methods for raising and lowering indices of
    covectors and vectors, respectively, and calling this object with two
    vectors will compute the scalar product.

    The inverse matrix (representing the metric with "indices up") is computed
    on-demand and then cached for subsequent use.
    """
    __slots__ = ("_mat", "_inv",)
    def __init__(self, matrix):
        r"""Initialize this tensor with the given matrix representing the metric."""
        ## The metric matrix.
        self._mat = matrix
        ## The inverse matrix (computed and then cached when needed).
        self._inv = None

    @property
    def mat(self):
        r"""Internal numpy matrix representing the metric tensor."""
        return self._mat

    @property
    def inv(self):
        r"""Internal numpy matrix representing the inverse of the metric tensor."""
        if self._inv is None:
            self._inv = linalg.inv(self._mat)
        return self._inv

    def dot(self, v):
        r"""Multiply a vector by the metric tensor (\ie lower its index)."""
        return self.lower_idx(v)

    def raise_idx(self, covector):
        r"""Raise the index of a co-vector.

        This computes \f$ X^i = g^{ij} X_j \f$.
        """
        return self.inv.dot(covector)

    def lower_idx(self, vector):
        r"""Lower the index of a vector.

        This computes \f$ X_i = g_{ij} X^j \f$.
        """
        return self._mat.dot(vector)

    def norm(self, vector):
        r"""Compute the norm of a vector \wrt the metric.

        This computes \f$ \sqrt{g_{ij} v^i v^j} \f$.
        """
        return np.sqrt(self(vector, vector))

    def __call__(self, v1, v2):
        r"""Compute g(v1, v2) for two vectors v1 and v2."""
        return self.lower_idx(v1).dot(v2)

    def trace(self):
        r"""Compute the trace of the metric tensor."""
        return self._mat.trace()


@add_metaclass(ABCMeta)
class _GeneralMetric(object):
    r"""Base class for general metrics (\ie of any dimension and signature).

    The methods child classes have to implement are:
        * _mat_at() computing the metric tensor (matrix) at a given point
        * diff() computing derivatives of the metric at a given point
    """

    def __call__(self, point):
        return self.diff(point, diff=0)

    def at(self, point):
        r"""Return the metric tensor at a given point."""
        return MetricTensor(self._mat_at(point))

    @abstractmethod
    def _mat_at(self, point):
        r"""Compute the metric tensor matrix at a given point."""
        raise NotImplementedError

    @abstractmethod
    def diff(self, point, inverse=False, diff=1):
        r"""Return derivatives of the metric at the given point.

        If `inverse==True`, the derivatives of the inverse metric
        \f$g^{ij}\f$ are returned.

        @param point
            Point (tuple/list/array) at which to compute the derivatives.
        @param inverse
            Whether to compute the derivatives of the inverse metric (indices
            up).
        @param diff
            Derivative order. Default is 1.

        @return Multidimensional list with indices `i1, i2, ..., k, l`
            corresponding to \f$\partial_{i_1}\partial_{i_2}\ldots g_{kl}\f$
            (or the inverse components if ``inverse==True``).

        @b Notes

        Subclasses may use _compute_inverse_diff() to compute the derivatives
        of inverse metrics. This is not implemented as default action to allow
        these classes to handle caching the results if they so desire.

        A simple implementation of diff() might start with:
        ~~~.py
        def diff(self, point, inverse=False, diff=1):
            if inverse:
                return self._compute_inverse_diff(point, diff=diff)
            if diff == 0:
                return self._mat_at(point)
            # ... your code to compute derivatives ...
        ~~~
        """
        raise NotImplementedError

    def _compute_inverse_diff(self, point, diff=1):
        r"""Compute derivatives of the inverse metric from those of the metric."""
        if diff == 0:
            return linalg.inv(self._mat_at(point))
        dg = self.diff(point, diff=1)
        g_inv = self.diff(point, inverse=True, diff=0)
        if diff == 1:
            return -g_inv.dot(dg).dot(g_inv).swapaxes(0, 1)
        ddg = self.diff(point, diff=2)
        dg_inv = self.diff(point, inverse=True, diff=1)
        if diff == 2:
            terms1 = np.einsum('ijab,bc->ijac', ddg, g_inv)
            terms2 = np.einsum('iab,jbc->ijac', dg, dg_inv)
            terms3 = terms2.swapaxes(0, 1)
            # pylint: disable=invalid-unary-operand-type
            return -np.einsum('ab,ijbc->ijac', g_inv, terms1+terms2+terms3)
        raise NotImplementedError

    def diff_lnsqrtg(self, point):
        r"""Return derivatives of ln(sqrt(det(g))).

        This computes the terms \f$\partial_i \ln(\sqrt{g})\f$
        and returns the results as a list (i.e. one element per value of `i`).
        """
        g_inv = self.diff(point, inverse=True, diff=0)
        dg = self.diff(point, diff=1)
        return 0.5 * np.einsum('jk,ijk', g_inv, dg)

    def christoffel(self, point):
        r"""Compute the Christoffel symbols of this metric.

        @return NumPy array with indices `[a,b,c]` corresponding to
            \f$\Gamma^a_{bc}\f$.
        """
        g_inv = self.at(point).inv
        dg = self.diff(point, diff=1)
        return christoffel_symbols(g_inv, dg)

    def christoffel_deriv(self, point):
        r"""Compute the derivatives of the Christoffel symbols.

        @return NumPy array with indices `[a,b,c,d]` corresponding to
            \f$\partial_a \Gamma^b_{cd}\f$.
        """
        g_inv = self.at(point).inv
        dg_inv = self.diff(point, diff=1, inverse=True)
        dg = self.diff(point, diff=1)
        ddg = self.diff(point, diff=2)
        return christoffel_deriv(g_inv, dg_inv, dg, ddg)

    def ricci_tensor(self, point):
        r"""Compute the components of the Ricci tensor.

        This computes the components \f$R_{ab} := R^c_{\ a\,cb}\f$.
        """
        G = self.christoffel(point)
        dG = self.christoffel_deriv(point)
        ra = range(G.shape[0])
        def _ric(a, b):
            return fsum(riemann_components(G, dG, c, a, c, b) for c in ra)
        return np.array(
            [[_ric(a, b) for b in ra] for a in ra]
        )

    def ricci_scalar(self, point):
        r"""Compute the scalar curvature (Ricci scalar)."""
        g_inv = self.at(point).inv
        Ric = self.ricci_tensor(point)
        return g_inv.dot(Ric).trace()


class _ThreeMetric(_GeneralMetric):
    r"""Base class for 3-metrics on a spatial slice of spacetime.

    The methods that child classes have to implement are:
        * _mat_at() computing the metric tensor (matrix) at a given point
        * diff() computing derivatives of the metric at a given point

    Furthermore, child classes may implement the methods for returning the
    lapse function and shift vector field (along with methods returning their
    respective time derivatives). This will enable construction of a 4-metric
    that can be evaluated at points of the spatial slice.

    If a 4-metric should be constructed but it does not matter which
    particular choice of lapse and shift is made, you can implement the
    methods trivially by returning one of the `trivial_*` function objects as
    documented in the respective `get_*` method.
    """
    # pylint: disable=abstract-method

    def get_curv(self):
        r"""Return the extrinsic curvature belonging to this 3-metric.

        By default, we assume a time-symmetric slice, i.e. vanishing extrinsic
        curvature. Implementations of non-time-symmetric slices should provide
        a callable with signature ``curv(point, diff=n)``, where ``n`` is the
        derivative order.
        """
        return None

    def get_lapse(self):
        r"""Return the lapse function on the slice.

        This will be used e.g. when constructing a 4-metric from this
        3-metric. This function should return the `trivial_lapse` function
        object if the trivial lapse function (constant 1) should be used.
        For example:

        ```
            def get_lapse(self):
                return trivial_lapse
        ```
        """
        return None

    def get_shift(self):
        r"""Return the shift vector field on the slice.

        This will be used e.g. when constructing a 4-metric from this
        3-metric. This function should return the `trivial_shift` function
        object if the trivial shift vector (zero) should be used.
        For example:

        ```
            def get_shift(self):
                return trivial_shift
        ```
        """
        return None

    def get_dtlapse(self):
        r"""Return the time derivative of lapse.

        This will be used e.g. when constructing a 4-metric from this
        3-metric. This function should return the `trivial_dtlapse` function
        object if the trivial lapse function's time derivative (zero) should
        be used.
        """
        return None

    def get_dtshift(self):
        r"""Return the time derivative of shift.

        This will be used e.g. when constructing a 4-metric from this
        3-metric. This function should return the `trivial_dtshift` function
        object if the trivial shift vector's time derivative (zero) should be
        used.
        """
        return None


def trivial_lapse(point, diff=0):
    r"""Trivial lapse function set to constant 1.0.

    To use it, return this in your get_lapse() implementation.
    """
    if diff:
        return np.zeros(shape=[3] * diff)
    return 1.0


def trivial_dtlapse(point, diff=0):
    r"""Trivial lapse time derivative set to constant 0.0.

    To use it, return this in your get_dtlapse() implementation.
    """
    if diff:
        return np.zeros(shape=[3] * diff)
    return 0.0


def trivial_shift(point, diff=0):
    r"""Trivial shift vector field set to constant zero.

    To use it, return this in your get_shift() implementation.
    """
    return np.zeros(shape=[3] * (diff + 1))


def trivial_dtshift(point, diff=0):
    r"""Trivial shift time derivative set to constant zero vector.

    To use it, return this in your get_dtshift() implementation.
    """
    return np.zeros(shape=[3] * (diff + 1))
