r"""@package motsfinder.metric.base

Base classes for implementations of different metrics.

These base classes implement some of the functionality any metric needs to
provide such as computing Christoffel symbols. Child classes may override some
of these (notably computing the derivatives) for better accuracy/perfomance.

Good examples of implementations are the analytical metrics in
motsfinder.metric.analytical.
"""

from __future__ import print_function
from abc import ABCMeta, abstractmethod
from math import fsum
from six import add_metaclass

import numpy as np
from scipy import linalg
from scipy.misc import derivative

from .helpers import christoffel_symbols, christoffel_deriv
from .helpers import riemann_components


__all__ = []


class MetricTensor(object):
    r"""Metric tensor at a certain point in a spatial slice.

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
class _ThreeMetric(object):
    r"""Base class for metrics, \ie symmetric covariant 2-Tensor fields.

    The methods child classes have to implement are:
        * _mat_at() computing the metric tensor (matrix) at a given point

    The may also override the default implementations of diff() and
    diff_lnsqrtg() if analytical derivatives are available. Otherwise,
    numerical finite difference (FD) differentiation is performed.
    """
    def __init__(self, dx=1e-8, fd_order=3):
        r"""Base class init for metrics.

        Args:
            dx: Step length for numerical differentiation. Default is `1e-8`.
            fd_order: Finite difference approximation order for numerical
                differentiation. Default is `3`.
        """
        ## Step length for numerical differentiation.
        self.dx = dx
        ## Finite difference approximation order for numerical
        ## differentiation.
        self.fd_order = fd_order

    def at(self, point):
        r"""Return the metric tensor at a given point."""
        return MetricTensor(self._mat_at(point))

    @abstractmethod
    def _mat_at(self, point):
        r"""Compute the metric tensor matrix at a given point."""
        raise NotImplementedError

    def diff(self, point, inverse=False, diff=1, **kw):
        r"""Return derivatives in x,y,z directions at the given point.

        If `inverse==True`, the derivatives of the inverse
        metric \f$ g^{ij} \f$ are returned.

        The optional argument `dx` may be set to a floating point value to
        control numerical differentiation step-size. By default, the value of
        `dx` as set during object creation is used. Other arguments are passed
        to `scipy.misc.derivative` to modify numerical finite difference
        order.

        Note that subclasses may implement analytical differentiation, in
        which case `dx` and other optional arguments may be ignored.

        @param point
            3D point (tuple/list/array) at which to compute the derivatives.
        @param inverse
            Whether to compute the derivatives of the inverse metric (indices
            up).
        @param diff
            Derivative order. Default is 1.

        @return Multidimensional list with indices `i1, i2, ..., k, l`
            corresponding to \f$\partial_{i_1}\partial_{i_2}\ldots g_{kl}\f$
            (or the inverse components if ``inverse==True``).
        """
        if diff == 0:
            return self.at(point).inv if inverse else self.at(point).mat
        if diff != 1:
            raise NotImplementedError
        point = np.array(point)
        g0 = self.at(point)
        def f(d, axis):
            v = [0., 0., 0.]
            v[axis] = d
            g = g0 if d == 0 else self.at(point + v)
            return g.inv if inverse else g.mat
        # TODO: Scale `dx` with norm(point)?
        kw["dx"] = kw.get("dx", self.dx)
        kw["order"] = kw.get("order", self.fd_order)
        return [derivative(f, x0=0.0, n=diff, args=(i,), **kw) for i in range(3)]

    def diff_lnsqrtg(self, point, **kw):
        r"""Return x,y,z derivatives of ln(sqrt(det(g))).

        This computes the terms \f$ \partial_i \ln(\sqrt{g}) \f$
        and returns the results as a list (i.e. one element per value of `i`).

        Similar to diff(), child classes may implement more efficient
        analytical derivatives and ignore optional arguments such as `dx`.
        """
        point = np.array(point)
        g0 = self.at(point)
        def f(d, axis):
            v = [0., 0., 0.]
            v[axis] = d
            g = g0 if d == 0 else self.at(point + v)
            return np.log(np.sqrt(np.linalg.det(g.mat)))
        kw["dx"] = kw.get("dx", self.dx)
        return [derivative(f, x0=0.0, n=1, args=(i,), **kw) for i in range(3)]

    def christoffel(self, point):
        r"""Compute the Christoffel symbols of this 3-metric.

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
