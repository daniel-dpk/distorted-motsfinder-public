r"""@package motsfinder.ndsolve.bases.base

Base class(es) for spectral bases.

The _SpectralBasis class already does much of the work of e.g. composing the
operator matrix.
"""

from __future__ import print_function
from abc import ABCMeta, abstractmethod

from builtins import range
from six import add_metaclass
import numpy as np
from mpmath import mp, fp

from ...utils import lmap, isiterable
from ...exprs.series import transform_domains


__all__ = []


@add_metaclass(ABCMeta)
class _SpectralBasis(object):
    r"""Abstract basis class for pseudospectral bases.

    Sub classes should implement the following methods:
        * internal_domain() returning the native domain of the basis
        * _collocation_points() computing the collocation points in the native
          domain
        * _compute_deriv_mat() computing the matrix we get by evaluating all
          basis functions at all collocation points
        * solution_function() constructing the solution from a set of
          coefficients
        * evaluate_all_at() evaluating all (derivatives of the) basis
          functions at a given point
    """
    def __init__(self, domain, num):
        ## Physical domain of the problem.
        self._domain = domain
        ## Number of basis functions to include (\ie resolution).
        self._num = num
        ## Whether to compute everything using mpmath.
        self._use_mp = False
        ## The collocation points mapped to the physical domain.
        self._pts = None
        ## The collocation points in the basis' native domain.
        self._pts_internal = None
        ## The context (`mp` or `fp`) to use for computations.
        self.ctx = None
        ## Cached derivative matrices.
        self._deriv_mats = dict()
        ## Scaling to apply due to derivatives being taken in the native
        ## domain instead of the physical one.
        self._scl = None

    def init(self, use_mp=False):
        r"""This should be called during the solving process to prepare the
        internal state.

        The main purpose of splitting initialization from class construction
        is that during construction (which is done by the user of the
        `NDSolver` and not controlled by the `NDSolver` itself), we might not
        have the desired state of the mpmath arbitrary precision library
        context. Once the solver is entered, we know the desired `dps`
        (decimal places) and that is when all the computation should take
        place.
        """
        self._use_mp = use_mp
        self.ctx = mp if use_mp else fp
        self._pts_internal = self._collocation_points()
        self._pts = transform_domains(
            self.pts_internal, self.internal_domain(), self._domain,
            use_mp=use_mp
        )
        fl = mp.mpf if use_mp else float
        with mp.extradps(15 if use_mp else 0):
            a, b = lmap(fl, self.internal_domain())
            c, d = lmap(fl, self.domain)
            self._scl = (b-a)/(d-c)

    def transform(self, x, back=False):
        r"""Transform a point between the physical and native/internal domain.

        Args:
            back: Whether to transform back from the physical to the native
                domain. Default is `False`, i.e. we transform from the native
                internal domain to the physical domain.
        """
        from_domain = self.internal_domain()
        to_domain = self.domain
        if back:
            from_domain, to_domain = to_domain, from_domain
        return transform_domains([x], from_domain, to_domain,
                                 use_mp=self._use_mp)[0]

    @property
    def pts(self):
        r"""All collocation points on the physical domain."""
        return self._pts

    @property
    def pts_internal(self):
        r"""All collocation points on the internal domain.

        The internal domain is dependent on the basis used. For example, a
        Chebyshev polynomial basis is defined on the domain `[-1, 1]`, while a
        Fourier basis is defined on `[0, 2pi]`.
        """
        return self._pts_internal

    @property
    def num(self):
        r"""Number of basis functions after which to truncate the expansion.

        This is the same number as the number of collocation points and basis
        functions. It should NOT be confused with the formula symbol `N`,
        which is highly dependent on convention and often is chosen to be
        `num-1`, i.e. indices often run over `0, ..., N`.
        """
        return self._num

    @property
    def domain(self):
        r"""Physical domain of the basis.

        This is the domain on which to solve the problem.
        """
        return self._domain

    @abstractmethod
    def internal_domain(self):
        r"""Return the internal domain of the basis set."""
        pass

    @abstractmethod
    def _collocation_points(self):
        r"""Compute the collocation points on the internal (native) domain."""
        pass

    @abstractmethod
    def _compute_deriv_mat(self, n):
        r"""Compute the n'th derivative of all basis functions at all collocation points.

        Rows of the matrix should correspond to different collocation points
        and columns to the different basis functions.
        """
        pass

    @abstractmethod
    def solution_function(self, sol_coeffs):
        r"""Construct the solution function from a given set of coefficients."""
        pass

    @abstractmethod
    def evaluate_all_at(self, x, n=0):
        """Evaluate n'th derivative of all basis functions at x.

        The result is a list of N values.

        Args:
            x: Point at which to evaluate all basis function.
            n: Derivative order of the basis functions to use.
        """
        pass

    def get_closest_collocation_point(self, point):
        """Return the collocation point closest to the given point.

        The index of this point and the point itself is returned.
        The `point` should be given in the internal domain-space of this
        basis.
        """
        return min(enumerate(self.pts_internal), key=lambda p: abs(p[1]-point))

    def zero_op(self, as_numpy, real=True):
        r"""Return a zero matrix with dimensions compatible with current resolution.

        It will either be a numpy or mpmath matrix, depending on `as_numpy`.
        """
        if as_numpy:
            return np.zeros(shape=(self._num, self._num),
                            dtype=float if real else complex)
        return self.ctx.zeros(self._num, self._num)

    def sample_operator_func(self, func):
        r"""Sample the given function at all collocation points.

        The collocation points will be those in the physical domain.
        """
        return lmap(func, self.pts)

    def sample_operator_funcs(self, funcs):
        r"""Sample the list of functions at all collocation points.

        The collocation points will be those in the physical domain.
        """
        return [self.sample_operator_func(f) for f in funcs]

    def construct_operator_matrix(self, op, as_numpy=False):
        r"""Compute the operator matrix for the given differential operator.

        This matrix, `L`, is defined as the matrix that results from applying
        the differential operator of the given problem to the matrix `M`
        consisting of the basis functions evaluated at the collocation
        points, i.e. \f[
            M_{ij} = \phi_j(x_i),
        \f]
        where \f$\phi_j\f$ is the j'th basis function and \f$x_i\f$ the i'th
        collocation point.

        The differential operator is given by a list of coefficient functions
        `op`, where the elements correspond to the different values of `n`
        used for calling deriv_mat().

        For example (for a 1D basis):

        \code{.unparsed}
        op[0]: multiplies the function phi(x) itself
        op[1]: multiplies phi'(x)
        op[2]: multiplies phi''(x)
        ...
        \endcode

        Alternatively, `op` may be a callable. In this case, it will be called
        with the full list of collocation points to evaluate the operator at.
        It should return an array/list of operator components (as above), each
        with as many elements as the given points list has.

        Args:
            op: Operator coefficient functions or a callable to compute them
                all at once or a list of lists already containing the
                evaluated operator.
            as_numpy: Whether to construct a NumPy operator matrix (if `True`)
                or an mpmath matrix (default), the latter being much slower
                and requiring much memory.
        """
        if callable(op):
            sampled_op = op(np.array(self.pts) if as_numpy else self.pts)
        elif callable(op[0]):
            sampled_op = self.sample_operator_funcs(op)
        else:
            sampled_op = op
        real = self._is_op_real(sampled_op)
        L = self.zero_op(as_numpy, real=real)
        rows = cols = self._num
        if self._use_mp:
            fl = mp.mpf if real else mp.mpc
        else:
            fl = float if real else complex
        zeros = mp.zeros(1, cols) if self._use_mp else np.zeros(cols, dtype=fl)
        for row_idx in range(rows):
            row = zeros.copy()
            for n in range(len(sampled_op)):
                try:
                    op_coeff = sampled_op[n][row_idx]
                except TypeError:
                    op_coeff = sampled_op[n]
                if not self._use_mp:
                    op_coeff = fl(op_coeff)
                if op_coeff != 0:
                    op_row = self._compute_deriv_mat_row(n, row_idx)
                    if op_coeff is None:
                        # Coefficient failed to compute. Let's hope the basis
                        # is zero there...
                        if any(op_row):
                            raise RuntimeError("Operator (n=%d) evaluated to "
                                               "an invalid value at %s."
                                               % (n, self.pts[row_idx]))
                        continue
                    row += op_coeff * op_row
            if as_numpy:
                L[row_idx,:] = lmap(fl, row)
            else:
                L[row_idx,:] = row
        return L

    def _is_op_real(self, op):
        for coeff_values in op:
            if not isiterable(coeff_values):
                coeff_values = [coeff_values]
            for val in coeff_values:
                if isinstance(val, (np.complex, mp.mpc)):
                    return False
        return True

    def _compute_deriv_mat_row(self, n, row_idx):
        r"""Return just one row of the derivative matrix of a certain order.

        Should be overridden by child classes if they can optimize it.
        """
        return self.deriv_mat(n)[row_idx,:]

    def deriv_mat(self, n):
        r"""Return derivative matrix of derivative order `n`."""
        try:
            return self._deriv_mats[n]
        except KeyError:
            M = self._compute_deriv_mat(n)
            self._deriv_mats[n] = M
            return M


class _SpectralSeriesBasis(_SpectralBasis):
    r"""Spectral basis base class utilizing implemented series expansions.

    This convenience base class simplifies implementation of spectral bases
    for which a series expression has been implemented as a subclass of
    exprs.series.SeriesExpression.

    Child classes need only provide the methods:
        * get_series_cls()
        * evaluate_all_at()
        * _compute_deriv_mat()
    """
    def __init__(self, domain, num, lobatto=True):
        r"""Create a basis object with a given domain and resolution.

        @param domain
            (2-tuple/list)
            Physical domain as `(a, b)` of the problem. This is independent of
            the domain of the basis functions.
        @param num
            (int)
            Number of basis functions to expand the solution in. This defines
            the resolution (and computational cost).
        @param lobatto
            (boolean, optional)
            Whether to use the Gauss-Lobatto points (default), which include
            the two endpoints, or the Gauss points, which don't.
        """
        super(_SpectralSeriesBasis, self).__init__(domain=domain, num=num)
        ## Whether to use the Gauss-Lobatto points or the Gauss points.
        self._lobatto = lobatto
        ## The (dummy) series object (created during init()).
        self._series = None

    @abstractmethod
    def get_series_cls(self):
        r"""The series class for creating new objects of the series type."""
        pass

    def init(self, use_mp=False):
        self._series = self.series_factory()
        super(_SpectralSeriesBasis, self).init(use_mp=use_mp)

    def series_factory(self, a_n=(), **kw):
        series_cls = self.get_series_cls()
        series = series_cls(a_n, domain=self._domain, **kw)
        return series

    def _collocation_points(self):
        return self._series.collocation_points(
            num=self._num, lobatto=self._lobatto, internal_domain=True,
            use_mp=self._use_mp,
        )

    def internal_domain(self):
        return self._series.internal_domain()

    def solution_function(self, sol_coeffs):
        return self.series_factory(sol_coeffs, name='sol')
