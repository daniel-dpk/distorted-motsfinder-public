r"""@package motsfinder.ndsolve.bcs

Classes for imposing boundary conditions.
"""

from __future__ import print_function

from builtins import range
import numpy as np
from mpmath import mp

from ..utils import isiterable, lmap
from .common import _make_callable


__all__ = [
    "DirichletCondition",
    "NeumannCondition",
    "RobinCondition",
]


class NDSolveError(Exception):
    r"""Raised for problems of the numerical task (like ill-conditioned
    boundary conditions)."""
    pass


class RobinCondition(object):
    r"""General Robin-type boundary condition.

    This class represents general Dirichlet, Neumann, or mixed (Robin) type
    boundary conditions suitable for 1D and 2D problems.

    After you have constructed such a condition object, it is used as an
    argument for the spectral solver (ndsolve() or NDSolver) in order to be
    imposed in the resulting matrix equation.
    """
    def __init__(self, x, alpha, beta, value, corners=True, add_rows=False):
        r"""Define the boundary condition.

        The general form of this boundary condition is \f[
            \alpha(x) u(x) + \beta(x) \partial_\nu u(x) = g(x),
        \f]
        where \f$ x \f$ is specified by the `x` argument (see below) and
        \f$ g(x) \f$ is given by `value`. The derivative of `u` is taken to be
        perpendicular to the direction of the line specified by `x`, i.e. it
        will be the outward pointing normal at the upper boundary and inward
        pointing normal at the lower boundary of the respective dimension.

        In 1D, we simply have \f$ \partial_\nu u(x) = u'(x) \f$.

        For a pure Dirichlet condition, set ``alpha=1, beta=0`` and for a pure
        Neumann condition ``alpha=0, beta=1``.

        Args:
            x: (float or tuple/list)
                Value at which to impose the condition. For 1D problems, this
                should simply be the `x`-value at which to impose the condition.
                For 2D problems, it should be a tuple/list specifying the
                `x=const` or `y=const` line to impose the condition at. For
                example, to define the line `x=5`, this argument should be set to
                `x=(5.0, None)`.
            alpha: (float or callable)
                Coefficient of the 'Dirichlet' part of the condition (see above).
            beta: (float or callable)
                Coefficient of the 'Neumann' part of the condition (see above).
            value: (float or callable)
                Value of the condition (see above).
            corners: (boolean, optional)
                Only relevant in 2D. If `True` (default), apply the condition at
                all collocation points along the specified axis including the
                boundary ones. If `False`, the boundary points are excluded. This
                may be important to make the resulting matrix equation a full-rank
                equation, since imposing multiple conditions at the same points
                (e.g. the corners) may make the matrix singular.
            add_rows: (boolean, optional)
                Whether to add or replace rows in the matrix and the inhomogeneity
                vector to impose the condition. Adding rows will create a
                non-square matrix, i.e. the system may become overdetermined and
                must be solved using e.g. a least squares method such as
                `scipy.lstsq`. Default is `False`, i.e. to replace rows, which is
                what you should do to get a determined system of equations.
        """
        ## Where to impose the condition
        self._x = x
        ## Callable or value multiplying the function
        self._alpha = alpha
        ## Callable or value multiplying the function's first derivative
        self._beta = beta
        ## Callable or value the function + derivative (as specified by
        ## `alpha` and `beta`) should attain.
        self._value = value
        ## Whether to include the corners/ends for the 2D case.
        self._corners = corners
        ## Whether to add more equations or replace existing ones.
        self._add_rows = add_rows
        if isiterable(x) and len(x) == 1:
            x = x[0]
        if isiterable(x):
            self._dim = len(x)
            if x.count(None) != self._dim - 1:
                raise ValueError("Condition must be imposed along one axis.")
            self._idx = [i for i, p in enumerate(x) if p is not None][0]
        else:
            if not corners:
                raise ValueError("Corners can't be skipped in 1D.")
            ## Auto-determined dimension of the problem
            self._dim = 1
            ## Axis along which to impose the condition
            self._idx = 0

    def impose(self, basis, L, f, blocked_indices, L_done=False, f_done=False,
               use_mp=False):
        r"""Impose the condition by modifying the operator and inhomogeneity.

        This should be called only during the solving process when the
        operator matrix `L` and inhomogeneity vector `f` have been built. The
        caller should supply the indices of rows (via `blocked_indices`) on
        which previous conditions have been placed to avoid overwriting them
        with the new conditions. The indices used here will be added to this
        list (i.e. the caller only needs to repeatedly supply this list,
        starting with an empty one).

        Args:
            basis: Spectral basis used during solving.
            L: Fully populated operator matrix.
            f: Inhomogeneity vector.
            blocked_indices: List which is updated by this method to store
                which equations have been replaced (in order not to replace a
                condition in later calls).
            L_done: Whether the operator matrix `L` has already been processed
                and should be left untouched. May be useful if caching of
                matrices with conditions already imposed is done, which may
                speed up computation significantly.
            f_done: Whether the inhomogeneity has already been processed and
                should be left untouched. Might be useful when re-using the
                same inhomogeneity (and same basis) to solve the same equation
                with different boundary conditions.
            use_mp: Whether to use arbitrary precision math operations (`True`) or
                faster floating point precision operations (`False`, default).
        """
        if L_done and f_done:
            return L, f
        alpha = _make_callable(self._alpha, use_mp=use_mp)
        beta = _make_callable(self._beta, use_mp=use_mp)
        value = _make_callable(self._value, use_mp=use_mp)
        num = basis.num
        idx = range(num)
        zero_row = mp.zeros(1, num) if use_mp else np.zeros(num)
        diff = self._idx + 1
        rows_to_append = []
        vals_to_append = []
        for i, x in self._get_points(basis):
            i = self._find_free_index(i, num-1, blocked_indices)
            if i is not None:
                blocked_indices.append(i)
            x_phys = basis.transform(x, back=False)
            a = alpha(x_phys)
            b = beta(x_phys)
            v = value(x_phys)
            if a == b == 0:
                continue
            if not L_done:
                Cx = DCx = zero_row
                if a != 0:
                    Cx = basis.evaluate_all_at(x, 0)
                if b != 0:
                    if self._dim > 2:
                        raise NotImplementedError("Neumann or mixed conditions not "
                                                  "implemented for dim > 2.")
                    DCx = basis.evaluate_all_at(x, diff)
                # NOTE: Originally, we had sparse indices (e.g. for basis functions
                #       left out from the set). This led to high complexity in
                #       index-types, i.e. the consecutive running indices vs.
                #       the subscript of the actual basis function. This might
                #       be required if we add further functionality to the
                #       system. At this point, one remainder is that the
                #       indices in `idx` could be sparse, whereas `j` can't.
                row = [a*Cx[j] + b*DCx[j] for j in range(len(idx))]
                if not any(row):
                    raise NDSolveError("Boundary condition numerically ill-conditioned. "
                                       "The condition might already be satisfied by the "
                                       "chosen basis or alpha = beta = 0.")
                if i is None:
                    rows_to_append.append(row)
                else:
                    self._replace_row(basis.ctx, L, i, row)
            if not f_done:
                if i is None:
                    vals_to_append.append([v])
                else:
                    f[i] = v
        if rows_to_append:
            L = self._append_rows(L, rows_to_append)
        if vals_to_append:
            f = self._append_rows(f, vals_to_append)
        return L, f

    def _replace_row(self, ctx, mat, n, row):
        r"""Replace a row of a matrix by a different row.

        This is a convenience function that handles the case of a NumPy matrix
        and that of an `mpmath` matrix.

        Args:
            ctx: `mp` or `fp`, the mpmath context to use if not a NumPy matrix.
            mat: The NumPy or mpmath matrix to replace a row in.
            n:   The row index of the row to replace.
            row: The new data to write into the row. May be a `list`.
        """
        if isinstance(mat, np.ndarray):
            mat[n,:] = lmap(float, row)
        else:
            mat[n,:] = ctx.matrix([row])

    def _append_rows(self, mat, rows):
        r"""Add rows to a NumPy or mpmath matrix.

        Args:
            mat: NumPy or mpmath matrix to add the rows to.
            rows: List of rows, each being an iterable containing the
                respective row's values.
        """
        if isinstance(mat, np.ndarray):
            mat = np.append(mat, [lmap(float, r) for r in rows], axis=0)
        elif isinstance(mat, list):
            if len(rows[0]) != 1:
                raise TypeError("Can only append 1D column to list.")
            mat.extend(r[0] for r in rows)
        else:
            mat = mp.matrix(mat.tolist() + rows)
        return mat

    def _find_free_index(self, i, max_i, blocked):
        r"""Return an index of a row we should replace next.

        This is called after the row to replace has been determined. It is
        responsible for checking that we don't replace a blocked row, i.e. one
        that already has a condition imposed.

        Args:
            i: Desired row that should be replaced if it is free.
            max_i: Maximum valid row index (total number of rows minus 1).
            blocked: List of row indices that are blocked and cannot replaced.
        """
        if self._add_rows:
            return None
        if len(blocked) > max_i:
            return 0
        if i not in blocked:
            return i
        add = 1
        while add <= max_i:
            if i+add <= max_i and i+add not in blocked:
                return i+add
            if i-add >= 0 and i-add not in blocked:
                return i-add
            add += 1

    def _transform(self, x, basis):
        """Transform a point from window to domain space.

        For multi-dimensional points, only the active dimension is returned,
        that is, this method always returns a floating type.
        """
        if self._dim == 1:
            return basis.transform(x, back=True)
        p = [0] * self._dim
        p[self._idx] = x[self._idx]
        p = basis.transform(p, back=True)
        return p[self._idx]

    def _filter_corners(self, pts, basis):
        r"""A generator to return only those points not on corner positions.

        Does nothing (i.e. no filtering) in the 1D case.
        """
        if self._dim == 1:
            for x in pts: yield x
        domains = [basis.get_basis(d).domain() for d in range(self._dim)]
        for i, p in pts:
            is_corner = True
            for d in range(self._dim):
                if d != self._idx and not (mp.almosteq(p[d], domains[d][0]) or mp.almosteq(p[d], domains[d][1])):
                    is_corner = False
            if not is_corner:
                yield i, p

    def _get_points(self, basis):
        r"""Return the row indices and points to impose the condition at.

        See _get_point_candidates() for more details.
        """
        pts = self._get_point_candidates(basis)
        if not self._corners:
            pts = list(self._filter_corners(pts, basis))
        return pts

    def _get_point_candidates(self, basis):
        r"""Collect a list of all points at which to impose the condition.

        Note that this will include the corners irrespective of the `corner`
        setting. The caller has to filter the corners out if desired.

        The result will be a list of 2-tuples containing the optimal row index
        (as detailed below) and exact point to impose a condition at. In the
        1D case, this list will only have one element (we want to impose the
        condition at one point only), whereas in the 2D case, we want to
        impose the condition along one axis, i.e. there will be multiple
        points.

        The points are chosen in the following way:
            * In the 1D case, we search for the collocation point closest to
              the point we want to impose the condition at and use that
              point's index (which will also be the row index in the operator
              matrix at which the equation has been evaluated for that
              collocation point). The point will be the one to impose the
              condition at, i.e. not necessarily the collocation point but
              it should be close to it.
            * In the 2D case, we first find the closest collocation point of
              the 1D basis for the fixed axis and then combine the fixed axis'
              value with all collocation points of the free axis to get a
              "line" of points to impose the condition at.

        Returns:
            A list of 2-tuples, each containing one index and one 1D or 2D
            point. The index will indicate the row in the matrix to replace
            with the condition evaluated at the returned point. The points
            will be returned in the native domain of the basis.
        """
        # x is the scalar value of the fixed axis transformed to the native
        # domain of the basis.
        x = self._transform(self._x, basis)
        points = basis.pts_internal # all collocation points
        basis_dim = len(points[0]) if isiterable(points[0]) else 1
        if basis_dim != self._dim:
            raise ValueError("Basis dimension does not match condition dimension.")
        if self._dim == 1:
            # We solving an ODE (i.e. one-dimensional).
            closest = basis.get_closest_collocation_point(x)[0]
            return [[closest, x]]
        # This gets the closest collocation point along the fixed axis.
        p1D = basis.get_basis(self._idx).get_closest_collocation_point(x)[1]
        ctx = basis.ctx
        # Collect all N-D grid points (with indices) along the one fixed axis.
        pts = [(i, list(p[:])) for i, p in enumerate(points) if ctx.almosteq(p[self._idx], p1D)]
        for p in pts:
            # The point `x` does not need to be a collocation point.
            p[1][self._idx] = x
        return pts


class DirichletCondition(RobinCondition):
    r"""A Dirichlet boundary condition.

    This represents a RobinCondition with `alpha==1` and `beta==0`.
    """
    def __init__(self, x, value=0, corners=True, add_rows=False):
        super(DirichletCondition, self).__init__(x, 1, 0, value, corners, add_rows)


class NeumannCondition(RobinCondition):
    r"""A Neumann boundary condition.

    This represents a RobinCondition with `alpha==0` and `beta==1`.
    """
    def __init__(self, x, value=0, corners=True, add_rows=False):
        super(NeumannCondition, self).__init__(x, 0, 1, value, corners, add_rows)
