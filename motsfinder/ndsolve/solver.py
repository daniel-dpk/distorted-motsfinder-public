r"""@package motsfinder.ndsolve.solver

The actual solver ndsolve() and the class NDSolver.

This is the main file of the pseudospectral solver containing the method(s)
needed to solve equations. Most of the actual work is distributed over the
basis base class bases.base._SpectralBasis and its derived classes like
bases.cheby.ChebyBasis.

The pseudospectral method is described in detail in [1] and [2].


@b Examples

Solving the following equation on the interval \f$ (1/2, 2) \f$
\f[ f''(x) - a b \sin(b x) f'(x) = a b^2 \cos(b x) f(x) \f]
with boundary conditions \f[
    f(1/2) = \exp(-a\cos(b/2)), \quad f(2) = \exp(-a\cos(2b))
\f]

The exact solution is \f$ f(x) = \exp(-a\cos(b x)) \f$. The numerical
solution can be obtained via:

```
sol = ndsolve(
    eq=((lambda x: -a*b**2*cos(b*x), lambda x: -a*b*sin(b*x), 1), 0),
    basis=ChebyBasis(domain=(0.5, 2), num=50),
    boundary_conditions=(
        DirichletCondition(x=0.5, value=exp(-a*cos(b/2))),
        DirichletCondition(x=2, value=exp(-a*cos(2*b))),
    )
)
```

@b References

[1] Boyd, J. P. "Chebyshev and Fourier Spectral Methods. Dover Publications
    Inc." New York (2001).

[2] Canuto, C., et al. "Spectral Methods: Fundamentals in Single Domains."
    Springer Verlag, 2006.
"""

from __future__ import print_function

from contextlib import contextmanager
import math

import numpy as np
from scipy import linalg
from mpmath import mp, fp

from ..utils import isiterable, lmap
from .common import _make_callable


__all__ = [
    "NDSolver",
    "ndsolve",
]


@contextmanager
def _noop_context(*args, **kwargs):
    r"""Dummy context used as placeholder."""
    yield


def _ensure_contextmanager(obj, attr):
    r"""Create any missing context managers as dummy context managers."""
    if isinstance(attr, (list, tuple)):
        for a in attr:
            _ensure_contextmanager(obj, a)
        return
    if not hasattr(obj, attr):
        setattr(obj, attr, _noop_context)


def _ensure_attr(obj, attr, attr_value):
    r"""Set/create an attribute in case it does not exist."""
    if not hasattr(obj, attr):
        setattr(obj, attr, attr_value)


class NDSolver(object):
    r"""Pseudospectral solver implemented as class.

    To call this solver, represent the (linear) operator as list of functions
    (or constants), choose and construct a basis, and construct the boundary
    conditions. Construct the `NDSolver` object with these and call its
    `solve` function to get the result as a NumericExpression created by the
    chosen basis.

    See the documentation of ndsolve() for more information and examples.
    """
    def __init__(self, eq, basis, boundary_conditions=None, use_mp=False,
                 dps=None, mat_solver='scipy.solve'):
        r"""Initialize the spectral solver.

        Args:
            eq: Definition of the differential equation. This should either be
                a 2-tuple containing the operator coefficient functions as
                first element and the inhomogeneity function or scalar as
                second, or a callable which evaluates all operator coefficient
                functions and the inhomogeneity at a given set of points.
                See ndsolve() for more information.
            basis: The spectral basis into which the solution should be expanded.
            boundary_conditions: An optional list of RobinCondition objects
                (or sub classes) imposing any desired conditions.
            use_mp: Whether to use arbitrary precision math operations
                (`True`) or faster floating point precision operations
                (`False`, default).
            dps: In case of ``use_mp==True``, this defines the number of
                decimal places to do the operations with. Default is to use
                the current global precision.
            mat_solver: Matrix solver method to use. If an mpmath method is
                chosen and ``use_mp==True``, the whole problem is solved using
                arbitrary precision operations. See ndsolve() for available solvers.
        """
        self._mat_solver = mat_solver
        self._basis = basis
        self._bcs = boundary_conditions
        self._use_mp = use_mp
        self._ctx = mp if use_mp else fp
        self._dps = dps if dps else self._ctx.dps
        _ensure_contextmanager(self._ctx, ('workprec','workdps','extraprec','extradps'))
        _ensure_attr(self._ctx, 'hypot', lambda x, y: math.sqrt(x**2 + y**2))
        with self.context():
            basis.init(use_mp=use_mp)
            op, inhom = self._evaluate_equation(eq)
            self._op = op
            self._inhom = inhom

    @property
    def basis(self):
        r"""The (initialized) basis object this solver was constructed with."""
        return self._basis

    def construct_operator_matrix(self):
        r"""Return the matrix resulting from applying the operator to the basis."""
        with self.context():
            return self._basis.construct_operator_matrix(
                self._op,
                as_numpy=not self._mat_solver.startswith('mp')
            )

    def eigenvalues(self):
        r"""Compute the eigenvalues of the operator.

        Note that currently, boundary conditions are ignored. This means that
        these need to be imposed by the right choice of the basis.

        @return `eigenvals, principal`, where `eigenvals` is a sequence of
            eigenvalues and `principal` is the principal eigenvalue (the one
            with the smallest real part).

        @b Notes

        To use this pseudospectral code for easy eigenvalue computation,
        consider the eigenvalue problem
        \f[
            \mathcal{L} u = \lambda u.
        \f]
        Expanding `u` into the chosen basis \f$\{\phi_k\}\f$, i.e.
        \f[
            u(x) = \sum_{k=0}^N a_k\,\phi_k(x),
        \f]
        and demanding that the eigenvalue equation holds at the collocation
        points, we get a system of equations
        \f[
            L_{ij} a_j = \lambda\,\Phi_{ij} a_j,
        \f]
        where
        \f[
            L_{ij} := (\mathcal{L}\phi_j)(x_i),
            \qquad
            \Phi_{ij} = \phi_j(x_i),
        \f]
        and the \f$x_i\f$ are the collocation points.
        Note that \f$\Phi_{ij}\f$ is invertible, so that the eigenvalue
        problem being solved here becomes
        \f[
            M_{ij} a_j = \lambda\, a_j,
        \f]
        where \f$M = \Phi^{-1} L\f$.
        """
        with self.context():
            L = self.construct_operator_matrix()
            Phi = self._basis.construct_operator_matrix(
                [1.0], as_numpy=True
            )
            Phi_inv = linalg.inv(Phi, overwrite_a=True)
            L = Phi_inv.dot(L)
            eigenvals = linalg.eigvals(a=L)
            principal = min(eigenvals, key=lambda v: v.real)
            return eigenvals, principal

    def _evaluate_equation(self, eq):
        r"""Interpret the given equation argument to get the operator and
        inhomogeneity.

        Returns:
            2-tuple containing as first element the operator coefficients
            evaluated at all collocation points, i.e. one list per derivative
            order. The second element returned will be the inhomogeneity
            evaluated at the collocation points.
        """
        pts = self._basis.pts
        if not self._use_mp:
            pts = np.array(pts, dtype=np.float64)
        if callable(eq):
            op, inhom = eq(pts)
            if not isiterable(inhom):
                inhom = [inhom] * self._basis.num
        else:
            op, inhom = eq
            op = [self._sample_func(f, pts) for f in op]
            inhom = self._sample_func(inhom, pts)
        return op, inhom

    def solve(self):
        r"""Solve the differential equation and return the solution."""
        with self.context():
            L = self.construct_operator_matrix()
            f = self._inhom
            L, f = self._impose_bcs(L, f)
            sol_coeffs = self.mat_solve(L, f)
            return self._basis.solution_function(sol_coeffs)

    def _impose_bcs(self, L=None, f=None, L_done=False, f_done=False):
        r"""Impose any stored boundary conditions."""
        bcs = self._bcs
        if L is None: L_done = True
        if f is None: f_done = True
        if not bcs or (L_done and f_done):
            return L, f
        if not isiterable(bcs):
            bcs = (bcs,)
        blocked = []
        for condition in bcs:
            L, f = condition.impose(self._basis, L, f, blocked, L_done,
                                    f_done, use_mp=self._use_mp)
        return L, f

    def _sample_func(self, func, pts):
        r"""Evaluate a function at a given set of points and return the results."""
        func = _make_callable(func, self._use_mp)
        return lmap(func, pts)

    def mat_solve(self, A, b):
        r"""Solve A x = b for x using the chosen solving method."""
        method = self._mat_solver
        if method == 'mp.lu_solve':
            x = mp.lu_solve(A, b)
        else:
            overwrite_a = overwrite_b = False
            if not isinstance(A, np.ndarray):
                if hasattr(A, 'tolist'): A = A.tolist()
                A = np.array(A, dtype=np.float64)
                overwrite_a = True
            if not isinstance(b, np.ndarray):
                if hasattr(b, 'tolist'): b = b.tolist()
                b = np.array(b, dtype=np.float64)
                overwrite_b = True
            if method == 'scipy.solve':
                x = linalg.solve(A, b, overwrite_a=overwrite_a, overwrite_b=overwrite_b)
            elif method == 'scipy.lstsq':
                x, _residues, _rank, _sigma = linalg.lstsq(
                    A, b, overwrite_a=overwrite_a, overwrite_b=overwrite_b
                )
            else:
                raise NotImplementedError("Solver method '%s' not implemented." % method)
        return x

    @contextmanager
    def context(self):
        r"""Set the mpmath decimal places to the configured value."""
        with self._ctx.workdps(self._dps):
            yield self._ctx


def ndsolve(eq, basis, boundary_conditions=None, use_mp=False, dps=None,
            mat_solver='scipy.solve'):
    r"""Numerically solve a linear differential equation.

    @param eq
            (array-like or callable)
            Defines the equation to solve. If an array-like is given, it must have
            two elements; the operator definition (LHS) and the inhomogeneity
            (RHS). The linear operator is defined by the coefficient functions of
            the different derivative orders, counted from low to high derivatives.
            If a callable is given, it will be called with the complete set of
            points to evaluate the operator and inhomogeneity at and it should
            return an array-like of resulting values for each derivative order and
            the inhomogeneity, in the same order as for the array-like argument.
            See examples below.
    @param basis
            The spectral basis to expand the solution into. This also defines
            the resolution and hence accuracy and speed of this call.
    @param boundary_conditions
            (condition or list of conditions, optional)
            Each condition should be a RobinCondition or a subclass like
            DirichletCondition or NeumannCondition.
    @param use_mp
            (boolean, optional)
            If `True`, evaluate the equation and all basis functions using
            `mpmath` arbitrary precision floating point operations. The full
            operator matrix will be computed with `mpmath`. Depending on the
            chosen `mat_solver`, each finished matrix row is immediately converted
            to floating point numbers in case a `SciPy` solver is specified. This
            significantly reduces memory footprint since storing a large `mpmath`
            matrix may become prohibitively expensive.
            Default is `False`.
    @param dps
            (int, optional)
            If `use_mp==True`, `dps` defines the number of decimal places to
            account for in arbitrary precision operations. The `mp` context is
            switched to this precision prior to evaluating the basis functions and
            the equation regardless of the value of `use_mp`. This has an effect
            on evaluation in case you forcibly use `mpmath` computations in the
            equation functions.
            Default is to use the current value of `mp.dps`.
    @param mat_solver
            (string, optional)
            Matrix solver method to use. Default is `'scipy.solve'`, which is a
            fast solver with floating point accuracy. Currently implemented
            solvers are:
                * `"scipy.solve"`
                * `"scipy.lstsq"`
                * `"mp.lu_solve"`

    @return
        The pseudospectral solution as a NumericExpression. To evaluate this
        solution, create an evaluator first, e.g.:

            f = solution.evaluator()
            y = f(2.5)

    @b Examples

    Solving the following equation on the interval \f$ (1/2, 2) \f$
    \f[  f''(x) - a b \sin(b x) f'(x) = a b^2 \cos(b x) f(x)  \f]
    with boundary conditions
    \f[  f(1/2) = \exp(-a\cos(b/2)), \quad f(2) = \exp(-a\cos(2b))  \f]

    The exact solution is \f$ f(x) = \exp(-a\cos(b x)) \f$. The numerical
    solution can be obtained via:

        sol = ndsolve(
            eq=((lambda x: -a*b**2*cos(b*x), lambda x: -a*b*sin(b*x), 1), 0),
            basis=ChebyBasis(domain=(0.5, 2), num=50),
            boundary_conditions=(
                DirichletCondition(x=0.5, value=exp(-a*cos(b/2))),
                DirichletCondition(x=2, value=exp(-a*cos(2*b))),
            )
        )


    An equation with the same boundary conditions and solution, but with an
    inhomogeneity is
    \f[  f''(x) - a b \sin(b x) f'(x) = a b^2 \cos(b x) \exp(-a\cos(b x))  \f]

    The numerical solution is obtained via:

    ```
    inhom = lambda x: a*b**2*cos(b*x)*exp(-a*cos(b*x))
    sol = ndsolve(
        eq=((0, lambda x: -a*b*sin(b*x), 1), inhom),
        basis=ChebyBasis(domain=(0.5, 2), num=50),
        boundary_conditions=(
            DirichletCondition(x=0.5, value=exp(-a*cos(b/2))),
            DirichletCondition(x=2, value=exp(-a*cos(2*b))),
        )
    )
    ```
    """
    solver = NDSolver(eq, basis, boundary_conditions, use_mp, dps, mat_solver)
    return solver.solve()
