r"""@package motsfinder.metric.discrete.numerical

Numerical low-level computations and helpers.

These are used by e.g. the .patch.DataPatch classes to perform their
interpolation and differentiation on grids of data.
"""

import itertools
from operator import add

import numpy as np
from scipy.interpolate import lagrange

from ...numutils import NumericalError, nan_mat


__all__ = [
    "fd_xz_derivatives",
]


# It is customary to denote indices of tensors without spaces, e.g.:
#   T_{ijk}  =>  T[i,j,k]
# We disable the respective pylint warning for this file.
# pylint: disable=bad-whitespace


# 1-D finite difference coefficients for first derivatives
COEFFS_1ST = [
    np.array([                  -1., 0.,   1.                 ]) /   2., # order=2
    np.array([            1.,   -8., 0.,   8.,   -1.          ]) /  12., # order=4
    np.array([     -1.,   9.,  -45., 0.,  45.,   -9.,  1.     ]) /  60., # order=6
    np.array([3., -32., 168., -672., 0., 672., -168., 32., -3.]) / 840., # order=8
]


# 1-D finite difference coefficients for second derivatives
COEFFS_2ND = [
    np.array([                      1.,     -2.,    1.                   ]),         # order=2
    np.array([              -1.,   16.,    -30.,   16.,    -1.           ]) /   12., # order=4
    np.array([       2.,   -27.,  270.,   -490.,  270.,   -27.,   2.     ]) /  180., # order=6
    np.array([-9., 128., -1008., 8064., -14350., 8064., -1008., 128., -9.]) / 5040., # order=8
]


class GridDataError(Exception):
    r"""Raised if discrete data is not compatible with this module."""
    pass


def interpolate(mat, coords, linear=False, cache=None, base_idx=None):
    r"""Given a (small) matrix patch, interpolate a value between grid points.

    This takes the whole matrix patch supplied, i.e. the order of the
    interpolating polynomial is determined by the size of the patch. For
    example, if `mat` is a 5x5x5 matrix, 5 point Lagrange interpolation is
    performed.

    @param mat
        Patch to interpolate within. The size of the patch determines the
        order of the interpolating polynomials.
    @param coords
        Coordinates in relative index space at which to interpolate. Should be
        closest to the center grid point for best results. For example, if
        `mat` is a 5x5 matrix with indices ``mat[i,j], i=0,...,4, j=0,...,4``,
        then possible `coords` are ``coords = (1.8, 2.2)``. If the grid points
        correspond to physical coordinates, they should be translated into
        relative index space for this function.
    @param linear
        If `True`, ignore the patch size and do simple linear interpolation.
        Default is `False`.
    @param cache
        Optional dictionary to store and reuse interpolating Lagrange
        polynomials for the first axis of the matrix patch. If given, you also
        need to supply `base_idx`, which maps the patch back to the full
        matrix and allows the cache to be used for the whole data.
    @param base_idx
        Tuple of as many indices as `mat` has axes. This allows the optional
        cache to be applicable to the full data when supplying only small
        patches of it to this function.
    """
    # "collapse" along each axis, one by one.
    try:
        for coord in coords:
            mat = _apply_along_first_axis(_interp1d, mat, coord, linear,
                                          cache, base_idx)
            cache = base_idx = None
    except (ValueError, IndexError) as e:
        raise NumericalError("%s" % e)
    return mat


def _apply_along_first_axis(func, mat, *args, **kwargs):
    r"""Similar to np.apply_along_axis(), but call func with the fixed indices.

    In contrast to `np.apply_along_axis()`, `func` is called with the tuple
    `ii` of indices that are fixed in `mat` as second positional argument.
    This provides context for where the function is being evaluated in the
    matrix which may be used to implement e.g. caching mechanisms.
    """
    shape = mat[1:] if isinstance(mat, tuple) else mat.shape[1:]
    out = np.zeros(shape)
    for ii in np.ndindex(out.shape):
        out[ii] = func(mat[np.index_exp[:] + ii], ii, *args, **kwargs)
    return out


def _interp1d(arr, ii, coord, linear, cache=None, base_idx=None):
    r"""Perform 1-D interpolation of a sequence of values.

    This takes a 1-D sequence of values, `arr`, and interpolates a value in
    relative index space at `coord` as described in interpolate(). The order
    of the interpolating polynomial is determined by the length of the
    sequence. The result is a floating point value.

    @param arr
        1-D array-like with values.
    @param ii
        In case a `cache` is used, the parameter `ii` should consist of
        indices that are *not* varied along the axis currently interpolated
        along. These are added to the last indices of `base_idx` to generate a
        unique key for the data of this axis, which will still be the same
        each time this strip of data is encountered (i.e. even if it is at a
        different position in the matrix patch currently considered).
    @param coord
        Float indicating the coordinate in relative index space at which to
        interpolate.
    @param linear
        If `True`, perform simple linear interpolation instead of Lagrange
        interpolation.
    @param cache
        Dictionary to use as cache. If given, `base_idx` must also be
        supplied.
    @param base_idx
        Tuple of as many indices as `mat` has axes. This allows the optional
        cache to be applicable to the full data.
    """
    if arr.size == 1:
        return arr[0]
    if linear:
        i = int(abs(coord))
        fa = arr[i]
        fb = arr[i+1]
        x = abs(coord)
        return fa + (x-i) * (fb - fa)
    else:
        if cache is not None:
            key = (base_idx[0],) + tuple(map(add, base_idx[1:], ii))
            try:
                poly = cache[key]
            except KeyError:
                # This call is the most expensive one when evaluating
                # numerical data during a MOTS search. It might benefit
                # significantly from a faster (e.g. Cython) implementation.
                poly = lagrange(range(arr.size), arr)
                cache[key] = poly
        else:
            poly = lagrange(range(arr.size), arr)
        return poly(coord)


def fd_xz_derivatives(mat, region, dx, dz, derivs, stencil_size=5):
    r"""Perform finite difference differentiation on specified grid points.

    Given a matrix `mat` containing values on a grid, this function
    approximates the first or second derivative for each grid point in the
    given `region` using 3-, 5-, 7- or 9-point stencils.

    Note that only `x` and `z` (and possibly mixed) derivatives are computed,
    even though `mat` needs to have three axes.

    @return For element of `derivs`, a matrix of the shape of `region`.

    @param mat
        Matrix with the values to use. Must have at least
        ``(stencil_size-1)/2`` additional points in the first and third axes
        on the borders of the given `region`.
    @param region
        Region of grid points at which to compute the derivatives. Should
        consist of three iterables of indices, the tensor product of which
        defines the actual set of indices in the region.
    @param dx,dz
        Physical distance of grid points in coordinate space along the axes.
    @param derivs
        Derivative orders to compute. To compute the x-, z-, and
        x-z-derivatives, use ``derivs=([1, 0], [0, 1], [1, 1])``.
    @param stencil_size
        Number of grid points to consider (i.e. the "size" of the stencil).
        This determines the order of accuracy of the derivative computed.
        Allowed values currently are 3, 5, 7, 9.
    """
    try:
        return _fd_xz_derivatives(mat, region, dx, dz, derivs, stencil_size)
    except (ValueError, IndexError) as e:
        raise NumericalError("%s" % e)


def _fd_xz_derivatives(mat, region, dx, dz, derivs, stencil_size):
    r"""Implement fd_xz_derivatives()."""
    n = int((stencil_size-1)/2)
    if n != (stencil_size-1)/2 or n > len(COEFFS_1ST):
        raise ValueError("Unsupported stencil size: %s" % stencil_size)
    shape = [len(r) for r in region]
    coeffs1 = COEFFS_1ST[n - 1]
    coeffs2 = COEFFS_2ND[n - 1]
    i0, j0, k0 = [r[0] for r in region]
    results = []
    for nx, nz in derivs:
        result = np.zeros(shape)
        for i, j, k in itertools.product(*region):
            if nx == 1 and nz == 0:
                result[i-i0, j-j0, k-k0] = 1/dx * (
                    mat[i-n:i+n+1, j, k].dot(coeffs1)
                )
            elif nx == 0 and nz == 1:
                result[i-i0, j-j0, k-k0] = 1/dz * (
                    mat[i, j, k-n:k+n+1].dot(coeffs1)
                )
            elif nx == 1 and nz == 1:
                result[i-i0, j-j0, k-k0] = 1/(dx*dz) * (
                    mat[i-n:i+n+1, j, k-n:k+n+1].dot(coeffs1).dot(coeffs1)
                )
            elif nx == 2 and nz == 0:
                result[i-i0, j-j0, k-k0] = 1/dx**2 * (
                    mat[i-n:i+n+1, j, k].dot(coeffs2)
                )
            elif nx == 0 and nz == 2:
                result[i-i0, j-j0, k-k0] = 1/dz**2 * (
                    mat[i, j, k-n:k+n+1].dot(coeffs2)
                )
            elif nx == 2 and nz == 1:
                result[i-i0, j-j0, k-k0] = 1/(dx*dx*dz) * (
                    mat[i-n:i+n+1, j, k-n:k+n+1].dot(coeffs1).dot(coeffs2)
                )
            elif nx == 1 and nz == 2:
                result[i-i0, j-j0, k-k0] = 1/(dx*dz*dz) * (
                    mat[i-n:i+n+1, j, k-n:k+n+1].dot(coeffs2).dot(coeffs1)
                )
            elif nx == 2 and nz == 2:
                result[i-i0, j-j0, k-k0] = 1/(dx*dz)**2 * (
                    mat[i-n:i+n+1, j, k-n:k+n+1].dot(coeffs2).dot(coeffs2)
                )
            else:
                raise NotImplementedError(
                    "Derivative order not implemented: %s, %s" % (nx, nz)
                )
        results.append(result)
    return results


def eval_sym_axisym_matrix(comp_funcs, *lower_orders, point, diff=0):
    r"""Evaluate (derivatives of) a symmetric tensor field at a point.

    This takes the six independent component functions of the `xx`, `xy`,
    `xz`, `yy`, `yz`, and `zz` components (in that order) of a tensor field
    `T`. These should be DataPatch objects. It then computes the
    derivatives of the requested order `diff` in all three coordinate
    directions using the axisymmetry of the tensor to infer the y-derivatives
    from the x-derivative.

    In order to compute derivatives, all lower order derivatives are required
    (including order 0). These have to be supplied as positional arguments
    after the list of component functions.

    @return For ``diff=0``, returns the 3x3 matrix representing `T`
        interpolated at `point`, i.e. \f$T_{ij}\f$.
        If ``diff=1``, returns ``dT[i,j,k]``, where the indices mean
        \f$\partial_i T_{jk}\f$ and if ``diff=2``, returns
        ``ddT[i,j,k,l]`` with indices \f$\partial_i\partial_j T_{kl}\f$.

    @param comp_funcs
        An iterable of the six independent component functions (DataPatch)
        of the tensor field.
    @param *lower_orders
        Further positional arguments supplying the lower order derivatives.
        For `diff=0`, none should be supplied. For `diff=1`, one argument, `T`
        itself, should be given. For `diff=2`, `T` and `dT` should be given in
        that order.
    @param point
        The point at which to compute.
    @param diff
        Derivative order to compute. Default is `0`.


    @b Notes

    Based on the considerations in [1], the y-derivatives of tensor field
    components in the xz-plane can be computed by differentiating eq. (7) in
    [1] w.r.t. `y` and evaluating at `y=0`. The results are
    \f{eqnarray*}{
        (\partial_y T_{ij}) &=& \frac1x
            \left(\begin{array}{@{}ccc@{}}
                -2T_{xy}      & T_{xx}-T_{yy} & -T_{yz} \\
                T_{xx}-T_{yy} & 2T_{xy}       & T_{xz} \\
                -T_{yz}       & T_{xz}        & 0
            \end{array}\right)
        \\
        (\partial_x\partial_y T_{ij}) &=& \frac1x
            \left(\begin{array}{@{}ccc@{}}
                -2T_{xy,x} + \frac{2 T_{xy}}{x}                       & T_{xx,x} - \frac{T_{xx}}{x} - T_{yy,x} + \frac{T_{yy}}{x} & - T_{yz,x} + \frac{T_{yz}}{x} \\
                T_{xx,x}-T_{yy,x}-\frac{T_{xx}}{x} + \frac{T_{yy}}{x} & 2 T_{xy,x} - \frac{2 T_{xy}}{x}                           & T_{xz,x} - \frac{T_{xz}}{x} \\
                -T_{yz,x}+\frac{T_{yz}}{x}                            & T_{xz,x} - \frac{T_{xz}}{x}                               & 0
            \end{array}\right)
        \\
        (\partial_y\partial_y T_{ij}) &=& \frac1x
            \left(\begin{array}{@{}ccc@{}}
                T_{xx,x}-\frac{2 T_{xx}}{x} + \frac{2 T_{yy}}{x} & T_{xy,x} - \frac{4 T_{xy}}{x}                      & T_{xz,x} - \frac{T_{xz}}{x}\\
                T_{xy,x}-\frac{4 T_{xy}}{x}                      & \frac{2 T_{xx}}{x} + T_{yy,x} - \frac{2 T_{yy}}{x} & T_{yz,x} - \frac{T_{yz}}{x}\\
                T_{xz,x}-\frac{T_{xz}}{x}                        & T_{yz,x} - \frac{T_{yz}}{x}                        & T_{zz,x}
            \end{array}\right)
        \\
        (\partial_y\partial_z T_{ij}) &=& \frac1x
            \left(\begin{array}{@{}ccc@{}}
                -2 T_{xy,z}       & T_{xx,z}-T_{yy,z} & -T_{yz,z}\\
                T_{xx,z}-T_{yy,z} & 2 T_{xy,z}        & T_{xz,z}\\
                -T_{yz,z}         & T_{xz,z}          & 0
            \end{array}\right).
    \f}
    Note that since we don't transform the derivative but use eq. (7) as
    defining `T` for ``y != 0``, the rotation matrices `R` have to be taken as
    dependent on `y` and not as rigid rotations.


    @b References

    [1] Alcubierre, Miguel, et al. "Symmetry without symmetry: Numerical
        simulation of axisymmetric systems using Cartesian grids."
        International Journal of Modern Physics D 10.03 (2001): 273-289.
    """
    if diff == 0:
        T00, T01, T02, T11, T12, T22 = [
            Tij.interpolate(point) for Tij in comp_funcs
        ]
        return np.array([[T00, T01, T02],
                         [T01, T11, T12],
                         [T02, T12, T22]])
    x = point[0]
    if diff == 1:
        T, = lower_orders
        (
            (T00x, T00z), (T01x, T01z), (T02x, T02z),
            (T11x, T11z), (T12x, T12z), (T22x, T22z)
        ) = [
            Tij.diff(point, diff=1)
            for Tij in comp_funcs
        ]
        Tx = np.array([[T00x, T01x, T02x],
                       [T01x, T11x, T12x],
                       [T02x, T12x, T22x]])
        Tz = np.array([[T00z, T01z, T02z],
                       [T01z, T11z, T12z],
                       [T02z, T12z, T22z]])
        if x == 0:
            Ty = nan_mat((3, 3))
        else:
            Ty = np.array([[-2*T[0,1]/x,       (T[0,0]-T[1,1])/x, -T[1,2]/x],
                           [(T[0,0]-T[1,1])/x, 2*T[0,1]/x,        T[0,2]/x],
                           [-T[1,2]/x,         T[0,2]/x,          0.]])
        return np.asarray([Tx, Ty, Tz])
    if diff == 2:
        T, dT = lower_orders
        (
            (T00xx, T00zz, T00xz),
            (T01xx, T01zz, T01xz),
            (T02xx, T02zz, T02xz),
            (T11xx, T11zz, T11xz),
            (T12xx, T12zz, T12xz),
            (T22xx, T22zz, T22xz),
        ) = [
            Tij.diff(point, diff=2)
            for Tij in comp_funcs
        ]
        Txx = np.array([[T00xx, T01xx, T02xx],
                        [T01xx, T11xx, T12xx],
                        [T02xx, T12xx, T22xx]])
        Tzz = np.array([[T00zz, T01zz, T02zz],
                        [T01zz, T11zz, T12zz],
                        [T02zz, T12zz, T22zz]])
        Txz = np.array([[T00xz, T01xz, T02xz],
                        [T01xz, T11xz, T12xz],
                        [T02xz, T12xz, T22xz]])
        if x == 0:
            Txy = nan_mat((3, 3))
            Tyy = nan_mat((3, 3))
            Tyz = nan_mat((3, 3))
        else:
            Txy = 1/x * np.array([
                [2 * (T[0,1]/x - dT[0,0,1]),
                 dT[0,0,0] - dT[0,1,1] + (T[1,1]-T[0,0])/x,
                 T[1,2]/x - dT[0,1,2]],
                [0, 2 * (dT[0,0,1] - T[0,1]/x), dT[0,0,2] - T[0,2]/x],
                [0, 0, 0]
            ])
            _sym3x3(Txy)
            Tyy = 1/x * np.array([
                [dT[0,0,0] - 2 * (T[0,0]-T[1,1])/x,
                 dT[0,0,1] - 4 * T[0,1]/x,
                 dT[0,0,2] - T[0,2]/x],
                [0, 2 * (T[0,0] - T[1,1])/x + dT[0,1,1], dT[0,1,2]-T[1,2]/x],
                [0, 0, dT[0,2,2]]
            ])
            _sym3x3(Tyy)
            Tyz = 1/x * np.array([
                [-2*dT[2,0,1], dT[2,0,0]-dT[2,1,1], -dT[2,1,2]],
                [0,            2*dT[2,0,1],         dT[2,0,2]],
                [0,            0,                   0]
            ])
            _sym3x3(Tyz)
        return np.asarray([[Txx, Txy, Txz],
                           [Txy, Tyy, Tyz],
                           [Txz, Tyz, Tzz]])
    raise ValueError("Unknown `diff` value: %s" % diff)


def _sym3x3(T):
    r"""Symmetrize a 3x3 matrix by replacing the lower-left three components."""
    T[1,0], T[2,0], T[2,1] = T[0,1], T[0,2], T[1,2]
