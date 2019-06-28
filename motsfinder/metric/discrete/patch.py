r"""@package motsfinder.metric.discrete.patch

Data patches containing axisymmetric data on grids.

The idea is to represent the numerical data via a matrix of values plus some
metadata containing information of how to map these values to physical
coordinates. This may then represent either a scalar field on some part of the
xz-plane, or a (part of a) single component of a vector or higher tensor
field. To construct complete fields such as the 3-metric or extrinsic
curvature, these patches may be used as individual components.

The BBox class represents a simple *index* bounding box (i.e. without mapping
to physical coordinates) used to map the 0-based matrix indices to indices of
a fictitious large grid covering the full (finite) domain.

The GridPatch then adds information needed to map matrix elements to physical
coordinates (and vice versa).

Finally, the DataPatch extends GridPatch with actual data. The main feature of
a GridPatch is the ability to evaluate the data at arbitrary points within the
domain using Lagrange interpolation, as well as evaluating and interpolating
the first and second derivatives of the discrete data. The patch objects also
handle an optional (active, by default) cache storing all Lagrange polynomials
of 5-point 1-D "strips" on the grid that have bean generated thus far.
Practical tests show that this greatly reduces computational cost in cases
where evaluations between grid points are close to each other. This happens in
practice e.g. when the grid is evaluated along trial surfaces in a non-linear
Newton-like search close to convergence of these surfaces.
"""

import itertools

import numpy as np
from scipy import linalg

from ...utils import lrange
from ...numutils import NumericalError
from .numerical import GridDataError, interpolate, fd_xz_derivatives


__all__ = []


# It is customary to denote indices of tensors without spaces, e.g.:
#   T_{ijk}  =>  T[i,j,k]
# We disable the respective pylint warning for this file.
# pylint: disable=bad-whitespace


class BBox():
    r"""Bounding box for index ranges of matrix patches.

    These can be used to "patch together" the individual blocks of data to
    construct one large matrix containing the data for the complete domain
    (i.e. there is no information here to map the grid to physical
    coordinates).
    """

    __slots__ = ("lower", "upper")

    def __init__(self, lower, upper):
        r"""Construct a bounding box using the lower and upper index limits."""
        ## NumPy array of lower index bounds.
        self.lower = np.asarray(lower)
        ## NumPy array of upper index bounds.
        self.upper = np.asarray(upper)

    @property
    def shape(self):
        r"""Shape of a data matrix covering the patch."""
        return self.upper - self.lower

    def __repr__(self):
        r"""Print a representation of the bounding box."""
        l = self.lower
        u = self.upper
        return "([%s]:[%s])" % (",".join("%d" % v for v in l),
                                ",".join("%d" % v for v in u))


class GridPatch():
    r"""This class represents an individual patch (block) of a field.

    This base class does not contain any actual field or field component data,
    it just represents the meta-like information of the patch, e.g. the
    bounding box, spatial grid resolution (i.e. the deltas) and physical
    coordinate of the origin.
    """

    def __init__(self, origin, deltas, box):
        r"""Construct a new patch object.

        Note that since all classes in this module assume the data to
        represent axisymmetric fields/tensors, we explicitly check that the
        xz-plane is contained in this patch. This allows the data to contain
        additional 3-D data around `y=0`, as will be the case for data
        containing ghost points at the domain boundaries. The public attribute
        `yidx` will hold the index in y-direction at which the xz-plane's data
        is found. If no ghost points are included, `yidx` should be zero.

        @param origin
            Origin of this data patch as obtained from the field component
            block.
        @param deltas
            3 by 3 matrix containing the deltas for the x, y, z physical grid
            translation vectors.
        @param box
            Bounding box of this patch. Should be a BBox object.
        """
        ## NumPy array of patch origin coordinates.
        self.origin = np.asarray(origin)
        ## 3x3 array with rows (fixed first index) indicating the three delta vectors.
        self.deltas = np.asarray(deltas)
        ## Step lengths of the three delta vectors.
        self.delta_norms = [linalg.norm(dx) for dx in self.deltas]
        ## Inverse delta vectors (used to find grid point from coordinates).
        self.inv_deltas = linalg.inv(deltas)
        ## Bounding box of this patch.
        self.box = box
        ## Physical domain of this patch.
        self.domain = self._get_domain()
        ## Index in y-direction at which the xz-plane is found in the data.
        self.yidx = -int(round(self.origin[1] / self.dy[1]))
        if not 0 <= self.yidx < self.shape[1]:
            raise GridDataError("Patch does not hit xz-plane")
        ## Index in x-direction at which ``x==0`` (not necessarily in patch domain).
        self.xidx = -int(round(self.origin[0] / self.dx[0]))

    @property
    def dx(self):
        r"""Translation (in coordinate space) between grid points on first axis."""
        return self.deltas[0]

    @property
    def dy(self):
        r"""Translation (in coordinate space) between grid points on second axis."""
        return self.deltas[1]

    @property
    def dz(self):
        r"""Translation (in coordinate space) between grid points on third axis."""
        return self.deltas[2]

    @property
    def dx_norm(self):
        r"""Length (Euclidean norm) of `dx`."""
        return self.delta_norms[0]

    @property
    def dy_norm(self):
        r"""Length (Euclidean norm) of `dy`."""
        return self.delta_norms[1]

    @property
    def dz_norm(self):
        r"""Length (Euclidean norm) of `dz`."""
        return self.delta_norms[2]

    @property
    def shape(self):
        r"""Shape of a matrix that can store the data of this patch."""
        return self.box.shape

    def coords(self, i, j, k=None):
        r"""Compute the physical coordinates of a grid point.

        The result is a point in coordinate space. This works even if the grid
        point lies outside the patch.

        The parameters `i`, `j`, `k` denote the indices of the patch matrix
        element belonging to the returned point. If `k` is not given, the two
        given indices are interpreted as lying in the xz-plane and the y-index
        is chosen to be at `y=0`.
        """
        if k is None:
            j, k = self.yidx, j
        if i < 0:
            i += self.shape[0]
        if j < 0:
            j += self.shape[1]
        if k < 0:
            k += self.shape[2]
        return self.origin + self.deltas.dot([i, j, k])

    def grid(self, xz_plane=True, ghost=0, full_output=False):
        r"""Generator to iterate over all grid points in the patch.

        It is more efficient to avoid iterating over the whole grid in Python.
        Instead, many operations can be performed by utilizing numpy's ability
        to operate on whole matrices. This method should only be used if
        efficiency is not of highest concern.

        @param xz_plane
            Whether to only output grid points on the xz-plane (i.e. for
            `y=0`). If the data contain ghost points in y-direction, this will
            pick out only those in the xz-plane. Default is `True`.
        @param ghost
            Number of ghost points to skip at the boundaries. Default is `0`.
        @param full_output
            If `True`, the yielded elements will consist of the matrix indices
            `(i,j,k)` and the physical coordinates `(x,y,z)`. Otherwise, only
            the matrix indices are yielded. Default is `False`.
        """
        nx, ny, nz = self.shape
        if xz_plane:
            yra = [self.yidx]
        else:
            yra = range(ghost, ny-ghost)
        grid_indices = itertools.product(range(ghost, nx-ghost),
                                         yra,
                                         range(ghost, nz-ghost))
        if full_output:
            for i, j, k in grid_indices:
                yield (i, j, k), self.coords(i, j, k)
        else:
            for ijk in grid_indices:
                yield ijk

    def closest_element(self, point, floating=False):
        r"""Compute the grid point indices closest to given physical coordinates.

        This takes a `point` in physical coordinates and computes the indices
        of the matrix element that lies closest. This is done in constant time
        (i.e. no "search" is performed).

        @param point
            3-D point in physical coordinates.
        @param floating
            If `True`, the returned indices will be of floating point type,
            the fractional portion denoting the precise location of the
            `point` in "grid coordinate space". This may be useful for
            interpolating the matrix values between the grid points.
        """
        indices = self.inv_deltas.dot(np.asarray(point) - self.origin)
        if not floating:
            return tuple(int(round(i)) for i in indices)
        return indices

    def cell_corner(self, point):
        r"""Return the grid point indices of the corner of the cell a point lies in.

        A "cell" is a 2x2 square patch of four points.
        """
        indices = self.inv_deltas.dot(np.asarray(point) - self.origin)
        return tuple(map(int, indices))

    def snap_to_grid(self, point):
        r"""Return the coordinates of the closest grid point to a given point."""
        return self.coords(*self.closest_element(point))

    def _get_domain(self):
        r"""Find the physical domain of this patch/block."""
        ra = [0, -1]
        points = np.array([self.coords(i, j, k)
                           for i in ra for j in ra for k in ra])
        return [(points[:,i].min(), points[:,i].max()) for i in range(3)]


class DataPatch(GridPatch):
    r"""Represents a patch of field (component) data in axisymmetry.

    The field data can be evaluated at the grid points using at() or
    interpolated between these using interpolate() or diff().

    To get the correct behavior at the z-axis (i.e. for `x=0`), we need to
    know the symmetry of the represented data under rotations by pi. The
    reason is that in order to interpolate derivatives close to the z-axis, we
    need the values at grid points around the interesting point (how many
    depends on the desired order of accuracy).
    """

    def __init__(self, origin, deltas, box, mat, symmetry,
                 interpolation='hermite5', fd_order=6, caching=True):
        r"""Create a new data patch object.

        @param origin
            Origin of this data patch as obtained from the field component
            block.
        @param deltas
            3 by 3 matrix containing the deltas for the x, y, z physical grid
            translation vectors.
        @param box
            Bounding box of this patch. Should be a BBox object.
        @param mat
            The matrix containing the actual field (or field component) data.
            This needs to have the same shape as given by `box`.
        @param symmetry
            Either ``'even'`` or ``'odd'``. For ``'even'``, the component or
            scalar field data is assumed to satisfy \f$f(x,0,z) = f(-x,0,z)\f$
            while for ``'odd'`` we assume \f$f(x,0,z) = -f(-x,0,z)\f$.
        @param interpolation
            Kind of interpolation to do. Default is ``'hermite5'``. Possible
            values are those documented for set_interpolation().
        @param fd_order
            Order of the finite difference differentiation. Default is `6`,
            i.e. using a 7-point stencil.
        @param caching
            Whether to cache interpolating Lagrange polynomials.
        """
        super(DataPatch, self).__init__(origin=origin, deltas=deltas, box=box)
        if symmetry not in ('even', 'odd'):
            raise ValueError("Unknown symmetry: %s" % symmetry)
        ## NumPy data matrix with the data of this patch.
        self.mat = mat
        ## Symmetry setting for auto-extending to negative x values.
        self.symmetry = symmetry
        self._lagrange_cache = [] if caching else None
        self._hermite_cell_matrices = dict()
        self._hermite_interpolant = dict()
        self._interpolation = None
        self._fd_order = fd_order
        self.set_interpolation(interpolation)

    @classmethod
    def from_patch(cls, patch, mat, symmetry, **kw):
        r"""Construct a DataPatch from a GridPatch and data matrix.

        @param patch
            The DataPatch object for which we have the data.
        @param mat,symmetry,**kw
            See the respective parameters of #__init__().
        """
        return cls(origin=patch.origin, deltas=patch.deltas, box=patch.box,
                   mat=mat, symmetry=symmetry, **kw)

    def __getstate__(self):
        r"""Return a picklable state object."""
        state = self.__dict__.copy()
        state['_lagrange_cache'] = []
        state['_hermite_cell_matrices'] = dict()
        state['_hermite_interpolant'] = dict()
        return state

    def __setstate__(self, state):
        r"""Restore this object from the given unpickled state."""
        # compatibility with data from previous versions
        self._lagrange_cache = []
        self._hermite_cell_matrices = dict()
        self._hermite_interpolant = dict()
        self._hermite_order = 5
        self._interpolation = 'lagrange' # previously the only option
        self._fd_order = 4 # previous default
        # Restore state. This overrides the above if contained in the data.
        self.__dict__.update(state)

    def set_caching(self, caching=True):
        r"""Specify whether to cache the interpolating polynomials."""
        if caching and not self.caching:
            self._lagrange_cache = []
        if not caching:
            self._lagrange_cache = None

    @property
    def caching(self):
        r"""Whether caching of interpolating Lagrange polynomials is enabled."""
        return isinstance(self._lagrange_cache, list)

    @property
    def fd_order(self):
        r"""Set the order of finite difference differentiation."""
        return self._fd_order
    @fd_order.setter
    def fd_order(self, order):
        r"""Current order of finite difference differentiation."""
        self._fd_order = order

    def set_interpolation(self, interpolation):
        r"""Change the kind of interpolation to do between grid points.

        Supported interpolations are:
            * ``'none'``: no interpolation (use closest grid point)
            * ``'linear'``: linear interpolation between grid points
            * ``'lagrange'``: 5-point Lagrange interpolation
            * ``'hermite3'``: cubic Hermite interpolation (using first
              derivatives)
            * ``'hermite5'``: quintic Hermite interpolation (using also second
              derivatives)
        """
        if interpolation is None:
            interpolation = 'none'
        interpolation = interpolation.lower()
        if interpolation == 'none':
            self._interpolation = None
        elif interpolation == 'hermite3':
            self._interpolation = 'hermite'
            self._hermite_order = 3
        elif interpolation == 'hermite5':
            self._interpolation = 'hermite'
            self._hermite_order = 5
        elif interpolation in ('lagrange', 'linear'):
            self._interpolation = interpolation
        else:
            raise ValueError("Unknown interpolation: %s" % interpolation)

    def get_interpolation(self):
        r"""Return the currently set interpolation."""
        interp = self._interpolation
        if interp == "hermite":
            interp = "hermite%s" % self._hermite_order
        return interp

    def _hermite_cell_matrix(self, order=3):
        r"""Compute the matrix used to solve for coefficients for Hermite interolation.

        @param order
            Order of the Hermite interpolating polynomial. Default is `3`,
            i.e. cubic Hermite interpolation, which uses the values and first
            partial derivatives at the grid points. The returned matrix will
            be a 16x16 matrix. The other implemented option is `5`, which also
            uses the second derivatives and hence produces a 36x36 matrix.
        """
        try:
            return self._hermite_cell_matrices[order]
        except KeyError:
            pass
        M = self._compute_hermite_cell_matrix(order=order)
        self._hermite_cell_matrices[order] = M
        return M

    def _compute_hermite_cell_matrix(self, order):
        r"""Compute the matrix for _hermite_cell_matrix()."""
        xs = [0, 1]
        zs = [0, 1]
        orders = list(range(order+1))
        M = [[xi**ii * zi**jj for ii in orders for jj in orders]
             for xi in xs for zi in zs]
        # \partial_x
        M.extend([[ii * xi**(ii-1) * zi**jj if ii else 0.
                   for ii in orders for jj in orders]
                  for xi in xs for zi in zs])
        # \partial_z
        M.extend([[jj * xi**ii * zi**(jj-1) if jj else 0.
                   for ii in orders for jj in orders]
                  for xi in xs for zi in zs])
        # \partial_x \partial_z
        M.extend([[ii * jj * xi**(ii-1) * zi**(jj-1) if ii and jj else 0.
                   for ii in orders for jj in orders]
                  for xi in xs for zi in zs])
        if order == 3:
            return np.asarray(M)
        # \partial_x^2
        M.extend([[ii * (ii-1) * xi**(ii-2) * zi**jj if ii > 1 else 0.
                   for ii in orders for jj in orders]
                  for xi in xs for zi in zs])
        # \partial_z^2
        M.extend([[jj * (jj-1) * xi**ii * zi**(jj-2) if jj > 1 else 0.
                   for ii in orders for jj in orders]
                  for xi in xs for zi in zs])
        # \partial_x^2 \partial_z
        M.extend([[ii * (ii-1) * jj * xi**(ii-2) * zi**(jj-1) if ii > 1 and jj > 0 else 0.
                   for ii in orders for jj in orders]
                  for xi in xs for zi in zs])
        # \partial_x \partial_z^2
        M.extend([[ii * jj * (jj-1) * xi**(ii-1) * zi**(jj-2) if ii > 0 and jj > 1 else 0.
                   for ii in orders for jj in orders]
                  for xi in xs for zi in zs])
        # \partial_x^2 \partial_z^2
        M.extend([[ii * (ii-1) * jj * (jj-1) * xi**(ii-2) * zi**(jj-2) if ii > 1 and jj > 1 else 0.
                   for ii in orders for jj in orders]
                  for xi in xs for zi in zs])
        if order == 5:
            return np.asarray(M)
        raise NotImplementedError("Interpolation order not implemented: %s" %
                                  order)

    def _hermite_cell(self, i, j, stencil_size=5, order=3):
        r"""Construct a Hermite interpolant for one grid cell.

        This creates a 2-D interpolant for the grid data cell with corner
        points ``i, i+1, j, j+1``. The returned callable has two parameters:
        the point at which to interpolate (should be the physical coordinates
        lying inside the cell) and optionally the derivative order `nu`, given
        by ``nu=(nu_x, nu_z)``, where `nu_x` is the derivative order in
        x-direction and `nu_z` in z-direciton.

        @param i,j
            Lower left corner of the cell to build the interpolant for.
        @param stencil_size
            Size of the derivative stencil for approximating the derivatives
            at the grid points. Default is `5`, leading to 4th order accurate
            derivatives. Implemented options are `3, 5, 7, 9`.
        @param order
            Order of the interpolating polynomial. Currently supported values
            are `3, 5`, with `3` being the default.
        """
        n = int((stencil_size-1)/2)
        mat = self.get_region(
            (i-n, i+n+2), (self.yidx, self.yidx+1),
            (j-n, j+n+2),
        )
        if order == 3:
            nu = [(1, 0), (0, 1), (1, 1)]
        elif order == 5:
            nu = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]
        else:
            raise ValueError("Unsupported order: %s" % order)
        derivs = fd_xz_derivatives(
            mat,
            region=([n, n+1], [0], [n, n+1]),
            dx=1, dz=1, stencil_size=stencil_size, derivs=nu,
        )
        y = [mat[n+nn, 0, n+mm] for nn in (0, 1) for mm in (0, 1)]
        for (nx, nz), der in zip(nu, derivs):
            y.extend(der.flatten().tolist())
        y = np.asarray(y)
        M = self._hermite_cell_matrix(order=order)
        c = linalg.solve(M, y)
        c = c.reshape((order+1, order+1))
        x0 = self.coords(abs(i), j)
        orders = list(range(order+1))
        if i < 0:
            x0[0] *= -1
        dx = self.dx_norm
        dz = self.dz_norm
        def interp(point, nu=None):
            x, _, z = np.asarray(point) - x0
            x /= dx
            z /= dz
            if nu is None or max(nu) == 0:
                return np.sum(c[i,j]*x**i*z**j for i in orders for j in orders)
            nx, nz = nu
            return np.sum(
                np.prod(range(i-nx+1, i+1)) / dx**nx
                * np.prod(range(j-nz+1, j+1)) / dz**nz
                * c[i,j]*x**(i-nx)*z**(j-nz)
                for i in range(nx, order+1) for j in range(nz, order+1)
            )
        return interp

    def hermite_2d(self, stencil_size=5, order=3):
        r"""Return a Hermite interpolant for the full data patch.

        Cells are set up on-demand and cached by default, so that this method
        is very light-weight. Whether caching is done or not can be controlled
        using set_caching().

        Also, the interpolants themselves are cached, so calling this method
        repeatedly is (almost) as efficient as keeping the interpolant for
        multiple evaluations.

        @param stencil_size
            Size of the derivative stencil for approximating the derivatives
            at the grid points. Default is `5`, leading to 4th order accurate
            derivatives. Implemented options are `3, 5, 7, 9`.
        @param order
            Order of the interpolating polynomial. Currently supported values
            are `3, 5`, with `3` being the default.
        """
        try:
            return self._hermite_interpolant[(stencil_size, order)]
        except KeyError:
            interp = self._hermite_2d(stencil_size, order)
            self._hermite_interpolant[(stencil_size, order)] = interp
            return interp

    def _hermite_2d(self, stencil_size, order):
        r"""Implement creating an interpolant for hermite_2d()."""
        if self.caching:
            cache = dict()
            def interp(point, nu=None):
                i, _, j = self.cell_corner(point)
                key = (i, j)
                try:
                    cell_interp = cache[key]
                except KeyError:
                    cell_interp = self._hermite_cell(
                        i, j, stencil_size=stencil_size, order=order
                    )
                    cache[key] = cell_interp
                return cell_interp(point, nu)
        else:
            def interp(point, nu=None):
                i, _, j = self.cell_corner(point)
                cell_interp = self._hermite_cell(
                    i, j, stencil_size=stencil_size, order=order
                )
                return cell_interp(point, nu)
        return interp

    def get_region(self, xra, yra, zra):
        r"""Create a matrix containing the data of a sub-patch.

        In most cases, the returned matrix will be a light-weight "view" of a
        patch of the full data matrix (so don't modify the data). The
        exception is if the x-range lies (fully or partly) outside the patch's
        domain across the z-axis. In this case, the symmetry of the data is
        respected to create the missing elements by rotating the existing data
        by pi.

        @param xra,yra,zra
            Range of x-, y-, z-indices (respectively) given as a 2-tuple/list
            ``xra = (xmin, xmax)``. Note that the index ``xmin`` will be
            included while ``xmax`` will not (like for e.g. `range()`).
        """
        try:
            return self._get_region(xra, yra, zra)
        except (ValueError, IndexError) as e:
            raise NumericalError("%s" % e)

    def _get_region(self, xra, yra, zra):
        r"""Implementation of get_region()."""
        change_sign_idx = 0
        if xra[0] < 0:
            xidx = self.xidx
            if xra[1] < 0:
                xslice = lrange(2*xidx-xra[0], 2*xidx-xra[1], -1)
            else:
                xslice = lrange(2*xidx-xra[0], 2*xidx, -1)
            if self.symmetry == 'odd':
                change_sign_idx = len(xslice)
            xslice += lrange(xra[1])
        else:
            xslice = slice(*xra)
        mat = self.mat[xslice,slice(*yra),slice(*zra)]
        if change_sign_idx:
            # When we're here, `mat` contains a copy of the data and not a
            # view (as `xslice` is a list), so we're safe to modify `mat`.
            mat[:change_sign_idx] *= -1
        return mat

    def interpolate(self, point):
        r"""Interpolate the data at a point within the patch.

        The kind of interpolation can be set using set_interpolation() and
        using the `fd_order` property.

        @param point
            (Physical) coordinates of the point at which to interpolate.
        """
        if self._interpolation == 'hermite':
            return self.hermite_2d(self.fd_order+1, self._hermite_order)(point)
        if self._interpolation is None:
            return self.at(*self.closest_element(point))
        linear = self._interpolation == 'linear'
        npts = self.fd_order + 1
        if linear or self._interpolation == 'lagrange':
            real_ijk = self.closest_element(point, floating=True)
            i, j, k = [int(round(i)) for i in real_ijk]
            if linear:
                npts = 3
            if npts % 2 == 0:
                raise ValueError("Need an odd number of points (`npts=%s`)" % npts)
            n = int((npts-1)/2)
            if abs(point[1]) < 1e-12:
                # we lie in the xz-plane
                yra = (self.yidx, self.yidx+1)
                real_ijk += [n-i, 0., n-k]
            else:
                yra = (j-n, j+n+1)
                real_ijk += [n-i, n-j, n-k]
            mat = self.get_region((i-n, i+n+1), yra, (k-n, k+n+1))
            return interpolate(mat, real_ijk, linear=linear,
                               cache=self._get_cache(diff=0), base_idx=(i,j,k))
        raise NotImplementedError("Interpolation method not implemented: %s" %
                                  self._interpolation)

    def _get_cache(self, diff=0, sub_idx=0):
        r"""Return a dict to use as cache for a given derivative order.

        The cache is implemented such that each partial derivative has its own
        dict. For example, there is a separate cache for:
            * the component values themselves
            * the x-derivative
            * the z-derivative
            * ...
            * the xz-derivative
            * ...

        To implement this, set `diff` to the derivative order and `sub_idx` to
        an index unique for the particular partial derivative (e.g. 0 for
        xx-derivative, 1 for the xy-derivative, etc.).
        """
        try:
            cache = self._lagrange_cache
            while len(cache) <= diff:
                cache.append([dict()])
            cache_slot = cache[diff]
            while len(cache_slot) <= sub_idx:
                cache_slot.append(dict())
            return cache_slot[sub_idx]
        except TypeError:
            return None

    def at(self, i, j, k, diff=0):
        r"""Get the value (or derivatives) at a grid point without interpolation.

        The parameters `i`, `j`, `k` are the indices of the patch matrix at
        which to compute. Given a point in physical `x, y, z` coordinates, you
        can get these indices using GridPatch.closest_element().

        Use `diff` to compute derivatives at the grid points. See diff() for
        more information.
        """
        if diff == 0:
            return self.mat[i, j, k]
        if j != self.yidx:
            raise ValueError("Evaluation must be in xz-plane.")
        dx = self.dx_norm
        dz = self.dz_norm
        n = int((self.fd_order)/2)
        mat = self.get_region((i-n, i+n+1), (self.yidx, self.yidx+1),
                              (k-n, k+n+1))
        if diff == 1:
            derivs = [(1, 0), (0, 1)]
        elif diff == 2:
            derivs = [(2, 0), (0, 2), (1, 1)]
        else:
            raise ValueError("Unsupported derivative order: %s" % (diff,))
        partial_derivs = fd_xz_derivatives(
            mat,
            region=([n], [0], [n]),
            dx=dx, dz=dz,
            stencil_size=self.fd_order+1,
            derivs=derivs,
        )
        return [v[0,0,0] for v in partial_derivs]

    def diff(self, point, diff=1):
        r"""Compute derivatives and interpolate at a given point.

        @return For ``diff=1`` a list with two elements ``[f_x, f_z]``, where
            `f_x` is the x-derivative and `f_z` the z-derivative. For
            ``diff=2``, returns a list with three elements
            ``[f_xx, f_zz, f_xz]``, where `f_xx` is the data twice
            differentiated w.r.t. x, etc. For ``diff=0``, returns the result
            of interpolate().

        @param point
            (Physical) coordinates of the point at which to interpolate.
        @param diff
            Derivative order. Can be `0`, `1` or `2`. Default is `1`.
        """
        if diff == 0:
            return self.interpolate(point)
        if self._interpolation == 'hermite':
            interp = self.hermite_2d(self.fd_order+1, self._hermite_order)
            if diff == 1:
                return [interp(point, nu=(1, 0)), interp(point, nu=(0, 1))]
            if diff == 2:
                return [interp(point, nu=(2, 0)),
                        interp(point, nu=(0, 2)),
                        interp(point, nu=(1, 1))]
            raise NotImplementedError("Unsupported derivative order: %s" %
                                      diff)
        linear = self._interpolation == 'linear'
        interp = self._interpolation is not None
        if not interp or linear or self._interpolation == 'lagrange':
            real_ijk = self.closest_element(point, floating=True)
            i, j, k = [int(round(i)) for i in real_ijk]
            if not interp:
                return self.at(i, j, k, diff=diff)
            n = int((self.fd_order)/2)
            real_ijk += [n-i, 0., n-k]
            dx = self.dx_norm
            dz = self.dz_norm
            if diff == 1:
                derivs = [(1, 0), (0, 1)]
            elif diff == 2:
                derivs = [(2, 0), (0, 2), (1, 1)]
            else:
                raise ValueError("Unsupported derivative order: %s" % (diff,))
            partial_derivs = fd_xz_derivatives(
                mat=self.get_region((i-2*n, i+2*n+1),
                                    (self.yidx, self.yidx+1),
                                    (k-2*n, k+2*n+1)),
                region=(range(n, 2*(n+1)+1), [self.yidx], range(n, 2*(n+1)+1)),
                dx=dx, dz=dz,
                stencil_size=self.fd_order+1,
                derivs=derivs,
            )
            return [
                interpolate(p, real_ijk, linear=linear,
                            cache=self._get_cache(diff, slot), base_idx=(i,j,k))
                for slot, p in enumerate(partial_derivs)
            ]
        raise NotImplementedError("Interpolation method not implemented: %s" %
                                  self._interpolation)
