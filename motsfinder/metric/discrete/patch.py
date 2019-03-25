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
from .numerical import GridDataError, interpolate, _diff_4th_order_xz


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
        if not full_output:
            return grid_indices
        for i, j, k in grid_indices:
            yield (i, j, k), self.coords(i, j, k)

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
    compute the derivatives on a small sub-patch around the closest grid point
    and then interpolate these derivative values. However, in order to
    evaluate the derivatives at each of the points of the sub-patch, we need
    sub-patches around these (usually 2 in each direction for a 5-point
    stencil/4th order accurate derivative). Hence, for a 4th order
    differentiation, we need a 9x9 patch around the grid point closest to the
    given point.
    """

    def __init__(self, origin, deltas, box, mat, symmetry, caching=True):
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
        return state

    def __setstate__(self, state):
        r"""Restore this object from the given unpickled state."""
        self.__dict__.update(state)
        # compatibility with data from previous versions
        self._lagrange_cache = self.__dict__.get('_lagrange_cache', [])

    def set_caching(self, caching=True):
        r"""Specify whether to cache the interpolating Lagrange polynomials."""
        if caching and not self.caching:
            self._lagrange_cache = []
        if not caching:
            self._lagrange_cache = None

    @property
    def caching(self):
        r"""Whether caching of interpolating Lagrange polynomials is enabled."""
        return isinstance(self._lagrange_cache, list)

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

    def interpolate(self, point, npts=5, linear=False, interp=True):
        r"""Interpolate the data at a point within the patch.

        We use Lagrange interpolation of a patch around the given point to
        compute an approximation of the data at the given point.

        @param point
            (Physical) coordinates of the point at which to interpolate.
        @param npts
            Number of points per axis to use for interpolation. The default is
            `5` and thus uses two points on each side of the grid point
            closest to `point`. Note that this must be an odd number.
        @param linear
            If `True`, ignore `npts` and perform linear interpolation. Default
            is `False`.
        @param interp
            Whether to interpolate at all. If `False`, the data at the closest
            grid point is returned. Default is `True`.
        """
        real_ijk = self.closest_element(point, floating=True)
        i, j, k = [int(round(i)) for i in real_ijk]
        if not interp:
            return self.at(i, j, k)
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
        dx, _, dz = [linalg.norm(dx) for dx in self.deltas]
        partial_derivs = _diff_4th_order_xz(
            self.mat, diff=diff, dx=dx, dz=dz,
            region=([i], [j], [k]),
        )
        return [v[0,0,0] for v in partial_derivs]

    def diff(self, point, diff=1, interp=True):
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
        @param interp
            Whether to interpolate at all or evaluate at the closest grid
            point. Default is `True` (i.e. do interpolation).
        """
        if diff == 0:
            return self.interpolate(point, interp=interp)
        real_ijk = self.closest_element(point, floating=True)
        i, j, k = [int(round(i)) for i in real_ijk]
        if not interp:
            return self.at(i, j, k, diff=diff)
        n = 2
        real_ijk += [n-i, 0., n-k]
        dx, _, dz = [linalg.norm(dx) for dx in self.deltas]
        partial_derivs = _diff_4th_order_xz(
            self.get_region((i-2*n, i+2*n+1), (self.yidx, self.yidx+1),
                            (k-2*n, k+2*n+1)),
            diff=diff, dx=dx, dz=dz,
            region=(range(n, 2*(n+1)+1), [self.yidx], range(n, 2*(n+1)+1)),
        )
        return [
            interpolate(
                p, real_ijk, cache=self._get_cache(diff, slot), base_idx=(i,j,k)
            )
            for slot, p in enumerate(partial_derivs)
        ]
