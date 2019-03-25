r"""@package motsfinder.metric.discrete.utils

Utility functions for inspecting discrete data.

Currently, this only includes a helper class, ConstraintAnalyzer, to compute
the residual violations of the constraints of numerical data along a surface
(represented by a curve in the xz-plane).
"""

import numpy as np


__all__ = [
    "ConstraintAnalyzer",
]


# It is customary to denote indices of tensors without spaces, e.g.:
#   T_{ijk}  =>  T[i,j,k]
# We disable the respective pylint warning for this file.
# pylint: disable=bad-whitespace


class ConstraintAnalyzer():
    r"""Collection of convenience methods to check constraint violations near
    a surface.

    This class is a collection of methods to compute the neighborhood of a
    curve (representing an axisymmetric surface) in the xz-plane. The results
    are suitable for visualising the numeric grid density near the curve and
    how the constraints are violated in that neighborhood.
    """

    def __init__(self, curve):
        r"""Construct a ConstraintAnalyzer for the given curve."""
        self._curve = curve

    def collocation_points(self, snap=True, indices=False, params=None):
        r"""Return the collocation points along the curve.

        @param snap
            Whether to return the numerical grid points closest to the
            collocation points of the curve instead of the collocation points
            themselves. Default is `True`.
        @param indices
            If `True`, return the indices of the discrete matrix elements
            close to the collocation points. If ``snap=True``, the result will
            be a list of 3-integer-elements, i.e. the indices of the matrix.
            If ``snap=False``, the elements will be 3 floats each representing
            the points in *index space*. Default is `False`.
        @param params
            Optional list of parameters along the curve to take instead of the
            actual collocation points.
        """
        c = self._curve
        if params is None:
            params = self._curve_params()
        pts = [c(l, xyz=True) for l in params]
        g00 = c.metric.component_matrix(0, 0)
        if snap:
            indices_list = [g00.closest_element(p) for p in pts]
            # Remove duplicates (points snapped to the same grid point).
            # Order is not too important, but Python 3.7's dict() will be
            # ordered (as is Python 3.6's in CPython implementation), so that
            # order is preserved.
            indices_list = list(dict.fromkeys(indices_list))
            if indices:
                return indices_list
            pts = [g00.coords(*ijk) for ijk in indices_list]
        elif indices:
            indices_list = [g00.closest_element(p, floating=True) for p in pts]
            return indices_list
        return pts

    def points_close_to_curve(self, radius=0, indices=False,
                              params=None, pts_indices=None):
        r"""Return the grid points in a neighborhood of the curve.

        This collects all grid points close to the curve's collocation points
        (by default) and then adds all neighbors of these points up to
        distance `radius` (which should hence be an integer). As a result, we
        get a more or less wide stripe of grid points along the curve, the
        physical width of which depends on the resolution of the grid.

        This may help in assessing whether a particular grid resolution is
        sufficient to resolve certain features or whether an interpolation
        method *keeps away* far enough from critical regions such as a
        singularity in the grid.

        Note that points are collected only up to the z-axis, even if
        neighbors would lie beyond.

        @param radius
            Number of neighboring grid points to add.
        @param indices
            If `True`, return the indices of the matrix elements instead of
            their physical coordinates. Default is `False`.
        @param params
            Optional list of parameters along the curve to take instead of the
            actual collocation points. Only used if no `pts_indices` are
            given.
        @param pts_indices
            Optional initial list of grid points (given via their indices) to
            take instead of collecting those close to the curve.
        """
        if pts_indices is None:
            pts_indices = self.collocation_points(snap=True, indices=True,
                                                  params=params)
        c = self._curve
        g00 = c.metric.component_matrix(0, 0)
        if radius == 0:
            indices_list = pts_indices
        else:
            mat = np.zeros(shape=g00.shape, dtype=bool)
            r = radius
            for i, j, k in pts_indices:
                mat[self._patch_slices(i,j,k,r)] = True
            indices_list = list(zip(*np.where(mat)))
        if indices:
            return indices_list
        pts = [g00.coords(*ijk) for ijk in indices_list]
        return pts

    def _patch_slices(self, i, j, k, radius):
        r"""Construct a tuple of 3 slice objects for a patch of given radius.

        Note that the slices will be confined to non-negative x-values even if
        the box around element `i,j,k` would extend to negative x-values
        across the z-axis.

        @param i,j,k
            Indices of the grid point around which to create a symmetric box.
        @param radius
            Number of neighbors on each side to include.
        """
        g00 = self._curve.metric.component_matrix(0, 0)
        r = radius
        return (
            slice(max(g00.xidx, i-r), i+r+1),
            slice(max(g00.yidx, j-r), j+r+1),
            slice(k-r, k+r+1)
        )

    def _curve_params(self):
        r"""Return the Gauss collocation points in curve parameter space."""
        return self._curve.h.collocation_points(lobatto=False)

    def constraints_on_patch(self, pts_indices):
        r"""Compute constraint violations on a patch of grid points.

        @return Two NumPy matrices `ham` and `mom` for the complete numerical
            domain, containing for each grid point in `pts_indices` the
            residual constraints. The remaining elements will be exact zeros.
            `ham` will contain the quantity
            \f$K_{ab}K^{ab} - (K^c_{\ c})^2 - R\f$ and `mom` the quantity
            \f$\sqrt{\mathcal{H}_a\mathcal{H}^a}\f$, where
            \f$\mathcal{H}_b = D^a(K_{ab} - g_{ab} K^c_{\ c})\f$.

        @param pts_indices
            List of grid points to consider.
        """
        g = self._curve.metric
        g00 = g.component_matrix(0, 0)
        pts = [g00.coords(*ijk) for ijk in pts_indices]
        ham = np.zeros(shape=g00.shape, dtype=float)
        mom = np.zeros(shape=g00.shape, dtype=float)
        interpolate = g.interpolate
        try:
            g.interpolate = False
            for pt, ijk in zip(pts, pts_indices):
                ijk = tuple(ijk)
                ham[ijk], mom[ijk] = g.constraints(pt, norm=True)
        finally:
            g.interpolate = interpolate
        return ham, mom

    def max_constraints_per_chunk(self, radii=(0,), params=None,
                                  full_output=False):
        r"""Compute the maximum constraint violation along the curve.

        For each radius in `radii`, this method walks along the (collocation
        points of the) curve and registers the maximum constraint violation on
        a patch of that radius around the current point. As a result, this
        allows for checking how the constraints become increasingly violated
        the further away from the curve we go. Plotting the results could
        produce a family of curves (one for each radius) showing the violation
        along the curve.

        @return NumPy array `results` where ``results[i,j,k]`` contains for
            the ``i'th`` radius and the ``j'th`` collocation point the
            ``k``-indexed triple ``(param, ham, mom)``, where ``param`` is the
            parameter along the curve, ``ham`` is the Hamiltonian and ``mom``
            the (norm of the) momentum constraint. This means that
            ``results[1,:,0]`` is a sequence of parameters along the curve
            while ``results[1,:,2]`` contains the respective momentum
            constraint violations for radius ``radii[1]``.
            If ``full_output=True``, also returns the matrices containing the
            computed Hamiltonian and momentum constraints as well as a list of
            matrix indices for which constraints have been computed.

        @param radii
            Sequence of radii (number of neighbors) to compute constraints
            for. Default is ``(0,)``.
        @param params
            Optional list of parameters along the curve to take instead of the
            actual collocation points.
        @param full_output
            If `True`, also return the computed constraints and at which
            points they were computed (see above). Default is `False`.
        """
        if params is None:
            params = self._curve_params()
        snapped_collocation_pts_indices = self.collocation_points(
            snap=True, indices=True, params=params
        )
        max_r = np.max(radii)
        all_pts_ijk = self.points_close_to_curve(
            radius=max_r, indices=True,
            pts_indices=snapped_collocation_pts_indices,
        )
        ham, mom = self.constraints_on_patch(all_pts_ijk)
        ham = np.absolute(ham)
        mom = np.absolute(mom)
        results = np.array(
            [[[l] + [np.nanmax(m[self._patch_slices(i,j,k,r)])
                     for m in (ham, mom)]
              for l, (i,j,k) in zip(params, snapped_collocation_pts_indices)]
             for r in radii]
        )
        if full_output:
            return results, ham, mom, all_pts_ijk
        return results

    def max_constraints_per_radius(self, radii, params=None,
                                   full_output=False):
        r"""Compute the maximum constraint violation at different distances
        from the curve.

        Similar to max_constraints_per_chunk(), but collapses the parameter
        along the curve by taking the respective maximum violation.

        @return `max_hams, max_moms`, where `max_hams` contains for each
            radius in `radii` the maximum violation of the Hamiltonian
            constraint on a stripe of grid points around the curve, and
            similarly `max_moms` the violation of the momentum constraint.
            If ``full_output=True``, also returns the matrices containing the
            computed Hamiltonian and momentum constraints as well as a list of
            matrix indices for which constraints have been computed.
        """
        constraints, ham, mom, all_pts_ijk = self.max_constraints_per_chunk(
            radii=radii, params=params, full_output=True
        )
        max_hams, max_moms = [
            np.array([np.nanmax(constraints[i,:,n]) for i in range(len(radii))])
            for n in (1, 2)
        ]
        if full_output:
            return max_hams, max_moms, ham, mom, all_pts_ijk
        return max_hams, max_moms
