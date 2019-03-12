r"""@package motsfinder.metric.discrete.utils

Utility functions for inspecting discrete data.
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
    def __init__(self, curve):
        self._curve = curve

    def collocation_points(self, snap=True, indices=False, params=None):
        c = self._curve
        if params is None:
            params = self._curve_params()
        pts = [c(l, xyz=True) for l in params]
        g00 = c.metric.component_matrix(0, 0)
        if snap:
            indices_list = [g00.closest_element(p) for p in pts]
            # Remove tuplicates (points snapped to the same grid point).
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
        g00 = self._curve.metric.component_matrix(0, 0)
        r = radius
        return (
            slice(max(g00.xidx, i-r), i+r+1),
            slice(max(g00.yidx, j-r), j+r+1),
            slice(k-r, k+r+1)
        )

    def _curve_params(self):
        return self._curve.h.collocation_points(lobatto=False)

    def constraints_on_patch(self, pts_indices):
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
