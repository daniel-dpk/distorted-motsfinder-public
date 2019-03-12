r"""@package motsfinder.metric.fourmetric

Spacetime metric constructed from 3-metric and lapse, shift.

This represents a spacetime 4-metric which is constructed from the spatial
slice 3-metric and lapse/shift fields. To compute derivatives of this metric
(in order to construct e.g. the Christoffel symbols), the extrinsic curvature
needs to be supplied as well.
"""

from __future__ import print_function

import numpy as np
from scipy import linalg

from .base import _GeneralMetric


__all__ = [
    'FourMetric',
]


# It is customary to denote indices of tensors without spaces, e.g.:
#   T_{ijk}  =>  T[i,j,k]
# We disable the respective pylint warning for this file.
# pylint: disable=bad-whitespace


class FourMetric(_GeneralMetric):
    r"""Spacetime metric class using 3-metric and lapse, shift for construction."""
    def __init__(self, three_metric):
        super(FourMetric, self).__init__()
        self._g3 = three_metric
        self._lapse = three_metric.get_lapse()
        self._shift = three_metric.get_shift()
        self._dtlapse = three_metric.get_dtlapse()
        self._dtshift = three_metric.get_dtshift()
        self._curv = three_metric.get_curv()
        if self._lapse is None:
            self._lapse = _default_lapse
        if self._shift is None:
            self._shift = _default_shift
        if self._dtlapse is None:
            self._dtlapse = _default_dtlapse
        if self._dtshift is None:
            self._dtshift = _default_dtshift

    @property
    def lapse(self):
        r"""Lapse function object."""
        return self._lapse

    @property
    def shift(self):
        r"""Shift vector field object."""
        return self._shift

    def _mat_at(self, point):
        g3 = self._g3.at(point).mat
        lapse = self._lapse(point)
        shift = self._shift(point)
        shift_cov = g3.dot(shift)
        g4 = np.zeros((4, 4))
        g4[1:,1:] = g3
        g4[0,0] = -(lapse**2 - shift_cov.dot(shift))
        g4[1:,0] = g4[0,1:] = shift_cov
        return g4

    def diff(self, point, inverse=False, diff=1):
        if inverse:
            return self._compute_inverse_diff(point, diff=diff)
        if diff == 0:
            return self._mat_at(point)
        g3 = self._g3.at(point).mat
        lapse = self._lapse(point)
        shift = self._shift(point) # beta^i
        dlapse = self.dlapse(point)
        dshift = self.dshift(point) # shape=(4,3), a,i -> del_a beta^i
        shift_cov = g3.dot(shift) # beta_i
        K = self._curv(point) if self._curv else 0
        G3 = self._g3.christoffel(point)
        cov_shift = ( # shape=(3,3), i,k -> D_i beta^k
            dshift[1:,:] + np.einsum('kil,l->ik', G3, shift)
        )
        cov_shift_cov = ( # shape=(3,3), i,j -> D_i beta_j
            np.einsum('jk,ik->ij', g3, cov_shift)
        )
        dg3 = self._dg3(point, K, lapse, cov_shift_cov) # shape=(4,3,3)
        if diff == 1:
            dg4 = np.zeros((4, 4, 4)) # a,b,c -> partial_a g_bc
            # partial_mu g_tt
            dg4[:,0,0] = (
                -2 * lapse * dlapse
                + np.einsum('amn,m,n->a', dg3, shift, shift)
                + 2 * np.einsum('mn,am,n->a', g3, dshift, shift)
            )
            # partial_mu g_ti = partial_mu g_it
            dshift_cov = ( # shape=(4,3), a,i -> del_a beta_i
                np.einsum('aij,j->ai', dg3, shift)
                + np.einsum('ij,aj->ai', g3, dshift)
            )
            dg4[:,0,1:] = dg4[:,1:,0] = dshift_cov
            dg4[:,1:,1:] = dg3
            return dg4
        d2lapse = self.d2lapse(point) # shape=(4,4), a,b -> del_a del_b lapse
        d2shift = self.d2shift(point) # shape=(4,4,3), a,b,i -> del_a del_b shift^i
        ddg3 = self._ddg3( # shape=(4,4,3,3), a,b,i,j -> del_a del_b h_ij
            point, g3, G3, dg3, K, shift, lapse, dlapse, dshift, d2shift,
            cov_shift
        )
        if diff == 2:
            ddg4 = np.zeros((4, 4, 4, 4)) # m,n,a,b -> del_m del_n g_ab
            ddg4[:,:,0,0] = (
                -2 * (np.outer(dlapse, dlapse) + lapse * d2lapse)
                + np.einsum('mnij,i,j->mn', ddg3, shift, shift)
                + 2 * np.einsum('mij,ni,j->mn', dg3, dshift, shift)
                + 2 * np.einsum('nij,mi,j->mn', dg3, dshift, shift)
                + 2 * np.einsum('ij,mni,j->mn', g3, d2shift, shift)
                + 2 * np.einsum('ij,mi,nj->mn', g3, dshift, dshift)
            )
            d2shift_cov = ( # shape=(4,4,3), a,b,i -> del_a del_b shift_i
                np.einsum('abij,j->abi', ddg3, shift)
                + np.einsum('aij,bj->abi', dg3, dshift)
                + np.einsum('bij,aj->abi', dg3, dshift)
                + np.einsum('ij,abj->abi', g3, d2shift)
            )
            ddg4[:,:,0,1:] = ddg4[:,:,1:,0] = d2shift_cov
            ddg4[:,:,1:,1:] = ddg3
            return ddg4
        raise NotImplementedError

    def dlapse(self, point):
        r"""Spacetime derivatives of lapse field, shape is `(4,)`."""
        dtlapse = self._dtlapse(point)
        dxlapse = self._lapse(point, diff=1)
        return np.array([dtlapse] + dxlapse.tolist())

    def dshift(self, point):
        r"""Spacetime derivatives of shift field, shape is `(4,3)`."""
        dtshift = self._dtshift(point)
        dxshift = self._shift(point, diff=1)
        return np.array([dtshift] + dxshift.tolist())

    def d2lapse(self, point):
        r"""Second spacetime derivative of lapse, shape is `(4, 4)`."""
        d2xlapse = self._lapse(point, diff=2)
        dtxlapse = self._dtlapse(point, diff=1)
        d2lapse = np.zeros((4, 4)) # a,b -> partial_a partial_b lapse
        d2lapse[0,0] = np.nan # second time derivative not implemented
        d2lapse[0,1:] = d2lapse[1:,0] = dtxlapse
        d2lapse[1:,1:] = d2xlapse
        return d2lapse

    def d2shift(self, point):
        r"""Second spacetime derivative of shift, shape is `(4, 4, 3)`."""
        d2xshift = self._shift(point, diff=2) # a,b,m -> partial_a partial_b shift^m
        dtxshift = self._dtshift(point, diff=1) # a,m -> partial_a partial_t shift^m
        d2shift = np.zeros((4, 4, 3)) # mu,nu,m -> partial_mu partial_nu shift^m
        d2shift[0,0,:] = np.nan # second time derivative not implemented
        d2shift[0,1:,:] = d2shift[1:,0,:] = dtxshift
        d2shift[1:,1:,:] = d2xshift
        return d2shift

    def _dg3(self, point, K, lapse, cov_shift_cov):
        r"""Spacetime derivatives of 3-metric, shape is `(4,3,3)`."""
        dtg3 = -2 * lapse * K + cov_shift_cov + cov_shift_cov.T # evolution equation
        dxg3 = np.asarray(self._g3.diff(point, diff=1))
        return np.array([dtg3] + dxg3.tolist())

    def _ddg3(self, point, g3, G3, dg3, K, shift, lapse, dlapse, dshift,
              d2shift, cov_shift):
        r"""Second spacetime derivatives of 3-metric, shape is `(4,4,3,3)`."""
        dK = self._curv(point, diff=1) if self._curv else 0
        dG3 = self._g3.christoffel_deriv(point)
        dcov_shift = (
            np.einsum('ink,mk->imn', dg3[1:], cov_shift)
            + np.einsum('nk,imk->imn', g3, d2shift[1:,1:])
            + np.einsum('nk,ikml,l->imn', g3, dG3, shift)
            + np.einsum('nk,kml,il->imn', g3, G3, dshift[1:])
        )
        d2xg = self._g3.diff(point, diff=2)
        dtxg = (
            - 2 * np.einsum('i,mn->imn', dlapse[1:], K)
            - 2 * lapse * dK
            + dcov_shift + dcov_shift.swapaxes(1, 2)
        )
        ddg3 = np.zeros((4, 4, 3, 3))
        ddg3[0,0] = np.nan # second time derivative not implemented
        ddg3[0,1:,:,:] = ddg3[1:,0,:,:] = dtxg
        ddg3[1:,1:,:,:] = d2xg
        return ddg3

    def normal(self, point, diff=0):
        r"""Timelike normal vector to the slice."""
        lapse = self._lapse(point)
        shift = self._shift(point)
        if diff == 0:
            n = np.ones(4)
            n[1:] = -shift
            n /= lapse
            return n
        dxlapse = self._lapse(point, diff=1)
        dxshift = self._shift(point, diff=1)
        if diff == 1:
            dn = np.zeros((4, 4)) # a,b -> del_a n^b
            dn[0,:] = np.nan # not needed, but we don't want *wrong* values even if not needed
            dn[1:,0] = -dxlapse/lapse**2
            dn[1:,1:] = (
                1/lapse**2 * np.einsum('i,m->im', dxlapse, shift)
                - 1/lapse * dxshift
            )
            return dn
        d2xlapse = self._lapse(point, diff=2) # shape=(3,3), i,j -> del_i del_j lapse
        d2xshift = self._shift(point, diff=2) # shape=(3,3,3), i,j,k -> del_i del_j shift^k
        if diff == 2:
            ddn = np.zeros((4, 4, 4)) # a,b,c -> del_a del_b n^c
            ddn[0,:,:] = ddn[:,0,:] = np.nan # time derivatives not computed
            ddn[1:,1:,0] = - d2xlapse/lapse**2 + 2 * np.outer(dxlapse, dxlapse)/lapse**3
            ddn[1:,1:,1:] = (
                np.einsum('ij,m', d2xlapse, shift) / lapse**2
                - np.einsum('i,j,m', dxlapse, dxlapse, shift) / lapse**3
                + np.einsum('i,jm', dxlapse, dxshift) / lapse**2
                + np.einsum('j,im', dxlapse, dxshift) / lapse**2
                - d2xshift / lapse
            )
            return ddn
        raise NotImplementedError


def _default_lapse(point, diff=0):
    if diff:
        return np.zeros(shape=[3] * diff)
    return 1.0

def _default_dtlapse(point, diff=0):
    if diff:
        return np.zeros(shape=[3] * diff)
    return 0.0

def _default_shift(point, diff=0):
    return np.zeros(shape=[3] * (diff + 1))

def _default_dtshift(point, diff=0):
    return np.zeros(shape=[3] * (diff + 1))
