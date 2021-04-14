r"""@package motsfinder.metric.fourmetric

Spacetime metric constructed from 3-metric and lapse, shift.

This represents a spacetime 4-metric which is constructed from the spatial
slice 3-metric and lapse/shift fields. To compute derivatives of this metric
(in order to construct e.g. the Christoffel symbols), the extrinsic curvature
as well as time derivatives of lapse and shift need to be supplied as well.

Note that some components of the resulting matrices (indicated in the
documentation of FourMetric.diff()) will not be computed at the moment since
these require second time derivatives of lapse and shift and are not used for
the components of the 4-Christoffel symbols entering the formulas that have
been used thus far.
"""

from __future__ import print_function

import numpy as np

from .base import _GeneralMetric, trivial_lapse, trivial_shift
from .base import trivial_dtlapse, trivial_dtshift


__all__ = [
    'FourMetric',
]


class FourMetric(_GeneralMetric):
    r"""Spacetime metric class using 3-metric and lapse, shift for construction.

    Note that currently, if either of the lapse or shift fields or their time
    derivatives is not available, a trivial default field is used silently.
    If you need derivatives of the metric (e.g. for Christoffel symbols), make
    sure the correct fields are being used (or that the default ones can be
    used without affecting your results in unexpected ways).

    @b Examples

    ```
        g3 = SioMetric('the/simulation/file.it0000001234.s5')
        g4 = FourMetric(g3)
        G = g4.christoffel(point=(0.2, 0.0, 0.5))
        dG = g4.christoffel_deriv(point=(0.2, 0.0, 0.5))
        # dG[0] will be four 4x4 nan-matrices
        # dG[1], dG[2], dG[3] will be four 4x4 matrices each
    ```

    @b Notes

    The spacetime metric \f$g_{\mu\nu}\f$ is constructed from the 3-metric
    \f$h_{ij}\f$ via
    \f[
        g = -(\alpha^2 - \beta_i \beta^i)\ dt^2
            + \beta_i (dt \otimes dx^i + dx^i \otimes dt)
            + h_{ij}\ dx^i \otimes dx^j,
    \f]
    that is, we have
    \f{eqnarray*}{
        g_{00} &=& -(\alpha^2 - \beta_i \beta^i), \\
        g_{i0} &=& g_{0i} = \beta_i := h_{ij} \beta^j \\
        g_{ij} &=& h_{ij}.
    \f}
    Here, \f$\alpha\f$ is the lapse function and \f$\beta^i\f$ the shift
    vector field.
    From the evolution equation
    \f[
        \dot h_{ij} = -2 \alpha K_{ij}
            + {}^{(3)}\nabla_i \beta_j
            + {}^{(3)}\nabla_j \beta_i,
    \f]
    where `K` is the extrinsic curvature of the spatial slice in spacetime and
    \f${}^{(3)}\nabla\f$ the Levi-Civita covariant derivative of the 3-metric
    `h`, we get the following components of the derivative of the 4-metric:
    \f{eqnarray*}{
        \partial_\mu g_{00} &=& -2 \alpha \alpha_{,\mu}
            + (\partial_\mu h_{ij}) \beta^i \beta^j
            + 2 h_{ij} \beta^i_{,\mu} \beta^j
            \\
        \partial_\mu g_{0i} &=& \partial_\mu \beta_i
            = (\partial_\mu h_{ij}) \beta^j + h_{ij} \beta^j_{,\mu}
            \\
        \partial_0 g_{ij} &=& -2 \alpha K_{ij}
            + {}^{(3)}\nabla_i \beta_j
            + {}^{(3)}\nabla_j \beta_i
            \\
        \partial_i g_{jk} &=& \partial_i h_{jk}.
    \f}
    We still need
    \f[
        {}^{(3)}\nabla_i \beta_j
            = h_{jk} {}^{(3)}\nabla_i \beta^k
            = h_{jk} \big(
                \partial_i \beta^k + {}^{(3)}\Gamma^k_{il} \beta^l
            \big)
    \f]
    to compute all the terms.

    For the second derivatives \f$\partial_\mu\partial_\nu g_{\alpha\beta}\f$,
    we explicitly exclude \f$\mu=\nu=0\f$ to avoid second time derivatives of
    lapse and shift (in computations, these values will be set to ``NaN``).
    The formulas then become simply (but a little lengthy):
    \f{eqnarray*}{
        \partial_\mu\partial_\nu g_{00} &=&
            -2(\alpha_{,\mu}\alpha_{,\nu} + \alpha\alpha_{,\mu\nu})
            +(\partial_\mu\partial_\nu h_{jk}) \beta^j \beta^k
            \\&&
            +2\big[ (\partial_\mu h_{jk}) \beta^j_{,\nu} \beta^k
                    +(\partial_\nu h_{jk}) \beta^j_{,\mu} \beta^k \big]
            +2h_{jk} (\beta^j_{,\mu\nu}\beta^k + \beta^j_{,\mu}\beta^k_{,\nu})
            \\
        \partial_\mu\partial_\nu g_{0i} &=&
            (\partial_\mu\partial_\nu h_{ij}) \beta^j
            +(\partial_\mu h_{ij}) \beta^j_{,\nu}
            +(\partial_\nu h_{ij}) \beta^j_{,\mu}
            +h_{ij} \beta^j_{,\mu\nu}
            \\
        \partial_i\partial_0 g_{jk} &=&
            -2\alpha_{,i} K_{jk}
            -2\alpha(\partial_i K_{jk})
            +\partial_i {}^{(3)}\nabla_j \beta_k
            +\partial_i {}^{(3)}\nabla_k \beta_j
            \\
        \partial_i\partial_j g_{kl} &=&
            \partial_i\partial_j h_{kl}.
    \f}
    For concrete computations, we again need to spell out one of the terms:
    \f[
        \partial_i {}^{(3)}\nabla_j \beta_k
            = (\partial_i h_{kl}) {}^{(3)}\nabla_j \beta^l
              +h_{kl} \big[
                \beta^l_{,ij}
                +(\partial_i {}^{(3)}\Gamma^l_{jm}) \beta^m
                +{}^{(3)}\Gamma^l_{jm} \beta^m_{,i}
              \big].
    \f]
    """

    def __init__(self, three_metric):
        r"""Create a 4-metric from a given 3-metric.

        The 3-metric is responsible for providing all information required for
        embedding the slice in spacetime, i.e. it provides the extrinsic
        curvature and lapse, shift (and their time derivatives).
        """
        super(FourMetric, self).__init__()
        self._g3 = three_metric
        self._lapse = three_metric.get_lapse()
        self._shift = three_metric.get_shift()
        self._dtlapse = three_metric.get_dtlapse()
        self._dtshift = three_metric.get_dtshift()
        self._curv = three_metric.get_curv()
        if not self._curv and self._curv != 0:
            print("WARNING: Extrinsic curvature not given. Using zero.")
        msg = ("\n"
               "         If this is correct, consider modifying your metric\n"
               "         class to make this choice explicit. See the\n"
               "         documentation of _ThreeMetric for more details.")
        if self._lapse is None:
            print("WARNING: No lapse given. Using constant 1." + msg)
            self._lapse = trivial_lapse
        if self._shift is None:
            print("WARNING: No shift given. Using zero." + msg)
            self._shift = trivial_shift
        if self._dtlapse is None:
            print("WARNING: No dtlapse given. Using zero." + msg)
            self._dtlapse = trivial_dtlapse
        if self._dtshift is None:
            print("WARNING: No dtshift given. Using zero." + msg)
            self._dtshift = trivial_dtshift

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
        r"""Return derivatives of the metric at the given point.

        @param point
            Point (tuple/list/array) in the current slice at which to compute
            the derivatives. Note that the time coordinate is inferred from
            the provided 3-metric, i.e. only supply the 3-D coordinates here.
        @param inverse
            Whether to compute the derivatives of the inverse metric (indices
            up).
        @param diff
            Derivative order. Default is 1.

        @return Multidimensional list with indices `i1, i2, ..., k, l`
            corresponding to \f$\partial_{i_1}\partial_{i_2}\ldots g_{kl}\f$
            (or the inverse components if ``inverse==True``).

        @b Notes

        First derivatives are completely implements and second derivatives are
        implement except for second time derivatives. This means that all
        returned matrices will contain computed data except the following:

            g4.diff(point, diff=2)[0,0]  # a 4x4 nan-matrix

        This corresponds to the double time derivative of the metric. Note
        that the elements not computed will be set to ``NaN`` to ensure that
        no wrong results will propagate through computations undetected.
        """
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
        r"""Second spacetime derivative of lapse, shape is `(4, 4)`.

        Double time derivatives are not computed and set to ``NaN``.
        """
        d2xlapse = self._lapse(point, diff=2)
        dtxlapse = self._dtlapse(point, diff=1)
        d2lapse = np.zeros((4, 4)) # a,b -> partial_a partial_b lapse
        d2lapse[0,0] = np.nan # second time derivative not implemented
        d2lapse[0,1:] = d2lapse[1:,0] = dtxlapse
        d2lapse[1:,1:] = d2xlapse
        return d2lapse

    def d2shift(self, point):
        r"""Second spacetime derivative of shift, shape is `(4, 4, 3)`.

        Double time derivatives are not computed and set to ``NaN``.
        """
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
        r"""Timelike normal vector to the slice.

        @param point
            3-D point in the slice to evaluate at. The time coordinate is
            implicitly taken from the 3-metric.
        @param diff
            Derivative order to compute. Default is 0, i.e. no derivative.
            Note that time derivatives are currently only implemented for
            ``diff=1``. However, all non-computed values are set to ``NaN``.
            This means that any component with a derivative index being zero
            will be ``NaN``. Spatial derivatives of the time component are
            computed, however.
        """
        lapse = self._lapse(point)
        shift = self._shift(point)
        if diff == 0:
            n = np.ones(4)
            n[1:] = -shift
            n /= lapse
            return n
        if diff == 1:
            dlapse = self.dlapse(point)
            dshift = self.dshift(point)
            dn = np.zeros((4, 4)) # a,b -> partial_a n^b
            dn[:,0] = -dlapse/lapse**2
            dn[:,1:] = (
                1/lapse**2 * np.einsum('i,m->im', dlapse, shift)
                - 1/lapse * dshift
            )
            return dn
        d2xlapse = self._lapse(point, diff=2) # shape=(3,3), i,j -> del_i del_j lapse
        d2xshift = self._shift(point, diff=2) # shape=(3,3,3), i,j,k -> del_i del_j shift^k
        if diff == 2:
            dxlapse = self._lapse(point, diff=1)
            dxshift = self._shift(point, diff=1)
            ddn = np.zeros((4, 4, 4)) # a,b,c -> del_a del_b n^c
            ddn[0,:,:] = ddn[:,0,:] = np.nan # time derivatives not computed
            ddn[1:,1:,0] = - d2xlapse/lapse**2 + 2 * np.outer(dxlapse, dxlapse)/lapse**3
            ddn[1:,1:,1:] = (
                np.einsum('ij,m', d2xlapse, shift) / lapse**2
                - 2 * np.einsum('i,j,m', dxlapse, dxlapse, shift) / lapse**3
                + np.einsum('i,jm', dxlapse, dxshift) / lapse**2
                + np.einsum('j,im', dxlapse, dxshift) / lapse**2
                - d2xshift / lapse
            )
            return ddn
        raise NotImplementedError
