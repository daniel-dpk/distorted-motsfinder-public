r"""@package motsfinder.axisym.curve.expcalc

Computation class storing interim results of expansion calculations.

The implementation here uses the formulas derived in
\ref thornburg2003_1 "[1]". Specifically, we make heavy use of the quantities
`A, B, C, D` defined in \ref thornburg2003_1 "[1]" in equation (12) to compute
the expansion \f$ \Theta \f$ using equation (11). See also
\ref pookkolb2018_1 "[2]" and the docstrings of the individual procedures.

In the base class ExpansionCalc defined in this module, we do not consider how
the used quantities \f$ s_i \f$ and \f$ \partial_i s_j \f$ are obtained. This
depends on how the surfaces are represented and hence is the responsibility of
subclasses to implement. Additionally, subclasses also need to supply surface
parameter derivatives defined in \ref thornburg2003_1 "[1]" as
\f$ X^u_i = \partial_i y^u \f$ and
\f$ X^u_{ij} = \partial_i\partial_j y^u \f$.
In the axisymmetric case considered here, we have only one parameter,
\f$ y^u = \lambda \f$ along the curve, and hence drop the `u` superscript.

Note that in this code, we call the covector field \f$ X_i \f$ simply `X` and
the 2nd rank tensor field \f$ X_{ij} \f$ simply `Y` (Python cannot
differentiate between objects based on how many indices you use).


@b Examples

See implementations starshapedcurve._StarShapedExpansionCalc and
refparamcurve._RefParamExpansionCalc.

@b References

\anchor thornburg2003_1 [1] Thornburg, Jonathan. "A fast apparent horizon finder
    for three-dimensional Cartesian grids in numerical relativity." Classical
    and quantum gravity 21.2 (2003): 743.

\anchor pookkolb2018_1 [2] D. Pook-Kolb, O. Birnholtz, B. Krishnan and E.
    Schnetter, "The existence and stability of marginally trapped surfaces."
    arXiv:1811.10405 [gr-qc].
"""

from abc import ABCMeta, abstractmethod
from math import fsum

from six import add_metaclass
import numpy as np
from scipy.misc import derivative

from ...utils import cache_method_results
from ...numutils import inverse_2x2_matrix_derivative
from ...metric import christoffel_symbols, christoffel_deriv
from ...metric import riemann_components


__all__ = []


# It is customary to denote indices of tensors without spaces, e.g.:
#   T_{ijk}  =>  T[i,j,k]
# We disable the respective pylint warning for this file.
# pylint: disable=bad-whitespace


@add_metaclass(ABCMeta)
class ExpansionCalc(object):
    r"""Abstract base class for computing the expansion at one point.

    This class serves as coordinator for computing the expansion and
    functional derivatives w.r.t. the horizon function. Sub classes need only
    implement a small number of computational methods.

    The purpose of having a separate class hierarchy for computing the
    expansion (as opposed to doing all the computations inside the curve
    classes) is to be able to store a number of interim results valid only for
    the results at one point of the surface. Including these as `cache` in the
    curve classes would in principle be possible. To ease management of cache
    invalidation (when computing at a different point), the complete cache
    should live on one object. The ExpansionCalc class and its sub classes can
    be interpreted as such a cache, with added functionality to do the
    necessary computations using the cached values.
    """
    def __init__(self, curve, h_fun, param, metric, extr_curvature):
        r"""Create a "calc" object for certain point of a curve.

        The curve represents an axisymmetric surface.

        @param curve (expcurve.ExpansionCurve)
            The curve representing the (trial) surface on which to compute the
            expansion and other quantities.
        @param h_fun (exprs.numexpr.NumericExpression)
            The (1D) "horizon" function. The subclasses implementing this
            ExpansionCalc class are free to interpret as they wish.
        @param param (float)
            The parameter value along the `curve` at which the quantities
            should be computed.
        @param metric
            The Riemannian 3-metric defining the geometry of the surrounding
            space.
        @param extr_curvature
            Symmetric covariant 2-tensor defining the extrinsic curvature `K`.
            If not specified, time-symmetry is assumed (i.e. `K=0`).
        """
        ## Step sizes for FD numerical differentiation of the expansion
        ## \wrt `h`, `h'`, ``h''``, respectively.
        self.dx_hdiffs = (1e-6, 1e-6, 1e-3)
        ## Finite difference differentiation order.
        self.fd_order = 3
        ## The curve representing the (trial) surface.
        self.curve = curve
        ## Horizon function (in case we need higher derivatives than ``h''``).
        self.h_fun = h_fun
        ## Value of horizon function `h` at the given parameter.
        self.h = h_fun(param)
        ## Value of `h'` at the given parameter.
        self.dh = h_fun.diff(param, n=1)
        ## Value of ``h''`` at the given parameter.
        self.ddh = h_fun.diff(param, n=2)
        ## Parameter on the curve at which to do the computations.
        self.param = param
        point = curve(param, xyz=True)
        ## 3D point in `x`,`y`,`z` coordinates.
        self.point = point
        ## Metric (tensor field).
        self.metric = metric
        ## Metric tensor at the point to do computations at.
        self.g = metric.at(point)
        if extr_curvature is None:
            ## Extrinsic curvature at the point to do computations at.
            self.K = None
        else:
            self.K = extr_curvature(point)
        # Cached metric derivatives (computed on-demand).
        self._dg = None
        self._dg_inv = None
        self._ddg = None
        self._ddg_inv = None
        ## Derivatives \f$ \partial_i \ln\sqrt{g} \f$
        self.dlnsqrtg = np.asarray(metric.diff_lnsqrtg(point))
        s, ds, X, Y = self._compute_s_ds_X_Y()
        ## Normal covector (not normalized).
        self.s = np.asarray(s)
        ## Derivative matrix \f$ \partial_i s_j \f$ of normal vector.
        self.ds = np.asarray(ds)
        ## Derivative covector \f$ X_i := \partial_i \lambda(\vec x) \f$.
        self.X = np.asarray(X)
        ## Second derivatives \f$ Y := X_{ij} := \partial_i\partial_j\lambda\f$.
        self.Y = np.asarray(Y)
        ## Contravariant normal vector (not normalized).
        self.s_up = self.g.raise_idx(s)
        ## Contravariant parameter derivative \f$ X^i := g^{ij}X_j \f$.
        self.X_up = self.g.raise_idx(X)
        ABCD, trK = self._compute_ABCDtrK()
        ## A, B, C, D terms of the Thornburg expansion formula.
        self.ABCD = ABCD
        ## Trace of the extrinsic curvature.
        self.trK = trK
        ## Cached expansion result.
        self._Th = None

    @property
    def dg(self):
        r"""Derivative of 3-metric components \wrt x,y,z."""
        if self._dg is None:
            self._dg = np.asarray(self.metric.diff(self.point, diff=1))
        return self._dg

    @property
    def dg_inv(self):
        r"""Derivative of inverse 3-metric components.

        This is computed using
        \f$0 = \partial_i \delta^a_b = \partial_i(g^{ac}g_{cb})\f$
        from which we get
        \f[
            \partial_i g^{-1} = -g^{-1} (\partial_i g) g^{-1}.
        \f]
        """
        if self._dg_inv is None:
            g_inv = self.g.inv
            dg = self.dg
            # explanation:
            #   X = g_inv.dot(dg) == g^ad partial_i g_db
            #   Y = X.dot(g_inv) == X^a_ib g^be
            #   => Y has indices Y[a,i,e] == (g^-1 partial_i g g^-1)^ae
            #   we want "i" to be the first axis => swapaxes(0, 1)
            # equivalent to: -np.einsum('ic,acd,dj', _g_inv, _dg, _g_inv)
            self._dg_inv = -(
                g_inv.dot(dg).dot(g_inv).swapaxes(0, 1)
            )
        return self._dg_inv

    @property
    def ddg(self):
        r"""Second derivatives of 3-metric components."""
        if self._ddg is None:
            self._ddg = np.asarray(self.metric.diff(self.point, diff=2))
        return self._ddg

    @property
    def ddg_inv(self):
        r"""Second derivatives of inverse 3-metric components.

        As for `dg_inv`, using
        \f$0 = \partial_i \partial_j \delta^a_b
        = \partial_i \partial_j (g^{ac}g_{cb})\f$
        we get
        \f[
            \partial_i \partial_j g^{-1}
                = -g^{-1}\big[
                    (\partial_i \partial_j g) g^{-1}
                    + (\partial_j g) (\partial_i g^{-1})
                    + (\partial_i g) (\partial_j g^{-1})
                \big].
        \f]
        """
        if self._ddg_inv is None:
            g_inv = self.g.inv
            dg = self.dg
            dg_inv = self.dg_inv
            ddg = self.ddg
            # equivalent to:
            #   -(
            #       + np.einsum('ij,abjk,kl', g_inv, ddg, g_inv)
            #       + np.einsum('ij,bjk,akl', g_inv, dg, dg_inv)
            #       + np.einsum('ij,ajk,bkl', g_inv, dg, dg_inv)
            #   )
            tmp = g_inv.dot(dg).dot(dg_inv)
            self._ddg_inv = -(
                + np.moveaxis(g_inv.dot(ddg).dot(g_inv), [1,2,0], [0,1,2])
                + np.moveaxis(tmp, [2,1,0], [0,1,2])
                + np.moveaxis(tmp, [1,2,0], [0,1,2])
            )
        return self._ddg_inv

    def _compute_ABCDtrK(self):
        r"""Compute the A, B, C, D and trace(K) terms.

        The computation only uses the cached covariant normal `s` and its
        derivatives `ds` (in addition to the metric and extrinsic curvature,
        of course). This means that any subclass only needs to implement
        computing `s` and `ds` in order to use this function.

        This computes the terms as defined in equation (12) in
        \ref thornburg2003_1 "[1]".
        """
        s, s_up, ds = self.s, self.s_up, self.ds
        g, dg_inv, dlnsqrtg = self.g, self.dg_inv, self.dlnsqrtg
        A = (
            - ds.dot(s_up).dot(s_up)
            - 0.5 * dg_inv.dot(s).dot(s).dot(s_up)
        )
        B = (
            dg_inv.dot(s).diagonal().sum()
            + g.inv.dot(ds).diagonal().sum()
            + dlnsqrtg.dot(s_up)
        )
        if self.K is None:
            trK = 0.0
            C = 0.0
        else:
            trK = g.inv.dot(self.K).diagonal().sum()
            C = self.K.dot(s_up).dot(s_up)
        D = s.dot(s_up)
        return (A, B, C, D), trK

    def expansion(self):
        r"""Compute the expansion at the configured point.

        This implements equation (11) in \ref thornburg2003_1 "[1]".
        """
        if self._Th is None:
            A, B, C, D = self.ABCD
            self._Th = A/D**1.5 + B/D**0.5 + C/D - self.trK
        return self._Th

    def diff(self, hdiff=0):
        r"""Compute derivative of expansion \wrt `h`, `h'`, or ``h''``.

        The argument `hdiff` controls the derivative order of `h` with
        respect to which to differentiate the expansion, i.e. `hdiff=0` will
        compute \f$ \partial_{h}\Theta \f$, while for `hdiff=2` we
        compute \f$ \partial_{h''}\Theta \f$.

        Numerical FD differentiation is performed if a `NotImplementedError`
        is raised in one of the subroutines.
        """
        try:
            return self._diff(hdiff=hdiff)
        except NotImplementedError:
            return self._diff_FD(hdiff=hdiff)

    def _diff_FD(self, hdiff):
        r"""Compute derivatives of the expansion using finite differencing.

        Since the expansion depends on `h` and its derivatives only
        ultra-locally, a reasonable approximation to the variational
        derivative of the expansion w.r.t. `h` can be obtained by varying `h`
        (or derivatives) point-wise, i.e. compute the usual partial derivative
        of the expansion w.r.t. `h`. This can be approximated using a finite
        difference differentiation, which is done in this function. Note that
        irrespective of the accuracy of this approximation, the test whether
        the expansion has the desired value (e.g. 0.0 for a MOTS) is
        independent of the results computed here.
        """
        h_orig = self.curve.h
        Th0 = self.expansion()
        param = self.param
        h_plus_eps = _FuncVariation(h_orig.evaluator(), diff=hdiff)
        with self.curve.override_evaluator(h_plus_eps):
            def f(eps):
                if eps == 0:
                    return Th0
                h_plus_eps.eps = eps
                with self.curve.suspend_calc_obj():
                    return self.curve.expansion(param)
            dx = self.dx_hdiffs[hdiff]
            return derivative(f, x0=0.0, n=1, dx=dx, order=self.fd_order)

    def _diff(self, hdiff):
        r"""Compute analytical functional derivatives of the expansion.

        This may raise a `NotImplementedError`, indicating that FD
        differentiation needs to be performed.

        @param hdiff
            Derivative order of `h` to differentiate the expansion by (see
            below). E.g., a value of `0` will compute \f$\partial_h \Theta\f$.

        @b Notes

        In general, due to the ultra-local dependency of the expansion on `h`
        and its first two derivatives, we can treat the variational
        differentiation like a simple partial differentiation. This can also
        be seen by taking the definition
        \f[
            (\delta\Theta)(h)\Delta
                := \frac{d}{d\varepsilon}\Big|_{\varepsilon=0}
                    \Theta(h+\varepsilon\Delta)
        \f]
        and separating the terms based on the derivative order of
        \f$\Delta\f$. The result will be of the form
        \f[
            (\delta\Theta)(h)\Delta =
            \partial_h\Theta \Delta
            + \partial_{h'}\Theta \Delta'
            + \partial_{h''}\Theta \Delta''.
        \f]
        These three terms are computed here using
        \f[
            \partial_f \Theta =
                \frac{A_f}{D^{3/2}}
                - \frac{3}{2} \frac{A D_f}{D^{5/2}}
                + \frac{B_f}{D^{1/2}}
                - \frac{1}{2} \frac{B D_f}{D^{3/2}}
                + \frac{C_f}{D}
                - \frac{C D_f}{D^2}
                - \partial_f \,\mathrm{tr} K,
        \f]
        where `f` is one of ``h, h', h''``.

        The terms `A`, `B`, `C`, and `D` are defined in [1], but here we
        repeat them for convenience:
        \f{eqnarray*}{
            A &:=& -s^i s^j \partial_i s_j - \frac{1}{2} s^i (\partial_i g^{kl}) s_k s_l \\
            B &:=& (\partial_i g^{ij}) s_j + g^{ij} \partial_i s_j + (\partial_i \ln\sqrt{g}) s^i \\
            C &:=& K^{ij} s_i s_j \\
            D &:=& s_i s^i.
        \f}

        @b References

        [1] Thornburg, Jonathan. "A fast apparent horizon finder for
            three-dimensional Cartesian grids in numerical relativity."
            Classical and quantum gravity 21.2 (2003): 743.
        """
        if hdiff == 0: # del_h H
            A, B, C, D = self.ABCD
            dhA, dhB, dhC, dhD, dhtrK = self.get_dh_ABCDtrK()
            return (
                - 3 * A * dhD / (2*D**2.5) - B * dhD / (2*D**1.5)
                - C/D**2 * dhD
                + dhC / D + dhB / np.sqrt(D) + dhA / D**1.5
                - dhtrK
            )
        if hdiff == 1: # del_h' H
            A, B, C, D = self.ABCD
            dhpA, dhpB, dhpC, dhpD = self.get_dhp_ABCD()
            return (
                - 3 * A * dhpD / (2*D**2.5) - B * dhpD / (2*D**1.5)
                - C/D**2 * dhpD
                + dhpC / D + dhpB / np.sqrt(D) + dhpA / D**1.5
            )
        if hdiff == 2: # del_h'' H
            D = self.ABCD[-1]
            dhppA, dhppB = self.get_dhpp_AB()
            return (D * dhppB + dhppA) / D**1.5
        raise NotImplementedError

    def get_dh_ABCDtrK(self):
        r"""Compute the derivative of A, B, C, D, tr(K) \wrt `h`.

        May raise `NotImplementedError` to indicate numerical differentiation
        should be done.

        Refer to the definition of `A,B,C,D` in the documentation of _diff().
        The terms computed here are:
        \f[
            \partial_h A = -2(\partial_h s^i) s^j \partial_i s_j
                - s^i s^j \partial_h \partial_i s_j
                - \frac{1}{2} (\partial_h s^i) (\partial_i g^{kl}) s_k s_l
                - \frac{1}{2} s^i (\partial_h \partial_i g^{kl}) s_k s_l
                - s^i (\partial_i g^{kl}) s_k \partial_h s_l
        \f]
        \f[
            \partial_h B =
                (\partial_h \partial_i g^{ij}) s_j
                + (\partial_i g^{ij}) \partial_h s_j
                + (\partial_h g^{ij}) \partial_i s_j
                + g^{ij} \partial_h \partial_i s_j
                + (\partial_h \partial_i \ln\sqrt{g}) s^i
                + (\partial_i \ln\sqrt{g}) \partial_h s^i
        \f]
        \f[
            \partial_h C =
                \big[(\partial_h g^{ik}) g^{jl} + g^{ik}(\partial_h g^{jl})\big]
                K_{kl} s_i s_j
                + g^{ik} g^{jl} (\partial_h K_{kl}) s_i s_j
                + 2 g^{ik} g^{jl} K_{kl} s_i \partial_h s_j
        \f]
        \f[
            \partial_h D =
                (\partial_h g^{ij}) s_i s_j + 2 g^{ij} s_i \partial_h s_j
        \f]
        \f[
            \partial_h \mathrm{tr}K =
                (\partial_h g^{ij}) K_{ij} + g^{ij} \partial_h K_{ij}
        \f]

        The individual terms are computed by simply applying the chain rule.
        We obtain for any quantity `f` which depends on the coordinates
        `x,y,z`:
        \f[
            \partial_h f = (\partial_i f) (\partial_h\gamma)^i,
        \f]
        where \f$\gamma\f$ is the curve along which the computation takes
        place.
        """
        dh_gamma = self.curve.h_diff(self.param)
        g_inv, dg_inv, dlnsqrtg = self.g.inv, self.dg_inv, self.dlnsqrtg
        dg = self.dg
        ddg = self.ddg
        ddg_inv = self.ddg_inv
        s, s_up, ds = self.s, self.s_up, self.ds
        dds = self.compute_dds()
        dhs = ds.dot(dh_gamma)
        dhg_inv = np.einsum('aij,a', dg_inv, dh_gamma)
        dhs_up = dhg_inv.dot(s) + g_inv.dot(dhs)
        dhdg_inv = np.einsum('aikl,a', ddg_inv, dh_gamma)
        dhds = dds.dot(dh_gamma)
        dhdlnsqrtg = (
            0.5 * np.einsum('icd,acd,a', dg_inv, dg, dh_gamma)
            + 0.5 * np.einsum('cd,iacd,a', g_inv, ddg, dh_gamma)
        )
        dhA = (
            - 2 * np.einsum('i,j,ij', dhs_up, s_up, ds)
            - np.einsum('i,j,ij', s_up, s_up, dhds)
            - 0.5 * np.einsum('i,ikl,k,l', dhs_up, dg_inv, s, s)
            - 0.5 * np.einsum('i,ikl,k,l', s_up, dhdg_inv, s, s)
            - np.einsum('i,ikl,k,l', s_up, dg_inv, s, dhs)
        )
        dhB = (
            np.einsum('iij,j', dhdg_inv, s)
            + np.einsum('iij,j', dg_inv, dhs)
            + dhg_inv.dot(ds).diagonal().sum()
            + g_inv.dot(dhds).diagonal().sum()
            + dhdlnsqrtg.dot(s_up)
            + dlnsqrtg.dot(dhs_up)
        )
        dhD = (
            np.einsum('ij,i,j', dhg_inv, s, s)
            + 2 * np.einsum('ij,i,j', g_inv, s, dhs)
        )
        if self.K is None:
            dhC = 0.0
            dhtrK = 0.0
        else:
            K = self.K
            dK = self.curve.extr_curvature(self.point, diff=1)
            dhK = np.einsum('aij,a', dK, dh_gamma)
            dhC = (
                np.einsum('ik,jl,kl,i,j', dhg_inv, g_inv, K, s, s)
                + np.einsum('ik,jl,kl,i,j', g_inv, dhg_inv, K, s, s)
                + np.einsum('ik,jl,kl,i,j', g_inv, g_inv, dhK, s, s)
                + 2 * np.einsum('ik,jl,kl,i,j', g_inv, g_inv, K, s, dhs)
            )
            dhtrK = (
                np.einsum('ij,ij', dhg_inv, K)
                + np.einsum('ij,ij', g_inv, dhK)
            )
        return dhA, dhB, dhC, dhD, dhtrK

    def get_dhp_ABCD(self):
        r"""Compute the derivative of A, B, C, D \wrt `h'`.

        May raise `NotImplementedError` to indicate numerical differentiation
        should be done.

        This implementation is correct iff
        \f{eqnarray*}{
            \partial_{h'} s_i &=& - X_i\\
            \partial_{h'} \partial_i s_j &=& - X_{ij},
        \f}
        where \f$X_i := \partial_i \lambda\f$ and
        \f$X_{ij} := \partial_i \partial_j \lambda\f$.

        The terms computed here then become (refer to _diff()):
        \f{eqnarray*}{
            \partial_{h'} A &=&
                2 X^i s^j \partial_i s_j + s^i s^j X_{ij}
                + \frac{1}{2} (\partial_i g^{kl}) (X^i s_k s_l + 2 s^i X_k s_l)
            \\
            \partial_{h'} B &=&
                -(\partial_i g^{ij}) X_j - g^{ij} X_{ij} - (\partial_i\ln\sqrt{g}) X^i
            \\
            \partial_{h'} C &=& -2 K_{ij} X^i s^j
            \\
            \partial_{h'} D &=& -2 X_i s^i
        \f}

        This method is agnostic as to how the surfaces are represented as long
        as the quantities \f$s_i\f$, \f$\partial_i s_j\f$, \f$X_i\f$, and
        \f$X_{ij}\f$ are available.
        """
        g_inv, dg_inv, dlnsqrtg = self.g.inv, self.dg_inv, self.dlnsqrtg
        s, s_up, ds = self.s, self.s_up, self.ds
        X, X_up, Y = self.X, self.X_up, self.Y
        dhpA = (
            2 * ds.dot(X_up).dot(s_up)
            + Y.dot(s_up).dot(s_up)
            + 0.5 * dg_inv.dot(s).dot(s).dot(X_up)
            + dg_inv.dot(X).dot(s).dot(s_up)
        )
        dhpB = (
            - dg_inv.dot(X).diagonal().sum()
            - g_inv.dot(Y).diagonal().sum()
            - dlnsqrtg.dot(X_up)
        )
        if self.K is None:
            dhpC = 0.0
        else:
            dhpC = - 2 * self.K.dot(X_up).dot(s_up)
        dhpD = - 2 * X.dot(s_up)
        return dhpA, dhpB, dhpC, dhpD

    def get_dhpp_AB(self):
        r"""Compute the derivative of A and B \wrt ``h''``.

        May raise `NotImplementedError` to indicate numerical differentiation
        should be done.

        This implementation is correct iff
        \f{eqnarray*}{
            \partial_{h''} s_i &=& 0\\
            \partial_{h''} \partial_i s_j &=& - X_i X_j.
        \f}

        We compute here (see also _diff()):
        \f{eqnarray*}{
            \partial_{h''} A &=& s^i s^j X_i X_j \\
            \partial_{h''} B &=& -X^i X_i \\
            \partial_{h''} C &=& \partial_{h''} D = 0
        \f}

        This method is agnostic as to how the surfaces are represented as long
        as the quantities \f$s_i\f$, \f$\partial_i s_j\f$, \f$X_i\f$, and
        \f$X_{ij}\f$ are available.
        """
        X, X_up = self.X, self.X_up
        s_up = self.s_up
        dhppA = np.outer(X, X).dot(s_up).dot(s_up)
        dhppB = - X_up.dot(X)
        return dhppA, dhppB

    @abstractmethod
    def _compute_s_ds_X_Y(self):
        r"""Compute the terms we need to compute the expansion.

        Subclasses need to interpret the horizon function and compute the
        covariant normal (not normalized), its derivatives, and the parameter
        first (`X = del_i lambda`) and second (`Y = del_i del_j lambda`)
        derivatives.
        """
        pass

    def _compute_dds_Z(self):
        r"""Compute second derivatives of the normal and third ones of lambda.

        This computes \f$\partial_i\partial_j s_k\f$ and
        \f$Z := X_{ijk} = \partial_i\partial_j\partial_k \lambda\f$.

        @return Two elements, the first containing the derivatives of the
            non-normalized covariant normal `s` and the second those of the
            parameter \f$\lambda\f$.
        """
        raise NotImplementedError

    def _compute_d2_Y(self):
        r"""Compute second derivatives of xi and lambda \wrt x,y,z."""
        raise NotImplementedError

    def _compute_d3_Z(self):
        r"""Compute third derivatives of xi and lambda \wrt x,y,z."""
        raise NotImplementedError

    def ricci_scalar(self):
        r"""Compute the Ricci scalar of the surface represented by the curve.

        The Ricci scalar of a 2-surface is defined as (see e.g. [1])
        \f$R = q^{AB}R_{AB}\f$, where `q` is the induced metric
        \f$q_{ab} = g_{ab} - \nu_a \nu_b\f$, \f$R_{AB}\f$ is the Ricci tensor
        \f$R_{AB} = R^C_{\ A\,CB}\f$ and \f$\nu\f$ the covariant outward unit
        normal of the surface.
        Here, \f$R^A_{\ B\,CD}\f$ is the Riemann tensor.
        Note that `A,B` run over the coordinates \f$(\lambda,\varphi)\f$ on
        the surface and `a,b` over `x,y,z`.
        See induced_metric() for a bit more details on the induced metric `q`
        and the coordinate transformation to get the components \f$q_{AB}\f$
        we need here.

        It is convenient to compute the Ricci scalar from the purely covariant
        Riemann tensor \f$R_{AB\,CD} = q_{AE}R^E_{\ B\,CD}\f$ as this is
        antisymmetric in the first and last two index pairs, i.e. it has only
        one independent component \f$R_{\lambda\varphi\,\lambda\varphi}\f$ in
        two dimensions.
        A short calculation reveals
        \f[
            R = q^{AB}R_{AB}
                = 2 R_{\lambda\varphi\,\lambda\varphi}
                  (q^{\lambda\lambda}q^{\varphi\varphi} - (q^{\lambda\varphi})^2).
        \f]

        @b References

        [1] Straumann, Norbert. General relativity. Springer Science &
            Business Media, 2004.
        """
        R_0101 = self.covariant_riemann()
        q_inv = self.induced_metric(inverse=True)
        return 2 * R_0101 * (q_inv[0,0]*q_inv[1,1] - q_inv[0,1]**2)

    def induced_metric(self, diff=0, inverse=False):
        r"""Compute the induced metric on the surface.

        This method can compute the components of the induced metric in
        \f$(\lambda,\varphi)\f$ coordinates as well as the components of the
        inverse (i.e. indices upstairs) and derivatives of these components.

        Since this class assumes axisymmetry throughout, this method requires
        (without loss of generality) that the point at which the metric is to
        be returned is located at `phi=0`, i.e. `y=0` and `x>0`.

        @param diff
            Derivative order to compute. Default is `0`.
        @param inverse
            Whether to return the (derivatives of the) inverse of the induced
            metric. Default is `False`.

        @return NumPy array with ``2+diff`` axes, such that the indices
            ``[A1,A2,...,B,C]`` correspond to
            \f$\partial_{A_1}\partial_{A_2}\ldots q_{BC}\f$ for
            `inverse==False` and with upstairs indices for `invers==True`.

        @b Notes

        As in expcurve.ExpansionCurve.area(), we consider the most general
        axisymmetric 3-metric, i.e. a Brill wave (cf. [1]), which we write as
        \f[
            g = a(\rho,z)\ (d\rho^2 + dz^2) + \rho^2 b(\rho,z)\ d\varphi^2,
        \f]
        where \f$a, b > 0\f$. We know that our 3-metric will be of this form
        if it is to be axisymmetric.

        Taking the usual transforms \f$\rho = \sqrt{x^2+y^2}\f$ and
        \f$\rho\cos\varphi = x\f$, we can relate the components to those in
        `x`, `y`, `z` coordinates in which the metric is given. We get
        \f[
            (g_{ab}) =
            \left(\begin{array}{@{}ccc@{}}
                a c^2 + b s^2 & a c s - b c s & 0 \\
                a c s - b c s & b c^2 + a s^2 & 0 \\
                0 & 0 & a
            \end{array}\right),
        \f]
        where \f$c := \cos\varphi\f$ and \f$s := \sin\varphi\f$.

        The components of the induced metric `q` on the surface are then given
        in any coordinates by (compare e.g. equation (2) in [2])
        \f[
            q_{ab} = g_{ab} - \nu_a \nu_b.
        \f]
        A straightforward way to compute these components is to express `g`
        and the covariant unit normal w.r.t. the coordinates
        \f$(\lambda,\varphi)\f$ on the surface. To this end, note that the
        surface is parameterized in the above cylindrical coordinates via
        \f$\rho = \rho(\lambda) = (\gamma(\lambda))_x\f$ and
        \f$z = z(\lambda) = (\gamma(\lambda))_z\f$. Hence we have
        \f{eqnarray*}{
            dx &=& \rho' \cos\varphi\ d\lambda - \rho \sin\varphi\ d\varphi \\
            dy &=& \rho' \sin\varphi\ d\lambda + \rho \cos\varphi\ d\varphi \\
            dz &=& z'\ d\lambda
        \f}
        and therefore
        \f{eqnarray*}{
            \underline{\nu} &:=& g(\nu,\,\cdot\,) = \nu_a\ dx^a
                = \big[\rho'\,(\nu_x c + \nu_y s) + z' \nu_z\big]\ d\lambda
                  + \rho\,(-\nu_x s + \nu_y c)\ d\varphi
              \\
                &=:& \nu_\lambda\ d\lambda + \nu_\varphi\ d\varphi,
        \f}
        leading finally to
        \f{eqnarray*}{
            q &=& (a\,(\rho'^2+z'^2) + \nu_\lambda^2)\ d\lambda^2
                + (b\,\rho^2 + \nu_\varphi^2)\ d\varphi^2
                + \nu_\lambda \nu_\varphi (d\lambda \otimes d\varphi
                                           + d\varphi \otimes d\lambda)
            \\
                &=& q_{\lambda\lambda}\ d\lambda^2
                    + q_{\varphi\varphi}\ d\varphi^2
                    + q_{\lambda\varphi}\ (d\lambda \otimes d\varphi
                                           + d\varphi \otimes d\lambda).
        \f}
        Note that for \f$\varphi=0\f$, we have \f$\nu_y = 0\f$ due to the
        axisymmetry making the induced metric become diagonal.

        The first derivatives are computed via (all evaluated at `phi==0`)
        \f{eqnarray*}{
            \partial_C q_{\lambda\lambda}
                &=& a_{,C}\,(\rho'^2+z'^2)
                    + 2a\,\delta_{C\lambda}(\rho'\rho'' + z'z'')
                    + 2 \nu_{\lambda,C} \nu_\lambda
            \\
            \partial_C q_{\varphi\varphi}
                &=& b_{,C} \rho^2 + 2b\,\delta_{C\lambda} \rho \rho'
            \\
            \partial_C q_{\lambda\varphi}
                &=& 0,
        \f}
        where \f$f_{,A} := \partial_A f\f$ and \f$\delta_{AB}\f$ is the
        Kronecker delta.
        The second derivatives are
        \f{eqnarray*}{
            \partial_D\partial_C q_{\lambda\lambda}
                &=& a_{,CD}(\rho'^2+z'^2)
                    + 2\delta_{D\lambda}a_{,C}(\rho'\rho''+z'z'')
                \\&&
                    + 2\delta_{C\lambda}\big[
                        a_{,D}(\rho'\rho''+z'z'')
                        + \delta_{D\lambda} a\,(
                            \rho''^2 + \rho'\rho''' + z''^2 + z'z'''
                        )
                    \big]
                \\&&
                    + 2\nu_{\lambda,CD}\nu_\lambda
                    + 2\nu_{\lambda,C} \nu_{\lambda,D}
            \\
            \partial_D\partial_C q_{\varphi\varphi}
                &=& b_{,CD}\rho^2 + 2 \delta_{D\lambda} b_{,C}\rho\rho'
                    + 2 \delta_{C\lambda} \big[
                        b_{,D}\rho\rho'
                        + \delta_{D\lambda}b\,(\rho'^2 + \rho\rho'')
                    \big]
                \\&&
                    + 2\nu_{\varphi,CD}\nu_\varphi
                    + 2\nu_{\varphi,C} \nu_{\varphi,D}
            \\
            \partial_D\partial_C q_{\lambda\varphi}
                &=& \nu_{\lambda,CD}\nu_\varphi
                    + \nu_{\lambda,C} \nu_{\varphi,D}
                    + \nu_{\varphi,C} \nu_{\lambda,D}
                    + \nu_{\varphi,CD}\nu_\lambda.
        \f}

        @b References

        [1] Brill, Dieter R. "On the positive definite mass of the
            Bondi-Weber-Wheeler time-symmetric gravitational waves." Annals of
            Physics 7.4 (1959): 466-483.

        [2] Gundlach, Carsten. "Pseudospectral apparent horizon finders: An
            efficient new algorithm." Physical Review D 57.2 (1998): 863.
        """
        return self._induced_metric(diff, bool(inverse))

    @cache_method_results()
    def _induced_metric(self, diff, inverse):
        r"""Implements induced_metric()."""
        if inverse:
            q = self.induced_metric(diff=0)
            if diff == 0:
                return np.array([[1/q[0,0], 0.0], [0.0, 1/q[1,1]]])
            dq = self.induced_metric(diff=1)
            if diff == 1:
                dq_inv = inverse_2x2_matrix_derivative(q, dq, diff=1)
                return dq_inv
            ddq = self.induced_metric(diff=2)
            if diff == 2:
                ddq_inv = inverse_2x2_matrix_derivative(q, dq, ddq, diff=2)
                return ddq_inv
            raise NotImplementedError
        a, b = self.get_a_b_from_metric(diff=0)
        rho = self.point[0]
        drho, dz = self.curve.diff(self.param, diff=1)
        nl, _ = self.covariant_normal(coords='la,ph')
        q11 = a * (drho**2+dz**2) + nl**2
        q22 = b * rho**2
        q = np.array([[q11, 0.0],
                      [0.0, q22]])
        if diff == 0:
            return q
        # partial_i nu_lambda,  (i=lambda,phi)
        dnl, dnp = self.covariant_normal(coords='la,ph', deriv_wrt='la,ph', diff=1)
        ddrho, ddz = self.curve.diff(self.param, diff=2)
        da, db = self.get_a_b_from_metric(deriv_wrt='la,ph', diff=1)
        # dq11 = [partial_lambda q_{la,la}, partial_phi q_{la,la}]
        # (la == lambda)
        dq11 = (
            2 * dnl * nl
            + da * (drho**2+dz**2)
            + 2*a * np.array([(drho*ddrho + dz*ddz), 0.0])
        )
        dq22 = (
            db * rho**2 + 2*b * np.array([rho*drho, 0.0])
        )
        # we want indices to mean [A,B,C] <=> \partial_A q_BC
        dq = np.array([[[dq11[i], 0.0], [0.0, dq22[i]]] for i in range(2)])
        if diff == 1:
            return dq
        dda, ddb = self.get_a_b_from_metric(deriv_wrt='la,ph', diff=2)
        ddnl, ddnp = self.covariant_normal(coords='la,ph', deriv_wrt='la,ph', diff=2)
        ddn = np.array([ddnl, ddnp])
        n = np.array([nl, 0.0])
        dn = np.array([dnl, dnp])
        d3rho, d3z = self.curve.diff(self.param, diff=3)
        def _ddq(D,C,A,B):
            ddq = (
                ddn[A,C,D] * n[B] + dn[A,C] * dn[B,D] + dn[A,D] * dn[B,C] + n[A] * ddn[B,C,D]
            )
            if A == B == 0:
                ddq += dda[C,D] * (drho**2 + dz**2)
                if D == 0:
                    ddq += 2 * da[C] * (drho*ddrho + dz*ddz)
                if C == 0:
                    ddq += 2 * da[D] * (drho*ddrho + dz*ddz)
                    if D == 0:
                        ddq += 2 * a * (ddrho**2 + drho*d3rho + ddz**2 + dz*d3z)
            if A == B == 1:
                ddq += ddb[C,D] * rho**2
                if D == 0:
                    ddq += 2 * db[C] * rho * drho
                if C == 0:
                    ddq += 2 * db[D] * rho * drho
                    if D == 0:
                        ddq += 2 * b * (drho**2 + rho*ddrho)
            return ddq
        if diff == 2:
            # indices should be [D,C,A,B] <=> \partial_D \partial_C q_AB
            ra = range(2)
            return np.array([[[[_ddq(D,C,A,B) for B in ra] for A in ra]
                              for C in ra] for D in ra])
        raise NotImplementedError

    def get_a_b_from_metric(self, deriv_wrt=None, diff=0):
        r"""Compute values of the functions a and b for the current metric.

        See the docstring of induced_metric() for the definition of
        axisymmetric metrics in terms of the functions `a` and `b`. As can
        also be seen in the docstring, we get `a` as the `g_zz` component and
        `b` as the `g_yy` component for `phi=0` (which we assume here
        throughout).

        @param deriv_wrt
            String indicating with respect to which coordinates any
            derivatives should be returned. Ignored if `diff==0`. Valid values
            are ``'x,y,z'`` and ``'la,ph'``.
        @param diff
            Derivative order to compute of `a` and `b`.

        @return Either two floats (if `diff==0`), one for `a` and `b`,
            respectively, or two NumPy arrays with `diff` axes corresponding
            to the partial derivatives.

        @b Notes

        To obtain the derivatives of `a` w.r.t. `x,y,z`, note that since
        `a == g_zz` even without setting `phi = 0`, the partial derivatives of
        the metric can directly be taken as those of `a`. For `b`, this is
        also the case, i.e.
        \f[
            \partial_i\partial_j b
                \stackrel{\varphi=0}{=} \partial_i\partial_j g_{yy},
        \f]
        even though one has to be careful to verify the above equation since
        \f$g_{yy} = b\cos^2\varphi + a\sin^2\varphi\f$.
        For example, we get
        \f[
            \partial_x \partial_y g_{yy}
                = b_{,xy} + 2 \varphi_{,x}\varphi_{,y}(a-b)
                = b_{,xy}.
        \f]
        The last equation is due to \f$\rho\cos\varphi = x\f$ and hence
        \f[
            -\varphi_{,x}\sin\varphi = \partial_x\cos\varphi
                = \frac{\rho^2-x^2}{\rho^3}
                = \frac{y^2}{\rho^3}
            \quad\Rightarrow\quad
            \varphi_{,x} = -\frac{y}{\rho^2}
                \stackrel{y=0}{=} 0.
        \f]

        For the derivatives w.r.t. \f$\lambda,\varphi\f$, we use the chain
        rule and the concrete transformation rules
        \f[
            x = \rho(\lambda)\cos\varphi,\quad
            y = \rho(\lambda)\sin\varphi,\quad
            z = z(\lambda),
        \f]
        to get (at \f$\varphi = 0\f$)
        \f[
            \begin{array}{@{}r@{\;}l@{\qquad}r@{\;}l@{\qquad}r@{\;}l@{\qquad}r@{\;}l@{\qquad}r@{\;}l@{}}
            x_{,\lambda} &= \rho' & x_{,\varphi} & = 0    & x_{,\lambda\lambda} & = \rho'' & x_{,\lambda\varphi} & = 0     & x_{,\varphi\varphi} & = -\rho \\
            y_{,\lambda} &= 0     & y_{,\varphi} & = \rho & y_{,\lambda\lambda} & = 0      & y_{,\lambda\varphi} & = \rho' & y_{,\varphi\varphi} & = 0 \\
            z_{,\lambda} &= z'    & z_{,\varphi} & = 0    & z_{,\lambda\lambda} & = z''    & z_{,\lambda\varphi} & = 0     & z_{,\varphi\varphi} & = 0
            \end{array}
        \f]
        and therefore, again at \f$\varphi = 0\f$,
        \f[
            \begin{array}{@{}r@{\;}l@{\qquad}r@{\;}l@{}}
            a_{,\lambda} &= \rho' a_{,x} + z' a_{,z} &
            a_{,\lambda\lambda} &= \rho'' a_{,x} + z'' a_{,z} + \rho'^2 a_{,xx} + z'^2 a_{,zz} + 2 \rho' z' a_{,xz}
            \\
            a_{,\varphi} &= 0 &
            a_{,\varphi\varphi} &= 0 \qquad\qquad
            a_{,\lambda\varphi} = 0.
            \end{array}
        \f]
        The same formulas hold for `b`.
        For some of these identities, we need to express `y`-derivatives of
        `a` in terms of \f$\rho\f$-derivatives using the rules explained at
        the end of the docstring of
        refparamcurve._RefParamExpansionCalc.coord_derivs().
        """
        if diff == 0:
            a, b = self.g.mat[2,2], self.g.mat[1,1]
            return a, b
        if deriv_wrt not in ('x,y,z', 'la,ph'):
            raise ValueError("Unknown coordinates: %s" % deriv_wrt)
        return self._get_a_b_from_metric(deriv_wrt, diff)

    @cache_method_results()
    def _get_a_b_from_metric(self, deriv_wrt, diff):
        r"""Cached version of get_a_b_from_metric() for computing derivatives."""
        if deriv_wrt == 'x,y,z':
            return self._a_b_wrt_xyz(diff=diff)
        return self._a_b_wrt_laph(diff=diff)

    def _a_b_wrt_xyz(self, diff):
        r"""Compute derivatives of a, b \wrt x,y,z (see get_a_b_from_metric())."""
        if diff == 1:
            # dg[i,j,k] = \partial_i g_jk
            dg = np.asarray(self.metric.diff(self.point, diff=1))
            da = dg[:,2,2]
            db = dg[:,1,1]
            da[1] = db[1] = 0.0
            return da, db
        if diff == 2:
            # ddg[i,j,k,l] = \partial_i \partial_j g_kl
            ddg = np.asarray(self.metric.diff(self.point, diff=2))
            dda = ddg[:,:,2,2]
            ddb = ddg[:,:,1,1]
            return dda, ddb
        raise NotImplementedError

    def _a_b_wrt_laph(self, diff):
        r"""Compute derivatives of a, b \wrt lambda,phi (see get_a_b_from_metric())."""
        dab = np.asarray(self.get_a_b_from_metric('x,y,z', diff=1))
        drho, dz = self.curve.diff(self.param, diff=1)
        if diff == 1:
            da_l, db_l = [
                drho * dab[i,0] + dz * dab[i,2]
                for i in range(2)
            ]
            return np.array([[da_l, 0.0], [db_l, 0.0]])
        ddab = np.asarray(self.get_a_b_from_metric('x,y,z', diff=2))
        ddrho, ddz = self.curve.diff(self.param, diff=2)
        if diff == 2:
            ddab_ll = [
                ddrho * dab[i,0] + ddz * dab[i,2]
                + drho**2 * ddab[i,0,0] + dz**2 * ddab[i,2,2]
                + 2 * drho * dz * ddab[i,0,2]
                for i in range(2)
            ]
            return np.array([[[ddab_ll[i], 0.0], [0.0, 0.0]] for i in range(2)])
        raise NotImplementedError

    def covariant_normal(self, coords='x,y,z', deriv_wrt=None, diff=0):
        r"""Compute (derivatives of) the normalized covariant normal.

        This method can compute the components of the normalized covariant
        normal in either `x,y,z` or \f$\lambda,\varphi\f$ coordinates. It can
        also compute derivatives of these components w.r.t. either of these
        coordinates in most cases (the exception being the `x,y,z` derivatives
        of the \f$\lambda,\varphi\f$ components).

        @param coords
            String indicating in which coordinates to return the components.
            Valid values are ``'x,y,z'`` (default) and ``'la,ph'``.
        @param deriv_wrt
            String indicating with respect to which coordinates any
            derivatives should be returned. Ignored if `diff==0`. Valid values
            are ``'x,y,z'`` and ``'la,ph'``. By default, derivatives are
            returned w.r.t. the requested `coords`.
        @param diff
            Derivative order to compute. Default is `0`.

        @return For ``coords=='x,y,z'``, returns three elements corresponding
            to (derivatives of) the `x, y, z` components \f$\nu_i\f$. For
            ``coords=='la,ph'``, two elements are returned which correspond to
            the two components \f$\nu_A\f$, \f$A=\lambda,\varphi\f$. In case
            of derivatives, each element will be a NumPy array with `diff`
            axes corresponding to the partial derivatives.

        @b Notes

        Given the non-normalized components \f$s_i\f$ of the covariant outward
        pointing normal on the surface, in the simplest case of x,y,z
        components without differentiation, the result is just
        \f[
            \nu_i = \frac{s_i}{\sqrt{D}}, \qquad D := g^{kl} s_k s_l.
        \f]
        From this formula, we get the x,y,z derivatives
        \f[
            \partial_i\nu_j =
                \frac{\partial_i s_j}{\sqrt{D}}
                - \frac{s_j}{2 D^{3/2}} D_i
        \f]
        and
        \f[
            \partial_i\partial_j\nu_k =
                \frac{\partial_i \partial_j s_k}{\sqrt{D}}
                - \frac{1}{2 D^{3/2}}
                    \Big(
                        (\partial_j s_k) D_i
                        + (\partial_i s_k) D_j
                        + s_k D_{ij}
                    \Big)
                + \frac{3}{4} \frac{s_k}{D^{5/2}} D_i D_j,
        \f]
        where
        \f{eqnarray*}{
            D_i &:=& \partial_i (g^{kl} s_k s_l)
                = (\partial_i g^{kl}) s_k s_l + 2 g^{kl} s_k\,\partial_i s_l
            \\
            D_{ij} &:=& \partial_j D_i \\
                &=&
                (\partial_i \partial_j g^{kl}) s_k s_l
                + 2 (\partial_i g^{kl}) s_k\,\partial_j s_l
                + 2 (\partial_j g^{kl}) s_k\,\partial_i s_l
                \\&&
                + 2 g^{kl} \big(
                    (\partial_j s_k)(\partial_i s_l)
                    + s_k \partial_i \partial_j s_l
                \big).
        \f}

        The derivatives of the x,y,z components w.r.t. \f$\lambda,\varphi\f$
        are obtained using the chain rule, i.e. the results at \f$\varphi=0\f$
        are
        \f{eqnarray*}{
            \partial_\lambda \nu_i &=& \rho' \nu_{i,x} + z' \nu_{i,z}
            \\
            \partial_\varphi \nu_i &=& \rho \nu_{i,y}
                \quad (= 0\ \mbox{if}\ i \neq y)
            \\
            \partial_\lambda^2 \nu_i &=&
                \rho'' \nu_{i,x} + \rho'^2 \nu_{i,xx}
                + z'' \nu_{i,z} + z'^2 \nu_{i,zz}
                + 2\rho' z' \nu_{i,xz}
            \\
            \partial_\lambda\partial_\varphi \nu_i &=&
                \rho' \nu_{i,y} + \rho\rho' \nu_{i,xy} + \rho z' \nu_{i,yz}
            \\
            \partial_\varphi^2 \nu_i &=&
                -\rho \nu_{i,x} + \rho^2 \nu_{i,yy}.
        \f}
        Here, \f$\nu_{i,x} := \partial_x\nu_i\f$, etc., and we have used the
        coordinate derivatives like \f$y_{,\varphi} = \rho\f$ derived in
        get_a_b_from_metric().

        Finally, the derivatives of the \f$\lambda,\varphi\f$ components
        w.r.t. \f$\lambda,\varphi\f$ can be spelled out by, again, applying
        the chain rule and then setting \f$\varphi=0\f$ to yield
        \f{eqnarray*}{
            \partial_\lambda \nu_\lambda
                &=& \nu_x \rho'' + \nu_z z''
                    + \nu_{x,\lambda} \rho'
                    + \nu_{z,\lambda} z'
            \\
            \partial_\varphi \nu_\lambda &=& \nu_{x,\varphi} \rho'
            \\
            \partial_\lambda \nu_\varphi &=& 0
            \\
            \partial_\varphi \nu_\varphi
                &=& - \nu_x \rho + \nu_{y,\varphi} \rho
            \\
            \partial_\lambda^2 \nu_\lambda
                &=& \nu_x \rho''' + 2 \nu_{x,\lambda} \rho''
                    + \nu_{x,\lambda\lambda} \rho'
                    + \nu_z z'''
                    + 2 \nu_{z,\lambda} z''
                    + \nu_{z,\lambda\lambda} z'
            \\
            \partial_\lambda\partial_\varphi \nu_\lambda
                &=& \rho' \nu_{x,\lambda\varphi} + z' \nu_{z,\lambda,\varphi}
            \\
            \partial_\varphi^2 \nu_\lambda
                &=& \rho'\, (\nu_{x,\varphi\varphi} + \nu_x + 2 \nu_{y,\varphi})
                    + z' \nu_{z,\varphi\varphi}
            \\
            \partial_\lambda^2 \nu_\varphi &=& \nu_{y,\lambda\lambda} \rho
            \\
            \partial_\lambda\partial_\varphi \nu_\varphi
                &=& \rho\,(-\nu_x + \nu_{y,\varphi})
                    + \rho\,(-\nu_{x,\lambda} + \nu_{y,\lambda\varphi})
            \\
            \partial_\varphi^2 \nu_\varphi
                &=& \rho \nu_{y,\varphi\varphi}.
        \f}
        """
        if not diff or deriv_wrt is None:
            deriv_wrt = coords
        if coords not in ('x,y,z', 'la,ph'):
            raise ValueError("Unknown coordinates: %s" % coords)
        if deriv_wrt not in ('x,y,z', 'la,ph'):
            raise ValueError("Unknown coordinates: %s" % deriv_wrt)
        return self._covariant_normal(coords, deriv_wrt, diff)

    @cache_method_results()
    def _covariant_normal(self, coords, deriv_wrt, diff):
        r"""Cached implementation of covariant_normal()."""
        if coords == 'x,y,z' and deriv_wrt == 'x,y,z':
            return self._nu_xyz_wrt_xyz(diff=diff)
        if coords == 'x,y,z' and deriv_wrt == 'la,ph':
            return self._nu_xyz_wrt_laph(diff=diff)
        if coords == 'la,ph' and deriv_wrt == 'la,ph':
            return self._nu_laph_wrt_laph(diff=diff)
        raise NotImplementedError

    def _nu_laph_wrt_laph(self, diff=0):
        r"""Compute lambda,phi derivatives of lambda,phi normal components.

        See covariant_normal() for the derivation of the used formulas.
        """
        rho = self.point[0]
        nx, _, nz = self.covariant_normal()
        drho, dz = self.curve.diff(self.param, diff=1)
        if diff == 0:
            nl = drho * nx + dz * nz # nu_lambda
            return nl, 0.0
        ddrho, ddz = self.curve.diff(self.param, diff=2)
        # derivatives of nu_x, nu_z w.r.t. lambda and phi
        dnx_lp, dny_lp, dnz_lp = self.covariant_normal(
            coords='x,y,z', deriv_wrt='la,ph', diff=1
        )
        dnl_la = nx * ddrho + dnx_lp[0] * drho + nz * ddz + dnz_lp[0] * dz
        dnl_ph = dnx_lp[1] * drho
        dnp_la = 0.0
        dnp_ph = rho * (-nx + dny_lp[1])
        dnl = np.array([dnl_la, dnl_ph])
        dnp = np.array([dnp_la, dnp_ph])
        if diff == 1:
            return dnl, dnp
        d3rho, d3z = self.curve.diff(self.param, diff=3)
        ddnx_lp, ddny_lp, ddnz_lp = self.covariant_normal(
            coords='x,y,z', deriv_wrt='la,ph', diff=2
        )
        ddnl_ll = (
            nx * d3rho + 2*dnx_lp[0]*ddrho + ddnx_lp[0,0]*drho + d3z*nz
            + 2*ddz*dnz_lp[0] + dz*ddnz_lp[0,0]
        )
        ddnl_lp = drho * ddnx_lp[0,1] + dz * ddnz_lp[0,1]
        ddnl_pp = drho * (ddnx_lp[1,1] - nx + 2*dny_lp[1]) + dz*ddnz_lp[1,1]
        ddnp_ll = rho * ddny_lp[0,0]
        ddnp_lp = drho*(-nx+dny_lp[1]) + rho*(-dnx_lp[0]+ddny_lp[0,1])
        ddnp_pp = rho * ddny_lp[1,1]
        if diff == 2:
            ddnl = np.array([[ddnl_ll, ddnl_lp], [ddnl_lp, ddnl_pp]])
            ddnp = np.array([[ddnp_ll, ddnp_lp], [ddnp_lp, ddnp_pp]])
            return ddnl, ddnp
        raise NotImplementedError

    def _nu_xyz_wrt_laph(self, diff=0):
        r"""Compute lambda,phi derivatives of x,y,z normal components.

        See covariant_normal() for the derivation of the used formulas.
        """
        if diff == 0:
            return self._nu_xyz_wrt_xyz(diff=0)
        rho = self.point[0]
        drho, dz = self.curve.diff(self.param, diff=1)
        # e.g.: dn[0] = partial_i nu_x,  for i = x, y, z
        dn = self.covariant_normal(
            coords='x,y,z', deriv_wrt='x,y,z', diff=1
        )
        dnx_la, dny_la, dnz_la = [drho * dn[i][0] + dz * dn[i][2]
                                  for i in range(3)]
        dnx_ph, dny_ph, dnz_ph = [rho * dn[i][1] for i in range(3)]
        if diff == 1:
            dnx_lp = np.array([dnx_la, dnx_ph])
            dny_lp = np.array([dny_la, dny_ph])
            dnz_lp = np.array([dnz_la, dnz_ph])
            return dnx_lp, dny_lp, dnz_lp
        # e.g.: ddn[0] = partial_i partial_j nu_x,  for i = x, y, z
        # so: ddn[0][1,2] = partial_y partial_z nu_x
        ddn = self.covariant_normal(
            coords='x,y,z', deriv_wrt='x,y,z', diff=2
        )
        ddrho, ddz = self.curve.diff(self.param, diff=2)
        ddnx_ll, ddny_ll, ddnz_ll = [
            ddrho * dn[i][0] + drho**2 * ddn[i][0,0] + 2*drho*dz * ddn[i][0,2]
            + dz**2 * ddn[i][2,2] + ddz * dn[i][2]
            for i in range(3)
        ]
        ddnx_lp, ddny_lp, ddnz_lp = [
            drho * dn[i][1] + rho*drho * ddn[i][0,1] + rho*dz * ddn[i][1,2]
            for i in range(3)
        ]
        ddnx_pp, ddny_pp, ddnz_pp = [
            -rho * dn[i][0] + rho**2 * ddn[i][1,1]
            for i in range(3)
        ]
        if diff == 2:
            ddnx = np.array([[ddnx_ll, ddnx_lp],
                             [ddnx_lp, ddnx_pp]])
            ddny = np.array([[ddny_ll, ddny_lp],
                             [ddny_lp, ddny_pp]])
            ddnz = np.array([[ddnz_ll, ddnz_lp],
                             [ddnz_lp, ddnz_pp]])
            return ddnx, ddny, ddnz
        raise NotImplementedError

    def _nu_xyz_wrt_xyz(self, diff=0):
        r"""Compute x,y,z derivatives of x,y,z normal components.

        See covariant_normal() for the derivation of the used formulas.
        """
        s = self.s
        D = self.ABCD[3]
        if diff == 0:
            return s / np.sqrt(D)
        ds = self.ds
        dg_inv = self.dg_inv
        g_inv = self.g.inv
        if diff == 1:
            # note: X.dot(y) for a n-d X and 1-d y contracts/sums the *last*
            #       index of X with y, i.e. X.dot(y) = sum_l X_ijkl y^l.
            #       This means X.dot(y) has n-1 free indices left.
            # We now compute partial_i nu_j  (note the indices i and j).
            dnx, dny, dnz = [
                ds[:,j] / np.sqrt(D) - 0.5 * (
                    s[j]/D**1.5 * np.array(
                        [dg_inv[i].dot(s).dot(s) + 2*g_inv.dot(s).dot(ds[i,:])
                         for i in range(3)]
                    )
                )
                for j in range(3)
            ]
            return dnx, dny, dnz
        dds = self.compute_dds()
        Di = self.compute_Di()
        Dij = self.compute_Dij()
        if diff == 2:
            # We now compute partial_i partial_j nu_k.
            ddnx, ddny, ddnz = [
                dds[:,:,k] / np.sqrt(D)
                - 1/(2*D**1.5) * (
                    np.outer(ds[:,k], Di) + np.outer(Di, ds[:,k]) + s[k] * Dij
                )
                + 3./4. * s[k] / D**2.5 * np.outer(Di, Di)
                for k in range(3)
            ]
            return ddnx, ddny, ddnz
        raise NotImplementedError

    def compute_Di(self):
        r"""Compute the D_i terms for covariant_normal().

        See covariant_normal() for the derivation of the used formulas.
        """
        g_inv = self.g.inv
        dg_inv = self.dg_inv
        s = self.s
        ds = self.ds
        return dg_inv.dot(s).dot(s) + 2 * ds.dot(g_inv.dot(s))

    def compute_Dij(self):
        r"""Compute the D_ij terms for covariant_normal().

        See covariant_normal() for the derivation of the used formulas.
        """
        g_inv = self.g.inv
        dg_inv = self.dg_inv
        ddg_inv = np.asarray(self.metric.diff(self.point, diff=2, inverse=True))
        s = self.s
        ds = self.ds
        dds = self.compute_dds()
        return (
            ddg_inv.dot(s).dot(s)
            + 2 * dg_inv.dot(s).dot(ds)
            + 2 * dg_inv.dot(s).dot(ds).T
            + 2 * g_inv.dot(ds).T.dot(ds) + 2 * dds.dot(g_inv.dot(s))
        )

    @cache_method_results()
    def compute_dds(self):
        r"""Compute the second derivatives of the non-normalized normal."""
        return self._compute_dds_Z()[0]

    @cache_method_results()
    def compute_d2_Y(self):
        r"""Compute second derivatives of xi and lambda \wrt x,y,z."""
        return self._compute_d2_Y()

    @cache_method_results()
    def compute_d3_Z(self):
        r"""Compute third derivatives of xi and lambda \wrt x,y,z."""
        return self._compute_d3_Z()

    def covariant_riemann(self):
        r"""Compute the purely covariant Riemann tensor.

        This computes the only independent component
        \f[
            R_{\lambda\varphi\,\lambda\varphi}
                = q_{\lambda A} R^A_{\ \varphi\,\lambda\varphi}
        \f]
        of the covariant Riemann tensor.
        """
        q = self.induced_metric()
        R0_101, R1_101 = self.riemann()
        R_0101 = q[0,0] * R0_101 + q[0,1] * R1_101
        return R_0101

    def riemann(self):
        r"""Compute the components of the Riemann tensor on the surface.

        The Riemann tensor computed here is defined as
        \f[
            R^A_{\ B\,CD} =
                \partial_C \Gamma^A_{DB}
                - \partial_D \Gamma^A_{CB}
                + \Gamma^A_{CE} \Gamma^E_{DB}
                - \Gamma^A_{DE} \Gamma^E_{CB},
        \f]
        where \f$\Gamma^{A}_{BC}\f$ are the Christoffel symbols of the induced
        2-metric `q`.

        Due to the antisymmetry in the last two indices, only two components
        may potentially be nonzero, namely
        \f$R^\lambda_{\ \varphi\,\lambda_\varphi}\f$ and
        \f$R^\varphi{\ \varphi\,\lambda_\varphi}\f$. These two components are
        returned here.
        """
        G = self.christoffel()
        dG = self.christoffel_deriv()
        R0_101 = riemann_components(G, dG, 0, 1, 0, 1)
        R1_101 = riemann_components(G, dG, 1, 1, 0, 1)
        return R0_101, R1_101

    def christoffel(self):
        r"""Compute the Christoffel symbols of the induced metric on the surface.

        @return NumPy array with indices `[A,B,C]` corresponding to
            \f$\Gamma^A_{BC}\f$.
        """
        q_inv = self.induced_metric(inverse=True)
        dq = self.induced_metric(diff=1)
        return christoffel_symbols(q_inv, dq)

    def christoffel_deriv(self):
        r"""Compute the derivatives of the Christoffel symbols on the surface.

        @return NumPy array with indices `[A,B,C,D]` corresponding to
            \f$\partial_A\Gamma^B_{CD}\f$.
        """
        q_inv = self.induced_metric(inverse=True)
        dq_inv = self.induced_metric(inverse=True, diff=1)
        dq = self.induced_metric(diff=1)
        ddq = self.induced_metric(diff=2)
        return christoffel_deriv(q_inv, dq_inv, dq, ddq)

    def extrinsic_curvature(self, trace=False, square=False):
        r"""Compute the extrinsic curvature.

        @param trace
            If `True`, returns the trace of the extrinsic curvature. Default
            is `False`. May not be used together with `square`.
        @param square
            If `True`, returns the square \f$k_{AB} k^{AB}\f$. Default is
            `False`. May not be used together with `trace`.

        @return If ``trace=square=False``, a NumPy 2x2 array containing the
            components of `k_AB`. Otherwise, returns a float.

        @b Notes

        To get the components \f$k_{AB} = -\nabla_A \nu_B\f$, first note that
        `k` annihilates any components transverse to the surface \f$\sigma\f$
        (see e.g. [1]), i.e. for any point \f$p \in \sigma\f$
        \f[
            k(v_p, X_p) = 0
                \qquad \forall\,X_p\in T_p M,
        \f]
        where \f$v\f$ is any vector field normal to \f$\sigma\f$, for example
        the normal \f$\nu\f$ in the current slice \f$\Sigma\f$ or the future
        pointing normal `n` of the slice in spacetime.
        Hence, we will in the following restrict all objects to \f$\sigma\f$.
        For example,
        \f[
            dx^\mu\big|_\sigma = \frac{\partial x^\mu}{\partial u^A}\ du^A
                =: x^\mu_{,A}\ du^A,
        \f]
        where \f$u^A = \lambda,\varphi\f$. The \f$x^\mu_{,A}\f$ are derived
        and spelled out in the description of get_a_b_from_metric(). Note that
        \f$x^0_{,A} = 0\f$ since \f$x^0 = t\f$ does not depend on
        \f$\lambda\f$ or \f$\varphi\f$.
        Observing further that \f$\partial_A = x^a_{,A}\partial_a\f$, we get
        \f{eqnarray*}{
            \nabla_{\!\partial_A} \nu_B
                &=& \big[x^a_{,A} \nabla_a \underline\nu\big]_B
            \\  &=& x^a_{,A} \big[
                    (\partial_a\nu_\beta - \Gamma^\alpha_{a\beta}\nu_\alpha)\ dx^\beta
                \big]_B
            \\  &=& x^a_{,A} \big[
                    (\partial_a\nu_b - \Gamma^c_{ab}\nu_c)\ dx^b
                \big]_B
            \\  &=& x^a_{,A} (\partial_a\nu_b - \Gamma^c_{ab}\nu_c) x^b_{,B}
            \\  &=& x^a_{,A} x^b_{,B}
                    (\partial_a\nu_b - {}^{(3)}\Gamma^c_{ab}\nu_c).
        \f}
        The third equality is due to \f$dx^0\big|_\sigma = 0\f$ and
        \f$\nu_0 = 0\f$. The reason we can take the Christoffel symbols of the
        3-metric in the slice is that, by their definition and using
        \f$g_{ab} = {}^{(4)}g_{ab} + n_a n_b\f$,
        \f{eqnarray*}{
            (\Gamma^k_{ab} - {}^{(3)}\Gamma^k_{ab}) \nu_k
                &=& \frac{1}{2} (g^{kc} - n^k n^c) \big[
                        - \partial_c (g_{ab} - n_a n_b)
                        + \partial_a (g_{bc} - n_b n_c)
                        + \partial_b (g_{ca} - n_c n_a)
                    \big] \nu_k
                \\&&
                    - \frac{1}{2} g^{kc} \big[
                        - \partial_c g_{ab}
                        + \partial_a g_{bc}
                        + \partial_b g_{ca}
                    \big] \nu_k
            \\  &=& \frac{1}{2} \nu^c \big[
                        \partial_c (n_a n_b)
                        - \partial_a (n_b n_c)
                        - \partial_b (n_c n_a)
                    \big]
            \\  &=& 0.
        \f}
        The first equality is due to \f$n^k \nu_k = 0\f$ (`n` is orthogonal to
        any horizontal vectors, i.e. in \f$T_p\Sigma\f$) and the last
        equation due to \f$n_\mu = 0\f$ for \f$\mu \neq 0\f$.

        @b References

        [1] D. Giulini. "Dynamical and Hamiltonian Formulation of General
            Relativity". In: Springer handbook of spacetime. Ed. by A.
            Ashtekar and V. Petkov. Springer, 2014. Chap. 17.
        """
        if trace and square:
            raise ValueError("Arguments `trace` and `square` are mutually exclusive.")
        ra2 = range(2)
        ra3 = range(3)
        G3 = self.metric.christoffel(self.point)
        nu = self.covariant_normal(coords='x,y,z', diff=0)
        dn = self.covariant_normal(coords='x,y,z', deriv_wrt='x,y,z', diff=1)
        # currently: dn[i,j] = partial_j nu_i
        # we want:   dn[i,j] = partial_i nu_j
        dn = np.asarray(dn).T
        rho = self.point[0]
        drho, dz = self.curve.diff(self.param, diff=1)
        # dx[A,i] = partial_A x^i
        dx = np.array([
            [drho, 0.0, dz], # partial_lambda [x, y, z]
            [0.0, rho, 0.0], # partial_phi    [x, y, z]
        ])
        def _K(A, B):
            return - (
                fsum(dx[A,i]*dx[B,j] * dn[i,j]
                     for j in ra3 for i in ra3)
                - fsum(dx[A,i]*dx[B,j] * G3[k,i,j]*nu[k]
                       for k in ra3 for j in ra3 for i in ra3)
            )
        K_AB = np.array([[_K(A,B) for B in ra2] for A in ra2])
        if trace or square:
            q_inv = self.induced_metric(inverse=True)
            if trace:
                return q_inv.dot(K_AB).diagonal().sum()
            return np.einsum('ac,bd,ab,cd', q_inv, q_inv, K_AB, K_AB)
        return K_AB


class _FuncVariation(object):
    r"""Helper class to apply an offset to a specific derivative of a function.

    Given a function `f`, an offset `eps` is applied to the n'th derivative of
    the function. Here `n` is given by the `diff` parameter.

    This is used to compute the finite difference approximation of the
    derivative of the expansion w.r.t. \f$ h \f$, \f$ h' \f$, and \f$ h'' \f$.
    """
    def __init__(self, f, diff, eps=0):
        r"""Create a callable and modify one derivative order.

        Args:
            f: Callable that should also implement `f.diff()`, e.g. an
                evaluator of the motsfinder.exprs system.
            diff: Derivative order of `f` to modify. `0` means that `eps` will
                be added to any function value computed by `f` but not to
                derivatives. A value of ``n>0`` means that `f` and all its
                derivatives are returned "as is", except for the n'th
                derivative to which the value of `eps` will be added.
            eps: Value to add to the results of computing the `diff`'th
                derivative of `f`.
        """
        ## The function to wrap.
        self._f = f
        ## The derivative order of the function to add `eps` to.
        self._diff = diff
        ## The value to add to the specified derivative order.
        self.eps = eps

    def __call__(self, x):
        r"""Evaluate the function at a point.

        In case `diff==0`, the `eps` will be added.
        """
        val = self._f(x)
        if self._diff == 0:
            val += self.eps
        return val

    def diff(self, x, n=1):
        r"""Evaluate the n'the derivaative of the function at a point.

        In case `diff==n`, the `eps` will be added.
        """
        val = self._f.diff(x, n=n)
        if self._diff == n:
            val += self.eps
        return val
