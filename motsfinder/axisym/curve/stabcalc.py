r"""@package motsfinder.axisym.curve.stabcalc

Compute quantities needed for the stability operator.

This is used by the ExpansionCurve to compute the stability operator in order
to evaluate its spectrum.
"""

import math

import numpy as np


__all__ = [
    "StabilityCalc",
]


# It is customary to denote indices of tensors without spaces, e.g.:
#   T_{ijk}  =>  T[i,j,k]
# We disable the respective pylint warning for this file.
# pylint: disable=bad-whitespace


class StabilityCalc():
    r"""Manage computation of quantities to evaluate stability operator.

    Given a MOTS and a point on the MOTS, this class computes the various
    quantities needed to evaluate the MOTS stability operator [1]. Notation is
    mostly compatible with [1]. In particular, we use
    \f$k_\mu \ell^\mu = -2\f$ unless otherwise noted in the docstrings.

    Note that we assume vacuum throughout.

    @b References

    [1] Andersson, Lars, Marc Mars, and Walter Simon. "Stability of
        marginally outer trapped surfaces and existence of marginally
        outer trapped tubes." arXiv preprint arXiv:0704.2889 (2007).
    """

    def __init__(self, curve, param, metric4=None, transform_torsion=False,
                 transformation=None):
        r"""Create a stability calculator at a point on a MOTS.

        @param curve
            The MOTS.
        @param param
            The parameter along the MOTS identifying the point at which to
            compute the quantities.
        @param metric4
            The 4-metric object. This is optional and may be omitted if we
            have time-symmetry.
        @param transform_torsion
            Whether to apply a transformation to the torsion \f$s_A\f$ of
            \f$\ell^\mu\f$ such that \f$D_A s^A = 0\f$ and (in our
            coordinates) \f$s_\lambda = 0\f$. Default is `False`.
        @param transformation
            An evaluator for the explicit transformation to change the torsion
            such that \f$D_A s^A = 0\f$ and (in our coordinates)
            \f$s_\lambda = 0\f$. Can be obtained from a NumericExpression
            using motsfinder.exprs.numexpr.NumericExpression.evaluator().
        """
        self.calc = curve.get_calc_obj(param)
        self.point = self.calc.point
        self.metric = curve.metric
        self.metric4 = metric4
        self._transform_torsion = transform_torsion
        self._transformation = transformation # assuming transformation to be a
                                              # function of only \lambda
        self._g3_inv = None
        self._g4 = None
        self._dg4 = None
        self._G4 = None
        self._n = None
        self._dn = None
        self._dg3_inv = None
        self._nu3_cov = None
        self._dnu3_cov = None
        self._nu3 = None
        self._k_cov = None
        self._l_cov = None
        self._dux3 = None
        self._dnu = None
        self._cov_a_l = None
        self._cov_A_l = None
        self._torsion_1form = None
        self._torsion_vector = None
        self._q_inv = None
        self._shear = None
        self._xi_A = None

    @property
    def has_transformation(self):
        r"""Whether a transformation for the null vectors has been specified."""
        return self._transformation is not None

    def exp_sigma(self, diff=0):
        r"""Compute the tranformation \f$\exp(\sigma)\f$, or any nonzero derivatives.

        Assuming the transformation \f$\exp(\sigma)\f$ is a function of lambda
        only.
        """
        if self.has_transformation:
            return self._transformation.diff(self.calc.param, n=diff)
        return None

    def compute_op_timesym(self):
        r"""Compute the time-symmetrical stability operator terms.

        This implementation assumes the slice to be time-symmetric.

        @return A list of three elements for the three derivative orders
            \f$\partial_\lambda^0, \partial_\lambda^1, \partial_\lambda^2\f$.
        """
        op = [0.0, 0.0, 0.0]
        calc = self.calc

        # Laplacian
        self.add_neg_axisym_laplacian(op)

        # Remaining terms
        k2 = calc.extrinsic_curvature(square=True)
        nu = self.nu3
        Ric = self.metric.ricci_tensor(calc.point)
        Rnn = Ric.dot(nu).dot(nu)
        op[0] += -Rnn - k2
        return np.asarray(op)

    def compute_op_general(self, slice_normal=True):
        r"""Compute the stability operator terms at the current point.

        This implementation is for the general (axisymmetric) case not
        assuming time-symmetry.

        @return A list of three elements for the three derivative orders
            \f$\partial_\lambda^0, \partial_\lambda^1, \partial_\lambda^2\f$.

        @param slice_normal
            Whether to compute the terms for the operator w.r.t. the slice
            normal or the past-pointing outward null normal \f$-k^\mu\f$.
            Default is `True`.
        """
        op = [0.0, 0.0, 0.0]

        # 1st term: Laplacian
        self.add_neg_axisym_laplacian(op)

        # 2nd term: 2 s^A D_A zeta = 2 s^lambda partial_lambda zeta
        # Note: A possible partial_phi term should be handled by separating
        #       out the phi-dependence.
        op[1] += 2 * self.torsion_vector[0]

        # Terms with no differentiation: (1/2 R_S - Y - s^2 + Ds) zeta
        R_S = self.calc.ricci_scalar()
        if slice_normal:
            Y = self.compute_Y()
        else:
            Y = 0.0
        s2 = self.compute_s_squared()
        if self._transform_torsion:
            Ds = 0.0
        else:
            Ds = self.compute_div_s()
        op[0] += 0.5 * R_S - Y - s2 + Ds
        return np.asarray(op)

    @property
    def g3_inv(self):
        r"""Inverse 3-metric at current point (3x3-matrix)."""
        if self._g3_inv is None:
            self._g3_inv = self.calc.g.inv
        return self._g3_inv

    @property
    def g4(self):
        r"""4-metric evaluated at current point (4x4-matrix)."""
        if self._g4 is None:
            self._g4 = self.metric4.at(self.point).mat
        return self._g4

    @property
    def dg4(self):
        r"""Derivatives of 4-metric (shape=(4,4,4))."""
        # c,a,b -> partial_c g_ab
        if self._dg4 is None:
            self._dg4 = self.metric4.diff(self.point, diff=1)
        return self._dg4

    @property
    def G4(self):
        r"""Christoffel symbols of the 4-metric."""
        if self._G4 is None:
            self._G4 = self.metric4.christoffel(self.point)
        return self._G4

    @property
    def n(self):
        r"""Timelike normal to slice (shape=4)."""
        if self._n is None:
            self._n = self.metric4.normal(self.point)
        return self._n

    @property
    def dn(self):
        r"""Partial derivative of timelike normal, \f$\partial_a n^b\f$ (4x4-matrix)."""
        # shape=(4,4), a,b -> partial_a n^b
        if self._dn is None:
            self._dn = self.metric4.normal(self.point, diff=1)
        return self._dn

    @property
    def dg3_inv(self):
        r"""Derivative of inverse metric (shape=(3,3,3))."""
        if self._dg3_inv is None:
            self._dg3_inv = self.metric.diff(self.point, diff=1, inverse=True)
        return self._dg3_inv

    @property
    def nu3_cov(self):
        r"""Covariant components of normal in slice (shape=3)."""
        if self._nu3_cov is None:
            self._nu3_cov = self.calc.covariant_normal(diff=0)
        return self._nu3_cov

    @property
    def dnu3_cov(self):
        r"""Derivatives of covariant normal in slice (shape=(3,3))."""
        if self._dnu3_cov is None:
            self._dnu3_cov = self.calc.covariant_normal(diff=1)
        return self._dnu3_cov

    @property
    def nu3(self):
        r"""Normal vector in slice (shape=3)."""
        if self._nu3 is None:
            self._nu3 = self.g3_inv.dot(self.nu3_cov)
        return self._nu3

    @property
    def nu(self):
        r"""Spacetime normal vector in slice (shape=4)."""
        nu = np.zeros((4,))
        nu[1:] = self.nu3
        return nu

    @property
    def k(self):
        r"""Ingoing future-pointing null normal (shape=4) \f$k^a\f$."""
        if not self.has_transformation:
            return self._k_orig
        return 1.0/self.exp_sigma() * self._k_orig

    @property
    def _k_orig(self):
        r"""Non-transformed \f$k^a\f$."""
        return self.n - self.nu

    @property
    def l(self):
        r"""Outgoing future-pointing null normal (shape=4) \f$\ell^a\f$."""
        if not self.has_transformation:
            return self._l_orig
        return self.exp_sigma() * self._l_orig

    @property
    def _l_orig(self):
        r"""Non-transformed \f$\ell^a\f$."""
        return self.n + self.nu

    @property
    def k_cov(self):
        r"""Covariant components of \f$k\f$."""
        if self._k_cov is None:
            self._k_cov = self.g4.dot(self.k)
        return self._k_cov

    @property
    def l_cov(self):
        r"""Covariant components of \f$\ell\f$."""
        if self._l_cov is None:
            self._l_cov = self.g4.dot(self.l)
        return self._l_cov

    @property
    def dux3(self):
        r"""Derivatives of \f$x,y,z\f$ wrt \f$\lambda,\phi\f$ (shape=(2,3))."""
        # A,i -> partial_A x^i
        if self._dux3 is None:
            self._dux3 = self.calc.diff_xyz_wrt_laph(diff=1)
        return self._dux3

    @property
    def dexpSigmadlaph(self):
        r"""First derivative of the transformation \f$\exp(\sigma)\f$ wrt
        \f$\lambda, \phi\f$ (shape = (2,)).

        Assuming that the transformation \f$\exp(\sigma)\f$ is a function of
        \f$\lambda\f$ only.
        """
        return np.array([self.exp_sigma(diff=1), 0])

    @property
    def dlaphdx4(self):
        r"""Jacobian matrix \f$\frac{\partial(\lambda,\phi)}{\partial(t,x,y,z)}\f$
        (shape=(2,4)).

        Note that \f$\partial_t^n \lambda = 0 = \partial_t^n\phi\f$ for
        \f$n > 0\f$. Also,
        \f$\partial_x\phi = 0, \partial_y\phi = \frac{1}{x}, \partial_z\phi = 0\f$.
        """
        return np.array([np.concatenate(([0.0], self.calc.X)),
                         [0.0, 0.0, 1.0/self.point[0], 0.0]])

    @property
    def dexpSigmadx4(self):
        r"""First derivative of transformation \f$\exp(\sigma)\f$ wrt
        \f$t,x,y,z\f$ (shape=(4,))."""
        return np.einsum('i,ij->j', self.dexpSigmadlaph, self.dlaphdx4)

    @property
    def dl(self):
        r"""First partial derivatives of l.

        Computes \f$\partial_a\ell^b\f$, where \f$a,b\in\{t,x,y,z\}\f$.
        If the transformation \f$\exp(\sigma)\f$ is given, computes
        \f$\partial_a (\exp(\sigma)\ell^b)\f$.
        """
        # shape=(4,4), a,b -> partial_a l^b
        if not self.has_transformation:
            return self._dl_orig

        # Note that self._l_orig has shape (4,)
        # Use np.outer to make a matrix when multiplying two vectors:
        return (self.exp_sigma() * self._dl_orig
                + np.outer(self.dexpSigmadx4, self._l_orig))

    @property
    def _dl_orig(self):
        r"""Non-transformed partial derivatives of l."""
        return self.dn + self.dnu

    @property
    def ddl(self):
        r"""Second partial derivatives of l.

        Computes \f$\partial_a\partial_b \ell^c\f$, where
        \f$a,b,c\in\{t,x,y,z\}\f$. If the transformation \f$\exp(\sigma)\f$ is
        given, computes
        \f$\partial_a\partial_b \left(\exp(\sigma)\ell^c\right)\f$.
        """
        if not self.has_transformation:
            return self._ddl_orig

        # Untransformed (original) \ell wrt t,x,y,z; shape = (4,)
        l_orig = self._l_orig

        # First partial of untransformed \ell wrt t,x,y,z; shape = (4,4)
        dl_orig = self._dl_orig

        # We want to calculate the second partials of transformed \ell:
        #
        # \partial_a\partial_b (\exp(\sigma)\ell^c) =
        # \exp(\sigma)\partial_a \partial_b \ell^c     (called 'firstTerm' below)
        # + \partial_a(\exp(\sigma)) \partial_b \ell^c (called 'secTerm1stHalf')
        # + \partial_b(\exp(\sigma)) \partial_a \ell^c (called 'secTerm2ndHalf')
        # + \partial_a(\partial_b(\exp(\sigma))) \ell^c  (called 'thirdTerm' below)
        #
        # Note the difference between secTerm1stHalf and secTerm2ndHalf,
        # i.e. placement of indices

        # shape = (4,4,4)
        firstTerm = self.exp_sigma() * self._ddl_orig

        # shape = (4,4,4)
        secTerm1stHalf = np.array([[[self.dexpSigmadx4[i] * dl_orig[j][k]
                                     for k in range(0,4)]
                                    for j in range(0,4)]
                                   for i in range(0,4)])
        # shape = (4,4,4)
        secTerm2ndHalf = np.array([[[self.dexpSigmadx4[j] * dl_orig[i][k]
                                     for k in range(0,4)]
                                    for j in range(0,4)]
                                   for i in range(0,4)])
        # shape = (4,4,4)
        secTerm = secTerm1stHalf + secTerm2ndHalf

        # Calculating thirdTerm

        # thirdTerm has two parts:
        #   thirdTerm = \partial_a(\partial_b(\exp(\sigma))) \ell^c
        #             = thirdTerm1stHalf + thirdTerm2ndHalf
        # where:
        #   thirdTerm1stHalf = \frac{\partial}{\partial x^a}\left(
        #           \frac{\partial(\lambda,\phi)}{\partial x^b}
        #       \right)
        #       \partial_{\lambda,\phi} (\exp(\sigma)) \ell^c
        # and
        #   thirdTerm2ndHalf = (
        #       \frac{\partial(\lambda,\phi)}{\partial x^a}
        #       \frac{\partial(\lambda,\phi)}{\partial x^b}
        #       \partial_{\lambda,\phi}^2 (\exp(\sigma)) \ell^c
        #   )

        # Recall that \partial_{\lambda,\phi} \exp(\sigma) is
        # [\partial_\lambda \exp\sigma, \partial_\phi \exp\sigma]
        #
        # And so \partial_{\lambda,\phi}^2 (\exp(\sigma)) is
        # [[\partial_\lambda^2 \exp\sigma, \partial_\lambda\partial_\phi \exp\sigma],
        #  [\partial_\phi\partial_\lambda \exp\sigma, \partial_\phi^2 \exp\sigma]]

        #-- Calculating thirdTerm1stHalf

        # ddlambdadx2 holds second derivatives of \lambda wrt t,x,y,z
        # i.e. [\frac{\partial^2 \lambda}{\partial x^a \partial x^b}]
        ddlambdadx2 = [np.zeros((4,))]
        for i in range(0, 3):
            ddlambdadx2.append(np.concatenate(([0.0], self.calc.Y[i])))
        ddlambdadx2 = np.array(ddlambdadx2) # shape = (4,4)

        # ddphidx2 holds second derivatives of \phi wrt t,x,y,z
        # i.e. [\frac{\partial^2 \phi}{\partial x^a \partial x^b}]
        ddphidx2 = np.array([
            np.zeros((4,)),
            [0.0,0.0, -1.0/math.pow(self.point[0], 2.0), 0.0],
            [0.0, -1.0/math.pow(self.point[0], 2.0), 0.0, 0.0],
            np.zeros((4,))
        ]) # shape = (4,4)

        # ddlaphddx4 holds second derivatives of \lambda as first element,
        # second derivatives of \phi as second element
        ddlaphddx4 = np.array([ddlambdadx2, ddphidx2]) # shape = (2,4,4)

        # ddlaphddx4_dexpSigmadlaph holds the product:
        #   \frac{\partial}{\partial x^a} \left(
        #     \frac{\partial(\lambda,\phi)}{\partial x^b}
        #   \right)
        #   \partial_{\lambda,\phi} (\exp(\sigma))
        ddlaphddx4_dexpSigmadlaph = np.einsum(
            'i,ijk->jk', self.dexpSigmadlaph, ddlaphddx4
        ) # shape = (4,4)

        # Calculating final product with \ell^c in 1st half of third term
        thirdTerm1stHalf = [
            [[ddlaphddx4_dexpSigmadlaph[i][j]*l_orig[k]
              for k in range(0,4)] for j in range(0,4)]
            for i in range(0,4)
        ]

        #-- Calculating thirdTerm2ndHalf
        # i.e.:
        #   \frac{\partial(\lambda,\phi)}{\partial x^a}
        #   \frac{\partial(\lambda,\phi)}{\partial x^b}
        #   \partial_{\lambda,\phi}^2 (\exp(\sigma)) \ell^c

        # Calculating \partial_{\lambda,\phi}^2 \exp(\sigma)
        # shape = (2,2)
        ddexpSigmaddlaphi = np.array([
            [self.exp_sigma(diff=2), 0.0],
            [0.0, 0.0]
        ])

        # Calculating
        # \frac{\partial(\lambda,\phi)}{\partial x^b} \partial_{\lambda,\phi}^2 \exp(\sigma)
        firstMult = np.einsum(
            'ij,jk->ik', ddexpSigmaddlaphi, self.dlaphdx4
        ) # shape = (2,4)
        # pylint: disable=assignment-from-no-return
        firstMult = np.matrix.transpose(firstMult) # shape = (4,2)

        # Calculating
        # \frac{\partial(\lambda,\phi)}{\partial x^a}
        # \frac{\partial(\lambda,\phi)}{\partial x^b}
        # \partial_{\lambda,\phi}^2 \exp(\sigma)
        secondMult = np.einsum('ij,jk->ik', firstMult, self.dlaphdx4) # shape = (4,4)

        # Calculating full second half of third term, shape = (4,4,4)
        thirdTerm2ndHalf = np.array([[[secondMult[i][j] * l_orig[k]
                                       for k in range(0,4)]
                                      for j in range(0,4)]
                                     for i in range(0,4)])

        # shape = (4,4,4)
        return firstTerm + secTerm + thirdTerm1stHalf + thirdTerm2ndHalf

    @property
    def _ddl_orig(self):
        r"""Non-transformed second partial derivatives of l."""
        calc = self.calc
        ddn = self.metric4.normal(self.point, diff=2)
        ddg3_inv = np.asarray(self.metric.diff(self.point, diff=2, inverse=True))
        # shape=(3,3,3), i,j,k -> partial_i partial_j nu_k
        ddnu3_cov = calc.covariant_normal(diff=2)
        ddnu3 = (
            np.einsum('ijkl,l->ijk', ddg3_inv, self.nu3_cov)
            + np.einsum('ikl,jl->ijk', self.dg3_inv, self.dnu3_cov)
            + np.einsum('jkl,il->ijk', self.dg3_inv, self.dnu3_cov)
            + np.einsum('kl,ijl->ijk', self.g3_inv, ddnu3_cov)
        )
        ddnu = np.zeros((4, 4, 4)) # a,b,c -> partial_a partial_b nu^c
        ddnu[1:,1:,1:] = ddnu3
        ddnu[0,:,1:] = ddnu[:,0,1:] = np.nan

        # Second partials of untransformed \ell wrt t,x,y,z; shape = (4,4,4)
        return ddn + ddnu

    @property
    def dk(self):
        r"""First partial derivatives of \f$k\f$.

        Computes \f$\partial_a k^b\f$, where \f$a,b\in\{t,x,y,z\}\f$.
        If the transformation \f$\exp(\sigma)\f$ given, computes
        \f$\partial_a (\exp(-\sigma)k^b)\f$.
        """
        # shape = (4,4), a,b -> partial_a k^b
        if not self.has_transformation:
            return self._dk_orig

        # Assuming that self.exp_sigma is a function of \lambda only
        # Note we perform \partial\exp(-\sigma)/\partial\lambda =
        # -\frac{1}{\exp(2\sigma)}\partial\exp(\sigma)/\partial\lambda

        # Note that self._k_orig has shape (4,)
        # Use np.outer to make a matrix when multiplying two vectors:
        exp_sigma = self.exp_sigma()
        return (
            1.0/exp_sigma * self._dk_orig
            - (1.0/math.pow(exp_sigma, 2.0)
               * np.outer(self.dexpSigmadx4, self._k_orig))
        )

    @property
    def _dk_orig(self):
        r"""Non-transformed partial derivatives of k."""
        return self.dn - self.dnu

    @property
    def dnu3(self):
        r"""Spatial derivatives of MOTS normal, \f$\partial_i \nu^j\f$."""
        return (
            np.einsum('ijk,k->ij', self.dg3_inv, self.nu3_cov)
            + np.einsum('jk,ik->ij', self.g3_inv, self.dnu3_cov)
        )

    @property
    def dnu(self):
        r"""Spacetime derivatives of MOTS normal in slice."""
        if self._dnu is None:
            dnu = np.zeros((4, 4)) # a,b -> partial_a nu^b (normal of MOTS)
            dnu[0,1:] = np.nan # don't know time derivatives of MOTS's normal
            dnu[1:,1:] = self.dnu3
            self._dnu = dnu
        return self._dnu

    @property
    def cov_a_l(self):
        r"""Covariant derivative of \f$\ell\f$ wrt \f$t,x,y,z\f$ (shape=(4,4))."""
        # a,b -> nabla_a l^b  (NaN for a=0)
        if self._cov_a_l is None:
            self._cov_a_l = self.dl + np.einsum('min,n', self.G4, self.l)
        return self._cov_a_l

    @property
    def cov_A_l(self):
        r"""Covariant derivative of \f$\ell\f$ wrt \f$\lambda,\phi\f$ (shape=(2,4))."""
        # A,mu -> nabla_A l^mu
        if self._cov_A_l is None:
            self._cov_A_l = np.einsum('Ai,im->Am', self.dux3, self.cov_a_l[1:])
        return self._cov_A_l

    @property
    def torsion_1form(self):
        r"""Torsion \f$s_A\f$ of \f$\ell\f$ (aka rotation 1-form \f$\Omega\f$, shape=2)."""
        # A -> -1/2 k_mu nabla_A l^mu
        if self._torsion_1form is None:
            self._torsion_1form = -0.5 * np.einsum(
                'm,Am->A', self.k_cov, self.cov_A_l
            )
            if self._transform_torsion:
                self._torsion_1form[0] = 0.0
        return self._torsion_1form

    @property
    def torsion_vector(self):
        r"""Contravariant version of torsion_1form() (shape=2)."""
        if self._torsion_vector is None:
            self._torsion_vector = self.q_inv.dot(self.torsion_1form)
        return self._torsion_vector

    @property
    def q_inv(self):
        r"""Inverse 2-metric \f$q^AB\f$ on MOTS (shape=(2,2))."""
        if self._q_inv is None:
            self._q_inv = self.calc.induced_metric(inverse=True) # q^AB
        return self._q_inv

    def add_neg_axisym_laplacian(self, op):
        r"""Add -Laplacian to the operator, assuming no \f$\phi\f$-dependence."""
        # Note: A partial_phi^2 term should be handled by separating out the
        #       phi-dependence.
        q_inv = self.q_inv # q^AB
        dq = self.calc.induced_metric(diff=1) # shape=(2,2,2), C,A,B -> partial_C q_AB
        op[1] += -(
            0.5 * np.einsum('AB,C,CAB', q_inv, q_inv[:,0], dq)
            - np.einsum('CA,B,CAB', q_inv, q_inv[0,:], dq)
        )
        op[2] += -q_inv[0,0]

    def compute_s_squared(self):
        r"""Compute \f$s_A s^A\f$."""
        # s^A s_A
        return self.torsion_vector.dot(self.torsion_1form)

    def compute_K_AB_l(self):
        r"""Compute the extrinsic curvature vector contracted with \f$\ell\f$.

        The quantity computed here is \f$K^\mu_{AB}\ell_\mu\f$.
        """
        dl_cov = ( # (\mu,\nu) -> \partial_\mu \ell_\nu
            np.einsum('ija,a->ij', self.dg4, self.l)
            + np.einsum('ja,ia->ij', self.g4, self.dl)
        )
        dux3 = self.dux3 # shape=(2,3), A,i -> partial_A x^i
        K_AB_l = -( # shape=(2,2), A,B -> K^mu_AB l_mu
            np.einsum('Ai,Bj,ij->AB', dux3, dux3, dl_cov[1:,1:])
            - np.einsum('Ai,Bj,aij,a->AB',
                        dux3, dux3, self.G4[:,1:,1:], self.l_cov)
        )
        return K_AB_l

    def compute_K_AB_k(self):
        r"""Compute the extrinsic curvature vector contracted with \f$k\f$.

        The quantity computed here is \f$K^\mu_{AB}k_\mu\f$, where \f$k^\mu\f$
        is the ingoing null normal. Note that the scaling of the returned
        quantity is such that \f$\ell \cdot k = -2\f$.
        """
        dk_cov = ( # (\mu,\nu) -> \partial_\mu k_\nu
            np.einsum('ija,a->ij', self.dg4, self.k)
            + np.einsum('ja,ia->ij', self.g4, self.dk)
        )
        dux3 = self.dux3 # shape=(2,3), A,i -> partial_A x^i
        K_AB_k = -( # shape=(2,2), A,B -> K^mu_AB k_mu
            np.einsum('Ai,Bj,ij->AB', dux3, dux3, dk_cov[1:,1:])
            - np.einsum('Ai,Bj,aij,a->AB',
                        dux3, dux3, self.G4[:,1:,1:], self.k_cov)
        )
        return K_AB_k

    def compute_Y(self):
        r"""Compute the Y quantity of [1] for the slice normal."""
        K_AB_l = self.compute_K_AB_l()
        q_inv = self.q_inv # q^AB
        Y = 0.5 * np.einsum('AB,AC,BD,CD', K_AB_l, q_inv, q_inv, K_AB_l)
        return Y

    def compute_shear(self):
        r"""Compute the shear tensor in coordinates on the surface.

        This computes \f$\sigma_{AB}\f$, where
        \f[
            \sqrt{2} \sigma_{AB}
                = \nabla_A \ell_B - \frac12 q_{AB} q^{CD} \nabla_C \ell_D
                = -K_{AB}^\mu\ell_\mu
                  + \frac12 q_{AB} q^{CD} K_{CD}^\mu \ell_\mu \,.
        \f]
        The factor of \f$\sqrt{2}\f$ results from our definition of the
        ingoing and outgoing null normals \f$k^\mu\f$ and \f$\ell^\mu\f$, i.e.
        we use a scaling where \f$\ell^\mu k_\mu = -2\f$, whereas e.g. [1]
        uses \f$\ell^\mu k_\mu = -1\f$. This means the values computed here
        are compatible with [1].

        Also note that for a MOTS, the second term above vanishes by
        definition.

        @b References

        [1] Dreyer, O., Krishnan, B., Shoemaker, D., & Schnetter, E. (2003).
            Introduction to isolated horizons in numerical relativity.
            Physical Review D, 67(2), 024018.
        """
        if self._shear is not None:
            return self._shear
        K_AB_l = self.compute_K_AB_l()
        q = self.calc.induced_metric()
        q_inv = self.q_inv # q^AB
        # Note: The second term below is zero for a MOTS (i.e. the first term
        #       is trace-free already).
        sigma = 1/np.sqrt(2) * (
            -K_AB_l
            + 0.5 * np.einsum('AB,CD,CD->AB', q, q_inv, K_AB_l)
        )
        self._shear = sigma
        return sigma

    def compute_shear_squared(self):
        r"""Compute ``|sigma|^2 = sigma_AB sigma^AB``.

        Uses the convention \f$\ell \cdot k = -1\f$.
        """
        sigma_AB = self.compute_shear()
        q_inv = self.q_inv # q^AB
        return np.einsum('AB,AC,BD,CD', sigma_AB, q_inv, q_inv, sigma_AB)

    def compute_shear_scalar(self):
        r"""Compute the shear scalar.

        See .expcurve.ExpansionCurve.shear_scalar() for the definition of the
        shear scalar computed here.

        Uses the convention \f$\ell \cdot k = -1\f$.
        """
        sigma_AB = self.compute_shear()
        q = self.calc.induced_metric()
        # The imaginary part should actually vanish.
        return (0.5 * (sigma_AB[0, 0]/q[0, 0] - sigma_AB[1, 1]/q[1, 1])
                + 1j * sigma_AB[0, 1]/np.sqrt(q[0, 0] * q[1, 1]))

    def compute_shear_k(self, full_output=False):
        r"""Compute the shear of the ingoing null normal.

        Similar to compute_shear(), this computes the symmetric tensor w.r.t.
        coordinates on the surface, i.e.
        \f[
            \sigma_{AB}
                = \frac{1}{\sqrt2} (\nabla_A k_B - \frac12 q_{AB} q^{CD} \nabla_C k_D)
                = \frac{1}{\sqrt2} (-K_{AB}^\mu k_\mu - \frac12 q_{AB} \Theta_{(k)}) \,.
        \f]

        The factor of \f$1/\sqrt2\f$ makes this result compatible with the
        convention \f$\ell \cdot k = -1\f$.

        @param full_output
            If `False` (default), return just the shear of `k`. If `True`,
            return
                * the symmetric tracefree shear of `k` as 2x2 matrix w.r.t.
                  the current coordinates on the MOTS
                * the (coordinate independent) square of this shear
                * the expansion of `k`
                * the trace of the shear (for diagnostic/debugging purposes)
        """
        K_AB_k = self.compute_K_AB_k() / np.sqrt(2) # convert to ell*k = -1 convention
        q = self.calc.induced_metric()
        q_inv = self.q_inv # q^AB
        theta_k = -np.einsum('AB,AB', q_inv, K_AB_k)
        sigma_k_AB = - K_AB_k - 0.5 * theta_k * q
        if full_output:
            sigma_k2 = np.einsum('AB,AC,BD,CD', sigma_k_AB, q_inv, q_inv, sigma_k_AB)
            tr = np.einsum('AB,AB', q_inv, sigma_k_AB)
            return sigma_k_AB, sigma_k2, theta_k, tr
        return sigma_k_AB

    def compute_ds_cov(self):
        r"""Compute the partial derivatives of the rotation 1-form \f$s_A\f$.

        Indices `A, B` correspond to \f$\partial_A s_B\f$, where `A, B` run
        over \f$\lambda\f$ and \f$\varphi\f$.
        """
        ddl = self.ddl # shape=(4,4,4), a,b,c -> partial_a partial_b l^c
        ddux3 = self.calc.diff_xyz_wrt_laph(diff=2) # shape=(2,2,3), A,B,i -> partial_A partial_B x^i
        dG4 = self.metric4.christoffel_deriv(self.point)
        dux3 = self.dux3
        # du_cov_i_l = partial_A nabla_i l^a
        #   = partial_A x^j (partial_i partial_j l^a + partial_j G^a_ib l^b + G^a_ib partial_j l^b)
        #   -> shape=(2,3,4)
        du_cov_i_l = (
            np.einsum('Aj,ija->Aia', dux3, ddl[1:,1:,:])
            + np.einsum('Aj,jaib,b->Aia', dux3, dG4[1:,:,1:,:], self.l)
            + np.einsum('Aj,aib,jb->Aia', dux3, self.G4[:,1:,:], self.dl[1:,:])
        )
        dk = self.dk # shape=(4,4), a,b -> partial_a k^b
        duk_cov = ( # shape=(2,4), A,a -> partial_A k_a
            np.einsum('Ai,iab,b->Aa', dux3, self.dg4[1:,:,:], self.k)
            + np.einsum('ab,Ai,ib->Aa', self.g4, dux3, dk[1:,:])
        )
        ds_cov = -0.5 * ( # shape=(2,2), A,B -> partial_A s_B
            np.einsum('Aa,Ba->AB', duk_cov, self.cov_A_l)
            + np.einsum('a,ABi,ia->AB', self.k_cov, ddux3, self.cov_a_l[1:,:])
            + np.einsum('a,Bi,Aia->AB', self.k_cov, dux3, du_cov_i_l)
        )
        return ds_cov

    def compute_div_s(self):
        r"""Compute the divergence div(s) of the rotation 1-form \f$s_A\f$."""
        # The quantity we compute is div(s) = D_A s^A = q^AB D_A s_B.
        # Here D_A is the covariant derivative compatible with the 2-metric q.
        ds_cov = self.compute_ds_cov()
        G2 = self.calc.christoffel() # A,B,C -> Gamma^A_BC (Christoffel on MOTS)
        Ds = (np.einsum('AB,AB', self.q_inv, ds_cov)
              - np.einsum('AB,CAB,C', self.q_inv, G2, self.torsion_1form))
        return Ds

    def nabla_m_ell_n(self, dt_normal):
        r"""Compute \f$\nabla_\mu \ell_\nu\f$ (shape=(4,4), no NaNs).

        In order to compute all values, including the time derivatives which
        usually get assigned NaNs, we need the time derivative of the normal
        to the MOTSs. This requires knowledge of the MOTT and hence is not the
        responsibility of this calculator class.
        """
        l = self.l
        dnu = self.dnu.copy()
        dnu[0,1:] = dt_normal
        dl = (self.dn + dnu)
        nabla_m_ell_n = np.einsum( # shape=(4,4), m,n -> nabla_m ell_n
            'bc,ac->ab', self.g4, dl + np.einsum('cad,d->ac', self.G4, l)
        )
        return nabla_m_ell_n

    def nabla_m_k_n(self, dt_normal):
        r"""Compute \f$\nabla_\mu k_\nu\f$ (shape=(4,4), no NaNs).

        In order to compute all values, including the time derivatives which
        usually get assigned NaNs, we need the time derivative of the normal
        to the MOTSs. This requires knowledge of the MOTT and hence is not the
        responsibility of this calculator class.
        """
        k = self.k
        dnu = self.dnu.copy()
        dnu[0,1:] = dt_normal
        dk = (self.dn - dnu)
        nabla_m_k_n = np.einsum( # shape=(4,4), m,n -> nabla_m k_n
            'bc,ac->ab', self.g4, dk + np.einsum('cad,d->ac', self.G4, k)
        )
        return nabla_m_k_n

    def surface_gravity(self, dt_normal, X=None):
        r"""Compute a (slicing dependent) surface gravity.

        See .expcurve.ExpansionCurve.surface_gravity() for more information.
        Computes
        \f[
            \kappa^{(X)} = - k_\mu X^\nu \nabla_\nu \ell^\mu \,,
        \f]
        where \f$X^\mu = \ell^\mu\f$ by default.

        Uses the convention \f$\ell \cdot k = -1\f$, in particular
        \f[
            \ell^\mu = \frac{1}{\sqrt{2}} (n^\mu + v^\mu) \,,
            \qquad
            k^\mu = \frac{1}{\sqrt{2}} (n^\mu - v^\mu) \,.
        \f]

        @param dt_normal
            The term \f$\partial_t v^\mu\f$, where `v` is the normal of the
            MOTS within the slice. Note that this should not simply be the
            time derivative of the normal's components. See
            .expcurve.ExpansionCurve.dt_normals().
        @param X
            4-vector with respect to wich to compute the surface gravity.
            Default is to use the outgoing null normal \f$\ell^\mu\f$.
        """
        nabla_m_ell_n = self.nabla_m_ell_n(dt_normal) / np.sqrt(2)
        # different normalization here: l*k = -1
        k_cov = self.k_cov / np.sqrt(2)
        if X is None:
            X = self.l / np.sqrt(2)
        g4_inv = np.linalg.inv(self.g4)
        kappa = - (
            np.einsum('m,n,mr,nr->', k_cov, X, g4_inv, nabla_m_ell_n)
        )
        return kappa

    def xi_vector(self, dt_normal, tev, up=True):
        r"""Compute the xi vector.

        See .expcurve.ExpansionCurve.xi_vector() for the definition of the
        quantity computed here. Note that the result is cached, making
        repeated calls to this method (e.g. with different values for `up`)
        cheap.

        @param dt_normal
            Derivative w.r.t. simulation time (t-coordinate) of the normal of
            the MOTS within the slice.
        @param tev
            Time evolution vector (i.e. a TimeVectorData object) at the
            current point of the MOTS.
        @param up
            Whether to return \f$\xi^A\f$ (`True`) or \f$\xi_A\f$ (`False`).
            Default is `True`.
        """
        if self._xi_A is None:
            nabla_m_ell_n = self.nabla_m_ell_n(dt_normal) / np.sqrt(2)
            V = tev.V_tilde
            V = V / np.sqrt(abs(self.g4.dot(V).dot(V)))
            # \chi_\nu = V^\rho \nabla_\rho \ell_\nu
            chi_n = np.einsum('r,rn->n', V, nabla_m_ell_n)
            # \xi_A = x^i_{,A} \chi_i, shape=(2,)
            self._xi_A = np.einsum('Ai,i->A', self.dux3, chi_n[1:])
        if up:
            return self.q_inv.dot(self._xi_A)
        return self._xi_A

    def xi_squared(self, dt_normal, tev):
        r"""Compute the square \f$\xi^A\xi_A\f$ of the xi vector."""
        xi_A = self.xi_vector(dt_normal, tev, up=False)
        xi_A_up = self.xi_vector(dt_normal, tev, up=True)
        return xi_A.dot(xi_A_up)

    def xi_scalar(self, dt_normal, tev):
        r"""Compute a complex scalar representing the xi vector.

        See .expcurve.ExpansionCurve.expand_xi_scalar() for details.
        """
        xi_A = self.xi_vector(dt_normal, tev, up=False)
        q = self.calc.induced_metric()
        return 1/np.sqrt(2) * (
            xi_A[0] / np.sqrt(q[0, 0])
            - 1j * xi_A[1] / np.sqrt(q[1, 1])
        )

    def print_debug_info(self):
        from motsfinder.utils import print_indented
        from motsfinder.ipyutils import disp
        from sympy import Matrix, Float
        dl_cov = ( # (\mu,\nu) -> \partial_\mu \ell_\nu
            np.einsum('ija,a->ij', self.dg4, self.l)
            + np.einsum('ja,ia->ij', self.g4, self.dl)
        )
        nabla_m_ell_n = dl_cov - ( # shape=(4,4), m,n -> nabla_m ell_n
            np.einsum('nij,n->ij', self.G4, self.l_cov)
        )
        q = self.calc.induced_metric()
        g3 = self.calc.g.mat
        dla, dph = self.dux3
        disp(r"\partial_\lambda = %s", Matrix(dla).T)
        disp(r"\partial_\varphi = %s", Matrix(dph).T)
        print()
        print("=" * 72)
        print("3-metric:")
        disp(r"\qquad h_{ab} = %s", Matrix(g3))
        g3_eig = np.linalg.eigvals(g3)
        print_indented("  eigenvalues: ", list(sorted(g3_eig)))
        print()
        tol = 1e-10
        ok = (r"\qquad\color{red}{\text{fail}}", r"\qquad\color{green}{\text{ok}}")
        disp(r"h(\partial_\lambda,\partial_\lambda) = %s"
             + ok[abs(g3.dot(dla).dot(dla)-q[0,0]) <= tol],
             Float(g3.dot(dla).dot(dla)))
        disp(r"h(\partial_\varphi,\partial_\varphi) = %s"
             + ok[abs(g3.dot(dph).dot(dph)-q[1,1]) <= tol],
             Float(g3.dot(dph).dot(dph)))
        print()
        nu3_cov = g3.dot(self.nu3)
        disp(r"v^a = %s", Matrix(self.nu3).T)
        disp(r"v_a = %s", Matrix(nu3_cov).T)
        disp(r"v^a v_a = %s", Float(nu3_cov.dot(self.nu3)))
        print()
        print("=" * 72)
        print("Induced 2-metric:")
        q_ab = g3 - np.outer(nu3_cov, nu3_cov)
        disp(r"\qquad q_{AB} = %s", Matrix(q))
        disp(r"\qquad q_{ab} = h_{ab} - v_a v_b = %s", Matrix(q_ab))
        q_ab_eig = np.linalg.eigvals(q_ab)
        print_indented("  eigenvalues: ", list(sorted(q_ab_eig)))
        print()
        disp(r"q_{ab} \lambda^a \lambda^b = %s, "
             r"\quad\text{where } \lambda^a = \frac{\partial x^i}{\partial\lambda},"
             r"\text{ i.e. } \partial_\lambda = \lambda^i \partial_i"
             + ok[abs(np.einsum('ab,a,b->', q_ab, dla, dla)-q[0,0]) <= tol],
             Float(np.einsum('ab,a,b->', q_ab, dla, dla)))
        disp(r"q_{ab} \varphi^a \varphi^b = %s, "
             r"\quad\text{where } \varphi^a = \frac{\partial x^i}{\partial\varphi}"
             r"\text{ i.e. } \partial_\varphi = \varphi^i \partial_i"
             + ok[abs(np.einsum('ab,a,b->', q_ab, dph, dph)-q[1,1]) <= tol],
             Float(np.einsum('ab,a,b->', q_ab, dph, dph)))
        disp(r"q_{ab} v^a = %s"
             + ok[np.allclose(q_ab.dot(self.nu3), 0.0, rtol=0.0, atol=tol)],
             Matrix(q_ab.dot(self.nu3)).T)
        q_ab_up = np.einsum('ac,bd,cd->ab', self.g3_inv, self.g3_inv, q_ab)
        disp(r"q^{ab} v_a = %s"
             + ok[np.allclose(q_ab_up.dot(nu3_cov), 0.0, rtol=0.0, atol=tol)],
             Matrix(q_ab_up.dot(nu3_cov)).T)
        nu = self.nu
        g4 = self.g4
        nu_cov = g4.dot(nu)
        n_cov = g4.dot(self.n)
        print()
        q_mn = g4 + np.outer(n_cov, n_cov) - np.outer(nu_cov, nu_cov)
        disp(r"\qquad q_{\mu\nu} = %s", Matrix(q_mn))
        q_mn_eig = np.linalg.eigvals(q_mn)
        print_indented("  eigenvalues: ", list(sorted(q_mn_eig)))
        disp(r"(q_{\mu\nu})_{ab} - q_{ab} = %s"
             + ok[np.allclose(q_mn[1:,1:]-q_ab, 0.0, rtol=0.0, atol=tol)],
             Matrix(q_mn[1:,1:]-q_ab))
        dla4 = np.zeros(4); dla4[1:] = dla
        dph4 = np.zeros(4); dph4[1:] = dph
        disp(r"q_{\mu\nu} \lambda^\mu \lambda^\nu = %s"
             + ok[abs(np.einsum('mn,m,n->', q_mn, dla4, dla4)-q[0,0]) <= tol],
             Float(np.einsum('mn,m,n->', q_mn, dla4, dla4)))
        disp(r"q_{\mu\nu} \varphi^\mu \varphi^\nu = %s"
             + ok[abs(np.einsum('mn,m,n->', q_mn, dph4, dph4)-q[1,1]) <= tol],
             Float(np.einsum('mn,m,n->', q_mn, dph4, dph4)))
        disp(r"q_{\mu\nu} v^\mu = %s"
             + ok[np.allclose(q_mn.dot(nu), 0.0, rtol=0.0, atol=tol)],
             Matrix(q_mn.dot(nu)).T)
        disp(r"q_{\mu\nu} \ell^\mu = %s"
             + ok[np.allclose(q_mn.dot(self.l), 0.0, rtol=0.0, atol=tol)],
             Matrix(q_mn.dot(self.l)).T)
        disp(r"q_{\mu\nu} n^\mu = %s"
             + ok[np.allclose(q_mn.dot(self.k), 0.0, rtol=0.0, atol=tol)],
             Matrix(q_mn.dot(self.k)).T)
        print()
        print("=" * 72)
        print("4-metric:")
        disp(r"\qquad g_{\mu\nu} = %s", Matrix(g4))
        g4_eig = np.linalg.eigvals(g4)
        print_indented("  eigenvalues: ", list(sorted(g4_eig)))
        h_mn = g4 - np.einsum('m,n->mn', n_cov, n_cov)
        disp(r"\qquad h_{\mu\nu} = %s", Matrix(h_mn))
        g4_inv = np.linalg.inv(g4)
        q_mn_up = np.einsum('mr,ns,rs->mn', g4_inv, g4_inv, q_mn)
        disp(r"\qquad q^{\mu\nu} = %s", Matrix(q_mn_up))
        #print_indented("  q^mn = ", q_mn_up)
        #print_indented("Gamma = ", self.G4)
        print()
        print("=" * 72)
        disp(r"\ell^\mu = %s", Matrix(self.l).T)
        disp(r"\ell_\mu = %s", Matrix(self.l_cov).T)
        disp(r"q^{\mu\nu} \ell_\mu = %s"
             + ok[np.allclose(q_mn_up.dot(self.l_cov), 0.0, rtol=0.0, atol=tol)],
             Matrix(q_mn_up.dot(self.l_cov)).T)
        print()
        disp(r"\partial_\mu \ell_\nu = %s", Matrix(dl_cov))
        disp(r"\nabla_\mu \ell_\nu = %s", Matrix(nabla_m_ell_n))
        print()
        # Currently NaN:
        #disp(r"q^{\mu\nu} \nabla_\mu \ell_\nu = %s",
        #     Float(np.einsum('mn,mn->', q_mn_up, nabla_m_ell_n)))
        #print()
        disp(r"v^\mu = %s", Matrix(nu).T)


class StabilitySpectrum():
    r"""Convenience class to organize stability spectra for different angular modes.

    The values are organized using a *main index* `l` and an angular mode
    index `m`, the latter running through the ``2 l + 1`` values
    ``-l, ..., 0, ..., l``.
    """

    def __init__(self, rtol, zeta=None):
        r"""Initialize an empty spectrum.

        Use the add() method to add the spectra of different angular modes.

        @param rtol
            Relative tolerance used to check for (approximate) equality of
            eigenvalues (in order to count multiplicities) and for checking if
            an eigenvalue is real.
        """
        ## Relative tolerance for an imaginary part to be considered zero.
        ## Used to determine if a value is real.
        self.rtol = rtol
        ## Maximum main mode to display when printing this object.
        self.print_max_l = 30
        ## Maximum angular mode to display when printing.
        self.print_max_m = None
        ## Azimuthal parameter mapping from curve parameter to zeta.
        self.zeta = zeta
        self._spectra = dict()
        self._eigenfunctions = dict()

    def __setstate__(self, state):
        r"""Restore this object from the given unpickled state."""
        # compatibility with data from previous versions
        self._eigenfunctions = dict()
        self.zeta = None
        # Restore state. This overrides the above if contained in the data.
        self.__dict__.update(state)

    def add(self, m, spectrum, eigenfunctions=()):
        r"""Add the spectrum of angular mode m."""
        if m in self._spectra:
            raise ValueError("Spectrum for `m=%s` already stored." % m)
        self.replace(m, spectrum, eigenfunctions)

    def replace(self, m, spectrum, eigenfunctions=()):
        r"""Replace the spectrum of angular mode m."""
        if eigenfunctions and len(eigenfunctions) != len(spectrum):
            raise ValueError(
                "Need the same number of eigenfunctions as eigenvalues."
            )
        if eigenfunctions:
            data = list(zip(spectrum, eigenfunctions))
            data.sort(key=lambda x: x[0].real)
            spectrum, eigenfunctions = list(zip(*data))
        self._spectra[m] = np.asarray(
            sorted(spectrum, key=lambda val: val.real)
        )
        self._eigenfunctions[m] = eigenfunctions

    @property
    def principal(self):
        r"""Eigenvalue with smallest real part (should be real)."""
        return self.get(l=0, m=0)

    @property
    def spectrum(self):
        r"""All eigenvalues in order of ascending real parts.

        Note that eigenvalues may appear multiple times in this list. However,
        it is not guaranteed that it will appear according to its multiplicity
        if higher angular modes have not been computed. Use multiplicity() to
        get a slightly more robust measure of multiplicity by raising an error
        if multiplicity seems to not be available in the data.
        """
        values = [v for m in self._spectra for v in self._spectra[m]]
        return sorted(values, key=lambda v: v.real)

    def get(self, l='all', m=0, simp=True):
        r"""Return eigenvalues.

        @param l
            Main eigenvalue index as ``int``. Use the string ``"all"``
            (default) to get all eigenvalues of the given angular mode `m`.
        @param m
            Angular mode. Runs over ``-l, ..., 0, ..., l``. Default is `0`.
        @param simp
            If `True` (default) and the value is found to have negligible
            imaginary part, return it as a `float` instead of a `complex`.
        """
        if m not in self._spectra:
            raise ValueError("Spectrum for angular mode `m=%s` not available." % m)
        if l in (None, 'all'):
            return self._spectra[m]
        if l < abs(m):
            raise ValueError("Angular mode `m > l`.")
        val = self._spectra[m][l-abs(m)]
        if simp and self.is_value_real(val):
            val = val.real
        return val

    def get_eigenfunction(self, l, m, evaluator=True):
        r"""Return the eigenfunction for the specified eigenvalue.

        Produces an error in case eigenfunctions have not been computed. Note
        that the eigenfunctions are neither normalized (in a meaningful way)
        nor will their sign be consistent.

        @param l
            Main eigenvalue index as ``int``. Valid values are `0` up to and
            including ``l_max + |m|``.
        @param m
            Angular mode. Runs over ``-l, ..., 0, ..., l``.
        @param evaluator
            Whether to return a callable (`True`, default), or the expression
            object itself.
        """
        if m not in self._spectra:
            raise ValueError("Spectrum for angular mode `m=%s` not available." % m)
        if l < abs(m):
            raise ValueError("Angular mode `m > l`.")
        func = self._eigenfunctions[m][l-abs(m)]
        if not evaluator:
            return func
        ev = func.evaluator()
        for attr in ("eigenvalue", "eigenfunction_number"):
            setattr(ev, attr, getattr(func, attr))
        return ev

    @property
    def angular_modes(self):
        r"""List of angular modes for which we have data."""
        return sorted(self._spectra.keys(), key=lambda m: (abs(m), m))

    @property
    def l_max(self):
        r"""Maximum main eigenvalue index `l`."""
        return len(self._spectra[0]) - 1

    def multiplicity(self, l, m=0, disp=True, verbose=False):
        r"""Determine the multiplicity of an eigenvalue based on available data.

        The returned value is not guaranteed to be correct if our data is not
        sufficiently complete. A simple heuristic is used to guess whether we
        *should* have enough data for the given eigenvalue. As a rule of
        thumb, if you find that the lowest eigenvalue you get when further
        increasing `m` is larger than the eigenvalue you want to know the
        multiplicity of, then the data should be complete. This is the
        implemented heuristic.

        @param l
            Main eigenvalue index as ``int``.
        @param m
            Angular mode. Runs over ``-l, ..., 0, ..., l``.
        @param disp
            Whether to raise a `ValueError` if the above heuristic indicates
            insufficient data for computing multiplicity. Default is `True`.
        @param verbose
            In case ``disp==False``, whether to at least print a warning in
            that case.
        """
        val = self.get(l, m)
        def _eq(v):
            return abs(val-v) <= self.rtol * abs(val)
        m_max = max(self._spectra.keys())
        m_min = min(self._spectra.keys())
        if (self.get(l=m_max, m=m_max).real < val.real
                or self.get(l=m_max, m=m_min).real < val.real):
            msg = ("Spectra for higher angular modes not computed. "
                   "Cannot reliably determine multiplicity.")
            if disp:
                raise ValueError(msg)
            if verbose:
                print("WARNING: %s" % msg)
        return sum(1 for v in self.spectrum if _eq(v))

    def is_real(self, l=0, m=0):
        r"""Whether the indicated eigenvalue is real. """
        return self.is_value_real(self.get(l=l, m=m))

    def is_value_real(self, value):
        r"""Whether a given value is real within some tolerance.

        The tolerance used is the `rtol` instance property created upon object
        initialization.
        """
        return abs(value.imag) <= self.rtol * abs(value.real)

    def __str__(self):
        if not self._spectra:
            return "empty spectrum object"
        if 0 not in self._spectra:
            return "spectrum without m=0 component"
        def _s(v):
            if self.is_value_real(v):
                return "%g" % v.real
            return "(%g%s%gj)" % (
                v.real, "+-"[v.imag >= 0], v.imag
            )
        msg = "Principal eigenvalue: %s\n" % self.principal.real
        msg += "Computed spectrum:\n"
        for l in range(len(self._spectra[0])):
            if l > self.print_max_l:
                msg += "...\n"
                break
            eigvals = [self.get(l=l, m=m) for m in self.angular_modes
                       if l >= abs(m)
                           and (self.print_max_m is None or abs(m) <= self.print_max_m)
                           and len(self._spectra[m]) > (l-abs(m))]
            msg += "l=%2s: %s\n" % (l, ", ".join(_s(v) for v in eigvals))
        return msg
