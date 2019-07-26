r"""@package motsfinder.axisym.curve.stabcalc

Compute quantities needed for the stability operator.

This is used by the ExpansionCurve to compute the stability operator in order
to evaluate its spectrum.
"""

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
    \f$k_\mu \ell^\mu = -2\f$.

    Note that we assume vacuum throughout.

    @b References

    [1] Andersson, Lars, Marc Mars, and Walter Simon. "Stability of
        marginally outer trapped surfaces and existence of marginally
        outer trapped tubes." arXiv preprint arXiv:0704.2889 (2007).
    """

    def __init__(self, curve, param, metric4=None, transform_torsion=False):
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
        """
        self.calc = curve.get_calc_obj(param)
        self.point = self.calc.point
        self.metric = curve.metric
        self.metric4 = metric4
        self._transform_torsion = transform_torsion
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
        r"""Timelike normal to slice (shape=3)."""
        if self._n is None:
            self._n = self.metric4.normal(self.point)
        return self._n

    @property
    def dn(self):
        r"""Partial derivative of timelike normal, partial_a n^b (4x4-matrix)."""
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
        r"""Ingoing future-pointing null normal (shape=4) k^a."""
        return self.n - self.nu

    @property
    def l(self):
        r"""Outgoing future-pointing null normal (shape=4) l^a."""
        return self.n + self.nu

    @property
    def k_cov(self):
        r"""Covariant components of k."""
        if self._k_cov is None:
            self._k_cov = self.g4.dot(self.k)
        return self._k_cov

    @property
    def l_cov(self):
        r"""Covariant components of l."""
        if self._l_cov is None:
            self._l_cov = self.g4.dot(self.l)
        return self._l_cov

    @property
    def dux3(self):
        r"""Derivatives of x,y,z wrt lambda,phi (shape=(2,3))."""
        # A,i -> partial_A x^i
        if self._dux3 is None:
            self._dux3 = self.calc.diff_xyz_wrt_laph(diff=1)
        return self._dux3

    @property
    def dl(self):
        r"""Partial derivatives of l."""
        # shape=(4,4), a,b -> partial_a l^b
        return self.dn + self.dnu

    @property
    def dnu(self):
        r"""Spacetime derivatives of MOTS normal in slice."""
        if self._dnu is None:
            dnu3 = (
                np.einsum('ijk,k->ij', self.dg3_inv, self.nu3_cov)
                + np.einsum('jk,ik->ij', self.g3_inv, self.dnu3_cov)
            )
            dnu = np.zeros((4, 4)) # a,b -> partial_a nu^b (normal of MOTS)
            dnu[0,1:] = np.nan # don't know time derivatives of MOTS's normal
            dnu[1:,1:] = dnu3
            self._dnu = dnu
        return self._dnu

    @property
    def cov_a_l(self):
        r"""Covariant derivative of l wrt t,x,y,z (shape=(4,4))."""
        # a,b -> nabla_a l^b  (NaN for a=0)
        if self._cov_a_l is None:
            self._cov_a_l = self.dl + np.einsum('min,n', self.G4, self.l)
        return self._cov_a_l

    @property
    def cov_A_l(self):
        r"""Covariant derivative of l wrt lambda,phi (shape=(2,4))."""
        # A,mu -> nabla_A l^mu
        if self._cov_A_l is None:
            self._cov_A_l = np.einsum('Ai,im->Am', self.dux3, self.cov_a_l[1:])
        return self._cov_A_l

    @property
    def torsion_1form(self):
        r"""Torsion s_A of l (aka rotation 1-form Omega, shape=2)."""
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
        r"""Inverse 2-metric q^AB on MOTS (shape=(2,2))."""
        if self._q_inv is None:
            self._q_inv = self.calc.induced_metric(inverse=True) # q^AB
        return self._q_inv

    def add_neg_axisym_laplacian(self, op):
        r"""Add -Laplacian to the operator, assuming no phi-dependence."""
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
        r"""Compute s_A s^A."""
        # s^A s_A
        return self.torsion_vector.dot(self.torsion_1form)

    def compute_Y(self):
        r"""Compute the Y quantity of [1] for the slice normal."""
        dl_cov = (
            np.einsum('ija,a->ij', self.dg4, self.l)
            + np.einsum('ja,ia->ij', self.g4, self.dl)
        )
        dux3 = self.dux3 # shape=(2,3), A,i -> partial_A x^i
        K_AB_l = -( # shape=(2,2), A,B -> K^mu_AB l_mu
            np.einsum('Ai,Bj,ij->AB', dux3, dux3, dl_cov[1:,1:])
            - np.einsum('Ai,Bj,aij,a->AB',
                        dux3, dux3, self.G4[:,1:,1:], self.l_cov)
        )
        q_inv = self.q_inv # q^AB
        Y = 0.5 * np.einsum('AB,AC,BD,CD', K_AB_l, q_inv, q_inv, K_AB_l)
        return Y

    def compute_div_s(self):
        r"""Compute the divergence div(s) of the rotation 1-form s_A."""
        # The quantity we compute is div(s) = D_A s^A = q^AB D_A s_B.
        # Here D_A is the covariant derivative compatible with the 2-metric q.
        calc = self.calc
        G2 = calc.christoffel() # A,B,C -> Gamma^A_BC (Christoffel on MOTS)
        # shape=(4,4,4), a,b,c -> partial_a partial_b n^c
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
        ddl = ddn + ddnu # shape=(4,4,4), a,b,c -> partial_a partial_b l^c
        ddux3 = calc.diff_xyz_wrt_laph(diff=2) # shape=(2,2,3), A,B,i -> partial_A partial_B x^i
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
        dk = self.dn - self.dnu # shape=(4,4), a,b -> partial_a k^b
        duk_cov = ( # shape=(2,4), A,a -> partial_A k_a
            np.einsum('Ai,iab,b->Aa', dux3, self.dg4[1:,:,:], self.k)
            + np.einsum('ab,Ai,ib->Aa', self.g4, dux3, dk[1:,:])
        )
        ds_cov = -0.5 * ( # shape=(2,2), A,B -> partial_A s_B
            np.einsum('Aa,Ba->AB', duk_cov, self.cov_A_l)
            + np.einsum('a,ABi,ia->AB', self.k_cov, ddux3, self.cov_a_l[1:,:])
            + np.einsum('a,Bi,Aia->AB', self.k_cov, dux3, du_cov_i_l)
        )
        Ds = (np.einsum('AB,AB', self.q_inv, ds_cov)
              - np.einsum('AB,CAB,C', self.q_inv, G2, self.torsion_1form))
        return Ds


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
                or self.get(l=m_max, m=m_max).real < val.real):
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
