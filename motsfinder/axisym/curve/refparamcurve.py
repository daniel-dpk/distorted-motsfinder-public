r"""@package motsfinder.axisym.curve.refparamcurve

Represents surfaces in local coordinates relative to a reference shape.


@b Examples

```
    # This is a simple curve in star-shaped parameterization.
    c0 = StarShapedCurve(CosineSeries([1.2, 0.2, 0.4, 0.1]),
                         metric=FlatThreeMetric)

    # Define an "offset function".
    from math import sin, cos, exp, pi
    func = lambda x: .3 * exp(cos(x)) * sin(x) * sin(5*x) - .2

    # Convert the offset function to a "horizon function" expression.
    f = CosineSeries.from_function(func, 30, domain=(0, pi))

    # Apply the offset using the reference parametrisation.
    curve = RefParamCurve(f, c0)
    curve.plot_curves((curve, 'curve'), (c0, 'ref', '--b'))
```
"""

import numpy as np
from scipy.linalg import norm
from scipy.optimize import newton as newton1d

from ...numutils import binomial_coeffs, inverse_2x2_matrix_derivative
from ...exprs.trig import CosineSeries
from .expcalc import ExpansionCalc
from .expcurve import ExpansionCurve


__all__ = [
    "RefParamCurve",
]


class RefParamCurve(ExpansionCurve):
    r"""Represent a curve as offset function relative to another curve.

    This class takes a reference curve and an *offset function* (i.e. the
    horizon function `h`) to represent a curve. The offset is multiplied by
    the (Euclidean) normal on the reference curve to determine the coordinates
    of the curve represented here, i.e. the curve is given by
    \f[
        \gamma(\lambda) = \gamma_R(\lambda) + h(\lambda) \nu_R(\lambda),
    \f]
    where \f$ \nu_R \f$ is the outward pointing normal vector on the reference
    curve. **Note** that this normal vector is not necessarily normalized.

    This class can compute geometric quantities like the expansion along the
    surface represented by this curve in axisymmetric spaces. The geometry of
    the spatial slice is specified by supplying a Riemannian 3-metric and
    optionally an extrinsic curvature when constructing this curve.

    See the examples in the package description of curve.refparamcurve.
    """

    def __init__(self, h, ref_curve, metric=None, freeze_ref_curve=True,
                 name=''):
        r"""Create a curve from a reference curve and a offset function.

        Note: The offset function is not guaranteed to be the Euclidean
              distance in coordinate space between the reference curve and the
              curve represented here. This depends on the length of the normal
              computed by the reference curve. It is better to think of `h` as
              a parameterized general coordinate *related* to the distance to
              the reference curve.

        @param h (exprs.numexpr.NumericExpression)
            Offset function, aka *horizon function* (see class description).
        @param ref_curve (basecurve.BaseCurve)
            Reference curve with respect to which `h` defines the curve.
        @param metric
            The Riemannian 3-metric of the slice. If not specified, the
            metric and extrinsic curvature is taken from the `ref_curve`. It
            is an error if in this case `ref_curve` is just a basic curve
            without metric information.
        @param freeze_ref_curve (boolean)
            Whether to assume the reference curve as being fixed such that
            evaluators can be frozen for efficiency. Default is `True`.
        @param name
            Name of this curve. This may be used when printing information
            about this curve or as label in plots.
        """
        if metric is None:
            metric = ref_curve.metric
        super(RefParamCurve, self).__init__(h, metric, name=name)
        self.ref_curve = ref_curve
        self._ref_curve_ev_override = None
        if freeze_ref_curve:
            self.ref_curve.freeze_evaluator()

    def __getstate__(self):
        with self.ref_curve.override_evaluator(None):
            return super(RefParamCurve, self).__getstate__()

    @staticmethod
    def from_curve(ref_curve, offset_coeffs=(), num=None, **kw):
        r"""Static method to create a curve from a reference curve.

        This method can be used to create a RefParamCurve from a given
        reference curve and optional coefficients for an offset curve.
        If no metric is given in the optional keyword arguments, it must be
        present on the `ref_curve`.

        The returned curve will be
        \f[
            \gamma(\lambda) = \gamma_R(\lambda)
                + \left(\sum_{k=0}^n a_k \cos(k\lambda)\right) \nu_R(\lambda),
        \f]
        where the \f$ a_k \f$ are the coefficients specified via
        `offset_coeffs`.

        @param ref_curve (basecurve.BaseCurve)
            Reference curve to construct the new curve with.
        @param offset_coeffs (iterable)
            Optional sequence of floats used as coefficients for a
            new exprs.trig.CosineSeries object. **Note** that since the normal
            on the reference curve is not required to have unit length, a
            constant offset cannot be achieved with just one coefficient here.
        @param num (int)
            Resolution of the newly created horizon function. The default is
            to take the current resolution of the reference curve. If smaller
            than the number of given `offset_coeffs`, trailing coefficients
            will be ignored.
        @param **kw
            Further keyword arguments supplied to the RefParamCurve
            constructor. This may be used to explicitly specify the metric
            and/or extrinsic curvature to use.
        """
        if num is None:
            num = ref_curve.num
        h = CosineSeries(offset_coeffs, domain=(0, np.pi))
        curve = RefParamCurve(h, ref_curve, **kw)
        curve.resample(num)
        return curve

    @staticmethod
    def from_star_shaped(star_shaped_curve, ref_curve, metric,
                         num=None, tol=1e-12):
        r"""Static method to create a curve from a star-shaped curve.

        This takes a curve in star-shaped parameterization (i.e. a
        starshapedcurve.StarShapedCurve object) and converts it to a
        RefParamCurve with the given `ref_curve` as reference curve.

        During the conversion, we need to do a numerical search for the
        offset function in normal direction at which we intersect with any
        point on the given `star_shaped_curve`. Therefore, some decline of
        accuracy is to be expected.

        @param star_shaped_curve (starshapedcurve.StarShapedCurve)
            The curve to reparameterize in reference curve parameterization.
        @param ref_curve (basecurve.BaseCurve)
            The reference curve with respect to which this curve should be
            represented.
        @param metric
            The Riemannian 3-metric defining the geometry of the surrounding
            space. You may explicitly set this to `None` to use the metric and
            extrinsic curvature from the `star_shaped_curve`.
        @param num (int)
            Resolution of the resulting curve. A higher value has the
            potential to lead to a more accurate modeling of the given
            `star_shaped_curve`, although the accuracy is also limited by the
            numerical search as detailed above. By default, the current
            resolution of the `star_shaped_curve` is used.
        @param tol (float)
            Tolerance for the numerical search described above. Default is
            `1e-12`.

        @b Notes

        The search is carried out as follows. For any fixed parameter value,
        we construct a line in normal direction from the reference curve at
        that parameter. For a point on this line, we compute the angle w.r.t.
        the `z`-axis, which must be the parameter value of the star-shaped
        curve if it were to intersect the line at that point. The Euclidean
        distance of the star-shaped curve at the computed angle to the point
        on the constructed line is then taken as the *error* function we wish
        to minimize. The minimum will be reached at a parameter value along
        the constructed line which must be the value of the horizon function
        for the initially fixed parameter value. Repeating this for many
        parameter values at the *collocation points* of the series we expand
        the horizon function into allows the computation of the coefficients
        of the series required to represent the curve.
        """
        if num is None:
            num = star_shaped_curve.num
        def _find_intersection(param):
            p = ref_curve(param)
            n = ref_curve.normal(param)
            def f(t):
                x = p + t*n
                r = norm(x)
                ta = np.arccos(x[1]/r)
                target = star_shaped_curve(ta)
                return norm(target - x)
            d = newton1d(f, x0=-0.2, tol=tol)
            # Note: d will be the correct `parameter` value for the horizon
            #       function. It is not necessarily the distance.
            return d
        with star_shaped_curve.fix_evaluator(), ref_curve.fix_evaluator():
            h = CosineSeries.from_function(
                _find_intersection, num=num, domain=(0, np.pi)
            )
        if metric is None:
            metric = star_shaped_curve.metric
        curve = RefParamCurve(h, ref_curve, metric=metric)
        return curve

    def h_diff(self, param):
        return self.ref_curve.normal(param, xyz=True)

    def copy(self):
        return type(self)(h=self.h.copy(), ref_curve=self.ref_curve,
                          metric=self.metric, name=self.name)

    def _create_calc_obj(self, param):
        return _RefParamExpansionCalc(
            curve=self, h_fun=self._get_evaluators(), param=param,
            metric=self.metric,
        )

    def __call__(self, param, xyz=False):
        p = self.ref_curve(param)
        n = self.ref_curve.normal(param)
        h = self._get_evaluators()(param)
        x, z = p + h*n
        return self._prepare_result(x, z, xyz)

    def _diff(self, param, diff):
        r"""Compute derivatives of the component functions.

        This is implemented to compute derivatives of the `x` and `z`
        component functions up to arbitrary orders using
        \f[
            \partial_n \gamma(\lambda) = \partial_n \gamma_R(\lambda)
                + \sum_{k=0}^n {n \choose k} h^{(n-k)}(\lambda) \nu_R^{(k)}(\lambda),
        \f]
        where `n==diff` and `lambda==param`.
        """
        ref_curve = self.ref_curve
        h = self._get_evaluators()
        coeffs = binomial_coeffs(diff)
        return ref_curve.diff(param, diff=diff) + np.sum(
            [coeffs[k] * h.diff(param, diff-k) * ref_curve.normal(param, k)
             for k in range(diff+1)],
            axis=0
        )

    def find_relative_coords(self, point, **kw):
        r"""Find the (d,lambda) coordinates of a point in (x,y,z) coordinates.

        This method performs a numerical search for the coordinates relative
        to the current reference surface of the given point.

        The returned parameter will be in the range `(-pi/2, 3*pi/2)`.
        Allowing the parameter to go slightly outside the actual domain of
        `(0,pi)` ensures we get correct values usable in FD differentiation
        should we step slightly into negative `x` values.

        The keyword arguments are passed to the closest_point() function and
        may e.g. contain an initial parameter `start_param` to start the
        search from.
        """
        x, z = self.rotate_to_xz_plane(point)
        param = self.ref_curve.closest_point((x, z), **kw)
        p = self.ref_curve(param)
        n = self.ref_curve.normal(param)
        d = np.sqrt((x-p[0])**2 + (z-p[1])**2) / norm(n)
        if n.dot([x, z] - p) < 0:
            d = -d
        if point[0] < 0:
            param = -param if z >= 0 else 2*np.pi-param
        return d, param


class _RefParamExpansionCalc(ExpansionCalc):
    r"""Expansion cache/calculator for RefParamCurve objects.

    This computes and caches interim results of the expansion and its
    functional derivatives w.r.t. the horizon function for curves in reference
    parameterization.

    This object is usually only created by the RefParamCurve object.
    """

    def _compute_s_ds_X_Y(self):
        r"""Compute the terms we need to compute the expansion.

        To compute the covariant normal \f$ s_i \f$ on the curve represented
        by the RefParamCurve, we define the function
        \f[
            F(\vec x) = d(\vec x) - h(\lambda(\vec x)),
        \f]
        where \f$ (d,\lambda) \f$ are the coordinates of a neighbourhood of
        the reference curve defining the parameter (\f$\lambda\f$) and
        distance factor (\f$\lambda\f$) in normal direction of the point. This
        means that on the surface defined by `h` we will have `F = 0`. The
        covariant normal is therefore obtained via
        \f[
            s_i = \nabla_i F = \partial_i F = \partial_i d - X_i h',
        \f]
        where we use the notation \f$ X_i = \partial_i \lambda \f$ introduced
        by Thornburg [1].
        For the formulas in [1], we furthermore need the derivative
        \f[
            \partial_i s_j = \partial_i \partial_j d - X_{ij}h' - X_i X_j h'',
        \f]
        where we set \f$ X_{ij} = \partial_i\partial_j\lambda \f$.

        Note that since we cannot differentiate between variables based on how
        many indices they have, we use `Y` for the components \f$ X_{ij} \f$.

        @b References

        [1] Thornburg, Jonathan. "A fast apparent horizon finder for
            three-dimensional Cartesian grids in numerical relativity."
            Classical and quantum gravity 21.2 (2003): 743.
        """
        dh, ddh = self.dh, self.ddh
        d1, X = self.coord_derivs(diff_order=1)
        d2, Y = self.compute_d2_Y()
        s = d1 - X * dh
        ds = d2 - Y * dh - np.outer(X, X) * ddh
        return s, ds, X, Y

    def _compute_d2_Y(self):
        return self.coord_derivs(diff_order=2)

    def _compute_d3_Z(self):
        return self.coord_derivs(diff_order=3)

    def _compute_dds_Z(self):
        r"""Compute second derivatives of the normal and third ones of lambda.

        This computes \f$\partial_i\partial_j s_k\f$ and
        \f$Z := X_{ijk} = \partial_i\partial_j\partial_k \lambda\f$.

        @return Two elements, the first containing the derivatives of the
            non-normalized covariant normal `s` and the second those of the
            parameter \f$\lambda\f$.

        @b Notes

        To compute the third derivatives, we continue from the formulas in
        _compute_s_ds_X_Y() and obtain
        \f[
            \partial_i\partial_j s_k
                = d_{ijk} - X_{ijk} h'
                  - (X_{ij}X_k + X_{jk}X_i + X_{ki}X_j) h''
                  - X_i X_j X_k h''',
        \f]
        where \f$d_{ijk} := \partial_i \partial_j \partial_k d\f$.
        """
        dh, ddh = self.dh, self.ddh
        d3h = self.h_fun.diff(self.param, n=3)
        d3, Z = self.compute_d3_Z()
        X, Y = self.X, self.Y
        n = 3
        YijXk = np.outer(Y, X).reshape(n, n, n)
        def YX(i,j,k):
            return np.moveaxis(YijXk, [0,1,2], [i,j,k])
        dds = (
            d3 - Z * dh
            - (YijXk + YX(1,2,0) + YX(2,0,1)) * ddh
            - np.outer(np.outer(X, X), X).reshape(n, n, n) * d3h
        )
        return dds, Z

    def coord_derivs(self, diff_order=1):
        r"""Compute derivatives of `d(x,y,z)` and `lambda(x,y,z)`.

        @param diff_order
            Derivative order. Either `1` for all first derivatives, `2` for
            the second derivatives or `3` for, well, the *third* partial
            derivatives. Default is `1`.

        @return A 2-tuple of the derivatives of `d` and `lambda`. For
            `diff_order==1`, each will be a numpy array with 3 elements
            corresponding to the derivatives in `x`, `y`, and `z`-direction.
            In case `diff_order==2`, both elements will be numpy 3x3 matrices
            containing all combinations of derivatives
            \f$ \partial_i\partial_j u \f$, where \f$u\f$ is either `d` or
            `lambda`.
            For `diff_order==3`, both elements will be 3x3x3 'matrices' with
            indices `d3u[i,j,k]` corresponding to
            \f$\partial_i \partial_j \partial_k u\f$.

        @b Notes

        The curves represent surfaces in axisymmetry. In other words, they
        live in the \f$(\rho, z)\f$ coordinates, where
        \f$\rho = \sqrt{x^2+y^2}\f$. The parameters `d` and `lambda` can be
        treated as a local coordinate system in a neighbourhood of the
        reference curve. We therefore have a coordinate transformation from
        the \f$(\rho,z)\f$ coordinates to the \f$(d,\lambda)\f$ coordinates.
        The Jacobian matrix of this transformation,
        \f[
            J_{(\rho,z)} =
            \left(\begin{array}{@{}cc@{}}
                \partial_d \rho & \partial_\lambda \rho\\
                \partial_d z & \partial_\lambda z
            \end{array}\right)
                =
            \left(\begin{array}{@{}cc@{}}
                (\nu_R)_x & (\gamma'_R)_x + d\ (\nu'_R)_x\\
                (\nu_R)_z & (\gamma'_R)_z + d\ (\nu'_R)_z
            \end{array}\right)
                =:
            \left(\begin{array}{@{}cc@{}}
                \tilde a & \tilde b\\
                \tilde c & \tilde d
            \end{array}\right)
        \f]
        may be inverted to obtain the Jacobian of the inverse transformation,
        i.e. we get
        \f[
            J_{(d,\lambda)}(\rho,z) = (J_{(\rho,z)})^{-1}
                =
            \left(\begin{array}{@{}cc@{}}
                \partial_\rho d & \partial_z d\\
                \partial_\rho \lambda & \partial_z \lambda
            \end{array}\right).
        \f]
        The second and third derivatives are then obtained from first and
        second derivatives of the Jacobian \f$J_{(\rho,z)}\f$
        which are obtained component-wise using
        \f{eqnarray*}{
            \tilde a_i &=& X_i\ (\nu'_R)_x \\
            \tilde b_i
                &=& X_i\ \big[(\gamma_R'')_x + d (\nu_R'')_x\big]
                    + d_i\ (\nu'_R)_x \\
            \tilde c_i &=& X_i\ (\nu'_R)_z \\
            \tilde d_i
                &=& X_i\ \big[(\gamma_R'')_z + d (\nu_R'')_z\big]
                    + d_i\ (\nu'_R)_z,
        \f}
        and
        \f{eqnarray*}{
            \tilde a_{ij} &=& X_{ij} (\nu_R')_x + X_i X_j (\nu_R'')_x,
            \\
            \tilde b_{ij} &=&
                X_{ij} ((\gamma_R'')_x + d\;(\nu_R'')_x)
                + X_i X_j ((\gamma_R''')_x + d\;(\nu_R''')_x)
            \\&&
                + (\nu_R'')_x (X_i d_j + X_j d_i)
                + d_{ij} (\nu_R')_x,
        \f}
        where \f$ d_i := \partial_i d \f$ and \f$X_i := \partial_i \lambda\f$.
        The derivatives \f$\tilde c_{ij}\f$ and \f$\tilde d_{ij}\f$ are
        analogous to those of `a` and `b`, respectively.

        As a last step, we convert the derivatives w.r.t. `rho` back to
        derivatives w.r.t. `x` and `y` using the chain rule. To do this, note
        that `d` and `lambda` are functions of `rho` and `z` only, even when
        considering them in 3-dimensions with coordinates \f$(\rho,\varphi,z)\f$.
        Now, for any function \f$f = f(\rho, z)\f$, we have
        \f{eqnarray*}{
            \partial_x f
                &=& \frac{\partial \rho}{\partial x}
                  \frac{\partial f}{\partial \rho}
                = \frac{x}{\rho} f
                \stackrel{\varphi=0}{=} \partial_\rho f,
            \\
            \partial_y f &=& \frac{y}{\rho} \partial_\rho f
                \stackrel{\varphi=0}{=} 0,
            \\
            \partial_x^2 f
                &\stackrel{\varphi=0}{=}& \partial_\rho^2 f,
            \\
            \partial_x\partial_y f
                &\stackrel{\varphi=0}{=}& 0,
            \\
            \partial_y^2 f
                &\stackrel{\varphi=0}{=}& \frac{\partial_\rho f}{\rho}
                = \frac{\partial_x f}{x},
            \\
            \partial_x\partial_y^2 f
                &\stackrel{\varphi=0}{=}&
                \frac{\partial_x^2 f}{x} - \frac{\partial_x f}{x^2}.
        \f}
        All `z`-derivatives can be absorbed in `f` in the above formulas and
        all the remaining derivatives involving ones w.r.t. `y` and not
        contained above vanish.
        """
        vc = lambda *xi: np.array(xi) # easily create numpy arrays/matrices
        la0 = self.param
        d0 = self.h
        ref_curve = self.curve.ref_curve
        dp = ref_curve.tangent(la0)
        n = ref_curve.normal(la0)
        dn = ref_curve.normal(la0, diff=1)
        # J_xz = Jacobian of transform (x,z) -> (d,lambda)
        ja, jc = n
        jb, jd = dp + d0 * dn
        J_xz = [[ja, jb], [jc, jd]]
        d1, X = inverse_2x2_matrix_derivative(J_xz, diff=0)
        d1_xyz = self.insert_y_derivatives(d1, 1)
        X_xyz = self.insert_y_derivatives(X, 1)
        if diff_order == 1:
            return d1_xyz, X_xyz
        ddp = ref_curve.diff(la0, diff=2)
        ddn = ref_curve.normal(la0, diff=2)
        # derivatives of ja, jb, jc, jd matrix elements
        dja = X * dn[0]
        djc = X * dn[1]
        djb = X * (ddp[0] + d0*ddn[0]) + d1 * dn[0]
        djd = X * (ddp[1] + d0*ddn[1]) + d1 * dn[1]
        dJ_xz = [[[dja[i], djb[i]], [djc[i], djd[i]]] for i in range(2)]
        # \partial_i J_(d, lambda)
        dJ_dl = inverse_2x2_matrix_derivative(J_xz, dJ_xz, diff=1)
        # Current axes of dJ_dl: [partial_i][u^j][partial_k]
        # However, we want:      [u^i][partial_j][partial_k]
        d2, Y = dJ_dl = dJ_dl.swapaxes(0, 1)
        d2_xyz = self.insert_y_derivatives(d2, 2, d1_xyz)
        Y_xyz = self.insert_y_derivatives(Y, 2, X_xyz)
        if diff_order == 2:
            return d2_xyz, Y_xyz
        d3p = ref_curve.diff(la0, diff=3)
        d3n = ref_curve.normal(la0, diff=3)
        ddja, ddjc = (
            Y * dn[i] + np.outer(X, X) * ddn[i]
            for i in range(2)
        )
        ddjb, ddjd = (
            Y * (ddp[i] + d0*ddn[i])
            + np.outer(X, X) * (d3p[i] + d0*d3n[i])
            + (np.outer(X, d1) + np.outer(d1, X)) * ddn[i]
            + d2 * dn[i]
            for i in range(2)
        )
        ddJ_xz = np.array([[ddja, ddjb], [ddjc, ddjd]])
        # currently: ddJ_xz[i,j,k,l] = \partial_k \partial_l J_ij
        # we want:   ddJ_xz[i,j,k,l] = \partial_i \partial_j J_kl
        ddJ_xz = np.moveaxis(ddJ_xz, [0,1,2,3], [2,3,0,1])
        ddJ_dl = inverse_2x2_matrix_derivative(J_xz, dJ_xz, ddJ_xz, diff=2)
        # axes: [partial_i, partial_j, u^k, partial_l]
        # need: [u^i, partial_j, partial_k, partial_l]
        d3, Z = ddJ_dl.swapaxes(0, 2)
        d3_xyz = self.insert_y_derivatives(d3, 3, d1_xyz, d2_xyz)
        Z_xyz = self.insert_y_derivatives(Z, 3, X_xyz, Y_xyz)
        if diff_order == 3:
            return d3_xyz, Z_xyz
        raise NotImplementedError("Derivative order not implemented: %s"
                                  % diff_order)

    def insert_y_derivatives(self, f, diff_order, *prev_diffs):
        r"""Convert partial derivatives \wrt rho to x- and y-derivatives.

        The formulas used here and their derivation can be found at the end of
        the docstring of coord_derivs().

        @param f
            Current derivative matrix with ``diff_order`` axes, one for each
            partial derivative. We expect each axis to have exactly two
            elements corresponding to `rho` and `z` derivatives, respectively.
        @param diff_order
            Order of derivatives in `f`.
        @param *prev_diffs
            Lower order partial derivatives of `f` used to construct the `y`
            derivatives. These need to be given in ascending derivative order,
            starting at `1` (i.e. we don't need/expect the non-differentiated
            values). Note that they should contain `y` derivatives, i.e. each
            axis should have 3 elements.

        @return The input `f` with `y` derivatives inserted at the correct
            places. The input array `f` is not modified.
        """
        f = np.asarray(f)
        prev_diffs = list(map(np.asarray, prev_diffs))
        x = self.point[0]
        for axis in range(diff_order):
            # this guarantees an independent copy is created
            f = np.insert(f, 1, 0.0, axis=axis)
        if diff_order == 1:
            pass # single y-derivatives are zero
        elif diff_order == 2:
            df = prev_diffs[0]
            f[1,1] = df[0]/x
        elif diff_order == 3:
            df, ddf = prev_diffs
            f[0,1,1] = f[1,0,1] = f[1,1,0] = ddf[0,0]/x - df[0]/x**2
            f[1,1,1] = 0.0
            f[2,1,1] = f[1,2,1] = f[1,1,2] = ddf[0,2]/x
        else:
            raise NotImplementedError
        return f
