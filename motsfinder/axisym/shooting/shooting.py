r"""@package motsfinder.axisym.shooting.shooting

Find axisymmetric MOTSs and MOTOSs using a shooting method.


@b Examples

```
    from motsfinder.axisym.shooting import shooting_method
    from motsfinder.metric import BrillLindquistMetric
    from motsfinder.ipyutils import plot_data

    g = BrillLindquistMetric(d=0.6, m1=0.2, m2=0.8)
    sol = shooting_method(
        metric=g, verbose=True,
        z0=0.44462406985175074,  # outer common MOTS (S_outer)
        #z0=0.3740659586129742,   # inner common MOTS (S_inner)
        #z0=0.36239042407655525,  # smaller individual MOTS (S_1)
        #z0=0.005761670102647809, # larger individual MOTS (S_2)
        #z0=0.4441263027,         # MOTS that wraps around
        #z0=0.445, max_domain=2,  # a MOTOS
    )
    S = sol.t
    P, Z, _, _ = sol.y

    plot_data(
        P, Z, figsize=(3.6, 3.6), dpi=120, xlabel="$x/M$", ylabel="$z/M$",
        subplot_kw=dict(aspect="equal"), pad="20%",
    )
```
"""

import numpy as np
from scipy.integrate import solve_ivp

from ...numutils import clip


__all__ = [
    "shooting_method",
    "get_termination_reason",
]


def shooting_method(metric, z0, x0=0.0, P_dot=1.0, Z_dot=0.0, s_max=np.inf,
                    axis_tol=1e-5, max_domain=10.0, rtol=1e-8, atol=1e-12,
                    method="RK45", eps=1e-6, bad_metric=False, normal="left",
                    verbose=False, unload=True, **kw):
    r"""Solve the MOT(O)S problem using a shooting method.

    Given a metric and initial conditions, this function integrates a system
    of ODEs to construct a curve representing a surface of zero outward
    expansion. If the curve starts and terminates on the z-axis (or closes
    upon itself away from the axis), it represents a closed surface, i.e. a
    MOTS (if it closes away from the z-axis, the surface has toroidal
    topology). Otherwise, the surface is open, i.e. a MOTOS.

    Integration is stopped when the z-axis is hit or when a given maximum
    distance from the coordinate origin is reached.

    Note that the current implementation assumes vanishing spin.

    @param metric
        The metric object which contains methods to obtain the slice 3-metric
        and extrinsic curvature.
    @param z0
        Coordinate on the symmetry axis (z-axis) at which to start the ray.
    @param x0
        x-coordinate of the starting point. Should be 0 for curves starting
        from the z-axis. Default is 0.
    @param P_dot,Z_dot
        Initial direction to shoot in (in the x,z coordinate plane). The given
        values will be normalized to a unity tangent vector. The direction
        should be perpendicular to the z-axis if the curve starts on the
        z-axis. This is also the default, i.e. ``P_dot=1.0, Z_dot=0.0``.
    @param s_max
        Maximal proper length of the curve. Default is to not limit the proper
        length and integrate until one of the other termination conditions is
        reached.
    @param axis_tol
        Tolerance for detecting termination when hitting the z-axis. A curve
        starting and ending at the z-axis represents a smooth closed surface
        (i.e. a MOTS). However, numerically the axis will not be reached
        exactly before small numerical noise leads to the curve turning away
        from the axis. If the curve gets closer than the given tolerance
        (default is ``1e-5``) in its x-coordinate, then a curve that actually
        terminates on the axis most likely exists in a small neighborhood of
        the found curve.
    @param max_domain
        Maximum coordinate distance from the origin in the (x,z)-coordinate
        plane the curve can travel. Integration is stopped as soon as the
        curve crosses this boundary. Default is `10.0`.
    @param rtol,atol
        Tolerances for the ``scipy.integrate.solve_ivp()`` call. Default is
        ``rtol=1e-8, atol=1e-12``. See the `SciPy` documentation for more
        information.
    @param method
        Method to use in the ``scipy.integrate.solve_ivp()`` call. Possible
        options depend on the installed `SciPy` version. Default is
        ``"RK45"``.
    @param eps
        Closest distance to the z-axis to consider. This function currently
        does not handle the case of `x` going to zero. We hence cut off the
        x-value at `eps` (default `1e-6`) and assume the error is negligible.
        If you get errors due to `NaN` values during integration, this value
        might have to be increased. Correctly handling this limit is planned.
    @param bad_metric
        Whether the metric cannot be evaluated closer than `eps` to the
        z-axis. Default is `False`, i.e. we may evaluate the metric and
        extrinsic curvature on the z-axis.
    @param normal
        The normal which should have vanishing expansion. Poissible are
        ``"left"`` (default) and ``"right"``.
    @param verbose
        Whether to print information on why integration was stopped, even in
        case of success. Default is `False`.
    @param unload
        Whether to unload any loaded numerical data from the metric after
        execution to free up memory. Default is `True`.
    @param **kw
        Further arguments are passed to ``scipy.integrate.solve_ivp()``
        without modification. Relevant arguments include e.g. ``max_step``.
    """
    try:
        return _shooting_method(
            metric=metric, z0=z0, x0=x0, P_dot=P_dot, Z_dot=Z_dot,
            s_max=s_max, axis_tol=axis_tol, max_domain=max_domain, rtol=rtol,
            atol=atol, method=method, eps=eps, bad_metric=bad_metric,
            normal=normal, verbose=verbose, **kw
        )
    finally:
        if unload:
            try:
                metric.unload_data()
            except AttributeError:
                pass


def _shooting_method(metric, z0, x0, P_dot, Z_dot, s_max, axis_tol,
                     max_domain, rtol, atol, method, eps, bad_metric, normal,
                     verbose, **kw):
    r"""Implements shooting_method()."""
    if normal == "left":
        ku_sgn = 1
    elif normal == "right":
        ku_sgn = -1
    else:
        raise ValueError("Unknown normal: %s" % (normal,))
    if eps > axis_tol:
        raise ValueError("`eps <= axis_tol` required to not cut off domain "
                         "before axis detection")
    curv = metric.get_curv()
    s0 = 0.0
    T0 = np.array([P_dot, 0.0, Z_dot], dtype=np.float)
    T0 = T0 / metric.at([max(eps, x0) if bad_metric else x0, 0.0, z0]).norm(T0)
    if abs(x0) < eps and abs(Z_dot) < 1e-12: # handle shooting off the z-axis
        if verbose:
            print("Handling departure from z-axis...")
        s0, x0, z0, T0[0], T0[2] = _handle_departure_from_axis(
            metric=metric, curv=curv, z0=z0, x0=x0, eps=eps,
            bad_metric=bad_metric, normal=normal,
            T=np.asarray([T0[0], T0[2]]), verbose=verbose,
        )
    y0 = np.array([x0, z0, T0[0], T0[2]])
    xmin = zmin = -np.inf
    xmax = zmax = np.inf
    try:
        (xmin, xmax), _, (zmin, zmax) = metric.safe_domain
    except AttributeError:
        pass
    def fun(s, y):
        P, Z, dP, dZ = y
        P = max(eps, P) # be sure to keep this distance to the z-axis
        P = clip(P, xmin, xmax)
        Z = clip(Z, zmin, zmax)
        X1 = dP
        X2 = dZ
        T = np.array([dP, dZ])     # vector T^a
        x = np.array([P, 0.0, Z])  # x,y,z coordinates of current point
        h = metric.at(x).mat       # 3-metric at current point in x,y,z coords
        R = x[0] * np.sqrt(h[1,1])
        G3 = metric.christoffel(x) # 3-Christoffel symbols
        G2 = G3[::2, ::2, ::2]     # remove all y-components
        det_g_bar = h[0,0] * h[2,2] - h[0,2]**2
        sdg = np.sqrt(det_g_bar)
        Nrho = -1/sdg * (h[0,2] * dP + h[2,2] * dZ)
        Nz = 1/sdg * (h[0,0] * dP + h[0,2] * dZ)
        N = np.array([Nrho, Nz]) # vector N^a
        K = np.zeros((3, 3)) if curv is None else curv(x)
        K = -K # numerics convention is different from mathematician's convention
        K2 = K[::2, ::2] # (rho, z) coordinates only
        # Note that k_phiphi is defined in the paper via K_phiphi = R^2 k_phiphi.
        # In the x-z-plane, we have R^2 = x^2 g_yy and K_phiphi = x^2 K_yy,
        # hence k_phiphi = K_phiphi / g_yy.
        k_phiphi = K[1,1] / h[1,1]
        ku = ku_sgn * (K2.dot(T).dot(T) + k_phiphi)
        dh = metric.diff(x)
        dR = np.array([
            np.sqrt(h[1,1]) + x[0]/(2*np.sqrt(h[1,1])) * dh[0,1,1],
            x[0]/(2*np.sqrt(h[1,1])) * dh[2,1,1]
        ])
        kappa = N.dot(dR) / R + ku
        fa = (
            - np.einsum('abc,b,c->a', G2, T, T) # pylint: disable=invalid-unary-operand-type
            + kappa * N
        )
        dy = (X1, X2, fa[0], fa[1])
        if np.isnan(dy).any():
            raise ShootingMethodError(
                "NaNs occurred in evaluation. "
                "Are we too close to the z-axis? "
                "Try increasing `eps` to a value > %s." % eps
            )
        return dy
    def axis_reached(s, y):
        P = y[0]
        return P - axis_tol
    axis_reached.terminal = True
    axis_reached.direction = -1
    def domain_left(s, y):
        P, Z = y[0], y[1]
        return np.sqrt(P**2 + Z**2) - max_domain
    domain_left.terminal = True
    domain_left.direction = 1
    sol = solve_ivp(
        fun, t_span=[s0, s_max], y0=y0, method=method,
        events=[axis_reached, domain_left],
        rtol=rtol, atol=atol, **kw,
    )
    if verbose:
        print(sol.message)
        if sol.status == 1:
            axis_events = sol.t_events[0]
            domain_events = sol.t_events[1]
            if len(axis_events):
                print("Curve hit the axis.")
            if len(domain_events):
                print("Curve left the coordinate domain radius of %s."
                      % max_domain)
    return sol


def _handle_departure_from_axis(metric, curv, x0, z0, eps, bad_metric, normal,
                                T, verbose=False):
    x = np.asarray([x0, 0.0, z0])
    dP, dZ = T
    if bad_metric:
        x[0] = max(eps, x[0])
    ku_sgn = 1 if normal == "left" else -1
    h = metric.at(x).mat
    dh = metric.diff(x)
    #ddh = metric.diff(x, diff=2)
    P1 = 1/np.sqrt(h[0,0])
    G3 = metric.christoffel(x)
    P2 = -G3[0,0,0] / h[0,0]
    #P2 = 0.0
    K = np.zeros((3, 3)) if curv is None else curv(x)
    K = -K # numerics convention
    k_phiphi = K[1,1] / h[1,1]
    dR = np.array([
        np.sqrt(h[1,1]) + x[0]/(2*np.sqrt(h[1,1])) * dh[0,1,1],
        0.0 # = x[0]/(2*np.sqrt(h[1,1])) * dh[2,1,1]
    ])
    R_T = np.einsum('a,a->', T, dR)
    G2 = G3[::2, ::2, ::2]
    shyy = np.sqrt(h[1,1])
    #Rxx = dh[0,1,1] / shyy - x[0] * dh[0,1,1]**2 / (4*shyy**3) + x[0] * ddh[0,0,1,1] / (2*shyy)
    Rxx = dh[0,1,1] / shyy # + 0
    # Can we prove Rxx = 0, that is, \partial_x h_{yy} = 0?
    #Rxz = dh[2,1,1] / (2*shyy) - x[0] * dh[0,1,1] * dh[2,1,1] / (4*shyy**3) + x[0]*ddh[0,2,1,1] / (2*shyy)
    Rxz = dh[2,1,1] / (2*shyy) # + 0
    #Rzz = -x[0] * dh[2,1,1]**2 / (4*shyy**3) + x[0] * ddh[2,2,1,1] / (2*shyy)
    Rzz = 0.0
    ddR = np.asarray([[Rxx, Rxz],
                      [Rxz, Rzz]])
    DDR = ddR - np.einsum('cab,c->ab', G2, dR)
    det_g_bar = h[0,0] * h[2,2] - h[0,2]**2
    sdg = np.sqrt(det_g_bar)
    Nrho = -1/sdg * (h[0,2] * dP + h[2,2] * dZ)
    Nz = 1/sdg * (h[0,0] * dP + h[0,2] * dZ)
    N = np.array([Nrho, Nz])
    R_TN = np.einsum('a,b,ab->', T, N, DDR)
    kappa0 = 1.0/2.0 * (R_TN / R_T + ku_sgn * (K[0,0] / h[0,0] + k_phiphi))
    Z2 = - G3[2,0,0] / h[0,0] + kappa0 / np.sqrt(h[2,2])
    # We want P = P1 * s0 + 1.0/2.0 * P2 * s0**2 = eps
    if P2 != 0.0:
        #s0 = - P1/P2 + np.sqrt((P1/P2)**2 + 2*eps/P2) # numerically unstable
        s0_1 = (- P1 - np.sqrt(P1**2 + 2*eps*P2)) / P2
        s0 = -2*eps / (P2 * s0_1) # Vieta
    else:
        s0 = eps / P1
    P_new = P1 * s0 + 1.0/2.0 * P2 * s0**2
    Z_new = z0 + 1.0/2.0 * Z2 * s0**2
    dP_new = P1 + s0 * P2
    dZ_new = s0 * Z2
    T0_new = np.array([dP_new, 0.0, dZ_new], dtype=np.float)
    T0_new = T0_new / metric.at([P_new, 0.0, Z_new]).norm(T0_new)
    dP_new, dZ_new = T0_new[0], T0_new[2]
    if verbose:
        print("  Initial conditions changed:")
        print("    z0    += %.16e" % (Z_new-z0))
        print("    x0    += %.16e" % (P_new-x0))
        print("    P_dot += %.16e" % (dP_new-dP))
        print("    Z_dot += %.16e" % (dZ_new-dZ))
    return s0, P_new, Z_new, dP_new, dZ_new


def get_termination_reason(sol):
    r"""Return a string indicating why the shooting method terminated.

    Possible reasons are:

        * ``"axis_hit"``: When the curve hit the axis.
        * ``"domain_reached"``: When the curve left the specified domain.
        * ``"end"``: `s_max` reached.
        * ``"failure"``: Integration step failed.
        * ``"other"``: Unknown reason.
    """
    if sol.status == 1:
        axis_events = sol.t_events[0]
        domain_events = sol.t_events[1]
        if len(axis_events):
            return "axis_hit"
        if len(domain_events):
            return "domain_reached"
    if sol.status == 0:
        return "end"
    if sol.status == -1:
        return "failure"
    return "other"


class ShootingMethodError(Exception):
    r"""Raised when an error occurs while integrating the coupled ODEs."""
    pass
