r"""@package motsfinder.metric.helpers

Utility functions for use in metric classes.

These utilities are used in computing e.g. Christoffel symbols or components
of the Riemann tensor.


@b Examples

```
    # Create a Brill-Lindquist metric.
    metric = BrillLindquistMetric(m1=1, m2=1, d=1.4, axis='z')

    # Define the point at which to evaluate the metric.
    point = np.array([1.5, 0.2, 0.3])

    # Get the metric tensor and its derivatives at a certain point.
    point = np.array([1.5, 0.2, 0.3])
    g = metric.at(point)
    dg = metric.diff(point)

    # Compute all the Christoffel symbols.
    G = christoffel_symbols(g.inv, dg)

    # Print a few of them.
    print_indented("Gamma^0_11 = ", G[0,1,1])
    print_indented("Gamma^1_bc = ", G[0,:,:])
```

Continuing from above, we can obtain components of the Riemann curvature
tensor.

```
    dg_inv = metric.diff(point, inverse=True)
    ddg = metric.diff(point, diff=2)
    dG = christoffel_deriv(g.inv, dg_inv, dg, ddg)
    ra = range(3)
    Ra_bcd = [[[[riemann_components(G, dG, a, b, c, d)
                 for d in ra] for c in ra] for b in ra] for a in ra]
    Ra_bcd = np.asarray(Ra_bcd)

    # Print the coordinate representation of the endomorphism (i.e. the
    # matrix) of R^a_b01.
    print_indented("R^a_b01 = ", Ra_bcd[:,:,0,1])

    # We get the purely covariant Riemann tensor by lowering the first index.
    R_abcd = np.einsum('ae,ebcd', g.mat, Ra_bcd)

    # Verify some of the symmetries.
    print("R_0012 = %s" % R_abcd[0,0,1,2])
    print("R_1200 = %s" % R_abcd[1,2,0,0])
    print("R_0121 - R_2101 = %s" % (R_abcd[0,1,2,1] - R_abcd[2,1,0,1]))
    print("R_0101 + R_0110 = %s" % (R_abcd[0,1,0,1] + R_abcd[0,1,1,0]))
    print("R_0101 + R_1001 = %s" % (R_abcd[0,1,0,1] + R_abcd[1,0,0,1]))
```
"""

from math import fsum

import numpy as np


__all__ = [
    "christoffel_symbols",
    "christoffel_deriv",
    "riemann_components",
]


def christoffel_symbols(g_inv, dg):
    r"""Compute the Christoffel symbols of a metric.

    This computes
    \f[
        \Gamma^c_{ab} = \frac{1}{2} g^{cd} \big(
                - \partial_d g_{ab}
                + \partial_a g_{bd}
                + \partial_b g_{da}
            \big),
    \f]
    where `a,b,c` each run over `0, ..., n-1`. Here `n` is the dimension of
    the manifold on which the metric lives.

    @param g_inv
        NumPy array containing the elements of the inverse metric at the point
        where the Christoffel symbols should be computed. The dimension of the
        manifold is inferred from this array (i.e. from the number of rows).
    @param dg
        NumPy array of the first derivatives of the metric. Indices `g[a,b,c]`
        should have the meaning \f$\partial_a g_{bc}\f$.

    @return NumPy array with indices `[a,b,c]` corresponding to
        \f$\Gamma^a_{bc}\f$.
    """
    dim = g_inv.shape[0]
    ra = range(dim)
    def _G(c,a,b):
        return 0.5 * fsum(
            g_inv[c,d] * (-dg[d,a,b] + dg[a,b,d] + dg[b,d,a])
            for d in ra
        )
    return np.array(
        [[[_G(c,a,b) for b in ra] for a in ra] for c in ra]
    )


def christoffel_deriv(g_inv, dg_inv, dg, ddg):
    r"""Compute the derivatives of the Christoffel symbols.

    This computes
    \f{eqnarray*}{
        \partial_e \Gamma^c_{ab} &=&
            \frac{1}{2} (\partial_e q^{cd}) \big(
                - \partial_d q_{ab}
                + \partial_a q_{bd}
                + \partial_b q_{da}
            \big)
            \\&&
            + \frac{1}{2} q^{cd} \big(
                - \partial_e\partial_d q_{ab}
                + \partial_e\partial_a q_{bd}
                + \partial_e\partial_b q_{da}
            \big).
    \f}

    @param g_inv
        Inverse metric components as an `(n, n)` matrix.
    @param dg_inv
        Derivatives of `g_inv` with indices `dg_inv[a,b,c]` corresponding to
        \f$\partial_a g^{bc}\f$.
    @param dg
        Derivatives of the metric with indices `dg[a,b,c]` corresponding to
        \f$\partial_a g_{bc}\f$.
    @param ddg
        Second derivatives of the metric with indices `ddg[a,b,c,d]`
        corresponding to \f$\partial_a \partial_b g_{cd}\f$.

    @return NumPy array with indices `[a,b,c,d]` corresponding to
        \f$\partial_a\Gamma^b_{cd}\f$.
    """
    dim = g_inv.shape[0]
    ra = range(dim)
    def _dG(e,c,a,b):
        term1 = 0.5 * fsum(
            dg_inv[e,c,d] * (-dg[d,a,b] + dg[a,b,d] + dg[b,d,a])
            for d in ra
        )
        term2 = 0.5 * fsum(
            g_inv[c,d] * (-ddg[e,d,a,b] + ddg[e,a,b,d] + ddg[e,b,d,a])
            for d in ra
        )
        return term1 + term2
    return np.array(
        [[[[_dG(e,c,a,b) for b in ra] for a in ra] for c in ra] for e in ra]
    )


def riemann_components(G, dG, a, b, c, d):
    r"""Compute a component of the Riemann tensor.

    This computes components of the endomorphism-valued 2-form,
    \f[
        R^a_{b\,cd},
    \f]
    i.e. *not* the purely covariant components. The indices `c,d` are the
    2-form indices and `a,b` the endomorphism ones.

    @param G
        Christoffel symbols with indices `a,b,c` meaning \f$\Gamma^a_{bc}\f$.
    @param dG
        Derivatives of the Christoffel symbols such that `dG[a,b,c,d]` means
        \f$\partial_a \Gamma^b_{cd}\f$.
    @param a
        Indices of the computed component.
    @param b
        Indices of the computed component.
    @param c
        Indices of the computed component.
    @param d
        Indices of the computed component.
    """
    ra = range(G.shape[0])
    return (
        dG[c,a,d,b] - dG[d,a,c,b]
        + fsum(G[a,c,e]*G[e,d,b] for e in ra)
        - fsum(G[a,d,e]*G[e,c,b] for e in ra)
    )
