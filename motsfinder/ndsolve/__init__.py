r"""@package motsfinder.ndsolve

Pseudospectral differential equation solver.

This package contains the basis classes and the pseudospectral solver method
itself. Furthermore, it has the general boundary condition classes
bcs.DirichletCondition, bcs.NeumannCondition,
and bcs.RobinCondition. See also [1], [2].


@b Examples

Solving the following equation on the interval \f$ (1/2, 2) \f$
\f[ f''(x) - a b \sin(b x) f'(x) = a b^2 \cos(b x) f(x) \f]
with boundary conditions \f[
    f(1/2) = \exp(-a\cos(b/2)), \quad f(2) = \exp(-a\cos(2b))
\f]

The exact solution is \f$ f(x) = \exp(-a\cos(b x)) \f$. The numerical
solution can be obtained via:

```
sol = ndsolve(
    eq=((lambda x: -a*b**2*cos(b*x), lambda x: -a*b*sin(b*x), 1), 0),
    basis=ChebyBasis(domain=(0.5, 2), num=50),
    boundary_conditions=(
        DirichletCondition(x=0.5, value=exp(-a*cos(b/2))),
        DirichletCondition(x=2, value=exp(-a*cos(2*b))),
    )
)
```

@b References

[1] Boyd, J. P. "Chebyshev and Fourier Spectral Methods. Dover Publications
    Inc." New York (2001).

[2] Canuto, C., et al. "Spectral Methods: Fundamentals in Single Domains."
    Springer Verlag, 2006.
"""

from .solver import ndsolve, NDSolver
from .bases.cheby import ChebyBasis
from .bases.trig import SineBasis, CosineBasis
from .bcs import DirichletCondition, NeumannCondition, RobinCondition
