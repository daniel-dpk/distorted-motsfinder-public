r"""@package motsfinder.axisym

Axisymmetric MOTS finder with and without star-shaped assumption.

This package contains the MOTS finders for the axisymmetric case. Prototypes
of time symmetric cases are the Brill-Lindquist metric (and, of course,
spatical Schwarzschild slices).

The code in this file allows finding MOTSs in axisymmetric cases using an
elliptic-PDE type solver based on the descriptions of
\ref thornburg2003 "[1]", but simplified to the axisymmetric case.

@b References

\anchor thornburg2003 [1] Thornburg, Jonathan. "A fast apparent horizon
     finder for three-dimensional Cartesian grids in numerical relativity."
     Classical and quantum gravity 21.2 (2003): 743.

@b Examples

```
    metric = BrillLindquistMetric(m1=1, m2=1, d=1.4, axis='z')
    c0 = StarShapedCurve.create_sphere(radius=2.0, num=100, metric=metric)
    c_AH = newton_kantorovich(c0, verbose=True)
    c_AH.plot(label='AH')
```
"""

from .newton import newton_kantorovich
from .curve import StarShapedCurve, RefParamCurve, ParametricCurve
