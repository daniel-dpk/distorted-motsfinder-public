r"""@package motsfinder.axisym.curve

Curves in the x-z-plane representing axisymmetric surfaces in 3D.

The various curve classes are used to represent general curves (like
parametriccurve.ParametricCurve) usable as reference shapes. The subclasses of
expcurve.ExpansionCurve are, in addition to representing surfaces, aware of
the geometry of the slice and can compute the expansion along the surface they
represent.


@b Examples

```
    # Create a Brill-Lindquist metric with a common apparent horizon and a
    # common inner horizon in addition to the two individual BH horizons.
    metric = BrillLindquistMetric(m1=1, m2=1, d=1.4, axis='z')

    # Use a simple sphere as initial trial surface. We use a resolution of 80
    # to indicate the desired resolution for the following search.
    c0 = StarShapedCurve.create_sphere(2.0, 80, metric=metric)

    # Find the apparent horizon starting from the trial surface.
    c_AH = newton_kantorovich(c0)

    # Create a curve in reference parameterization from the AH. We make a step
    # inward (the -0.3) to kick off the search inward from the AH.
    c0 = RefParamCurve.from_curve(c_AH, [-0.3], num=200)

    # Starting from the above curve in reference parameterization, find the
    # inner common MOTS.
    c_inner = newton_kantorovich(c0)

    # Plot what we have so far.
    c_AH.plot_curves((c_AH, 'AH', '-b'), (c_inner, 'inner MOTS', '-g'))
```
"""

from .basecurve import BaseCurve
from .parametriccurve import ParametricCurve
from .refparamcurve import RefParamCurve
from .starshapedcurve import StarShapedCurve
