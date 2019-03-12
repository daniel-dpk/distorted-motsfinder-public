r"""@package motsfinder.metric

Metric and metric tensors.

The classes in this package implement various kinds of metrics for use in MOTS
finder. This currently includes analytically implemented metrics (e.g.
Schwarzschild slices) but may be extended to numerically obtained (possibly
discretized) metric components.


@b Examples

```
    # Create a Brill-Lindquist metric.
    metric = BrillLindquistMetric(m1=1, m2=1, d=1.4, axis='z')

    # Get the metric at a certain coordinate.
    g = metric.at([1.5, 0.2, 0.3])

    # Raise the index of a covector at this point.
    covector = [0.1, 2.3, 4.5]
    v = g.raise_idx(covector)
```
"""

from .analytical import FlatThreeMetric, SchwarzschildSliceMetric
from .analytical import BrillLindquistMetric
from .helpers import christoffel_symbols, christoffel_deriv
from .helpers import riemann_components
from .fourmetric import FourMetric
