r"""@package motsfinder.metric.analytical

Analytically implemented metrics.

These metrics are known analytically and hence provide the optimal (floating
point) accuracy.


@b Examples

```
    # Create a Schwarzschild slice metric.
    metric = SchwarzschildSliceMetric(m=2)
    print("Schwarzschild radius: %s" % (2*metric.m))
    print("Horizon radial coordinate: %s" % metric.horizon_coord_radius())
```
"""

from .simple import FlatThreeMetric, SchwarzschildSliceMetric
from .simple import BrillLindquistMetric
