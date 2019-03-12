r"""@package motsfinder.metric.simulation

Axisymmetric metrics read from simulation data.

The classes in this module interpret simulation results on a grid of points
and build 3-metric objects that can be evaluated at any point in the
simulation domain. Lagrange interpolation is used to evaluate between the grid
points. Derivatives of the metric are obtained by first computing derivatives
on a patch of grid points using finite differencing and then interpolating the
results using Lagrange interpolation.

This module requires the `SimulationIO` Python library available at:

    https://github.com/eschnett/SimulationIO


@b Examples

This example opens a data file and prints out the detected patch structure.
```
    fname = 'the/simulation/file.it0000001234.s5'
    proj = SioProject(fname)
    for i, patch in enumerate(proj.patches):
        (x1, x2), (y1, y2), (z1, z2) = patch.domain
        print("Domain of patch %2d: "
              "x:[%+10.6f, %+10.6f]  "
              "y:[%+10.6f, %+10.6f]  "
              "z:[%+10.6f, %+10.6f]"
              % (i, x1, x2, y1, y2, z1, z2))
```

We can read in a component of e.g. the metric and plot it using:
```
    g00 = proj.field_component_matrix('admbase::metric', '00')
    plot_mat(g00.mat[:,g00.yidx,:], log10=True, cmap=cm.viridis, vmax=2.2,
             figsize=(10, 4), grid=False,
             title=(r'$\log_{10}(|g_{xx}|)$', dict(y=1.1)))
```

Using the data as a metric for the MOTS finder is easy too:
```
    metric = SioMetric(fname)
    cfg = GeneralMotsConfig(metric, metric.get_curv(), c_ref=1.0, num=20,
                            atol=1e-5, accurate_test_res=None,
                            auto_resolution=False)
    c = find_mots(cfg, verbose=True)
    c.plot(label="MOTS")
```

The above obviously requires the initial guess to be sufficiently close to the
MOTS in order to converge. Note also that we use no auto-resolution and a
relatively high tolerance due to the data usually having less accuracy than
analytically implemented metrics like the Brill-Lindquist metric.
"""

from .sioproject import SioProject
from .siometric import SioMetric
