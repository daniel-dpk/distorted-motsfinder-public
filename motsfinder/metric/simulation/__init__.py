r"""@package motsfinder.metric.simulation

Axisymmetric metrics read from simulation data using `SimulationIO`.

The classes in this module implement a ..discrete.metric.DiscreteMetric by
reading in the data using SimulationIO [1]. Examples of parameter files
producing compatible output using the Einstein Toolkit [2,3] can be found in
the ``paper2/parfiles/`` directory.

To simplify handling of multi-patch data, the fields are collected into single
large matrices.


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
    cfg = GeneralMotsConfig.preset(
        "discrete2", hname="AH", metric=metric, num=30,
    )
    c = find_mots(cfg, c_ref=1.0, verbose=True)
    c.plot(label="MOTS")
```

In the above example, we take a round (coordinate) sphere of radius 1.0 as
initial guess. Whether this converges to a MOTS obviously depends on the data
at hand.


@b References

[1] Erik Schnetter, & Jonah Miller. (2019, January 27). eschnett/SimulationIO:
    Allow specifying compression options when writing datasets (Version
    version/8.0.0). Zenodo. http://doi.org/10.5281/zenodo.2550800

[2] Einstein Toolkit: Open software for relativistic astrophysics.
    http://einsteintoolkit.org/.

[3] F. Löffler, J. Faber, E. Bentivegna, T. Bode, P. Diener, R. Haas, I.
    Hinder, B. C. Mundim, C. D. Ott, E. Schnetter, G. Allen, M. Campanelli,
    and P. Laguna. The Einstein Toolkit: A Community Computational
    Infrastructure for Relativistic Astrophysics. Class. Quantum Grav.,
    29(11):115001, 2012.
"""

from .sioproject import SioProject
from .siometric import SioMetric
