# Source Code Documentation

The source code is documented using `docstrings` which are translated to html
help using [Doxygen](http://www.stack.nl/~dimitri/doxygen/) with the help of
[doxypypy](https://pypi.org/project/doxypypy/).


## Building the documentation

The prerequisites for building the documentation are:

* Doxygen
* dot (e.g. via graphviz)
* LaTeX (e.g. via texlive)
* doxypypy (e.g. via `pip install doxypypy` in your virtual environment)

To (re-)generate the complete docs, execute the following in the root of this
project:

```
doxygen Doxyfile
```

A `virtualenv` environment can be used for the `doxypypy` call by copying
`./config.cfg` to the untracked `config.mine.cfg`, where you can configure
which python binary should be used for the unit tests and the Doxygen
filtering.

For example:

```.sh
$ cd ~/src/distorted-motsfinder-public
$ cp config.cfg config.mine.cfg
```

Then edit the file `config.mine.cfg` with your preferred text editor and in
e.g. the ``[linux]`` section, change the ``python: ...`` line to something
like:

```
python: LC_ALL=en_US.UTF-8 ~/py3env/bin/python3
```

> **NOTE**
>
> Currently, the combination of Doxygen and doxypypy seems to have the
> following problems:
>     * Documentation of parent classes implementations do not get copied to
>       child class overrides of methods.
>     * It is not able to correctly identify properties and instead displays
>       them as functions.


## Top-Level Helper Scripts

The files in the top-level directory (like `cythonize_all.py`) are not part of
the `motsfinder` module and used just to e.g. help initiating unittests
running inside `virtualenv` environments.

Executable helper scripts:

```bash
call_python.py      # Run a Python script from a virtual environment
cythonize_all.py    # Compile all Cython (.pyx) files in sub folders
cythonize_helper.py # Call `cythonize_all` in a virtual environment
tests.py            # Run all test_*.py unittest scripts in sub folders
teststarter.py      # Run `tests` in a virtual environment
testutils.py        # Tweaked `unittest.TestCase` class with more features
```

The remaining files have the following role:

```bash
config.cfg           # Config template for setting up the virtualenv (see contents)
Doxyfile             # Doxygen configuration
doxystyle_tweaks.css # Custom style settings for the html documentation
LICENSE.txt          # Project license (MIT)
README.md            # Main readme file with examples runnable in Jupyter
README_SRC.md        # This readme describing the source
```

### Running the tests

Tests are implemented using the standard `unittest` module. The tests for a
certain `module_file.py` are contained in `test_module_file.py` in the same
folder to keep them close to the actual source.

To run the currently implemented tests, use:

```bash
./teststarter.py
```

The following options are understood:

```bash
    -f/--failfast       # stop at first fail/error
    -b/--buffer         # turn on output buffering
    -t/--timing         # print execution time for each test*
    -s/--run-slow-tests # don't skip tests marked as slow*
    -w/-wait            # prompt before exiting the run
```

* The `-t` and `-s` options require the tests to be subclassed from
testutils.DpkTestCase instead of `unittest.TestCase`. To mark a test as
slow, apply the testutils.slowtest decorator.


## The Actual Source

The source of the project is located in the `./motsfinder` directory. A short
breakdown of the most important files is as follows (`.py` suffix suppressed):

```bash
axisym/           # Axisymmetric MOTS finder with and without star-shaped assumption
    curve/        # Curves in the x-z-plane representing axisymmetric surfaces in 3D
        expcurve        # Base class for curves that can calculate their expansion
        parametriccurve # General parametric curve in the x-z-plane
        refparamcurve   # Surfaces in local coordinates relative to a reference shape
        starshapedcurve # Surfaces in star-shaped parameterization relative to some origin
    findmots            # Convenience function(s) to coordinate finding MOTSs
    initialguess  # Helpers for determining MOTSs without data from previous steps
    newton        # Implementation of a Newton-Kantorovich method to find MOTSs
    trackmots     # Track a single MOTS through slices of a simulation

exprs/            # Expression system for composing functions and efficiently evaluating them and their derivatives
    cheby         # Expression representing a truncated Chebyshev polynomial series
    trig	      # Expression representing a truncated sine or cosine series

ipyutils/         # Utility functions for interactive IPython/Jupyter sessions
    plotting      # Helper functions for more convenient plotting
    plotting3d    # Functions for plotting in 3D, e.g. 2-dimensional surfaces via plot_2d()
    reloading     # Helpers for reloading changed sources at runtime

metric/           # Metric and metric tensors
    analytical	  # Analytically implemented metrics
    discrete      # Metrics interpolated from data on a grid
    fourmetric    # Spacetime metric constructed from 3-metric and lapse, shift
    simulation    # Axisymmetric metrics read from simulation data using SimulationIO

ndsolve/          # Pseudospectral differential equation solver
    bases/        # Bases that can be used in the pseudospectral solver
        cheby     # Chebyshev polynomial basis for the pseudospectral solver
        cosine    # Fourier cosine basis for the pseudospectral solver
    bcs           # Classes for imposing boundary conditions
    solver        # The actual solver ndsolve() and the class NDSolver

numutils          # Miscellaneous numerical utilities and helpers
pickle_helpers    # Helper functions to (un)pickle problematic objects
utils             # General utilities for simplifying certain tasks in Python
```

### Walking Through an Example

Consider the problem of finding the apparent horizon (AH) for a
Brill-Lindquist metric. We will use the star-shaped assumption in this
example.

First, we create the metric object using BrillLindquistMetric from
`./metric/analytical/simple.py`:

```.py
metric = BrillLindquistMetric(m1=1, m2=1, d=1.4, axis='z')
```

Next, create the initial guess curve which we take as a sphere of radius 2 (in
coordinate space). We already specify the resolution of the spectral
representation of the curve and store the metric of the space this curve lives
in. Internally, a cosine series is created with 100 basis functions and
coefficients `[2.0, 0.0, 0.0, ..., 0.0]`.

```.py
c0 = StarShapedCurve.create_sphere(2.0, num=100, metric=metric)
```

The main function to find the AH/MOTS is
motsfinder.axisym.newton.newton_kantorovich in `./axisym/newton.py`:

```.py
c_AH = newton_kantorovich(c0)
```

What happens here is the following:

* We iterate through the allowed number of steps and in each step:
    * We solve the linearized expansion equation for the current horizon
      function. The terms are provided by the curve itself.
    * We check for convergence.
    * The solution of the above equation is added to the current horizon
      function. This is the Newton step. We don't add the full solution, only
      a fraction of it until we reach the linear regime to increase the
      convergence radius. (Implementing an actual line-search could
      dramatically improve convergence radius and speed, though not accuracy,
      of course).

#### Plotting the result

To plot the horizon, you may simply do:

```.py
c_AH.plot(l='-b', label='AH', figsize=(6, 6))
```

To plot the coefficients of the final horizon and verify exponential
convergence, use:

```.py
c_AH.plot_coeffs()
```

We see that every other coefficient is (essentially) zero, reflecting the
`pi/2` symmetry due to two equal mass black holes. More importantly, the other
coefficients exponentially converge to zero.

We can also check the vanishing of the expansion along the AH:

```.py
c_AH.plot_expansion(points=200)
```
