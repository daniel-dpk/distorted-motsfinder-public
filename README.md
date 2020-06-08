# MOTS Finder


[![DOI](https://zenodo.org/badge/159724600.svg)](https://zenodo.org/badge/latestdoi/159724600)


This project implements an algorithm for numerically finding MOTSs that are
not star-shaped. The current implementation is limited to the axisymmetric
case but the ideas it is based on easily generalize to the fully
non-symmetrical case.


## Using this project in your research

Anyone is free and welcome to use the code provided in this project
(see the `LICENSE.txt`).
For any publications produced using this code, please cite:

> D. Pook-Kolb, O. Birnholtz, B. Krishnan and E. Schnetter "Existence and
> stability of marginally trapped surfaces in black-hole spacetimes." Physical
> Review D 99.6 (2019): 064005.

For any questions/comments/ideas, feel free to contact
`daniel.pook.kolb at aei.mpg.de`.


## Documentation

HTML documentation (including this readme) is available in:

```
./docs/html/index.html
```

> **IMPORTANT**
>
> If a class method is not documented, check if the parent class one is
> instead.


## Examples and tutorials

* [Brill-Lindquist initial data](docs_input/examples_bl.md)
* [Using the finder with a numerical metric](docs_input/examples_num_metric.md)
* Finding stability spectra in Kerr slices
  ([html](docs_input/Tutorial_Find_Eigenvalues_Kerr_Slice.html),
  [notebook](docs_input/Tutorial_Find_Eigenvalues_Kerr_Slice.ipynb))
* Implementing a new analytical metric
  ([html](docs_input/Tutorial_Implement_New_Analytical_Metric.html),
  [notebook](docs_input/Tutorial_Implement_New_Analytical_Metric.ipynb))


## Installation

The prerequisites for running the MOTS Finder are as follows:

* Python 3.6 or 3.7 (not tested with Python 3.8; use e.g. `pyenv` to obtain
  any Python version)
* SciPy 1.2.1
* NumPy 1.16.1
* SymPy 1.3
* mpmath 1.1.0
* matplotlib 2.2.5
* Jupyter
* Optionally, for speeding up several computations, cython 0.29.15
* Optionally, for reading simulation data:
  [SimulationIO](https://github.com/eschnett/SimulationIO) with its
  prerequisites and Python 3 bindings
* Optionally, for expansions into spin-weighted spherical harmonics:
  [spinsfast](https://github.com/moble/spinsfast) with its prerequisites


The simplest way to set up the environment on a Linux-based machine is as
follows. Here, we use `apt` (e.g. in Ubuntu), please translate to your
preferred package management method.

First, create a new Python environment to not interfere with any of your
system Python configurations:

```.sh
$ sudo apt install virtualenv
$ cd ~
$ virtualenv --always-copy --python=python3 py3env
```

Next, clone this repo to get the MOTS Finder source code:

```.sh
$ mkdir -p ~/src
$ cd ~/src
$ git clone https://github.com/daniel-dpk/distorted-motsfinder-public.git
```

Finally, install all the required Python packages. The following installs the
exact package versions the MOTS Finder was tested with. You may also try to
install the most up to date versions. To do this, replace the second line
below with `pip install numpy scipy sympy mpmath matplotlib jupyter`.

```.sh
$ source ~/py3env/bin/activate
$ pip install -r ~/src/distorted-motsfinder-public/python-requirements.txt
```

### Using the MOTS Finder

To use the finder, start a notebook server in a dedicated folder:

```.sh
$ source ~/py3env/bin/activate  # in case it's not already active
$ mkdir -p ~/src/motsfinder-local-notebooks
$ cd ~/src/motsfinder-local-notebooks
$ jupyter notebook
```

**NOTE:** If you already use Jupyter from a different Python environment, the
above will probably override your system Jupyter settings. To avoid that,
replace the last line with:

```.sh
$ JUPYTER_CONFIG_DIR="$HOME/.jupyter-motsfinder" JUPYTER_DATA_DIR="$HOME/.local/share/jupyter-motsfinder" jupyter notebook
```

Likewise, to install the Jupyter extensions into this environment, execute the
following and then restart the Jupyter notebook server using the above
command:

```.sh
$ source ~/py3env/bin/activate  # in case it's not already active
$ JUPYTER_CONFIG_DIR="$HOME/.jupyter-motsfinder" JUPYTER_DATA_DIR="$HOME/.local/share/jupyter-motsfinder" jupyter contrib nbextension install --user
```


### (Optional) Using Cython to speed up some computations

A short summary of the required steps to utilize the Cython implementations
is:

```.sh
$ sudo apt install python3-dev
$ source ~/py3env/bin/activate
$ cd ~/src/distorted-motsfinder-public
$ pip install cython==0.29.15  # remove the '==...' part to use the current version
$ ./cythonize_helper.py
```


### (Optional) For reading simulation data using the `SioMetric` class

If you need to read simulation data and would like to use
[SimulationIO](https://github.com/eschnett/SimulationIO) for this purpose,
please see its instructions for building and installation.


### (Optional) For expanding into spin-weighted spherical harmonics

The following works on Ubuntu-based systems. Please translate to your
environment as needed.

```.sh
$ sudo apt install libfftw3-dev
... restart shell ...
$ source ~/py3env/bin/activate
$ pip install spinsfast
```

## Authors

Maintainer: Daniel Pook-Kolb

Further authors and contributors:

* Ofek Birnholtz
* Jose Luis Jaramillo
* Badri Krishnan
* Erik Schnetter
* Victor Zhang


## Developing

An overview of how the project source is organized and documented is available
in [README_SRC.md](README_SRC.md).
