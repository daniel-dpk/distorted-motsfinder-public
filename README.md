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


## Examples

* [Brill-Lindquist initial data](docs_input/examples_bl.md)
* [Using the finder with a numerical metric](docs_input/examples_num_metric.md)


## Installation

The prerequisites to running the MOTS Finder are as follows:

* Python 3.6.6
* SciPy 1.1.0
* NumPy 1.14.3
* SymPy 1.1.1
* mpmath 1.0.0
* matplotlib
* Jupyter
* For reading simulation data:
  [SimulationIO](https://github.com/eschnett/SimulationIO) with its
  prerequisites and Python 3 bindings


The simplest way to set up the environment on a Linux-based machine is as
follows. Here, we use `apt` (e.g. in Ubuntu), please translate to your
preferred package management method.

```.sh
$ sudo apt install virtualenv
$ cd ~
$ virtualenv --always-copy --python=python3 py3env
$ source ~/py3env/bin/activate
$ pip install numpy scipy sympy mpmath matplotlib jupyter
$ mkdir -p ~/src
$ cd ~/src
$ git clone https://github.com/daniel-dpk/distorted-motsfinder-public.git
```

Then, to start a notebook server in a dedicated folder:

```.sh
$ source ~/py3env/bin/activate
$ mkdir -p ~/src/motsfinder-local-notebooks
$ cd ~/src/motsfinder-local-notebooks
$ jupyter notebook
```

If you need to read simulation data and would like to use
[SimulationIO](https://github.com/eschnett/SimulationIO) for this purpose,
please see its instructions for building and installation.


## Authors

Maintainer: Daniel Pook-Kolb

Further authors and contributors:

* Ofek Birnholtz
* Badri Krishnan
* Erik Schnetter
* Victor Zhang


## Developing

An overview of how the project source is organized and documented is available
in [README_SRC.md](README_SRC.md).
