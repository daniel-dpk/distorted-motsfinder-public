r"""@package motsfinder.metric.discrete.discretize

Helpers to create discrete versions of (\eg analytical) metrics.

These can be used to compare results obtained with analytically implemented
metrics with those of the discrete metric classes.

@b Examples

```
    # Use a simple Brill-Lindquist metric as example here.
    m1 = 0.2; m2 = 0.8; d = 0.6
    gBL = BrillLindquistMetric(d=d, m1=m1, m2=m2)

    # This creates the discrete version of it.
    res = 128; radius = 2
    g = DiscretizedMetric(
        patch=DiscretizedMetric.construct_patch(res=res, radius=radius),
        metric=gBL,
        curv=gBL.get_curv(), # not needed here as gBL is time symmetric
    )

    # To demonstrate that this metric can be used as usual, we find the four
    # MOTSs in it.
    h = InitHelper(
        metric=g,
        out_base="some/output/folder_res%s" % res,
        suffix="discrete",
    )
    curves = h.find_four_MOTSs(m1=m1, m2=m2, d=d, plot=True)
```
"""

import numpy as np

from ...numutils import nan_mat, raise_all_warnings
from .patch import GridPatch, DataPatch, BBox
from .metric import DiscreteMetric
from .tensors import DiscreteScalarField, DiscreteVectorField
from .tensors import DiscreteSym2TensorField


__all__ = [
    "DiscretizedMetric",
]


class _ScalarField(DiscreteScalarField):
    r"""Axisymmetric discrete scalar field created from a scalar function.

    This samples a (e.g. analytical) function on a grid to create a discrete
    version of it.

    The grid is defined by the configuration of the metric this field is
    associated with.
    """

    def __init__(self, metric, func):
        r"""Create a scalar field from the given function.

        The `metric` defines the discretization (i.e. resolution, domain).
        """
        super(_ScalarField, self).__init__(metric)
        self.__func = func

    def _load_data(self):
        f = self.__func
        patch = self.metric.patch
        mat = self.metric.empty_mat()
        with raise_all_warnings():
            for (i, j, k), pt in patch.grid(full_output=True):
                mat[i, j, k] = _eval(f, pt, np.nan)
        return [DataPatch.from_patch(patch, mat, 'even')]


class _VectorField(DiscreteVectorField):
    r"""Axisymmetric discrete vector field created from a vector-valued function.

    This samples a (e.g. analytical) function on a grid to create a discrete
    version of it.

    The grid is defined by the configuration of the metric this field is
    associated with.
    """

    def __init__(self, metric, func):
        r"""Create a vector field from the given function.

        The `metric` defines the discretization (i.e. resolution, domain).

        `func` should be a callable returning three floats: the x-, y-, and
        z-components of the vector.
        """
        super(_VectorField, self).__init__(metric)
        self.__func = func

    def _load_data(self):
        f = self.__func
        patch = self.metric.patch
        x_mat = self.metric.empty_mat()
        y_mat = self.metric.empty_mat()
        z_mat = self.metric.empty_mat()
        nan = nan_mat(3)
        with raise_all_warnings():
            for (i, j, k), pt in patch.grid(full_output=True):
                x, y, z = _eval(f, pt, nan)
                x_mat[i, j, k] = x
                y_mat[i, j, k] = y
                z_mat[i, j, k] = z
        data = [(x_mat, 'odd'), (y_mat, 'odd'), (z_mat, 'even')]
        return [DataPatch.from_patch(patch, mat, sym) for mat, sym in data]


class _Sym2TensorField(DiscreteSym2TensorField):
    r"""Axisymmetric discrete tensor field created from a matrix-valued function.

    This samples a (e.g. analytical) function on a grid to create a discrete
    version of it.

    The grid is defined by the configuration of the metric this field is
    associated with.
    """

    def __init__(self, metric, func):
        r"""Create a tensor field from the given function.

        The `metric` defines the discretization (i.e. resolution, domain).

        `func` should be a callable returning a symmetric 3x3 matrix.
        """
        super(_Sym2TensorField, self).__init__(metric)
        self.__func = func

    def _load_data(self):
        f = self.__func
        patch = self.metric.patch
        xx_mat = self.metric.empty_mat()
        xy_mat = self.metric.empty_mat()
        xz_mat = self.metric.empty_mat()
        yy_mat = self.metric.empty_mat()
        yz_mat = self.metric.empty_mat()
        zz_mat = self.metric.empty_mat()
        nan = nan_mat((3, 3))
        with raise_all_warnings():
            for (i, j, k), pt in patch.grid(full_output=True):
                mat = _eval(f, pt, nan)
                xx_mat[i, j, k] = mat[0, 0]
                xy_mat[i, j, k] = mat[0, 1]
                xz_mat[i, j, k] = mat[0, 2]
                yy_mat[i, j, k] = mat[1, 1]
                yz_mat[i, j, k] = mat[1, 2]
                zz_mat[i, j, k] = mat[2, 2]
        data = [
            (xx_mat, 'even'),
            (xy_mat, 'even'),
            (xz_mat, 'odd'),
            (yy_mat, 'even'),
            (yz_mat, 'odd'),
            (zz_mat, 'even'),
        ]
        return [DataPatch.from_patch(patch, mat, sym) for mat, sym in data]


class DiscretizedMetric(DiscreteMetric):
    r"""Full discrete slice geometry generated from (analytical) functions.

    This takes a 3-metric (..base._ThreeMetric) and optionally callables to
    evaluate e.g. the extrinsic curvature and builds matrices for all the
    different components. This allows discretization of analytical metrics to
    e.g. compare results at different discrete resolutions.
    """

    def __init__(self, patch, metric, curv=None, lapse=None, shift=None,
                 dtlapse=None, dtshift=None):
        r"""Construct a discrete metric on a given patch.

        @param patch
            Definition of the discretization (i.e. domain, resolution). Use
            the convenience class method construct_patch() to easily generate
            such a patch.
        @param metric
            3-metric tensor field. Should be axisymmetric. A valid example is
            ..analytical.simple.BrillLindquistMetric.
        @param curv
            Callable returning symmetric 3x3 matrices representing the
            extrinsic curvature of the 3-slice embedded in spacetime.
        @param lapse,dtlapse
            Lapse function (scalar) and its time derivative (also scalar),
            respectively. Both are callables returning scalar values.
        @param shift,dtshift
            Shift vector field and its time derivative, respectively. Both
            callables should return 3 floats for the x-, y-, z-components of
            the field at the specified point.
        """
        super(DiscretizedMetric, self).__init__()
        self._patch = patch
        self._metric = _Sym2TensorField(self, metric)
        self._curv = _Sym2TensorField(self, curv) if curv else None
        self._lapse = _ScalarField(self, lapse) if lapse else None
        self._shift = _VectorField(self, shift) if shift else None
        self._dtlapse = _ScalarField(self, dtlapse) if dtlapse else None
        self._dtshift = _VectorField(self, dtshift) if dtshift else None

    @classmethod
    def construct_patch(cls, res, radius, origin=(0., 0., 0.)):
        r"""Class method to create a patch definition from a given resolution.

        This creates a patch to be used for constructing a DiscretizedMetric.
        The domain is specified using an origin and "radius" (the domain is,
        of course, rectangular).

        @param res
            Resolution. There will be `res` grid points per unit per axis.
        @param radius
            Coordinate distance from `origin` to include in the domain.
        @param origin
            Origin of the patch around which the radius defines the full
            domain. Default is ``0,0,0``.
        """
        origin = np.asarray(origin)
        origin[2] -= radius
        deltas = 1./res * np.identity(3)
        box = BBox(
            lower=[0, 0, 0],
            upper=[int(round(res*radius))+1, 1, int(round(2*res*radius))+1],
        )
        return GridPatch(origin=origin, deltas=deltas, box=box)

    @property
    def patch(self):
        r"""Patch property used during construction."""
        return self._patch

    def empty_mat(self):
        r"""Empty (zero) matrix of the correct shape for the full domain."""
        return np.zeros(shape=self.patch.shape)

    def all_field_objects(self):
        return [
            self._metric, self._curv, self._lapse, self._shift,
            self._dtlapse, self._dtshift,
        ]

    def _get_metric(self):
        return self._metric

    def get_curv(self):
        return self._curv

    def get_lapse(self):
        return self._lapse
    def get_dtlapse(self):
        return self._dtlapse

    def get_shift(self):
        return self._shift
    def get_dtshift(self):
        return self._dtshift


def _eval(func, arg, default):
    r"""Evaluate a function, returning a given default in case of error.

    If evaluating the function succeeds, the produced value is returned. If,
    however, a `FloatingPointError` is raised, the given default value is
    returned instead.

    @param func
        Callable to evaluate.
    @param arg
        Argument to call `func` with.
    @param default
        Value to return in case of `FloatingPointError`.
    """
    try:
        return func(arg)
    except FloatingPointError:
        return default
