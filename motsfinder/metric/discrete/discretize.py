r"""@package motsfinder.metric.discrete.discretize

Helpers to create discrete versions of (\eg analytical) metrics.

These can be used to compare results obtained with analytically implemented
metrics with those of the discrete metric classes.
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
    def __init__(self, metric, func):
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
    def __init__(self, metric, func):
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
    def __init__(self, metric, func):
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
    def __init__(self, patch, metric, curv=None, lapse=None, shift=None,
                 dtlapse=None, dtshift=None):
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
        return self._patch

    def empty_mat(self):
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
    try:
        return func(arg)
    except FloatingPointError:
        return default
