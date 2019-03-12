r"""@package motsfinder.metric.discrete.tensors

Various discrete tensors, from scalars to (0,2)-tensors.
"""

from abc import ABCMeta, abstractmethod

from six import add_metaclass
import numpy as np

from ...numutils import nan_mat
from .numerical import eval_sym_axisym_matrix


__all__ = []


# It is customary to denote indices of tensors without spaces, e.g.:
#   T_{ijk}  =>  T[i,j,k]
# We disable the respective pylint warning for this file.
# pylint: disable=bad-whitespace


# TODO: Redo docstrings
@add_metaclass(ABCMeta)
class _DiscreteField():
    r"""General field baseclass for scalars, vectors, or higher tensors.

    This baseclass coordinates lazy-loading of the data from disk when
    subclasses access the `components` property. It also handles caching
    results evaluated at the most recent point.

    Subclasses need to implement the following methods:
        * _eval() - to evaluate the field (components) at a particular given
          point
    """

    def __init__(self, discrete_metric):
        r"""Create a new field object belonging to the given metric."""
        ## DiscreteMetric object this field curvature belongs to.
        self.metric = discrete_metric
        ## Lazily loaded numerical data.
        self._field_data = None
        ## Cached tensor components interpolated at the last requested coordinate.
        self._field_cache = [None] * 3

    def __getstate__(self):
        r"""Return a picklable state object."""
        self.reset_cache()
        state = self.__dict__.copy()
        if not self.metric.save_full_data:
            state['_field_data'] = None
        return state

    @abstractmethod
    def _eval(self, point, interp, diff):
        r"""Evaluate the field (or derivatives) at a given point."""
        pass

    @abstractmethod
    def _load_data(self):
        r"""Generate or load the data from a file."""
        pass

    def load_data(self):
        r"""Generate or load the data from a file."""
        if self._field_data is None:
            self._field_data = self._load_data()

    def unload_data(self):
        self._field_data = None

    def reset_cache(self):
        r"""Reset cached values to force re-evaluation upon next request.

        This should be called when the point at which to evaluate/interpolate
        is different from the last point at which evaluation was done.
        Users usually don't need to call this themselves as it is called
        indirectly by the updating mechanism.
        """
        self._field_cache = [None] * 3

    @property
    def components(self):
        r"""Full set of component matrices (lazily loaded).

        For vector and covector fields, the returned elements are `x`, `y`,
        `z`, while for 2-tensors, they are `xx`, `xy`, `xz`, `yy`, `yz`, and
        `zz` in that order. For a scalar field, the returned list will have
        only one element. Each component is a DataPatch object.
        """
        self.load_data()
        return self._field_data

    def __call__(self, point, diff=0, interp=None):
        r"""Evaluate/interpolate the field at a point.

        In case `interp` is not specified, respects the `interpolate`
        attribute of the corresponding metric object.

        @return If ``diff=0``, returns the field
            (components) at the given `point`. For a scalar field, this will
            be a float while e.g. for a `0,2`-tensor, a 3x3 matrix is
            returned. If ``diff>0``, the first `diff` axes run over the
            partial derivative direction. For example, for a `0,2`-tensor `T`
            and with ``diff==1``, returns a 3x3x3 matrix `dT`, where
            ``dT[i,j,k]`` corresponds to \f$\partial_i T_{jk}\f$.

        @param point
            Point in coordinate space at which to evaluate/interpolate.
        @param diff
            Derivative order to compute. Default is `0`.
        @param interp
            Override the `interpolate` attribute of the metric for this
            evaluation.

        @b Notes

        This caches the results of evaluating the field for the most recent
        point. This is especially useful if computing higher derivatives
        requires the results of lower derivatives, since these will then be
        computed only once.

        To ensure the cache is used, sub-classes should use the __call__()
        function for evaluation instead of doing a full computation via an
        _eval() call.
        """
        if interp is None:
            interp = self.metric.interpolate
        point = self.metric.prepare_for_point(point)
        if len(self._field_cache) <= diff:
            self._field_cache += [None] * (diff - len(self._field_cache) + 1)
        if self._field_cache[diff] is None:
            self._field_cache[diff] = self._eval(
                point=point, interp=interp, diff=diff
            )
        return self._field_cache[diff]


class DiscreteScalarField(_DiscreteField):
    r"""Represents a scalar field with one component."""
    # pylint: disable=abstract-method

    def _eval(self, point, interp, diff):
        field = self.components[0]
        if diff == 0:
            return field.interpolate(point, interp=interp)
        if diff == 1:
            fx, fz = field.diff(point, diff=1, interp=interp)
            return np.array([fx, 0., fz])
        x = point[0]
        # pylint: disable=unsubscriptable-object
        fx = self(point=point, interp=interp, diff=1)[0]
        if diff == 2:
            fxx, fzz, fxz = field.diff(point, diff=2, interp=interp)
            fxy = fyz = 0.
            if x == 0:
                fyy = np.nan
            else:
                fyy = fx/x
            return np.asarray([[fxx, fxy, fxz],
                               [fxy, fyy, fyz],
                               [fxz, fyz, fzz]])
        raise ValueError("Unknown `diff` value: %s" % diff)


class DiscreteVectorField(_DiscreteField):
    r"""Represents a vector field with three spatial components."""
    # pylint: disable=abstract-method

    def _eval(self, point, interp, diff):
        if diff == 0:
            T = np.array([
                Ti.interpolate(point, interp=interp) for Ti in self.components
            ])
            return T
        x = point[0]
        T = self(point=point, interp=interp, diff=0)
        if diff == 1:
            (T0x, T0z), (T1x, T1z), (T2x, T2z) = [
                Ti.diff(point, diff=1, interp=interp)
                for Ti in self.components
            ]
            Tx = np.array([T0x, T1x, T2x])
            Tz = np.array([T0z, T1z, T2z])
            if x == 0:
                Ty = nan_mat((3,))
            else:
                Ty = 1/x * np.array([-T[1], T[0], 0.])
            return np.asarray([Tx, Ty, Tz])
        dT = self(point=point, interp=interp, diff=1)
        if diff == 2:
            (
                (T0xx, T0zz, T0xz),
                (T1xx, T1zz, T1xz),
                (T2xx, T2zz, T2xz),
            ) = [
                Ti.diff(point, diff=2, interp=interp)
                for Ti in self.components
            ]
            Txx = np.array([T0xx, T1xx, T2xx])
            Tzz = np.array([T0zz, T1zz, T2zz])
            Txz = np.array([T0xz, T1xz, T2xz])
            # pylint: disable=unsubscriptable-object
            if x == 0:
                Txy = nan_mat((3,))
                Tyy = nan_mat((3,))
                Tyz = nan_mat((3,))
            else:
                # note: dT[i,j] == partial_i T^j == T^j_{,i}
                Txy = 1/x * np.array([T[1]/x - dT[0,1],
                                      -T[0]/x + dT[0,0],
                                      0.])
                Tyy = 1/x * np.array([-T[0]/x + dT[0,0],
                                      -T[1]/x + dT[0,1],
                                      dT[0,2]])
                Tyz = 1/x * np.array([-dT[2,1],
                                      dT[2,0],
                                      0.])
            return np.asarray([[Txx, Txy, Txz],
                               [Txy, Tyy, Tyz],
                               [Txz, Tyz, Tzz]])
        raise ValueError("Unknown `diff` value: %s" % diff)


class DiscreteSym2TensorField(_DiscreteField):
    # pylint: disable=abstract-method

    def _eval(self, point, interp, diff):
        if diff == 0:
            return eval_sym_axisym_matrix(
                self.components, point=point, interp=interp, diff=0
            )
        if diff == 1:
            return eval_sym_axisym_matrix(
                self.components,
                self(point=point, interp=interp, diff=0),
                point=point, interp=interp, diff=1
            )
        if diff == 2:
            return eval_sym_axisym_matrix(
                self.components,
                self(point=point, interp=interp, diff=0),
                self(point=point, interp=interp, diff=1),
                point=point, interp=interp, diff=2
            )
        raise NotImplementedError
