r"""@package motsfinder.metric.discrete.tensors

Various abstract discrete tensor classes, from scalars to (0,2)-tensors.

These abstract classes provide all the functionality needed for representing
tensors such as the extrinsic curvature, metric, lapse function and shift
vector field. The only missing part is the actual generation/acquisition of
data using the _DiscreteField._load_data() method.

Note that even though a symmetric (0,2)-tensor can represent a metric, by
convention a full .metric.DiscreteMetric also has the responsibility to
provide the other fields required to define the geometry and embedding
(extrinsic curvature) of the slice in spacetime as well as (possibly) the
lapse function and shift vector field. Therefore, the DiscreteSym2TensorField
class may be used to *implement* such a metric using numerical data on a grid.

Examples of such implementations currently include
.discretize.DiscretizedMetric and ..simulation.siometric.
"""

from abc import ABCMeta, abstractmethod
import gc

from six import add_metaclass
import numpy as np

from .numerical import eval_sym_axisym_matrix, _get_fy, _get_fxy_fyy_fyz
from .numerical import _get_Vy, _get_Vxy_Vyy_Vyz


__all__ = []


@add_metaclass(ABCMeta)
class _DiscreteField():
    r"""General field base class for scalars, vectors, or higher tensors.

    This base class coordinates lazy-loading of the data from disk when
    subclasses access the `components` property. It also handles caching
    results evaluated at the most recent point.

    Subclasses need to implement the following methods:
        * _eval() - to evaluate the field (components) at a particular given
          point
        * _load_data() - to load (or generate) the discrete data

    Usually, one would subclass not this class directly but one of the other
    classes of this module, namely DiscreteScalarField, DiscreteVectorField,
    DiscreteSym2TensorField. Then, only _load_data() needs to be implemented.
    """

    def __init__(self, discrete_metric):
        r"""Create a new field object belonging to the given metric.

        The metric is responsible for coordinating the cache as well as
        loading/unloading of the data (hence the bi-directional relationship).
        """
        ## DiscreteMetric object this field curvature belongs to.
        self.metric = discrete_metric
        ## Lazily loaded numerical data.
        self._field_data = None
        ## Cached tensor components interpolated at the last requested coordinate.
        self._field_cache = [None] * 3
        self._interpolation = None
        self._fd_order = None
        ## The point (as 3-tuple) for which the cached values are valid.
        self._prev_pt = None

    def __getstate__(self):
        r"""Return a picklable state object."""
        self.reset_cache()
        state = self.__dict__.copy()
        state['_prev_pt'] = None
        if not self.metric.save_full_data:
            state['_field_data'] = None
        return state

    def __setstate__(self, state):
        r"""Restore this object from the given unpickled state."""
        # compatibility with data from previous versions
        self._interpolation = 'lagrange' # previously the only option
        self._fd_order = 4 # previous default
        self._prev_pt = None
        # Restore state. This overrides the above if contained in the data.
        self.__dict__.update(state)

    def set_fd_order(self, fd_order):
        r"""Set the finite difference differentiation accuracy order for all
        componenets."""
        self._fd_order = fd_order
        if self._field_data and self._fd_order is not None:
            for patch in self._field_data:
                patch.fd_order = self._fd_order

    def set_interpolation(self, interpolation):
        r"""Set the kind of interpolation for all componenets.

        Refer to .patch.DataPatch.set_interpolation() for details.
        """
        self._interpolation = interpolation
        if self._field_data and self._interpolation is not None:
            for patch in self._field_data:
                patch.set_interpolation(self._interpolation)

    @abstractmethod
    def _eval(self, point, diff):
        r"""Evaluate the field (or derivatives) at a given point.

        @param point
            Point in physical space.
        @param diff
            Derivative order of the field to compute. Results depend on the
            tensor type of the field.
        """
        pass

    @abstractmethod
    def _load_data(self):
        r"""Generate or load the data from a file.

        This method should be overridden to generate or load the full data
        into memory. This includes all components in case the field is a
        tensor having more than one component. This method should not set any
        member variables but instead return the data as a list of component
        DataPatch objects.

        Note that even scalars should return a (one-element) list of data
        patches.
        """
        pass

    def load_data(self):
        r"""Generate or load the data from a file.

        Note that this method coordinates the lazy-loading of the actual data
        and hence it should not be overridden by child classes for this
        purpose. Instead, _load_data() should be overridden.
        """
        if self._field_data is None:
            # We may need a lot of memory now, so better clean up before.
            gc.collect()
            self._field_data = self._load_data()
            self.set_fd_order(self._fd_order)
            self.set_interpolation(self._interpolation)

    def unload_data(self):
        r"""Free the memory occupied by loaded numerical data.

        If the data is needed at a later time, a reloading/regeneration of the
        data will be performed, so only use this if it is not needed anymore.

        On the other hand, since this data may occupy large amounts of memory,
        it should be unloaded if results referring to it are to be kept in
        memory. A prime example is determining MOTSs in a sequence of slices,
        while reusing previous results for the next steps. In that case, the
        data should be unloaded prior to moving on to the next slice to not
        accumulate multiple slices in memory.
        """
        if self._field_data is not None:
            self._field_data = None
            # Ensure the data is actually released from memory. This may free
            # hundreds of MB of memory in extreme cases, justifying its direct
            # use here.
            gc.collect()

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

    def __call__(self, point, diff=0):
        r"""Evaluate/interpolate the field at a point.

        @return If ``diff=0``, returns the field (components) at the given
            `point`. For a scalar field, this will be a float while e.g. for a
            `0,2`-tensor, a 3x3 matrix is returned. If ``diff>0``, the first
            `diff` axes run over the partial derivative direction. For
            example, for a `0,2`-tensor `T` and with ``diff==1``, returns a
            3x3x3 matrix `dT`, where ``dT[i,j,k]`` corresponds to
            \f$\partial_i T_{jk}\f$.

        @param point
            Point in coordinate space at which to evaluate/interpolate.
        @param diff
            Derivative order to compute. Default is `0`.

        @b Notes

        This caches the results of evaluating the field for the most recent
        point. This is especially useful if computing higher derivatives
        requires the results of lower derivatives, since these will then be
        computed only once.

        To ensure the cache is used, sub-classes should use the __call__()
        function for evaluation instead of doing a full computation via an
        _eval() call.
        """
        if self.components[0].get_interpolation() is None:
            # we snap only in order not to clear the cache for near-by points
            # that would snap to the same grid point
            point = self.components[0].snap_to_grid(point)
        point_tuple = tuple(point)
        if point_tuple != self._prev_pt:
            self._prev_pt = point_tuple
            self.reset_cache()
        if len(self._field_cache) <= diff:
            self._field_cache += [None] * (diff - len(self._field_cache) + 1)
        result = self._field_cache[diff]
        if result is None:
            result = self._eval(
                point=point, diff=diff
            )
            self._field_cache[diff] = result
        return result


class DiscreteScalarField(_DiscreteField):
    r"""Represents a scalar field with one component.

    Subclass this class and implement the _load_data() method to generate or
    load the data.
    """
    # pylint: disable=abstract-method

    def _eval(self, point, diff):
        field = self.components[0]
        if diff == 0:
            return field.interpolate(point)
        if diff == 1:
            fx, fz = field.diff(point, diff=1)
            return np.array([fx, _get_fy(), fz])
        if diff == 2:
            fxx, fzz, fxz = field.diff(point, diff=2)
            fxy, fyy, fyz = _get_fxy_fyy_fyz(point, df=self(point, diff=1))
            return np.asarray([[fxx, fxy, fxz],
                               [fxy, fyy, fyz],
                               [fxz, fyz, fzz]])
        raise ValueError("Unknown `diff` value: %s" % diff)


class DiscreteVectorField(_DiscreteField):
    r"""Represents a vector field with three spatial components.

    Subclass this class and implement the _load_data() method to generate or
    load the data.
    """
    # pylint: disable=abstract-method

    def _eval(self, point, diff):
        if diff == 0:
            T = np.array([
                Ti.interpolate(point) for Ti in self.components
            ])
            return T
        T = self(point=point, diff=0)
        if diff == 1:
            (T0x, T0z), (T1x, T1z), (T2x, T2z) = [
                Ti.diff(point, diff=1)
                for Ti in self.components
            ]
            Tx = np.array([T0x, T1x, T2x])
            Tz = np.array([T0z, T1z, T2z])
            Ty = _get_Vy(point, T)
            return np.asarray([Tx, Ty, Tz])
        dT = self(point=point, diff=1)
        if diff == 2:
            (
                (T0xx, T0zz, T0xz),
                (T1xx, T1zz, T1xz),
                (T2xx, T2zz, T2xz),
            ) = [
                Ti.diff(point, diff=2)
                for Ti in self.components
            ]
            Txx = np.array([T0xx, T1xx, T2xx])
            Tzz = np.array([T0zz, T1zz, T2zz])
            Txz = np.array([T0xz, T1xz, T2xz])
            Txy, Tyy, Tyz = _get_Vxy_Vyy_Vyz(point, T, dT)
            return np.asarray([[Txx, Txy, Txz],
                               [Txy, Tyy, Tyz],
                               [Txz, Tyz, Tzz]])
        raise ValueError("Unknown `diff` value: %s" % diff)


class DiscreteSym2TensorField(_DiscreteField):
    r"""Represents a symmetric 2-tensor field.

    Subclass this class and implement the _load_data() method to generate or
    load the data.
    """
    # pylint: disable=abstract-method

    def _eval(self, point, diff):
        if diff == 0:
            return eval_sym_axisym_matrix(
                self.components, point=point, diff=0
            )
        if diff == 1:
            return eval_sym_axisym_matrix(
                self.components,
                self(point=point, diff=0),
                point=point, diff=1
            )
        if diff == 2:
            return eval_sym_axisym_matrix(
                self.components,
                self(point=point, diff=0),
                self(point=point, diff=1),
                point=point, diff=2
            )
        raise NotImplementedError
