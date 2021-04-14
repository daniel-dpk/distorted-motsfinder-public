r"""@package motsfinder.metric.discrete.metric

Base class for discrete metrics.

The DiscreteMetric class is an abstract class implementing most of the
functionality required for a ..base._ThreeMetric. The missing part is the
definition of actual numerical data. There are currently two implementations
of this abstract class, serving at the same time as examples:
    * .discretize.DiscretizedMetric
    * ..simulation.siometric.SioMetric
"""

from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

from six import add_metaclass
import numpy as np

from ...utils import save_to_file, load_from_file
from ..base import _ThreeMetric


__all__ = [
    "DiscreteMetric",
]


@add_metaclass(ABCMeta)
class DiscreteMetric(_ThreeMetric):
    r"""Base class for discrete axisymmetric metrics.

    This subclass of ..base._ThreeMetric implements the memory management of
    numerical data for a metric and other fields defining the geometry and
    embedding of the slice into spacetime.

    Subclasses should implement:
        * _get_metric() - constructing the metric as a .tensors.DiscreteSym2TensorField
        * all_field_objects() - return all fields as a list

    The reason to have the latter of the two is to allow metrics that don't
    supply a lapse and/or shift field but still have an easy way to keep track
    of all fields (for caching and memory management purposes).

    Optionally, subclasses may implement:
        * get_curv() - extrinsic curvature (.tensors.DiscreteSym2TensorField)
        * get_lapse() - lapse function (.tensors.DiscreteScalarField)
        * get_shift() - lapse function (.tensors.DiscreteVectorField)
        * get_dtlapse() - (.tensors.DiscreteScalarField)
        * get_dtshift() - (.tensors.DiscreteVectorField)

    If all of these are supplied, then the full 4-metric can be evaluated on
    the slice.
    """

    def __init__(self):
        r"""This base constructor initializes the properties."""
        super(DiscreteMetric, self).__init__()
        ## The metric discrete field object.
        self._metric_field = None
        ## Whether all matrices should be saved.
        self._save_full_data = False

    @abstractmethod
    def _get_metric(self):
        r"""Abstract method to create/load numerical metric data.

        This method should return a .tensors.DiscreteSym2TensorField built
        from the component matrices of the metric. It is called only once
        (lazily) and the object is *not* destroyed even when calling
        unload_data(). Instead, the field's unload_data() method is called.

        This method is the hook for subclasses to implement their method of
        generating/loading the numerical data.
        """
        pass

    @abstractmethod
    def all_field_objects(self):
        r"""Abstract method supplying all defined field objects as a list.

        See .discretize.DiscretizedMetric for a simple example.
        """
        pass

    @property
    def field(self):
        r"""Field attribute containing the actual field object.

        The object is lazily loaded (i.e. on first access) and kept as
        instance attribute. Note that this access does not imply that data is
        loaded or generated. This is handled by the field object itself.

        The result is a .tensors.DiscreteSym2TensorField.
        """
        if self._metric_field is None:
            self._metric_field = self._get_metric()
        return self._metric_field

    def set_interpolation(self, interpolation):
        r"""Set the interpolation for all fields/components.

        Refer to .patch.DataPatch.set_interpolation() for details.
        """
        for field in self.all_field_objects():
            if field:
                field.set_interpolation(interpolation)

    def set_fd_order(self, fd_order):
        r"""Set the finite difference derivative order of accuracy."""
        for field in self.all_field_objects():
            if field:
                field.set_fd_order(fd_order)

    @contextmanager
    def temp_interpolation(self, interpolation, fd_order=None):
        prev_interp = self.field.components[0].get_interpolation()
        prev_fd_order = self.field.components[0].fd_order
        try:
            self.set_interpolation(interpolation)
            if fd_order is not None:
                self.set_fd_order(fd_order)
            yield
        finally:
            self.set_interpolation(prev_interp)
            if fd_order is not None:
                self.set_fd_order(prev_fd_order)

    @property
    def save_full_data(self):
        r"""Read/write property specifying whether the full grid data should
        be stored.

        This is `False` by default. If set to `True` instead, saving this
        object (or more generally "pickling" it) will include the numerical
        data on the full grid. For large slice data, this will basically store
        the full slice.
        """
        return self._save_full_data
    @save_full_data.setter
    def save_full_data(self, value):
        self._save_full_data = value

    def save(self, filename, full_data=False, overwrite=False, verbose=True):
        r"""Save the metric to disk.

        @param filename
            The file to store the data in. The extension ``'.npy'`` will be
            added if not already there.
        @param full_data
            Whether to store all the matrix data (if `True`) or regenerate it
            on-demand after loading (if `False`). Default is `False`.
        @param overwrite
            Whether to overwrite an existing file with the same name. If
            `False` (default) and such a file exists, a `RuntimeError` is
            raised.
        @param verbose
            Whether to print a message upon success. Default is `True`.
        """
        prev_value = self.save_full_data
        try:
            self.__dict__['_prev_save_full_data'] = prev_value
            self.save_full_data = full_data
            if full_data:
                self.load_data()
            save_to_file(
                filename, self, overwrite=overwrite, verbose=verbose,
                showname=self.__class__.__name__
            )
        finally:
            self.save_full_data = prev_value
            self.__dict__.pop('_prev_save_full_data')

    @staticmethod
    def load(filename):
        r"""Static function to load a metric from disk."""
        metric = load_from_file(filename)
        metric.save_full_data = metric.__dict__.pop(
            '_prev_save_full_data', False
        )
        return metric

    def constraints(self, point, norm=False):
        r"""Compute the Hamiltonian and momentum constraints.

        If the constraints are satisfied exactly, the returned values will be
        zero. However, for numerical simulation data this will rarely ever be
        the case. The returned values can then be used to determine the
        accuracy of the simulation, e.g. by comparing them at different
        spatial resolutions (grid densities).

        Note that this function is not optimized in any way and thus will
        perform poorly when evaluated on a large number of points.

        @return A 2-tuple ``(scal_constr, vec_constr)``, where `scal_constr`
            is a float representing the Hamiltonian constraint
            \f$K_{ab}K^{ab} - (K^c_{\ c})^2 - R\f$ and `vec_constr` the
            momentum constraint \f$D^a(K_{ab} - g_{ab} K^c_{\ c})\f$. The
            latter is a (3-D) covector (indices downstairs). Here, `D` is the
            Levi-Civita covariant derivative compatible with `g`, the slice's
            Riemannian 3-metric, `K` is the extrinsic curvature of the
            slice and `R` the scalar curvature of the slice. If ``norm=True``,
            the second element is a float, the `g`-norm of the momentum
            constraint.

        @param point
            The point at which to compute the constraints.
        @param norm
            If `True`, compute the `g`-norm of the momentum constraint instead
            of the covector. Default is `False`.
        """
        g_inv = self.diff(point, inverse=True, diff=0)
        curv = self.get_curv() # pylint: disable=assignment-from-none
        K = curv(point)
        R = self.ricci_scalar(point)
        K_up = g_inv.dot(g_inv.dot(K).T).T
        KK = K_up.dot(K.T).trace()
        trK = g_inv.dot(K).trace()
        scal_constr = KK - trK**2 - R
        dg = self.diff(point, diff=1)
        dK = np.asarray(curv(point, diff=1))
        G = self.christoffel(point)
        vec_constr = (
            np.einsum('ac,cab', g_inv, dK)
            - np.einsum('ac,ica,ib', g_inv, G, K)
            - np.einsum('ac,icb,ai', g_inv, G, K)
            - np.einsum('ac,bac', g_inv, dK)
            + np.einsum('bac,ac', dg, K_up)
        )
        if norm:
            vec_constr = np.sqrt(g_inv.dot(vec_constr).dot(vec_constr))
        return scal_constr, vec_constr

    def load_data(self, *which):
        r"""Load/generate the full numerical data.

        Without arguments, all fields are loaded to memory. If one or more
        arguments are given, only those fields are loaded. Possible arguments
        are: ``metric, curv, lapse, shift, dtlapse, dtshift``.
        """
        if len(which) > 1:
            for field_name in which:
                self.load_data(field_name)
            return
        field_name, = which if which else [None]
        def _load(field):
            if field:
                field.load_data()
        if field_name is None:
            for field in self.all_field_objects():
                _load(field)
        elif field_name == "metric":
            _load(self.field)
        elif field_name == "curv":
            _load(self.get_curv())
        elif field_name == "lapse":
            _load(self.get_lapse())
        elif field_name == "shift":
            _load(self.get_shift())
        elif field_name == "dtlapse":
            _load(self.get_dtlapse())
        elif field_name == "dtshift":
            _load(self.get_dtshift())
        else:
            raise ValueError("Unknown field: %s" % field_name)

    def unload_data(self):
        r"""Free memory from all numerical field matrix data."""
        for field in self.all_field_objects():
            if field:
                field.unload_data()
        self.reset_cache()

    def release_file_handle(self):
        r"""Convenience method to signal child classes they should release any file handles.

        This does nothing by default. Subclasses may implement this method to
        free access to any files currently opened by this class. It may be
        called by users of this class in case they deem file access to be
        finished (e.g. after loading all required data).
        """
        pass

    def reset_cache(self):
        r"""Reset the cache of all fields."""
        for field in self.all_field_objects():
            if field:
                field.reset_cache()

    def grid(self, xz_plane=True, ghost=0, full_output=False):
        r"""Convenience method delegating to .patch.DataPatch.grid()."""
        return self.field.components[0].grid(
            xz_plane=xz_plane, ghost=ghost, full_output=full_output
        )

    def snap_to_grid(self, point):
        r"""Convenience method delegating to .patch.DataPatch.snap_to_grid()."""
        return self.field.components[0].snap_to_grid(point)

    @property
    def shape(self):
        r"""Shape of the domain, \ie of individual component matrices."""
        return self.field.components[0].shape

    @property
    def box(self):
        r"""Index bounding box of the full domain."""
        return self.field.components[0].box

    @property
    def domain(self):
        return self.field.components[0].domain

    @property
    def safe_domain(self):
        return self.field.components[0].safe_domain

    def component_matrix(self, i, j):
        r"""Return the DataPatch of a component of the metric."""
        gij = self.field.components
        if i == j == 0:
            return gij[0]
        if i == j == 1:
            return gij[3]
        if i == j == 2:
            return gij[5]
        if j < i:
            i, j = j, i
        if i == 0 and j == 1:
            return gij[1]
        if i == 0 and j == 2:
            return gij[2]
        if i == 1 and j == 2:
            return gij[4]
        raise ValueError("Unknown component: (%s, %s)" % (i, j))

    def _mat_at(self, point):
        return self.diff(point, diff=0)

    def diff(self, point, inverse=False, diff=1):
        r"""Compute (derivatives of) the metric tensor at a given point.

        @return For ``diff=0``, returns the 3x3 matrix representing the metric
            interpolated at `point`, i.e. \f$g_{ij}\f$.
            If ``diff=1``, returns ``dg[i,j,k]``, where the indices mean
            \f$\partial_i g_{jk}\f$ and if ``diff=2``, returns
            ``ddg[i,j,k,l]`` with indices \f$\partial_i\partial_j g_{kl}\f$.
            In each case, if ``inverse=True`` the inverse metric is used with
            indices upstairs.

        @param point
            The point at which to compute.
        @param inverse
            Whether to compute (derivatives of) the inverse metric. Default is
            `False`.
        @param diff
            Derivative order to compute. Default is `1`.
        """
        if inverse:
            return self._compute_inverse_diff(point, diff=diff)
        return self.field(point, diff=diff)
