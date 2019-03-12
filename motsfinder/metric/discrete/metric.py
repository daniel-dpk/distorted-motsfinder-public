r"""@package motsfinder.metric.discrete.metric

Metrics interpolated from data on a grid.
"""

from abc import ABCMeta, abstractmethod

from six import add_metaclass
import numpy as np

from ...utils import save_to_file, load_from_file
from ..base import _ThreeMetric


__all__ = [
    "DiscreteMetric",
]


@add_metaclass(ABCMeta)
class DiscreteMetric(_ThreeMetric):
    def __init__(self):
        super(DiscreteMetric, self).__init__()
        ## Whether Lagrange interpolation should be done.
        self._interpolate = True
        ## The point (as 3-tuple) for which the cached values are valid.
        self._prev_pt = None
        ## The metric discrete field object.
        self._metric_field = None
        ## Whether all matrices should be saved.
        self._save_full_data = False

    @abstractmethod
    def _get_metric(self):
        pass

    @abstractmethod
    def all_field_objects(self):
        pass

    def __getstate__(self):
        r"""Return a picklable state object representing the whole metric."""
        state = self.__dict__.copy()
        state['_prev_pt'] = None
        return state

    @property
    def field(self):
        if self._metric_field is None:
            self._metric_field = self._get_metric()
        return self._metric_field

    @property
    def interpolate(self):
        r"""Read/write property indicating whether interpolation should be performed.

        This is `True` by default. If set to `False`, each evaluation will
        first find the closest grid point to evaluate at and then compute
        all requested values at that grid point without need of interpolation.
        """
        return self._interpolate
    @interpolate.setter
    def interpolate(self, value):
        self.reset_cache()
        self._interpolate = value

    @property
    def save_full_data(self):
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
            Whether to print a message upon success.
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
        r"""Static function to load an expression object from disk."""
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
        perform poorly when evaluated on a large number of points (especially
        when ``interpolate=True`` and interpolation is performed).

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
        point = self.prepare_for_point(point)
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

    def prepare_for_point(self, point):
        r"""Prepare object for evaluation at given point (called internally).

        This is called internally e.g. by diff(); users of this class usually
        don't have to call this directly. Its responsibility is to check
        whether the internal cache has to be updated. In case the
        `interpolate` property is set to `False`, the point will be snapped to
        the nearest grid point and returned. Otherwise, the point is returned
        unchanged.
        """
        if not self._interpolate:
            point = self.snap_to_grid(point)
        x, y, z = point
        if (x, y, z) == self._prev_pt:
            return point
        self._prev_pt = (x, y, z)
        self.reset_cache()
        return point

    def load_data(self, *which):
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
        for field in self.all_field_objects():
            if field:
                field.unload_data()
        self.reset_cache()

    def release_file_handle(self):
        pass

    def reset_cache(self):
        for field in self.all_field_objects():
            if field:
                field.reset_cache()

    def grid(self, xz_plane=True, ghost=0, full_output=False):
        return self.field.components[0].grid(
            xz_plane=xz_plane, ghost=ghost, full_output=full_output
        )

    def snap_to_grid(self, point):
        return self.field.components[0].snap_to_grid(point)

    @property
    def shape(self):
        r"""Shape of the domain, \ie of individual component matrices."""
        return self.field.components[0].shape

    @property
    def box(self):
        r"""Index bounding box of the full domain."""
        return self.field.components[0].box

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
