r"""@package motsfinder.metric.discrete.construct

Construct an axisymmetric discrete metric from NumPy arrays.

The functions in this module help in supplying your slice data to the MOTS
Finder. The data should be prepared in form of a sequence of 6 component
matrices for the metric and extrinsic curvature, respectively. See the example
in metric_from_data().

Optionally, lapse and shift can be supplied in order to evaluate the spacetime
metric on points of the slice (by constructing a ..fourmetric.FourMetric).
Note that lapse and shift are not required to compute the *stability spectrum*
as these are guage choices just altering the coordinate system the 4-metric is
expressed in.


@b Examples

See documentation of metric_from_data().
"""

import numbers

import numpy as np

from ...utils import save_to_file, load_from_file
from .patch import GridPatch, DataPatch, BBox
from .metric import DiscreteMetric
from .tensors import DiscreteScalarField, DiscreteVectorField
from .tensors import DiscreteSym2TensorField


__all__ = [
    "metric_from_data",
    "NumericalMetric",
]


def metric_from_data(out_fname, res, origin, metric, curv, lapse=None,
                     shift=None, dtlapse=None, dtshift=None, overwrite=False,
                     interpolation='hermite5', fd_order=6):
    r"""Convert slice data from a list of matrices to a compatible form.

    This functions converts slice data for the metric, extrinsic curvature,
    (and optionally lapse and shift and their time derivatives) into a metric
    object that can be used e.g. to find MOTSs.

    **Note** that currently, all components need to be represented by single
    matrices each. The matrices need to have the same shape, resolution (grid
    spacing), and origin.

    @param out_fname
        File name to write the constructed metric data to (see below).
    @param res
        Spatial resolution of the data. Must be uniform across all fields and
        dimensions. For a grid spacing of `h`, we have `res = 1/h`.
    @param origin
        Physical location of the `[0, 0]` component of each field. Can be
        given as a float (location on the z-axis) or a 2- or 3-element tuple
        with the `(x,z)` or `(x,y,z)` components, respectively.
    @param metric
        Sequence of the 6 component matrices ``xx, xy, xz, yy, yz, zz`` of the
        metric. Each should be a matrix of shape ``(N, M)``, where `N` is the
        number of grid points in x-direction and `M` the number of grid points
        in z-direction. The domain is calculated from `origin`, `res` and
        these numbers `N` and `M`.
    @param curv
        Extrinsic curvature of the slice in the same format as `metric`.
        Alternatively, explicitly set this parameter `None` to imply vanishing
        extrinsic curvature (time-symmetry).
    @param lapse
        Lapse function as ``(N, M)`` matrix.
    @param shift
        Shift (contravariant) vector field as three ``(N, M)`` matrices.
    @param dtlapse
        As `lapse` but for the time derivative if nonzero.
    @param shift
        As `shift` but for the time derivative if nonzero.
    @param interpolation
        Kind of interpolation to do. Default is ``'hermite5'``. Possible
        values are those documented for .patch.DataPatch.set_interpolation().
    @param fd_order
        Finite difference convergence order for computing derivatives at grid
        points.


    @b Notes

    Since each found MOTS stores a reference to the metric, saving the curves
    to disk would create many duplicates of the full slice data. To avoid
    this, the saved metric object only keeps the name of a file containing all
    the required data. This function creates this file and then constructs a
    metric pointing to that file.


    @b Examples

    ```
        # Assume you have matrices containing NumPy arrays with the metric
        # components (in the x-z plane)
        g_xx = ... # np.ndarray of shape=(N, N)
        g_xy = ... # np.ndarray of shape=(N, N)
        g_xz = ... # np.ndarray of shape=(N, N)
        g_yy = ... # np.ndarray of shape=(N, N)
        g_yz = ... # np.ndarray of shape=(N, N)
        g_zz = ... # np.ndarray of shape=(N, N)

        # The same goes for the extrinsic curvature and optionally lapse and
        # shift
        K_xx = ...; ..., K_zz = ...;
        lapse = ...
        shift_x = ...; shift_y = ...; shift_z = ...;

        # Say the [0, 0] component of each array sits at x,y,z = 0,0,-5
        origin = -5.0

        # With grid spacing h, the `res` is given by:
        res = 1/h

        # Now, create the metric
        fname_data = 'data/my_numerical_metric_data.npy'
        g = metric_from_data(
            fname_data, res=res, origin=origin,
            metric=[g_xx, g_xy, g_xz, g_yy, g_yz, g_zz],
            curv=[K_xx, K_xy, K_xz, K_yy, K_yz, K_zz],
            lapse=lapse,
            shift=[shift_x, shift_y, shift_z],
        )

        # Let's store the metric itself too. Notice that this will be a small
        # file even for large numerical data sets.
        # NOTE: We use a *different* filename here!
        fname_metric = 'data/my_numerical_metric.npy'
        g.save(fname_metric)
    ```

    The last step above is not strictly required, you can also use the metric
    directly and then also skip the first line below:

    ```
        g = DiscreteMetric.load(fname_metric)

        find_mots(GeneralMotsConfig.preset(
            "discrete8",
            metric=g,
            hname='AH',
            out_folder='data/MOTSs',
            c_ref=StarShapedCurve.create_sphere(1.5, num=50, metric=g),
            ref_num=2, reparam=False,
            verbose=True, plot_steps=True,
        ))
    ```

    Obviously, some of the above settings like the initial guess `c_ref`, the
    resolution `num=50`, `ref_num`, etc. are highly dependent on the slice
    data.
    """
    if not out_fname.endswith('.npy'):
        out_fname += '.npy'
    _verify_data_input(metric, curv, shift, dtshift)
    metric = [np.asarray(comp) for comp in metric]
    patch = _construct_patch(res, origin, metric[0].shape)
    g = _construct_data_patch(metric, patch, rank=(0, 2))
    K = _construct_data_patch(curv, patch, rank=(0, 2))
    alp = _construct_data_patch([lapse], patch, rank=(0,))
    dtalp = _construct_data_patch([dtlapse], patch, rank=(0,))
    beta = _construct_data_patch(shift, patch, rank=(1,))
    dtbeta = _construct_data_patch(dtshift, patch, rank=(1,))
    data = dict(metric=g, curv=K, lapse=alp, dtlapse=dtalp, shift=beta,
                dtshift=dtbeta, patch=patch, res=res)
    save_to_file(out_fname, data=data, showname='slice data',
                 overwrite=overwrite)
    return NumericalMetric(data_file=out_fname, data=data,
                           interpolation=interpolation, fd_order=fd_order)


def _verify_data_input(metric, curv, shift, dtshift):
    r"""Basic sanity checks for the given data objects."""
    if len(metric) != 6:
        raise ValueError("Need matrices for 6 metric components: "
                         "xx, xy, xz, yy, yz, zz")
    if curv is not None and len(curv) != 6:
        raise ValueError("Need matrices for 6 curv components: "
                         "xx, xy, xz, yy, yz, zz")
    if shift is not None and len(shift) != 3:
        raise ValueError("Need matrices for 3 shift components: "
                         "x, y, z")
    if dtshift is not None and len(dtshift) != 3:
        raise ValueError("Need matrices for 3 dtshift components: "
                         "x, y, z")


def _construct_patch(res, origin, shape):
    r"""Create the grid patch for the supplied settings."""
    if isinstance(origin, numbers.Number) or len(origin) == 1:
        # origin given as z value on the z-axis
        origin = [0., 0., origin]
    elif len(origin) == 2:
        # given as x-z pair for y=0
        origin = [origin[0], 0., origin[1]]
    origin = np.array(origin)
    deltas = 1./res * np.identity(3)
    if len(shape) == 2:
        shape = (shape[0], 1, shape[1])
    box = BBox(lower=[0, 0, 0], upper=shape)
    patch = GridPatch(origin=origin, deltas=deltas, box=box)
    return patch


def _construct_data_patch(components, patch, rank):
    r"""Create the patches containing the actual data.

    @param components
        List of `n` component matrices.
    @param patch
        Patch object defining the physical location/size of the matrices.
    @param rank
        Rank of the tensor. Note that for a rank `(0, 2)` tensor we assume the
        tensor to be symmetric (invariant under transposition of the 3x3
        matrix constructed at each point from the given components). Hence, we
        expect only 6 instead of 9 components here.
    """
    if components is None or all([comp is None for comp in components]):
        return None
    if rank == (0, 2):
        symmetries = ['even', 'even', 'odd', 'even', 'odd', 'even']
    elif rank == (1,):
        symmetries = ['odd', 'odd', 'even']
    elif rank == (0,):
        symmetries = ['even']
    else:
        raise ValueError("Unsupported rank: %s" % (rank,))
    if len(symmetries) != len(components):
        raise ValueError("Incorrect number of components for tensor of rank "
                         "%s: %s" % (rank, len(components)))
    def _to_array(mat):
        mat = np.asarray(mat)
        if len(mat.shape) == 2:
            # missing y dimension
            mat = mat.reshape(mat.shape[0], 1, mat.shape[1])
        return mat
    components = [_to_array(comp) for comp in components]
    return [
        DataPatch.from_patch(patch, mat, sym)
        for mat, sym in zip(components, symmetries)
    ]


class NumericalMetric(DiscreteMetric):
    r"""Metric containing given slice data."""
    def __init__(self, data_file, data=None, interpolation='hermite5',
                 fd_order=6):
        r"""Create a metric containing a full slice definition.

        This object is usually created from metric_from_data(). See its
        documentation for more information and examples.

        @param data_file
            Filename of the data as stored by metric_from_data().
        @param data
            Optional parameter containing the data in the given file. This
            avoids loading the data to memory twice should it already be
            available.
        @param interpolation
            Kind of interpolation to do. Default is ``'hermite5'``. Possible
            values are those documented for
            .patch.DataPatch.set_interpolation().
        @param fd_order
            Finite difference convergence order for computing derivatives at
            grid points.
        """
        super().__init__()
        self._data_file = data_file
        if data is None:
            data = load_from_file(data_file)
        self.__data = data
        curv, lapse, shift, dtlapse, dtshift, res = [
            data[key] for key in ("curv", "lapse", "shift", "dtlapse",
                                  "dtshift", "res")
        ]
        self._metric = _Sym2TensorField(self, 'metric')
        self._curv = _Sym2TensorField(self, 'curv') if curv else None
        self._lapse = _ScalarField(self, 'lapse') if lapse else None
        self._shift = _VectorField(self, 'shift') if shift else None
        self._dtlapse = _ScalarField(self, 'dtlapse') if dtlapse else None
        self._dtshift = _VectorField(self, 'dtshift') if dtshift else None
        self._res = res
        self.set_interpolation(interpolation)
        self.set_fd_order(fd_order)

    def all_field_objects(self):
        return [
            self._metric, self._curv, self._lapse, self._shift,
            self._dtlapse, self._dtshift,
        ]

    @property
    def data_file(self):
        r"""Filename of the data file containing the prepared slice data."""
        return self._data_file

    def set_data_file(self, new_fname):
        r"""Convenience method for setting a new filename for the data file.

        This may be used to specify a new location of the data file. It is
        assumed that it contains the *same* data as before, no check is done
        to ensure this.
        """
        self._data_file = new_fname

    @property
    def res(self):
        r"""Resolution 1/h of the data grids."""
        return self._res

    @property
    def data_dict(self):
        r"""Dictionary containing """
        if self.__data is None:
            self.__data = load_from_file(self.data_file)
        return self.__data

    def __getstate__(self):
        r"""Return a picklable state object."""
        data = self.__data
        try:
            self.__data = None
            state = self.__dict__.copy()
            return state
        finally:
            self.__data = data

    def unload_data(self):
        self.__data = None
        super().unload_data()

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


class _Sym2TensorField(DiscreteSym2TensorField):
    def __init__(self, metric, key):
        super().__init__(metric)
        self.__key = key

    def _load_data(self):
        return self.metric.data_dict[self.__key]


class _ScalarField(DiscreteScalarField):
    def __init__(self, metric, key):
        super().__init__(metric)
        self.__key = key

    def _load_data(self):
        return self.metric.data_dict[self.__key]


class _VectorField(DiscreteVectorField):
    def __init__(self, metric, key):
        super().__init__(metric)
        self.__key = key

    def _load_data(self):
        return self.metric.data_dict[self.__key]
