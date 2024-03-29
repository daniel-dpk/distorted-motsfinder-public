r"""@package motsfinder.metric.simulation.siometric

The simulation-data based metric class for axisymmetric cases.

See the package docstring of motsfinder.metric.simulation (in the file
`motsfinder/metric/simulation/__init__.py`) for more general information and
examples.

The classes in this module implement the field classes of ..discrete.tensors
and the ..discrete.metric.DiscreteMetric class.
"""

from ..discrete import DiscreteMetric, DiscreteScalarField
from ..discrete import DiscreteVectorField, DiscreteSym2TensorField
from .sioproject import SioProject


__all__ = [
    "SioMetric",
]


class SioScalarField(DiscreteScalarField):
    r"""Represents an axisymmetric scalar field with one component."""

    def __init__(self, sio_metric, field_name):
        r"""Create a scalar field using the given metric.

        The data file containing the actual data is inferred from the given
        metric object. The `field_name` should be the field's name in the
        data, e.g. ``'admbase::lapse'``.
        """
        super(SioScalarField, self).__init__(sio_metric)
        self.__field_name = field_name

    def _load_data(self):
        return [
            self.metric.project.field_component_matrix(self.__field_name, c)
            for c in ['scalar']
        ]


class SioVectorField(DiscreteVectorField):
    r"""Represents an axisymmetric vector field with three spatial components."""

    def __init__(self, sio_metric, field_name):
        r"""Create a vector field using the given metric.

        The data file containing the actual data is inferred from the given
        metric object. The `field_name` should be the field's name in the
        data, e.g. ``'admbase::shift'``.
        """
        super(SioVectorField, self).__init__(sio_metric)
        self.__field_name = field_name

    def _load_data(self):
        return [
            self.metric.project.field_component_matrix(self.__field_name, c)
            for c in ['0', '1', '2']
        ]


class SioSym2TensorField(DiscreteSym2TensorField):
    r"""Represents an axisymmetric symmetric 2-tensor field."""

    def __init__(self, sio_metric, field_name):
        r"""Create a symmetric 2-tensor field using the given metric.

        The data file containing the actual data is inferred from the given
        metric object. The `field_name` should be the field's name in the
        data, e.g. ``'admbase::curv'``.
        """
        super(SioSym2TensorField, self).__init__(sio_metric)
        self.__field_name = field_name

    def _load_data(self):
        return [
            self.metric.project.field_component_matrix(self.__field_name, c)
            for c in ['00', '01', '02', '11', '12', '22']
        ]


class SioMetric(DiscreteMetric):
    r"""Axisymmetric Riemannian 3-metric of a slice read from simulation data.

    The filename of the data file is stored as instance variable. When storing
    this object to disk (or in general when pickling it), the actual data is
    skipped and only the configuration and file name is stored. Upon loading
    (unpickling), the data is not immediately read from disk. Instead, it is
    loaded lazily, i.e. when first needed. This allows efficient loading of,
    for example, curve objects for plotting without loading the slice into
    memory.

    See the package docstring of motsfinder.metric.simulation (in the file
    `motsfinder/metric/simulation/__init__.py`) for more general information
    and examples.
    """

    def __init__(self, hdf5_file, interpolation='hermite5', fd_order=6):
        r"""Create a 3-metric from a simulation data file.

        @param hdf5_file
            The filename to load.
        @param interpolation
            Kind of interpolation to perform. Refer to
            ..discrete.patch.DataPatch.set_interpolation() for available
            options.
            Default is ``'hermite5'``.
        @param fd_order
            Order of accuracy for finite difference differentiation stencils.
            Default is `6`.
        """
        super(SioMetric, self).__init__()
        p = SioProject(hdf5_file)
        self._project = p
        ## The filename containing the simulation data.
        self._data_file = hdf5_file
        ## Coordinate time of the slice.
        self._time = p.time()
        ## Coordinate time delta.
        self._delta_time = p.delta_time()
        ## Iteration number of the slice.
        self._iteration = p.iteration()
        self._curv = create_field(SioSym2TensorField, self, 'admbase::curv')
        self._lapse = create_field(SioScalarField, self, 'admbase::lapse')
        self._dtlapse = create_field(SioScalarField, self, 'admbase::dtlapse')
        self._shift = create_field(SioVectorField, self, 'admbase::shift')
        self._dtshift = create_field(SioVectorField, self, 'admbase::dtshift')
        self._all_fields = [
            self.field, self._curv, self._lapse, self._shift,
            self._dtlapse, self._dtshift,
        ]
        self.set_interpolation(interpolation)
        self.set_fd_order(fd_order)

    def _get_metric(self):
        return SioSym2TensorField(self, 'admbase::metric')

    def all_field_objects(self):
        return self._all_fields

    def load_data(self, *which):
        super().load_data(*which)
        if not which:
            # all data loaded, no need to access anything in the future
            self.release_file_handle()

    def unload_data(self):
        super().unload_data()
        self.release_file_handle()

    def release_file_handle(self):
        self._project = None

    @property
    def project(self):
        r"""SioProject pointing to the slice data (lazily loaded)."""
        if self._project is None:
            self._project = SioProject(self.data_file)
        return self._project

    def __getstate__(self):
        r"""Return a picklable state object representing the whole metric."""
        self.reset_cache()
        state = self.__dict__.copy()
        state['_project'] = None
        return state

    @property
    def data_file(self):
        r"""Filename of the data file containing the numerical simulation data."""
        return self._data_file

    @property
    def time(self):
        r"""Simulation coordinate time of the slice.

        Note that requesting this attribute will not trigger the data to be
        loaded.
        """
        return self._time

    @property
    def delta_time(self):
        r"""Time step distance in coordinate time.

        Note that requesting this attribute will not trigger the data to be
        loaded.
        """
        return self._delta_time

    @property
    def iteration(self):
        r"""Iteration number of the current slice.

        Note that requesting this attribute will not trigger the data to be
        loaded.
        """
        return self._iteration

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


def create_field(cls, metric, field_name):
    r"""Create a field object only if the slice contains its data.

    @param cls
        Field class to create (e.g. SioVectorField).
    @param metric
        The metric to connect with the field object.
    @param field_name
        Name under which this field is stored in the numerical data, for
        example ``"admbase::lapse"``.
    """
    if metric.project.has_field(field_name):
        return cls(metric, field_name)
    return None
