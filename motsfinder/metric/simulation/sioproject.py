r"""@package motsfinder.metric.simulation.sioproject

SimulationIO low-level wrapper functions and classes.

See the package docstring of motsfinder.metric.simulation (in the file
`motsfinder/metric/simulation/__init__.py`) for more general information and
examples.
"""

import os.path as op

import numpy as np

from ..discrete import BBox, GridPatch, DataPatch


__all__ = [
    "SioProject",
]


class SioProject():
    r"""Wraps a simulation data file to conveniently extract data from it.

    This class can be used to get field (or field component) data and other
    information stored in the data file of a slice of spacetime.

    Note that the data is assumed to be axisymmetric.
    """

    def __init__(self, hdf5_file):
        r"""Create a new project from a simulation slice data file.

        @param hdf5_file
            The filename to load.
        """
        if not op.isfile(hdf5_file):
            raise FileNotFoundError("File not found: %s" % hdf5_file)
        import SimulationIO # pylint: disable=import-outside-toplevel
        ## The SimulationIO project object.
        self.project = SimulationIO.readProjectHDF5(hdf5_file)
        self._patches = None

    @property
    def patches(self):
        r"""List of patches (GridPatch) in the data."""
        if self._patches is None:
            self._patches = [GridPatch(*data) for data in self.coord_blocks()]
        return self._patches

    def get_value(self, data):
        r"""Conveniently extract data in the correct type.

        Given a "data object", this checks the value type and extracts the
        respective data. Returns `None` if the value type is not recognized.
        """
        if data.value_type == data.type_double:
            return data.getValueDouble()
        if data.value_type == data.type_int:
            return data.getValueInt()
        if data.value_type == data.type_string:
            return data.getValueString()
        return None

    def total_bbox(self):
        r"""Return the index bounding box of the full patch system.

        This assumes the first patch to contain the minimum x and z
        coordinates and the last patch to contain the maximum x and z
        coordinates covered by the data.

        @return A 2 by 3 matrix with the first index being `0` for "lower" and
            `1` for "upper" and the second indicating the axis (`x,y,z`).
        """
        patches = self.patches
        # indices [i,j]: [lower,upper][i], [x,y,z][j]
        return np.asarray(
            [[patches[0].box.lower[i] for i in range(3)]]
            + [[patches[-1].box.upper[i] for i in range(3)]]
        )

    def total_shape(self):
        r"""Shape a matrix needs to have to store all data."""
        bbox = self.total_bbox()
        return [bbox[1,i]-bbox[0,i] for i in range(3)]

    def has_field(self, field_name):
        r"""Return whether data for a particular field is available."""
        return field_name in self.project.fields()

    def field_blocks(self, field, component=None, read=False):
        r"""Generator to iterate over all blocks/patches and extract data.

        @param field
            Either the name of the field as a string, or the field object
            itself.
        @param component
            The component to extract (if any) as a string. If not specified,
            the block objects are returned by themselves. If given, that
            component is extracted.
        @param read
            Whether to read and copy the data contained in this block.
        """
        if isinstance(field, str):
            field = self.project.fields()[field]
        discrete_field = field.discretefields().values()[0]
        blocks = discrete_field.discretefieldblocks()
        for block in blocks.value_iterator():
            if component is None:
                result = block
            else:
                result = block.discretefieldblockcomponents()[component]
            if read:
                data = result.copyobj()
                shape = data.shape()
                data = np.asarray(data.readData_double())
                Nx, Ny, Nz = [shape[i] for i in range(3)]
                result = data.reshape((Nz, Ny, Nx)).T
            yield result

    def field_blocks_tensor3d(self, field):
        r"""Return a generator to iterate over patches and their data.

        The data consists of a tuple of the six independent components
        (matrices) of the symmetric tensor field. The order of components is
        `xx`, `xy==yx`, `xz==zx`, `yy`, `yz==zy`, `zz`.

        @param field
            Field name or object as understood by field_blocks().
        """
        comp = ['00', '01', '02', '11', '12', '22']
        blocks = [self.field_blocks(field, c, read=True) for c in comp]
        return zip(self.patches, zip(*blocks))

    def coord_blocks(self):
        r"""Generator to iterate over all coordinate blocks/patches.

        Each element we yield consists of the `origin` of the patch, the delta
        vectors representing the grid structure, and the bounding box.
        """
        p = self.project
        fields = p.coordinatesystems()['cctkGH.space'].coordinatefields()
        xyz_fields = [
            fields['cctkGH.space[%s]' % i].field() for i in range(3)
        ]
        for xyz in zip(*[self.field_blocks(f, 'scalar') for f in xyz_fields]):
            ranges = [c.datarange() for c in xyz]
            box_obj = ranges[0].box()
            lower, upper = box_obj.lower(), box_obj.upper()
            box = BBox(
                lower=[lower[i] for i in range(3)],
                upper=[upper[i] for i in range(3)]
            )
            origin = np.asarray([r.origin() for r in ranges])
            deltas = np.asarray([r.delta() for r in ranges])
            yield origin, deltas, box

    def field_component_matrix(self, field, component):
        r"""Construct one big matrix by stitching together all patches of a field component.

        The result is a DataPatch object containing the full data of the
        component of the specified field. Note that only the ``y==0`` slice of
        the data is extracted even if the data contains ghost points in the
        y-direction.

        @param field
            The field object or name (as understood by field_blocks()).
        @param component
            The field's component to extract the data of.
        """
        offsets = self.total_bbox()[0,:]
        shape = self.total_shape()
        shape[1] = 1
        full_mat = np.zeros(shape)
        for patch, mat in zip(self.patches,
                              self.field_blocks(field, component, True)):
            box = patch.box
            full_mat[box.lower[0]-offsets[0]:box.upper[0]-offsets[0],
                     0,
                     box.lower[2]-offsets[2]:box.upper[2]-offsets[2]] = mat[:,patch.yidx,:]
        origin = self.patches[0].origin.copy()
        lower = self.patches[0].box.lower
        upper = self.patches[-1].box.upper
        # We extracted only the y=0 data above, so the origin and bounding box
        # information needs to be updated.
        origin[1] = 0
        lower[1] = 0
        upper[1] = 1
        box = BBox(lower=lower, upper=upper)
        return DataPatch(
            origin=origin,
            deltas=self.patches[0].deltas,
            box=box,
            mat=full_mat,
            symmetry='odd' if component in ('02', '12', '0', '1') else 'even',
        )

    def metric_component_matrix(self, component):
        r"""Stitch together all patches of a component of the metric.

        The result is one DataPatch object containing the data in one big
        matrix.
        """
        return self.field_component_matrix('admbase::metric', component)

    def curv_component_matrix(self, component):
        r"""Stitch together all patches of a component of the extrinsic curvature.

        The result is one DataPatch object containing the data in one big
        matrix.
        """
        return self.field_component_matrix('admbase::curv', component)

    def get_single_parameter(self, name):
        r"""Get the value of a single-valued parameter.

        See time() for an example.

        @param name
            Name of the parameter to retrieve the value of.
        """
        params = self.project.parameters()[name]
        values = params.parametervalues().values()
        assert len(values) == 1
        return self.get_value(values[0])

    def time(self):
        r"""Return the time parameter of the current slice."""
        return self.get_single_parameter('cctk_time')

    def delta_time(self):
        r"""Return the time delta in coordinate time."""
        return self.get_single_parameter('cctk_delta_time')

    def iteration(self):
        r"""Return the iteration number of the current slice."""
        return self.get_single_parameter('iteration')

    def cactus_version(self):
        r"""Return the Cactus version as stored in the data."""
        return self.get_single_parameter('Cactus version')
