r"""@package motsfinder.metric.discrete

Metrics interpolated from data on a grid.

The classes in this module mainly serve as base classes for concrete discrete
metric implementations. An exception is the .metric.DiscreteMetric class,
which is used mainly for testing discrete metrics against analytical metrics
and determine e.g. required resolutions.
"""

from .metric import DiscreteMetric
from .numerical import GridDataError
from .patch import BBox, GridPatch, DataPatch
from .tensors import DiscreteScalarField, DiscreteVectorField
from .tensors import DiscreteSym2TensorField
from .utils import ConstraintAnalyzer
from .discretize import DiscretizedMetric
