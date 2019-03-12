r"""@package motsfinder.metric.discrete

Metrics interpolated from data on a grid.
"""

from .metric import DiscreteMetric
from .numerical import GridDataError
from .patch import BBox, GridPatch, DataPatch
from .tensors import DiscreteScalarField, DiscreteVectorField
from .tensors import DiscreteSym2TensorField
from .utils import ConstraintAnalyzer
from .discretize import DiscretizedMetric
