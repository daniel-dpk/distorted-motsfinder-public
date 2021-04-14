r"""@package motsfinder.ndsolve.bases

This package contains the various bases that can be used in the pseudospectral
solver.
"""

from .cheby import ChebyBasis
from .trig import SineBasis, CosineBasis, FourierBasis

def choose_basis_class(expr):
    from ...exprs.trig import SineSeries, CosineSeries, FourierSeries
    for basis_cls in [SineBasis, CosineBasis, FourierBasis]:
        series_cls = expr.__class__
        if basis_cls.get_series_cls() is series_cls:
            return basis_cls
    raise TypeError("Could not find correct series class.")
