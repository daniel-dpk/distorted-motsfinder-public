r"""@package motsfinder.ipyutils

Utility functions for interactive IPython/Jupyter sessions.
"""

from .reloading import reload_all
from .plotctx import matplotlib_rc
from .plotting import plot_multi
from .plotting import plot_1d, plot_data, plot_curve, plot_polar, plot_mat
from .plotting import plot_data_disconnected
from .plotting3d import plot_2d, plot_scatter, plot_tri, plot_tri_cut
from .printing import disp
