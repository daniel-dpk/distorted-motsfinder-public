r"""@package motsfinder.axisym.trackmots

Helpers to track a MOTS through simulation slices.

See .trackmots.tracker for more information and examples.
"""

from .tracker import MOTSTracker, track_mots, find_slices
from .props import compute_props, max_constraint_along_curve
from .props import compute_tube_signature
from .props import compute_tev_divergence
from .props import compute_time_evolution_vector
from .props import compute_shear_hat_scalar
from .props import compute_shear_hat2_integral
from .props import compute_xi_hat2_integral
from .props import compute_xi_tilde2_integral
from .props import compute_xi_vector
from .props import compute_xi2_integral
from .props import compute_xi_scalar
from .props import compute_xi_hat_scalar
from .props import compute_xi_tilde_scalar
from .props import compute_surface_gravity
from .props import compute_surface_gravity_tev
from .props import compute_extremality_parameter
from .props import compute_timescale_tau2
from .props import compute_timescale_tau2_option01
from .props import compute_timescale_tau2_option02
from .props import compute_timescale_tau2_option03
from .props import compute_timescale_tau2_option04
from .props import compute_timescale_T2
from .findslices import find_slices
from .optimize import optimize_bipolar_scaling
