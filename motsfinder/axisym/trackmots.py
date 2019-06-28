r"""@package motsfinder.axisym.trackmots

Track a single MOTS through slices of a simulation.

@b Examples

```
    # Example of a run finding the AH in a sequence of slices.
    track_mots(
        hname='AH',
        sim_dir='data/simulation/brill-lindquist-res256',
        out_base_dir='results',
        folder='BL-res256/run1/AH',
        initial_guess='results/BL-res256/init/AH_discrete1_it0000000000.npy',
        compute=True,
        two_pass=False,
        props=[],
        strategy=2,
        verbosity=2,
        timings=True,
    )

    # Example of a run for computing the properties in parallel.
    track_mots(
        hname='AH',
        sim_dir='data/simulation/brill-lindquist-res256',
        out_base_dir='results',
        folder='BL-res256/run1/AH',
        initial_guess=None,  # not needed for a property run
        compute=False,
        two_pass=False,
        props=['all'],
        strategy=2,
        verbosity=2,
        timings=True,
        retry_after=180,  # wait 3 minutes for new results and try again
    )
```

These examples assume the initial guess for the MOTS has been found e.g. using
.initialguess.InitHelper and that the apparent horizon already exists in the
first slice.

In case the initial guess for the AH is successful in, say, slice 2048, then
we need to run two sets of the above trackers, one going forward in slices and
one going backwards. This would be accomplished by adding the
following arguments:

@code
    initial_guess='results/BL-res256/init/AH_discrete1_it0000002048.npy',
    start_slice=2048,
    backwards=False,
@endcode

And for the backwards direction:

@code
    initial_guess='results/BL-res256/init/AH_discrete1_it0000002048.npy',
    start_slice=2048,
    backwards=True,
@endcode
"""

from glob import glob
import numbers
import os.path as op
import re
import time

import numpy as np
from scipy.interpolate import lagrange

from ..utils import timethis, find_file, find_files
from ..numutils import inf_norm1d, clip, IntegrationResults
from ..numutils import try_quad_tolerances
from ..metric.simulation import SioMetric
from .curve import BaseCurve, ParametricCurve, BipolarCurve, RefParamCurve
from .findmots import GeneralMotsConfig, find_mots


__all__ = [
    "track_mots",
    "MOTSTracker",
    "compute_props",
    "find_slices",
]


# Valid properties to compute.
ALL_PROPS = (None, 'none', 'all', 'length_maps', 'constraints', 'ingoing_exp',
             'avg_ingoing_exp', 'area', 'ricci', 'mean_curv', 'curv2',
             'stability', 'stability_convergence', 'neck', 'dist_top_bot',
             'z_dist_top_inner', 'z_dist_bot_inner', 'signature',
             'point_dists', 'area_parts', 'multipoles')

# Of the above, those not included in 'all'.
NEED_ACTIVATION_PROPS = ('stability_convergence',)


def track_mots(hname, sim_dir, out_base_dir, folder, initial_guess,
               MOTS_map=None, **kw):
    r"""Track one MOTS through slices of simulation data.

    See the package documentation for ..axisym.trackmots for examples of
    invoking the tracker.

    The tracker basically has two "modes" to run in:
        1. Find mode: Try to locate MOTSs in each slice
        2. Property mode: Compute area, stability, etc., of previously found
           MOTSs

    The idea is that the two modes can potentially be run in parallel: You
    start with a finding run. After the first few MOTSs have been found and
    stored in the output folder, a property runner can be invoked to compute
    physical properties parallel to the finding run. If desired, the property
    runner can be instructed to wait for results of the finder in case it
    catches up.

    Computed properties are stored as ``user_data`` of the curve object. After
    computing the properties, the file from where the curve was loaded is
    overwritten with the new curve containing the computed data. Note that
    this introduces an issue happening if two *property runners* compute
    (potentially different) properties of the same curve in parallel, since
    the one finishing last will effectively erase the results computed by the
    others. This means you should let only one property runner work on a curve
    at a time. No check is done to enforce this.

    For some tasks, the finder or property runner need to consider sibling
    MOTSs (e.g. to compute distances between MOTSs) or future/past MOTSs of
    the same tube (e.g. to compute the signature). Sibling MOTSs are located
    using the following assumptions:
        * Each MOTS has their own output folder
        * Each of these folders belong to a "run", i.e. a common parent folder

    This should be accomplished by setting `folder` appropriately, e.g.:

        folder="foo/run1/AH"

    An example file structure (with ``out_base_dir="some/path"``) might be:

        some/path/foo/run1/AH/AH_discrete_it0000000000.npy
        some/path/foo/run1/AH/AH_discrete_it0000000001.npy
        some/path/foo/run1/top/top_discrete_it0000000000.npy
        some/path/foo/run1/top/top_discrete_it0000000001.npy
        some/path/foo/run1/bot/bot_discrete_it0000000000.npy
        some/path/foo/run1/bot/bot_discrete_it0000000001.npy
        some/path/foo/run1/inner/inner_discrete_it0000000000.npy
        some/path/foo/run2/inner/inner_discrete_it0000000000.npy
        some/path/foo/run2/inner/inner_discrete_it0000000001.npy

    Here, the apparent horizon, the top, and bottom MOTSs were found in slices
    0 and 1. The inner common MOTS was only found in slice 0. Upon changing
    settings, we managed to find it in slice 0 and 1 and stored the results
    under `run2`.

    To make the tracker aware of the `top` and `bot` MOTSs from `run1` when
    computing properties of `run2/inner`, you supply a mapping from MOTS name
    (e.g. `top`, `bot`) to the run to take it from (e.g. `run1`). This is done
    using the `MOTS_map` parameter like so:

        MOTS_map=dict(top="run1", bot="run1")

    A missing entry in the `MOTS_map` means we should take the MOTS from the
    current run.

    @param hname
        Name of the horizon to track. This determines the file names as well
        as which properties are computed. For example, the distance between
        top and bottom MOTS is computed only if this MOTS has ``hname="bot"``
        and a sibling MOTS for the same slice named ``hname="top"`` has
        already been found and stored. Similarly, the *neck* location and
        properties as well as the distance to the ``"top"`` and ``"bot"``
        MOTSs is computed only for a run with ``hname="inner"``.
        Examples: ``"AH"``, ``"top"``, ``"bot"``, ``"inner"``.
    @param sim_dir
        Folder to find the simulation slices in. Slices from all subfolders
        are found and sorted alphanumerically. Slices with the same base name
        (i.e. ignoring differently named parent folders) as one already in the
        list are skipped.
        Example: ``"data/simulation/brill-lindquist-2-res256"``.
    @param out_base_dir
        Parent folder for all the results. Example: ``"./results"``.
    @param folder
        Sub folder of the `out_base_dir` containing results for this specific
        configuration and run. Example: ``"BL-2-res256/run1/AH"``.
        If configured as in the examples and the MOTS configuration is set to
        use a `simple_name` (see the documentation of
        .findmots.MotsFindingConfig.__init__()), then the curves will be
        stored in ``./results/BL-2-res256/run1/AH``.
    @param initial_guess
        Curve or filename or float to use as initial guess for tracking this
        MOTS. Differentiation between the different MOTSs to track through
        slices happens by using different initial guesses. You may use the
        .initialguess.InitHelper to help obtaining initial guesses for
        different MOTSs in a particular slice.
    @param MOTS_map
        Optional map (`dict`) specifying from which run the auxiliary MOTSs
        required for certain tasks should be taken. See above for examples.
    @param **kw
        Additional keyword arguments configure the settings of the
        MOTSTracker (see its documented public attributes).
    """
    tracker = MOTSTracker(
        hname=hname,
        sim_dir=sim_dir,
        out_base_dir=out_base_dir,
        folder=folder,
        initial_guess=initial_guess,
        MOTS_map=MOTS_map,
        **kw
    )
    return tracker.track()


class MOTSTracker():
    r"""Class for coordinating tracking a MOTS through slices.

    This is usually created and used by track_mots() but may also be created
    on its own to have a more object-oriented interface.

    The tracker object has a number of public (documented) attributes
    controlling how slices are traversed and with which settings the MOTS
    finder is invoked.
    """

    def __init__(self, hname, sim_dir, out_base_dir, folder, initial_guess, **kw):
        r"""Create an configure new tracker for a given MOTS.

        @param hname
            Name of the MOTS to track. This determines which properties are
            being computed (see the documentation of track_mots()).
        @param sim_dir
            Directory containing simulation data of all slices to track the
            MOTS in (see the documentation of track_mots()).
        @param out_base_dir
            Parent directory for results (see the documentation of
            track_mots()).
        @param folder
            Specific folder to store curves in (see the documentation of
            track_mots()).
        @param initial_guess
            Starting curve (or filename or float radius) determining the MOTS
            to track (see the documentation of track_mots()).
        @param **kw
            Additional options controlling the public (documented) instance
            attributes. Keywords that are not instance attributes are used as
            options for .findmots.find_mots() (see its documentation).
        """
        ## Name of MOTS to track (determines the properties to compute).
        self.hname = hname
        ## Folder to find all simulation data.
        self.sim_dir = sim_dir
        ## Parent folder for results.
        self.out_base_dir = out_base_dir
        ## Subfolder to store actual results in.
        self.folder = folder
        ## Initial curve determining which MOTS to track.
        self.initial_guess = initial_guess
        ## Dictionary determining the runs from which to take certain
        ## auxiliary curves (see the documentation of track_mots()).
        self.MOTS_map = None
        ## Whether to find curves at all (set this to `False` for a *property
        ## run*).
        self.compute = True
        ## If `True`, use a found MOTS as reference curve to find the same
        ## MOTS again (may reduce the resolution required to converge).
        self.two_pass = True
        ## Maximum number of slices to consider (default is all slices).
        self.max_slices = None
        ## First slice number to consider (inclusive when going forward,
        ## exclusive when going `backwards` (see below)).
        self.start_slice = 0
        ## Last slice to consider (inclusive).
        self.end_slice = None
        ## Number of *slices* to advance in each step (considers only the
        ## actual files, not the slices' iteration numbers).
        self.initial_stride = 1
        ## Trial counter to start with.\ The trial number is set to 1 by
        ## default and increased when a step fails and the stride is
        ## reduced.\ Using an initial value other than one may be useful for
        ## computing properties for slices without considering previous steps
        ## where the trial number had been increased.
        self.initial_try_count = None
        ## Whether to go through slices forward in coordinate time (going back
        ## may be useful if the AH is initially found at a later slice and we
        ## want to check where it appeared).
        self.backwards = False
        ## Whether to skip the first slice when going in backwards
        ## direction.\ This makes sense when starting both, a forward and a
        ## backwards tracker, so that the starting slice is not done twice.
        self.backwards_skip_first = True
        ## In case of a *property run*, wait that many seconds for results of
        ## a parallel *finding run* to appear before retrying.
        self.retry_after = None
        ## Callback function to modify the find_mots() settings during runtime
        ## on a slice-by-slice basis.\ The callable is called with signature
        ## ``cfg, pass_nr=pass_nr``.
        self.cfg_callback = None
        ## Which properties to compute.\ See compute_props() for a full list
        ## of properties that may be computed.\ Use ``"all"`` to compute all
        ## applicable.
        self.props = None
        ## Relative tolerance for the area integral.\ Setting this too low
        ## will result in integration warnings being produced and possibly
        ## underestimated residual errors.
        self.area_rtol = 1e-6
        ## Minimum number of stability eigenvalues to compute.\ Default is `30`.
        self.min_stability_values = 30
        ## Factors by which to multiply the curve resolution to
        ## compute convergence of the stability spectrum.\ Default is
        ## ``(0.2, 0.4, 0.6, 0.8, 0.9, 1.1)``.
        self.stability_convergence_factors = (0.2, 0.4, 0.6, 0.8, 0.9, 1.1)
        ## Strategy for finding MOTS in each step.\ Controls mainly the
        ## *preset* of .findmots.GeneralMotsConfig.preset() by indicating just
        ## the suffix number of the various ``discrete*`` presets.\ As
        ## indicated there, use ``2`` for best results if the plateau is
        ## unknown.\ Default is (still) ``1``.
        self.strategy = 1
        ## How verbose to be.\ Newton steps are printed for ``verbosity >= 2``.
        self.verbosity = 1
        ## Whether to indicate running times in printed status messages.
        self.timings = False
        ## Whether to return the full list of curves found.\ By default, the
        ## curves are found and stored to disk but not returned.
        self.full_output = False
        ## Initial resolution to start in the first slice with.\ If not set,
        ## uses the setting from the preset (if not overridden by specifying
        ## e.g.\ the `num` parameter).
        self.initial_num = None
        ## If `True`, use the resolution of the previous curve to start the
        ## search (default).\ If `False` and an `initial_num` is given, that
        ## resolution is used instead.\ Note that this is independent of any
        ## dynamic resolution determined by e.g.\ the plateau detection.
        self.follow_resolution = True
        ## If `True` (default), use the smoothing factor for the ``'curv2'``
        ## reparameterization strategy optimized for the previous MOTS.\ If
        ## `False` and `ref_smoothing` is specified, that value is used
        ## instead.\ Has no effect if `ref_smoothing` is not specified, as we
        ## then follow unconditionally.\ Note that this smoothing factor
        ## optimization is currently only available together with bipolar
        ## coordinates.
        self.follow_curv2_smoothing = True
        ## If `True` (default), use the optimal bipolar scaling determined for
        ## the previous MOTS.\ If `False` and a `bipolar_kw` is specified and
        ## contains a ``'scale'`` key, that value is used instead.\ Has no
        ## effect if ``'scale'`` is not specified, as we then follow
        ## unconditionally.
        self.follow_bipolar_scaling = True
        ## If `True`, move the origin of the bipolar coordinate system to the
        ## center between the lower end of the top and upper end of the bottom
        ## MOTS.
        self.auto_bipolar_move = True
        ## Minimum resolution of the reference curve.
        self.min_ref_num = 5
        ## Factor to multiply the previous curve's resolution with to get the
        ## reference curve resolution.
        self.ref_num_factor = 0.1
        ## Smoothing factor for the reference curve reparameterization.\ If
        ## given, forces the ``"curv2"`` reparameterization strategy (see
        ## .curve.parametriccurve.ParametricCurve.reparameterize()).
        self.ref_smoothing = None
        ## If `True`, read in the two individual MOTSs ``"top"`` and ``"bot"``
        ## (respecting any `MOTS_map` settings) to perform what is called here
        ## the *neck trick*.\ This essentially moves the "neck" of the
        ## reference curve in the middle between the two individual curves to,
        ## in some cases, significantly increase the chance to find the inner
        ## common MOTS.\ Whether the neck trick is actually performed depends
        ## on the `neck_trick_thresholds`.
        self.do_neck_trick = False
        ## Thresholds for performing the *neck trick* (if enabled).\ The first
        ## float is compared against the ratio of the smaller individual
        ## MOTS's coordinate width to the neck's coordinate width.\ The second
        ## against the ratio of coordinate distance of the two individual
        ## MOTSs to the neck's width.\ If any of these two ratios lies above
        ## the respective threshold, the *neck trick* is performed.\ Note that
        ## an error is raised if either of the two individual MOTSs for this
        ## slice is not found.
        self.neck_trick_thresholds = (10.0, 2.0)
        ## Kind of interpolation to perform on numerical data on a grid.\ Refer
        ## to motsfinder.metric.discrete.patch.DataPatch.set_interpolation()
        ## for more details.
        self.interpolation = None
        ## Finite difference order of accuracy for computing numerical
        ## derivatives.\ This controls the stencil size and is currently only
        ## used for the Hermite-type interpolations.
        self.fd_order = None
        self._last_neck_info = None
        self._files = None
        self._c_ref = None
        self._c_ref_prev = None
        self._metrics = None
        self._parent_dir = None
        self._run_name = None
        ## Additional options for each .findmots.find_mots() call.\ Should not
        ## be set directly as it is automatically populated with any extra
        ## keyword arguments.
        self.cfg = dict()
        for k, v in kw.items():
            if hasattr(self, k):
                super().__setattr__(k, v)
            else:
                self.cfg[k] = v
        if self.MOTS_map is None:
            self.MOTS_map = dict()

    @property
    def files(self):
        r"""List of all simulation files considered (in order)."""
        if self._files is None:
            self._files = self.get_sim_files()
        return self._files

    def get_metric(self, i):
        r"""Return the metric for the i'th slice in the list of files."""
        if self._metrics is None:
            self._metrics = [None] * len(self.files)
        if self._metrics[i] is None:
            opts = dict()
            if self.fd_order is not None:
                opts['fd_order'] = self.fd_order
            if self.interpolation is not None:
                opts['interpolation'] = self.interpolation
            g = SioMetric(self.files[i], **opts)
            g.release_file_handle()
            self._metrics[i] = g
        return self._metrics[i]

    def _p(self, msg, level=1):
        r"""Print the given message in case we should be verbose."""
        if self.verbosity >= level:
            print(msg)

    def _init_tracking(self):
        r"""Prepare for starting the actual tracking.

        This method prepares/validates configuration, prints init messages and
        loads the initial guess for the first slice. We also check the output
        folders here.
        """
        if self.props and not isinstance(self.props, (list, tuple)):
            self.props = [self.props]
        self._p("=" * 72)
        self._p("Tracking MOTS '%s' in (at most) %d slices..."
                % (self.hname, len(self.files)))
        self._c_ref = self.load_init_curve()
        path = self._base_cfg().get_out_dir()
        self._parent_dir = op.normpath(op.join(path, op.pardir, op.pardir))
        self._run_name = op.basename(op.dirname(path))

    def track(self):
        r"""Start the tracker.

        This method initiates the tracking procedure and walks through the
        considered slices. In each slice, the currently tracked MOTS is found
        after which we move on to the next slice. The MOTS found in the
        previous slice is then prepared as reference shape and initial guess
        for the current slice.

        The number of slices we advance in each step is `initial_stride` until
        a MOTS is not found. In this case, the stride is reduced and the
        smaller step is retried.

        Note that if the tracker is run multiple times with the same output
        configuration (same folders and ``hname``), then any curves found
        previously will be loaded instead of being re-computed. Effectively, a
        second run will then only take seconds when the first one finished at
        some point (even if this means it diverged at a particular slice).

        In case the tracker is running in *property* mode (i.e. with
        ``compute=False`` and ``props != None``), the same exact logic is
        applied, the difference being that we error out (or wait) if no
        previous curve is found.
        """
        self._init_tracking()
        curves = []
        try_no = 1 if self.initial_stride > 1 else None
        if self.initial_try_count is not None:
            try_no = self.initial_try_count
        stride = self.initial_stride
        i = 0
        while True:
            if i >= len(self.files):
                break
            c, fname = self._step(i, try_no=try_no, stride=stride)
            converged = c and c.user_data.get('converged', True)
            if i and stride > 1 and not converged:
                i -= stride # back to previous successful slice
                stride = int(stride/2)
                i += stride # advance to closer next slice
                try_no += 1
                self._c_ref = self._c_ref_prev
                continue
            if not c:
                break
            curves.append([c, fname])
            if not converged:
                # A curve was produced (and we computed its properties), but it
                # "formally" did not converge. This may be due to insufficient
                # maximum resolution, in which case it might still be a good
                # approximation of the solution. Another reason could be that the
                # step limit was reached. In this case the curve could be
                # completely off. Anyway, we still saved it to allow inspection.
                # The user can determine whether the result converged by checking
                # the values of `user_data['converged']` and `user_data['reason']`.
                self._p("Previous surface did not converge. Reason: %s"
                        % c.user_data.get('reason', 'unspecified'))
                self._p("Stopping analysis here.")
                break
            i += stride
        if self.full_output:
            return curves

    def _step(self, i, try_no, stride):
        r"""Find the MOTS in the i'th slice.

        @param i
            Slice file index.
        @param try_no
            How many times we reduced the stride length already. This changes
            the file name, since a first try at a larger stride length which
            failed may now succeed with smaller steps.
        @param stride
            Current stride length (number of files to advance). This is *not*
            the number of iterations to advance if the stored simulation files
            do not contain every slice.
        """
        g = self.get_metric(i)
        try:
            return self._do_step(g, i, try_no, stride)
        finally:
            g.unload_data()

    def _copy_curve(self, c):
        r"""Helper to copy a curve.

        If this curve is `None` (i.e. not found/given) or a number (e.g. for
        an initial guess of a round sphere), then no copying is required.
        """
        if c is None:
            return c
        if isinstance(c, numbers.Number):
            return c
        return c.copy()

    def _do_step(self, g, i, try_no, stride):
        r"""Implement _step()"""
        self._c_ref_prev = self._copy_curve(self._c_ref)
        self._p("=" * 72)
        self._p("Data file %d/%d: %s" % (i+1, len(self.files), self.files[i]))
        self._p("  %s" % time.strftime('%Y-%m-%d %H:%M:%S %z'))
        if self.initial_stride > 1:
            self._p("  evolution time: %s,  iteration: %s,  try: %s,  stride: %d"
                    % (g.time, g.iteration, try_no, stride))
        else:
            self._p("  evolution time: %s,  iteration: %s"
                    % (g.time, g.iteration))
        cfg = self._get_cfg(g, try_no)
        c, fname, cfg = self._load_existing_curve(cfg)
        if not c and fname:
            # curve was computed but did not converge (only `None` was saved)
            return None, None
        if not c and not self.compute:
            raise FileNotFoundError("Curve missing. Previous results expected.")
        self._prepare_metric_for_computation(g, c)
        if not c:
            cfg.update(c_ref=self._current_ref_curve(cfg))
            suffix = cfg.suffix
            c, fname = self._call_find_mots(
                cfg, pass_nr=1, timings=self.timings,
                callback=lambda curve: self._optimize_parameterization(curve, cfg)
            )
            if not c:
                return None, None
            if self.two_pass:
                cfg.update(suffix=suffix) # reset suffix if changed by cfg_callback
                self._c_ref = c
                c, fname = self._call_find_mots(
                    cfg, pass_nr=2, timings=self.timings,
                    c_ref=self._current_ref_curve(cfg, allow_neck_trick=False),
                )
                if not c:
                    return None, None
        # The following line has the side-effect of removing the 'cfg' key
        # from self._c_ref.user_data. Hence we make sure to store our precious
        # curve `c` as reference curve for the next slice *after* the
        # following line.
        c_past, c_future = self._get_future_past_curves(c, i, stride, try_no)
        self._compute_properties(c, fname, c_future=c_future, c_past=c_past)
        self._c_ref = c
        return c, fname

    def _current_ref_curve(self, cfg, allow_neck_trick=True):
        r"""Prepare and return the current reference curve.

        This also updates the given configuration with the bipolar coordinate
        setup to use (in case these coordinates are activated) and performs
        the *neck trick*.
        """
        g = cfg.metric
        c_ref = self._c_ref
        if cfg.bipolar_ref_curve:
            bipolar_kw = cfg.bipolar_kw or dict()
            if self.auto_bipolar_move:
                bipolar_kw['move'] = self._get_bipolar_origin(cfg)
            if self._do_bipolar_scaling_optimization(cfg):
                bipolar_kw['scale'] = self._get_bipolar_autoscale(c_ref, cfg)
            cfg.bipolar_kw = bipolar_kw
            if self._do_curv2_optimization(cfg):
                cfg.reparam = self._get_curv2_reparam_settings(c_ref, cfg)
        if allow_neck_trick:
            c_ref = self.neck_trick(g, c_ref)
        return c_ref

    def _reparam_settings(self, cfg):
        if isinstance(cfg.reparam, (list, tuple)):
            strategy, opts = cfg.reparam
        else:
            strategy = cfg.reparam
            opts = dict()
        return strategy, opts

    def _do_bipolar_scaling_optimization(self, cfg):
        bipolar_kw = cfg.bipolar_kw or dict()
        return (bipolar_kw.get('scale', None) is None
                or (not self._is_first_slice(cfg.metric)
                    and self.follow_bipolar_scaling))

    def _do_curv2_optimization(self, cfg):
        # We optimize the curv2 smoothing iff we also optimize the bipolar
        # scaling and smoothing is not fixed. This should be made independent!
        # need fresh cfg without auto-settings
        cfg = self._get_cfg(cfg.metric, 1) # we should solve this differently, though
        reparam, opts = self._reparam_settings(cfg)
        return (reparam == 'curv2'
                and (opts.get('smoothing', None) is None
                     or (not self._is_first_slice(cfg.metric)
                         and self.follow_curv2_smoothing))
                and cfg.bipolar_ref_curve)

    def _get_curv2_reparam_settings(self, curve, cfg):
        reparam_strategy, reparam_opts = self._reparam_settings(cfg)
        key = 'optimized_curv2_smoothing'
        if key in curve.user_data:
            reparam_opts['smoothing'] = curve.user_data[key]
        return reparam_strategy, reparam_opts

    def _optimize_parameterization(self, curve, cfg):
        r"""Optimize and update the stored parameterization for the given MOTS.

        The "optimal" settings are stored in the `user_data` of the current
        curve and may then be used for the search for the next MOTS. The curve
        data itself (shape and reference curve) is not modified in any way.
        """
        if curve is None or not cfg.bipolar_ref_curve:
            return
        if self._reparam_settings(cfg)[0] == 'curv2':
            scale, smoothing = self._determine_optimal_parameters(
                curve=curve, cfg=cfg,
                initial_smoothing=self._get_curv2_reparam_settings(
                    curve, cfg
                )[1].get('smoothing', 0.05)
            )
            curve.user_data['optimized_curv2_smoothing'] = smoothing
        else:
            scale = self._determine_optimal_parameters(curve=curve, cfg=cfg)
        curve.user_data['optimized_bipolar_scale'] = scale

    def _get_bipolar_autoscale(self, curve, cfg):
        r"""Return previously determined bipolar scale or a rough estimate."""
        key = 'optimized_bipolar_scale'
        if key in curve.user_data:
            return curve.user_data[key]
        scale, _ = self._get_parameter_guesses(curve, cfg)
        return scale

    def _get_bipolar_origin(self, cfg):
        r"""Return the origin for bipolar coordinates.

        If the origin is specified using configuration options (`bipolar_kw`),
        then this value is returned. Otherwise, we use the center between the
        top and bottom MOTSs (assuming they have already been found).
        """
        g = cfg.metric
        neck_info = self._get_neck_info(g)
        if neck_info.has_data:
            return neck_info.z_center
        c_ref = self._c_ref
        if callable(c_ref):
            return (c_ref(0)[1]+c_ref(np.pi)[1]) / 2.0
        if isinstance(c_ref, (list, tuple)):
            return c_ref[-1] # z-offset in case of circle
        return 0.0

    def _get_parameter_guesses(self, curve, cfg):
        if not cfg.bipolar_ref_curve:
            return None, None
        g = cfg.metric
        try:
            scale = curve.ref_curve.scale
            self._p("Using previous scale=%s" % scale, level=2)
        except AttributeError:
            # crude estimate that should be OK but not optimal
            neck_info = self._get_neck_info(g)
            if not neck_info.has_data:
                if callable(curve):
                    scale = 0.25 * (curve(0)[1]-curve(np.pi)[1])
                else:
                    if isinstance(curve, (list, tuple)):
                        scale = curve[0] / 2.0
                    else:
                        scale = curve / 2.0
            else:
                scale = abs(neck_info.z_dist)**(2/3.)
                scale = min(scale, 2/3. * neck_info.smaller_z_extent)
                scale = max(1e-3*neck_info.z_extent, scale)
            self._p("Using estimated scale=%s" % scale, level=2)
        move = self._get_bipolar_origin(cfg)
        return scale, move

    def _determine_optimal_parameters(self, curve, cfg,
                                      initial_smoothing=None):
        r"""Return (close to optimal) scaling for bipolar coordinates."""
        with timethis("Optimizing ref curve parameterization...",
                      silent=not self.timings):
            scale, move = self._get_parameter_guesses(curve, cfg)
            try:
                result = optimize_bipolar_scaling(
                    curve=curve, move=move,
                    initial_scale=scale,
                    initial_smoothing=initial_smoothing,
                    verbose=self.verbosity > 1,
                )
            except IndexError as e:
                # Optimization failed. Keep previous values.
                self._p("  ERROR: Optimization failed (%s)" % (e,))
                self._p("         Keeping previous values.")
                return scale if initial_smoothing is None else scale, initial_smoothing
        if initial_smoothing is None:
            scale, smoothing = result, None
        else:
            scale, smoothing = result
        self._p("  estimated optimal scale: %s" % scale)
        if initial_smoothing is None:
            return scale
        self._p("  estimated optimal curv2 reparam smoothing: %s"
                % smoothing)
        return scale, smoothing

    def _get_future_past_curves(self, c, i, stride, try_no):
        r"""Search for a curve for the next and previous slice.

        Only if both are found and have the same iteration distance
        (coordinate time distance) to this slice will they be returned.
        Otherwise, both curves are returned as `None`.
        """
        if not self._has_prop_to_do('signature'):
            return None, None
        c_past = c_future = None
        if i+stride < len(self.files):
            c_future = self._load_existing_curve(
                self._get_cfg(self.get_metric(i+stride), try_no),
                quiet=True, only_ok=True,
            )[0]
        if i-stride >= 0:
            c_past = self._load_existing_curve(
                self._get_cfg(self.get_metric(i-stride), try_no),
                quiet=True, only_ok=True,
            )[0]
        if not c_past or not c_future:
            return None, None
        # pylint: disable=no-member
        it_delta1 = c.metric.iteration - c_past.metric.iteration
        it_delta2 = c_future.metric.iteration - c.metric.iteration
        if it_delta1 != it_delta2:
            c_past = c_future = None
        return c_past, c_future

    def _prepare_metric_for_computation(self, g, c):
        r"""Load data and release file handles in common cases."""
        # Handle just the most common cases where we trivially know which data
        # will be needed. In these cases, we can load the data now and then
        # release the file handle (if any).
        do_load = False
        what = []
        if (c and self._has_prop_to_do('stability')
                and "stability" not in c.user_data):
            do_load = True
        if (c and self._has_prop_to_do('stability_convergence')
                and "stability_convergence" not in c.user_data):
            do_load = True
        if not c and not self._has_prop_to_do():
            do_load = True
            what = ["metric", "curv"]
        if not c and self._has_prop_to_do('stability'):
            do_load = True
            what = []
        if do_load:
            with timethis("Loading slice data...", " elapsed: {}",
                          silent=not self.timings, eol=False):
                g.load_data(*what)
            g.release_file_handle()

    def _has_prop_to_do(self, prop='any'):
        r"""Return whether any or a particular property should be computed."""
        props = self.props
        if not isinstance(props, (list, tuple)):
            props = [props]
        if prop == 'any':
            if len(props) == 1:
                p = props[0]
                return p is not None and p != 'none'
            return bool(props)
        return (prop in props
                or ('all' in props and prop not in NEED_ACTIVATION_PROPS))

    def _compute_properties(self, c, fname, c_future=None, c_past=None):
        r"""Call compute_props() for the given curve.

        In case properties have been computed and hence the curve data
        changed, the curve is re-saved to disk with the new data.
        """
        resave = False
        with timethis("Computing properties...",
                      silent=not (self.timings and self._has_prop_to_do())):
            stability_factors = self.stability_convergence_factors
            if compute_props(hname=self.hname, c=c, props=self.props,
                             area_rtol=self.area_rtol,
                             min_stability_values=self.min_stability_values,
                             stability_convergence_factors=stability_factors,
                             MOTS_map=self.MOTS_map, verbosity=self.verbosity,
                             fname=fname, c_future=c_future, c_past=c_past):
                resave = True
        if resave and fname:
            c.save(fname, overwrite=True)

    def _aux_MOTS_dir(self, hname):
        r"""Get the directory (respecting the MOTS map) of auxiliary curves."""
        run = self.MOTS_map.get(hname, self._run_name)
        return op.join(self._parent_dir, run, hname)

    def _get_neck_info(self, g):
        if self._last_neck_info and self._last_neck_info.iteration == g.iteration:
            return self._last_neck_info
        self._last_neck_info = self._compute_neck_info(g)
        return self._last_neck_info

    def _compute_neck_info(self, g):
        c = self._c_ref
        top_dir = self._aux_MOTS_dir('top')
        bot_dir = self._aux_MOTS_dir('bot')
        threshold1, threshold2 = self.neck_trick_thresholds
        neck_info = _NeckInfo(threshold1=threshold1, threshold2=threshold2,
                              iteration=g.iteration)
        if not hasattr(c, 'find_neck'):
            self._p("Not an ExpansionCurve. Neck moving not available.")
            return neck_info
        try:
            x_neck, z_neck = c(c.find_neck('coord')[0])
        except ValueError:
            self._p("Neck not found.")
            return neck_info
        c_top = find_file(
            pattern="%s/top_*_it%010d*.npy" % (top_dir, g.iteration),
            skip_regex=r"_CE", load=True, verbose=self.verbosity > 1
        )
        c_bot = find_file(
            pattern="%s/bot_*_it%010d*.npy" % (bot_dir, g.iteration),
            skip_regex=r"_CE", load=True, verbose=self.verbosity > 1
        )
        with c_top.fix_evaluator():
            x_top = max(
                c_top(la)[0] for la in c_top.h.collocation_points(lobatto=False)
            )
        with c_bot.fix_evaluator():
            x_bot = max(
                c_bot(la)[0] for la in c_bot.h.collocation_points(lobatto=False)
            )
        pinching1 = min(x_top, x_bot) / x_neck
        self._p("Smaller MOTS's width / neck width: %s (threshold=%s)"
                % (pinching1, threshold1))
        z_top = c_top(np.pi)[1]
        z_bot = c_bot(0.0)[1]
        z_top_outer = c_top(0.0)[1]
        z_bot_outer = c_bot(np.pi)[1]
        pinching2 = abs(z_top - z_bot) / (2*x_neck)
        self._p("Individual horizons' distance / neck width: %s (threshold=%s)"
                % (pinching2, threshold2))
        neck_info.update(
            x_top=x_top, z_top=z_top, x_bot=x_bot, z_bot=z_bot, x_neck=x_neck,
            z_neck=z_neck, z_top_outer=z_top_outer, z_bot_outer=z_bot_outer,
            pinching1=pinching1, pinching2=pinching2,
        )
        return neck_info

    def neck_trick(self, g, c):
        r"""Check and possibly perform the *neck trick*.

        @param g
            Current slice's metric.
        @param c
            Reference curve (or initial guess) to apply the neck trick to.
            Should be a .curve.expcurve.ExpansionCurve.
        """
        if not self.do_neck_trick:
            return c
        neck_info = self._get_neck_info(g)
        if not neck_info.has_data:
            return c
        if not neck_info.do_move_neck:
            self._p("Neck pinching below threshold. Not moving reference curve.")
            return c
        self._p("Neck pinching above threshold. Moving...")
        z_neck = neck_info.z_neck
        z_center = neck_info.z_center
        if hasattr(c, 'ref_curve') and hasattr(c.ref_curve, 'add_z_offset'):
            c = c.copy()
            c.ref_curve = c.ref_curve.copy()
            c.ref_curve.add_z_offset(-(z_neck - z_center))
        else:
            # Not a RefParamCurve. Convert to parametric curve.
            c = ParametricCurve.from_curve(c, num=c.num)
            c.add_z_offset(-(z_neck - z_center))
            self._p("Curve converted to ParametricCurve with resolution %s." % c.num)
        self._p("Neck moved by dz=%s to z=%s" % (z_center - z_neck, z_center))
        return c

    @classmethod
    def max_constraint_along_curve(cls, curve, points=None, x_padding=1,
                                   fd_order=None, full_output=False):
        r"""Compute the maximum violation of the Hamiltonian constraint.

        This computes the constraints on a set of grid points close to the
        given curve and returns the maximum value of the Hamiltonian
        constraint.

        @param curve
            The curve along which to sample the constraint.
        @param points
            Number of points to sample at along the curve. If not specified,
            uses either the curve's *accurate* residual measuring points or
            (if these are not available) twice the resolution of the curve.
            May also be a list or tuple of parameter values in the range
            ``[0,pi]``.
        @param x_padding
            How many grid points to stay away from the z-axis. Default is 1.
        @param fd_order
            Finite difference convergence order for computing derivatives at
            grid points. Default is to use the current stencil.
        @param full_output
            If `True`, return the evaluated constraints along with the maximum
            violation.
        """
        if points is None and 'accurate_abs_delta' in curve.user_data:
            params = curve.user_data['accurate_abs_delta']['points']
        else:
            if points is None:
                points = 2 * curve.num
            if isinstance(points, (list, tuple, np.ndarray)):
                params = np.asarray(points)
            else:
                params = np.linspace(0, np.pi, points+1, endpoint=False)[1:]
        g = curve.metric
        gxx = g.component_matrix(0, 0)
        i0 = gxx.closest_element((0, 0, 0))[0]
        constraints = dict()
        full_data = []
        with g.temp_interpolation(interpolation='none', fd_order=fd_order):
            for la in params:
                i, j, k = gxx.closest_element(curve(la, xyz=True))
                if x_padding:
                    i = max(i0+x_padding, i)
                key = (i, j, k)
                if key in constraints:
                    ham, mom = constraints[key]
                else:
                    pt = gxx.coords(i, j, k)
                    ham, mom = g.constraints(pt, norm=True)
                    constraints[key] = [ham, mom]
                full_data.append([la, ham, mom])
        data = np.array(list(constraints.values()))
        constraints = data[:,0]
        max_ham = np.absolute(constraints).max()
        if full_output:
            full_data = np.asarray(full_data)
            return max_ham, full_data
        return max_ham

    def _get_cfg(self, g, try_no):
        r"""Construct a configuration for the given metric's slice."""
        suffix = "it%010d" % g.iteration
        if try_no is not None:
            suffix += "_try%04d" % try_no
        c_ref = self._c_ref
        cfg = self._base_cfg(metric=g, suffix=suffix, c_ref=c_ref)
        if self.strategy >= 2:
            if c_ref and not isinstance(c_ref, numbers.Number):
                cfg.update(
                    num=c_ref.num,
                    ref_num=max(
                        self.min_ref_num,
                        int(round(self.ref_num_factor*c_ref.num))
                    )
                )
            else:
                cfg.update(
                    num=int(round(
                        self.min_ref_num / max(self.ref_num_factor, 0.1)
                    )),
                    ref_num=self.min_ref_num,
                )
        if self._should_override_resolution(g):
            cfg.update(num=self.initial_num)
        if self.ref_smoothing is not None:
            self._change_reparam_settings(
                cfg, "curv2", smoothing=self.ref_smoothing
            )
        return cfg

    def _should_override_resolution(self, g):
        r"""Return whether we should override the curve resolution for this slice."""
        if self.initial_num is None:
            return False
        if self.follow_resolution and not self._is_first_slice(g):
            return False
        return True

    def _is_first_slice(self, g):
        r"""Return whether the given metric belongs to the first slice considered."""
        return g.iteration == self.get_metric(0).iteration

    def _change_reparam_settings(self, cfg, new_strategy=None, **new_settings):
        r"""Merge new settings into current `reparam` parameters.

        No check is done to ensure consistent settings, e.g. we don't remove
        `smoothing` from the arguments even if `new_strategy` is incompatible.
        Currently, the caller of this class has to provide consistent
        settings.
        """
        opts = dict()
        reparam = cfg.reparam
        if isinstance(cfg.reparam, (list, tuple)):
            reparam, old_opts = cfg.reparam
            opts.update(old_opts)
        opts.update(new_settings)
        if new_strategy is not None:
            reparam = new_strategy
        cfg.update(reparam=(reparam, opts))

    def _base_cfg(self, **kw):
        r"""Create a base configuration independently of the slice."""
        cfg = GeneralMotsConfig.preset(
            'discrete%s' % self.strategy, hname=self.hname,
            save=True, base_folder=self.out_base_dir, folder=self.folder,
            dont_compute=not self.compute, verbose=self.verbosity > 1,
        )
        cfg.update(**self.cfg)
        cfg.update(**kw)
        # ensure we can edit these settings without changing the original
        cfg.update(bipolar_kw=(cfg.bipolar_kw or dict()).copy())
        return cfg

    def _load_existing_curve(self, cfg, quiet=False, only_ok=False):
        r"""Taking the current configuration, try to load an existing result.

        This will not initiate a MOTS finding process. If the curve is not
        found (e.g. because it was not yet computed/found), `None` is
        returned, unless we are in a *property run* and should wait for
        results to appear.
        """
        cfg_loading = cfg.copy()
        while True:
            c, fname = self._try_to_load_curve(cfg_loading)
            if (not quiet and not c and not fname and not self.compute
                    and self.retry_after):
                cfg.metric.release_file_handle()
                print("Data not yet ready. Waiting %s seconds."
                      % self.retry_after)
                time.sleep(self.retry_after)
                continue
            break
        if c and only_ok:
            ok_reasons = [
                'converged', 'plateau', 'insufficient resolution or roundoff',
            ]
            # pylint: disable=no-member
            reason = c.user_data.get('reason', 'converged') if c else 'missing'
            if reason not in ok_reasons:
                c = None
        if c:
            if not quiet:
                # pylint: disable=no-member
                self._p("Curve loaded: %s" % fname)
                max_err = c.user_data.get('accurate_abs_delta',
                                          dict(max='N/A'))['max']
                self._p("  resolution: %s, residual error: %s"
                        % (c.num, max_err))
            # Our cfg.metric object has the open file handle to the data, so
            # that's the one we should be using. Note that the loaded curve
            # currently points to the exact same metric (unless files were
            # moved around manually or runs for different spacetimes were
            # called the same, in which case we'd have a lot of other
            # consistency problems).
            c.metric = cfg.metric
            c.extr_curvature = cfg.metric.get_curv()
            cfg = cfg_loading
        return c, fname, cfg

    def _try_to_load_curve(self, cfg):
        r"""Try to load a previous result.

        This does not wait nor prepare the curve in any way and returns
        immediately.
        """
        pass_nr = 2 if self.two_pass else 1
        try:
            return self._call_find_mots(cfg, pass_nr, dont_compute=True,
                                        save=True)
        except FileNotFoundError:
            return None, None

    def _call_find_mots(self, cfg, pass_nr, timings=False, **kw):
        r"""Perform final preparations and then call the MOTS finder.

        These preparations include calling the configuration callback to tweak
        individual settings using user-supplied functions.
        """
        suffix = cfg.suffix
        if pass_nr != 1:
            if suffix:
                suffix += "_"
            suffix += "pass%s" % pass_nr
            cfg.update(suffix=suffix)
        if kw:
            cfg.update(**kw)
        if self.cfg_callback:
            self.cfg_callback(cfg, pass_nr=pass_nr) # pylint: disable=not-callable
        if hasattr(cfg.c_ref, 'user_data') and 'cfg' in cfg.c_ref.user_data:
            # Avoid curves to store all their ancestors' data.
            del cfg.c_ref.user_data['cfg']
        save_cfg = cfg.copy().update(callback=None, veto_callback=None)
        user_data = dict(
            cfg=dict(cfg=save_cfg, two_pass=self.two_pass, pass_nr=pass_nr)
        )
        with timethis("Calling find_mots()...", silent=not timings):
            return find_mots(cfg, user_data=user_data, full_output=True)

    def load_init_curve(self):
        r"""Interpret the `initial_guess` and convert it to a usable form.

        Valid initial guesses are actual curves, as well as floats (taken to
        be coordinate radii of round spheres) or string containing the file
        name of the curve to load.
        """
        curve = self.initial_guess
        try:
            # Curve given as float.
            return float(curve)
        except (TypeError, ValueError):
            pass
        if isinstance(curve, str):
            # Curve given by file name.
            return BaseCurve.load(curve)
        # Curve probably given directly (as curve object).
        return curve

    def get_sim_files(self, disp=True):
        r"""Gather all simulation data files we should consider."""
        return find_slices(
            sim_dir=self.sim_dir,
            backwards=self.backwards,
            backwards_skip_first=self.backwards_skip_first,
            start_slice=self.start_slice,
            end_slice=self.end_slice,
            max_slices=self.max_slices,
            disp=disp,
        )


def find_slices(sim_dir, backwards=False, start_slice=None, end_slice=None,
                max_slices=None, backwards_skip_first=True,
                skip_checkpoints=True, disp=True):
    r"""Gather all simulation data files.

    @param sim_dir
        Folder to start searching recursively for simulation data files.
    @param backwards
        Whether to reverse the order of the slices found. Default is `False`.
    @param start_slice
        First slice to consider. When ``backwards==True``, this should be the
        *latest* slice in the series. Note that if a start slice is specified
        in the ``backwards==True`` case, the slice itself is skipped to allow
        simultaneously starting forward and backwards runs.
        Default is to find all slices (including the last one in the backwards
        case).
    @param end_slice
        Last slice to consider. Take all (except for `start_slice`) if not
        given (default).
    @param max_slices
        Truncate the number of slices returned after this many if given.
    @param backwards_skip_first
        If `True` (default), skip the starting slice when going backwards as
        described above. If `False`, include the starting slice in all cases.
    @param skip_checkpoints
        Whether to ignore checkpoints in the simulation folders. Default is
        `True`.
    @param disp
        Raise a `ValueError` if no slices are found.
    """
    files = glob("%s/**/*.it*.s5" % sim_dir, recursive=True)
    # remove duplicates (caused by symlinks) by constructing a dict
    files = dict([(op.basename(fn), fn) for fn in files
                  if re.search(r"\.it\d{10}\.s5", fn)
                  and not re.search(r"output-\d{4}-active", fn)
                  and (not skip_checkpoints or not re.search(r"checkpoint\.", fn))])
    files = list(files.items())
    files.sort(key=lambda f: f[0]) # sort based on base name
    files = [fn for _, fn in files] # return the relative names
    if backwards:
        files.reverse()
    active = start_slice is None
    result = []
    for fn in files:
        if (start_slice is not None
                and re.search(r"\.it%010d\.s5" % start_slice, fn)):
            active = True
            if backwards and backwards_skip_first:
                # Starting a forward and backward search simultaneously
                # requires one of them to ignore the start slice.
                continue
        if active:
            result.append(fn)
        if (end_slice is not None
                and re.search(r"\.it%010d\.s5" % end_slice, fn)):
            break
    if disp and not result:
        raise ValueError("No simulation data found in: %s" % sim_dir)
    if max_slices and len(result) > max_slices:
        result = result[:max_slices]
    return result


def compute_props(hname, c, props, area_rtol=1e-6, min_stability_values=30,
                  stability_convergence_factors=(0.2, 0.4, 0.6, 0.8, 0.9, 1.1),
                  verbosity=True, MOTS_map=None, fname=None, c_past=None,
                  c_future=None, remove_invalid=True):
    r"""Compute properties and store them in the curve object.

    All computed properties will be stored under the property's name in the
    curve's ``user_data`` dictionary. The arguments used for this computation
    are additionally stored under the key ``prop_args``, where ``prop`` is the
    property name. Some properties produce extra output. This is then stored
    under ``prop_extras``.

    The properties that can be computed here are:
        * **length_maps**:
            Mappings from current parameterization to proper-length based
            parameterizations (see
            ..curve.basecurve.BaseCurve.proper_length_map()). The value will
            be a dictionary with elements `length_map`, `inv_map`,
            `proper_length`, where the latter is the length of the curve in
            curved space. Note that the two functions will actually be series
            objects (motsfinder.exprs.series.SeriesExpression), so that
            ``evaluator()`` needs to be called to get actual callables.
        * **constraints**:
            Compute the Hamiltonian (scalar) and momentum (vector) constraints
            on grid points close to the curve. The stored value will have the
            structure ``dict(near_curve=dict(std=[x, ham, mom],
            proper=[x, ham, mom]))``. Here, `x` is the set of parameter values
            of the curve close to which the values are computed, `ham` are the
            Hamiltonian constraint values and `mom` the momentum ones. The
            difference between `std` and `proper` is that for `std`, the
            parameters are equidistant in curve parameter `lambda`, while for
            `proper` they are equidistant in the curve's proper length
            parameterization, though the stored `x` values are in the curve's
            current parameterization. This way, you can plot in the curve's
            proper length parameterization by taking the `x` values of `std`
            and values of `proper`.
        * **ingoing_exp**:
            Expansion of the ingoing null normals. This is computed on a grid
            of points (in curve parameter space) the density of which is
            determined by the curve's resolution.
        * **avg_ingoing_exp**:
            Average ingoing expansion, averaged over the MOTSs surface area.
        * **area**:
            Surface area of the MOTS. This produces extra data (as explained
            above) containing information about the convergence of the
            integral.
        * **ricci**:
            Ricci scalar computed on a grid of points with density dependent
            on the curve resolution. Additionally, the maximum absolute value
            of the Ricci scalar along the curve is found and stored. Inspect
            the actual data (dictionary) for details.
        * **mean_curv**:
            Similar to ``ricci``, but compute the trace of the extrinsic
            curvature of the MOTS in the slice instead.
        * **curv2**:
            Similar to ``mean_curv``, but compute the "square"
            \f$k_{AB}k^{AB}\f$ of the MOTS's extrinsic curvature.
        * **stability**:
            Compute the spectrum of the stability operator. The number of
            eigenvalues returned depends on the curve's resolution, but at
            least 30 eigenvalues (by default) are computed. See also the
            `min_stability_values` parameter and the following property.
        * **stability_convergence**:
            In addition to computing the stability spectrum above, recompute
            it at (by default) ``0.2, 0.4, 0.6, 0.8, 0.9, 1.1`` times the
            resolution used for the ``"stability"`` property. This allows
            analyzing convergence of the individual eigenvalues.
        * **neck**:
            Find the neck based on various criteria and compute various
            properties at this point, like proper circumference or proper
            distance to the z-axis. Only computed for a MOTS with
            ``hname="inner"``.
        * **dist_top_bot**:
            Proper distance (along the z-axis) of the two individual MOTSs.
            Only computed for a MOTS with ``hname="bot"``.
        * **z_dist_top_inner**:
            Proper distance of the highest point on the z-axis of the inner
            common MOTS and the top individual MOTS. Only compute for a MOTS
            with ``hname="inner"``.
        * **z_dist_bot_inner**:
            Proper distance of the lowest point on the z-axis of the inner
            common MOTS and the bottom individual MOTS. Only compute for a
            MOTS with ``hname="inner"``.
        * **signature**:
            Signature of the MOTT traced out by the series of MOTSs. The
            number of points at which the signature is computed depends on the
            curve's resolution
        * **point_dists**:
            Coordinate distances of the collocation points along the curve.
            May be useful to compare against the spatial coordinate resolution
            of the numerical grid.
        * **area_parts**:
            In case ``hname="bot"`` or ``"top"`` and the top and bottom MOTSs
            intersect, compute separately the area of the non-intersecting
            part and the part lying inside the other MOTS.
            For ``hname="inner"``, compute the parts before and after the
            *neck* separately. In case it self-intersects, find all the
            intersection points of the MOTS with itself and the top and bottom
            MOTSs and compute the area of each section, including splitting at
            the neck.
            If a result is found (i.e. the property's value is not `None`),
            the stored value will be a numutils.IntegrationResults object. Its
            `info` dictionary contains all splitting points. Sorting these
            gives you all the intervals of which the areas were computed.
        * **multipoles**:
            Compute the first 10 multipole moments of the MOTS. The first and
            second moments are computed numerically even though these have
            analytically known values of `sqrt(pi)` and `0`, respectively.
            This allows comparison of the integration results with known
            values.

    Some properties are only computed for certain horizons. The horizon
    specific properties are:
        * for ``"top"``: `area_parts`
        * for ``"bot"``: `dist_top_bot`, `area_parts`
        * for ``"inner"``: `neck`, `z_dist_top_inner`, `z_dist_bot_inner`,
          `area_parts`

    The ``signature`` is only computed if at least one of `c_past` or
    `c_future` is supplied.

    @param hname
        Name of horizon. Determines which kinds of properties are being
        computed (see above).
    @param c
        Curve representing the MOTS in axisymmetry.
    @param props
        Sequence of properties to compute (see above for the possible values).
        Use the string ``"all"`` to compute all possible properties.
    @param area_rtol
        Relative tolerance for the area integral. Setting this too low will
        result in integration warnings being produced and possibly
        underestimated residual errors.
    @param min_stability_values
        Minimum number of MOTS-stability eigenvalues to compute. The default
        is `30`.
    @param stability_convergence_factors
        Factors by which to multiply the resolution used for computing the
        stability spectrum. Each of the resulting lower or higher resolutions
        is used to compute the same spectrum, so that convergence of the
        individual eigenvalues can be examined. Defaults to ``(0.2, 0.4, 0.6,
        0.8, 0.9, 1.1)``.
    @param verbosity
        Whether to print progress information.
    @param MOTS_map
        Dictionary indicating from which runs auxiliary MOTSs should be loaded
        (used e.g. for the various distances).
    @param fname
        File name under which the MOTS is stored. This is currently just used
        to infer the run name for finding auxiliary MOTSs.
    @param c_past,c_future
        Optional MOTSs of the previous and next slices, respectively. These
        are used to compute the signature of the world tube traced out by the
        evolution of the MOTS.
    @param remove_invalid
        Whether to check for and remove invalid data prior to recomputing it.
        This only affects data that is specified to being computed. This may
        be useful for updating data for which updated methods have been
        developed and a check for validity can be done. Default is `True`.
    """
    if MOTS_map is None:
        MOTS_map = dict()
    out_dir = run_name = None
    if fname:
        out_dir = op.dirname(fname)
        run_name = op.basename(op.dirname(out_dir))
    if not isinstance(props, (list, tuple)):
        props = [props]
    for p in props:
        if p not in ALL_PROPS:
            raise ValueError(
                "Invalid property to compute: %s. Valid properties are: %s"
                % (p, ",".join("%s" % pr for pr in ALL_PROPS))
            )
    data = c.user_data
    did_something = [False]
    def do_prop(p, func, args=None, **kw):
        r"""(Re)compute a given property using a given function.

        @param p
            Property name to compute. The curve's `user_data` will get
            keys ``{p}``, ``{p}_args``, and potentially ``{p}_extras``
            containing the data, defining arguments and any extra data
            returned.
        @param func
            Function to call to produce the result. Will be called as
            ``func(**args, **kw)``.
        @param args
            Argument dictionary uniquely defining the result.
        @param **kw
            Extra arguments not influencing the result or reproducible
            from context (e.g. the curve to operate on).
        """
        if p not in ALL_PROPS:
            raise RuntimeError("Unkown property: %s" % p)
        is_extra = p in NEED_ACTIVATION_PROPS
        do = p in props or ('all' in props and not is_extra)
        if not do:
            return
        if args is None:
            args = dict()
        p_args = '%s_args' % p
        p_extras = '%s_extras' % p
        if remove_invalid and p in data and p_args in data and data[p_args] == args:
            if not _data_is_valid(p, data[p], data[p_args], hname):
                print("Removing invalid '%s' data from curve to trigger "
                      "recomputation." % p)
                del data[p]
        if p not in data or p_args not in data or data[p_args] != args:
            msg = "Computing property %-21s" % ("'%s'..." % p)
            with timethis(msg, " elapsed: {}", silent=verbosity == 0, eol=False):
                try:
                    result = func(**args, **kw)
                except AuxResultMissing:
                    if verbosity > 0:
                        print(" [cancelled due to missing data]", end="")
                    return
            if isinstance(result, _PropResult):
                if result.extras:
                    data[p_extras] = result.extras
                else:
                    data.pop(p_extras, None)
                result = result.result
            data[p] = result
            data[p_args] = args
            did_something[0] = True
    if hname == 'bot' and out_dir:
        do_prop('dist_top_bot', _dist_top_bot,
                dict(rtol=1e-5,
                     allow_intersection=True,
                     other_run=MOTS_map.get('top', run_name),
                     other_hname='top'),
                c=c, out_dir=out_dir)
    do_prop('length_maps', _length_maps, dict(num=None), c=c)
    do_prop('constraints', _constraints, dict(num=None, fd_order=None), c=c)
    do_prop('area', _area, dict(epsrel=area_rtol, limit=100, v=1), c=c)
    do_prop('ingoing_exp', _ingoing_exp, dict(Ns=None, eps=1e-6), c=c)
    do_prop('avg_ingoing_exp', _avg_ingoing_exp,
            dict(epsabs='auto', limit='auto'), c=c)
    if c.num >= min_stability_values:
        # This code ensures that the stability spectrum is not recomputed
        # unnecessarily (i.e. when the minimum `min_num` has no effect because
        # the curve has a higher resolution anyway). Since the property is
        # recomputed on any argument changes, we set it to its previous
        # default value in these cases so that it matches the previously
        # stored value and does not trigger recomputation.
        min_stability_values = 30 # previous default, has no effect here
    do_prop('stability', _stability, dict(min_num=min_stability_values,
                                          method='direct'), c=c)
    do_prop('stability_convergence', _stability_convergence,
            dict(min_num=min_stability_values,
                 convergence_factors=stability_convergence_factors,
                 method='direct'),
            c=c, verbose=verbosity > 0)
    do_prop('ricci', _ricci, dict(Ns=None, eps=1e-6, xatol=1e-6), c=c)
    do_prop('mean_curv', _mean_curv, dict(Ns=None, eps=1e-6, xatol=1e-6), c=c)
    do_prop('curv2', _curv_squared, dict(Ns=None, eps=1e-6, xatol=1e-6), c=c)
    do_prop('point_dists', _point_dists, c=c)
    if hname == 'inner':
        do_prop('neck', _neck, dict(xtol=1e-6, epsrel=1e-6), c=c)
        for other_hname, where in zip(['top', 'bot'], ['top', 'bottom']):
            do_prop('z_dist_%s_inner' % other_hname, _z_dist_to_inner,
                    dict(rtol=1e-5, where=where,
                         other_run=MOTS_map.get(other_hname, run_name),
                         other_hname=other_hname),
                    c=c, out_dir=out_dir)
        do_prop('area_parts', _area_parts_inner,
                dict(epsrel=area_rtol, limit=100, v=1,
                     top_run=MOTS_map.get('top', run_name),
                     bot_run=MOTS_map.get('bot', run_name)),
                c=c, out_dir=out_dir)
    if hname in ('top', 'bot'):
        other_hname = 'bot' if hname == 'top' else 'top'
        do_prop('area_parts', _area_parts_individual,
                dict(epsrel=area_rtol, limit=100, v=2,
                     other_run=MOTS_map.get(other_hname, run_name),
                     other_hname=other_hname),
                c=c, out_dir=out_dir)
    do_prop('multipoles', _multipoles, dict(max_n=10), c=c)
    if c_past or c_future:
        do_prop('signature', _signature,
                dict(N=None, eps=1e-6, has_past=bool(c_past),
                     has_future=bool(c_future)),
                c=c, c_past=c_past, c_future=c_future)
    return did_something[0]


def _area(c, v=None, **kw):
    # pylint: disable=unused-argument
    r"""Compute the area for the curve.

    The parameter `v` is ignored here and can be used to force recomputation
    of the area for curves already containing an area result.
    """
    # Note: `v` can be passed to enforce recomputation of results previously
    # stored in curve files. When the arguments passed to `_area()` are
    # different than those used to compute the original result, recomputation
    # is triggered in `compute_props()`.
    A, abserr, info, message = c.area(full_output=True, **kw)
    return _PropResult(
        result=A,
        extras=dict(abserr=abserr, info=info, message=message),
    )


def _area_parts_individual(c, out_dir, other_run, other_hname, v=None, **kw):
    # pylint: disable=unused-argument
    r"""Compute the area of different parts of an individual MOTS.

    If the other individual MOTS does not intersect this MOTS, nothing is done
    and `None` is returned. If it does intersect, computes the area of the
    exterior and interior parts separately and returns an
    numutils.IntegrationResults object. The returned object's `info`
    dictionary contains an `intersection` key with the curve paramters of this
    (first) and the other (second) curve at which they intersect.
    """
    c_other = _get_aux_curve(
        "%s/../../%s/%s" % (out_dir, other_run, other_hname),
        hname=other_hname,
        it=c.metric.iteration,
        disp=False,
    )
    if not c_other:
        raise AuxResultMissing()
    z_dist = c.z_distance_using_metric(
        metric=None, other_curve=c_other, allow_intersection=True,
    )
    if z_dist >= 0:
        return None
    la1, la2 = c.locate_intersection(c_other, N1=100, N2=100)
    kw['full_output'] = True
    return IntegrationResults(
        c.area(domain=(0, la1), **kw),
        c.area(domain=(la1, np.pi), **kw),
        info=dict(intersection=(la1, la2)),
    )


def _area_parts_inner(c, out_dir, top_run, bot_run, v=None, **kw):
    # pylint: disable=unused-argument
    r"""Compute areas of different parts of the inner common MOTS.

    Note that a precondition for this function is computation of the 'neck'.
    If a neck was actually found, computes the area of the upper and lower
    parts separately. The result is an numutils.IntegrationResults object with
    the two values and a `info` dictionary containing the used `neck`
    location.

    If, in addition to a neck, the MOTS self-intersects, then it will also
    intersect the two individual MOTSs. The four intersection points (two
    parameters for the self-intersection, plus two for the intersection with
    the other MOTSs) together with the neck partition the curve into six
    sections, of which the area is computed separately. The resulting
    numutils.IntegrationResults object's `info` dictionary contains the
    location of all detected intersections. For the individual MOTSs, the
    second parameter in the values refers to the parameter of the individual
    MOTS at which this intersection happens.
    """
    c_top = _get_aux_curve(
        "%s/../../%s/%s" % (out_dir, top_run, 'top'),
        hname='top',
        it=c.metric.iteration,
        disp=False,
    )
    c_bot = _get_aux_curve(
        "%s/../../%s/%s" % (out_dir, bot_run, 'bot'),
        hname='bot',
        it=c.metric.iteration,
        disp=False,
    )
    if not c_top or not c_bot:
        raise AuxResultMissing()
    if c_top(0)[1] < c_bot(0)[1]:
        c_top, c_bot = c_bot, c_top
    z_dist = c_top.z_distance_using_metric(
        metric=None, other_curve=c_bot, allow_intersection=True,
    )
    self_intersecting = z_dist < 0
    neck = c.user_data['neck']['circumference']['param']
    if not neck:
        return None
    kw['full_output'] = True
    if not self_intersecting:
        A1 = c.area(domain=(0, neck), **kw)
        A2 = c.area(domain=(neck, np.pi), **kw)
        return IntegrationResults(
            A1, A2, info=dict(neck=neck, self_intersection=None)
        )
    # A self-intersecting inner common MOTS also crosses both individual
    # horizons.
    la_self1, la_self2 = c.locate_self_intersection(neck=neck)
    # To make the search more robust, we use our knowledge that the inner
    # common MOTS intersects the individual MOTSs close to the point the
    # intersect each other.
    la_top, la_bot = c_top.locate_intersection(c_bot)
    la_top1, la_top2 = c.locate_intersection(
        c_top,
        domain1=(neck, np.pi), strict1=True, N1=50,
        domain2=(clip(la_top-0.2, 0, np.pi), clip(la_top+0.2, 0, np.pi)),
        strict2=False, N2=10,
    )
    if None in (la_top1, la_top2):
        # search failed, retry with different settings
        la_top1, la_top2 = c.locate_intersection(
            c_top,
            domain1=(la_self2, np.pi), # start at self-intersection point
            strict1=True,
            N1=max(100, min(1000, int(c.num/2))), # use much finer initial grid
            domain2=(clip(la_top-0.2, 0, np.pi), clip(la_top+0.2, 0, np.pi)),
            strict2=False, N2=30,
        )
    la_bot1, la_bot2 = c.locate_intersection(
        c_bot,
        domain1=(0, neck), strict1=True, N1=50,
        domain2=(clip(la_bot-0.2, 0, np.pi), clip(la_bot+0.2, 0, np.pi)),
        strict2=False, N2=10,
    )
    if None in (la_bot1, la_bot2):
        # search failed, retry with different settings
        la_bot1, la_bot2 = c.locate_intersection(
            c_bot,
            domain1=(0, la_self1), # start at self-intersection point
            strict1=True,
            N1=max(100, min(1000, int(c.num/2))), # use much finer initial grid
            domain2=(clip(la_bot-0.2, 0, np.pi), clip(la_bot+0.2, 0, np.pi)),
            strict2=False, N2=30,
        )
    params = sorted([0.0, la_bot1, la_self1, neck, la_self2, la_top1, np.pi])
    return IntegrationResults(
        *[c.area(domain=(params[i], params[i+1]), **kw)
          for i in range(len(params)-1)],
        info=dict(neck=neck,
                  self_intersection=(la_self1, la_self2),
                  top_intersection=(la_top1, la_top2),
                  bot_intersection=(la_bot1, la_bot2)),
    )


def _data_is_valid(prop, value, args, hname):
    r"""Return whether the current data is valid or should be removed."""
    if prop == 'area_parts' and hname == 'inner':
        ap = value
        if (len(ap) == 6 and (ap.info['self_intersection'][0] <= 0
                              or ap.info['self_intersection'][1] <= 0)):
            # MOTS self-intersection found at negative parameter values.
            return False
    return True


def _multipoles(c, max_n):
    r"""Compute the first multipoles for the given curve.

    @return numutils.IntegrationResults object containing the results with
        estimates of their accuracy.

    @param c
        Curve representing the horizon.
    @param max_n
        Highest multipole to compute. It will compute the elements
        ``0, 1, 2, ..., max_n`` (i.e. ``max_n+1`` elements).

    @b Notes

    The strategy for computation is to first try to compute using the
    estimated curve's residual expansion as absolute tolerance. If this fails
    or produces integration warnings, the tolerance is successively increased
    and computation is retried.
    """
    if 'accurate_abs_delta' in c.user_data:
        residual = c.user_data['accurate_abs_delta']['max']
    else:
        pts = np.linspace(0, np.pi, 2*c.num+1, endpoint=False)[1:]
        residual = np.absolute(c.expansions(pts)).max()
    return try_quad_tolerances(
        func=lambda tol: c.multipoles(
            max_n=max_n, epsabs=tol,
            limit=max(100, int(round(c.num/2))),
            full_output=True, disp=True,
        ),
        tol_min=residual,
        tol_max=max(1e-3, 10*residual),
    )


def _stability(c, min_num, method):
    r"""Compute the stability spectrum for the curve."""
    if method != 'direct':
        raise ValueError("Unsupported method: %s" % (method,))
    result = c.stability_parameter(num=max(min_num, c.num), full_output=True)
    return _PropResult(
        result=result,
        extras=dict(method=method),
    )


def _stability_convergence(c, min_num, convergence_factors, method,
                           verbose=False):
    r"""Compute the stability spectrum at different resolutions."""
    if method != 'direct':
        raise ValueError("Unsupported method: %s" % (method,))
    prev_spectra = c.user_data.get('stability_convergence', [])
    if 'stability' in c.user_data:
        prev_spectra.append(c.user_data['stability'])
    main_num = max(min_num, c.num)
    def _compute(factor):
        num = int(round(factor * main_num))
        for principal, spectrum in prev_spectra:
            if spectrum.shape[0] == num:
                return principal, spectrum
        if verbose:
            print(" x%s(%s)" % (round(factor, 12), num), end='',
                  flush=True)
        return c.stability_parameter(num=num, full_output=True)
    result = [_compute(f) for f in convergence_factors]
    return _PropResult(
        result=result,
        extras=dict(method=method),
    )


def _ricci(c, Ns, eps, xatol):
    r"""Compute the Ricci scalar and its maximum along the curve."""
    return _sample_and_extremum(c, c.ricci_scalar, Ns=Ns, eps=eps, xatol=xatol)


def _mean_curv(c, Ns, eps, xatol):
    r"""Compute the mean curvature and its maximum along the curve."""
    def func(x):
        return c.extrinsic_surface_curvature(x, trace=True)
    return _sample_and_extremum(c, func, Ns=Ns, eps=eps, xatol=xatol)


def _curv_squared(c, Ns, eps, xatol):
    r"""Compute the curvature square and its maximum along the curve."""
    def func(x):
        return c.extrinsic_surface_curvature(x, square=True)
    return _sample_and_extremum(c, func, Ns=Ns, eps=eps, xatol=xatol)


def _sample_and_extremum(c, func, Ns, eps, xatol):
    r"""Sample a property along the curve and find its maximum."""
    if Ns is None:
        Ns = max(100, 2*c.num)
    pts = np.linspace(eps, np.pi-eps, Ns)
    step = pts[1] - pts[0]
    with c.fix_evaluator():
        fx = np.array([[la, func(la)] for la in pts])
        x0, _ = max(fx, key=lambda xy: abs(xy[1]))
        x_max, f_max = inf_norm1d(
            func, domain=(max(eps, x0-step), min(np.pi-eps, x0+step)), Ns=0, xatol=xatol
        )
    return dict(values=fx, x_max=x_max, f_max=f_max)


def _length_maps(c, num):
    r"""Compute the conversion mappings from parameterization to proper length."""
    length_map, inv_map, proper_length = c.proper_length_map(
        num=num, evaluators=False, full_output=True
    )
    return dict(
        length_map=length_map, inv_map=inv_map, proper_length=proper_length
    )


def _constraints(c, num, fd_order):
    r"""Compute the constraints close to the curve."""
    if num is None:
        num = max(100, 2*c.num)
    std_params = np.linspace(0, np.pi, num+1, endpoint=False)[1:]
    inv_map = c.user_data['length_maps']['inv_map'].evaluator()
    proper_params = [inv_map(la) for la in std_params]
    _, std_data = MOTSTracker.max_constraint_along_curve(
        curve=c, points=std_params, fd_order=fd_order, full_output=True,
    )
    _, proper_data = MOTSTracker.max_constraint_along_curve(
        curve=c, points=proper_params, fd_order=fd_order, full_output=True,
    )
    key = "near_curve"
    if fd_order is not None:
        key = "%s_order%s" % (key, fd_order)
    return {
        key: dict(
            std=[std_data[:,0], std_data[:,1], std_data[:,2]],
            proper=[proper_data[:,0], proper_data[:,1], proper_data[:,2]],
        )
    }


def _ingoing_exp(c, Ns, eps):
    r"""Compute the ingoing expansion along the curve."""
    if Ns is None:
        Ns = max(100, 2*c.num)
    pts = np.linspace(eps, np.pi-eps, Ns)
    fvals = c.expansions(params=pts, ingoing=True)
    fx = np.array(list(zip(pts, fvals)))
    return dict(values=fx) # compatible with _sample_and_extremum() format


def _avg_ingoing_exp(c, epsabs, limit):
    r"""Compute the average ingoing expansion across the MOTS."""
    area = c.user_data['area']
    if epsabs == 'auto':
        if 'accurate_abs_delta' in c.user_data:
            epsabs = max(1e-6, c.user_data['accurate_abs_delta']['max'])
        else:
            epsabs = 1e-6
    if limit == 'auto':
        limit = max(50, int(c.num/2))
    return c.average_expansion(
        ingoing=True, area=area, epsabs=epsabs, limit=limit
    )


def _point_dists(c):
    r"""Compute the coordinate distances of all collocation points."""
    params = c.collocation_points()
    point_dists = c.point_distances(params)
    min_dist = point_dists.min()
    max_dist = point_dists.max()
    return dict(
        params=params,
        coord=dict(dists=point_dists, min=min_dist, max=max_dist),
        # leave open the option to include proper distances as well
    )


def _neck(c, xtol, epsrel):
    r"""Find and compute various neck properties for the inner common MOTS."""
    def _values(param):
        return dict(
            param=param,
            x_coord=c(param)[0],
            circumference=c.circumference(param),
            proper_x_dist=c.x_distance(param, epsrel=epsrel),
        )
    compute_props('inner', c, ['ricci', 'curv2'], verbosity=0)
    with c.fix_evaluator():
        return dict(
            coord=_values(c.find_neck('coord', xtol=xtol)[0]),
            circumference=_values(c.find_neck('circumference', xtol=xtol)[0]),
            proper_x_dist=_values(c.find_neck('proper_x_dist', xtol=xtol)[0]),
            max_ricci=_values(c.user_data['ricci']['x_max']),
            max_curv2=_values(c.user_data['curv2']['x_max']),
        )


def _dist_top_bot(c, rtol, allow_intersection, other_run, other_hname, out_dir):
    r"""Compute the proper distance of the two individual MOTSs."""
    c_other = _get_aux_curve(
        "%s/../../%s/%s" % (out_dir, other_run, other_hname),
        hname=other_hname,
        it=c.metric.iteration,
        disp=False,
    )
    if not c_other:
        raise AuxResultMissing()
    if c_other.metric.time != c.metric.time:
        raise RuntimeError("Auxiliary curve not in correct slice.")
    return c.z_distance(c_other, rtol=rtol,
                        allow_intersection=allow_intersection)


def _z_dist_to_inner(c, rtol, where, other_run, other_hname, out_dir):
    r"""Compute the proper distance of inner and individual MOTSs."""
    c_other = _get_aux_curve(
        "%s/../../%s/%s" % (out_dir, other_run, other_hname),
        hname=other_hname,
        it=c.metric.iteration,
        disp=False,
    )
    if not c_other:
        raise AuxResultMissing()
    if c_other.metric.time != c.metric.time:
        raise RuntimeError("Auxiliary curve not in correct slice.")
    return c.inner_z_distance(c_other, rtol=rtol, where=where)


def _signature(c, N, eps, c_past, c_future, has_past, has_future):
    r"""Compute the signature of the world tube traces out by MOTSs."""
    if N is None:
        N = max(200, c.num)
    pts = np.linspace(eps, np.pi-eps, N)
    return c.signature_quantities(
        pts,
        past_curve=c_past if has_past else None,
        future_curve=c_future if has_future else None,
    )


def _get_aux_curve(out_dir, hname, it, disp=True, verbose=False):
    r"""Load an auxiliary curve for the given iteration."""
    files = find_files(
        pattern="%s/%s_*_it%010d*.npy" % (out_dir, hname, it),
        skip_regex=r"_CE",
    )
    if not files:
        if disp:
            raise FileNotFoundError(
                "Auxiliary curve for horizon '%s' at iteration %s not found."
                % (hname, it)
            )
        return None
    files.sort()
    fname = files[-1]
    if verbose:
        print("Loading curve: %s" % fname)
    return BaseCurve.load(fname)


class _PropResult():
    r"""Simple storage class to help return results of computations."""

    def __init__(self, result, extras=None):
        r"""Store a result and optional extras in this instance."""
        self.result = result
        self.extras = extras


class AuxResultMissing(Exception):
    r"""Raised when a result is missing required for computing a property."""
    pass


class _NeckInfo():
    r"""Collection of data about the *neck* of an inner common MOTS."""

    def __init__(self, threshold1, threshold2, iteration, x_top=None,
                 z_top=None, x_bot=None, z_bot=None, x_neck=None, z_neck=None,
                 z_top_outer=None, z_bot_outer=None, pinching1=None,
                 pinching2=None):
        self.x_top = x_top
        self.z_top = z_top
        self.x_bot = x_bot
        self.z_bot = z_bot
        self.z_top_outer = z_top_outer
        self.z_bot_outer = z_bot_outer
        self.x_neck = x_neck
        self.z_neck = z_neck
        self.pinching1 = pinching1
        self.pinching2 = pinching2
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.iteration = iteration

    def update(self, **kw):
        r"""Update public instance attributes."""
        for k, v in kw.items():
            if hasattr(self, k):
                super().__setattr__(k, v)
            else:
                raise TypeError("Unknown parameter: %s" % (k,))

    @property
    def has_data(self):
        r"""Boolean whether we have any data about the neck at all."""
        return self.pinching1 is not None and self.pinching2 is not None

    @property
    def do_move_neck(self):
        r"""Boolean indicating whether neck pinching is above thresholds."""
        if not self.has_data:
            return False
        return (self.pinching1 >= self.threshold1
                or self.pinching2 >= self.threshold2)

    @property
    def z_center(self):
        r"""Center between top end of bottom and bottom end of top MOTS."""
        return (self.z_top + self.z_bot) / 2.0

    @property
    def z_dist(self):
        r"""Distance between top end of bottom and bottom end of top MOTS."""
        return abs(self.z_top - self.z_bot)

    @property
    def z_extent(self):
        r"""Distance between bottom end of bottom and top end of top MOTS."""
        return abs(self.z_top_outer - self.z_bot_outer)

    @property
    def smaller_z_extent(self):
        r"""Smaller distance from `z_center` to the other end of the MOTS."""
        center = self.z_center
        values = abs(self.z_top_outer - center), abs(center - self.z_bot_outer)
        return min(values)


def optimize_bipolar_scaling(curve, initial_scale, move, factor=0.75,
                             threshold_factor=5.0, res_decrease_factor=0.9,
                             initial_smoothing=None, max_smoothing=0.1,
                             verbose=True, recursions=1, max_shrinking=0.05):
    r"""Perform successive tests to try to find optimal bipolar scale parameter.

    This function parameterizes the given MOTS in bipolar coordinates and
    reparameterizes in the s-t-coordinate-space. Afterwards, the residual
    expansion between collocation points is measured. By changing the scaling
    parameter (half-distance of the two foci), the result will generally
    become better or worse, depending on how well the coordinates are adapted
    to the curve's shape. A very simple numerical search is used to find a
    scale that is better than the given `initial_scale`.

    @param curve
        MOTS to base the measurements on. Should have low residual expansion
        already.
    @param initial_scale
        Scaling to start with.
    @param move
        Origin, i.e. center between the foci of the bipolar coordinates.
    @param factor
        Factor by which to increase or decrease the scaling each step,
        respectively. Default is `0.75`, i.e. increase using ``value/0.75``
        and decrease using ``value*0.75``.
    @param threshold_factor
        Factor by which to increase the initial residual expansion. This
        defines the target resolution used in each reparameterization. The
        idea is to avoid being in the roundoff plateau, so that values react
        to how well the scaling is suited to the problem, and not fluctuate a
        lot. Default is `0.9`.
    @param res_decrease_factor
        Factor used for initially finding the resolution with which to work
        during the search. The initial resolution is reduced successively by
        this factor until the residual expansion is above
        ``threshold_factor*residual``. Default is `10`.
    @param initial_smoothing
        If given, perform an optimization for the smoothing parameter of the
        ``'curv2'`` reparameterization strategy, starting from the given
        value. This is done after the bipolar scaling parameter is optimized.
    @param max_smoothing
        Maximum smoothing value to allow. For high smoothing values, the
        results usually don't change anymore. Under certain circumstances, the
        search may start to increase this smoothing value indefinitely
        (effectively amounting to the limit of arc-length parameterization).
        Default is `0.1`.
    @param verbose
        Whether to print status messages. Default is `True`.
    @param recursions
        If the result gets worse for the initial steps up and down, put a
        parabola through the three obtained values. Its minimum is then taken
        as the initial step for another search. The `recursions` value
        determines how often this should happen (i.e. it is reduced by one
        each recursion). If ``recursions==0``, the minimum of the parabola is
        taken as optimal value.
    @param max_shrinking
        Upon recursion (see above), don't move too close to not moving at all
        (i.e. a `factor` of `1.0`). This parameter ensures that
        ``(new_factor-1)/(factor-1) >= max_shrinking``.
    """
    def set_value_for_scale(value, num):
        bipolar_curve = BipolarCurve.from_curve(curve, num=num, scale=value,
                                                move=move)
        opts = dict(coord_space=True)
        if initial_smoothing is None:
            reparam = 'arc_length'
        else:
            reparam = 'curv2'
            opts['smoothing'] = initial_smoothing
        bipolar_curve.reparameterize(reparam, **opts)
        return bipolar_curve
    if verbose:
        print("Finding bipolar scaling parameter...")
    scale, num = _optimize_curve_parameter(
        curve, x0=initial_scale, set_value=set_value_for_scale,
        factor=factor, threshold_factor=threshold_factor,
        res_decrease_factor=res_decrease_factor, verbose=verbose,
        recursions=recursions, full_output=True, max_shrinking=max_shrinking,
    )
    if initial_smoothing is None:
        return scale
    def set_value_for_smoothing(value, num):
        bipolar_curve = BipolarCurve.from_curve(curve, num=num, scale=scale,
                                                move=move)
        bipolar_curve.reparameterize('curv2', smoothing=value,
                                     coord_space=True)
        return bipolar_curve
    if verbose:
        print("Finding 'curv2' reparameterization smoothing using scale=%s..."
              % scale)
    smoothing = _optimize_curve_parameter(
        curve, x0=initial_smoothing, set_value=set_value_for_smoothing,
        factor=factor, num=num, verbose=verbose, recursions=recursions,
        max_value=max_smoothing, max_shrinking=max_shrinking,
    )
    return scale, smoothing


def _optimize_curve_parameter(curve, x0, set_value, factor, max_shrinking,
                              max_value=None, threshold_factor=None,
                              res_decrease_factor=None, verbose=True,
                              recursions=1, num=None, full_output=False,
                              _cache=None):
    r"""Vary a parameter to minimize the required resolution.

    This varies a parameter and repeatedly checks the residual expansion of
    the resulting curve to see at which value the residual has its minimal
    value. This should lead to a lower required resolution when using the
    found parameter value for reparametrizing the curve.
    """
    def _p(msg):
        if verbose:
            print(msg)
    _res_cache = dict() if _cache is None else _cache
    def _max_residual(num, value):
        key = (num, value)
        try:
            return _res_cache[key]
        except KeyError:
            pass
        modified_curve = set_value(value, num=num)
        c1 = RefParamCurve.from_curve(modified_curve, num=0,
                                      metric=curve.metric)
        space = np.linspace(0, np.pi, 2*num+1, endpoint=False)[1:]
        residual = np.absolute(c1.expansions(params=space)).max()
        _res_cache[key] = residual
        return residual
    a = x0
    if num is None:
        if threshold_factor is None or res_decrease_factor is None:
            raise TypeError("With `num` not specified, `threshold_factor` "
                            "and `res_decrease_factor` are mandatory.")
        err0 = _max_residual(num=curve.num, value=a)
        _p("residual expansion after conversion: %s" % err0)
        threshold = threshold_factor * err0
        num = curve.num
        while True:
            num = res_decrease_factor * num
            if _max_residual(num=int(num), value=a) >= threshold:
                break
        num = int(num)
    _p("performing search with resolution %s" % num)
    def f(x):
        value = _max_residual(num=num, value=x)
        _p("  residual for value=%s is: %.6e" % (x, value))
        return value
    def _step(x):
        x1 = factor * x
        return min(x1, max_value) if max_value else x1
    fa = f(a)
    b = _step(a)
    fb = f(b)
    if fb >= fa:
        factor = 1./factor
        c = _step(a)
        fc = f(c)
        if fc >= fa:
            # worse or equal in both directions
            data = sorted(
                # set() removes any duplicate entries
                set([(a, fa), (b, fb), (c, fc)]), key=lambda x: x[0]
            )
            xs, ys = zip(*data)
            if len(xs) < 3:
                if recursions > 0:
                    smaller_factor = 0.5 * (factor - 1.0) + 1.0
                    opt = x0 * smaller_factor
                else:
                    _p("Ending search at boundary.")
                    return (a, num) if full_output else a
            else:
                opt = lagrange(xs, np.log10(ys)).deriv().roots.real[0]
            if recursions > 0:
                _p("Worse or equal in both directions. Recursing...")
                new_factor = _limit_factor_shrinking(
                    opt/x0, factor, max_shrinking, verbose=verbose,
                )
                opt = _optimize_curve_parameter(
                    curve=curve, x0=x0, max_shrinking=max_shrinking,
                    max_value=max_value, set_value=set_value,
                    factor=new_factor, num=num, verbose=verbose,
                    recursions=recursions-1,
                    _cache=_res_cache,
                )
            return (opt, num) if full_output else opt
        a, b = b, c
        fa, fb = fb, fc
    while True:
        c = _step(b)
        if c == b:
            _p("Search reached boundary. Ending here.")
            return (c, num) if full_output else c
        fc = f(c)
        if fc > fb:
            opt = lagrange([a, b, c], np.log10([fa, fb, fc])).deriv().roots.real[0]
            return (opt, num) if full_output else opt
        a, b = b, c
        fa, fb = fb, fc


def _limit_factor_shrinking(new_factor, factor, max_shrinking, verbose):
    r"""Make sure the new factor is not too much closer to 1 than the old one."""
    if max_shrinking is None:
        return new_factor
    # Make sure both factors are on "the same side" of 1.
    if (new_factor-1) * (factor-1) < 0:
        factor = 1./factor
    if (new_factor-1) / (factor-1) < max_shrinking:
        fixed_factor = max_shrinking * (factor-1) + 1
        if verbose:
            print("Factor %s too much closer to 1.0 than the old one %s. "
                  "Changed to %s." % (new_factor, factor, fixed_factor))
        new_factor = fixed_factor
    return new_factor
