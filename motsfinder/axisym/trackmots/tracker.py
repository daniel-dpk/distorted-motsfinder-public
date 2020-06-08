r"""@package motsfinder.axisym.trackmots.tracker

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

import numbers
import os.path as op
import time

import numpy as np

from ...utils import timethis, find_file
from ...metric.simulation import SioMetric
from ..curve import BaseCurve, ParametricCurve
from ..findmots import GeneralMotsConfig, find_mots
from .findslices import find_slices
from .optimize import optimize_bipolar_scaling
from .props import compute_props, max_constraint_along_curve
from .props import NEED_ACTIVATION_PROPS


__all__ = [
    "track_mots",
    "MOTSTracker",
]


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

    For some tasks, the finder or property runner needs to consider sibling
    MOTSs (e.g. to compute distances between MOTSs). Sibling MOTSs are located
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
        ## Highest mass multipole moment to compute.\ Default is `10`.
        self.max_multipole_n = 10
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
        self._compute_properties(c, fname)
        # NOTE: Assigning to `self._c_ref` has the effect that *any*
        #       subsequent invocation of `_call_find_mots()` deletes the
        #       search config of the assigned curve (the data in
        #       `c_ref.user_data['cfg']`). Hence, we need to take care not to
        #       trigger such a call before the result has been saved. Note
        #       that this is not a problem here, since the above `find_mots()`
        #       calls have stored the result already and/or the
        #       `_compute_properties()` call did too. From now on, the curve
        #       is only used as reference curve can *can* have its search
        #       config be removed.
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

    def _compute_properties(self, c, fname):
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
                             max_multipole_n=self.max_multipole_n,
                             MOTS_map=self.MOTS_map, verbosity=self.verbosity,
                             fname=fname):
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
    def max_constraint_along_curve(cls, *args, **kwargs):
        r"""Calls `props.max_constraint_along_curve()`."""
        return max_constraint_along_curve(*args, **kwargs)

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
                'converged', 'plateau', 'knee',
                'insufficient resolution or roundoff',
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
