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

from ..utils import timethis, find_file, find_files
from ..numutils import inf_norm1d
from ..metric.simulation import SioMetric
from .curve import BaseCurve, ParametricCurve
from .findmots import GeneralMotsConfig, find_mots


__all__ = [
    "track_mots",
    "compute_props",
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
        ## Whether to go through slices forward in coordinate time (going back
        ## may be useful if the AH is initially found at a later slice and we
        ## want to check where it appeared).
        self.backwards = False
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
        ## MOTSs to the neck's width. If any of these two ratios lies above
        ## the respective threshold, the *neck trick* is performed.\ Note that
        ## an error is raised if either of the two individual MOTSs for this
        ## slice is not found.
        self.neck_trick_thresholds = (10.0, 2.0)
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
            g = SioMetric(self.files[i])
            g.release_file_handle()
            self._metrics[i] = g
        return self._metrics[i]

    def _p(self, msg):
        r"""Print the given message in case we should be verbose."""
        if self.verbosity >= 1:
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
            cfg.update(c_ref=self.neck_trick(g, self._c_ref))
            suffix = cfg.suffix
            c, fname = self._call_find_mots(cfg, pass_nr=1,
                                            timings=self.timings)
            if not c:
                return None, None
            if self.two_pass:
                cfg.update(suffix=suffix) # reset suffix if changed by cfg_callback
                self._c_ref = c
                c, fname = self._call_find_mots(
                    cfg, pass_nr=2, c_ref=self._c_ref, timings=self.timings
                )
                if not c:
                    return None, None
        self._c_ref = c
        c_past, c_future = self._get_future_past_curves(c, i, stride, try_no)
        self._compute_properties(c, fname, c_future=c_future, c_past=c_past)
        return c, fname

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
        if not c and not self._has_prop_to_do():
            do_load = True
            what = ["metric", "curv"]
        if not c and self._has_prop_to_do('stability'):
            do_load = True
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
        return prop in props or 'all' in props

    def _compute_properties(self, c, fname, c_future=None, c_past=None):
        r"""Call compute_props() for the given curve.

        In case properties have been computed and hence the curve data
        changed, the curve is re-saved to disk with the new data.
        """
        resave = False
        with timethis("Computing properties...",
                      silent=not (self.timings and self._has_prop_to_do())):
            if compute_props(hname=self.hname, c=c, props=self.props,
                             area_rtol=self.area_rtol, MOTS_map=self.MOTS_map,
                             verbosity=self.verbosity, fname=fname,
                             c_future=c_future, c_past=c_past):
                resave = True
        if resave and fname:
            c.save(fname, overwrite=True)

    def _aux_MOTS_dir(self, hname):
        r"""Get the directory (respecting the MOTS map) of auxiliary curves."""
        run = self.MOTS_map.get(hname, self._run_name)
        return op.join(self._parent_dir, run, hname)

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
        top_dir = self._aux_MOTS_dir('top')
        bot_dir = self._aux_MOTS_dir('bot')
        threshold1, threshold2 = self.neck_trick_thresholds
        if not hasattr(c, 'find_neck'):
            self._p("Not an ExpansionCurve. Neck moving not available.")
            return c
        try:
            x_neck, z_neck = c(c.find_neck('coord')[0])
        except ValueError:
            self._p("Neck not found.")
            return c
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
        pinching2 = abs(z_top - z_bot) / (2*x_neck)
        self._p("Individual horizons' distance / neck width: %s (threshold=%s)"
                % (pinching2, threshold2))
        if pinching1 < threshold1 and pinching2 < threshold2:
            self._p("Neck pinching below threshold. Not moving reference curve.")
            return c
        self._p("Neck pinching above threshold. Moving...")
        z_center = (z_top + z_bot) / 2.0
        if hasattr(c, 'ref_curve') and hasattr(c.ref_curve, 'z_fun'):
            c = c.copy()
            c.ref_curve = c.ref_curve.copy()
            c.ref_curve.z_fun.a_n[0] -= z_neck - z_center
            c.ref_curve.force_evaluator_update()
        else:
            # Not a RefParamCurve. Convert to parametric curve.
            c = ParametricCurve.from_curve(c, num=c.num)
            c.z_fun.a_n[0] -= z_neck - z_center
        self._p("Neck moved by dz=%s to z=%s" % (z_center - z_neck, z_center))
        return c

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
                    num=int(round(self.min_ref_num/self.ref_num_factor)),
                    ref_num=self.min_ref_num,
                )
        if self.initial_num is not None:
            cfg.update(num=self.initial_num)
        if self.ref_smoothing is not None:
            cfg.update(reparam=("curv2", dict(smoothing=self.ref_smoothing)))
        return cfg

    def _base_cfg(self, **kw):
        r"""Create a base configuration independently of the slice."""
        cfg = GeneralMotsConfig.preset(
            'discrete%s' % self.strategy, hname=self.hname,
            save=True, base_folder=self.out_base_dir, folder=self.folder,
            dont_compute=not self.compute, verbose=self.verbosity > 1,
        )
        cfg.update(**self.cfg)
        cfg.update(**kw)
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
        user_data = dict(
            cfg=dict(cfg=cfg, two_pass=self.two_pass, pass_nr=pass_nr)
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
        files = glob("%s/**/*.it*.s5" % self.sim_dir, recursive=True)
        # remove duplicates (caused by symlinks) by constructing a dict
        files = dict([(op.basename(fn), fn) for fn in files
                      if re.search(r"\.it\d{10}\.s5", fn)
                      and not re.search(r"output-\d{4}-active", fn)])
        files = list(files.items())
        files.sort(key=lambda f: f[0]) # sort based on base name
        files = [fn for _, fn in files] # return the relative names
        if self.backwards:
            files.reverse()
        active = False
        result = []
        for fn in files:
            if re.search(r"\.it%010d\.s5" % self.start_slice, fn):
                active = True
                if self.backwards:
                    # Starting a forward and backward search simultaneously
                    # requires one of them to ignore the start slice.
                    continue
            if active:
                result.append(fn)
            if (self.end_slice is not None
                    and re.search(r"\.it%010d\.s5" % self.end_slice, fn)):
                break
        if disp and not result:
            raise ValueError("No simulation data found in: %s" % self.sim_dir)
        if self.max_slices and len(result) > self.max_slices:
            result = result[:self.max_slices]
        return result


def compute_props(hname, c, props, area_rtol=1e-6, verbosity=True,
                  MOTS_map=None, fname=None, c_past=None, c_future=None):
    r"""Compute properties and store them in the curve object.

    All computed properties will be stored under the property's name in the
    curve's ``user_data`` dictionary. The arguments used for this computation
    are additionally stored under the key ``prop_args``, where ``prop`` is the
    property name. Some properties produce extra output. This is then stored
    under ``prop_extras``.

    The properties that can be computed here are:
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
            least 30 eigenvalues are computed.
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

    Some properties are only computed for certain horizons. The horizon
    specific properties are:
        * for ``"bot"``: ``dist_top_bot``
        * for ``"inner"``: ``neck, z_dist_top_inner, z_dist_bot_inner``

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
    """
    if MOTS_map is None:
        MOTS_map = dict()
    out_dir = run_name = None
    if fname:
        out_dir = op.dirname(fname)
        run_name = op.basename(op.dirname(out_dir))
    if not isinstance(props, (list, tuple)):
        props = [props]
    all_props = (None, 'none', 'all', 'ingoing_exp', 'avg_ingoing_exp',
                 'area', 'ricci', 'mean_curv', 'curv2', 'stability', 'neck',
                 'dist_top_bot', 'z_dist_top_inner', 'z_dist_bot_inner',
                 'signature', 'point_dists')
    for p in props:
        if p not in all_props:
            raise ValueError(
                "Invalid property to compute: %s. Valid properties are: %s"
                % (p, ",".join("%s" % pr for pr in all_props))
            )
    data = c.user_data
    did_something = [False]
    def do_prop(p, func, args=None, is_extra=False, **kw):
        if p not in all_props:
            raise RuntimeError("Unkown property: %s" % p)
        do = p in props or ('all' in props and not is_extra)
        if not do:
            return
        if args is None:
            args = dict()
        p_args = '%s_args' % p
        p_extras = '%s_extras' % p
        if p not in data or p_args not in data or data[p_args] != args:
            data[p_args] = args
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
            did_something[0] = True
    if hname == 'bot' and out_dir:
        do_prop('dist_top_bot', _dist_top_bot,
                dict(rtol=1e-5,
                     allow_intersection=True,
                     other_run=MOTS_map.get('top', run_name),
                     other_hname='top'),
                c=c, out_dir=out_dir)
    do_prop('area', _area, dict(epsrel=area_rtol, limit=100, v=1), c=c)
    do_prop('ingoing_exp', _ingoing_exp, dict(Ns=None, eps=1e-6), c=c)
    do_prop('avg_ingoing_exp', _avg_ingoing_exp,
            dict(epsabs='auto', limit='auto'), c=c)
    do_prop('stability', _stability, dict(min_num=30), c=c)
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
    if c_past or c_future:
        do_prop('signature', _signature,
                dict(N=None, eps=1e-6, has_past=bool(c_past),
                     has_future=bool(c_future)),
                c=c, c_past=c_past, c_future=c_future)
    return did_something[0]


def _area(c, v=None, **kw):
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

def _stability(c, min_num):
    r"""Compute the stability spectrum for the curve."""
    return c.stability_parameter(num=max(min_num, c.num), full_output=True)


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
        epsabs = max(1e-6, c.user_data['accurate_abs_delta']['max'])
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


def _get_aux_curve(out_dir, hname, it, disp=True):
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
    return BaseCurve.load(files[-1])


class _PropResult():
    r"""Simple storage class to help return results of computations."""

    def __init__(self, result, extras=None):
        r"""Store a result and optional extras in this instance."""
        self.result = result
        self.extras = extras


class AuxResultMissing(Exception):
    r"""Raised when a result is missing required for computing a property."""
    pass
