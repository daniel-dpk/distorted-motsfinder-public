r"""@package motsfinder.axisym.trackmots

Track a single MOTS through slices of a simulation.
"""

from glob import glob
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
    def __init__(self, hname, sim_dir, out_base_dir, folder, initial_guess, **kw):
        self.hname = hname
        self.sim_dir = sim_dir
        self.out_base_dir = out_base_dir
        self.folder = folder
        self.initial_guess = initial_guess
        self.MOTS_map = None
        self.compute = True
        self.two_pass = True
        self.max_slices = None
        self.start_slice = 0
        self.end_slice = None
        self.initial_stride = 1
        self.backwards = False
        self.retry_after = None
        self.cfg_callback = None
        self.props = None
        self.area_rtol = 1e-6
        self.strategy = 1
        self.verbosity = 1
        self.timings = False
        self.full_output = False
        self.initial_num = None
        self.min_ref_num = 5
        self.ref_num_factor = 0.1
        self.ref_smoothing = None
        self.do_neck_trick = False
        self.neck_trick_thresholds = (10.0, 2.0)
        self._files = None
        self._c_ref = None
        self._c_ref_prev = None
        self._metrics = None
        self._parent_dir = None
        self._run_name = None
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
        if self._files is None:
            self._files = self.get_sim_files()
        return self._files

    def get_metric(self, i):
        if self._metrics is None:
            self._metrics = [None] * len(self.files)
        if self._metrics[i] is None:
            g = SioMetric(self.files[i])
            g.release_file_handle()
            self._metrics[i] = g
        return self._metrics[i]

    def _p(self, msg):
        if self.verbosity >= 1:
            print(msg)

    def _init_tracking(self):
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
        g = self.get_metric(i)
        try:
            return self._do_step(g, i, try_no, stride)
        finally:
            g.unload_data()

    def _do_step(self, g, i, try_no, stride):
        self._c_ref_prev = self._c_ref.copy()
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
        run = self.MOTS_map.get(hname, self._run_name)
        return op.join(self._parent_dir, run, hname)

    def neck_trick(self, g, c):
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
        suffix = "it%010d" % g.iteration
        if try_no is not None:
            suffix += "_try%04d" % try_no
        c_ref = self._c_ref
        cfg = self._base_cfg(metric=g, suffix=suffix, c_ref=c_ref)
        if self.strategy >= 2:
            cfg.update(
                num=c_ref.num,
                ref_num=max(
                    self.min_ref_num,
                    int(round(self.ref_num_factor*c_ref.num))
                )
            )
        if self.initial_num is not None:
            cfg.update(num=self.initial_num)
        if self.ref_smoothing is not None:
            cfg.update(reparam=("curv2", dict(smoothing=self.ref_smoothing)))
        return cfg

    def _base_cfg(self, **kw):
        cfg = GeneralMotsConfig.preset(
            'discrete%s' % self.strategy, hname=self.hname,
            save=True, base_folder=self.out_base_dir, folder=self.folder,
            dont_compute=not self.compute, verbose=self.verbosity > 1,
        )
        cfg.update(**self.cfg)
        cfg.update(**kw)
        return cfg

    def _load_existing_curve(self, cfg, quiet=False, only_ok=False):
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
        pass_nr = 2 if self.two_pass else 1
        try:
            return self._call_find_mots(cfg, pass_nr, dont_compute=True,
                                        save=True)
        except FileNotFoundError:
            return None, None

    def _call_find_mots(self, cfg, pass_nr, timings=False, **kw):
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
    return c.stability_parameter(num=max(min_num, c.num), full_output=True)


def _ricci(c, Ns, eps, xatol):
    return _sample_and_extremum(c, c.ricci_scalar, Ns=Ns, eps=eps, xatol=xatol)


def _mean_curv(c, Ns, eps, xatol):
    def func(x):
        return c.extrinsic_surface_curvature(x, trace=True)
    return _sample_and_extremum(c, func, Ns=Ns, eps=eps, xatol=xatol)


def _curv_squared(c, Ns, eps, xatol):
    def func(x):
        return c.extrinsic_surface_curvature(x, square=True)
    return _sample_and_extremum(c, func, Ns=Ns, eps=eps, xatol=xatol)


def _sample_and_extremum(c, func, Ns, eps, xatol):
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
    if Ns is None:
        Ns = max(100, 2*c.num)
    pts = np.linspace(eps, np.pi-eps, Ns)
    fvals = c.expansions(params=pts, ingoing=True)
    fx = np.array(list(zip(pts, fvals)))
    return dict(values=fx) # compatible with _sample_and_extremum() format


def _avg_ingoing_exp(c, epsabs, limit):
    area = c.user_data['area']
    if epsabs == 'auto':
        epsabs = max(1e-6, c.user_data['accurate_abs_delta']['max'])
    if limit == 'auto':
        limit = max(50, int(c.num/2))
    return c.average_expansion(
        ingoing=True, area=area, epsabs=epsabs, limit=limit
    )


def _point_dists(c):
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
    if N is None:
        N = max(200, c.num)
    pts = np.linspace(eps, np.pi-eps, N)
    return c.signature_quantities(
        pts,
        past_curve=c_past if has_past else None,
        future_curve=c_future if has_future else None,
    )


def _get_aux_curve(out_dir, hname, it, disp=True):
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
    def __init__(self, result, extras=None):
        self.result = result
        self.extras = extras


class AuxResultMissing(Exception):
    r"""Raised when a result is missing required for computing a property."""
    pass
