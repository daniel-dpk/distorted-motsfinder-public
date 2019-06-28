r"""@package motsfinder.axisym.initialguess

Helpers for determining MOTSs without data from previous steps.

The InitHelper class assumes the metric to roughly resemble a Brill-Lindquist
setting to construct estimates for the four different MOTSs that usually
appear there:
    * the common apparent horizon (`AH`)
    * the two individual MOTSs (`top` and `bot`)
    * the inner (unstable) branch of the common horizon (`inner`)

The AH and the two individual horizons are found starting from suitable round
spheres as initial guesses. These should be relatively easy to find. Adjust
the individual masses and distance parameters in case these are not found
successfully.

For the inner common MOTS, the strategy here is to take the AH and then find
constant expansion surfaces (CESs) close to it with expansion `< 0` to go
inwards. At some point, the minimal expansion for which a CES exists is found
(approximately). At this point, increasing the expansion can either take us on
the way back to the AH, or towards the inner common MOTS. To take the correct
direction, we "jump" over the exact minimum (assuming we're away from it on
the order of the smallest expansion step size considered) by repeating the
previous shape change and finding a CES of the same expansion but different
shape (i.e. "on the other side" of the minimum). Increasing the expansion in
small steps from there then leads to the inner MOTS.

This "jump" is a critical point in the current strategy: It may fail to
converge (if the artificial shape change is too large) or it may simply find
the previous CES again (if the change is too small). There is therefore the
possibility to tweak this change using, among others, the `jump_mult`
parameter to scale the applied change.

During the whole search, interim results are stored to disk and reused when
the helper is rerun, at least if an `out_base` folder is specified. This means
that you can easily tweak the settings and quickly find out if they help in
finding all the MOTSs. In most cases, though, the defaults will suffice and
even tweaking `jump_mult` is rarely needed.


@b Examples

The first example finds the initial MOTSs of an analytical Brill-Lindquist
configuration:
```
    g = BrillLindquistMetric(d=0.6, m1=0.2, m2=0.8)
    h = InitHelper(
        metric=g, out_base="some/output/folder",
        suffix="d%s_m%s_%s" % (g.d, g.m1, g.m2)
    )
    curves = h.find_four_MOTSs(m1=g.m1, m2=g.m2, d=g.d, plot=True)
```

This example uses simulation data to find the initial MOTSs in:
```
    it = 0 # initial iteration of the simulation
    g = SioMetric(
        find_file("simulation_run/**/*it%010d.s5" % it,
                  recursive=True, verbose=True)
    )
    h = InitHelper(metric=g, out_base="some/output/folder",
                   suffix="it%010d" % it)

    # Note: The numbers are just examples for a specific case.
    curves = h.find_four_MOTSs(m1=0.5, m2=0.8, d=0.75, plot=True)
```
"""

from contextlib import contextmanager

import numpy as np

from ..utils import merge_dicts, update_dict, insert_missing
from ..numutils import clip
from .findmots import GeneralMotsConfig, find_mots
from .curve import BaseCurve, ParametricCurve


__all__ = [
    "InitHelper",
]


class InitHelper():
    r"""Helper to find the initial MOTSs for a Brill-Lindquist-like metric.

    Note that the metric itself can be arbitrary (except it must be
    axisymmetric), so even non-time-symmetric metrics are supported. However,
    the ability to find the initial MOTSs depends on how well the actual
    metric resembles the Brill-Lindquist case. Most importantly, the apparent
    horizon and the two individual MOTSs have to be found using an assumed
    coordinate distance `d` of the singularities and approximate individual
    (bare) masses `m1` and `m2`. The inner common MOTS is then found by
    walking along the area-expansion curve from the apparent horizon inward.

    See the package documentation for motsfinder.axisym.initialguess for more
    information and examples.
    """

    def __init__(self, metric, out_base=None, suffix='', **kw):
        r"""Create an initial guess helper object.

        @param metric
            The metric of the slice to find MOTSs in. Note that any non-zero
            extrinsic curvature has to be supplied by the metric as specified
            by the interface motsfinder.metric.base._ThreeMetric.
        @param out_base
            Optional folder to write interim results to. If given, a subfolder
            ``{out_base}/init`` is created and all found MOTSs and CESs are
            stored there.
        @param suffix
            Optional suffix for all files written.
        @param **kw
            Additional keyword arguments supplied to all .findmots.find_mots()
            calls. A notable one is `verbose`, which controls the printed
            output of all methods of this class.
        """
        ## The metric supplied during construction.
        self.metric = metric
        ## The output base folder for storing (interim) results.
        self.out_base = out_base
        ## Suffix to use for filenames.
        self.suffix = suffix
        ## Additional options for the .findmots.find_mots() calls.
        self.opts = kw

    def find_AH(self, m1, m2, d, hname='AH', **kw):
        r"""Find the apparent horizon (AH).

        This is based on a (crude) estimate of the coordinate radius of the AH
        based on the bare masses and mutual coordinate distances of the two
        punctures.
        """
        guessed_radius = (m1+m2)/2. + d/3.
        return self._find(hname, **insert_missing(kw, c_ref=guessed_radius))

    def find_individual(self, hname, m, z, **kw):
        r"""Find one of the two individual MOTSs.

        @param hname
            "Name" of this MOTS. Should be either ``"top"`` or ``"bot"``.
        @param m
            Bare mass of the puncture (i.e. Brill-Lindquist mass parameter).
            Does not need to be very accurate, it is just used for the initial
            round sphere's coordinate radius.
        @param z
            Rough coordinate location of the puncture on the z-axis.
        @param **kw
            Further keyword arguments used in the .findmots.find_mots() call.
            These override any keyword arguments given upon object
            construction.
        """
        guessed_radius = min(m/2., abs(z))
        if guessed_radius == 0.0:
            raise ValueError(
                "Use `find_AH()` in case of only one BH. "
                "Otherwise, make sure the BHs are distributed at roughly "
                "equal distances from the origin."
            )
        return self._find(
            hname,
            **insert_missing(kw, c_ref=(guessed_radius, z))
        )

    def find_inner(self, c_AH, start_ex=-0.01, max_step=0.05, min_step=0.005,
                   jump_mult=1, jump_offset=(), num=60, hname='inner',
                   unload=True, back_kw=None, **kw):
        r"""Try to find the inner MOTS by walking along an area-expansion curve.

        See figure 13 in [1] for an example and description of area-expansion
        curves. The idea here is basically that we start at the AH (the bottom
        dot at expansion zero) and walk along the curve to end up at the inner
        common MOTS. To accomplish this, we find constant expansion surfaces
        (CESs) of increasingly negative expansion until we are close to the
        minimum expansion. Extrapolating the previous shape change, we "jump"
        over the minimum and then find CESs of increasing expansion. This then
        leads to the inner common MOTS. This jump is currently done based on
        simple heuristics and hence is not robust, i.e. it may fail in
        particular situations. Usually, however, simply tweaking the
        `jump_mult` parameter suffices to "make the jump".

        Upon successful steps in expansion, the step size is increased by the
        "golden ratio" (`(1+sqrt(5))/2`) to speed up the process. Failing
        steps are retried with a smaller step size. If the smallest step size
        has been reached this way, we define the current expansion as "close
        enough" to the actual minimum and try to make the jump. Note that if
        this happens on our way back up to zero expansion, the MOTS probably
        does not exist and we give up instead.

        A useful strategy for tweaking the settings in case the inner MOTS is
        not found successfully is to first let this function produce and store
        interim results up to the point it fails. Then, rerun it with the
        argument ``plot_steps=True`` to see what the issue is. In most cases,
        the jump is either too large or too small and a human can easily
        identify the required changes to the strategy and tweak the
        `jump_mult` parameter. In some cases (depending on how far apart the
        AH and inner MOTS already are), it may also be necessary to reduce or
        increase the scale of the minimum and/or maximum expansion step sizes
        using `min_step` and `max_step`, respectively. Other things to try
        include reducing the `linear_regime_threshold` and/or `step_mult`.
        As a last resort, you may specify `jump_offset` coefficient to
        directly influence the shape of the jumped to guess.

        @param c_AH
            Apparent horizon to start the CES progression.
        @param start_ex
            First (negative) expansion to try to find a CES for. This also
            becomes the first expansion step size tried. Default is `-0.01`.
        @param max_step
            Maximum expansion step size. Default is `0.05`.
        @param min_step
            Smallest expansion step size. Default is `0.005`.
        @param jump_mult
            Factor by which to multiply the last shape change for jumping.
            Default is `1.0`.
        @param jump_offset
            Coefficients added to the horizon function for making the jump.
        @param num
            Resolution used for the curves during the search. Default is `60`.
            No check is done to ensure this resolution is sufficient to
            satisfy any tolerance for the expansion, i.e. the resulting
            surfaces are not necessarily accurate MOTSs. However, they should
            be ideal starting points for finding more accurate MOTSs.
        @param hname
            Name of this MOTS. Default is ``"inner"``.
        @param unload
            Whether to unload any numerical data from the metric after finding
            the MOTS (or failing to find it). Default is `True`. Has no effect
            for metrics without an `unload_data()` method.
        @param back_kw
            Keyword arguments to update the search parameters with after a
            successful jump. Can be used, e.g., to increase the resolution on
            the way back up to the inner MOTS.
        @param **kw
            Additional keyword arguments supplied to the .findmots.find_mots()
            call. These may (partially) be overridden by `back_kw` applied
            after the jump.

        @b References

        [1] D. Pook-Kolb, O. Birnholtz, B. Krishnan and E. Schnetter.
            "Existence and stability of marginally trapped surfaces in
            black-hole spacetimes." Physical Review D 99.6 (2019): 064005.
        """
        with self._unload_after(unload=unload):
            return self._find_inner(
                c_AH, start_ex, max_step, min_step, jump_mult, jump_offset,
                num, hname, back_kw, **kw
            )

    def _find_inner(self, c_AH, start_ex, max_step, min_step, jump_mult,
                    jump_offset, num, hname, back_kw, **kw):
        r"""Implements find_inner()."""
        if self.out_base:
            try:
                return self._find(hname, dont_compute=True, disp=True,
                                  verbose=False, unload=False)
            except FileNotFoundError:
                pass
        verbose = kw.get('verbose', self.opts.get('verbose', True))
        def _p(msg):
            if verbose:
                print(msg)
        golden = 1.61803398874989 # (1+sqrt(5)) / 2 "golden ratio"
        opts = dict(reparam=True, reparam_with_metric=False, ref_num=10,
                    num=num, save_failed_curve=True, unload=False, disp=False,
                    verbose=verbose)
        max_step = abs(max_step)
        min_step = abs(min_step)
        step = start_ex
        if step >= 0.0:
            raise ValueError("Must start with negative expansion.")
        c = c_AH.copy()
        ex = 0
        jump = 0
        ex_steps = []
        def _s(s=''):
            return "j%s_CE%s%s" % (jump, ex, ("_%s" % s) if s else '')
        def _ok(c):
            return c and c.user_data.get("converged", True)
        while True:
            ex_prev = ex
            ex = min(0.0, ex + step)
            _p("Expansion: %s, current step: %s" % (ex, step))
            c_prev = c
            if ex == 0.0:
                _p("Trying to find actual MOTS")
                c = self._find(
                    hname=hname, c_ref=c,
                    **update_dict(opts, save_failed_curve=False, **kw)
                )
                if _ok(c):
                    return c
            else:
                c = self._find(
                    hname='AH', c_ref=c, c=ex, steps=30,
                    suffix=_s('from%s' % ex_prev),
                    **merge_dicts(opts, kw)
                )
            sgn = -1 if step < 0 else 1
            if _ok(c):
                _p("CES found for ex=%s. Increasing step size." % (ex))
                ex_steps.append(ex)
                step = sgn * clip(golden*abs(step), min_step, max_step)
            if not _ok(c):
                _p("CES not found for ex=%s" % (ex))
                c = c_prev
                ex = ex_prev
                if abs(step) > min_step:
                    step = sgn * clip(abs(step)/3., min_step, max_step)
                    _p("  shrinking step size to %s" % (step))
                    continue
                if jump:
                    raise RuntimeError("Inner common MOTS not found.")
                _p("Minimum step size reached. "
                   "Trying to jump with ex=%s while turning around..." % ex)
                step = -step # turn around towards zero expansion
                jump = 1
                offset_coeffs = list(
                    jump_mult * 1/(50*abs(ex_steps[-1]-ex_steps[-2]))
                    * np.asarray(c.h.a_n)
                )
                for i, val in enumerate(jump_offset):
                    if len(offset_coeffs) == i:
                        offset_coeffs.append(val)
                    else:
                        offset_coeffs[i] += val
                def veto_cb(curve, cfg):
                    if _is_same_curve(curve, c_prev, tol=1e-4):
                        raise RuntimeError(
                            "Did not make the jump. Not saving curve. "
                            "Change jump settings and try again."
                        )
                    return False # don't veto
                c = self._find(
                    hname='AH', c_ref=c, c=ex, steps=30,
                    offset_coeffs=offset_coeffs, veto_callback=veto_cb,
                    suffix=_s('from%s' % ex_prev),
                    **merge_dicts(opts, kw)
                )
                _p("Jump successful. Continuing towards zero expansion.")
                opts.update(back_kw or dict())

    def find_individuals(self, m1, m2, d, kw_top=None, kw_bot=None,
                         unload=True, plot=False, plot_opts=None, **kw):
        r"""Find the two individual MOTSs.

        This calls find_individual() once for the top and once for the bottom
        MOTS and returns the resulting curves.

        @param m1,m2
            Bare masses of the two punctures (i.e. Brill-Lindquist mass
            parameters). These do not need to be very accurate, they are just
            used for the initial round spheres' coordinate radii.
        @param d
            Rough coordinate distance of the punctures on the z-axis. Note
            that currently the punctures need to be approximately the same
            coordinate distance from the origin.
        @param kw_top,kw_bot
            Extra arguments used in the .findmots.find_mots() call for finding
            the top and bottom MOTSs, respectively.
        @param unload
            Whether to unload any numerical data from the metric afterwards.
            Default is `True`.
        @param plot
            Whether to plot the found MOTSs. Default is `False`.
        @param plot_opts
            Options supplied to the plotting command.
        @param **kw
            Common options for both ``find_individual()`` calls. These are
            overridden by the individual `kw_...` arguments.
        """
        with self._unload_after(unload=unload):
            c_top = self.find_individual('top', m=m1, z=d/2., unload=False,
                                         **merge_dicts(kw, kw_top or dict()))
            c_bot = self.find_individual('bot', m=m2, z=-d/2., unload=False,
                                         **merge_dicts(kw, kw_bot or dict()))
        if plot:
            title = "MOTSs"
            if hasattr(self.metric, 'time'):
                title += r" $t = %s$" % self.metric.time
            BaseCurve.plot_curves(
                (c_top, 'top', ':m'), (c_bot, 'bottom', '--r'),
                **insert_missing(
                    plot_opts or dict(), title=title,
                )
            )
        return c_top, c_bot

    def find_four_MOTSs(self, m1, m2, d, kw_AH=None, kw_top=None, kw_bot=None,
                        kw_inner=None, unload=True, plot=False,
                        plot_opts=None, **kw):
        r"""Find the two individual and the two common MOTSs.

        Based on a Brill-Lindquist-like configuration given by `m1`, `m2`,
        `d`, this method tries to find all the four MOTSs that can
        individually be obtained using find_AH(), find_individual() and
        find_inner(). See their documentation for more info.

        @param m1,m2
            Bare masses of the two punctures (i.e. Brill-Lindquist mass
            parameters). These do not need to be very accurate, they are just
            used for the initial round spheres' coordinate radii.
        @param d
            Rough coordinate distance of the punctures on the z-axis. Note
            that currently the punctures need to be approximately the same
            coordinate distance from the origin.
        @param kw_AH,kw_top,kw_bot,kw_inner
            Optional arguments for the individual ``find_...()`` calls.
        @param unload
            Whether to unload any numerical data from the metric afterwards.
            Default is `True`.
        @param plot
            Whether to plot the found MOTSs. Default is `False`.
        @param plot_opts
            Options supplied to the plotting command.
        @param **kw
            Common options for all ``find_...()`` calls. These are overridden
            by the individual `kw_...` arguments.
        """
        with self._unload_after(unload=unload):
            c_AH = self.find_AH(
                m1=m1, m2=m2, d=d, unload=False,
                **merge_dicts(kw, kw_AH or dict())
            )
            c_top, c_bot = self.find_individuals(
                m1=m1, m2=m2, d=d,
                kw_top=merge_dicts(kw, kw_top or dict()),
                kw_bot=merge_dicts(kw, kw_bot or dict()),
                unload=False
            )
            c_inner = self.find_inner(
                c_AH, unload=False,
                reference_curves=[
                    (c_AH, 'AH', '-b'), (c_top, 'top', '-m'),
                    (c_bot, 'bottom', '-r')
                ],
                **merge_dicts(kw, kw_inner or dict())
            )
        if plot:
            title = "MOTSs"
            if hasattr(self.metric, 'time'):
                title += r" $t = %s$" % self.metric.time
            BaseCurve.plot_curves(
                (c_AH, 'AH', '-b'), (c_top, 'top', ':m'),
                (c_bot, 'bottom', '--r'), (c_inner, 'inner', '-g'),
                **insert_missing(
                    plot_opts or dict(), title=title,
                )
            )
        return c_AH, c_top, c_bot, c_inner

    @contextmanager
    def _unload_after(self, unload=True):
        r"""Context to temporarily fix the evaluator(s) of used expressions."""
        try:
            yield
        finally:
            if unload:
                self.unload_data()

    def unload_data(self):
        r"""Unload any numerical data from the metric to free memory.

        Note that this has no effect in case the metric does not have an
        `unload_data()` method, as is the case for analytical metrics.
        """
        try:
            self.metric.unload_data()
        except AttributeError:
            pass

    def _find(self, hname, suffix='', unload=True, preset='discrete1', **kw):
        r"""Construct the configuration and call find_mots()."""
        with self._unload_after(unload=unload):
            out_folder = None
            if self.out_base:
                out_folder = "%s/init" % self.out_base
            cfg = GeneralMotsConfig.preset(
                preset, metric=self.metric, hname=hname,
                reparam=False, linear_regime_threshold=0.1,
                save_failed_curve=False, simple_name=True,
                out_folder=out_folder,
            )
            if suffix:
                suffix = "%s_%s" % (self.suffix, suffix)
            else:
                suffix = self.suffix
            cfg.update(suffix=suffix, **self.opts)
            cfg.update(save_verbose=cfg.newton_args.get('verbose', True))
            return find_mots(cfg, **kw)


def _is_same_curve(curve1, curve2, tol=1e-8):
    r"""Return whether two given curves appear to be the same.

    This aims to find out whether the curves describe the same geometrical
    object (i.e. irrespective of their parameterization). To do this, the
    curves are both first parameterized by arc length in coordinate space and
    then compared point-by-point. If the largest distance is smaller than the
    given tolerance `tol`, then the curves are assumed to be equivalent.

    Note that the curves themselves are not modified by this function.
    """
    num = max(curve1.num, curve2.num)
    c1 = ParametricCurve.from_curve(curve1, num)
    c1.reparameterize(metric=None)
    c2 = ParametricCurve.from_curve(curve2, num)
    c2.reparameterize(metric=None)
    c_norm = c1.inf_norm(c2)[1]
    return c_norm <= tol
