r"""@package motsfinder.axisym.initialguess

Helpers for determining MOTSs without data from previous steps.

@b Examples

```
    it = 0 # initial iteration of the simulation
    g = SioMetric(find_file("simulation_run/**/*it%010d.s5" % it),
                  recursive=True, verbose=True)
    h = InitHelper(metric=g, out_base="some/output/folder",
                   suffix="it%010d" % it)

    # This will try to find the AH and the two individual MOTSs first,
    # followed by a search for the inner common MOTS by starting at the AH and
    # using constant expansion surfaces (CESs) to approach the MOTS.
    h.find_four_MOTSs(m1=0.5, m2=0.8, d=0.75, plot=True)
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
    def __init__(self, metric, out_base, suffix='', **kw):
        self.metric = metric
        self.out_base = out_base
        self.suffix = suffix
        self.opts = kw

    def find_AH(self, m1, m2, d, hname='AH', **kw):
        guessed_radius = (m1+m2)/2. + d/3.
        return self._find(hname, c_ref=guessed_radius, **kw)

    def find_individual(self, hname, m, z, **kw):
        guessed_radius = min(m/2., abs(z))
        if guessed_radius == 0.0:
            raise ValueError(
                "Use `find_AH()` in case of only one BH. "
                "Otherwise, make sure the BHs are distributed at roughly "
                "equal distances from the origin."
            )
        return self._find(hname, c_ref=(guessed_radius, z), **kw)

    def find_inner(self, c_AH, start_ex=-0.01, max_step=0.05, min_step=0.005,
                   jump_mult=1, jump_offset=(), num=60, hname='inner',
                   unload=True, verbose=True, back_kw=None, **kw):
        with self._unload_after(unload=unload):
            return self._find_inner(
                c_AH, start_ex, max_step, min_step, jump_mult, jump_offset,
                num, hname, verbose, back_kw, **kw
            )

    def _find_inner(self, c_AH, start_ex, max_step, min_step, jump_mult,
                    jump_offset, num, hname, verbose, back_kw, **kw):
        try:
            return self._find(hname, dont_compute=True, disp=True,
                              verbose=False, unload=False)
        except FileNotFoundError:
            pass
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
                         unload=True, plot=False, plot_opts=None):
        with self._unload_after(unload=unload):
            c_top = self.find_individual('top', m=m1, z=d/2., unload=False,
                                         **(kw_top or dict()))
            c_bot = self.find_individual('bot', m=m2, z=-d/2., unload=False,
                                         **(kw_bot or dict()))
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
                        plot_opts=None):
        with self._unload_after(unload=unload):
            c_AH = self.find_AH(m1=m1, m2=m2, d=d, unload=False,
                                **(kw_AH or dict()))
            c_top, c_bot = self.find_individuals(
                m1=m1, m2=m2, d=d, kw_top=kw_top, kw_bot=kw_bot, unload=False
            )
            c_top = self.find_individual('top', m=m1, z=d/2., unload=False,
                                         **(kw_top or dict()))
            c_bot = self.find_individual('bot', m=m2, z=-d/2., unload=False,
                                         **(kw_bot or dict()))
            c_inner = self.find_inner(
                c_AH, unload=False,
                reference_curves=[
                    (c_AH, 'AH', '-b'), (c_top, 'top', '-m'),
                    (c_bot, 'bottom', '-r')
                ],
                **(kw_inner or dict())
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
        try:
            self.metric.unload_data()
        except AttributeError:
            pass

    def _find(self, hname, suffix='', unload=True, **kw):
        with self._unload_after(unload=unload):
            cfg = GeneralMotsConfig.preset(
                "discrete1", metric=self.metric, hname=hname,
                reparam=False, linear_regime_threshold=0.1,
                save_failed_curve=False, simple_name=True,
                out_folder="%s/init" % self.out_base,
            )
            if suffix:
                suffix = "%s_%s" % (self.suffix, suffix)
            else:
                suffix = self.suffix
            cfg.update(suffix=suffix, **self.opts)
            return find_mots(cfg, **kw)


def _is_same_curve(curve1, curve2, tol=1e-8):
    num = max(curve1.num, curve2.num)
    c1 = ParametricCurve.from_curve(curve1, num)
    c1.reparameterize(metric=None)
    c2 = ParametricCurve.from_curve(curve2, num)
    c2.reparameterize(metric=None)
    c_norm = c1.inf_norm(c2)[1]
    return c_norm <= tol
