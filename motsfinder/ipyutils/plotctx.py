r"""@package motsfinder.ipyutils.plotctx

Contexts for convenient plot setup/management.
"""

from __future__ import print_function

from contextlib import contextmanager
import subprocess
import os
import os.path as op
import inspect

import numpy as np
import matplotlib as mpl
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    mpl.use('agg', warn=False, force=True)  # switch to a more basic backend
    import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
# This import has side-effects we need.
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import

from ..utils import insert_missing, merge_dicts
from .plotutils import add_zero_line


__all__ = [
    "matplotlib_rc",
    "simple_plot_ctx",
    "wrap_simple_plot_ctx",
    "plot_ctx",
    "plot_ctx_3d",
    "pi_ticks",
]


def has_latex():
    """Return whether the 'latex' command is available on the system."""
    try:
        subprocess.check_output(['latex', '--version'])
        return True
    except OSError:
        return False


@contextmanager
def matplotlib_rc(opts):
    r"""Context manager to temporarily modify Matplotlib settings.

    The option keys should be valid keys in the `matplotlib.rcParams` dict.
    """
    old_values = dict((k, mpl.rcParams[k]) for k in opts)
    try:
        for k in opts:
            if k == 'text.usetex' and opts[k] and not has_latex():
                print("WARNING: LaTeX command not available. Using default.")
                continue
            mpl.rcParams[k] = opts[k]
        yield
    finally:
        for k in old_values:
            mpl.rcParams[k] = old_values[k]


def _equal_lengths(axes, ax):
    r"""Modify an Axis object to have equal length axes.

    @param ax
        String containing the axes, like ``"xyz"`` that should have equal
        length.
    """
    lims = [getattr(ax, "get_%slim" % x)() for x in axes]
    max_length = max(x[1]-x[0] for x in lims)
    def _lim(a, b, l):
        c = 0.5 * (a + b)
        d = 0.5 * l
        return c - d, c + d
    for x, lim in zip(axes, lims):
        getattr(ax, "set_%slim" % x)(*_lim(*lim, l=max_length))


@contextmanager
def simple_plot_ctx(figsize=(6, 2), projection=None, usetex=None,
                    fontsize=None, save=None, ax=None, show=True, close=False,
                    cfg_callback=None, dpi=None, save_opts=None,
                    subplot_kw=None, subplots_kw=None, fixed_layout=None,
                    pad=0, ypad=0, zero_line=False, xlog=False, ylog=False,
                    xlim=(None, None), ylim=(None, None), xtick_spacing=None,
                    ytick_spacing=None, pi_xticks=None, pi_yticks=None,
                    nrows=1, ncols=1):
    r"""Simple context for creating an setting up a figure and axis/axes.

    The given `ax` axis object can be a callable, in which case it is used to
    create an axis object (which it should return) and infer the figure from
    that axis. This will be done in context of the other settings, most
    notably the `fontsize` and `usetex` configurations, which should be active
    during axis creation.
    """
    if close and show:
        raise ValueError("Cannot close and show figures.")
    if fixed_layout:
        if fixed_layout is True:
            fixed_layout = {}
        if isinstance(fixed_layout, (list, tuple)):
            left, bottom, right, top = fixed_layout
            fixed_layout = dict(left=left, right=right, bottom=bottom, top=top)
        fixed_layout = insert_missing(
            fixed_layout, left=0.2, right=0.95, top=0.85, bottom=0.2
        )
        fixed_layout.update((subplots_kw or {}).get("gridspec_kw", {}))
        save_opts = insert_missing(save_opts or {}, bbox_inches=None)
        subplots_kw = insert_missing(
            subplots_kw or {}, gridspec_kw=fixed_layout,
        )
    rc_opts = dict()
    if usetex is not None:
        rc_opts['text.usetex'] = usetex
        if usetex:
            rc_opts['font.family'] = 'DejaVu Serif', 'serif'
            rc_opts['font.serif'] = ['Computer Modern']
    if fontsize is not None:
        rc_opts['font.size'] = fontsize
    with matplotlib_rc(rc_opts):
        if ax is None:
            if subplot_kw is None:
                subplot_kw = dict()
            if projection:
                subplot_kw.setdefault('projection', projection)
            figopts = dict(figsize=figsize)
            if dpi:
                figopts['dpi'] = dpi
            if nrows == 1 == ncols and subplots_kw is None:
                fig = plt.figure(**figopts)
                ax = fig.add_subplot(111, **(subplot_kw or dict()))
            else:
                fig, ax = plt.subplots(
                    nrows, ncols, **(subplots_kw or dict()), **figopts,
                    subplot_kw=subplot_kw,
                )
        else:
            if callable(ax):
                ax = ax()
            fig = np.asarray([ax]).flat[0].figure
        ax_result = ax
        axes = list(np.asarray([ax_result]).flat)
        ax = None
        if zero_line:
            for ax in axes:
                add_zero_line(ax, zero_line)
        yield ax_result
        for ax in axes:
            if xlog:
                ax.set_xscale('log')
            if ylog:
                ax.set_yscale('log')
            xmin, xmax = xlim
            if xmin is not None:
                ax.set_xlim(left=xmin)
            if xmax is not None:
                ax.set_xlim(right=xmax)
            ymin, ymax = ylim
            if ymin is not None:
                ax.set_ylim(bottom=ymin)
            if ymax is not None:
                ax.set_ylim(top=ymax)
            if pad:
                ax.set_xlim(auto=True)
                ax.set_xlim(*_interpret_pad(pad, *xlim, *ax.get_xlim()))
            if ypad:
                ax.relim()
                ax.autoscale(axis='y')
                ax.set_ylim(auto=True)
                ax.set_ylim(*_interpret_pad(ypad, *ylim, *ax.get_ylim()))
        if cfg_callback:
            cfg_callback(ax_result)
        for ax in axes:
            if xtick_spacing is not None:
                if xlog:
                    ax.xaxis.set_major_locator(
                        plticker.LogLocator(base=10, numticks=xtick_spacing)
                    )
                else:
                    ax.xaxis.set_major_locator(plticker.MultipleLocator(xtick_spacing))
            elif pi_xticks:
                pi_ticks(ax, **(dict() if pi_xticks is True else pi_xticks))
            if ytick_spacing is not None:
                if ylog:
                    ax.yaxis.set_major_locator(
                        plticker.LogLocator(base=10, numticks=ytick_spacing)
                    )
                else:
                    ax.yaxis.set_major_locator(plticker.MultipleLocator(ytick_spacing))
            elif pi_yticks:
                pi_ticks(ax, axis='y', **(dict() if pi_yticks is True else pi_yticks))
        if save is not None and save is not False:
            if "." not in op.basename(save):
                save += ".pdf"
            fname = op.expanduser(save)
            os.makedirs(op.normpath(op.dirname(fname)), exist_ok=True)
            fig.savefig(fname, **insert_missing(save_opts or dict(),
                                                bbox_inches='tight'))
        if show:
            plt.show()
        elif close:
            plt.close(fig)


@contextmanager
def wrap_simple_plot_ctx(opts=None, **kw):
    r"""Plotting context, extracting figure options.

    Use this to extract figure options (like `figsize`, `dpi`, `ax`, etc.) and
    create the axis object. All remaining parameters will be supplied to the
    context.

    The returned options will be configured to use the new axis and defer
    showing until leaving the context.

    @b Examples
    ```
        with wrap_simple_plot_ctx(dpi=120, grid=False, l='.-') as (ax, opts):
            plot_data([1, 2, 3], **opts)
            plot_data([4, 5, 6], **opts)
    ```

    @param opts
        Dictionary with options. Default is the empty dictionary.
    @param **kw
        Default options. These are overridden by `opts` for keys that exist in
        both.
    """
    sig = inspect.signature(simple_plot_ctx)
    ctx_opts = {k: v.default for k, v in sig.parameters.items()}
    remaining_opts = merge_dicts(kw, opts or {})
    for k in list(remaining_opts.keys()):
        if k in ctx_opts:
            ctx_opts[k] = remaining_opts.pop(k)
    if "xlim" in ctx_opts:
        xmin, xmax = ctx_opts["xlim"]
        if xmin is not None or xmax is not None:
            remaining_opts["xlim"] = ctx_opts["xlim"]
    with simple_plot_ctx(**ctx_opts) as ax:
        if not isinstance(ax, np.ndarray):
            remaining_opts['ax'] = ax
        remaining_opts['show'] = False
        if ctx_opts["fixed_layout"]:
            remaining_opts['tight_layout'] = False
        yield ax, remaining_opts


@contextmanager
def plot_ctx(figsize=(6, 2), projection=None, grid=True, xlog=False,
             ylog=False, xlim=(None, None), ylim=(None, None), pad=0, ypad=0,
             yscilimits=(-3, 3), xscilimits=None, xtick_spacing=None,
             pi_xticks=None, pi_yticks=None, zero_line=False,
             ytick_spacing=None, title=None, xlabel=None, ylabel=None,
             tight=True, tight_layout=True, usetex=None, fontsize=None,
             save=None, ax=None, show=True, close=False, cfg_callback=None,
             equal_lengths=False, dpi=None, save_opts=None, subplot_kw=None,
             subplots_kw=None, fixed_layout=None):
    r"""Context manager for preparing a figure and applying configuration.

    This is a convenience function with which it is possible to easily create
    new plots or reuse existing (not yet displayed) figures. It will
    initialize and yield an axis object suitable for plotting into using its
    functions. After this, the object is configured with the chosen settings.

    Finally, the figure is optionally saved to a file and/or shown using the
    current Matplotlib driver.

    Most of the options should be described in the (user-visible) function
    plot_1d().
    """
    def _cb(ax):
        if cfg_callback:
            cfg_callback(ax)
        if equal_lengths:
            axes = "xy" if equal_lengths is True else equal_lengths
            _equal_lengths(axes, ax)
    if fixed_layout:
        tight_layout = False
    with simple_plot_ctx(figsize=figsize, projection=projection,
                         usetex=usetex, fontsize=fontsize, save=save, ax=ax,
                         show=show, close=close, cfg_callback=_cb, dpi=dpi,
                         save_opts=save_opts, subplot_kw=subplot_kw,
                         subplots_kw=subplots_kw,
                         fixed_layout=fixed_layout, pad=pad, ypad=ypad,
                         zero_line=zero_line, xlog=xlog, ylog=ylog, xlim=xlim,
                         ylim=ylim, xtick_spacing=xtick_spacing,
                         ytick_spacing=ytick_spacing, pi_xticks=pi_xticks,
                         pi_yticks=pi_yticks) as ax:
        yield ax
        ax.grid(grid)
        if tight is not None:
            ax.autoscale(axis='x', tight=tight)
        if yscilimits is not None and ax.get_yscale() == 'linear':
            ax.ticklabel_format(axis='y', style='sci', scilimits=yscilimits)
        if xscilimits is not None and ax.get_xscale() == 'linear':
            ax.ticklabel_format(axis='x', style='sci', scilimits=xscilimits)
        if title is not None:
            opts = dict()
            if isinstance(title, (list, tuple)):
                title, opts = title
            ax.set_title(title, **opts)
        if ylabel is not None:
            opts = dict()
            if isinstance(ylabel, (list, tuple)):
                ylabel, opts = ylabel
            ax.set_ylabel(ylabel, **opts)
        if xlabel is not None:
            opts = dict()
            if isinstance(xlabel, (list, tuple)):
                xlabel, opts = xlabel
            ax.set_xlabel(xlabel, **opts)
        if tight_layout:
            opts = tight_layout if isinstance(tight_layout, dict) else dict()
            ax.figure.tight_layout(**opts)


@contextmanager
def plot_ctx_3d(zlim=(None, None), zscilimits=None, ztick_spacing=None,
                zlabel=None, figsize=(6, 5), azim=-60, elev=32, dist=None,
                cfg_callback=None, equal_lengths=False, **kw):
    r"""As plot_ctx(), but for 3-dimensional plots of \eg surfaces."""
    def _cb(ax):
        if cfg_callback:
            cfg_callback(ax)
        if equal_lengths:
            axes = "xyz" if equal_lengths is True else equal_lengths
            _equal_lengths(axes, ax)
    with plot_ctx(figsize=figsize, projection='3d', cfg_callback=_cb, **kw) as ax:
        ax.azim = azim
        ax.elev = elev
        if dist is not None:
            ax.dist = dist
        yield ax
        if zscilimits is not None:
            ax.ticklabel_format(axis='z', style='sci', scilimits=zscilimits)
        zmin, zmax = zlim
        if zmin is not None:
            ax.set_zlim(bottom=zmin)
        if zmax is not None:
            ax.set_zlim(top=zmax)
        if ztick_spacing is not None:
            ax.zaxis.set_major_locator(plticker.MultipleLocator(ztick_spacing))
        if zlabel is not None:
            opts = dict()
            if isinstance(zlabel, (list, tuple)):
                zlabel, opts = zlabel
            ax.set_zlabel(zlabel, **opts)


def _interpret_pad(pad, lower, upper, cur_lower, cur_upper):
    if lower is None:
        lower = cur_lower
    if upper is None:
        upper = cur_upper
    if not isinstance(pad, (list, tuple)):
        pad = (pad, pad)
    w = upper - lower
    pad = [
        w*float(p[:-1])/100 if isinstance(p, str) and p.endswith('%') else p
        for p in pad
    ]
    return lower - pad[0], upper + pad[1]


def pi_ticks(ax, axis='x', major=2, minor=1):
    r"""Show ticks in multiples of pi.

    @param ax
        The axis object to modify.
    @param axis
        String representing the axis to change. Default: ``"x"``
    @param major
        How many sub-intervals of `[0,pi]` to create. Default is `2`, i.e. a
        label will appear every ``pi/2``.
    @param minor
        Steps between the major steps. Set to `1` for no steps and `2` for one
        additional marker. Default is `1`.
    """
    a = getattr(ax, "%saxis" % axis)
    pi = np.pi
    denom = major
    a.set_major_locator(plticker.MultipleLocator(pi / major))
    if minor > 1:
        a.set_minor_locator(plticker.MultipleLocator(pi / (major * minor)))
    def _gcd(a, b):
        return _gcd(b, a % b) if b else a
    def _formatter(x, pos):
        num = int(round(denom*x/pi))
        if num == 0:
            return "$0$"
        common = _gcd(num, denom)
        num, den = int(num/common), int(denom/common)
        den = "" if den == 1 else "/%s" % den
        if num == 1:
            num = ""
        elif num == -1:
            num = "-"
        return r"$%s\pi%s$" % (num, den)
    a.set_major_formatter(plt.FuncFormatter(_formatter))
