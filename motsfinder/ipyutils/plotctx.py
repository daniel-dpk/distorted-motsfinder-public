r"""@package motsfinder.ipyutils.plotctx

Contexts for convenient plot setup/management.
"""

from __future__ import print_function

from contextlib import contextmanager
import subprocess
import os
import os.path as op

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
# This import has side-effects we need.
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import

from ..utils import insert_missing


__all__ = [
    "matplotlib_rc",
    "plot_ctx",
    "plot_ctx_3d",
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
def plot_ctx(figsize=(6, 2), projection=None, grid=True, xlog=False,
             ylog=False, xlim=(None, None), ylim=(None, None), pad=0,
             yscilimits=(-3, 3), xscilimits=None, xtick_spacing=None,
             ytick_spacing=None, title=None, xlabel=None, ylabel=None,
             tight=True, tight_layout=True, usetex=None, fontsize=None,
             save=None, ax=None, show=True, close=False, cfg_callback=None,
             equal_lengths=False, save_opts=None):
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
    if close and show:
        raise ValueError("Cannot close and show figures.")
    rc_opts = dict()
    if usetex is not None:
        rc_opts['text.usetex'] = usetex
    if fontsize is not None:
        rc_opts['font.size'] = fontsize
    with matplotlib_rc(rc_opts):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            if projection is not None:
                ax = fig.add_subplot(111, projection=projection)
            else:
                ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        yield ax
        ax.grid(grid)
        if tight is not None:
            ax.autoscale(axis='x', tight=tight)
        if ylog: ax.set_yscale('log')
        if xlog: ax.set_xscale('log')
        if yscilimits is not None and ax.get_yscale() == 'linear':
            ax.ticklabel_format(axis='y', style='sci', scilimits=yscilimits)
        if xscilimits is not None and ax.get_xscale() == 'linear':
            ax.ticklabel_format(axis='x', style='sci', scilimits=xscilimits)
        xmin, xmax = xlim
        if xmin is not None: ax.set_xlim(left=xmin)
        if xmax is not None: ax.set_xlim(right=xmax)
        ymin, ymax = ylim
        if ymin is not None: ax.set_ylim(bottom=ymin)
        if ymax is not None: ax.set_ylim(top=ymax)
        if xtick_spacing is not None:
            if xlog:
                ax.xaxis.set_major_locator(plticker.LogLocator(xtick_spacing))
            else:
                ax.xaxis.set_major_locator(plticker.MultipleLocator(xtick_spacing))
        if ytick_spacing is not None:
            if ylog:
                ax.yaxis.set_major_locator(plticker.LogLocator(ytick_spacing))
            else:
                ax.yaxis.set_major_locator(plticker.MultipleLocator(ytick_spacing))
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
            fig.tight_layout(**opts)
        if pad:
            if not isinstance(pad, (list, tuple)):
                pad = (pad, pad)
            ax.set_xlim(auto=True)
            xmin, xmax = ax.get_xlim()
            ax.set_xlim(xmin-pad[0], xmax+pad[1])
        _cb(ax)
        if save is not None:
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
