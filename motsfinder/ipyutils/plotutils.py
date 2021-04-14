r"""@package motsfinder.ipyutils.plotutils

Utilities used by the plotting functions.
"""

import numpy as np
import matplotlib as mpl
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    mpl.use('agg', warn=False, force=True)  # switch to a more basic backend
    import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..utils import insert_missing


__all__ = [
    "add_zero_line",
    "add_zero_vline",
]


def _crop(image, left, bottom, right, top):
    h, w, _ = image.shape
    return image[
        top:h-bottom,
        left:w-right
    ]


def add_zero_line(ax, opts=None):
    r"""Add a line at y=0 to the given axis object.

    If the `opts` argument is a `dict`, it can be used to configure the
    `ax.axhline` call. Otherwise it is ignored.
    """
    if not isinstance(opts, dict):
        opts = dict()
    ax.axhline(
        **insert_missing(
            opts, linewidth=0.75, linestyle='-', color='k', alpha=0.33
        )
    )


def add_zero_vline(ax, opts=None):
    r"""Add a line at x=0 to the given axis object.

    If the `opts` argument is a `dict`, it can be used to configure the
    `ax.axhline` call. Otherwise it is ignored.
    """
    if not isinstance(opts, dict):
        opts = dict()
    ax.axvline(
        **insert_missing(
            opts, linewidth=0.75, linestyle='-', color='k', alpha=0.33
        )
    )


def add_two_colorbars(ax, values1, cmap1, values2, cmap2, fraction1=(0, 1),
                      fraction2=(0, 1), label1="", label2="", gap=5,
                      width="5%", pad=0.05):
    r"""Convenience function to add *two* colorbars to an axis object.

    @b Examples
    ```
        ax = plot_1d(lambda x: x**2, show=False, figsize=(3, 4))
        add_two_colorbars(
            ax,
            values1=[1*0.01, 7*0.01], cmap1=cm.coolwarm, fraction1=(1, 0.5),
            values2=[-1*0.01, -7*0.01], cmap2=cm.coolwarm, fraction2=(0.5, 0),
            label1=dict(label="positive CE values", labelpad=15),
            label2=dict(label="negative CE values", labelpad=6),
            gap=8,
        )
    ```
    """
    divider = make_axes_locatable(ax)
    iax = divider.append_axes("right", size="0%", pad=0.0)
    im1 = iax.imshow(
        np.array([values1]),
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "map", cmap1(np.linspace(*fraction1, 100))
        )
    )
    im2 = iax.imshow(
        np.array([values2]),
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "map", cmap2(np.linspace(*fraction2, 100))
        )
    )
    iax.set_visible(False)
    cax = inset_axes(
        ax, width=width, height="%.2f%%" % (50-gap/2.), loc='upper left',
        bbox_to_anchor=(1+pad, 0., 1, 1), bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = plt.colorbar(im1, cax=cax)
    if label1:
        cbar.set_label(**label1 if isinstance(label1, dict) else label1)
    cax = inset_axes(
        ax, width=width, height="%.2f%%" % (50-gap/2.), loc='lower left',
        bbox_to_anchor=(1+pad, 0., 1, 1), bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = plt.colorbar(im2, cax=cax)
    if label2:
        cbar.set_label(**label2 if isinstance(label2, dict) else label2)


def _configure_legend(ax, legendargs=None, labelspacing=0, **defaults):
    defaults['labelspacing'] = labelspacing
    defaults.update(legendargs or {})
    if defaults.get('loc', None) in ('outside', 'outside left'):
        defaults['loc'] = 'upper left'
        defaults = insert_missing(
            defaults, ncol=1, bbox_to_anchor=(1.05, 1.0), fancybox=False,
            borderaxespad=0,
        )
    if defaults.get('loc', None) == 'above':
        defaults['loc'] = 'lower center'
        defaults = insert_missing(
            defaults, ncol=1, bbox_to_anchor=(0.5, 1.0), frameon=False, #fancybox=False,
            borderaxespad=0,
        )
    ax.legend(**defaults)


def _extract_data_in_xlim(xs, ys, xlim):
    r"""Remove data outside the given x-limits.

    One data point to the left and right is kept (if possible) in order to
    have the line extend to plot boundary.
    """
    if xlim is None:
        return xs, ys
    xmin, xmax = xlim
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if xmin is None:
        start = 0
    else:
        start = max(0, xs.searchsorted(xmin)-1)
    if xmax is None:
        end = len(xs)
    else:
        end = xs.searchsorted(xmax) + 1
    return xs[start:end], ys[start:end]
