r"""@package motsfinder.ipyutils.plotting

Helper functions for more convenient plotting.
"""


from __future__ import print_function
from itertools import zip_longest
import os
import os.path as op
import copy

import numpy as np
from mpmath import mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .plotctx import plot_ctx
from ..utils import isiterable, insert_missing, merge_dicts, update_dict, lmap
from ..numutils import clip


__all__ = [
    "plot_1d",
    "plot_data",
    "plot_curve",
    "plot_polar",
    "plot_mat",
    "plot_multi",
    "video_from_folder",
    "video_from_images",
    "plot_image",
    "add_two_colorbars",
]


def plot_1d(f, f2=None, points=500, l=('-k', '-g'), color=None, lw=1.5,
            label=None, legendargs=None, rel_range=(0, 1), domain=None,
            value_pad=0, difference=False, offset=0, absolute=False,
            mark_points=None, zero_line=False, transform_x=None, xlog=False,
            plot_kw=None, show=True, close=False, **kw):
    r"""Plot one or two 1D functions and optionally save the plot to disk.

    By default, this function will simply plot the given callable `f` (which
    is required to return a scalar numeric value) in black and, if specified,
    the `f2` function in green.

    If no domain (x-range) is specified via `domain`, it is taken from the
    `domain` attribute of `f`, if that exists, or simply set to `[0,1]`.

    @param f (callable or object with an evaluator method)
        Function to be plotted. If `f` is not callable, its `evaluator` method
        is called without arguments to create a callable.
    @param f2 (callable, optional)
        Second function to be plotted or ``None`` to plot only one function.
        May also be a non-callable with an `evaluator` method.
    @param points (int or iterable, optional)
        Number of (equidistant) points for the plot. Default is `500`.
        Can also be an iterable. In this case, the function is sampled at
        exactly the points in this iterable and the parameters `domain` and
        `rel_range` as well as any `domain` attribute of `f` have no effect.
    @param l (string or 2-tuple, optional)
        Linestyles for the two plotted functions. Default is `('-k','-g')`,
        which will plot the first function with a black line and the second
        with a green line. If just a string, the linestyle is used for both
        functions.
    @param color
        Optional linecolor. Colors set via the `l` or `plot_kw` arguments take
        precedence over any color specified here.
    @param lw (float)
        Linewidth to draw the data with. Default is `1.5`.
    @param figsize (2-tuple, optional)
        Size of the plot. Default is `(6,2)`.
    @param rel_range (2-tuple, optional)
        Which area of the full domain to actually plot. Default is `[0,1]`,
        which will plot the complete domain. A value of e.g. `[.5,1]` will
        plot only the second half of the domain.
    @param domain (2-tuple, optional)
        Domain (x-range) to plot. If not specified (or ``None``), the domain
        is taken from the `domain` attribute of `f` if that exists or `[0,1]`
        otherwise.
    @param value_pad (float)
        Padding on the boundaries of `domain`. If nonzero, the domain is moved
        inward on both sides by this amount.
    @param difference (boolean, optional)
        If `True`, instead of plotting `f` and `f2`, plot their
        difference. Default is `False`.
    @param offset (float, optional)
        Offset the function and the range by this value (default `0`). This
        will only change the values on the axis. It is useful for cases where
        logarithmic scaling of the x-axis is desired but the function's domain
        starts at some other finite value.
    @param absolute (boolean, optional)
        Whether to plot the absolute value of the function results. Default is
        `False`.
    @param transform_x
        Optional callable to transform the x-axis values. Useful in case the
        inverse transform is not easily accessible. The function will be
        called once for each x-value and should return the transformed value.
    @param xlog (boolean, optional)
        Whether to plot the x-axis logarithmically.
    @param ylog (boolean, optional)
        Whether to plot the y-axis logarithmically.
    @param xlim (2-tuple, optional)
        Custom limit for the x-axis range.
    @param ylim (2-tuple, optional)
        Custom limit for the shown y-axis range. If any (or both) of the
        values are ``None`` (the default), the range is chosen automatically
        by the called plotting function to show all function values.
    @param pad (float or 2-tuple, optional)
        Usually, plots are cut off at the boundaries of the data. This
        parameter allows adding a bit of extra space.
    @param mark_points (list/tuple or (list/tuple, linestyle), optional)
        Mark these points on the plot of the function. The default linestyle
        is ``'o'``, which sets large dots at the points. The color is chosen
        to be the same as the plotted line. To get e.g. smaller black dots,
        use ``'.k'``. If `True` is used instead of actual values, uses the
        points at which the function is sampled.
        Also allows an argument ``(pts, linestyle, opts)``, where `opts` is a
        dict of options passed to `ax.plot`.
    @param zero_line (boolean, optional)
        Whether to draw a line for ``y == 0``. If a string, it is used as the
        linestyle. Default is `False`.
    @param yscilimits (tuple, optional)
        `yscilimits` option passed to `ax.ticklabel_format` to activate
        scientific notation for the `y`-axis labels. E.g., `(-2, 2)` will use
        scientific notation if the axis exceeds values ``>= 100`` stays below
        ``0.01``. Default is `(-3, 3)`.
    @param xscilimits (tuple, optional)
        Same as `yscilimits` but for the `x`-axis. Default is `None`.
    @param xtick_spacing (float, optional)
        Sets the distance between labeled ticks on the x-axis.
    @param title (string, optional)
        Title to be rendered above the plot.
    @param xlabel (string or (string, dict), optional)
        Label for the x-axis. The optional `dict` will be passed to the
        `ax.set_xlabel` call.
    @param ylabel (string or (string, dict), optional)
        Label for the y-axis. Optional `dict` used as for `xlabel`.
    @param tight (boolean or None, optional)
        If true (the default), the range of the x-axis will be exactly as
        needed to show all data. Otherwise, the range might be extended a bit.
        For example, with `tight=False`, the range `[0,pi]` will probably
        result in the x-axis showing `[0,3.5]`. Setting this to `None` will
        not change the current default (useful if an `ax` object is supplied).
    @param tight_layout (boolean or dict, optional)
        If `True`, calls `tight_layout()` on the figure object without
        arguments. This leads to the elements on the plot to be contained in a
        smaller area, potentially fixing cut-out elements in stored image
        files. If this does not fix the problems, the argument may also be a
        `dict` containing the options to pass to `tight_layout`, e.g.
        ``tight_layout=dict(pad=1.2)``.
    @param label (string, optional)
        Optional label string. If any of the plots added to the `ax` have a
        label, a legend is displayed showing the configured labels.
    @param legendargs (dict, optional)
        Arguments for the `ax.legend()` call for fine-tuning e.g. placement
        and style. By default, sets `labelspacing=0`, which may be overridden
        by this argument. Has no effect when no `label` is given.
    @param usetex (boolean, optional)
        Whether to use a LaTeX to render labels or matplotlib's built-in
        `mathtext`. Default is to use the current setting.
    @param fontsize (int, optional)
        Global size for all text elements.
    @param save (string, optional)
        Optional filename for saving the created plot to disk. If this
        contains no dot, ``'.pdf'`` is appended. The file is saved in the
        current working directory, unless a path is specified too.
    @param ax (Axis object, optional)
        Axis object to add the plot to. If none is given, creates a new one.
        This may be useful when you created multiple Axis objects using e.g.
        `ax1, ax2 = matplotlib.pyplot.subplots(ncols=2)[1]`.
    @param show (boolean, optional)
        Whether to conclude by showing the whole plot (the default). Any Axis
        object will be invalid after this call, which is why nothing is
        returned in this case. If you set `show=False` (and `close` is false
        too), the `ax` object will be returned for further processing (like
        adding more plots).
    @param close (boolean, optional)
        Whether to close the figure at the end. Default is `False`. This can
        only be used if `show=False`, since showing all open figures closes
        them anyway. Also, no axis object will be returned if `close=True`.
        This can be useful to save plots to files without displaying them.
    @param cfg_callback (callable, optional)
        Optional callback for more custom configuration tasks. It is called as
        ``cfg_callback(ax)``, where `ax` is the current axis object. Use
        `ax.figure` to obtain the figure object. It is called as the last step
        before saving/showing the plot.
    @param dpi
        Optional `dpi` value for construction the figure. Higher values
        magnify the plot (including e.g. labels). Default is
        ``rcParams["figure.dpi"]`` (usually `75` or `100`, depending on the
        matplotlib version).
    @param plot_kw (dict or (dict, dict), optional)
        Additional options passed to the `ax.plot` call. If two dicts are
        given, the second one is used when plotting `f2`.
    @param **kw
        Additional keyword arguments are passed to the plot_ctx() function.
    """
    if legendargs is None:
        legendargs = dict()
    if plot_kw is None:
        plot_kw = dict()
    if not isinstance(plot_kw, (list, tuple)):
        plot_kw = (plot_kw, plot_kw)
    if not isinstance(lw, (list, tuple)):
        lw = (lw, lw)
    kw1, kw2 = plot_kw
    kw1 = update_dict(kw1, lw=lw[0])
    kw2 = update_dict(kw2, lw=lw[1])
    if color and l and not l[-1].isalpha():
        kw1 = insert_missing(kw1, color=color)
        kw2 = insert_missing(kw2, color=color)
    if difference and f2 is None:
        raise ValueError("Cannot plot difference without two functions.")
    if not callable(f):
        f = f.evaluator()
    if f2 is not None and not callable(f2):
        f2 = f2.evaluator()
    if isinstance(l, (list, tuple)):
        l1, l2 = l
    else:
        l1 = l2 = l
    if l1 is None:
        l1 = '-'
    if l2 is None:
        l2 = '-'
    if isinstance(label, (list, tuple)):
        label1, label2 = label
    else:
        label1, label2 = label, None
    if 'abs_range' in kw:
        domain = kw.pop('abs_range')
    if domain is None:
        domain = getattr(f, 'domain', (0, 1))
    aa, ab = domain
    ra, rb = rel_range
    a = float(aa + ra * (ab-aa) + offset) + value_pad
    b = float(aa + rb * (ab-aa) + offset) - value_pad
    if isiterable(points):
        space = points
    else:
        if xlog:
            space = np.logspace(np.log10(a), np.log10(b), points)
        else:
            space = np.linspace(a, b, points)
    try:
        f.prepare_evaluation_at(space)
    except AttributeError: pass
    try:
        if f2 is not None:
            f2.prepare_evaluation_at(space)
    except AttributeError: pass
    xs = space
    if transform_x:
        if not callable(transform_x):
            transform_x = transform_x.evaluator()
        xs = [transform_x(x) for x in space]
    if difference:
        other_fun = f2
        f2 = None
        fun = lambda x: f(x) - other_fun(x)
    else:
        fun = f
    with plot_ctx(show=show, close=close, xlog=xlog, **kw) as ax:
        if zero_line:
            ax.plot(xs, [0] * len(space), '-k' if zero_line is True else zero_line)
        f1 = (lambda x: abs(fun(x))) if absolute else fun
        line, = ax.plot(xs, [f1(x-offset) for x in space], l1, label=label1, **kw1)
        if f2 is not None:
            _f2 = f2
            f2 = (lambda x: abs(_f2(x))) if absolute else _f2
            ax.plot(xs, [f2(x-offset) for x in space], l2, label=label2, **kw2)
        if mark_points is not None:
            marker_opts = dict()
            if mark_points is True:
                mark_points = xs
            if mark_points[0] is True:
                _, ls = mark_points
                pts = xs
            elif isiterable(mark_points[0]):
                try:
                    pts, ls = mark_points
                except ValueError:
                    pts, ls, marker_opts = mark_points
            else:
                pts, ls = mark_points, 'o'
            if 'color' not in marker_opts and not any(c in ls for c in 'bgrcmykw'):
                marker_opts['color'] = line.get_color()
            pts = [p for p in pts if xs[0] <= p <= xs[-1]]
            ax.plot(pts, [f1(x-offset) for x in pts], ls, **marker_opts)
        if label is not None:
            ax.legend(**insert_missing(legendargs, labelspacing=0))
    if not show and not close:
        return ax


def plot_data(x, y=None, l='.-k', color=None, lw=1.5, absolute=False,
              zero_line=False, label=None, legendargs=None, plot_kw=None,
              copy_x=False, copy_y=False, show=True, close=False, **kw):
    r"""Plot the data given as x- and y-values and optionally save the plot to disk.

    This is a convenience function to quickly plot the given pairs of x- and
    y-values.

    Parameters not documented here have the same meaning as in plot_1d().

    The `plot_kw` parameter is added to the `Axis.plot` call.

    @param x (array_like)
        List of x-positions. Each element in this list corresponds to the
        element at the same index in the `y` list and defines the abscissa of
        the data point. Alternatively, `y` may be omitted, in which case `x`
        is interpreted as containing the values and the actual `x` values will
        be set to ``range(len(y))``.
    @param y (array_like)
        List of y-positions, i.e. values of the corresponding element in `x`.
        This defines the ordinate of the data points.
    @param l (string, optional)
        Linestyle used for plotting the data. The default ``'.-k'`` will show
        black dots at the data points, connected with black lines.
    @param lw (float)
        Linewidth to draw the data with. Default is `1.5`.
    @param copy_x (boolean, optional)
        Repeat the plot with all x-values replaced by their negatives.
    @param copy_y (boolean, optional)
        Repeat the plot with all y-values replaced by their negatives. If both
        `copy_x` and `copy_y` are set to `True`, a third plot is added where
        both sets of values are sign-changed.
    """
    if legendargs is None:
        legendargs = dict()
    if plot_kw is None:
        plot_kw = dict()
    if l is None:
        l = '-'
    plot_kw = update_dict(plot_kw, lw=lw)
    if y is None:
        y = x
        x = range(len(y))
    if color and l and not l[-1].isalpha():
        plot_kw = insert_missing(plot_kw, color=color)
    with plot_ctx(show=show, close=close, **kw) as ax:
        if absolute:
            y = np.array(lmap(abs, y))
        if zero_line:
            ax.plot(x, [0] * len(x), '-k' if zero_line is True else zero_line)
        line, = ax.plot(x, y, l, label=label, **plot_kw)
        plot_kw['color'] = line.get_color()
        if copy_x:
            ax.plot(-x, y, l, **plot_kw)
        if copy_y:
            ax.plot(x, -y, l, **plot_kw)
        if copy_x and copy_y:
            ax.plot(-x, -y, l, **plot_kw)
        if label is not None:
            ax.legend(**insert_missing(legendargs, labelspacing=0))
    if not show and not close:
        return ax


def plot_curve(curve, domain=None, points=500, l='-k', label=None,
               mirror_x=False, mirror_y=False, offset=(0, 0), figsize=(4, 4),
               **kw):
    r"""Plot a given parametric curve.

    The `curve` should return `x,y` pairs of values when evaluated at a given
    parameter value.

    Parameters not described here are described in plot_data() or plot_1d().

    @param curve (callable or NumericExpression)
        Parameterized curve to plot.
    @param domain (2-tuple/list, optional)
        Parameter domain as `(a, b)` of the curve. If not given, assumes the
        curve object has a `domain` attribute.
    @param points (int or iterable, optional)
        Number of (equally spaced w.r.t. the parameter) samples to take along
        the curve. If an iterable, uses the items as values to sample at. In
        that case, the `domain` argument is ignored.
        Default is `500`.
    @param mirror_x (boolean, optional)
        Change sign of all `x` values before applying the offset.
    @param mirror_y (boolean, optional)
        Change sign of all `y` values before applying the offset.
    @param offset (2-tuple/list, optional)
        x/y offset to apply to all values. Useful when the curve is
        parameterized w.r.t. an origin different from `(0, 0)`.
    """
    if not callable(curve):
        curve = curve.evaluator()
    if isiterable(points):
        t = np.array(points)
    else:
        if domain is None:
            domain = curve.domain
        a, b = domain
        t = np.linspace(a, b, points)
    data = np.array(lmap(curve, t))
    xs, ys = data.T
    if mirror_x:
        xs = -xs
    if mirror_y:
        ys = -ys
    xs += offset[0]
    ys += offset[1]
    return plot_data(xs, ys, l=l, label=label, figsize=figsize, **kw)


def plot_polar(f, origin=(0, 0), points=500, l='-k', pi_symmetry=False,
               mirror_x=False, mirror_y=False, figsize=(4, 4),
               equal_lengths=True, **kw):
    r"""Plot a function parameterized by an angle.

    Contrary to the usual convention, the angle parameter is taken to be the
    angle w.r.t. the `y`-axis. Options are similar to plot_curve().

    Parameters not described here are described in plot_data() or plot_1d().

    @param origin
        Move the origin of the coordinate system to the desired location.
        Default is `(0, 0)`.
    @param points
        Number of equally spaced samples to take along the curve. May also be
        an iterable of the precise points at which to sample. Default is
        `500`.
    @param l
        Linestyle of the curve. Default is ``'-k'``, i.e. a black line.
    @param pi_symmetry
        Whether the function is symmetric w.r.t. `theta=pi`. If `True`, the
        function will be evaluated only in the interval `[0,pi]` and the plot
        will be repeated mirrored across the y-axis. Otherwise, the function
        is evaluated on the interval `[0,2pi]`.
    @param mirror_x
        Mirror the plot across the y-axis (i.e. in x-direction).
    @param mirror_y
        Mirror the plot across the x-axis (i.e. in y-direction).
    """
    if not callable(f):
        f = f.evaluator()
    if isiterable(points):
        ta = np.array(points)
    else:
        if pi_symmetry:
            ta = np.linspace(0, np.pi, points)
        else:
            ta = np.linspace(0, 2*np.pi, points)
    r = np.array(lmap(f, ta))
    if pi_symmetry:
        ta = np.concatenate((ta[:-1], ta+np.pi))
        r = np.concatenate((r[:-1], r[::-1]))
    xs = r * np.sin(ta)
    ys = r * np.cos(ta)
    if mirror_x:
        xs = -xs
    if mirror_y:
        ys = -ys
    return plot_data(xs + origin[0], ys + origin[1], l=l, figsize=figsize,
                     equal_lengths=equal_lengths, **kw)


def plot_mat(mat, figsize=(5, 4), colorbar=True, cmap=cm.jet, vmin=None,
             vmax=None, absolute=False, log10=False, normalize=False,
             offset=0, bad_color=None, is_img=False, plot_kw=None, show=True,
             close=False, **kw):
    r"""Visualize a matrix using a colored rectangle.

    Parameters not described here are described in plot_data() or plot_1d().

    @param colorbar
        Whether to show a bar next to the plot indicating the numerical values
        the colors represent. Default is `True`.
    @param cmap
        Color map to use for coloring. Default is `cm.jet`, which is a
        "heat-map" like color map.
    @param vmin
        Minimum value represented by the color map. By default, the smallest
        occurring value is used.
    @param vmax
        Maximum value represented by the color map. By default, the largest
        occurring value is used.
    @param absolute
        Whether to plot the absolute values.
    @param log10
        Whether to plot the base-10 logarithm of the values. Best combined
        with `absolute`.
    @param normalize
        If `True`, each value is divided such that the maximum magnitude is 1.
    @param offset
        Value to add to each element of the matrix. Might be useful when
        plotting logarithms and there are exact zeros. For example, adding a
        value slightly below the roundoff plateau of spectral coefficients can
        make the plot more readable.
    @param bad_color
        Color to use for NaNs in the data.
    @param is_img
        If `True`, use `imshow()` instead of `matshow()` for plotting. This
        affects the placement of axes labels. Default is `False`.
    @param plot_kw
        Dictionary of arguments to pass to the `ax.matshow()` function.
    """
    if plot_kw is None:
        plot_kw = dict()
    if isinstance(mat, mp.matrix):
        mat = np.array(mat.tolist(), dtype=float)
    if not isinstance(mat, (np.ndarray, np.matrix)):
        mat = np.array(mat)
    if normalize:
        mx = np.absolute(mat).max()
        if mx > 0:
            mat = mat / mx
    if absolute:
        mat = np.absolute(mat)
    if offset != 0:
        mat = mat + offset
    if log10:
        mat = np.ma.log10(mat)
    if bad_color:
        cmap = copy.copy(cmap)
        if not isinstance(bad_color, (list, tuple)):
            bad_color = [bad_color]
        cmap.set_bad(*bad_color)
    with plot_ctx(figsize=figsize, show=show, close=close, **kw) as ax:
        opts = dict(cmap=cmap, vmin=vmin, vmax=vmax, **plot_kw)
        if is_img:
            img = ax.imshow(mat, **opts)
        else:
            img = ax.matshow(mat, **opts)
        if colorbar:
            opts = colorbar if isinstance(colorbar, dict) else dict()
            ax.figure.colorbar(img, **insert_missing(opts, ax=ax))
    if not show and not close:
        return ax


def plot_multi(*args, tight_layout=False, **kwargs):
    r"""Plot multiple 1d or 2d functions in a grid.

    All the positional arguments are interpreted as functions to plot. These
    arguments may optionally be tuples/lists with two elements. The first must
    be the function to plot, the second can either be a string (title of the
    plot) or a `dict` with options passed to the individual plot commands.

    These individual plot options may optionally contain a special key
    `plotter`, which will be used as the plotting function instead of
    plot_1d() (for 1d plots) or plot_2d() (for 2d plots).

    If 1D plots should be done (which is the default), the plot_1d() function
    is called for each plot, whereas for 3D plots (if `surfaces==True`),
    plot_2d() is used.

    All parameters not listed below are merged with the individual options of
    each plot (if any) and passed to the plot command.

    @param surfaces (boolean, optional)
        If `True`, all individual plots are surface plots. Default is `False`.
    @param cols (int, optional)
        Maximum number of plots in one row. If more plots are given, they are
        put into multiple rows. Default is `4`.
    @param w (float, optional)
        Width of each row. Default is `15`.
    @param h (float, optional)
        Relative height of each individual plot. This is scaled such that the
        aspect ratio of ``12 : h`` is kept.
        Default is `5` for 1d and `8` for 2d plots.
    @param maxwidth (float, optional)
        Maximum width of an individual sub-plot. Default is `5.9`. This is
        useful to prevent e.g. single plots to become very large.
    @param figsize (2-tuple or list, optional)
        Overrides `w` and `h` with a custom size.

    @b Example

    \code
    plot_multi(
        lambda x: x**2,
        (lambda x: x**2, r'Square of $x$'),
        (lambda x: x**2, dict(title=r'$x^2$', l='--g', domain=(-1,1))),
        l='-b', points=100, xlabel=r'$x$', ylabel=r'$y$'
    )
    \endcode
    """
    n = len(args)
    kwargs['tight_layout'] = tight_layout
    surfaces = kwargs.get('surfaces', False)
    cols = kwargs.get('cols', 4)
    width = kwargs.get('w', 15)
    height = kwargs.get('h', 8 if surfaces else 5)
    maxw = kwargs.get('maxwidth', 5.9)
    if maxw is not None:
        width = min(width, maxw*min(cols, n))
    figsize = kwargs.get('figsize', (width, width/12.0 * float(height)/n))
    dpi = kwargs.get('dpi', 70)
    if cols < n:
        for i in range(0, n, cols):
            sub_list = args[i:i+cols]
            sub_n = len(sub_list)
            if sub_n < cols: # last one not filled
                plot_multi(*sub_list,
                           **merge_dicts(kwargs,
                                         dict(w=width*(float(sub_n)/cols))))
            else:
                plot_multi(*sub_list, **kwargs)
        return
    kwargs.pop('surfaces', None)
    kwargs.pop('cols', None)
    kwargs.pop('w', None)
    kwargs.pop('h', None)
    kwargs.pop('maxwidth', None)
    kwargs.pop('figsize', None)
    if surfaces:
        axes = plt.subplots(
            ncols=n, figsize=figsize, dpi=dpi, squeeze=False,
            subplot_kw={'projection':'3d'}
        )[1].flatten()
    else:
        axes = plt.subplots(
            ncols=n, figsize=figsize, dpi=dpi, squeeze=False
        )[1].flatten()
    for i, ax in enumerate(axes):
        try:
            f, opts = args[i]
        except (TypeError, ValueError):
            f = args[i]
            opts = dict()
        if not isinstance(opts, dict):
            opts = dict(title=opts)
        opts = insert_missing(opts, **kwargs)
        if surfaces:
            from .plotting3d import plot_2d
            plotter = plot_2d
        else:
            plotter = plot_1d
        plotter = opts.pop('plotter', plotter)
        plotter(f, ax=ax, show=False, **opts)
    plt.show()


def video_from_folder(*folders, ext='png', filter_cb=None, **kw):
    r"""Display a movie from a sequence of images in one or more folders.

    This takes all images in the given folder(s) and renders them into a video
    to view inside the Jupyter notebook and/or save into a file.

    @param *folders
        Positional arguments are the various folders to search for images.
        Each element may either be
        \code
            folder_name
            (folder_name, reverse)
        \endcode
        where `folder_name` is a string and `reverse` and optional boolean
        indicating whether to reverse the order of images in this particular
        folder.
    @param ext
        Extension of the images. All files in the given folders with this
        extension are loaded.
    @param **kw
        Further keyword arguments are passed to video_from_images().
    """
    def isimg(folder, fname):
        fname = "%s/%s" % (folder, fname)
        return op.isfile(fname) and fname.endswith(".%s" % ext)
    image_files = []
    for folder in folders:
        if isinstance(folder, (list, tuple)):
            folder, reverse = folder
        else:
            reverse = False
        folder = op.normpath(folder)
        files = list("%s/%s" % (folder, f)
                     for f in os.listdir(folder) if isimg(folder, f))
        files.sort()
        if reverse:
            files.reverse()
        image_files += files
    if filter_cb:
        image_files = [img for i, img in enumerate(image_files)
                       if filter_cb(i, img)]
    if not image_files:
        raise ValueError("No images found.")
    return video_from_images(image_files, **kw)


def video_from_images(image_files, fps=5, callback=None, reverse=False,
                      dpi=72, interpolation=None, repeat_first=0,
                      repeat_last=0, save=None, repetitions=None, crop=None,
                      snap_first=False):
    r"""Display a movie from a sequence of images.

    This renders the given images into a video to view inside the Jupyter
    notebook and/or save into a file.

    @param image_files
        Images to render into a video.
    @param fps
        Frames/images per second to show. Default is `5`.
    @param callback
        Optional callback called for each frame with the signature
        `callback(i, ax, ctrl)`, where `i` runs from `0` through the frame
        numbers, `ax` is the axes object, and `ctrl` the image control created
        by `ax.imshow()`.
    @param reverse
        If `True`, play the images in reverse order.
    @param dpi
        Dots per inch of the stored image. When set correctly, the image will
        be shown in its native resolution (i.e. without resizing).
    @param interpolation
        Pixel interpolation in case the rendered size is not exactly the image
        size. Default is `None`.
    @param repeat_first
        Number of extra repetitions of the first image. This adds additional
        time at the start. Default is `0`.
    @param repeat_last
        Number of extra repetitions of the last image. This adds additional
        time at the end, so that the result can be inspected a little longer.
        Default is `0`.
    @param save
        Optional filename for storing the resulting video.
    @param repetitions
        How often to repeat individual images. Should be a sequence of
        integers corresponding to the individual images. Missing elements are
        taken to be 1.
    @param crop
        4-tuple with the amount of cropping (in pixel) in order (left, botton,
        right, top).
    @param snap_first
        If `True`, save a snapshot of the first frame.
    """
    from IPython.display import HTML
    from matplotlib.animation import FuncAnimation
    if not image_files:
        raise ValueError("No images specified.")
    if reverse:
        image_files = list(reversed(image_files))
    if repetitions is not None:
        repeated = []
        for img, rep in zip_longest(image_files, repetitions, fillvalue=1):
            if img == 1:
                break
            repeated.extend([img] * rep)
        image_files = repeated
    ax, image_ctrl = plot_image(image_files[0], dpi=dpi, show=False,
                                animated=True, interpolation=interpolation,
                                crop=crop)
    if snap_first and save:
        fname = op.expanduser(save)
        if fname.endswith(".mp4"):
            fname = "%s.png" % fname[:-4]
        os.makedirs(op.normpath(op.dirname(fname)), exist_ok=True)
        ax.figure.savefig(fname)
    def update(i):
        idx = clip(i-repeat_first, 0, len(image_files)-1)
        image = plt.imread(image_files[idx])
        if crop:
            image = _crop(image, *crop)
        image_ctrl.set_array(image)
        if callback:
            callback(i, ax, image_ctrl)
    a = FuncAnimation(ax.figure, update,
                      frames=len(image_files) + repeat_first + repeat_last,
                      interval=1000/fps)
    v = a.to_html5_video()
    if save is not None:
        if "." not in op.basename(save):
            save += ".mp4"
        fname = op.expanduser(save)
        os.makedirs(op.normpath(op.dirname(fname)), exist_ok=True)
        a.save(fname)
    plt.close(ax.figure)
    return HTML(v)


def plot_image(image_file, dpi=72, show=True, crop=None, **kw):
    r"""Plot a single image file into the notebook.

    @param image_file
        File containing the image.
    @param dpi
        Dots per inch of the stored image. When set correctly, the image will
        be shown in its native resolution (i.e. without resizing).
        Default is `72`.
    @param show (boolean, optional)
        Whether to conclude by showing the figure (default). If `False`,
        returns the axis and image objects.
    @param crop
        4-tuple with the amount of cropping (in pixel) in order (left, botton,
        right, top).
    @param **kw
        Extra arguments passed to `ax.imshow()`. May e.g. contain
        interpolation or animation arguments.

    @return `None` if `show==True` (default). Otherwise, returns the axis and
        image objects.
    """
    image = plt.imread(image_file)
    if crop:
        image = _crop(image, *crop)
    h, w, _ = image.shape
    fig = plt.figure(figsize=(float(w)/dpi, float(h)/dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    image_ctrl = ax.imshow(image, **kw)
    if not show:
        return ax, image_ctrl
    plt.show()


def _crop(image, left, bottom, right, top):
    h, w, _ = image.shape
    return image[
        top:h-bottom,
        left:w-right
    ]


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
            "map", cm.coolwarm(np.linspace(*fraction1, 100))
        )
    )
    im2 = iax.imshow(
        np.array([values2]),
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "map", cm.coolwarm(np.linspace(*fraction2, 100))
        )
    )
    iax.set_visible(False)
    cax = inset_axes(
        ax, width=width, height="%.2f%%" % (50-gap/2.), loc='upper left',
        bbox_to_anchor=(1+pad, 0., 1, 1), bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = plt.colorbar(im1, cax=cax);
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
