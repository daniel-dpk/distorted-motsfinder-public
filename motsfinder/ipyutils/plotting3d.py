r"""@package motsfinder.ipyutils.plotting3d

Functions for plotting in 3D, \eg 2-dimensional surfaces via plot_2d().
"""

from matplotlib import cm

import numpy as np

from .plotctx import plot_ctx_3d
from .plotting import plot_data
from ..utils import isiterable, insert_missing


__all__ = [
    "prepare_grid",
    "plot_2d",
    "plot_scatter",
    "plot_tri",
    "plot_tri_cut",
]


def prepare_grid(fun, space_x, space_y):
    """Evaluate `fun` to build X, Y, Z arrays usable for plotting with `Axes3D.plot_surface`."""
    X, Y = np.meshgrid(space_x, space_y)
    zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    return X, Y, Z


def plot_scatter(points, s=20, c='b', plot_kw=None, show=True, close=False,
                 **kw):
    r"""Create a scatter plot of 3D points.

    Options are similar to the ones of plot_1d() with the below additions.

    @param points
        Iterable of 3D points to plot.
    @param s
        Size of the individual point markers. Default is `20`.
    @param c
        Color of the markers. Default is blue, i.e. ``'b'``.
    @param zlim
        Optional interval of the z-axis to show. Default is automatic.
    @param zscilimits
        As the `xscilimits` parameter but for the z-axis.
    @param ztick_spacing
        As `xtick_spacing` but for the z-axis.
    @param zlabel
        Label for the z-axis
    @param azim
        Azimuthal angle used for the 3D projection. Default is `-60`.
    @param elev
        Elevation angle used for the 3D projection. Default is `32`.
    @param dist
        Custom "distance" to the plotted area. Modifies perspective and can
        help in cases where important parts are cropped off the image.
    """
    if plot_kw is None:
        plot_kw = dict()
    xs, ys, zs = zip(*points)
    with plot_ctx_3d(show=show, close=close, **kw) as ax:
        ax.scatter(xs, ys, zs, s=s, c=c, **plot_kw)
    if not show and not close:
        return ax


def plot_2d(f, domain=None, rel_range=((0, 1), (0, 1)), points=30, lw=.4,
            c='w', ec='k', cmap=None, absolute=False, zoffset=0, zlog=False,
            plot_kw=None, show=True, close=False, **kw):
    r"""Plot a surface parameterized as `z = f(x, y)`.

    Parameters not described here are described in plot_scatter() or plot_1d().

    @param f
        Function to plot. Should take a 2-tuple/list/array and return a
        floating type. Alternatively, this may be a (NumPy) matrix which will
        be plotted as ``f(i, j) = mat[i, j]``, where `i` and `j` run over the
        full shape (i.e. ranges and `points` arguments are ignored).
    @param domain
        Domain to plot `f` on as ``((x1, x2), (y1, y2))``. If not given, the
        function's `domainX` and `domainY` attributes are queried. If these
        don't exist, the interval `(0,1)` is used.
    @param rel_range
        Similar as `rel_range` in plot_1d(), but containing the relative range
        to plot for both axes.
    @param points
        Number of samples to take in each direction. The function will be
        evaluated on a grid of `points*points` points. May also be a
        2-tuple/list with a resolution for each axis. Default is `30`.
    @param lw
        Line width for drawing the edges of the 3D mesh. Default is `0.4`.
    @param c
        Color of the mesh surfaces. Default is white, i.e. ``'w'``.
    @param ec
        Edge color. Default is black, i.e. ``'k'``.
    @param cmap
        Optional color map to indicate height of the surface.
    @param absolute
        If `True`, uses the absolute of the values returned by `f`.
    @param zoffset
        Value to add to the values returned by `f`.
    @param zlog
        Whether to plot the base-10 logarithm of the function values.
    """
    if isinstance(f, (np.ndarray, list, tuple)):
        mat = np.asarray(f)
        f = lambda xy: mat[xy[0], xy[1]]
        points = range(mat.shape[0]), range(mat.shape[1])
    if plot_kw is None:
        plot_kw = dict()
    if 'abs_range' in kw:
        domain = kw.pop('abs_range')
    domain_x, domain_y = (None, None) if domain is None else domain
    x1, x2 = getattr(f, 'domainX', (0, 1)) if domain_x is None else domain_x
    y1, y2 = getattr(f, 'domainY', (0, 1)) if domain_y is None else domain_y
    x1, x2, y1, y2 = map(float, (x1, x2, y1, y2))
    (rx1, rx2), (ry1, ry2) = rel_range
    if not isiterable(points):
        points = [points, points]
    x_space, y_space = points
    if not isiterable(x_space):
        x_space = np.linspace(x1 + rx1*(x2-x1), x1 + rx2*(x2-x1), x_space)
    if not isiterable(y_space):
        y_space = np.linspace(y1 + ry1*(y2-y1), y1 + ry2*(y2-y1), y_space)
    with plot_ctx_3d(show=show, close=close, **kw) as ax:
        if absolute:
            f_with_sign = f
            f = lambda x: abs(f_with_sign(x))
        if zoffset != 0:
            f_no_offset = f
            f = lambda x: f_no_offset(x) + zoffset
        if zlog:
            f_linear = f
            f = lambda x: np.log10(f_linear(x))
        X, Y, Z = prepare_grid(lambda x, y: float(f([x, y])), x_space, y_space)
        ax.plot_surface(
            X, Y, Z,
            **insert_missing(plot_kw, rstride=1, cstride=1,
                             cmap=cmap, color=c, edgecolor=ec,
                             linewidth=lw, antialiased=True)
        )
    if not show and not close:
        return ax


def plot_tri(points, triangles=None, lw=.4, c='w', ec='k', facecolors=None,
             cmap=cm.jet, colorbar=True, plot_kw=None, show=True, close=False,
             **kw):
    r"""Create a 3D surface plot with manual triangulation.

    Options not described here are similar to the ones of plot_2d().

    @param points
        Vertex points of the mesh
    @param triangles
        List of 3-tuples indicating which vertices form triangles. If not
        given, `points` will be interpreted as an object with a `surface()`
        method returning the actual points and triangles.
    @param lw (float, optional)
        Linewidth of the triangle edges. Default is `.4`.
    @param c (color, optional)
        Color of the triangle faces. May be overridden by `facecolors`.
    @param ec (color, optional)
        Edge color for the triangle edges. Set to `None` to not draw the wire.
    @param facecolors (iterable of floats, optional)
        A value assigned to each "face" (i.e. triangle). The values will be
        associated with colors of the colormap specified via `cmap`.
    @param cmap (colormap, optional)
        A colormap (e.g. one of the `matplotlib.cm` properties) to use for the
        face colors. Has no effect without `facecolors`.
    @param colorbar (boolean, optional)
        Whether to show a color bar indicating the values associated with
        colors of the `cmap`. Has no effect without `facecolors`.
    """
    if not isiterable(points):
        points, triangles = points.surface()
    if plot_kw is None:
        plot_kw = dict()
    xs, ys, zs = zip(*points)
    with plot_ctx_3d(show=show, close=close, **kw) as ax:
        pcollection = ax.plot_trisurf(
            xs, ys, zs, triangles=triangles, color=c,
            linewidth=lw, edgecolor=ec, **plot_kw
        )
        if facecolors is not None:
            pcollection.set_cmap(cmap)
            pcollection.set_array(np.array(facecolors))
            if colorbar:
                opts = colorbar if isinstance(colorbar, dict) else dict()
                ax.figure.colorbar(pcollection, **opts)
    if not show and not close:
        return ax


def _plane_intersection(p1, p2):
    r"""Return the point at which the line between two points crosses the x-y-plane.

    If the line does not cross the x-y-plane, `None` is returned.
    """
    z1, z2 = p1[2], p2[2]
    if z1 * z2 < 0.0 or (z1 * z2 == 0.0 and z1 + z2 != 0.0):
        t = - z1 / (z2-z1)
        return (p1 + t * (p2 - p1))[0:2]
    return None


def plot_tri_cut(points, triangles=None, figsize=(4, 4), **kw):
    r"""Plot the intersection curve of a triangle surface and the x-y-plane.

    The options are the same as for plot_data().
    """
    if not isiterable(points):
        points, triangles = points.surface()
    points = np.array(points)
    pts = []
    for tri in triangles:
        p1, p2, p3 = [points[t] for t in tri]
        for pa, pb in ((p1, p2), (p1, p3), (p2, p3)):
            pt = _plane_intersection(pa, pb)
            if pt is not None:
                pts.append(pt)
    # sort based on angle
    c = sum(pts) / len(pts)
    pts.sort(key=lambda pt: np.arctan2(*(pt - c)))
    if pts:
        pts.append(pts[0])
        xs, ys = zip(*pts)
    else:
        xs = ys = []
    return plot_data(xs, ys, figsize=figsize, **kw)
