r"""@package motsfinder.axisym.trackmots.optimize

Functions to perform various MOTS optimizations during tracking.

The main optimization performed here is the numerical search for close to
optimal bipolar coordinate parameters.
"""

import numpy as np
from scipy.interpolate import lagrange

from ..curve import BipolarCurve, RefParamCurve


__all__ = [
    "optimize_bipolar_scaling",
]


def optimize_bipolar_scaling(curve, initial_scale, move, factor=0.75,
                             threshold_factor=5.0, res_decrease_factor=0.9,
                             initial_smoothing=None, max_smoothing=0.1,
                             verbose=True, recursions=1, max_shrinking=0.05):
    r"""Perform successive tests to try to find optimal bipolar scale parameter.

    This function parameterizes the given MOTS in bipolar coordinates and
    reparameterizes in the s-t-coordinate-space. Afterwards, the residual
    expansion between collocation points is measured. By changing the scaling
    parameter (half-distance of the two foci), the result will generally
    become better or worse, depending on how well the coordinates are adapted
    to the curve's shape. A very simple numerical search is used to find a
    scale that is better than the given `initial_scale`.

    @param curve
        MOTS to base the measurements on. Should have low residual expansion
        already.
    @param initial_scale
        Scaling to start with.
    @param move
        Origin, i.e. center between the foci of the bipolar coordinates.
    @param factor
        Factor by which to increase or decrease the scaling each step,
        respectively. Default is `0.75`, i.e. increase using ``value/0.75``
        and decrease using ``value*0.75``.
    @param threshold_factor
        Factor by which to increase the initial residual expansion. This
        defines the target resolution used in each reparameterization. The
        idea is to avoid being in the roundoff plateau, so that values react
        to how well the scaling is suited to the problem, and not fluctuate a
        lot. Default is `0.9`.
    @param res_decrease_factor
        Factor used for initially finding the resolution with which to work
        during the search. The initial resolution is reduced successively by
        this factor until the residual expansion is above
        ``threshold_factor*residual``. Default is `10`.
    @param initial_smoothing
        If given, perform an optimization for the smoothing parameter of the
        ``'curv2'`` reparameterization strategy, starting from the given
        value. This is done after the bipolar scaling parameter is optimized.
    @param max_smoothing
        Maximum smoothing value to allow. For high smoothing values, the
        results usually don't change anymore. Under certain circumstances, the
        search may start to increase this smoothing value indefinitely
        (effectively amounting to the limit of arc-length parameterization).
        Default is `0.1`.
    @param verbose
        Whether to print status messages. Default is `True`.
    @param recursions
        If the result gets worse for the initial steps up and down, put a
        parabola through the three obtained values. Its minimum is then taken
        as the initial step for another search. The `recursions` value
        determines how often this should happen (i.e. it is reduced by one
        each recursion). If ``recursions==0``, the minimum of the parabola is
        taken as optimal value.
    @param max_shrinking
        Upon recursion (see above), don't move too close to not moving at all
        (i.e. a `factor` of `1.0`). This parameter ensures that
        ``(new_factor-1)/(factor-1) >= max_shrinking``.
    """
    def set_value_for_scale(value, num):
        bipolar_curve = BipolarCurve.from_curve(curve, num=num, scale=value,
                                                move=move)
        opts = dict(coord_space=True)
        if initial_smoothing is None:
            reparam = 'arc_length'
        else:
            reparam = 'curv2'
            opts['smoothing'] = initial_smoothing
        bipolar_curve.reparameterize(reparam, **opts)
        return bipolar_curve
    if verbose:
        print("Finding bipolar scaling parameter...")
    scale, num = _optimize_curve_parameter(
        curve, x0=initial_scale, set_value=set_value_for_scale,
        factor=factor, threshold_factor=threshold_factor,
        res_decrease_factor=res_decrease_factor, verbose=verbose,
        recursions=recursions, full_output=True, max_shrinking=max_shrinking,
    )
    if initial_smoothing is None:
        return scale
    def set_value_for_smoothing(value, num):
        bipolar_curve = BipolarCurve.from_curve(curve, num=num, scale=scale,
                                                move=move)
        bipolar_curve.reparameterize('curv2', smoothing=value,
                                     coord_space=True)
        return bipolar_curve
    if verbose:
        print("Finding 'curv2' reparameterization smoothing using scale=%s..."
              % scale)
    smoothing = _optimize_curve_parameter(
        curve, x0=initial_smoothing, set_value=set_value_for_smoothing,
        factor=factor, num=num, verbose=verbose, recursions=recursions,
        max_value=max_smoothing, max_shrinking=max_shrinking,
    )
    return scale, smoothing


def _optimize_curve_parameter(curve, x0, set_value, factor, max_shrinking,
                              max_value=None, threshold_factor=None,
                              res_decrease_factor=None, verbose=True,
                              recursions=1, num=None, full_output=False,
                              _cache=None):
    r"""Vary a parameter to minimize the required resolution.

    This varies a parameter and repeatedly checks the residual expansion of
    the resulting curve to see at which value the residual has its minimal
    value. This should lead to a lower required resolution when using the
    found parameter value for reparametrizing the curve.
    """
    def _p(msg):
        if verbose:
            print(msg)
    res_cache = dict() if _cache is None else _cache
    def _max_residual(num, value):
        key = (num, value)
        try:
            return res_cache[key]
        except KeyError:
            pass
        modified_curve = set_value(value, num=num)
        c1 = RefParamCurve.from_curve(modified_curve, num=0,
                                      metric=curve.metric)
        space = np.linspace(0, np.pi, 2*num+1, endpoint=False)[1:]
        residual = np.absolute(c1.expansions(params=space)).max()
        res_cache[key] = residual
        return residual
    a = x0
    if num is None:
        if threshold_factor is None or res_decrease_factor is None:
            raise TypeError("With `num` not specified, `threshold_factor` "
                            "and `res_decrease_factor` are mandatory.")
        err0 = _max_residual(num=curve.num, value=a)
        _p("residual expansion after conversion: %s" % err0)
        threshold = threshold_factor * err0
        num = curve.num
        while True:
            num = res_decrease_factor * num
            if _max_residual(num=int(num), value=a) >= threshold:
                break
        num = int(num)
    _p("performing search with resolution %s" % num)
    def f(x):
        value = _max_residual(num=num, value=x)
        _p("  residual for value=%s is: %.6e" % (x, value))
        return value
    def _step(x):
        x1 = factor * x
        return min(x1, max_value) if max_value else x1
    fa = f(a)
    b = _step(a)
    fb = f(b)
    if fb >= fa:
        factor = 1./factor
        c = _step(a)
        fc = f(c)
        if fc >= fa:
            # worse or equal in both directions
            data = sorted(
                # set() removes any duplicate entries
                set([(a, fa), (b, fb), (c, fc)]), key=lambda x: x[0]
            )
            xs, ys = zip(*data)
            if len(xs) < 3:
                if recursions > 0:
                    smaller_factor = 0.5 * (factor - 1.0) + 1.0
                    opt = x0 * smaller_factor
                else:
                    _p("Ending search at boundary.")
                    return (a, num) if full_output else a
            else:
                opt = lagrange(xs, np.log10(ys)).deriv().roots.real[0]
            if recursions > 0:
                _p("Worse or equal in both directions. Recursing...")
                new_factor = _limit_factor_shrinking(
                    opt/x0, factor, max_shrinking, verbose=verbose,
                )
                opt = _optimize_curve_parameter(
                    curve=curve, x0=x0, max_shrinking=max_shrinking,
                    max_value=max_value, set_value=set_value,
                    factor=new_factor, num=num, verbose=verbose,
                    recursions=recursions-1,
                    _cache=res_cache,
                )
            return (opt, num) if full_output else opt
        a, b = b, c
        fa, fb = fb, fc
    while True:
        c = _step(b)
        if c == b:
            _p("Search reached boundary. Ending here.")
            return (c, num) if full_output else c
        fc = f(c)
        if fc > fb:
            opt = lagrange([a, b, c], np.log10([fa, fb, fc])).deriv().roots.real[0]
            return (opt, num) if full_output else opt
        a, b = b, c
        fa, fb = fb, fc


def _limit_factor_shrinking(new_factor, factor, max_shrinking, verbose):
    r"""Make sure the new factor is not too much closer to 1 than the old one."""
    if max_shrinking is None:
        return new_factor
    # Make sure both factors are on "the same side" of 1.
    if (new_factor-1) * (factor-1) < 0:
        factor = 1./factor
    if (new_factor-1) / (factor-1) < max_shrinking:
        fixed_factor = max_shrinking * (factor-1) + 1
        if verbose:
            print("Factor %s too much closer to 1.0 than the old one %s. "
                  "Changed to %s." % (new_factor, factor, fixed_factor))
        new_factor = fixed_factor
    return new_factor
