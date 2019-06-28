r"""@package motsfinder.axisym.utils

Utilities for curve analysis/modification.
"""

import numpy as np

from ..utils import lrange


__all__ = [
    "detect_coeff_knee",
]


def detect_coeff_knee(coeffs, n_min=10, limit_order_up=8, limit_order_down=5,
                      threshold=0.05, min_window=4, window_ratio=0.01):
    r"""Find the point at which the coefficients stop converging exponentially.

    This does a crude but somewhat conservative estimate of whether there is a
    point after which the coefficients seem to not decay exponentially
    anymore. It does so by first finding a region to look at. The assumption
    is that there may be "noise" (i.e. important actual information) in the
    first few coefficients, followed by a region of exponential decay,
    followed by sub-exponential decay or roundoff noise. Sub-exponential decay
    may be a sign of overfitting.

    @return Index of the coefficient where a cutoff is suggested. `None` if no
        such suggestion can be made.

    @param coeffs
        The coefficient list to look at.
    @param n_min
        Minimum number of elements to ignore at the beginning of the data.
    @param limit_order_up,limit_order_down
        Determine the region to look at by taking the largest coefficient and
        moving `limit_order_down` orders of magnitude down or the smallest
        coefficient and moving `limit_order_up` orders of magnitude up. The
        smaller of the two values is taken and the first coefficient lying
        below that value is taken as start of the region. If no such
        coefficient exists, all the data is looked at.
    @param threshold
        Value of the *defect* function demarcating the knee. Default is
        `0.05`. Use a larger value to be more conservative (i.e. tending to
        larger resolutions).
    @param min_window
        Number of points in each direction of a point to consider as one
        point. This is necessary to remove or at least reduce issues of large
        jumps in coefficient values due to e.g. symmetries in the problem. For
        example, if fitting a symmetric function with a Cosine series, every
        other coefficient will be zero. The points
        ``i-window,...,i,...,i+window`` are taken and the maximum (absolute)
        value is used in each instance where a coefficient order of magnitude
        is looked at.
    @param window_ratio
        Fraction of the full list to use as window, if larger than
        `min_window`.

    @b Examples

    ```
        data = []
        data += [5*r for i, r in enumerate(np.random.randn(40))]
        data += [r*np.exp(-i/10.) for i, r in enumerate(np.random.randn(200))]
        data += [r*(i+10)**(-3)*1e-5 for i, r in enumerate(np.random.randn(600))]
        knee = detect_coeff_knee(data)
        print("Found knee at %s" % knee)
        ax = plot_data(data, absolute=True, ylog=True, figsize=(14, 6), show=False)
        ax.axvline(knee)
        plt.show()
    ```
    """
    coeffs = np.absolute(np.asarray(coeffs))
    max_coeff = coeffs.max()
    coeffs = coeffs[n_min:]
    coeffs = np.log10(coeffs)
    max_coeff = np.log10(max_coeff)
    N = len(coeffs)
    window = max(min_window, int(window_ratio*N))
    def f(i):
        return max(coeffs[i-window:i+window])
    wrange = lrange(window, N-window)
    best = min([max(coeffs[i-window:i+window]) for i in wrange])
    limit = min(best + limit_order_up, max_coeff - limit_order_down)
    start_idx = wrange[-1]
    while start_idx > wrange[0] and f(start_idx) < limit:
        start_idx -= 1
    if f(start_idx) >= limit and start_idx-window > 0:
        coeffs = coeffs[start_idx-window:]
        n_min += start_idx-window
        N = len(coeffs)
        wrange = range(window, N-window)
    worst = coeffs.max()
    def max_defect(i):
        # Return the maximum (normalized) distance of the *knee* to the
        # straight connection from 0 to i.
        a = np.asarray([0, worst/best])
        b = np.asarray([i/N, f(i)/best])
        def _defect(j):
            # Return the (normalized) distance of the point j to the
            # connecting line 0 to i.
            return f(j)/best - ((b[1]-a[1]) * float(j)/i + a[1])
        return max([0] + [_defect(j) for j in range(window, i-window, window)])
    # Search from the end to catch the highest resolution we might want to
    # keep.
    knee = next((i+n_min for i in reversed(wrange) if max_defect(i) < threshold), None)
    if knee + window > wrange[-1]:
        # We're below the threshold near the end already, so we should not
        # detect a knee at all.
        knee = None
    return knee
