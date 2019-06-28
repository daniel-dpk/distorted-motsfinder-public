r"""@package motsfinder.axisym.newton

Implementation of a Newton-Kantorovich method to find MOTSs.

This module contains the Newton-Kantorovich finder newton_kantorovich(), which
is a high-level function to search for MOTSs based on an initial guess.

A basic description of the method can be found in \ref boyd2001 "[1]",
Appendix C, and in \ref pookkolb2018_1 "[2]".

@b Examples

```
    metric = BrillLindquistMetric(m1=1, m2=1, d=1.4, axis='z')
    c0 = StarShapedCurve.create_sphere(radius=2.0, num=100, metric=metric)
    c_AH = newton_kantorovich(c0, verbose=True)
    c_AH.plot(label='AH')
```

@b References

\anchor boyd2001 [1] Boyd, J. P. "Chebyshev and Fourier Spectral Methods.
    Dover Publications Inc." New York (2001).

\anchor pookkolb2018_1 [2] D. Pook-Kolb, O. Birnholtz, B. Krishnan and E.
    Schnetter, "The existence and stability of marginally trapped surfaces."
    arXiv:1811.10405 [gr-qc].
"""

from __future__ import print_function
from builtins import range

from scipy.linalg import LinAlgWarning, LinAlgError
import numpy as np

from ..utils import insert_missing, process_pool
from ..numutils import raise_all_warnings, NumericalError
from ..ndsolve import ndsolve, CosineBasis
from .utils import detect_coeff_knee


__all__ = [
    "newton_kantorovich",
]


class NoConvergence(Exception):
    r"""Base for exceptions indicating failed convergence of Newton steps.

    This exception is raised directly when an error is raised by methods
    called during the Newton steps when these are related to convergence (e.g.
    `scipy.linalg.LinAlgWarning`.
    """
    pass


class InsufficientResolutionOrRoundoff(NoConvergence):
    r"""Raised when resolution is too low or roundoff prevents convergence.

    We currently cannot differentiate between these two cases.
    """
    pass


class StepLimitExceeded(NoConvergence):
    r"""Raised when convergence not achieved within the step count limit."""
    pass


def newton_kantorovich(initial_curve, c=0.0, steps=50, step_mult=0.5, rtol=0,
                       atol=1e-12, accurate_test_res=500,
                       auto_resolution=True, max_resolution=1000,
                       linear_regime_threshold=1e-2, mat_solver='scipy.solve',
                       fake_steps=3,
                       plot_steps=False, reference_curves=(),
                       plot_deltas=False, verbose=False, disp=True,
                       parallel=False, **kw):
    r"""Do a Newton type search to find a MOTS.

    This function performs scaled Newton steps to solve the nonlinear problem
    of finding a MOTS.

    @param initial_curve
        (ExpansionCurve subclass)
        Initial guess for the trial surface. The current resolution of the
        horizon function in this curve will be used for solving the linear
        problem for each step. The specific type of this curve determines
        the parameterization of the curve. Also, this curve knows its
        geometry, so the metric does not need to be supplied here.
    @param c
        (float, optional)
        The target expansion. Default is `0`, i.e. we're trying to find MOTSs.
        If nonzero, constant expansion surfaces are found.
    @param steps
        (int, optional)
        Maximum number of Newton steps to take. Default is `50`.
    @param step_mult
        (float, optional)
        Factor to multiply the Newton steps with.
    @param rtol
        Float indicating when to stop taking more steps. If the maximum
        expansion (or its deviation from `c`) divided by the maximum
        expansion of the initial evaluation is less or equal to `rtol`, we
        consider the Newton steps to have converged. Default is `0`.
    @param atol
        Float indicating the absolute expansion below which we stop
        taking more Newton steps. Default is `1e-12`.
    @param accurate_test_res
        Number of samples to check the expansion at to decide upon
        convergence. If ``'twice'``, uses two times the curve's resolution. If
        zero or `None`, convergence is only checked at the collocation points
        and `auto_resolution` has no effect. Default is `500`.
    @param auto_resolution
        Boolean indicating whether the resolution should be increased if
        insufficient resolution is detected. Default: `True`.
    @param max_resolution
        Maximum resolution to increase to. Failure to converge at this
        resolution will terminate the search. Default: `1000`.
    @param linear_regime_threshold
        Float indicating the absolute (see `atol`) expansion below which we
        take full Newton steps to accelerate convergence. Default is `1e-2`.
    @param mat_solver
        (string, optional)
        Matrix solver method to use. Default is ``'scipy.solve'``, which is a
        fast solver with floating point accuracy. See
        ..ndsolve.solver.ndsolve() for more information.
    @param fake_steps
        Number of additional steps taken when the solution has converged at
        the collocation points but not yet between them. After these steps,
        the resolution is increased (or the search terminated).
    @param plot_steps
        (boolean, optional)
        Whether to plot each Newton step. Default is `False`.
    @param reference_curves
        (iterable, optional)
        Iterable of ``(curve, options)`` pairs to be plotted together with
        each step for comparison. The options part should either be a
        label string or a `dict` with options to be passed to
        `plot_curve`.
    @param plot_deltas
        (boolean, optional)
        Whether to plot the individual pseudospectral solutions of the linear
        problems. Default is `False`.
    @param verbose
        Whether to print status information during the Newton search. Default
        is `False`.
    @param disp
        Whether to raise NoConvergence (or subclasses) when the desired
        tolerance could not be reached in the number of steps or resolution.
        Default is `True`.
    @param parallel
        Whether to evaluate the equation using multiple processes in parallel.
        If `True`, uses all available threads. If an integer, uses that many
        threads. Default is `False`, i.e. don't compute in parallel.
    @param **kw
        Remaining keyword arguments are set as attributes on the solver
        object. They are documented as public class attributes of
        NewtonKantorovich.
    """
    solver = NewtonKantorovich(max_steps=steps, step_mult=step_mult,
                               atol=atol, rtol=rtol, verbose=verbose)
    solver.target_expansion = c
    solver.accurate_test_res = accurate_test_res
    solver.auto_resolution = auto_resolution
    solver.max_resolution = max_resolution
    solver.linear_regime_threshold = linear_regime_threshold
    solver.mat_solver = mat_solver
    solver.max_fake_convergent_steps = fake_steps
    solver.plot_steps = plot_steps
    solver.plot_deltas = plot_deltas
    solver.reference_curves = reference_curves
    solver.disp = disp
    solver.parallel = parallel
    for key, val in kw.items():
        setattr(solver, key, val)
    return solver.solve(initial_curve)


class NewtonKantorovich(object):
    r"""Class implementing the Newton-Kantorovich steps.

    The Newton-Kantorovich method is implemented as a class to keep track of
    the current state across several methods while taking successive steps.

    The docstring of newton_kantorovich() contains the actual description of
    the method and explains the parameters.

    After constructing a NewtonKantorovich object, configure it using its
    public instance attributes. Then, call solve() to perform the search.
    """

    __slots__ = ("max_steps", "step_mult", "atol", "rtol", "verbose",
                 "target_expansion", "accurate_test_res", "auto_resolution",
                 "max_resolution", "linear_regime_threshold", "mat_solver",
                 "__fake_convergent_steps", "max_fake_convergent_steps",
                 "res_increase_factor", "min_res_increase_factor",
                 "max_res_increase", "res_decrease_factor",
                 "res_decrease_accel", "res_init_factor",
                 "downsampling_tol_factor", "linear_regime_res_factor",
                 "min_res", "detect_plateau", "plateau_tol",
                 "knee_detection", "knee_increase_factor", "knee_threshold",
                 "_knee_detected",
                 "liberal_downsampling", "plot_steps", "plot_deltas",
                 "plot_opts", "reference_curves", "disp", "parallel",
                 "_prev_accurate_delta", "_prev_res_accurate_delta",
                 "user_data")

    def __init__(self, max_steps=50, step_mult=0.5, atol=1e-12, rtol=0,
                 verbose=False):
        r"""Create a Newton-Kantorovich solver object.

        Most of the configuration of the method is done via public instance
        variables. Since the class uses ``__slots__``, there is no chance that
        typos in these attributes go by undetected.

        The public instance variables which correspond to parameters of the
        newton_kantorovich() method are described there.

        @param max_steps
            Maximum number of Newton steps to take. If this number is reached
            before convergence is detected, we raise an error or return the
            current result, depending on the `disp` instance variable. This
            corresponds to the `steps` parameter in newton_kantorovich().
            Default is `50`.
        @param step_mult
            Corresponds to the same parameter in newton_kantorovich().
        @param atol
            Corresponds to the same parameter in newton_kantorovich().
        @param rtol
            Corresponds to the same parameter in newton_kantorovich().
        @param verbose
            Corresponds to the same parameter in newton_kantorovich().
        """
        ## Maximum number of Newton steps to take.
        self.max_steps = max_steps
        ## Factor to multiply the Newton steps with.
        self.step_mult = step_mult
        ## Absolute tolerance. If the error at the collocation points falls
        ## below this threshold, the result is assumed to be converged.
        ## However, if accurate convergence detection is active, this
        ## tolerance serves as threshold to reach before the result is
        ## considered converged. The `auto_resolution` setting then controls
        ## whether the resolution is increased to reach this tolerance.
        self.atol = atol
        ## Relative tolerance currently not implemented in a meaningful way.
        ## Until then, it is best to set (or leave) it at zero.
        self.rtol = rtol
        ## Whether to print status information during the Newton search.
        self.verbose = verbose
        ## Expansion of the surface to find. Default is zero (i.e. to find a
        ## MOTS).
        self.target_expansion = 0.0
        ## Resolution used to check convergence between collocation points. If
        ## ``'twice'``, uses two times the curve's resolution. If zero or
        ## `None`, convergence is only checked at the collocation points and
        ## `auto_resolution` has no effect. Default is `500`.
        self.accurate_test_res = 500
        ## Whether the resolution should be increased if insufficient
        ## resolution is detected. Default: `True`.
        self.auto_resolution = True
        ## Maximum resolution to increase to. Failure to converge at this
        ## resolution will terminate the search. Default: `1000`.
        self.max_resolution = 1000
        ## Whether to raise NoConvergence (or subclasses) when the desired
        ## tolerance could not be reached in the number of steps or
        ## resolution. Default is `True`.
        self.disp = True
        ## Float indicating the absolute (see `atol`) expansion below which we
        ## take full Newton steps to accelerate convergence. Default is
        ## `1e-2`.
        self.linear_regime_threshold = 1e-2
        ## Matrix solver method to use. Default is ``'scipy.solve'``, which is
        ## a fast solver with floating point accuracy. See
        ## ..ndsolve.solver.ndsolve() for more information.
        self.mat_solver = 'scipy.solve'
        self.__fake_convergent_steps = 0
        ## Steps to take when we have already converged at the collocation
        ## points but not everywhere yet.
        self.max_fake_convergent_steps = 3
        ## Factor by which to increase the resolution in case insufficient
        ## resolution is detected (should be > 1.0).
        self.res_increase_factor = 1.5
        ## If res increase is limited by `max_resolution`, this is the minimal
        ## factor by which the resolution must effectively increase for
        ## plateau detection to be reliable. Default is to use
        ## `res_increase_factor`. Note that this only applies to resolutions
        ## near the `max_resolution`, where smaller relative resolution
        ## increases may be sufficient for plateau detection.
        self.min_res_increase_factor = None
        ## Maximum resolution increase per step. When the tolerance is reached
        ## at the collocation points but not between them, the resolution is
        ## increased by multiplying the current resolution with
        ## `res_increase_factor`. This increase can be limited using
        ## `max_res_increase`. Note that plateau detection is *not* turned off
        ## if this limit is reached even if the `min_res_increase_factor` is
        ## *not* reached. This may help limiting the resolution increase in
        ## cases where convergence is worse than exponential. A use-case is
        ## when using Hermite interpolation of numerical data which may lead
        ## to an explosion of collocation points if the solver, after having
        ## reached the actual plateau exponentially, still sees
        ## sub-exponential convergence occurring from making its complicated
        ## way between the smooth numerical inaccuracies (i.e. when it tries
        ## to do sub-grid-point corrections).
        self.max_res_increase = None
        ## Once converged and with `auto_resolution==True`, this controls in
        ## which steps the resolution is reduced as long as the tolerance is
        ## maintained.
        self.res_decrease_factor = 0.9 # should be < 1
        ## This accelerates the resolution decreasing steps with each step.
        self.res_decrease_accel = 0.95 # should be <= 1
        ## Factor by which errors can increase during downsampling while still
        ## staying below the configured absolute tolerance.
        self.downsampling_tol_factor = 5.0
        ## Initial resolution multiplier before starting the search.
        self.res_init_factor = 1
        ## Once the linear regime is reached, the resolution is multiplied by
        ## this factor.
        self.linear_regime_res_factor = 1
        ## Minimum resolution when decreasing the resolution.
        self.min_res = 1
        ## Whether to stop increasing the resolution when it stops producing
        ## more accurate results.
        self.detect_plateau = False
        ## Plateau is detected if new results are not at least `plateau_tol`
        ## times better than previous results.
        self.plateau_tol = 1.5
        ## Whether to estimate the plateau based on "knee" detection in the
        ## horizon function's coefficient list.\ Any time convergence occurs
        ## at the collocation points, this test is performed (before any
        ## plateau detection) and if a cutoff is suggested, the result is
        ## defined to have converged.\ In that case, the suggested resolution
        ## (plus a few extra points) is used for a few more steps until we
        ## have convergence at these new collocation points.
        self.knee_detection = False
        ## How many extra collocation points (as a fraction of the suggested
        ## cutoff) to add to the suggested cutoff.\ If the resulting
        ## resolution is larger than the current resolution, knee detection is
        ## ignored for that round.
        self.knee_increase_factor = 0.2
        ## Threshold for the "knee" detection in case ``knee_detection==True``.
        self.knee_threshold = 0.05
        ## Whether we're doing the final few steps after the knee was found.
        self._knee_detected = False
        ## If `True`, ignore `atol` while downsampling and only consider
        ## `downsampling_tol_factor`.
        self.liberal_downsampling = False
        ## Whether to plot each Newton step.
        self.plot_steps = False
        ## Whether to plot the solutions to the linear problem of the Newton
        ## steps.
        self.plot_deltas = False
        ## Options used for plotting the Newton steps.
        self.plot_opts = dict()
        ## Optional curves to add to the Newton step plots.
        self.reference_curves = ()
        ## Whether to evaluate the equation at the collocation points in
        ## parallel processes. Note that for numerical (grid) data, this may
        ## lead to an immense increase in memory usage, so it is best to use
        ## this only for analytical metrics.
        self.parallel = False
        ## Custom data to store in the final curve (excluding convergence data).
        self.user_data = dict()
        self._prev_accurate_delta = None
        self._prev_res_accurate_delta = None

    def solve(self, initial_curve):
        r"""Perform the actual Newton steps to find a solution.

        See the docstring of newton_kantorovich() for more information.
        """
        with raise_all_warnings():
            try:
                if self.parallel:
                    with process_pool() as pool:
                        return self._solve(initial_curve, pool=pool)
                return self._solve(initial_curve)
            except (LinAlgWarning, LinAlgError, FloatingPointError,
                    NumericalError) as e:
                self._raise(NoConvergence, str(e))

    def _solve(self, initial_curve, pool=None):
        r"""Wrapped function for performing the Newton steps."""
        curve = initial_curve.copy()
        if self.user_data:
            curve.user_data.update(**self.user_data)
        step_mult = self.step_mult
        self.__fake_convergent_steps = 0
        eq = _LinearExpansionEquation(
            curve,
            target_expansion=self.target_expansion,
            accurate_test_res=self.accurate_test_res,
            parallel=self.parallel,
            pool=pool
        )
        initial_res = curve.num
        new_res = max(self.min_res, int(round(self.res_init_factor * curve.num)))
        if new_res != curve.num and step_mult != 1.0:
            if self.verbose:
                print("Initial steps with resolution %s" % new_res)
            curve.resample(new_res)
        curve.user_data['converged'] = False
        curve.user_data['reason'] = "unspecified"
        for i in range(self.max_steps):
            self._plot_step(curve, step=i)
            d = ndsolve(
                eq=eq,
                basis=CosineBasis(domain=curve.domain, num=curve.num,
                                  lobatto=False),
                mat_solver=self.mat_solver,
            )
            if self.verbose:
                print("%02d: max error at %d collocation points: %s"
                      % (i+1, d.N, eq.prev_abs_delta))
            if self._has_converged(eq, curve, d):
                self._downsample(eq, curve)
                break
            elif i+1 == self.max_steps:
                self._raise(
                    StepLimitExceeded,
                    "Newton-Kantorovich search did not reach "
                    "desired tolerance within %d steps."
                    % (self.max_steps)
                )
                curve.user_data['reason'] = "step limit"
            if eq.prev_abs_delta <= self.linear_regime_threshold:
                if step_mult != 1.0:
                    if self.linear_regime_res_factor == "restore":
                        new_res = initial_res
                    elif self.linear_regime_res_factor != 1:
                        new_res = int(round(self.linear_regime_res_factor*curve.num))
                    if new_res != curve.num:
                        if self.verbose:
                            print("  Upsampling to %d in linear regime" % new_res)
                        self._change_resolution(curve, d, new_res)
                step_mult = 1.0
            curve.h.a_n += step_mult * np.asarray(d.a_n)
            self._plot_delta(d, step=i)
        return curve

    def _downsample(self, eq, curve):
        r"""Downsample the given curve without making the result worse.

        This is done by performing the *accurate* convergence test on
        increasingly lower resolution copies of the `curve` and making sure
        the result is not (much) worse. We allow the result to become worse by
        a small amount but no worse than the tolerance.
        """
        prev_delta = self._prev_accurate_delta
        if (not self.auto_resolution or not eq.has_accurate_test()
                or not self.atol or self.rtol):
            # downsampling only safe if we're just using `atol`
            return
        tol = self.downsampling_tol_factor * prev_delta
        if not self.liberal_downsampling:
            if prev_delta >= self.atol:
                return
            tol = min(tol, self.atol)
        test_curve = curve.copy()
        res = test_curve.num
        f = self.res_decrease_factor
        downsampled_delta = prev_delta
        while True:
            if res <= self.min_res:
                break
            prev_delta = downsampled_delta
            downsampled_delta = eq.accurate_abs_delta(
                test_curve.resample(max(self.min_res, int(f*res)))
            )
            if downsampled_delta > tol:
                # Downsampled curve not OK anymore, use previous resolution
                break
            if res == test_curve.num:
                # new_res == res  =>  seems the horizon function is zero
                break
            res = test_curve.num
            # sample down slightly more aggressive next round
            f *= self.res_decrease_accel
        if self.verbose and res != curve.num:
            print("  Curve downsampled to %s, accurate error now %g"
                  % (res, prev_delta))
        curve.resample(res)

    def _has_converged(self, eq, curve, d):
        r"""Check whether we have converged and found a solution.

        This checks if either the relative or absolute tolerance has been
        achieved and, in case it has, returns `True`.

        In case the tolerance has been achieved at the collocation points
        (which is a fast O(1) test), a more detailed convergence check is
        performed at more points using _accurate_convergence_test().

        @param eq
            _LinearExpansionEquation object.
        @param curve
            The current state of the solution.
        @param d
            The current solution of the linear problem.
        """
        if eq.prev_rel_delta > self.rtol and eq.prev_abs_delta > self.atol:
            self.__fake_convergent_steps = 0
            return False
        # The definition of "relative tolerance" is currently work in progress
        # and no "accurate" test has been developed for it yet. This means
        # that, in case `rtol` is used (i.e. nonzero) and satisfied, we have
        # converged by that criterion.
        if eq.prev_rel_delta <= self.rtol:
            curve.user_data['converged'] = True
            curve.user_data['reason'] = "converged"
            return True
        # Expansion is below tolerance at the collocation points. Now check it
        # is low elsewhere too.
        return self._accurate_convergence_test(eq, curve, d)

    def _accurate_convergence_test(self, eq, curve, d):
        r"""Check for convergence and detect insufficient resolution.

        This method checks if we have converged only at the collocation points
        or everywhere (i.e. at a reasonable number of points not identical to
        the collocation points). If only at the collocation points, this step
        is called a "fake convergent step" and the respective counter is
        increased. After a few of such fake steps were taken without actual
        convergence, we likely need to increase the resolution (or the
        tolerance lies below the roundoff plateau). In case we're allowed to,
        the resolution is then increased and more steps are taken.
        """
        # Note: We enter this function each time we have converged at the
        #       collocation points. Any logic that should occur at these
        #       instants is put here. (This does not mean we should not
        #       refactor the logic into smaller methods...)
        def _p(msg):
            if self.verbose:
                print(msg)
        accurate_delta = eq.accurate_abs_delta(curve)
        self._prev_accurate_delta = accurate_delta
        if not self.atol or accurate_delta <= self.atol:
            # We really have converged (or should not check).
            if eq.has_accurate_test():
                if self.atol:
                    _p("  Accurate error = %g <= %g at resolution %s"
                       % (accurate_delta, self.atol, curve.num))
                else:
                    _p("  Accurate error = %g at resolution %s"
                       % (accurate_delta, curve.num))
            # Even if a knee was detected, we've now *officially* converged
            # even between collocation points.
            curve.user_data['converged'] = True
            curve.user_data['reason'] = "converged"
            return True
        if self._knee_detected:
            ## We've now converged at the collocation points after a new
            ## resolution was set due to a "knee" having been detected in the
            ## coefficients.
            _p("  Accurate error %g > %g." % (accurate_delta, self.atol))
            return True
        # Expansion is not low enough. This is a "fake convergent" step.
        self.__fake_convergent_steps += 1
        if self.__fake_convergent_steps < self.max_fake_convergent_steps:
            # We allow some fake steps to account for the case that the
            # violating values need just one or two extra steps to converge.
            return False
        if self.knee_detection:
            suggested_res = self._knee_detection(curve)
            if suggested_res:
                # Knee detection successful. This means we *have* converged
                # but still need to do one more round to converge at the
                # collocation points.
                _p("  Accurate error still %g > %g."
                   % (accurate_delta, self.atol))
                _p("  Sub-exponential convergence of coefficients found. "
                   "Suggested maximum resolution: %s" % suggested_res)
                _p("  Continuing to converge at (new) collocation points.")
                curve.user_data['converged'] = True
                curve.user_data['reason'] = "knee"
                self._knee_detected = True
                self._change_resolution(curve, d, suggested_res)
                self.detect_plateau = False
                return False
        if self.detect_plateau:
            prev_res_delta = self._prev_res_accurate_delta
            if prev_res_delta and self.plateau_tol*accurate_delta >= prev_res_delta:
                # increasing the resolution did not help appreciably
                _p("  Plateau detected; remaining error = %g > %g at resolution %s"
                   % (accurate_delta, self.atol, curve.num))
                curve.user_data['converged'] = True
                curve.user_data['reason'] = "plateau"
                return True
        # We probably need a higher resolution since several fake convergent
        # steps have been taken now.
        if not self.auto_resolution or curve.num >= self.max_resolution:
            _p("  Accurate error still %g > %g... stopping with resolution %d"
               % (accurate_delta, self.atol, curve.num))
            self._raise(  # only raises if self.disp==True
                InsufficientResolutionOrRoundoff,
                "Newton-Kantorovich search converged only "
                "at the collocation points. Resolution "
                "insufficient or roundoff error too high."
            )
            # We can't expect further progress. Since we should not report
            # convergence problems, we might as well return now.
            curve.user_data['reason'] = "insufficient resolution or roundoff"
            return True
        # Increase resolution and take more steps.
        self._prev_res_accurate_delta = accurate_delta
        num = int(self.res_increase_factor * curve.num)
        if self.max_res_increase:
            num = min(curve.num+self.max_res_increase, num)
        if num > self.max_resolution:
            num = self.max_resolution
            # can't detect plateau for small resolution changes
            if (self.detect_plateau
                    and not self._enough_res_increase(curve.num, num)):
                self.detect_plateau = False
                _p("Too little resolution increase. Plateau detection deactivated.")
        _p("  Accurate error still %g > %g... increasing resolution %d -> %d"
           % (accurate_delta, self.atol, curve.num, num))
        self._change_resolution(curve, d, num)
        self.__fake_convergent_steps = 0
        return False

    def _enough_res_increase(self, num1, num2):
        r"""Check if the increase `num1` -> `num2` is large enough for plateau detection."""
        min_factor = self.res_increase_factor
        if self.min_res_increase_factor:
            min_factor = self.min_res_increase_factor
        return num2 >= min_factor*num1 or (
            self.max_res_increase and num2-num1 >= self.max_res_increase
        )

    def _change_resolution(self, curve, d, res):
        r"""Resample the curve and current linear solution.

        This respects the configured minimal resolution.
        """
        res = max(self.min_res, res)
        curve.resample(res)
        d.resample(res)

    def _knee_detection(self, curve):
        r"""Check if there is a *knee* in the coefficient list."""
        try:
            knee = detect_coeff_knee(curve.h.a_n, threshold=self.knee_threshold)
        except FloatingPointError:
            return None
        if knee is None:
            return None
        knee *= int(round(1 + self.knee_increase_factor))
        knee = max(self.min_res, knee)
        if knee >= curve.num:
            return None
        return knee

    def _raise(self, ex_cls, msg):
        r"""Raise an exception depending on the `disp` setting.

        This is a no-op if `disp==False`.
        """
        if self.disp:
            raise ex_cls(msg)
        if self.verbose:
            print("Stopping Newton search: %s" % msg)

    def _plot_step(self, curve, step):
        r"""Plot the result of a Newton step together with reference curve."""
        if not self.plot_steps:
            return
        opts = self.plot_opts.copy()
        opts['title'] = opts.get('title', "step {step}").format(step=step)
        ax = curve.plot_curves(
            (curve, 'trial', '-k'), *self.reference_curves,
            show=False, **opts
        )
        curve.plot(
            points=curve.h.collocation_points(min(50, curve.num)),
            label=None, ax=ax, **insert_missing(opts, l='.k')
        )

    def _plot_delta(self, delta, step):
        r"""Plot the solution function of the linearized equation."""
        if not self.plot_deltas:
            return
        from ..ipyutils import plot_1d
        plot_1d(delta, title=r"$\Delta(\theta)$ at step %d" % step,
                xlabel=r"$\theta$", mark_points=delta.collocation_points())


class _LinearExpansionEquation(object):
    r"""Compute the linearized expansion equation and keep track of results."""

    def __init__(self, curve, target_expansion=0,
                 accurate_test_res=None, parallel=False, pool=None):
        r"""Init function.

        @param curve
            Curve along which the expansion/equation should be computed.
        @param target_expansion
            Constant expansion to aim for. Will be added to the inhomogeneity
            of the linear equation. Default is `0`.
        @param accurate_test_res
            Resolution for the more accurate convergence check.
        @param parallel
            Whether to evaluate the equation using multiple processes in parallel.
        @param pool
            Optional processing pool to re-use instead of creating a new one.
        """
        ## Current curve object.
        self.curve = curve
        ## Target expansion (0 for MOTSs).
        self.c = target_expansion
        ## Resolution for the more accurate convergence check.
        self.accurate_test_res = accurate_test_res
        ## Maximum deviations from target expansion of all evaluations.
        self._max_deltas = []
        self._parallel = parallel
        self._pool = pool

    def __call__(self, pts):
        r"""Return the equation suitable for ndsolve()."""
        eq = self.curve.linearized_equation(pts, target_expansion=self.c,
                                            parallel=self._parallel,
                                            pool=self._pool)
        inhom = eq[1]
        # Due to boundary conditions, the equation is not evaluated at the
        # boundary points (where the operator is not defined anyway).
        self._max_deltas.append(np.absolute(inhom[1:-1]).max())
        return eq

    @property
    def prev_rel_delta(self):
        r"""Maximum deviation from target expansion relative to first evaluation."""
        return self._max_deltas[-1] / self._max_deltas[0]

    @property
    def prev_abs_delta(self):
        r"""Maximum deviation from target expansion."""
        return self._max_deltas[-1]

    def has_accurate_test(self):
        r"""Return whether we should perform accurate convergence tests
        between collocation points."""
        return bool(self.accurate_test_res)

    def accurate_abs_delta(self, curve):
        r"""Compute the deviation from the target expansion more accurately.

        This uses the `accurate_test_res` to evaluate the expansion at a
        different set of points than the collocation points and returns the
        maximum absolute value.
        """
        if not self.has_accurate_test():
            return self.prev_abs_delta
        if self.accurate_test_res == "twice":
            res = 2 * curve.num
        else:
            res = int(self.accurate_test_res)
        pts = np.linspace(0, np.pi, res+1, endpoint=False)[1:]
        values = np.asarray(curve.expansions(pts))
        accurate_delta = np.max(np.absolute(values - self.c))
        curve.user_data['accurate_abs_delta'] = dict(
            points=pts, values=values, max=accurate_delta
        )
        return accurate_delta
