r"""@package motsfinder.axisym.findmots

Convenience function(s) to coordinate finding MOTSs.

This module provides convenience functions to configure and run a MOTS search.
Results are optionally stored to disk. If a result is found on disk, it is
loaded instead of being recomputed from scratch.


@b Examples

```
    # Define the base configuration.
    base_cfg = BrillLindquistConfig(
        d=0.65, m1=0.2, m2=0.8, save=True, folder='tests',
        ref_num=25, reparam=(0.1, 0.5)
    )

    # Find the Apparent Horizon (AH).
    cfg = base_cfg.copy().update(c_ref=0.8, num=40, suffix='AH', name='AH')
    c_AH = find_mots(cfg)

    # From there, step inward and find the inner common MOTS.
    cfg = base_cfg.copy().update(c_ref=c_AH, suffix='inner', name='inner',
                                 offset_coeffs=[-0.2, 0.0, 0.1])
    c_inner = find_mots(cfg)

    # Plot the results.
    BaseCurve.plot_curves(c_AH, c_inner)
```
"""

from abc import ABCMeta, abstractmethod
import os.path as op
import numbers
from collections import OrderedDict
from six import add_metaclass

from ..utils import insert_missing, timethis
from ..numutils import raise_all_warnings
from ..metric import BrillLindquistMetric
from ..exprs.numexpr import save_to_file
from .curve import StarShapedCurve, RefParamCurve, BaseCurve
from .curve import ParametricCurve, BipolarCurve
from .newton import newton_kantorovich, NoConvergence


__all__ = [
    "find_mots",
    "GeneralMotsConfig",
    "BrillLindquistConfig",
    "prepare_ref_curve",
]


def find_mots(cfg, user_data=None, full_output=False, **kw):
    r"""Convenience function to prepare and perform a Newton-Kantorovich search.

    This function takes a *configuration* object `cfg`, which should be of a
    subclass of MotsFindingConfig. It then prepares an appropriate reference
    shape for the search and the initial "guess" (identical to the prepared
    reference shape) and calls the newton.newton_kantorovich() method.

    Additionally, a somewhat unique filename for the specified configuration
    is generated. If this file exists, its is loaded and returned instead of
    actually running a search. Otherwise, a generated result will be stored to
    this file by default.

    @param cfg
        Configuration object. Subclass of MotsFindingConfig.
    @param user_data
        Optional dictionary of custom data to write into the new curve's
        `user_data` dict. Note that convergence data stored under the keys
        ``"converged"``, ``"reason"``, ``"accurate_abs_delta"``, will override
        items supplied here. Also, if a curve is loaded from disk, this
        argument is ignored.
    @param full_output
        If `True`, return the curve and the filename it is stored under.
        Default is `False`.
    @param **kw
        Any keyword arguments take precedence over the configuration objects'
        settings. See the MotsFindingConfig documentation for the possible
        arguments.

    @return The computed or loaded curve object
        (curve.refparamcurve.RefParamCurve).
    """
    # version prefix updated when the algorithm changes to differentiate
    # different results
    v = 'v8_exp_reparam'
    if kw.pop('get_version', False):
        return v
    cfg = cfg.copy().update(**kw)
    if cfg.v is None:
        cfg.v = v
    cfg.verify_configuration()
    fname = cfg.get_fname() # None in case we shouldn't save/load
    if fname and not cfg.recompute and op.isfile(fname):
        curve = BaseCurve.load(fname)
        return (curve, fname) if full_output else curve
    if cfg.dont_compute:
        raise FileNotFoundError("File: %s" % fname)
    c0 = _prepare_initial_curve(cfg)
    newton_args = cfg.newton_args.copy()
    ref_curves = (
        [(c0.ref_curve, "ref", "--k")] + newton_args.get('reference_curves', [])
    )
    newton_args['reference_curves'] = ref_curves
    if cfg.accurate_test_res == "auto":
        cfg.accurate_test_res = max(500, 1.9*getattr(cfg.c_ref, 'num', 500))
    try:
        curve = newton_kantorovich(
            c0, accurate_test_res=cfg.accurate_test_res,
            user_data=user_data or dict(),
            **newton_args
        )
    except NoConvergence as e:
        if not cfg.save_failed_curve:
            raise
        if cfg.newton_args.get('verbose', False):
            print("Did not converge: %s" % (e,))
        curve = None
    if cfg.callback:
        cfg.callback(curve)
    if fname and not cfg.veto(curve, cfg):
        if curve:
            curve.save(fname, verbose=cfg.save_verbose,
                       overwrite=cfg.recompute,
                       msg="res:%s" % curve.num)
        elif cfg.save_failed_curve:
            save_to_file(
                fname, None, showname="'None'", verbose=cfg.save_verbose
            )
    return (curve, fname) if full_output else curve


def _prepare_initial_curve(cfg):
    r"""Create the initial guess curve based on the given configuration."""
    c_ref = prepare_ref_curve(cfg)
    num = cfg.num
    if num == "auto":
        num = cfg.c_ref.num
    c0 = RefParamCurve.from_curve(
        c_ref, offset_coeffs=cfg.offset_coeffs, num=num,
        metric=cfg.get_metric(), name=cfg.name
    )
    return c0


def prepare_ref_curve(cfg):
    r"""Prepare the reference curve based on the given configuration.

    If the initial reference curve `cfg.c_ref` is just a numeric value, it is
    interpreted as (coordinate) radius of a perfect circle/sphere to use as
    reference shape.

    Otherwise, the shape is converted to a
    curve.parametriccurve.ParametricCurve and optionally reparameterized using
    the stored settings.
    """
    verbose = cfg.newton_args.get('verbose', False)
    metric = cfg.get_metric()
    if isinstance(cfg.c_ref, numbers.Number):
        return StarShapedCurve.create_sphere(
            radius=cfg.c_ref, num=1, metric=metric,
        )
    if isinstance(cfg.c_ref, (list, tuple)):
        radius, origin = cfg.c_ref
        return StarShapedCurve.create_sphere(
            radius=radius, num=1, metric=metric, origin=origin
        )
    ref_num = cfg.ref_num
    reparam = cfg.reparam
    with_metric = metric if cfg.reparam_with_metric else None
    if ref_num and reparam:
        if cfg.bipolar_ref_curve:
            bipolar_kw = cfg.bipolar_kw or dict()
            c_ref = BipolarCurve.from_curve(
                cfg.c_ref, num=cfg.c_ref.num, **bipolar_kw,
            )
            if verbose:
                print("Using bipolar coordinates with settings: %s" % bipolar_kw)
        else:
            c_ref = ParametricCurve.from_curve(cfg.c_ref, num=cfg.c_ref.num)
        strategy = "arc_length"
        opts = dict()
        if isinstance(reparam, (list, tuple)):
            strategy, opts = reparam
            if isinstance(strategy, numbers.Number): # backwards compatibility
                alpha, beta = strategy, opts
                strategy = "experimental"
                opts = dict(alpha=alpha, beta=beta)
        if isinstance(reparam, str):
            strategy = reparam
        if isinstance(cfg.reparam_with_metric, numbers.Number):
            blend = cfg.reparam_with_metric
            if blend and blend != 1.0 and 'blend' not in opts:
                opts['blend'] = blend
        opts = insert_missing(opts, num=ref_num)
        if verbose:
            metric_info = "curved space" if with_metric else "flat space"
            print("Reparameterization strategy: %s (%s)" % (strategy, metric_info))
            print("Reparameterization options:  %s" % opts)
        with raise_all_warnings(),\
                timethis("Reparameterizing reference curve...",
                         " done after {}", eol=False, silent=not verbose):
            c_ref.reparameterize(
                strategy=strategy,
                **insert_missing(opts, metric=with_metric)
            )
    else:
        if ref_num is None:
            c_ref = cfg.c_ref
        else:
            c_ref = ParametricCurve.from_curve(cfg.c_ref, num=ref_num)
    return c_ref


@add_metaclass(ABCMeta)
class MotsFindingConfig():
    r"""Abstract configuration base class for find_mots().

    This stores all the settings for the find_mots() runs.

    The folder to store results in is constructed as follows:

        ./{base_folder}/{v}/{folder}/{prefix}{cfg_str}_n{num}_refn{ref_num}_{suffix}.npy

    The individual parts are described in the respective parameter docs of
    #__init__() except for `cfg_str` which is generated by the respective
    subclass and may e.g. indicate the configuration of the metric.
    """

    def __init__(self, **kw):
        r"""Initialize a new configuration object.

        @param name
            Name of the curve (used e.g. for labels in some plots).
            Default is `''`.
        @param c_ref
            Reference shape. Either a float, i.e. the radius of a circle to use,
            or any curve object which will be converted to a parametric curve.
            Must be set unless overridden via the `c_ref` kwarg to
            find_mots().
        @param num
            Initial resolution of the solution. The default ``'auto'`` will use
            the reference shape's resolution. Note that if `auto_resolution` is
            `True` (either due to its default value or by setting it explicitly
            here as extra search argument (see below)), this is likely not going
            to be the actual resolution of the solution, which may be higher or
            lower.
            Default is `'auto'`.
        @param ref_num
            Resolution of the prepared reference shape. Downsampling significantly
            (to e.g. `20..80`) can increase the attainable accuracy by a large
            amount.
            If `c_ref` is a float (i.e. a sphere is created), then `ref_num`
            is ignored. Otherwise, if not given, the reference shape is used
            as-is without reparameterization.
        @param recompute
            Whether to start the search even if a result file exists.
            Default is `False`.
        @param dont_compute
            If `True`, load results from files. If a file does not exist, no
            computation is started and a `FileNotFoundError` is raised.
            Default is `False`.
        @param save
            Boolean indicating whether to save/load results using files on disk.
            Default is `False`.
        @param save_verbose
            Whether to print a message containing the filename when saving a
            curve.
            Default is `True`.
        @param save_failed_curve
            Whether to save a dummy file to indicate a curve was searched but not
            found.
            Default is `False`.
        @param offset_coeffs
            Coefficients used as initial offset from the reference curve. See
            curve.refparamcurve.RefParamCurve.from_curve().
            Default is the empty list.
        @param reparam
            Whether and how to reparameterize the reference shape. Possible values
            are:
            \code{.unparsed}
                False   # don't reparameterize
                True    # reparameterize based on arc-length
                2-tuple # custom reparameterization (see below)
            \endcode
            In case of a 2-tuple, the first element defines the strategy as a
            string and the second can be a dictionary with arguments. Possible
            strategies are those supported by
            curve.parametriccurve.ParametricCurve.reparameterize().
            For backwards compatibility, the 2-tuple can also consist of two
            floats. This chooses the ``"experimental"`` strategy and sets
            `alpha` and `beta` to the two values, respectively.
            Default is `True`.
        @param reparam_with_metric
            Whether to reparameterize in curved (`True`) or flat (`False`) space.
            Default is `True`.
        @param bipolar_ref_curve
            Parameterize the reference curve in bipolar coordinates using a
            .curve.bipolarcurve.BipolarCurve. Default is `False`, i.e. to
            parameterize in cartesian coordinates using a
            .curve.parametriccurve.ParametricCurve.
        @param bipolar_kw
            Optional configuration options for the bipolar reference curve
            parameterization (i.e. constructor arguments `scale` and `move`).
        @param accurate_test_res
            Resolution for performing the accurate convergence tests. The default
            ``'auto'`` will use `1.9` times the resolution of the (unprepared)
            reference shape but at least `500`.
            Default is ``'auto'``.
        @param callback
            Optional function called after finding the MOTS but before storing
            it. This may be used to save additional information about the MOTS
            in the curve object before it is saved to disk. It is called as
            ``callback(curve)``, where `curve` may be `None` if no curve was
            found.
        @param veto_callback
            Optional callback to veto saving a result to file. It is called as
            ``veto_callback(curve, cfg)``. If this function returns `False`,
            no veto is issued and the curve is saved. A return value of `True`
            raises a veto and the curve is not stored to disk.
        @param prefix
            Prefix string for the filenames.
            Default is ``''``.
        @param suffix
            Suffix string for the filenames.
            Default is ``''``.
        @param folder
            Name of the folder to put result files in.
            Default is ``''``.
        @param base_folder
            Name of the base folder to store result files in.
            Default is ``'data/'``.
        @param v
            Optional override for the version string indicating the current
            algorithm in the folder name to store results in. This is usually set
            by a hard-coded variable in find_mots() and is changed when e.g. the
            reparameterization formula changes to avoid accidentally mixing
            results of different strategies.
        @param simple_name
            If `True`, the auto-generated filename contains less info. Otherwise
            (default), it will contain the resolution of the curve and of the
            reference curve, as well as whether the reference curve has been
            reparameterized.
        @param **kw
            Further keyword arguments are passed directly to the
            .newton.newton_kantorovich() call. Useful settings include
            `auto_resolution`, `max_resolution`, `step_mult`, `verbose`, and
            others. Those not documented elsewhere are documented as public
            class attributes of .newton.NewtonKantorovich.
        """
        self.name = ''
        self.c_ref = None
        self.num = 'auto'
        self.ref_num = None
        self.recompute = False
        self.dont_compute = False
        self.save = False
        self.save_verbose = True
        self.save_failed_curve = False
        self.offset_coeffs = ()
        self.reparam = True
        self.reparam_with_metric = True
        self.bipolar_ref_curve = False
        self.bipolar_kw = None
        self.accurate_test_res = 'auto'
        self.callback = None
        self.veto_callback = None
        self.prefix = ''
        self.suffix = ''
        self.folder = ''
        self.base_folder = 'data/'
        self.v = None
        self.simple_name = False
        self.newton_args = dict()
        self.update(**kw)
        self._init_done = True

    @classmethod
    def from_curve(cls, curve, **kw):
        r"""Copy the metric settings from the given curve.

        All other settings will be at their defaults except for those
        specified via the ``**kw`` arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def get_metric(self):
        r"""Construct/get a metric from the specified settings."""
        pass

    @abstractmethod
    def config_str(self):
        r"""String indicating e.g. the configuration of the metric.

        Used in the filename of generated results.
        """
        pass

    def get_fname(self):
        r"""Generate a filename for saving/loading results."""
        if not self.save:
            return None
        prefix = self.prefix
        suffix = self.suffix
        if prefix and not prefix.endswith("_"):
            prefix = "%s_" % prefix
        if suffix and not suffix.startswith("_"):
            suffix = "_%s" % suffix
        folder = self.folder
        if folder and not folder.endswith('/'):
            folder += '/'
        if self.simple_name:
            fname = "{base}/{folder}{prefix}{cfg}{suffix}.npy"
        else:
            if not isinstance(self.c_ref, (numbers.Number, list, tuple)):
                suffix = "_refn%d%s" % (
                    self.ref_num if self.ref_num else self.c_ref.num,
                    suffix
                )
                if self.ref_num and self.reparam:
                    suffix += "_reparam"
            fname = "{base}/{v}/{folder}{prefix}{cfg}_n{num}{suffix}.npy"
        fmt = dict(base=self.base_folder, v=self.v, folder=folder, prefix=prefix,
                   cfg=self.config_str(), num=self.num, suffix=suffix)
        return fname.format(**fmt)

    def get_out_dir(self):
        r"""Return the directory into which results should be stored."""
        fname = self.get_fname()
        if fname is None:
            return None
        return op.dirname(fname)

    def veto(self, curve, cfg):
        r"""Called after a curve is found to check whether to store it.

        This is called with the solution `curve` and current configuration
        `cfg` prior to saving the curve. If we return a `True` value here, the
        curve is not saved to disk.

        The actual veto callback being executed is set as `veto_callback`
        configuration option. If given, it should be a callable accepting the
        two arguments ``curve, cfg``. By default, no veto is issued.
        """
        if self.veto_callback:
            return self.veto_callback(curve, cfg)
        return False

    def __setattr__(self, attr, value):
        if attr.startswith('_') or not getattr(self, '_init_done', False):
            super(MotsFindingConfig, self).__setattr__(attr, value)
        else:
            self.update(**{attr: value})

    def update(self, **kw):
        r"""Replace settings with the given parameters.

        Any setting not recognized as option for find_mots() is stored as
        extra argument for the newton_kantorovich() call.
        """
        for k, v in kw.items():
            if hasattr(self, k):
                super(MotsFindingConfig, self).__setattr__(k, v)
            else:
                self.newton_args[k] = v
        return self

    def copy(self):
        r"""Create a (shallow) independent copy of the current settings.

        The settings in the copy may be modified independently of the original
        settings. The extra arguments for the newton_kantorovich() call are
        also copied, so that these may be modified independently too.
        """
        instance = type(self).__new__(type(self))
        instance.__dict__ = self.__dict__.copy()
        instance.newton_args = self.newton_args.copy()
        return instance

    def _asdict(self):
        r"""Return a dictionary version of the current settings."""
        return OrderedDict([(k, v) for k, v in self.__dict__.items()
                            if not k.startswith("_")])

    def __repr__(self):
        r"""Represent the current settings as a string."""
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join("%s=%r" % (k, v) for k, v in
                                     self._asdict().items()))

    def verify_configuration(self):
        r"""Check that the given configuration makes some sense."""
        if not self.save and self.dont_compute:
            raise ValueError("Option `dont_compute` cannot be used with "
                             "`save==False`.")
        if self.recompute and self.dont_compute:
            raise ValueError("Options `recompute` and `dont_compute` are "
                             "mutually exclusive.")


class GeneralMotsConfig(MotsFindingConfig):
    r"""General purpose MOTS finding configuration settings.

    These settings are applicable to both analytical (e.g. Brill-Lindquist)
    metrics and numerical discrete data. You have to supply the actual metric
    to consider prior to or while calling find_mots(). From the metric itself,
    any extrinsic curvature will be taken and used to compute the expansion
    and linearised expansion equation.

    While using the MOTS finder, some combinations of the many available
    configuration options have turned out to produce good results. The most
    important ones have been collected in so-called *presets*, which can be
    used via the preset() class method.
    """

    def __init__(self, metric=None, fname_desc='general', **kw):
        r"""Create a new configuration with default settings.

        By default, no metric is specified. However, before calling
        find_mots(), a metric should be set using the update() method or using
        a keyword argument in the find_mots() call itself.

        @param metric
            The slice's metric to use. The extrinsic curvature will be
            obtained from this metric.
        @param fname_desc
            Short string used to construct filenames for resulting curves.
        @param **kw
            Additional arguments set the specific options described in
            MotsFindingConfig.__init__() and .newton.newton_kantorovich().
            Further options not explained there are documented as public class
            attributes of .newton.NewtonKantorovich.
        """
        self.metric = metric
        self.fname_desc = fname_desc
        super(GeneralMotsConfig, self).__init__(**kw)

    @classmethod
    def from_curve(cls, curve, **kw):
        r"""Create default settings taking just the metric of the given curve.

        The optional keyword arguments can be used to set all the remaining
        options.
        """
        g = curve.metric
        return cls(**insert_missing(kw, metric=g))

    def get_metric(self):
        if self.metric is None:
            raise RuntimeError("Metric not specified for run.")
        return self.metric

    def config_str(self):
        return self.fname_desc

    @classmethod
    def preset(cls, preset, hname, out_folder=None, **kw):
        r"""Create a configuration based on a manually crafted preset.

        The currently available presets are:
            * **discrete1**:
                Fixed resolution with no accuracy detection and flat space
                reparameterization of the reference curve. This provides
                relatively fast results with no control over accuracy.
                Probably rarely used in practice.
            * **discrete2**:
                Best option if the roundoff plateau is not known and the curve
                is roughly round (see `discrete3` for highly distorted MOTSs).
                Accuracy is optimized by increasing the resolution until the
                roundoff plateau is detected. Results are downsampled as long
                as the residual error does not increase significantly, even if
                this residual error lies above the absolute tolerance.
            * **discrete3**:
                Best option if the accuracy that can be obtained (i.e. where
                the plateau lies) is known at call time. As with `discrete2`,
                this is for mostly round curves. The analogue for distorted
                MOTSs is `discrete5`. For best results, set the `atol`
                tolerance slightly above this plateau level. The maximum
                resolution is set to 12000 and a found curve is downsampled
                more aggressively as long as `atol` is not surpassed by the
                error. However, no downsampling is allowed if the `atol`
                setting would be violated. Hence, setting `atol` too low will
                push the resolution up to the maximum of 12000 within just a
                few steps.
            * **discrete4**:
                As `discrete2`, but using bipolar coordinates to parameterize
                the reference curve. Depending on the scaling of the
                coordinate system, a much higher density of collocation points
                is obtained in the very distorted "neck" region.
            * **discrete5**:
                This is to `discrete3` what `discrete4` is to `discrete2`
                (i.e. bipolar coordinates for the reference curve in case the
                plateau is known and `atol` can be specified).
            * **discrete6**:
                As `discrete4`, but with dynamically optimized ``'curv2'``
                reparameterization in s-z-coordinate space.
            * **discrete7**:
                As `discrete5`, but with dynamically optimized ``'curv2'``
                reparameterization in s-z-coordinate space.
            * **discrete8**:
                As `discrete2`, but better suited for Hermite interpolation.
                This preset allows a maximum single-step resolution increase
                of 100 while demanding at least a factor of 2 lower residual
                error. This means that plateau detection is triggered slightly
                easier (which we need for Hermite interpolation, as there the
                hard plateau we get for Lagrange interpolation does not appear
                and instead a very slow sub-exponential convergence up to
                floating point errors is observed). Furthermore, plateau
                detection is supported by *knee detection* in the coefficient
                list of the horizon function. This should prevent the worst
                cases of overfitting.
            * **discrete9**:
                A combination of `discrete3` (i.e. for known plateau) with the
                easier palteau detection of `discrete8`.
            * **discrete10**:
                A combination of `discrete6` with the easier palteau detection
                of `discrete8`. Here, the maximum single-step resolution
                increase is set to 400.
            * **discrete11**:
                A combination of `discrete7` (i.e. for known plateau) with the
                easier palteau detection of `discrete8`. Here, the maximum
                single-step resolution increase is set to 800.

        In practice, for Lagrange interpolation, ``"discrete2"`` has turned
        out to be a well rounded configuration in all but the very extreme
        cases. Analogously, ``"discrete6"`` or ``'discrete7'`` (for known
        plateau) may be good choices for the inner common MOTS.

        For Hermite interpolation, ``"discrete8"`` should be preferred for the
        general case and ``"discrete10"`` or ``"discrete11"`` (for known
        plateau) for the distorted inner common MOTS.

        @param preset
            One of the above preset names.
        @param hname
            Name given to the horizon to find (e.g. ``'AH'``). This is used as
            a filename prefix.
        @param out_folder
            Optional folder to store curves in. By default, curves are not
            stored. If this folder is specified (and neither `save`,
            `base_folder`, or `folder` is given as extra keyword arg), saving
            of curves is activated and files are stored there.
        @param **kw
            Extra options overriding those in the preset. These are documented
            in MotsFindingConfig.__init__() and .newton.newton_kantorovich().
            Further options not explained there are documented as public class
            attributes of .newton.NewtonKantorovich.
        """
        def _hermite_settings(**kw):
            return insert_missing(
                kw, max_res_increase=100, plateau_tol=2.0,
                knee_detection=True,
            )
        if preset == "discrete1":
            cfg = cls(
                ref_num=5, num=30, atol=1e-8,
                reparam=True, reparam_with_metric=False,
                auto_resolution=False, accurate_test_res=None, prefix=hname,
                save_failed_curve=True, verbose=True, v=''
            )
        elif preset == "discrete2":
            cfg = cls(
                atol=1e-8, reparam="curv2", reparam_with_metric=False,
                disp=False, auto_resolution=True, max_resolution=8000,
                fake_steps=0, detect_plateau=True, plateau_tol=1.5,
                min_res=30,
                res_increase_factor=1.4,
                min_res_increase_factor=1.333, # allow 6000 -> 8000
                res_init_factor=1,
                linear_regime_res_factor=1,
                res_decrease_factor=0.75, res_decrease_accel=0.9,
                downsampling_tol_factor=1.5, liberal_downsampling=True,
                accurate_test_res="twice", prefix=hname,
                save_failed_curve=True, verbose=True, simple_name=True,
            )
        elif preset == "discrete3":
            # Strategy for high resolution situations with known plateau.
            # Absolute tolerance should be specified correctly for this to
            # work best.
            cfg = cls.preset(
                "discrete2", hname,
                max_resolution=12000,
                min_res_increase_factor=None,
                res_increase_factor=1.2,
                res_decrease_factor=0.9,
                downsampling_tol_factor=2.0,
                liberal_downsampling=False,
            )
        elif preset == "discrete4":
            cfg = cls.preset(
                "discrete2", hname,
                reparam=("arc_length", dict(coord_space=True)),
                reparam_with_metric=False,
                bipolar_ref_curve=True,
            )
        elif preset == "discrete5":
            cfg = cls.preset(
                "discrete4", hname,
                max_resolution=12000,
                min_res_increase_factor=None,
                res_increase_factor=1.2,
                res_decrease_factor=0.9,
                downsampling_tol_factor=2.0,
                liberal_downsampling=False,
            )
        elif preset == "discrete6":
            cfg = cls.preset(
                "discrete4", hname,
                reparam=("curv2", dict(coord_space=True)),
            )
        elif preset == "discrete7":
            cfg = cls.preset(
                "discrete5", hname,
                reparam=("curv2", dict(coord_space=True)),
            )
        elif preset == "discrete8":
            cfg = cls.preset(
                "discrete2", hname, **_hermite_settings()
            )
        elif preset == "discrete9":
            cfg = cls.preset(
                "discrete3", hname, **_hermite_settings()
            )
        elif preset == "discrete10":
            cfg = cls.preset(
                "discrete6", hname, **_hermite_settings(max_res_increase=400)
            )
        elif preset == "discrete11":
            cfg = cls.preset(
                "discrete7", hname, **_hermite_settings(max_res_increase=800)
            )
        else:
            raise ValueError("Unknown preset: %s" % preset)
        if out_folder:
            cfg.update(save=True, base_folder=out_folder, folder="",
                       simple_name=True)
        cfg.update(fname_desc=preset)
        cfg.update(**kw)
        return cfg


class BrillLindquistConfig(MotsFindingConfig):
    r"""Configuration for finding MOTSs in Brill-Lindquist metrics.

    This is a specialization of the MotsFindingConfig for Brill-Lindquist
    configurations. By specifying a distance parameter `d` and the two masses,
    the metric can be created dynamically.

    See the documentation of MotsFindingConfig for a detailed description of
    all available options.
    """

    def __init__(self, d, m1=1, m2=1, **kw):
        r"""Create a Brill-Lindquist find_mots() configuration object.

        This shares all the parameters of MotsFindingConfig (see there for the
        documentation). The additional parameters are described next.

        @param d
            Distance parameter for the Brill-Lindquist metric.
        @param m1
            Mass parameter (i.e. not physical/irreducible mass) of the black
            hole with puncture on the positive z-axis.
        @param m2
            As `m1` but for the puncture on the negative z-axis.
        @param **kw
            Arguments shared with MotsFindingConfig.
        """
        self.d = d
        self.m1 = m1
        self.m2 = m2
        super(BrillLindquistConfig, self).__init__(**kw)

    @classmethod
    def from_curve(cls, curve, **kw):
        g = curve.metric
        return cls(**insert_missing(kw, d=g.d, m1=g.m1, m2=g.m2))

    def get_metric(self):
        return BrillLindquistMetric(d=self.d, m1=self.m1, m2=self.m2,
                                    axis='z')

    def config_str(self):
        return "d%s_m%s_%s" % (self.d, self.m1, self.m2)
