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

from ..utils import insert_missing
from ..numutils import raise_all_warnings
from ..metric import BrillLindquistMetric
from ..exprs.numexpr import save_to_file
from .curve import StarShapedCurve, RefParamCurve, ParametricCurve, BaseCurve
from .newton import newton_kantorovich, NoConvergence


__all__ = [
    "find_mots",
    "GeneralMotsConfig",
    "BrillLindquistConfig",
]


def find_mots(cfg, **kw):
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
    fname = _get_fname(cfg) # None in case we shouldn't save/load
    if fname and not cfg.recompute and op.isfile(fname):
        return BaseCurve.load(fname)
    if cfg.dont_compute:
        raise FileNotFoundError("File: %s" % fname)
    c0 = _prepare_initial_curve(cfg)
    if cfg.accurate_test_res == "auto":
        cfg.accurate_test_res = max(500, 1.9*getattr(cfg.c_ref, 'num', 500))
    try:
        curve = newton_kantorovich(c0, **cfg.newton_args)
    except NoConvergence:
        if not cfg.save_failed_curve:
            raise
        curve = None
    if fname and not cfg.veto(curve, cfg):
        if curve:
            curve.save(fname, verbose=cfg.save_verbose,
                       msg="res:%s" % curve.num)
        elif cfg.save_failed_curve:
            save_to_file(
                fname, None, showname="'None'", verbose=cfg.save_verbose
            )
    return curve


def _get_fname(cfg):
    r"""Generate a (somewhat) unique filename for saving/loading the result.

    This is used by find_mots() to save/load the resulting curves.
    """
    if not cfg.save:
        return None
    prefix = cfg.prefix
    suffix = cfg.suffix
    if prefix and not prefix.endswith("_"):
        prefix = "%s_" % prefix
    if suffix and not suffix.startswith("_"):
        suffix = "_%s" % suffix
    if not isinstance(cfg.c_ref, (numbers.Number, list, tuple)):
        suffix = "_refn%d%s" % (cfg.ref_num if cfg.ref_num else cfg.c_ref.num,
                                suffix)
        if cfg.ref_num and cfg.reparam:
            suffix += "_reparam"
    folder = cfg.folder
    if folder and not folder.endswith('/'):
        folder += '/'
    fname = ("%s/%s/%s%s%s_n%s%s.npy"
             % (cfg.base_folder, cfg.v, folder, prefix, cfg.config_str(),
                cfg.num, suffix))
    return fname


def _prepare_initial_curve(cfg):
    r"""Create the initial guess curve based on the given configuration."""
    c_ref = _prepare_ref_curve(cfg)
    num = cfg.num
    if num == "auto":
        num = cfg.c_ref.num
    c0 = RefParamCurve.from_curve(
        c_ref, offset_coeffs=cfg.offset_coeffs, num=num,
        metric=cfg.get_metric(),
        extr_curvature=cfg.get_extr_curvature(),
        name=cfg.name
    )
    return c0


def _prepare_ref_curve(cfg):
    r"""Prepare the reference curve based on the given configuration.

    If the initial reference curve `cfg.c_ref` is just a numeric value, it is
    interpreted as (coordinate) radius of a perfect circle/sphere to use as
    reference shape.

    Otherwise, the shape is converted to a
    curve.parametriccurve.ParametricCurve and optionally reparameterized using
    the stored settings.
    """
    metric = cfg.get_metric()
    curv = cfg.get_extr_curvature()
    if isinstance(cfg.c_ref, numbers.Number):
        return StarShapedCurve.create_sphere(
            radius=cfg.c_ref, num=1, metric=metric, extr_curvature=curv,
        )
    if isinstance(cfg.c_ref, (list, tuple)):
        radius, origin = cfg.c_ref
        return StarShapedCurve.create_sphere(
            radius=radius, num=1, metric=metric, extr_curvature=curv,
            origin=origin
        )
    ref_num = cfg.ref_num
    reparam = cfg.reparam
    with_metric = metric if cfg.reparam_with_metric else None
    if ref_num and reparam:
        c_ref = ParametricCurve.from_curve(cfg.c_ref, num=cfg.c_ref.num)
        with raise_all_warnings():
            blend = 1.0
            if isinstance(cfg.reparam_with_metric, numbers.Number):
                blend = cfg.reparam_with_metric
            if isinstance(reparam, (list, tuple)):
                alpha, beta = reparam
                c_ref.reparameterize_by_curvature(
                    metric=with_metric, alpha=alpha, beta=beta,
                    num=ref_num, blend=blend
                )
            else:
                c_ref.reparameterize(metric=with_metric, num=ref_num,
                                     blend=blend)
    else:
        if ref_num is None:
            c_ref = cfg.c_ref
        else:
            c_ref = ParametricCurve.from_curve(cfg.c_ref, num=ref_num)
    return c_ref


@add_metaclass(ABCMeta)
class MotsFindingConfig():
    r"""Configuration class for find_mots().

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
                2-tuple # reparameterize using experimental formula
            \endcode
            In the last case, the two elements are taken as `alpha` and `beta`
            parameters for the `reparameterize_by_curvature()` call.
            Default is `True`.
        @param reparam_with_metric
            Whether to reparameterize in curved (`True`) or flat (`False`) space.
            Default is `True`.
        @param accurate_test_res
            Resolution for performing the accurate convergence tests. The default
            ``'auto'`` will use `1.9` times the resolution of the (unprepared)
            reference shape but at least `500`.
            Default is ``'auto'``.
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
        @param **kw
            Further keyword arguments are passed directly to the
            newton_kantorovich() call. Useful settings include `auto_resolution`,
            `max_resolution`, `step_mult`, `verbose`, and others.
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
        self.accurate_test_res = 'auto'
        self.veto_callback = None
        self.prefix = ''
        self.suffix = ''
        self.folder = ''
        self.base_folder = 'data/'
        self.v = None
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

    def get_extr_curvature(self):
        r"""Construct/get the extrinsic curvature from the specified settings."""
        return None

    @abstractmethod
    def config_str(self):
        r"""String indicating e.g. the configuration of the metric.

        Used in the filename of generated results.
        """
        pass

    def veto(self, curve, cfg):
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
    def __init__(self, metric, extr_curvature=None, fname_desc='general', **kw):
        self.metric = metric
        self.extr_curvature = extr_curvature
        self.fname_desc = fname_desc
        super(GeneralMotsConfig, self).__init__(**kw)

    @classmethod
    def from_curve(cls, curve, **kw):
        g = curve.metric
        K = curve.extr_curvature
        return cls(**insert_missing(kw, metric=g, extr_curvature=K))

    def get_metric(self):
        return self.metric

    def get_extr_curvature(self):
        return self.extr_curvature

    def config_str(self):
        return self.fname_desc


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
