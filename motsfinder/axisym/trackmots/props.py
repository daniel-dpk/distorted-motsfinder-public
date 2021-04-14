r"""@package motsfinder.axisym.trackmots.props

Functions to compute and store physical properties of MOTS curves.

The functions in this module can be used to compute various properties of
MOTSs like their area, signature, stability spectrum, etc. The actual
calculation happens inside the curve classes themselves, but the methods here
also store and update the stored results inside the curves' `user_data`
dictionary.
"""

import os.path as op

from scipy.integrate import IntegrationWarning
import numpy as np

from ...utils import timethis, find_files, update_user_data_in_file
from ...numutils import inf_norm1d, clip, IntegrationResults
from ...numutils import IntegrationResult
from ...numutils import try_quad_tolerances, _fixed_quad
from ...exprs.cheby import Cheby
from ..utils import _replace_metric
from ..curve import BaseCurve
from ..curve.expcurve import interpret_basis_class


__all__ = [
    "compute_props",
    "compute_tube_signature",
    "compute_tev_divergence",
    "compute_time_evolution_vector",
    "compute_shear_hat_scalar",
    "compute_shear_hat2_integral",
    "compute_xi_hat2_integral",
    "compute_xi_tilde2_integral",
    "compute_xi_vector",
    "compute_xi2_integral",
    "compute_xi_scalar",
    "compute_xi_hat_scalar",
    "compute_xi_tilde_scalar",
    "compute_surface_gravity",
    "compute_surface_gravity_tev",
    "compute_timescale_tau2",
    "compute_timescale_tau2_option01",
    "compute_timescale_tau2_option02",
    "compute_timescale_tau2_option03",
    "compute_timescale_tau2_option04",
    "compute_timescale_T2",
    "compute_extremality_parameter",
    "max_constraint_along_curve",
]


# Valid properties to compute.
ALL_PROPS = (None, 'none', 'all', 'length_maps', 'constraints', 'ingoing_exp',
             'avg_ingoing_exp', 'area', 'ricci', 'mean_curv', 'curv2',
             'shear', 'shear2_integral', 'shear_scalar', 'stability',
             'stability_null', 'stability_convergence',
             'stability_convergence_null', 'neck', 'dist_top_bot',
             'z_dist_top_inner', 'z_dist_bot_inner',
             'ricci_difference_integral', 'ricci_interp', 'point_dists', 'area_parts',
             'multipoles', 'zeta')


# Of the above, those not included in 'all'.
NEED_ACTIVATION_PROPS = (
    'stability_convergence',
    'stability_convergence_null',
    'ricci_difference_integral',
    'ricci_interp',
)


def compute_props(hname, c, props, arg_dict=None, verbosity=True,
                  MOTS_map=None, fname=None, remove_invalid=True,
                  update_file=False, **kwargs):
    r"""Compute properties and store them in the curve object.

    All computed properties will be stored under the property's name in the
    curve's ``user_data`` dictionary. The arguments used for this computation
    are additionally stored under the key ``prop_args``, where ``prop`` is the
    property name. Some properties produce extra output. This is then stored
    under ``prop_extras``.

    The properties that can be computed here are:
        * **length_maps**:
            Mappings from current parameterization to proper-length based
            parameterizations (see
            ..curve.basecurve.BaseCurve.proper_length_map()). The value will
            be a dictionary with elements `length_map`, `inv_map`,
            `proper_length`, where the latter is the length of the curve in
            curved space. Note that the two functions will actually be
            NumericExpression objects
            (motsfinder.exprs.numexpr.NumericExpression), so that
            ``evaluator()`` needs to be called to get actual callables.
        * **constraints**:
            Compute the Hamiltonian (scalar) and momentum (vector) constraints
            on grid points close to the curve. The stored value will have the
            structure ``dict(near_curve=dict(std=[x, ham, mom],
            proper=[x, ham, mom]))``. Here, `x` is the set of parameter values
            of the curve close to which the values are computed, `ham` are the
            Hamiltonian constraint values and `mom` the momentum ones. The
            difference between `std` and `proper` is that for `std`, the
            parameters are equidistant in curve parameter `lambda`, while for
            `proper` they are equidistant in the curve's proper length
            parameterization, though the stored `x` values are in the curve's
            current parameterization. This way, you can plot in the curve's
            proper length parameterization by taking the `x` values of `std`
            and values of `proper`.
        * **ingoing_exp**:
            Expansion of the ingoing null normals. This is computed on a grid
            of points (in curve parameter space) the density of which is
            determined by the curve's resolution.
        * **avg_ingoing_exp**:
            Average ingoing expansion, averaged over the MOTSs surface area.
        * **area**:
            Surface area of the MOTS. This produces extra data (as explained
            above) containing information about the convergence of the
            integral.
        * **ricci**:
            Ricci scalar computed on a grid of points with density dependent
            on the curve resolution. Additionally, the maximum absolute value
            of the Ricci scalar along the curve is found and stored. Inspect
            the actual data (dictionary) for details.
        * **ricci_interp**:
            Ricci scalar as series expansion into a Cosine series. The
            resolution of this representation is set to the resolution of the
            curve.
        * **mean_curv**:
            Similar to ``ricci``, but compute the trace of the extrinsic
            curvature of the MOTS in the slice instead.
        * **curv2**:
            Similar to ``mean_curv``, but compute the "square"
            \f$k_{AB}k^{AB}\f$ of the MOTS's extrinsic curvature.
        * **shear**:
            Compute the shear \f$\sigma_{AB}\f$ and \f$|\sigma|^2\f$. See also
            ..curve.expcurve.ExpansionCurve.shear(). We compute the tensor and
            the square at a set of Gauss-Chebyshev collocation points. We use
            two times the curve's resolution as number of points. The result
            is that the minimum point density is slightly higher than the
            curve's collocation points (which are equidistant in lambda).
            The result is hence suitable to accurately expand the quantities
            into a Chebyshev series expression. Note that the collocation
            points are reversed in order to be increasing from 0 to pi. To
            create a Chebyshev series, use e.g.:
            @code
                shear2_values = curve.user_data['shear']['shear2']
                f = Cheby([], domain=(0, np.pi))
                f.set_coefficients(shear2_values[::-1], lobatto=False,
                                   physical_space=True)
            @endcode
        * **shear2_integral**:
            Compute the integral of the shear squared
            \f$\sigma_{AB}\sigma^{AB}\f$ over the MOTS. See also
            ..curve.expcurve.ExpansionCurve.shear_square_integral().
        * **shear_scalar**:
            Expand the complex shear scalar into spin 2 weighted spherical
            harmonics. For this, the MOTS is parameterized using the invariant
            angle `zeta`. For more information, see
            ..curve.expcurve.ExpansionCurve.expand_shear_scalar().
        * **stability**, **stability_null**:
            Compute the spectrum of the stability operator. The number of
            eigenvalues returned depends on the curve's resolution, but at
            least 30 eigenvalues (by default) are computed. See also the
            `min_stability_values` parameter and the following property.
            The stored value is a tuple ``principal, spectrum0``,
            where `principal` is the principal eigenvalue and
            `spectrum0` is the spectrum of the ``m=0`` angular mode.
            The full computed spectrum including higher modes is
            stored in the ``stability_extras`` dictionary under the
            ``spectrum`` key. Use `m_terminate_index` to control how
            many angular modes to consider.
            The ***_null*** variant of this property computes the spectrum
            w.r.t. the past outward null normal \f$-k^\mu\f$ instead of the
            outward normal in the spatial slice.
        * **stability_convergence**, **stability_convergence_null**:
            In addition to computing the stability spectrum above, recompute
            it at (by default) ``0.2, 0.4, 0.6, 0.8, 0.9, 1.1`` times the
            resolution used for the ``"stability"`` property. This allows
            analyzing convergence of the individual eigenvalues.
            As above, the ***_null*** variant is for the past outward null
            normal \f$-k^\mu\f$.
        * **neck**:
            Find the neck based on various criteria and compute various
            properties at this point, like proper circumference or proper
            distance to the z-axis. Only computed for a MOTS with
            ``hname="inner"``.
        * **dist_top_bot**:
            Proper distance (along the z-axis) of the two individual MOTSs.
            Only computed for a MOTS with ``hname="bot"``.
        * **z_dist_top_inner**:
            Proper distance of the highest point on the z-axis of the inner
            common MOTS and the top individual MOTS. Only compute for a MOTS
            with ``hname="inner"``.
        * **z_dist_bot_inner**:
            Proper distance of the lowest point on the z-axis of the inner
            common MOTS and the bottom individual MOTS. Only compute for a
            MOTS with ``hname="inner"``.
        * **point_dists**:
            Coordinate distances of the collocation points along the curve.
            May be useful to compare against the spatial coordinate resolution
            of the numerical grid.
        * **area_parts**:
            In case ``hname="bot"`` or ``"top"`` and the top and bottom MOTSs
            intersect, compute separately the area of the non-intersecting
            part and the part lying inside the other MOTS.
            For ``hname="inner"``, compute the parts before and after the
            *neck* separately. In case it self-intersects, find all the
            intersection points of the MOTS with itself and the top and bottom
            MOTSs and compute the area of each section, including splitting at
            the neck.
            If a result is found (i.e. the property's value is not `None`),
            the stored value will be a numutils.IntegrationResults object. Its
            `info` dictionary contains all splitting points. Sorting these
            gives you all the intervals of which the areas were computed.
        * **ricci_difference_integral**:
            Compute the integral of the difference of Ricci scalars between
            the inner common MOTS and the individual ones. Only done for
            ``hname="inner"``. The result is stored as a dictionary with keys
            ``"value", "value1", "value2", "info1", "info2"``. Here,
            ``"value"`` stores the value of the integral and ``"valueX"`` the
            integrations of the portion before (``X=1``) and after (``X=2``)
            the neck. The respective ``"infoX"`` entries contain the
            parameters and values of the Ricci scalar along the inner and the
            two individual MOTSs (where ``X=1`` refers to the top and ``X=2``
            to the bottom MOTS).
        * **multipoles**:
            Compute the first 10 multipole moments of the MOTS. The first and
            second moments are computed numerically even though these have
            analytically known values of `sqrt(pi)` and `0`, respectively.
            This allows comparison of the integration results with known
            values.
        * **zeta**:
            Compute the invariant angle zeta. See
            ..curve.expcurve.ExpansionCurve.compute_zeta().

    Some properties are only computed for certain horizons. The horizon
    specific properties are:
        * for ``"top"``: `area_parts`
        * for ``"bot"``: `dist_top_bot`, `area_parts`
        * for ``"inner"``: `neck`, `z_dist_top_inner`, `z_dist_bot_inner`,
          `area_parts`, `ricci_difference_integral`

    @param hname
        Name of horizon. Determines which kinds of properties are being
        computed (see above).
    @param c
        Curve representing the MOTS in axisymmetry.
    @param props
        Sequence of properties to compute (see above for the possible values).
        Use the string ``"all"`` to compute all possible properties except
        ``"stability_convergence"``, ``"stability_convergence_null"``,
        ``"ricci_difference_integral"``, ``"ricci_interp"``.
    @param arg_dict
        Dictionary containing (optional) detailed configuration for each
        property. Properties not computed are ignored. For each computed
        property, this dictionary may contain a set of options (as dictionary)
        under the key of the property. For example:
        ``arg_dict=dict(stability=dict(rtol=1e-9), area=dict(limit=200))``.
        Note that there are (more or less) sensible default values chosen for
        many of the options. Their current values can be inspected in the
        source for the _prepare_arg_dict() function and other functions in the
        trackmots.props module.
    @param verbosity
        Whether to print progress information.
    @param MOTS_map
        Dictionary indicating from which runs auxiliary MOTSs should be loaded
        (used e.g. for the various distances).
    @param fname
        File name under which the MOTS is stored. This is used to infer the
        run name for finding auxiliary MOTSs (in case there is no entry in the
        `MOTS_map`) and for writing the file if ``update_file=True``.
    @param remove_invalid
        Whether to check for and remove invalid data prior to recomputing it.
        This only affects data that is specified to being computed. This may
        be useful for updating data for which updated methods have been
        developed and a check for validity can be done. Default is `True`.
    @param update_file
        Whether to immediately write any computed results into the curve's
        data file (in case ``use_user_data==True``). This is done in a way
        safe for being used when different processes work on different
        properties simultaneously, even across nodes in a compute cluster
        accessing the same file via a network file system. Note that the data
        in the supplied curve object will not reflect updates performed by
        other processes in the meantime. The file, however, will contain this
        data. You should hence not save the given curve to the file after
        calling this function, since you would potentially overwrite the
        updates of other processes.
    @param **kwargs
        Additional keyword arguments are passed to _prepare_arg_dict() for
        configuring default options.
    """
    if MOTS_map is None:
        MOTS_map = dict()
    arg_dict = _prepare_arg_dict(arg_dict, **kwargs)
    out_dir = run_name = None
    if update_file and not fname:
        raise ValueError("Cannot update file when no filename is given.")
    if fname:
        out_dir = op.dirname(fname)
        run_name = op.basename(op.dirname(out_dir))
    if not isinstance(props, (list, tuple)):
        props = [props]
    for p in props:
        if p not in ALL_PROPS:
            raise ValueError(
                "Invalid property to compute: %s. Valid properties are: %s"
                % (p, ",".join("%s" % pr for pr in ALL_PROPS))
            )
    data = c.user_data
    did_something = [False]
    def do_prop(p, func, args=None, **kw):
        r"""(Re)compute a given property using a given function.

        @param p
            Property name to compute. The curve's `user_data` will get
            keys ``{p}``, ``{p}_args``, and potentially ``{p}_extras``
            containing the data, defining arguments and any extra data
            returned.
        @param func
            Function to call to produce the result. Will be called as
            ``func(**args, **kw)``.
        @param args
            Argument dictionary uniquely defining the result.
        @param **kw
            Extra arguments not influencing the result or reproducible
            from context (e.g. the curve to operate on).
        """
        if p not in ALL_PROPS:
            raise RuntimeError("Unkown property: %s" % p)
        is_extra = p in NEED_ACTIVATION_PROPS
        do = p in props or ('all' in props and not is_extra)
        if not do:
            return
        if args is None:
            args = dict()
        args.update(arg_dict.get(p, {}))
        stored_args = args.copy()
        for arg in stored_args:
            if isinstance(stored_args[arg], BaseCurve):
                stored_args[arg] = stored_args[arg].loaded_from
        p_args = '%s_args' % p
        p_extras = '%s_extras' % p
        if remove_invalid and p in data and p_args in data and data[p_args] == stored_args:
            if not _data_is_valid(p, data[p], data[p_args], hname):
                print("Removing invalid '%s' data from curve to trigger "
                      "recomputation." % p)
                # We don't save the file just now (for update_file) as data[p]
                # will be recomputed anyway.
                del data[p]
        if p not in data or p_args not in data or data[p_args] != stored_args:
            msg = "Computing property %-21s" % ("'%s'..." % p)
            with timethis(msg, " elapsed: {}", silent=verbosity == 0, eol=False):
                try:
                    result = func(**args, **kw)
                except AuxResultMissing:
                    if verbosity > 0:
                        print(" [cancelled due to missing data]", end="")
                    return
            updates = {p: result, p_args: stored_args}
            keys_to_remove = []
            if isinstance(result, _PropResult):
                updates[p] = result.result
                if result.extras:
                    updates[p_extras] = result.extras
                else:
                    keys_to_remove.append(p_extras)
            data.update(updates)
            for key in keys_to_remove:
                data.pop(key, None)
            if update_file:
                update_user_data_in_file(
                    fname=fname, data=updates,
                    keys_to_remove=keys_to_remove,
                )
            did_something[0] = True
    if hname == 'bot' and out_dir:
        do_prop('dist_top_bot', _dist_top_bot,
                dict(other_run=MOTS_map.get('top', run_name),
                     other_hname='top'),
                c=c, out_dir=out_dir)
    do_prop('zeta', _zeta, c=c)
    do_prop('length_maps', _length_maps, c=c)
    do_prop('constraints', _constraints, c=c)
    do_prop('area', _area, dict(v=1), c=c)
    do_prop('ingoing_exp', _ingoing_exp, c=c)
    do_prop('avg_ingoing_exp', _avg_ingoing_exp, c=c)
    for prop in ["stability", "stability_null", "stability_convergence",
                 "stability_convergence_null"]:
        if c.num >= arg_dict[prop]["min_num"]:
            # This code ensures that the stability spectrum is not recomputed
            # unnecessarily (i.e. when the minimum `min_num` has no effect because
            # the curve has a higher resolution anyway). Since the property is
            # recomputed on any argument changes, we set it to its previous
            # default value in these cases so that it matches the previously
            # stored value and does not trigger recomputation.
            arg_dict[prop]["min_num"] = 30 # previous default, has no effect here
    do_prop('stability', _stability, dict(v=2), slice_normal=True, c=c)
    do_prop('stability_null', _stability, dict(v=2), slice_normal=False, c=c)
    do_prop('stability_convergence', _stability_convergence, dict(v=2),
            slice_normal=True, c=c, verbose=verbosity > 0)
    do_prop('stability_convergence_null', _stability_convergence, dict(v=2),
            slice_normal=False, c=c, verbose=verbosity > 0)
    do_prop('ricci', _ricci, c=c)
    do_prop('ricci_interp', _ricci_interp, c=c)
    do_prop('mean_curv', _mean_curv, c=c)
    do_prop('curv2', _curv_squared, c=c)
    do_prop('shear', _shear, dict(v=2), c=c)
    do_prop('shear2_integral', _shear2_integral, c=c)
    do_prop('shear_scalar', _shear_scalar, c=c)
    do_prop('point_dists', _point_dists, c=c)
    if hname == 'inner':
        do_prop('neck', _neck, c=c)
        for other_hname, where in zip(['top', 'bot'], ['top', 'bottom']):
            do_prop('z_dist_%s_inner' % other_hname, _z_dist_to_inner,
                    dict(other_run=MOTS_map.get(other_hname, run_name),
                         other_hname=other_hname),
                    c=c, out_dir=out_dir)
        do_prop('area_parts', _area_parts_inner,
                dict(v=1,
                     top_run=MOTS_map.get('top', run_name),
                     bot_run=MOTS_map.get('bot', run_name)),
                c=c, out_dir=out_dir)
        do_prop('ricci_difference_integral', _ricci_difference_integral,
                dict(v=1,
                     top_run=MOTS_map.get('top', run_name),
                     bot_run=MOTS_map.get('bot', run_name)),
                c=c, out_dir=out_dir)
    if hname in ('top', 'bot'):
        other_hname = 'bot' if hname == 'top' else 'top'
        do_prop('area_parts', _area_parts_individual,
                dict(v=2,
                     other_run=MOTS_map.get(other_hname, run_name),
                     other_hname=other_hname),
                c=c, out_dir=out_dir)
    do_prop('multipoles', _multipoles, dict(v=1), c=c)
    return did_something[0]


def _prepare_arg_dict(arg_dict, area_rtol=1e-6, min_stability_values=30,
                      m_terminate_index=30, max_multipole_n=10,
                      stability_convergence_factors=(0.2, 0.4, 0.6, 0.8, 0.9, 1.1),
                      shear_max_lmax=None):
    r"""Prepare default arguments for all computable properties.

    Note that any existing values of dictionaries contained in `arg_dict` have
    precedence over the extra arguments like `area_rtol`.

    @param arg_dict
        Dictionary containing a dictionary for each property. Missing
        dictionaries are filled with empty ones (plus default values for some
        options).
    @param area_rtol
        Relative tolerance for the area integral. Setting this too low will
        result in integration warnings being produced and possibly
        underestimated residual errors.
    @param min_stability_values
        Minimum number of MOTS-stability eigenvalues to compute. The default
        is `30`.
    @param m_terminate_index
        Index of the eigenvalue of the `m=0` mode to use to as stopping
        criterion for the angular mode. Default is `30`.
        See .curve.expcurve.ExpansionCurve.stability_parameter() for details.
    @param stability_convergence_factors
        Factors by which to multiply the resolution used for computing the
        stability spectrum. Each of the resulting lower or higher resolutions
        is used to compute the same spectrum, so that convergence of the
        individual eigenvalues can be examined. Defaults to ``(0.2, 0.4, 0.6,
        0.8, 0.9, 1.1)``.
    @param shear_max_lmax
        Maximum shear mode to compute in case it does not converge earlier.
        Default is to use the default value in
        ..curve.expcurve.ExpansionCurve.expand_shear_scalar().
    """
    if arg_dict is None:
        arg_dict = dict()
    for prop in ALL_PROPS:
        arg_dict.setdefault(prop, {})
    arg_dict["area"].setdefault("epsrel", area_rtol)
    arg_dict["area"].setdefault("limit", 100)
    arg_dict["stability_convergence"].setdefault(
        "convergence_factors", stability_convergence_factors
    )
    arg_dict["stability_convergence_null"].setdefault(
        "convergence_factors", stability_convergence_factors
    )
    for prop in ["stability", "stability_null", "stability_convergence",
                 "stability_convergence_null"]:
        arg_dict[prop].setdefault("min_num", min_stability_values)
    for prop in ["stability", "stability_null"]:
        arg_dict[prop].setdefault("m_terminate_index", m_terminate_index)
        arg_dict[prop].setdefault("method", "direct")
    arg_dict["dist_top_bot"].setdefault("rtol", 1e-5)
    arg_dict["dist_top_bot"].setdefault("allow_intersection", True)
    arg_dict["zeta"].setdefault("num", None)
    arg_dict["length_maps"].setdefault("num", None)
    arg_dict["length_maps"].setdefault("accurate", True)
    arg_dict["constraints"].setdefault("num", None)
    arg_dict["constraints"].setdefault("fd_order", None)
    arg_dict["ingoing_exp"].setdefault("Ns", None)
    arg_dict["ingoing_exp"].setdefault("eps", 1e-6)
    arg_dict["avg_ingoing_exp"].setdefault("epsabs", "auto")
    arg_dict["avg_ingoing_exp"].setdefault("limit", "auto")
    for prop in ["ricci", "mean_curv", "curv2"]:
        arg_dict[prop].setdefault("Ns", None)
        arg_dict[prop].setdefault("eps", 1e-6)
        arg_dict[prop].setdefault("xatol", 1e-6)
    arg_dict["shear"].setdefault("num", None)
    arg_dict["shear"].setdefault("eps", 1e-6)
    arg_dict["shear"].setdefault("xatol", 1e-6)
    arg_dict["shear2_integral"].setdefault("n", "auto")
    arg_dict["shear_scalar"].setdefault("lmax", "auto")
    if shear_max_lmax is not None:
        arg_dict["shear_scalar"].setdefault("max_lmax", shear_max_lmax)
    arg_dict["neck"].setdefault("xtol", 1e-6)
    arg_dict["neck"].setdefault("epsrel", 1e-6)
    for other_hname, where in zip(['top', 'bot'], ['top', 'bottom']):
        prop = "z_dist_%s_inner" % other_hname
        arg_dict[prop].setdefault("rtol", 1e-5)
        arg_dict[prop].setdefault("where", where)
    arg_dict["area_parts"].setdefault("epsrel", area_rtol)
    arg_dict["area_parts"].setdefault("limit", 100)
    arg_dict["multipoles"].setdefault("max_n", max_multipole_n)
    return arg_dict


def _area(c, v=None, **kw):
    # pylint: disable=unused-argument
    r"""Compute the area for the curve.

    The parameter `v` is ignored here and can be used to force recomputation
    of the area for curves already containing an area result.
    """
    # Note: `v` can be passed to enforce recomputation of results previously
    # stored in curve files. When the arguments passed to `_area()` are
    # different than those used to compute the original result, recomputation
    # is triggered in `compute_props()`.
    A, abserr, info, message = c.area(full_output=True, **kw)
    return _PropResult(
        result=A,
        extras=dict(abserr=abserr, info=info, message=message),
    )


def _area_parts_individual(c, out_dir, other_run, other_hname, v=None, **kw):
    # pylint: disable=unused-argument
    r"""Compute the area of different parts of an individual MOTS.

    If the other individual MOTS does not intersect this MOTS, nothing is done
    and `None` is returned. If it does intersect, computes the area of the
    exterior and interior parts separately and returns an
    numutils.IntegrationResults object. The returned object's `info`
    dictionary contains an `intersection` key with the curve paramters of this
    (first) and the other (second) curve at which they intersect.
    """
    c_other = _get_aux_curve(
        "%s/../../%s/%s" % (out_dir, other_run, other_hname),
        hname=other_hname,
        it=c.metric.iteration,
        disp=False,
    )
    if not c_other:
        raise AuxResultMissing()
    z_dist = c.z_distance_using_metric(
        metric=None, other_curve=c_other, allow_intersection=True,
    )
    if z_dist >= 0:
        return None
    la1, la2 = c.locate_intersection(c_other, N1=100, N2=100)
    kw['full_output'] = True
    return IntegrationResults(
        c.area(domain=(0, la1), **kw),
        c.area(domain=(la1, np.pi), **kw),
        info=dict(intersection=(la1, la2)),
    )


def _area_parts_inner(c, out_dir, top_run, bot_run, v=None, **kw):
    # pylint: disable=unused-argument
    r"""Compute areas of different parts of the inner common MOTS.

    Note that a precondition for this function is computation of the 'neck'.
    If a neck was actually found, computes the area of the upper and lower
    parts separately. The result is an numutils.IntegrationResults object with
    the two values and a `info` dictionary containing the used `neck`
    location.

    If, in addition to a neck, the MOTS self-intersects, then it will also
    intersect the two individual MOTSs. The four intersection points (two
    parameters for the self-intersection, plus two for the intersection with
    the other MOTSs) together with the neck partition the curve into six
    sections, of which the area is computed separately. The resulting
    numutils.IntegrationResults object's `info` dictionary contains the
    location of all detected intersections. For the individual MOTSs, the
    second parameter in the values refers to the parameter of the individual
    MOTS at which this intersection happens.
    """
    c_top = _get_aux_curve(
        "%s/../../%s/%s" % (out_dir, top_run, 'top'),
        hname='top',
        it=c.metric.iteration,
        disp=False,
    )
    c_bot = _get_aux_curve(
        "%s/../../%s/%s" % (out_dir, bot_run, 'bot'),
        hname='bot',
        it=c.metric.iteration,
        disp=False,
    )
    if not c_top or not c_bot:
        raise AuxResultMissing()
    if c_top(0)[1] < c_bot(0)[1]:
        c_top, c_bot = c_bot, c_top
    z_dist = c_top.z_distance_using_metric(
        metric=None, other_curve=c_bot, allow_intersection=True,
    )
    self_intersecting = z_dist < 0
    neck = c.user_data['neck']['circumference']['param']
    if not neck:
        return None
    kw['full_output'] = True
    if not self_intersecting:
        A1 = c.area(domain=(0, neck), **kw)
        A2 = c.area(domain=(neck, np.pi), **kw)
        return IntegrationResults(
            A1, A2, info=dict(neck=neck, self_intersection=None)
        )
    # A self-intersecting inner common MOTS also crosses both individual
    # horizons.
    la_self1, la_self2 = c.locate_self_intersection(neck=neck)
    # To make the search more robust, we use our knowledge that the inner
    # common MOTS intersects the individual MOTSs close to the point the
    # intersect each other.
    la_top, la_bot = c_top.locate_intersection(c_bot)
    la_top1, la_top2 = c.locate_intersection(
        c_top,
        domain1=(neck, np.pi), strict1=True, N1=50,
        domain2=(clip(la_top-0.2, 0, np.pi), clip(la_top+0.2, 0, np.pi)),
        strict2=False, N2=10,
    )
    if None in (la_top1, la_top2):
        # search failed, retry with different settings
        la_top1, la_top2 = c.locate_intersection(
            c_top,
            domain1=(la_self2, np.pi), # start at self-intersection point
            strict1=True,
            N1=max(100, min(1000, int(c.num/2))), # use much finer initial grid
            domain2=(clip(la_top-0.2, 0, np.pi), clip(la_top+0.2, 0, np.pi)),
            strict2=False, N2=30,
        )
    la_bot1, la_bot2 = c.locate_intersection(
        c_bot,
        domain1=(0, neck), strict1=True, N1=50,
        domain2=(clip(la_bot-0.2, 0, np.pi), clip(la_bot+0.2, 0, np.pi)),
        strict2=False, N2=10,
    )
    if None in (la_bot1, la_bot2):
        # search failed, retry with different settings
        la_bot1, la_bot2 = c.locate_intersection(
            c_bot,
            domain1=(0, la_self1), # start at self-intersection point
            strict1=True,
            N1=max(100, min(1000, int(c.num/2))), # use much finer initial grid
            domain2=(clip(la_bot-0.2, 0, np.pi), clip(la_bot+0.2, 0, np.pi)),
            strict2=False, N2=30,
        )
    params = sorted([0.0, la_bot1, la_self1, neck, la_self2, la_top1, np.pi])
    return IntegrationResults(
        *[c.area(domain=(params[i], params[i+1]), **kw)
          for i in range(len(params)-1)],
        info=dict(neck=neck,
                  self_intersection=(la_self1, la_self2),
                  top_intersection=(la_top1, la_top2),
                  bot_intersection=(la_bot1, la_bot2)),
    )


def _ricci_difference_integral(c, out_dir, top_run, bot_run, v):
    # pylint: disable=unused-argument
    r"""Compute the integral of the difference of Ricci scalars between inner
    and the two individual MOTSs.

    The following properties are prerequisites for computing this one:
        * `length_maps`
    """
    c_top = _interpret_aux_curve(c=c, aux_run=top_run, aux_hname="top",
                                 out_dir=out_dir)
    c_bot = _interpret_aux_curve(c=c, aux_run=bot_run, aux_hname="bot",
                                 out_dir=out_dir)
    # Prevent loading top/bot metric files
    with _replace_metric(c_top, c.metric), _replace_metric(c_bot, c.metric), \
            c.fix_evaluator(), c_top.fix_evaluator(), c_bot.fix_evaluator():
        lm_inner = c.user_data["length_maps"]["length_map"].evaluator()
        im_inner = c.user_data["length_maps"]["inv_map"].evaluator()
        neck = c.user_data['neck']['circumference']['param']
        area_element = c.get_area_integrand()
        n = 2 * max(c.num, c_top.num + c_bot.num)
        def _integrate(a, b, other_curve, a_other=0.0, b_other=np.pi):
            a_proper = lm_inner(a)
            b_proper = lm_inner(b)
            lm_other = other_curve.user_data["length_maps"]["length_map"].evaluator()
            im_other = other_curve.user_data["length_maps"]["inv_map"].evaluator()
            params = []
            params_other = []
            measure_inner = []
            ricci_inner = []
            ricci_other = []
            def integrand(xs):
                params[:] = xs
                integrand_values = []
                for param in params:
                    measure_inner.append(area_element(param))
                    ricci_inner.append(c.ricci_scalar(param))
                    rel_proper = (lm_inner(param) - a_proper)/(b_proper-a_proper)
                    param_other = im_other(a_other + rel_proper*(b_other-a_other))
                    params_other.append(param_other)
                    ricci_other.append(other_curve.ricci_scalar(param_other))
                    integrand_values.append(
                        measure_inner[-1] * (ricci_inner[-1] - ricci_other[-1])
                    )
                return integrand_values
            value = 2*np.pi * _fixed_quad(integrand, a=a, b=b, n=n)
            params, params_other, measure_inner, ricci_inner, ricci_other = map(
                np.asarray,
                [params, params_other, measure_inner, ricci_inner, ricci_other]
            )
            ricci_integral = 2*np.pi * _fixed_quad(
                lambda _: measure_inner*ricci_inner, a=a, b=b, n=n
            )
            area = 2*np.pi * _fixed_quad(
                lambda _: measure_inner, a=a, b=b, n=n
            )
            return value, dict(params=params, params_other=params_other,
                               measure_inner=measure_inner,
                               ricci_inner=ricci_inner,
                               ricci_other=ricci_other,
                               ricci_inner_integral=ricci_integral,
                               area_inner=area)
        value1, info1 = _integrate(a=0.0, b=neck, other_curve=c_top)
        value2, info2 = _integrate(a=neck, b=np.pi, other_curve=c_bot)
        return dict(value=value1+value2,
                    value1=value1, info1=info1,
                    value2=value2, info2=info2)


def _data_is_valid(prop, value, args, hname):
    r"""Return whether the current data is valid or should be removed."""
    if prop == 'area_parts' and hname == 'inner':
        ap = value
        if (len(ap) == 6 and (ap.info['self_intersection'][0] <= 0
                              or ap.info['self_intersection'][1] <= 0)):
            # MOTS self-intersection found at negative parameter values.
            return False
    return True


def _multipoles(c, max_n, v=None):
    r"""Compute the first multipoles for the given curve.

    @return numutils.IntegrationResults object containing the results with
        estimates of their accuracy.

    @param c
        Curve representing the horizon.
    @param max_n
        Highest multipole to compute. It will compute the elements
        ``0, 1, 2, ..., max_n`` (i.e. ``max_n+1`` elements).
    @param v
        "Version". Can be used to force recomputation from the `do_prop()`
        call and determines if previous results are re-used.

    @b Notes

    The strategy for computation is to first try to compute using the
    estimated curve's residual expansion as absolute tolerance. If this fails
    or produces integration warnings, the tolerance is successively increased
    and computation is retried. This is done separately for each multipole
    moment.

    Previously computed multipole moments stored under the ``'multipoles'``
    key in ``c.user_data`` will be re-used in case their version parameter
    equals the value of `v`.
    """
    if 'accurate_abs_delta' in c.user_data:
        residual = c.user_data['accurate_abs_delta']['max']
    else:
        pts = np.linspace(0, np.pi, 2*c.num+1, endpoint=False)[1:]
        residual = np.absolute(c.expansions(pts)).max()
    prop = 'multipoles'
    if c.user_data.get("%s_args" % prop, {}).get("v", None) == v:
        prev_multipoles = c.user_data[prop]
    else:
        prev_multipoles = []
    results = []
    det_q = c.get_det_q_func(cached=True)
    ricci_scal = c.get_ricci_scalar_func(cached=True)
    zeta = c.user_data.get('zeta', None)
    if zeta is None:
        zeta = c.compute_zeta()
    def _get_I_n(n):
        if n < len(prev_multipoles):
            return prev_multipoles[n]
        try:
            res = try_quad_tolerances(
                func=lambda tol: c.multipoles(
                    min_n=n, max_n=n, epsabs=tol, zeta=zeta,
                    limit=max(100, int(round(c.num/2))),
                    det_q_function=det_q, ricci_scalar_function=ricci_scal,
                    full_output=True, disp=True,
                ),
                tol_min=residual,
                tol_max=max(1e-3, 10*residual),
            )
        except IntegrationWarning as e:
            res = [IntegrationResult(np.nan, np.nan, info=None, warning=str(e))]
        return res[0]
    for n in range(max_n+1):
        results.append(_get_I_n(n))
    return IntegrationResults(*results, sum_makes_sense=False)


def _stability(c, min_num, method, m_terminate_index, slice_normal, v=None, **kw):
    # pylint: disable=unused-argument
    r"""Compute the stability spectrum for the curve.

    The parameter `v` is ignored here and can be used to force recomputation
    of this property in case the code has changed.
    """
    if method != 'direct':
        raise ValueError("Unsupported method: %s" % (method,))
    num = max(min_num, c.num)
    principal, spectrum = c.stability_parameter(
        num=num, m_terminate_index=m_terminate_index,
        slice_normal=slice_normal, full_output=True, **kw
    )
    return _PropResult(
        result=(principal, spectrum.get(l='all', m=0)),
        extras=dict(method=method, spectrum=spectrum),
    )


def _stability_convergence(c, min_num, convergence_factors, method,
                           slice_normal, verbose=False, v=None, **kw):
    # pylint: disable=unused-argument
    r"""Compute the stability spectrum at different resolutions.

    The parameter `v` can be used to force recomputation of this property in
    case the code has changed. Note that results from the ``"stability"``
    property are reused only if the `v` parameters match.
    """
    if method != 'direct':
        raise ValueError("Unsupported method: %s" % (method,))
    # Collect previously computed results (but only if computed with the
    # correct method and version).
    if slice_normal:
        prop = 'stability_convergence'
        prop_main = 'stability'
    else:
        prop = 'stability_convergence_null'
        prop_main = 'stability_null'
    def _get_method(key):
        if c.user_data.get("%s_args" % key, {}).get("v", None) != v:
            # Computed with different version of the code. Don't reuse.
            return None
        return c.user_data.get("%s_extras" % key, dict(method=None))['method']
    prev_spectra = c.user_data.get(prop, [])
    if _get_method(prop) != method:
        prev_spectra = []
    if prop_main in c.user_data and _get_method(prop_main) == method:
        prev_spectra.append(c.user_data[prop_main])
    main_num = max(min_num, c.num)
    def _compute(factor):
        r"""Compute the stability spectrum for `factor*main_num`."""
        num = int(round(factor * main_num))
        for principal, spectrum in prev_spectra:
            if spectrum.shape[0] == num:
                return principal, spectrum
        if verbose:
            print(" x%s(%s)" % (round(factor, 12), num), end='',
                  flush=True)
        principal, spectrum = c.stability_parameter(
            num=num, m_max=0, slice_normal=slice_normal, full_output=True, **kw
        )
        return principal, spectrum.get(l='all', m=0)
    result = [_compute(f) for f in convergence_factors]
    return _PropResult(
        result=result,
        extras=dict(method=method),
    )


def _ricci(c, Ns, eps, xatol):
    r"""Compute the Ricci scalar and its maximum along the curve."""
    return _sample_and_extremum(c, c.ricci_scalar, Ns=Ns, eps=eps, xatol=xatol)


def _mean_curv(c, Ns, eps, xatol):
    r"""Compute the mean curvature and its maximum along the curve."""
    def func(x):
        return c.extrinsic_surface_curvature(x, trace=True)
    return _sample_and_extremum(c, func, Ns=Ns, eps=eps, xatol=xatol)


def _curv_squared(c, Ns, eps, xatol):
    r"""Compute the curvature square and its maximum along the curve."""
    def func(x):
        return c.extrinsic_surface_curvature(x, square=True)
    return _sample_and_extremum(c, func, Ns=Ns, eps=eps, xatol=xatol)


def _sample_and_extremum(c, func, Ns, eps, xatol):
    r"""Sample a property along the curve and find its maximum."""
    if Ns is None:
        Ns = max(100, 2*c.num)
    pts = np.linspace(eps, np.pi-eps, Ns)
    step = pts[1] - pts[0]
    with c.fix_evaluator():
        x_max, f_max, fx = _find_extremum(func, pts, eps=eps, xatol=xatol)
        fx = np.array(list(zip(pts, fx)))
    return dict(values=fx, x_max=x_max, f_max=f_max)


def _find_extremum(func, pts, eps, xatol, fx=None, domain=(0, np.pi)):
    if fx is None:
        fx = [func(x) for x in pts]
    fx = np.asarray(fx)
    idx = max(range(len(pts)), key=lambda i: abs(fx[i]))
    a = pts[max(0, idx-1)]
    b = pts[min(len(pts)-1, idx+1)]
    x_max, f_max = inf_norm1d(
        func, domain=(max(domain[0], a), min(domain[1]-eps, b)),
        Ns=0, xatol=xatol
    )
    return x_max, f_max, fx


def _ricci_interp(c, num=None, basis_cls="cos"):
    r"""Expand the Ricci scalar into a Cosine series."""
    if num is None:
        num = c.num
    basis_cls = interpret_basis_class(basis_cls).get_series_cls()
    with c.fix_evaluator():
        ricci = basis_cls.from_function(
            f=c.ricci_scalar, domain=(0, np.pi), num=num, lobatto=False,
        )
    return ricci


def _shear(c, num, eps, xatol, v=None):
    # pylint: disable=unused-argument
    r"""Compute the shear and shear-square at Chebyshev collocation points.

    The parameter `v` is ignored here and can be used to force recomputation
    of this property in case the code has changed.
    """
    if num is None:
        num = max(100, 2*c.num)
    pts = Cheby([], domain=(0, np.pi)).collocation_points(
        num=num, lobatto=False,
    )
    pts.reverse()
    pts = np.asarray(pts)
    with c.fix_evaluator():
        shear = []
        shear2 = []
        for x in pts:
            sigma, sigma2 = c.shear(x, full_output=True)
            shear.append(sigma)
            shear2.append(sigma2)
        shear2 = np.asarray(shear2)
        x_max, f_max, _ = _find_extremum(
            lambda x: c.shear(x, full_output=True)[1],
            pts=pts, eps=eps, xatol=xatol, fx=shear2
        )
    return dict(
        params=pts, shear=shear, shear2=shear2,
        shear2_max_value=f_max,
        shear2_max_point=x_max
    )


def _shear2_integral(c, n):
    r"""Integrate the shear^2 over the MOTS."""
    return dict(
        value=c.shear_square_integral(n=n)
    )


def _shear_scalar(c, lmax, **kw):
    r"""Expand the shear scalar into spin weighted spherical harmonics."""
    zeta = c.user_data.get('zeta', None)
    al0, values, ta_space = c.expand_shear_scalar(
        lmax=lmax, full_output=True, zeta=zeta, **kw
    )
    return dict(
        ta_space=ta_space, values=values, al0=al0,
    )


def compute_shear_hat_scalar(curve, curves, steps=3, lmax="auto", min_lmax=64,
                             max_lmax=512, use_user_data=True,
                             recompute=False, full_output=False,
                             update_file=False):
    r"""Expand \f$\hat\sigma_{(\ell)}\f$ into spin weighted spherical harmonics.

    See .curve.expcurve.ExpansionCurve.expand_shear_scalar() for details and the
    meaning of the parameters.
    """
    zeta = curve.user_data.get('zeta', None)
    def _func(curve, curves, **kw):
        al0, values, ta_space = curve.expand_shear_scalar(
            curves=curves, steps=None, lmax=lmax, min_lmax=min_lmax,
            max_lmax=max_lmax, zeta=zeta, hat=True, full_output=True,
        )
        return dict(
            ta_space=ta_space, values=values, al0=al0,
        )
    return _compute_tube_property(
        _func, kwargs=dict(lmax=lmax, min_lmax=min_lmax, max_lmax=max_lmax),
        curve=curve, curves=curves, steps=steps, prop="shear_hat_scalar",
        use_user_data=use_user_data, recompute=recompute, version=1,
        full_output=full_output, update_file=update_file,
    )


def compute_shear_hat2_integral(curve, curves, steps=3, n="auto",
                                use_user_data=True, recompute=False,
                                full_output=False, update_file=False):
    r"""Integrate \f$|\hat\sigma|^2\f$ over the MOTS.

    See .curve.expcurve.ExpansionCurve.shear_hat_square_integral() for details.

    The following properties are prerequisites for computing this one:
        * `length_maps`
        * `tev_divergence`
    """
    tevs = curve.user_data["tev_divergence"]["vectors"]
    def _func(curve, curves, **kw):
        value = curve.shear_hat_square_integral(tevs=tevs)
        return dict(value=value)
    return _compute_tube_property(
        _func, kwargs=dict(n=n), curve=curve, curves=curves,
        steps=steps, prop="shear_hat2_integral", use_user_data=use_user_data,
        recompute=recompute, version=1, full_output=full_output,
        update_file=update_file,
    )


def compute_xi_hat2_integral(curve, curves, steps=3, n="auto",
                             use_user_data=True, recompute=False,
                             full_output=False, update_file=False):
    r"""Integrate \f$|\hat\xi|^2\f$ over the MOTS.

    See .curve.expcurve.ExpansionCurve.xi_hat_square_integral() for details.

    The following properties are prerequisites for computing this one:
        * `length_maps`
        * `tev_divergence`
    """
    tevs = curve.user_data["tev_divergence"]["vectors"]
    def _func(curve, curves, **kw):
        value = curve.xi_hat_square_integral(tevs, curves=curves, steps=None)
        return dict(value=value)
    return _compute_tube_property(
        _func, kwargs=dict(n=n), curve=curve, curves=curves,
        steps=steps, prop="xi_hat2_integral", use_user_data=use_user_data,
        recompute=recompute, version=1, full_output=full_output,
        update_file=update_file,
    )


def compute_xi_tilde2_integral(curve, curves, steps=3, n="auto",
                               use_user_data=True, recompute=False,
                               full_output=False, update_file=False):
    r"""Integrate \f$|\tilde\xi|^2\f$ over the MOTS.

    We define \f$\tilde\xi^\mu\f$ here as
    \f[
        \tilde\xi^\mu = q^{AB} \mathcal{V}^\mu \nabla_\mu \ell_B \,,
    \f]
    where \f$\mathcal{V}^\mu\f$ is the slicing-adapted evolution vector. See
    .curve.expcurve.TimeVectorData() for the difference.

    See .curve.expcurve.ExpansionCurve.xi_hat_square_integral() for details.

    The following properties are prerequisites for computing this one:
        * `length_maps`
        * `tev_divergence`
    """
    tevs = curve.user_data["tev_divergence"]["vectors"]
    def _func(curve, curves, **kw):
        value = curve.compute_xi_square_integral(
            tevs, curves=curves, steps=None, hat=False, r_hat=False,
        )
        return dict(value=value)
    return _compute_tube_property(
        _func, kwargs=dict(n=n), curve=curve, curves=curves,
        steps=steps, prop="xi_tilde2_integral", use_user_data=use_user_data,
        recompute=recompute, version=1, full_output=full_output,
        update_file=update_file,
    )


def compute_xi_vector(curve, curves, steps=3, num=None, eps=1e-6,
                      use_user_data=True, recompute=False, full_output=False,
                      update_file=False):
    r"""Compute the xi vector.

    See .curve.expcurve.ExpansionCurve.xi_vector() for details. We store all
    four quantities in the returned dictionary.

    @param curve
        MOTS along which to compute the xi vector.
    @param curves
        Curves defining the MOTT.
    @param steps
        Number of surrounding MOTSs used for interpolating the tube. Default
        is `3`.
    @param num
        Number of equidistant points at which to compute the xi vector
        quantities along the MOTS. Note that the points will be equidistant in
        proper length.
    @param eps
        Distance to stay away from the domain boundaries 0 and pi (in scaled
        proper length coordinate). Default is `1e-6`.
    @param use_user_data
        Whether to store the results in the curve's `user_data` dictionary and
        retrieve it from there instead of recomputing on subsequent calls. If
        the curve is saved afterwards, this data will persist.
    @param recompute
        Whether compute even if ``use_user_data=True`` and the data is already
        stored in `user_data`. This can be used to force an update of
        previously computed result.
    @param full_output
        If `True`, return the result dictionary and a boolean indicating
        whether computation was done (`True`) or data was taken from
        `user_data` (`False`). Default is `False`, i.e. only the result
        dictionary is returned.
    @param update_file
        Whether to immediately write any computed results into the curve's
        data file (in case ``use_user_data==True``). This is done in a way
        safe for being used when different processes work on different
        properties simultaneously, even across nodes in a compute cluster
        accessing the same file via a network file system. Note that the data
        in the supplied curve object will not reflect updates performed by
        other processes in the meantime. The file, however, will contain this
        data. You should hence not save the given curve to the file after
        calling this function, since you would potentially overwrite the
        updates of other processes.
    """
    if num is None:
        num = max(200, curve.num)
    proper_pts = np.linspace(eps, np.pi-eps, num)
    def _func(curve, curves, **kw):
        xi_A_up, xi_A, xi2, xi_scalar = curve.xi_vector(
            pts=proper_pts, curves=curves, steps=None, full_output=True,
        )
        return dict(xi_A_up=xi_A_up, xi_A=xi_A, xi2=xi2,
                    xi_scalar=xi_scalar, proper_pts=proper_pts)
    return _compute_tube_property(
        _func, kwargs=dict(num=num, eps=eps), curve=curve, curves=curves,
        steps=steps, prop="xi_vector", use_user_data=use_user_data,
        recompute=recompute, version=1, full_output=full_output,
        update_file=update_file,
    )


def compute_xi2_integral(curve, curves, steps=3, n="auto", use_user_data=True,
                         recompute=False, full_output=False,
                         update_file=False):
    r"""Integrate xi^2 over the MOTS.

    See .curve.expcurve.ExpansionCurve.xi_square_integral() for details.

    The `n` parameter defines the fixed quadrature resolution for integrating.
    """
    def _func(curve, curves, **kw):
        value = curve.xi_square_integral(curves=curves, steps=None, n=n)
        return dict(value=value)
    return _compute_tube_property(
        _func, kwargs=dict(n=n), curve=curve, curves=curves,
        steps=steps, prop="xi2_integral", use_user_data=use_user_data,
        recompute=recompute, version=2, full_output=full_output,
        update_file=update_file,
    )


def compute_xi_scalar(curve, curves, steps=3, lmax="auto", min_lmax=64,
                      max_lmax=512, use_user_data=True, recompute=False,
                      full_output=False, update_file=False):
    r"""Expand xi_(ell) into spin weighted spherical harmonics.

    See .curve.expcurve.ExpansionCurve.expand_xi_scalar() for details and the
    meaning of the parameters.
    """
    zeta = curve.user_data.get('zeta', None)
    def _func(curve, curves, **kw):
        al0, values, ta_space = curve.expand_xi_scalar(
            curves=curves, steps=None, lmax=lmax, min_lmax=min_lmax,
            max_lmax=max_lmax, zeta=zeta, full_output=True,
        )
        return dict(
            ta_space=ta_space, values=values, al0=al0,
        )
    return _compute_tube_property(
        _func, kwargs=dict(lmax=lmax, min_lmax=min_lmax, max_lmax=max_lmax),
        curve=curve, curves=curves, steps=steps, prop="xi_scalar",
        use_user_data=use_user_data, recompute=recompute, version=1,
        full_output=full_output, update_file=update_file,
    )


def compute_xi_hat_scalar(curve, curves, steps=3, lmax="auto", min_lmax=64,
                          max_lmax=512, use_user_data=True, recompute=False,
                          full_output=False, update_file=False):
    r"""Expand \f$\hat\xi_{(\ell)}\f$ into spin weighted spherical harmonics.

    See .curve.expcurve.ExpansionCurve.expand_xi_scalar() for details and the
    meaning of the parameters.
    """
    zeta = curve.user_data.get('zeta', None)
    def _func(curve, curves, **kw):
        al0, values, ta_space = curve.expand_xi_scalar(
            curves=curves, steps=None, lmax=lmax, min_lmax=min_lmax,
            max_lmax=max_lmax, zeta=zeta, hat=True, full_output=True,
        )
        return dict(
            ta_space=ta_space, values=values, al0=al0,
        )
    return _compute_tube_property(
        _func, kwargs=dict(lmax=lmax, min_lmax=min_lmax, max_lmax=max_lmax),
        curve=curve, curves=curves, steps=steps, prop="xi_hat_scalar",
        use_user_data=use_user_data, recompute=recompute, version=1,
        full_output=full_output, update_file=update_file,
    )


def compute_xi_tilde_scalar(curve, curves, steps=3, lmax="auto", min_lmax=64,
                            max_lmax=512, use_user_data=True, recompute=False,
                            full_output=False, update_file=False):
    r"""Expand \f$\tilde\xi_{(\ell)}\f$ into spin weighted spherical harmonics.

    We define \f$\tilde\xi^\mu\f$ here as
    \f[
        \tilde\xi^\mu = q^{AB} \mathcal{V}^\mu \nabla_\mu \ell_B \,,
    \f]
    where \f$\mathcal{V}^\mu\f$ is the slicing-adapted evolution vector. See
    .curve.expcurve.TimeVectorData() for the difference.

    See .curve.expcurve.ExpansionCurve.expand_xi_scalar() for details and the
    meaning of the parameters.
    """
    zeta = curve.user_data.get('zeta', None)
    def _func(curve, curves, **kw):
        al0, values, ta_space = curve.expand_xi_scalar(
            curves=curves, steps=None, lmax=lmax, min_lmax=min_lmax,
            max_lmax=max_lmax, zeta=zeta, hat=False, r_hat=False,
            full_output=True,
        )
        return dict(
            ta_space=ta_space, values=values, al0=al0,
        )
    return _compute_tube_property(
        _func, kwargs=dict(lmax=lmax, min_lmax=min_lmax, max_lmax=max_lmax),
        curve=curve, curves=curves, steps=steps, prop="xi_tilde_scalar",
        use_user_data=use_user_data, recompute=recompute, version=1,
        full_output=full_output, update_file=update_file,
    )


def compute_surface_gravity(curve, curves, steps=3, num=None, eps=1e-6,
                            use_user_data=True, recompute=False,
                            full_output=False, update_file=False):
    r"""Compute the surface gravity at a set of points on a MOTS.

    See .curve.expcurve.ExpansionCurve.surface_gravity() for details and the
    meaning of the first few parameters. The remaining parameters have the
    same meaning as in compute_xi_vector().
    """
    if num is None:
        num = max(200, curve.num)
    proper_pts = np.linspace(eps, np.pi-eps, num)
    def _func(curve, curves, **kw):
        kappa = curve.surface_gravity(
            pts=proper_pts, curves=curves, steps=None,
        )
        return dict(kappa=kappa, proper_pts=proper_pts)
    return _compute_tube_property(
        _func, kwargs=dict(num=num, eps=eps),
        curve=curve, curves=curves, steps=steps, prop="surface_gravity",
        use_user_data=use_user_data, recompute=recompute, version=1,
        full_output=full_output, update_file=update_file,
    )


def compute_surface_gravity_tev(curve, curves, steps=3, use_user_data=True,
                                recompute=False, full_output=False,
                                update_file=False):
    r"""Compute the surface gravity with respect to the time evolution vector.

    See .curve.expcurve.ExpansionCurve.surface_gravity() for details and the
    meaning of the first few parameters. This uses the option ``wrt="tev"``.
    The remaining parameters have the same meaning as in compute_xi_vector().

    The following properties are prerequisites for computing this one:
        * `length_maps`
        * `tev_divergence`
    """
    length_map = curve.user_data["length_maps"]["length_map"].evaluator()
    tevs = curve.user_data["tev_divergence"]["vectors"]
    proper_pts = [length_map(tev.param) for tev in tevs]
    def _func(curve, curves, **kw):
        kappa = curve.surface_gravity(
            pts=proper_pts, curves=curves, steps=None, wrt="tev", tevs=tevs,
        )
        return dict(kappa=kappa, proper_pts=proper_pts)
    return _compute_tube_property(
        _func, kwargs=dict(
            tev_divergence_args=curve.user_data["tev_divergence_args"],
        ),
        curve=curve, curves=curves, steps=steps, prop="surface_gravity_tev",
        use_user_data=use_user_data, recompute=recompute, version=1,
        full_output=full_output, update_file=update_file,
    )


def compute_timescale_tau2(curve, curves, steps=3, use_user_data=True,
                           recompute=False, full_output=False,
                           update_file=False, option=0):
    r"""Compute the square of the timescale `tau`.

    See .curve.expcurve.ExpansionCurve.timescale_tau2() for details and the
    meaning of the first few parameters. The remaining parameters have the
    same meaning as in compute_xi_vector().

    The following properties are prerequisites for computing this one:
        * `length_maps`
        * `tev_divergence`
        * `surface_gravity_tev`

    Note that for ``option=4``, all curves in `curves` are also required to
    have data for the `tev_divergence` property.
    """
    if option in (0, 3, 4):
        kwargs = dict(
            tev_divergence_args=curve.user_data["tev_divergence_args"],
            surface_gravity_tev_args=curve.user_data["surface_gravity_tev_args"],
        )
    else:
        kwargs = dict(
            tev_divergence_args=curve.user_data["tev_divergence_args"],
        )
    prop = "timescale_tau2"
    if option:
        prop = "%s_option%02d" % (prop, option)
    kappas_tev = curve.user_data["surface_gravity_tev"]["kappa"]
    proper_pts = curve.user_data["surface_gravity_tev"]["proper_pts"]
    tevs = curve.user_data["tev_divergence"]["vectors"]
    def _func(curve, curves, **kw):
        tau2 = curve.timescale_tau2(
            tevs=tevs, kappas_tev=kappas_tev, proper_pts=proper_pts,
            curves=curves, steps=None, option=option,
        )
        if option == 2:
            return dict(tau=tau2)
        return dict(tau2=tau2)
    return _compute_tube_property(
        _func, kwargs=kwargs,
        curve=curve, curves=curves, steps=steps, prop=prop,
        use_user_data=use_user_data, recompute=recompute, version=1,
        full_output=full_output, update_file=update_file,
    )


def compute_timescale_tau2_option01(*args, **kwargs):
    r"""Compute option 1 in .curve.expcurve.ExpansionCurve.timescale_tau2()."""
    return compute_timescale_tau2(*args, option=1, **kwargs)


def compute_timescale_tau2_option02(*args, **kwargs):
    r"""Compute option 2 in .curve.expcurve.ExpansionCurve.timescale_tau2()."""
    return compute_timescale_tau2(*args, option=2, **kwargs)


def compute_timescale_tau2_option03(*args, **kwargs):
    r"""Compute option 3 in .curve.expcurve.ExpansionCurve.timescale_tau2()."""
    return compute_timescale_tau2(*args, option=3, **kwargs)


def compute_timescale_tau2_option04(*args, **kwargs):
    r"""Compute option 4 in .curve.expcurve.ExpansionCurve.timescale_tau2()."""
    return compute_timescale_tau2(*args, option=4, **kwargs)


def compute_timescale_T2(curve, curves, steps=3, use_user_data=True,
                         recompute=False, full_output=False,
                         update_file=False):
    r"""Compute the square of the timescale `T`.

    See .curve.expcurve.ExpansionCurve.timescale_T2() for details and the
    meaning of the first few parameters. The remaining parameters have the
    same meaning as in compute_xi_vector().

    The following properties are prerequisites for computing this one:
        * `length_maps`
        * `tev_divergence`
    """
    tevs = curve.user_data["tev_divergence"]["vectors"]
    def _func(curve, curves, **kw):
        T2 = curve.timescale_T2(tevs=tevs)
        return dict(T2=T2)
    return _compute_tube_property(
        _func, kwargs=dict(
            tev_divergence_args=curve.user_data["tev_divergence_args"],
        ),
        curve=curve, curves=curves, steps=steps, prop="timescale_T2",
        use_user_data=use_user_data, recompute=recompute, version=1,
        full_output=full_output, update_file=update_file,
    )


def compute_extremality_parameter(curve, curves, steps=3, n="auto",
                                  use_user_data=True, recompute=False,
                                  full_output=False, update_file=False):
    r"""Compute the extremality parameter.

    See .curve.expcurve.ExpansionCurve.extremality_parameter() for details.

    The `n` parameter defines the fixed quadrature resolution for integrating.
    """
    def _func(curve, curves, **kw):
        e, v_bots, params = curve.extremality_parameter(
            curves=curves, steps=None, n=n, full_output=True,
        )
        return dict(e=e, v_bots=v_bots, params=params)
    return _compute_tube_property(
        _func, kwargs=dict(n=n), curve=curve, curves=curves,
        steps=steps, prop="extremality_parameter",
        use_user_data=use_user_data, recompute=recompute, version=1,
        full_output=full_output, update_file=update_file,
    )


def _compute_tube_property(func, kwargs, curve, curves, steps, prop,
                           use_user_data, recompute, version, full_output,
                           update_file):
    r"""Compute a property along a MOTT.

    The given `func` should be a callable that computes the property and
    returns e.g. a dictionary with the results. It is called as

        result = func(curve, curves, **kwargs)

    and the result is stored in the given `prop` property of the
    `curve.user_data` dictionary. The arguments in `kwargs` are also stored in
    the key ``*_args``, where ``*`` is replaced with the property name. Note
    that we will add the following items to the stored arguments:
        * `steps`: the steps parameter
        * `version`: the given version, used just to reject deprecated results
        * `iterations`: slice numbers of MOTSs used to interpolate the tube

    The remaining parameters are explained in e.g. compute_xi_vector().
    """
    curves = curve.collect_close_in_time(curves, steps=steps)
    stored_args = kwargs.copy()
    stored_args["steps"] = steps
    stored_args["version"] = version
    stored_args["iterations"] = [c.metric.iteration for c in curves]
    results_dir = op.dirname(op.dirname(op.dirname(curve.loaded_from)))
    def _curve_fname(c):
        fname = op.relpath(c.loaded_from, start=results_dir)
        return fname
    stored_args["curves"] = [_curve_fname(c) for c in curves]
    key = prop
    key_args = "%s_args" % key
    data = curve.user_data
    if (use_user_data and not recompute and key in data
            and data.get(key_args, None) == stored_args):
        if full_output:
            return data[key], False
        return data[key]
    with timethis("%-40s" % ("Computing '%s'..." % prop), " elapsed: {}",
                  eol=False):
        result = func(curve, curves, **kwargs)
    if use_user_data:
        curve.user_data[key] = result
        curve.user_data[key_args] = stored_args
        if update_file:
            update_user_data_in_file(
                fname=curve.loaded_from,
                data={key: result, key_args: stored_args},
            )
    if full_output:
        return data[key], True
    return result


def _zeta(c, num):
    r"""Compute the invariant angle zeta."""
    return c.compute_zeta(num=num)


def _length_maps(c, num, accurate):
    r"""Compute the conversion mappings from parameterization to proper length."""
    length_map, inv_map, proper_length = c.proper_length_map(
        num=num, evaluators=False, accurate=accurate, full_output=True
    )
    return dict(
        length_map=length_map, inv_map=inv_map, proper_length=proper_length
    )


def _constraints(c, num, fd_order):
    r"""Compute the constraints close to the curve."""
    if num is None:
        num = max(100, 2*c.num)
    std_params = np.linspace(0, np.pi, num+1, endpoint=False)[1:]
    inv_map = c.user_data['length_maps']['inv_map'].evaluator()
    proper_params = [inv_map(la) for la in std_params]
    _, std_data = max_constraint_along_curve(
        curve=c, points=std_params, fd_order=fd_order, full_output=True,
    )
    _, proper_data = max_constraint_along_curve(
        curve=c, points=proper_params, fd_order=fd_order, full_output=True,
    )
    key = "near_curve"
    if fd_order is not None:
        key = "%s_order%s" % (key, fd_order)
    return {
        key: dict(
            std=[std_data[:,0], std_data[:,1], std_data[:,2]],
            proper=[proper_data[:,0], proper_data[:,1], proper_data[:,2]],
        )
    }


def max_constraint_along_curve(curve, points=None, x_padding=1, fd_order=None,
                               full_output=False):
    r"""Compute the maximum violation of the Hamiltonian constraint.

    This computes the constraints on a set of grid points close to the
    given curve and returns the maximum value of the Hamiltonian
    constraint.

    @param curve
        The curve along which to sample the constraint.
    @param points
        Number of points to sample at along the curve. If not specified,
        uses either the curve's *accurate* residual measuring points or
        (if these are not available) twice the resolution of the curve.
        May also be a list or tuple of parameter values in the range
        ``[0,pi]``.
    @param x_padding
        How many grid points to stay away from the z-axis. Default is 1.
    @param fd_order
        Finite difference convergence order for computing derivatives at
        grid points. Default is to use the current stencil.
    @param full_output
        If `True`, return the evaluated constraints along with the maximum
        violation.
    """
    if points is None and 'accurate_abs_delta' in curve.user_data:
        params = curve.user_data['accurate_abs_delta']['points']
    else:
        if points is None:
            points = 2 * curve.num
        if isinstance(points, (list, tuple, np.ndarray)):
            params = np.asarray(points)
        else:
            params = np.linspace(0, np.pi, points+1, endpoint=False)[1:]
    g = curve.metric
    gxx = g.component_matrix(0, 0)
    i0 = gxx.closest_element((0, 0, 0))[0]
    constraints = dict()
    full_data = []
    with g.temp_interpolation(interpolation='none', fd_order=fd_order):
        for la in params:
            i, j, k = gxx.closest_element(curve(la, xyz=True))
            if x_padding:
                i = max(i0+x_padding, i)
            key = (i, j, k)
            if key in constraints:
                ham, mom = constraints[key]
            else:
                pt = gxx.coords(i, j, k)
                ham, mom = g.constraints(pt, norm=True)
                constraints[key] = [ham, mom]
            full_data.append([la, ham, mom])
    data = np.array(list(constraints.values()))
    constraints = data[:,0]
    max_ham = np.absolute(constraints).max()
    if full_output:
        full_data = np.asarray(full_data)
        return max_ham, full_data
    return max_ham


def _ingoing_exp(c, Ns, eps):
    r"""Compute the ingoing expansion along the curve."""
    if Ns is None:
        Ns = max(100, 2*c.num)
    pts = np.linspace(eps, np.pi-eps, Ns)
    fvals = c.expansions(params=pts, ingoing=True)
    fx = np.array(list(zip(pts, fvals)))
    return dict(values=fx) # compatible with _sample_and_extremum() format


def _avg_ingoing_exp(c, epsabs, limit):
    r"""Compute the average ingoing expansion across the MOTS."""
    area = c.user_data['area']
    if epsabs == 'auto':
        if 'accurate_abs_delta' in c.user_data:
            epsabs = max(1e-6, c.user_data['accurate_abs_delta']['max'])
        else:
            epsabs = 1e-6
    if limit == 'auto':
        limit = max(50, int(c.num/2))
    return c.average_expansion(
        ingoing=True, area=area, epsabs=epsabs, limit=limit
    )


def _point_dists(c):
    r"""Compute the coordinate distances of all collocation points."""
    params = c.collocation_points()
    point_dists = c.point_distances(params)
    min_dist = point_dists.min()
    max_dist = point_dists.max()
    return dict(
        params=params,
        coord=dict(dists=point_dists, min=min_dist, max=max_dist),
        # leave open the option to include proper distances as well
    )


def _neck(c, xtol, epsrel):
    r"""Find and compute various neck properties for the inner common MOTS."""
    def _values(param):
        return dict(
            param=param,
            x_coord=c(param)[0],
            circumference=c.circumference(param),
            proper_x_dist=c.x_distance(param, epsrel=epsrel),
        )
    with c.fix_evaluator():
        return dict(
            coord=_values(c.find_neck('coord', xtol=xtol)[0]),
            circumference=_values(c.find_neck('circumference', xtol=xtol)[0]),
            proper_x_dist=_values(c.find_neck('proper_x_dist', xtol=xtol)[0]),
            max_ricci=_values(c.user_data['ricci']['x_max']),
            max_curv2=_values(c.user_data['curv2']['x_max']),
        )


def _dist_top_bot(c, rtol, allow_intersection, other_run, other_hname, out_dir):
    r"""Compute the proper distance of the two individual MOTSs."""
    c_other = _interpret_aux_curve(c=c, aux_run=other_run,
                                   aux_hname=other_hname, out_dir=out_dir)
    return c.z_distance(c_other, rtol=rtol,
                        allow_intersection=allow_intersection)


def _z_dist_to_inner(c, rtol, where, other_run, other_hname, out_dir):
    r"""Compute the proper distance of inner and individual MOTSs."""
    if isinstance(other_run, BaseCurve):
        c_other = other_run
    else:
        c_other = _get_aux_curve(
            "%s/../../%s/%s" % (out_dir, other_run, other_hname),
            hname=other_hname,
            it=c.metric.iteration,
            disp=False,
        )
        if not c_other:
            raise AuxResultMissing()
        if c_other.metric.time != c.metric.time:
            raise RuntimeError("Auxiliary curve not in correct slice.")
    return c.inner_z_distance(c_other, rtol=rtol, where=where)


def compute_tube_signature(curve, curves, num=None, eps=1e-6, proper=True,
                           steps=3, use_user_data=True, recompute=False,
                           full_output=False, update_file=False):
    r"""Compute the signature of the world tube traced out by MOTSs.

    @param curve
        The curve representing the MOTS at the time slice the signature of the
        MOTT should be computed.
    @param curves
        All known curves before and after `curve` in the simulation. These are
        interpolated to obtain the tangent of the coordinate along the tube.
        May optionally contain `curve` (has no effect) and does not need to be
        sorted.
    @param num
        Number of points to compute the signature at. Default is to use the
        curve's resolution but at least `200`.
    @param eps
        How close to approach the poles. Default is `1e-6`.
    @param proper
        Whether to distribute the points using the curve's proper length or
        its current parameterization. This will also be used to interpret the
        other `curves` of the tube, which is why setting this to `True`
        (default) is highly recommended.
    @param steps
        How many curves before and after this current curve to consider for
        interpolation. Default is `3`.
    @param use_user_data
        Whether to store the results in the curve's `user_data` dictionary and
        retrieve it from there instead of recomputing on subsequent calls. If
        the curve is saved afterwards, this data will persist.
    @param recompute
        Whether to recompute the signature even if ``use_user_data=True`` and
        the signature data is already there. This can be used to force an
        update of the stored signature data.
    @param full_output
        If `True`, return the result dictionary and a boolean indicating
        whether computation was done (`True`) or data was taken from
        `user_data` (`False`). Default is `False`, i.e. only the result
        dictionary is returned.
    @param update_file
        Whether to immediately write any computed results into the curve's
        data file (in case ``use_user_data==True``). This is done in a way
        safe for being used when different processes work on different
        properties simultaneously, even across nodes in a compute cluster
        accessing the same file via a network file system. Note that the data
        in the supplied curve object will not reflect updates performed by
        other processes in the meantime. The file, however, will contain this
        data. You should hence not save the given curve to the file after
        calling this function, since you would potentially overwrite the
        updates of other processes.
    """
    if num is None:
        num = max(200, curve.num)
    proper_pts = np.linspace(eps, np.pi-eps, num)
    def _func(curve, curves, **kw):
        return curve.signature_quantities(
            proper_pts, curves, proper=proper, steps=None
        )
    return _compute_tube_property(
        _func, kwargs=dict(num=num, eps=eps, proper=proper),
        curve=curve, curves=curves,
        steps=steps, prop="tube_signature", use_user_data=use_user_data,
        recompute=recompute, version=1, full_output=full_output,
        update_file=update_file,
    )


def compute_time_evolution_vector(curve, curves, num=None, eps=1e-6,
                                  proper=True, steps=3, integral=True,
                                  use_user_data=True, recompute=False,
                                  full_output=False, update_file=False):
    r"""Compute the time evolution vector along the world tube of MOTSs.

    This function can compute the time evolution vectors at a grid of points
    along the MOTS. By default, it also integrates the divergence of the time
    evolution vector field over the MOTS. The result is returned as a
    dictionary with the following structure:

        result = dict(
            vectors=...,
            pts=...,
            integral=dict(value=..., vectors=...),
        )

    Here, the two `vectors` keys contain lists of objects of type
    ..curve.expcurve.TimeVectorData. The `pts` key contains the proper length
    values at which the `vectors` are computed by default. If
    ``proper=False``, then `pts` contains the parameter positions of the
    `vectors` instead. Both kinds of information in `pts` can be reconstructed
    easily from the TimeVectorData objects and the length maps (see
    ..curve.expcurve.ExpansionCurve.proper_length_map()), since the former
    contain their curve parameter value as an attribute.

    @param curve
        Curve representing the MOTS at the time slice we should compute at.
    @param curves
        All known curves before and after `curve` in the simulation. These are
        interpolated in time to find the evolution vector. May optionally
        contain `curve` (has no effect) and does not need to be sorted.
    @param num
        Number of points to compute the vector at. Default is to use the
        curve's resolution but at least `200`.
    @param eps
        How close to approach the poles. Default is `1e-6`.
    @param proper
        Whether to distribute the points using the curve's proper length or
        its current parameterization. Default is `True`.
    @param steps
        How many curves before and after this current curve to consider for
        interpolation. Default is `3`.
    @param integral
        Compute the integral of the divergence of the time evolution vector
        over the MOTS. If `True` (default), this will add an ``"integral"``
        key to the result dictionary containing a dictionary with the `value`
        of the integral and the `vectors` created during evaluation of the
        integrand (a list of ..curve.expcurve.TimeVectorData objects). Note
        that `num` has no effect on the computation of the integral.
    @param use_user_data
        Whether to store the results in the curve's `user_data` dictionary and
        retrieve it from there instead of recomputing on subsequent calls. If
        the curve is saved afterwards, this data will persist.
    @param recompute
        Whether compute even if ``use_user_data=True`` and the data is already
        stored in `user_data`. This can be used to force an update of
        previously computed result.
    @param full_output
        If `True`, return the result dictionary and a boolean indicating
        whether computation was done (`True`) or data was taken from
        `user_data` (`False`). Default is `False`, i.e. only the result
        dictionary is returned.
    @param update_file
        Whether to immediately write any computed results into the curve's
        data file (in case ``use_user_data==True``). This is done in a way
        safe for being used when different processes work on different
        properties simultaneously, even across nodes in a compute cluster
        accessing the same file via a network file system. Note that the data
        in the supplied curve object will not reflect updates performed by
        other processes in the meantime. The file, however, will contain this
        data. You should hence not save the given curve to the file after
        calling this function, since you would potentially overwrite the
        updates of other processes.
    """
    if num is None:
        num = max(200, curve.num)
    pts = np.linspace(eps, np.pi-eps, num)
    if proper:
        proper_pts = pts
    else:
        length_map = curve.cached_length_maps()[0]
        proper_pts = np.asarray([length_map(x) for x in pts])
    def _func(curve, curves, **kw):
        tevs = curve.time_evolution_vector(
            proper_pts, curves=curves, steps=steps, full_output=True
        )
        result = dict(vectors=tevs, pts=pts)
        if integral:
            value, vectors = curve.integrate_tev_divergence(
                curves=curves, unity_b=True, full_output=True
            )
            result["integral"] = dict(value=value, vectors=vectors)
        return result
    return _compute_tube_property(
        _func,
        kwargs=dict(num=num, eps=eps, proper=proper, integral=integral),
        curve=curve, curves=curves, steps=steps, prop="time_evolution_vector",
        use_user_data=use_user_data, recompute=recompute, version=1,
        full_output=full_output, update_file=update_file,
    )


def compute_tev_divergence(curve, curves, n="auto", steps=3,
                           use_user_data=True, recompute=False,
                           full_output=False, update_file=False):
    r"""Integrate the divergence of the time evolution vector (TEV).

    This is similar to the ``"integral"`` key of the results produced by
    compute_time_evolution_vector(). The main differences are:

    * We use the TEV scaled with unity time component (see [1]).
    * We only compute the integral, not the vectors at another set of points.

    The result dictionary will have the form:

        result = dict(integral=..., vectors=...)

    The `vectors` key contains a list of TEV objects at the Gaussian
    quadrature points in intrinsic curve parameter space. From these, you can
    extract the coefficients `b` and `c` for the decomposition into the
    ingoing and outgoing null normals (scaled to have product -1). This allows
    the values to be re-used for other quantities and even integrals without
    recomputing all the vectors.

    @param curve
        Curve representing the MOTS at the time slice we should compute at.
    @param curves
        All known curves before and after `curve` in the simulation. These are
        interpolated in time to find the evolution vector. May optionally
        contain `curve` (has no effect) and does not need to be sorted.
    @param n
        Order of the fixed quadrature integration (equal to the number of
        points at which the integrand is evaluated). The default ``"auto"``
        will use twice the current curve's resolution, but at least 30 points.
    @param steps
        How many curves before and after this current curve to consider for
        interpolation. Default is `3`.
    @param use_user_data
        Whether to store the results in the curve's `user_data` dictionary and
        retrieve it from there instead of recomputing on subsequent calls. If
        the curve is saved afterwards, this data will persist.
    @param recompute
        Whether compute even if ``use_user_data=True`` and the data is already
        stored in `user_data`. This can be used to force an update of
        previously computed result.
    @param full_output
        If `True`, return the result dictionary and a boolean indicating
        whether computation was done (`True`) or data was taken from
        `user_data` (`False`). Default is `False`, i.e. only the result
        dictionary is returned.
    @param update_file
        Whether to immediately write any computed results into the curve's
        data file (in case ``use_user_data==True``). This is done in a way
        safe for being used when different processes work on different
        properties simultaneously, even across nodes in a compute cluster
        accessing the same file via a network file system. Note that the data
        in the supplied curve object will not reflect updates performed by
        other processes in the meantime. The file, however, will contain this
        data. You should hence not save the given curve to the file after
        calling this function, since you would potentially overwrite the
        updates of other processes.

    @b References

    [1] Booth, Ivan, and Stephen Fairhurst. "Extremality conditions for
        isolated and dynamical horizons." Physical review D 77.8 (2008):
        084005.
    """
    def _func(curve, curves, **kw):
        value, vectors = curve.integrate_tev_divergence(
            curves=curves, n=n, unity_b=False, full_output=True
        )
        return dict(integral=value, vectors=vectors)
    return _compute_tube_property(
        _func,
        kwargs=dict(n=n),
        curve=curve, curves=curves, steps=steps, prop="tev_divergence",
        use_user_data=use_user_data, recompute=recompute, version=1,
        full_output=full_output, update_file=update_file,
    )


def _interpret_aux_curve(c, aux_run, aux_hname, out_dir, disp=True):
    r"""Return an auxiliary curve in the same slice as the given one.

    This returns a previously computed MOTS for horizon `aux_hname` in the
    same slice as the given curve `c`. The given `aux_run` may also be the
    curve itself, in which case it is returned without further checks.

    @param c
        The curve object from which the current iteration and time is used to
        identify the slice to get the auxiliary curve in.
    @param aux_run
        The name of the finder "run" in which the auxiliary curve should be
        found. May also be a curve object, which in this case is returned
        immediately without further checks.
    @param out_dir
        Path to the folder containing the file for curve `c`.
    @param disp
        Whether to raise an error if the curve is either missing or not
        contained in the same slice. If it is missing, an AuxResultMissing
        error is raised. Curves in the wrong slice trigger a `RuntimeError`
        (both only in case `aux_run` is not the curve itself).
    """
    if isinstance(aux_run, BaseCurve):
        c_aux = aux_run
    else:
        c_aux = _get_aux_curve(
            "%s/../../%s/%s" % (out_dir, aux_run, aux_hname),
            hname=aux_hname,
            it=c.metric.iteration,
            disp=False,
        )
        if disp:
            if not c_aux:
                raise AuxResultMissing()
            if c_aux.metric.time != c.metric.time:
                raise RuntimeError("Auxiliary curve not in correct slice.")
    return c_aux


def _get_aux_curve(out_dir, hname, it, disp=True, verbose=False):
    r"""Load an auxiliary curve for the given iteration."""
    files = find_files(
        pattern="%s/%s_*_it%010d*.npy" % (out_dir, hname, it),
        skip_regex=r"_CE",
    )
    if not files:
        if disp:
            raise FileNotFoundError(
                "Auxiliary curve for horizon '%s' at iteration %s not found."
                % (hname, it)
            )
        return None
    files.sort()
    fname = files[-1]
    if verbose:
        print("Loading curve: %s" % fname)
    return BaseCurve.load(fname)


class _PropResult():
    r"""Simple storage class to help return results of computations."""

    def __init__(self, result, extras=None):
        r"""Store a result and optional extras in this instance."""
        self.result = result
        self.extras = extras


class AuxResultMissing(Exception):
    r"""Raised when a result is missing required for computing a property."""
    pass
