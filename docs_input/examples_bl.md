# Example: MOTSs in Brill-Lindquist initial data

> **IMPORTANT**
>
> See [Preparation to running the examples](./examples_init.md) for setting up (necessary imports, etc).


## Example 0 -- Creating a metric

This shows how to create various metrics to use with the MOTS finder.
No MOTS is found here.

```.py
# Schwarzschild `t=const` slice in isotropic coordinates:
g = SchwarzschildSliceMetric(2.0) # the only parameter is mass

# Schwarzschild `t=const` slice in Kerr-Schild coordinates:
g = SchwarzschildKSSlice(2.0) # the only parameter is mass

# Brill-Lindquist metric of two non-spinning BHs at rest:
g = BrillLindquistMetric(d=0.5, m1=0.2, m2=0.8)

# Kerr `t=const` slice in Kerr-Schild coordinates:
gKerr = KerrKSSlice(M=1.0, a=0.5)
# NOTE: The above can only be used with finite difference differentiation.
#       See below for details.

# Metric from numerical simulation data in SimulationIO format:
g = SioMetric('the/simulation/file.it0000001234.s5')

# Metric from a set of component matrices.
# See the documentation of `motsfinder.discrete.construct.metric_from_data()`.
fname_data = 'data/my_numerical_metric_data.npy'
g = metric_from_data(
    fname_data, res=res, origin=origin,
    metric=[g_xx, g_xy, g_xz, g_yy, g_yz, g_zz],
    curv=[K_xx, K_xy, K_xz, K_yy, K_yz, K_zz],
    lapse=lapse,
    shift=[shift_x, shift_y, shift_z],
)
```

If the metric does not support computing its first and second derivatives
(e.g. the motsfinder.metric.analytical.kerrks.KerrKSSlice metric),
you may wrap it in a finite difference calculator like this:

```.py
gKerr = KerrKSSlice(M=1.0, a=0.5)
g = FDDerivMetric(
    metric=gKerr, # metric to discretize
    res=192,      # resolution of finite difference method
    fd_order=6    # order of the finite difference method (can be 2, 4, 6, 8)
)
```


## Example 1 -- Axisymmetric non-star-shaped

Here, we consider a Brill-Lindquist two black hole metric with mass parameters
`m1 = 0.2`, `m2 = 0.8`, and distance parameter `d = 0.5`
(see Ref [1] for more details on Brill-Lindquist data).

```.py
# This defines the metric for a mass-ratio 1:4 system with d = 0.5 M.
g = BrillLindquistMetric(d=0.5, m1=0.2, m2=0.8)

# The initial guess dictates which MOTS is found. We create a sphere of a
# suitable radius to find the apparent horizon (AH).
initial_guess = StarShapedCurve.create_sphere(
    radius=(g.m1 + g.m2)/2. + g.d/3., # just some guess based on parameters
    origin=(0, 0, 0),
)

# To use the finder, we first create a search configuration.
# This is highly configurable, but we use one of the "presets" to have good
# defaults.
cfg = GeneralMotsConfig.preset(
    'general',      # The preset to use. Any further settings override the
                    # preset settings.
    metric=g,
    num=50,         # Initial resolution for the horizon function.
    atol=1e-12,     # Tolerance. The residual expansion must be below this
                    # value for the finder to stop.
    reparam=False,  # Do not reparameterize the reference curve.
    verbose=True,   # Print convergence messages.
    #plot_steps=True, # If you want to see all the Newton steps.
    #out_folder="data/my_MOTSs/", # If you want to save the found MOTS.
    #suffix="BL_d%s_m%s_%s" % (g.d, g.m1, g.m2), # Make filename unique.
)

# Start the actual finder.
c_AH = find_mots(
    cfg,
    prefix='AH',         # The "horizon name" to be used as file name prefix.
    c_ref=initial_guess, # Use the initial guess as reference curve.
)

# Plot the curve and how the coefficients converge.
c_AH.plot()
c_AH.plot_coeffs()

# We can also plot the expansion along the curve.
c_AH.plot_expansion()
```

As a next step, we find the two individual horizons.

```.py
# Find the smaller individual MOTS at z=+d/2.
c_smaller = find_mots(
    cfg,
    prefix='smaller', # instead of 'AH'
    c_ref=StarShapedCurve.create_sphere(
        radius=min(g.m1/2., g.d/2.), # just some heuristics
        origin=g.d/2.,               # center the sphere on the upper puncture
    ),
)

# Find the larger individual MOTS at z=-d/2.
c_larger = find_mots(
    cfg,
    prefix='larger',
    c_ref=StarShapedCurve.create_sphere(
        radius=min(g.m2/2., g.d/2.),
        origin=-g.d/2.,              # center the sphere on the lower puncture
    ),
)

# Plot all three MOTSs together.
BaseCurve.plot_curves(
    (c_AH, r"$\mathcal{S}_{\rm outer}$", '-b'),
    (c_smaller, r"$\mathcal{S}_1$", ':m'),
    (c_larger, r"$\mathcal{S}_2$", '--r'),
    dpi=100, grid=False, legendargs=dict(loc=4),
)
```

Finding the inner common MOTS is easier at larger distances where it is much
closer to the AH in shape. We can then vary the distance step by step and use
the previous inner common MOTS as reference surface (and thus starting
surface) for the search in the next step.

```.py
# First, find the AH when d=0.67. We'll use the previous AH as initial guess.
# Note that we downsample the reference curve significantly (using `ref_num`).
g_larger_d = BrillLindquistMetric(d=0.67, m1=g.m1, m2=g.m2)
c_AH_larger_d = find_mots(
    cfg,
    metric=g_larger_d,
    c_ref=c_AH,
    ref_num=10, # this downsamples the reference curve to smooth it out
)

# Next we find the inner common MOTS by stepping inward a little.
# The `offset_coeffs` define the coefficients `a_n` of the function we add:
#     sum_n{a_n cos(n lambda)}
c_inner_larger_d = find_mots(
    cfg, num=200, metric=g_larger_d, c_ref=c_AH_larger_d, ref_num=10,
    offset_coeffs=(-0.2, 0, 0.05),
    step_mult=0.3, # do smaller steps initially to help with converging
    #plot_steps=True,
)

# Now do a few steps in `d` to get to the desired value defined above.
# The argument `reparam='curv2'` activates curvature-based reparameterization
# of the reference curve to make sure the collocation points are evenly
# distributed (with higher density in highly curved regions).
c_inner = c_inner_larger_d.copy()
for d in np.linspace(0.67, g.d, num=5):
    g_interim = BrillLindquistMetric(d=d, m1=g.m1, m2=g.m2)
    c_inner = find_mots(
        cfg,
        metric=g_interim, # use the temporary metric
        num=c_inner.num,  # use previous MOTS's resolution initially
        c_ref=c_inner,    # use the previous MOTS as initial guess
        ref_num=25,
        reparam='curv2',  # enable reparameterization
    )

# Plot everything together.
BaseCurve.plot_curves(
    (c_AH, r"$\mathcal{S}_{\rm outer}$", '-b'),
    (c_smaller, r"$\mathcal{S}_1$", ':m'),
    (c_larger, r"$\mathcal{S}_2$", '--r'),
    (c_inner, r"$\mathcal{S}_{\rm inner}$", '-g'),
    dpi=100, grid=False, legendargs=dict(loc=4),
)
```

From here it is easy to reduce `d` step by step to get to the point where the
inner common MOTS and the bottom MOTS disappear. This will require choosing
smaller and smaller steps in `d`.


### Physical quantities

We can plot, e.g., the Ricci scalar of each curve like this:

```.py
plot_1d(c_AH.ricci_scalar, points=100, domain=(0, np.pi), value_pad=1e-6,
        title=r"Ricci Scalar $\mathcal{R}$ of AH")
```

Other quantities are computed simply via:

```.py
print("Stability of larger MOTS:", c_larger.stability_parameter())
print("Area of inner MOTS:", c_inner.area())

_, spectrum = c_inner.stability_parameter(full_output=True)
s1, s2 = sorted(spectrum.get().real)[:2]
print("Stability parameter of inner MOTS:", s1)
print("2nd stability eigenvalue of inner MOTS:", s2)

# Print the spectrum including all the higher angular modes.
spectrum.print_max_l = 5 # print only the first few ones
print(spectrum)
```


## Example 2 -- (Almost) fully automated

For a two-black-hole configuration like in Brill-Lindquist, there is a
convenience class motsfinder.axisym.initialguess.InitHelper, which automates
many of the above steps. It also does not need to modify the metric in order
to find the inner common MOTS and instead takes a sequence of constant
expansion surfaces to deform the apparent horizon into the inner common MOTS.

In many cases, this `InitHelper` successfully finds the four MOTSs we already
found above (in case they all exist). If not, several parameters can be
tweaked allowing it to find the four MOTSs in many other cases too.

Here is a very simple example:

```.py
g = BrillLindquistMetric(d=0.5, m1=0.2, m2=0.8)
h = InitHelper(
    metric=g, out_base="data/my_MOTSs/",
    suffix="d%s_m%s_%s" % (g.d, g.m1, g.m2)
)
curves = h.find_four_MOTSs(m1=g.m1, m2=g.m2, d=g.d, plot=True)
```

The generated MOTSs are not found to very high accuracy but they are perfectly
suited as initial guesses (i.e. reference curves) for a more precise search:

```.py
cfg = GeneralMotsConfig.preset(
    'general', metric=g, reparam='curv2', ref_num=15, verbose=True,
)
curves_accurate = []
for c_ref in curves:
    c = find_mots(cfg, c_ref=c_ref, num=c_ref.num)
    curves_accurate.append(c)

BaseCurve.plot_curves(
    (curves_accurate[0], r"$\mathcal{S}_{\rm outer}$", '-b'),
    (curves_accurate[1], r"$\mathcal{S}_1$", ':m'),
    (curves_accurate[2], r"$\mathcal{S}_2$", '--r'),
    (curves_accurate[3], r"$\mathcal{S}_{\rm inner}$", '-g'),
    dpi=100, grid=False, legendargs=dict(loc=4),
)
```


# References:

[1] Brill, Dieter R., Lindquist, R. W. "Interaction Energy in
    Geometrostatics." Phys. Rev. 131, 471 (1963).
    https://journals.aps.org/pr/abstract/10.1103/PhysRev.131.471
