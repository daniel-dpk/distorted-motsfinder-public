# Example: MOTSs in Brill-Lindquist initial data

See [preparation to running the examples](./examples_init.md) for setting up.


## Example 1 -- Axisymmetric non-star-shaped

Here, we consider a Brill-Lindquist two black hole metric with mass parameters
`m1 = 0.2`, `m2 = 0.8`, and distance parameter `d = 0.5`.

```.py
# This defines our setup of a mass-ratio 1:4 system with d = 0.5 M.
cfg = BrillLindquistConfig(d=0.5, m1=0.2, m2=0.8)

# The initial guess dictates which MOTS is found. The first one finds the
# apparent horizon (AH) starting with a resolution of 50.
initial_radius = (cfg.m1+cfg.m2)/2. + cfg.d/3.
initial_origin = 0.0
c_AH = find_mots(cfg, num=50, c_ref=(initial_radius, initial_origin),
                 verbose=True)

# Plot the curve and how the coefficients converge.
c_AH.plot()
c_AH.plot_coeffs()

# We can also plot the expansion along the curve.
c_AH.plot_expansion()
```

As a next step, we find the two individual horizons.

```.py
initial_radius = min(cfg.m1/2., cfg.d/2.)
initial_origin = cfg.d/2.
c_smaller = find_mots(cfg, num=50, c_ref=(initial_radius, initial_origin),
                      verbose=True)

initial_radius = min(cfg.m2/2., cfg.d/2.)
initial_origin = -cfg.d/2.
c_larger = find_mots(cfg, num=50, c_ref=(initial_radius, initial_origin),
                     verbose=True)

# Plot all three MOTSs together.
BaseCurve.plot_curves(
    (c_AH, 'AH', '-b'),
    (c_smaller, 'smaller MOTS', ':m'),
    (c_larger, 'larger MOTS', '--r'),
)
```

Finding the inner common MOTS is easier at larger distances where it is much
closer to the AH in shape. We can then vary the distance step by step and use
the previous inner common MOTS as reference surface (and thus starting
surface) for the search in the next step.

```.py
# First, find the AH at d=0.67. We'll use the previous AH as starting surface.
# Note that we downsample the reference curve significantly (using `ref_num`).
c_AH_larger_d = find_mots(cfg, d=0.67, c_ref=c_AH, ref_num=10, verbose=True)

# Next we find the inner common MOTS by stepping inward a little.
# The `offset_coeffs` define the coefficients `a_n` of the function we add:
#     sum_n{a_n cos(n lambda)}
c_inner_larger_d = find_mots(
    cfg, num=200, d=0.67, c_ref=c_AH_larger_d, ref_num=10,
    offset_coeffs=(-0.2, 0, 0.05),
    verbose=True
)

# Now do a few steps in `d` to get to the desired value defined above.
# The argument `reparam=True` activates arc-length reparameterization of the
# reference curve to make sure the collocation points are evenly distributed.
c_inner = c_inner_larger_d.copy()
for d in np.linspace(0.67, cfg.d, num=5):
    c_inner = find_mots(cfg, d=d, c_ref=c_inner, ref_num=25,
                        reparam=True, verbose=True)

# Plot everything together.
BaseCurve.plot_curves(
    (c_AH, 'AH', '-b'),
    (c_smaller, 'smaller MOTS', ':m'),
    (c_larger, 'larger MOTS', '--r'),
    (c_inner, 'inner common MOTS', '-g'),
)
```

From here it is easy to reduce `d` step by step to get to the point where the
inner common MOTS and the bottom MOTS disappear. This will require choosing
smaller and smaller steps in `d` and also eventually allowing for a higher
maximum spectral resolution.


### Physical quantities

We can plot e.g. the ricci scalar of each curve like this:

```.py
plot_1d(c_AH.ricci_scalar, points=100, domain=(0, np.pi), value_pad=1e-6,
        title=r"Ricci Scalar $\mathcal{R}$ of AH")
```

Other quantities are computed simply via:

```.py
print("Stability of larger MOTS:", c_larger.stability_parameter())
print("Area of inner MOTS:", c_inner.area())

_, eigenvalues = c_inner.stability_parameter(full_output=True)
s1, s2 = sorted(eigenvalues.real)[:2]
print("Stability parameter of inner MOTS:", s1)
print("2nd stability eigenvalue of inner MOTS:", s2)
```


## Example 2 -- More manual control

The first example used convenience contructs like the `BrillLindquistConfig`
and the `find_mots()` function. These are just wrappers that internally
construct the actual metric object and invoke the `newton_kantorovich()`
function to solve for the MOTS.

This second example avoids these high-level constructs to demonstrate what
actually happens.

We consider a Brill-Lindquist two black hole metric with mass parameters
`m1 = 1`, `m2 = 1`, and distance parameter `d = 0.75`. The star-shaped
approach seems to have problems finding the inner common MOTS as it is either
not star-shaped or too close to not being star-shaped anymore.

We first find the outermost (i.e. apparent) horizon `AH` using the assumption
that it is star-shaped. The inner individual MOTS are found the same way, by
moving the origin of the finder around.

To then find the inner common MOTS, we first find an inner common MOTS for a
larger distance of `d = 1.4` (where it is still star-shaped), very similar to
the above example.

```.py
d = 0.75
d_start = 1.4
num = 300 # initial spectral resolution of inner common MOTS

# Construct the metric
metric = BrillLindquistMetric(m1=1, m2=1, d=d, axis='z')

# This is the metric for finding the starting shape of our
# inner horizon search.
metric_start = BrillLindquistMetric(m1=1, m2=1, d=d_start, axis='z')

with timethis("Finding apparent horizon (AH)..."):
    c_AH = newton_kantorovich(
        StarShapedCurve.create_sphere(radius=2.0, num=50, metric=metric)
    )
with timethis("Finding individual horizons..."):
    c_BH = newton_kantorovich(
        StarShapedCurve.create_sphere(radius=d/2, num=50, metric=metric,
                                      origin=(0, d/2)),
    )
with timethis("Finding star-shaped inner common horizon for d=%s..." % d_start):
    c0 = c_AH.copy().resample(50)
    c0.metric = metric_start
    c_AH_start = newton_kantorovich(c0, steps=5, disp=False,
                                    auto_resolution=False)
    c0 = c_AH_start.copy()
    c0.h.a_n[0] -= 0.25
    c_inner_start = newton_kantorovich(
        c0, steps=5, step_mult=0.2, disp=False,
        #plot_steps=True, reference_curves=([(c_AH_start, 'AH start')]),
        #verbose=True
    )

# Plot what we have so far.
ax = c_AH.plot(l='-b', figsize=(6, 6), label='AH', show=False)
c_BH.plot(l='-r', copy_y=True, label='individual MOTS', ax=ax, show=False)
c_inner_start.plot(l='--g', label='starting surface', ax=ax)

# Represent the starting surface in our new relative coordinates.
c0 = RefParamCurve.from_star_shaped(c_inner_start, c_AH, num=num,
                                    metric=metric)

# The next step may create many plots (if `plot_steps` is `True`), for which
# 'inline' is better.
%matplotlib inline

# Run the Newton search for the inner horizon.
c_inner = newton_kantorovich(
    c0, step_mult=0.2, linear_regime_threshold=0.3,
    verbose=True,
    #plot_steps=True, plot_deltas=True,
    reference_curves=([
        (c_AH, dict(l='-b', label='AH')),
        (c_inner_start, dict(l='--g', label='inner MOTS for $d=%s$' % d_start)),
        (c_BH, dict(l='-r', copy_y=True, label='individual MOTS')),
    ]),
)

# Plot the result.
ax = c_AH.plot(l='-b', figsize=(5, 5), label='AH', show=False)
c_BH.plot(l='-r', copy_y=True, label='individual MOTS', ax=ax, show=False)
c_inner.plot(l='-g', points=c_inner.h.collocation_points(),
             label=r"common MOTS (non-star)", ax=ax, legendargs=dict(loc=1))
```
