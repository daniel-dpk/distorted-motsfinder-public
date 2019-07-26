# Example: Using the finder with a numerical metric

See [preparation to running the examples](./examples_init.md) for setting up.

This example considers the case that you have numerical data defining an
axisymmetric slice of spacetime. Further, we assume you have the components of
the metric and extrinsic curvature each as a 2- or 3-dimensional array
containing the component values for a grid covering the full domain containing
any potential MOTS you would like to find.

This means, we have 6 matrices of shape `(N, N)` for the metric:

```.py
metric = [
    np.array(...),  # xx component, shape=(N,N)
    np.array(...),  # xy component, shape=(N,N)
    np.array(...),  # xz component, shape=(N,N)
    np.array(...),  # yy component, shape=(N,N)
    np.array(...),  # yz component, shape=(N,N)
    np.array(...),  # zz component, shape=(N,N)
]
```

And the same for the extrinsic curvature:

```.py
curv = [
    np.array(...),  # xx component, shape=(N,N)
    np.array(...),  # xy component, shape=(N,N)
    np.array(...),  # xz component, shape=(N,N)
    np.array(...),  # yy component, shape=(N,N)
    np.array(...),  # yz component, shape=(N,N)
    np.array(...),  # zz component, shape=(N,N)
]
```

This allows us to define the slice data in a format understood by the MOTS
Finder using motsfinder.metric.discrete.construct.metric_from_data():

```.py
g = metric_from_data(
    "data/my_numerical_metric_data.npy",    # file to store bulk data
    res=256,                                # resolution 1/h of the grid
    origin=-5.0,                            # location of the [0,0] grid point
    metric=metric,
    curv=curv,
)
```

Now we can use the finder to look for a MOTS:

```.py
num = 50      # spectral resolution of the horizon function
radius = 1.5  # coordinate radius of the initial guess shape
c = find_mots(GeneralMotsConfig.preset(
    "discrete8",
    metric=g,
    hname='my_first_MOTS',
    out_folder='data/MOTSs',
    c_ref=StarShapedCurve.create_sphere(radius, num=num, metric=g),
    ref_num=2, reparam=False,
    verbose=True, plot_steps=True,
))
```

... and compute some of its properties:

```.py
print("Area: %s" % c.area())

# This will generate warnings about missing lapse and shift.
# However, for the stability spectrum the trivial implementations are fine to
# use as the spectrum is invariant under changes of lapse and shift.
_, spectrum = c.stability_parameter(full_output=True)
spectrum.print_l_max = 5
print(spectrum)
```
