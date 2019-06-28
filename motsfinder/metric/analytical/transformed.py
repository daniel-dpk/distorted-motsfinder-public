r"""@package motsfinder.metric.analytical.transformed

Implement a coordinate-transformed metric.
"""

import numpy as np
from scipy import linalg

from ..base import _ThreeMetric


__all__ = [
    "TransformedMetric",
    "transformation_preset",
]


class TransformedMetric(_ThreeMetric):
    def __init__(self, metric, x, x_u, x_w, z=None, z_u=None, z_w=None, **kw):
        r"""Create a transformed metric.

        All the transformation functions should take primed arguments `u, w`
        and return a scalar.

        @param x,z
            Inverse transformation functions (i.e. x w.r.t. `x_prime` and
            `z_prime`, etc.).
        @param x_u,x_w
            Derivatives of the inverse x-transform w.r.t. `x_prime` and
            `z_prime`.
        @param z_u,z_w
            Derivatives of the inverse z-transform w.r.t. `x_prime` and
            `z_prime`.
        """
        preset = kw.pop('_preset', None)
        params = kw.pop('_preset_params', dict())
        if kw:
            raise TypeError("Invalid parameters: %s" % ", ".join(kw.keys()))
        super().__init__()
        self._g = metric
        self._x = x
        self._z = z
        self._x_u = x_u
        self._x_w = x_w
        self._z_u = z_u
        self._z_w = z_w
        self._preset = preset
        self._preset_params = params
        self._fix_functions()

    @classmethod
    def from_preset(cls, metric, preset, **params):
        r"""Classmethod to create a transformed metric from a preset."""
        funcs = transformation_preset(preset, **params)
        return cls(metric, **funcs, _preset=preset, _preset_params=params)

    def __getstate__(self):
        r"""Return a picklable state object."""
        state = self.__dict__.copy()
        if self._preset is not None:
            state['_x'] = None
            state['_z'] = None
            state['_x_u'] = None
            state['_x_w'] = None
            state['_z_u'] = None
            state['_z_w'] = None
        return state

    def __setstate__(self, state):
        r"""Restore this object from the given unpickled state."""
        self.__dict__.update(state)
        if self._preset is not None:
            funcs = transformation_preset(self._preset, **self._preset_params)
            for key, func in funcs.items():
                setattr(self, "_"+key, func)
            self._fix_functions()

    @property
    def g(self):
        r"""Original (non-transformed) metric."""
        return self._g

    def _fix_functions(self):
        if self._z is None and self._z_u is None and self._z_w is None:
            self._z = lambda x, z: z
            self._z_u = lambda x, z: 0.0
            self._z_w = lambda x, z: 1.0

    def forward_transform(self):
        r"""Return the two functions to transform from `x,z` to `u,w` coordinates."""
        if self._preset is None:
            raise ValueError("Not a preset-based transformation.")
        funcs, inverse = transformation_preset(
            self._preset, **self._preset_params,
            full_output=True,
        )
        return inverse[0], inverse[1]

    def backward_transform(self):
        r"""Return the two functions going from `u,w` to `x,z`."""
        return self._x, self._z

    def _transform(self, point):
        u, _, w = point
        x_u = self._x_u(u, w)
        x_w = self._x_w(u, w)
        z_u = self._z_u(u, w)
        z_w = self._z_w(u, w)
        T = np.asarray([
            [x_u, 0.0, z_u],
            [0.0, x_u, 0.0],
            [x_w, 0.0, z_w],
        ])
        x = self._x(u, w)
        z = self._z(u, w)
        return np.array([x, 0.0, z]), T

    def _mat_at(self, point):
        p, T = self._transform(point)
        mat = self._g._mat_at(p)
        return np.einsum('ik,jl,kl->ij', T, T, mat)

    def diff(self, point, inverse=False, diff=1):
        if inverse:
            return self._compute_inverse_diff(point, diff=diff)
        if diff == 0:
            return self._mat_at(point)
        raise NotImplementedError

    def get_curv(self):
        curv = self._g.get_curv()
        return _CurvWrapper(curv, self._transform)

    def get_lapse(self):
        lapse = self._g.get_lapse()
        return _LapseWrapper(lapse, self._transform)

    def get_shift(self):
        shift = self._g.get_shift()
        return _ShiftWrapper(shift, self._transform)


def transformation_preset(preset, full_output=False, **params):
    r"""Create transformations suitable for TransformedMetric.

    The transformations here modify the x and/or z coordinate such that MOTSs
    appear highly distorted even in case of a Schwarzschild slice. The purpose
    is purely to test the performance and accuracy of the MOTS finder and the
    quantities can be computed.

    One advantage of using these presets over ad-hoc defined functions (i.e.
    not in a module) is that the resulting MOTS curves can be "pickled", i.e.
    stored to disk.
    """
    sinh, cosh = np.sinh, np.cosh
    if preset == "none":
        def _mk():
            return dict(
                x=  lambda u, w: u,
                x_u=lambda u, w: 1.0,
                x_w=lambda u, w: 0.0,
                z=  lambda u, w: w,
                z_u=lambda u, w: 0.0,
                z_w=lambda u, w: 1.0,
            ), (
                lambda x, z: x, # u(x,z)
                lambda x, z: z, # w(x,z)
            )
    elif preset == "stretch":
        def _mk(a=1, b=1):
            return dict(
                x=  lambda u, w: u / a,
                x_u=lambda u, w: 1.0 / a,
                x_w=lambda u, w: 0.0,
                z=  lambda u, w: w / b,
                z_u=lambda u, w: 0.0,
                z_w=lambda u, w: 1.0 / b,
            ), (
                lambda x, z: a*x, # u(x,z)
                lambda x, z: b*z, # w(x,z)
            )
    if preset == "pinched-x":
        def _mk(beta, gamma, z0=0.0):
            return dict(
                x=  lambda u, w: u / (1 - beta/cosh((w-z0)/gamma)),
                x_u=lambda u, w: 1 / (1 - beta/cosh((w-z0)/gamma)),
                x_w=lambda u, w: - (beta*u*sinh((w-z0)/gamma)) / (gamma*(beta-cosh((w-z0)/gamma))**2),
                z=  lambda u, w: w,
                z_u=lambda u, w: 0.0,
                z_w=lambda u, w: 1.0,
            ), (
                lambda x, z: x * (1 - beta/cosh((z-z0)/gamma)), # u(x,z)
                lambda x, z: z,                                 # w(x,z)
            )
    funcs, inverse = _mk(**params)
    if full_output:
        return funcs, inverse
    return funcs
    raise ValueError("Unknown preset '%s'" % (preset,))


class _CurvWrapper():
    r"""Wrapper class to make pickling/unpickling of `K` possible."""
    def __init__(self, curv, transform):
        self.curv = curv
        self._transform = transform

    def __call__(self, point, diff=0):
        if diff != 0:
            raise NotImplementedError
        p, T = self._transform(point)
        K = self.curv(p)
        return np.einsum('ik,jl,kl->ij', T, T, K)


class _LapseWrapper():
    r"""Wrapper class to make pickling/unpickling of `alpha` possible."""
    def __init__(self, lapse, transform):
        self.lapse = lapse
        self._transform = transform

    def __call__(self, point, diff=0):
        if diff != 0:
            raise NotImplementedError
        p, T = self._transform(point)
        alpha = self.lapse(p)
        return alpha


class _ShiftWrapper():
    r"""Wrapper class to make pickling/unpickling of `beta` possible."""
    def __init__(self, shift, transform):
        self.shift = shift
        self._transform = transform

    def __call__(self, point, diff=0):
        if diff != 0:
            raise NotImplementedError
        p, T = self._transform(point)
        Tinv = linalg.inv(T)
        beta = self.shift(p)
        return np.einsum('ik,k->i', Tinv.T, beta)
