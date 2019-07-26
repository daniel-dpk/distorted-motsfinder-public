r"""@package motsfinder.metric.discrete.fdderivs

Wrapper class to compute derivatives using finite differences.
"""

import itertools
from abc import ABCMeta, abstractmethod

from six import add_metaclass
import numpy as np

from ...utils import lrange, lmap
from ..base import _ThreeMetric
from .numerical import fd_xz_derivatives, _get_Ty, _get_Txy_Tyy_Tyz
from .numerical import _get_fy, _get_fxy_fyy_fyz, _get_Vy, _get_Vxy_Vyy_Vyz


__all__ = [
    "FDDerivMetric",
]


class FDDerivMetric(_ThreeMetric):
    r"""Wrapper for computing derivatives using finite differences.

    @b Examples

    ```
        # Create a Kerr-Schild slice
        M = 0.5
        gKS = SchwarzschildKSSlice(M=M)
        g = FDDerivMetric(metric=gKS, res=256, fd_order=6)

        # We know the apparent horizon in this case.
        c = StarShapedCurve.create_sphere(radius=2*M, num=2, metric=g)
        c_AH = RefParamCurve.from_curve(c)

        # Check the expansion to verify it is a MOTS and check the accuracy.
        c_AH.plot_expansion(title=r"AH")

        # Compute the stability spectrum (uses 2nd derivatives).
        spectrum = c_AH.stability_parameter(num=30, full_output=True)[1]
        spectrum.print_max_l = 5
        print(spectrum)
    ```
    """

    def __init__(self, metric, res, fd_order=6):
        r"""Wrap a metric for computing derivatives using finite differences.

        @param metric
            The metric to wrap. All derivatives of the metric, the extrinsic
            curvature, lapse, shift (and their time derivatives) will be
            computed using finite differences, even if some of the derivatives
            are implemented analytically.
        @param res
            Grid resolution for the finite differences. Has no impact on
            performance or memory.
        @param fd_order
            Finite difference method order of accuracy. The higher the order,
            the larger the stencil. This has a performance impact but
            increases the accuracy at lower resolutions.
        """
        super().__init__()
        self.__metric = metric
        self.__res = res
        self.__dx = 1./res
        self.__stencil_size = fd_order + 1
        self.__field = _Sym2TensorField(
            _CallableMetric(metric),
            dx=self.__dx,
            stencil_size=self.__stencil_size,
        )
        self.__curv = self._mk_field(_Sym2TensorField, self.__metric.get_curv())
        self.__lapse = self._mk_field(_ScalarField, self.__metric.get_lapse())
        self.__dtlapse = self._mk_field(_ScalarField, self.__metric.get_dtlapse())
        self.__shift = self._mk_field(_VectorField, self.__metric.get_shift())
        self.__dtshift = self._mk_field(_VectorField, self.__metric.get_dtshift())

    @property
    def metric(self):
        return self.__metric

    @property
    def res(self):
        r"""Current resolution for the FD derivatives."""
        return self.__res

    @property
    def fd_order(self):
        r"""Convergence order of the FD derivatives."""
        return self.__stencil_size - 1

    @property
    def __fields(self):
        r"""All field objects (may include `None`)."""
        return [
            self.__field,
            self.__curv,
            self.__lapse,
            self.__dtlapse,
            self.__shift,
            self.__dtshift,
        ]

    def set_res(self, res):
        r"""Change the resolution for the FD derivatives.

        Note that this has no impact on performance or memory.
        """
        self.__res = res
        self.__dx = 1./res
        for field in self.__fields:
            if field:
                field.dx = self.__dx

    def set_fd_order(self, fd_order):
        r"""Change the order of the FD derivative method.

        Note that this *does* have an impact on performance. The higher the
        order, the more accurate the computations are at lower resolutions.
        Once the round-off errors dominate, increasing the resolution reduces
        the accuracy and different order methods have the same accuracy once
        both run into this regime. However, the higher order methods reach the
        point at which the round-off errors dominate faster, producing much
        better maximum accuracy for the right resolution.
        """
        self.__stencil_size = fd_order + 1
        for field in self.__fields:
            if field:
                field.stencil_size = self.__stencil_size

    def _mat_at(self, point):
        return self.__metric._mat_at(point)

    def diff(self, point, inverse=False, diff=1):
        if inverse:
            return self._compute_inverse_diff(point, diff=diff)
        return self.__field(point, diff=diff)

    def _mk_field(self, cls, field):
        r"""Create a field object if the field exists."""
        if not field:
            return None
        return cls(field, dx=self.__dx, stencil_size=self.__stencil_size)

    def get_curv(self):
        return self.__curv

    def get_lapse(self):
        return self.__lapse

    def get_dtlapse(self):
        return self.__dtlapse

    def get_shift(self):
        return self.__shift

    def get_dtshift(self):
        return self.__dtshift


@add_metaclass(ABCMeta)
class _FDField():
    r"""Base class for FD field wrappers."""
    def __init__(self, field, dx, stencil_size):
        r"""Create a new FD field object.

        @param field
            The (callable) field to wrap.
        @param dx
            Distance for finite difference derivatives.
        @param stencil_size
            Size of the FD stencil, determining the order of the method. For
            first and second derivatives, the order will be
            ``stencil_size-1``.
        """
        self.field = field
        self.dx = dx
        self.stencil_size = stencil_size

    def __call__(self, point, diff=0):
        if diff == 0:
            return self.field(point)
        dx = dz = self.dx
        if diff == 1:
            Tx, Tz = func_xz_derivatives(
                lambda point: self.field(point),
                point=point, dx=dx, dz=dz,
                derivs=[(1, 0), (0, 1)],
                stencil_size=self.stencil_size,
            )
            Ty = self._get_Ty(point)
            return np.asarray([Tx, Ty, Tz])
        if diff == 2:
            Txx, Tzz, Txz = func_xz_derivatives(
                lambda point: self.field(point),
                point=point, dx=dx, dz=dz,
                derivs=[(2, 0), (0, 2), (1, 1)],
                stencil_size=self.stencil_size,
            )
            Txy, Tyy, Tyz = self._get_Txy_Tyy_Tyz(point)
            return np.asarray([[Txx, Txy, Txz],
                               [Txy, Tyy, Tyz],
                               [Txz, Tyz, Tzz]])
        raise NotImplementedError("Derivative order %s" % diff)

    @abstractmethod
    def _get_Ty(self, point):
        r"""Compute the y-derivative of the field."""
        pass

    @abstractmethod
    def _get_Txy_Tyy_Tyz(self, point):
        r"""Compute 2nd derivatives involving the y-direction."""
        pass


class _Sym2TensorField(_FDField):
    r"""Wrapper for symmetric 2-tensor fields."""
    def _get_Ty(self, point):
        return _get_Ty(point, T=self.field(point))

    def _get_Txy_Tyy_Tyz(self, point):
        return _get_Txy_Tyy_Tyz(point, T=self(point, diff=0), dT=self(point, diff=1))


class _ScalarField(_FDField):
    r"""Wrapper for scalar fields."""
    def _get_Ty(self, point):
        return _get_fy()

    def _get_Txy_Tyy_Tyz(self, point):
        return _get_fxy_fyy_fyz(point, df=self(point, diff=1))


class _VectorField(_FDField):
    r"""Wrapper for (contravariant) vector fields."""
    def _get_Ty(self, point):
        return _get_Vy(point, V=self.field(point))

    def _get_Txy_Tyy_Tyz(self, point):
        return _get_Vxy_Vyy_Vyz(point, V=self(point), dV=self(point, diff=1))


def func_xz_derivatives(func, point, dx, dz, derivs, stencil_size=5):
    r"""Compute finite difference derivatives of a function at a point.

    This uses finite difference (FD) derivatives to differentiate a scalar,
    vector, or matrix-valued function `func`.

    @b Examples

    ```
        f = lambda p: p[0]**3 + p[1]**2 + p[2]
        dxf = lambda p: func_xz_derivatives(f, p, 1e-6, 1e-6, [[1, 0]])
        plot_1d(lambda x: dxf([x,0,0])[0], lambda x: 3*x**2,
                title=r"$x$-derivative",
                domain=(-2, 2), label=("FD", "exact"))
    ```

    An example for a matrix-valued function:

    ```
        def f(p):
            x, y, z = p
            return np.array([[x**2*z,    x*x,    z*z],
                             [     x,      y,      z],
                             [  z**3, z**3*x, x**2*z]])
        df = lambda p: func_xz_derivatives(f, p, 1e-6, 1e-6, [[1, 0], [0, 1]])
        plot_1d(lambda x: df([x, 0, 1])[0][0, 0], lambda x: 2*x*1,
                title=r"$x$-derivative of $xx$-component",
                domain=(-2, 2), label=("FD", "exact"))
        plot_1d(lambda x: df([x, 0, 1])[1][2, 2], lambda x: x**2,
                title=r"$z$-derivative of $zz$-component",
                domain=(-2, 2), label=("FD", "exact"))
    ```
    """
    nx = nz = int((stencil_size-1)/2)
    if not any(d[0] for d in derivs):
        nx = 0
    if not any(d[1] for d in derivs):
        nz = 0
    val = func(point)
    if nx == nz == 0:
        return np.array([val for _ in derivs])
    def f(i, j):
        if i == j == 0:
            return val
        return func([point[0] + i*dx, point[1], point[2] + j*dz])
    mat = np.asarray(
        [[[f(i, j) for j in range(-nz, nz+1)]] for i in range(-nx, nx+1)]
    )
    def _diff(idx):
        return [result[0, 0, 0] for result in fd_xz_derivatives(
            mat[np.s_[:,:,:] + idx], region=[[nx], [0], [nz]], dx=dx, dz=dz,
            derivs=derivs, stencil_size=stencil_size
        )]
    shape = np.asarray(val).shape
    if not shape:
        return _diff(())
    results = []
    for idx in itertools.product(*lmap(lrange, shape)):
        results.append(_diff(idx))
    results = np.asarray(results).reshape(*shape, len(derivs))
    return np.moveaxis(results, [-1], [0])


class _CallableMetric():
    r"""Simple wrapper class to make metric fields callable."""

    def __init__(self, metric):
        self.metric = metric

    def __call__(self, point, diff=0):
        return self.metric.diff(point, diff=diff)
