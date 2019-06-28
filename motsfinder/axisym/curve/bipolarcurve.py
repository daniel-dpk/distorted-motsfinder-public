r"""@package motsfinder.axisym.curve.bipolarcurve

Parametric curve in bipolar coordinates.
"""

from math import sin, cos, sinh, cosh

import numpy as np

from ...exprs.basics import DivisionExpression, ScaleExpression
from ...exprs.basics import OffsetExpression, SumExpression
from ...exprs.basics import SimpleSinExpression, SimpleCosExpression
from ...exprs.basics import SimpleSinhExpression, SimpleCoshExpression
from ...exprs.trig import SineSeries, CosineSeries
from .baseparamcurve import BaseParametricCurve


__all__ = [
    "BipolarCurve",
]


class BipolarCurve(BaseParametricCurve):
    r"""Parametric curve in bipolar coordinates.

    Instead of parameterizing the x and z-coordinate functions directly (as is
    done in .curve.parametriccurve.ParametricCurve), we parameterize functions
    \f$s(\lambda), t(\lambda)\f$ and transform the resulting curve in
    t-s-coordinate-space to the x-z-coordinate-space using bipolar coordinates
    of the form
    \f[
        x(s, t) = a \frac{\sin s}{\cosh t - \cos s}, \qquad
        z(s, t) = a \frac{\sinh t}{\cosh t - \cos s} + c.
    \f]

    The scaling `a` and shift `c` should be chosen appropriately for the curve
    to represent.

    If the component functions `s` and `t` support it, this curve can compute
    up to 4th derivatives of the cartesian x-z-components (i.e. up to 3rd
    derivative of the normal vector).
    """

    def __init__(self, s_fun, t_fun, scale=1.0, move=0.0, freeze=True,
                 name=''):
        r"""Construct a curve from component functions s and t.

        @param s_fun,t_fun
            Component functions.
        @param scale
            Scaling parameter `a`, i.e. the half-distance of the two foci of
            the coordinates.
        @param move
            Offset of the center of the two foci on the z-axis.
        @param freeze
            If `True` (default) cache the evaluators of component functions.
            Changes to the component function (e.g. due to reparameterization,
            resampling, or even direct modifications to coefficients) will
            *not* be noticed until a refresh is done e.g. using
            force_evaluator_update(). Using functions of this class or its
            base classes to modify the functions should automatically perform
            this update.
        @param name
            Name of the curve (used e.g. for labels in some plots).
        """
        super().__init__(s_fun=s_fun, t_fun=t_fun, freeze=freeze, name=name)
        self.scale = scale
        self.move = move

    @classmethod
    def from_curve(cls, curve, num, s_cls=SineSeries, t_cls=CosineSeries,
                   scale=1.0, move=0):
        r"""Class method to create a bipolar curve from any other curve.

        The given `curve` is sampled along its affine parameter and the
        x-z-coordinates are transformed to the s-t-coordinate space. These
        results are then used to construct independent functions for the s and
        the t coordinate. The `scale` and `move` parameters are used to
        define the bipolar coordinates (see #__init__() for a description).
        """
        s_fun, t_fun = cls.construct_transforms(scale=scale, move=move)
        with curve.fix_evaluator():
            fs = lambda la: s_fun(*curve(la))
            ft = lambda la: t_fun(*curve(la))
            s_fun = s_cls.from_function(fs, num=num, domain=curve.domain, lobatto=False)
            t_fun = t_cls.from_function(ft, num=num, domain=curve.domain, lobatto=False)
            return cls(s_fun, t_fun, scale=scale, move=move)

    def copy(self):
        return type(self)(t_fun=self.t_fun.copy(), s_fun=self.s_fun.copy(),
                          scale=self.scale, move=self.move, name=self.name)

    def _add_z_offset(self, z_offset):
        self.move += z_offset

    @classmethod
    def construct_transforms(cls, scale, move):
        r"""Return the two function transforming from `x, z` to `s` and  `t`, respectively."""
        a = scale
        c = move
        def s_fun(x, z):
            r"""s-component for given `x, z` values."""
            z = z - c
            azx = a**2 - z**2 - x**2
            return (
                np.pi
                -2 * np.arctan2(
                    2*a*x, azx + np.sqrt(azx**2 + 4*a**2*x**2)
                )
            )
        def t_fun(x, z):
            r"""t-component for given `x, z` values."""
            z = z - c
            return 0.5 * np.log(((z+a)**2 + x**2) / ((z-a)**2 + x**2))
        return s_fun, t_fun

    def get_transforms(self):
        r"""Return the two function transforming from `x, z` to `s` and  `t`, respectively."""
        return self.construct_transforms(scale=self.scale, move=self.move)

    def get_back_transforms(self):
        r"""Return the two function transforming from `s, t` to `x` and  `z`, respectively."""
        a = self.scale
        c = self.move
        def x_fun(s, t):
            r"""x-component for given `s, t` values."""
            return a * sin(s) / (cosh(t) - cos(s))
        def z_fun(s, t):
            r"""z-component for given `s, t` values."""
            return a * sinh(t) / (cosh(t) - cos(s)) + c
        return x_fun, z_fun

    def __call__(self, param, xyz=False):
        s_ev, t_ev = self._get_evaluators()
        a = self.scale
        c = self.move
        s = s_ev(param)
        t = t_ev(param)
        denom = cosh(t) - cos(s)
        x = a * sin(s) / denom
        z = a * sinh(t) / denom + c
        return self._prepare_result(x, z, xyz)

    def _diff(self, param, diff):
        r"""Compute derivatives of the x- and z-components of this curve.

        Derivatives are taken w.r.t. affine parameter `param`. The formulas
        implemented here are trivial (though somewhat laborious) applications
        of the chain and product rule of derivatives applied to the formulas
        in the class description (see BipolarCurve).
        """
        s_ev, t_ev = self._get_evaluators()
        a = self.scale
        s = [s_ev.diff(param, n) for n in range(diff+1)]
        t = [t_ev.diff(param, n) for n in range(diff+1)]
        # compute d/dl of f_i(l)/g(l), where
        #   f_0(l) = sin(s(l))
        #   f_1(l) = sinh(t(l))
        #   g(l) = cosh(t(l)) - cos(s(l))
        sinh_t = sinh(t[0])
        cosh_t = cosh(t[0])
        sin_s = sin(s[0])
        cos_s = cos(s[0])
        f = [sin_s, sinh_t]
        df = [cos_s*s[1], cosh_t*t[1]]
        g = cosh_t - cos_s
        dg = sinh_t*t[1] + sin_s*s[1]
        if diff == 1:
            return [a*(df[i] / g - f[i]*dg / g**2) for i in (0, 1)]
        ddf = [
            -sin_s*s[1]**2 + cos_s*s[2],
            sinh_t*t[1]**2 + cosh_t*t[2]
        ]
        ddg = cosh_t*t[1]**2 + sinh_t*t[2] + cos_s*s[1]**2 + sin_s*s[2]
        if diff == 2:
            return [
                a * (
                    ddf[i]/g - (2*df[i]*dg + f[i]*ddg)/g**2 + 2*f[i]*dg**2/g**3
                )
                for i in (0, 1)
            ]
        d3f = [
            -cos_s*s[1]**3 - 3*sin_s*s[1]*s[2] + cos_s*s[3],
            cosh_t*t[1]**3 + 3*sinh_t*t[1]*t[2] + cosh_t*t[3]
        ]
        d3g = (
            sinh_t*t[1]**3 + 3*cosh_t*t[1]*t[2] + sinh_t*t[3]
            -sin_s*s[1]**3 + 3*cos_s*s[1]*s[2] + sin_s*s[3]
        )
        if diff == 3:
            return [
                a * (
                    d3f[i]/g - (3*(ddf[i]*dg+df[i]*ddg) + f[i]*d3g)/g**2
                    + 6 * (f[i]*dg*ddg + df[i]*dg**2) / g**3
                    - 6 * f[i] * dg**3 / g**4
                )
                for i in (0, 1)
            ]
        d4f = [
            sin_s * (-3*s[2]**2 + s[1]**4 - 4*s[1]*s[3])
            + cos_s * (s[4] - 6*s[2]*s[1]**2),
            sinh_t * (3*t[2]**2 + t[1]**4 + 4*t[1]*t[3])
            + cosh_t * (6*t[2]*t[1]**2 + t[4])
        ]
        d4g = (
            sin_s * (s[4] - 6*s[2]*s[1]**2)
            + cos_s * (3*s[2]**2 - s[1]**4 + 4*s[1]*s[3])
            + sinh_t * (6*t[2]*t[1]**2 + t[4])
            + cosh_t * (3*t[2]**2 + t[1]**4 + 4*t[1]*t[3])
        )
        if diff == 4:
            return [
                a * (
                    - 24*df[i]*dg**3 / g**4
                    + d4f[i] / g
                    + 6*f[i]*ddg**2 / g**3
                    + 24*f[i]*dg**4 / g**5
                    + dg**2 * (12*ddf[i]/g**3 - 36*f[i]*ddg/g**4)
                    + (-6*ddf[i]*ddg - 4*df[i]*d3g - 4*d3f[i]*dg - f[i]*d4g) / g**2
                    + (24*df[i]*ddg*dg + 8*f[i]*dg*d3g) / g**3
                )
                for i in (0, 1)
            ]
        raise NotImplementedError('Derivatives of order > 4 not implemented.')
