r"""@package motsfinder.ipyutils.reloading

Helpers for reloading changed sources at runtime.

@b Examples

@code
exec(reload_all())
@endcode
"""

import textwrap


__all__ = [
    "reload_all",
]


def _reload_code(*modules):
    r"""Return a string reloading the given list of module names."""
    return "\n".join("import {m}; reload({m}); from {m} import *".format(m=m) for m in modules)

def reload_all():
    """Return a string that can be 'exec'ed to reload modules under development.

    The modules are imported, reloaded, and finally a

        from the_module import *

    is done for each module.

    Note: If you are not the last person that modified this reload code, check
          the result of this function prior to executing it for security
          reasons.

    To check the result prior to executing it, do:

        print(reload_all())

    To actually reload the modules, do:

        exec(reload_all())
    """
    prefix = textwrap.dedent("""\
        try:
            import importlib
            from importlib import reload
        except ModuleNotFoundError:
            pass
        """)
    return prefix + _reload_code(
        "motsfinder.utils",
        "motsfinder.numutils",
        "motsfinder.exprs.common",
        "motsfinder.exprs.evaluators",
        "motsfinder.exprs.numexpr",
        "motsfinder.exprs.basics",
        "motsfinder.exprs.series",
        "motsfinder.exprs.cheby",
        "motsfinder.exprs.trig",
        "motsfinder.exprs",
        "motsfinder.ndsolve.common",
        "motsfinder.ndsolve.bases.base",
        "motsfinder.ndsolve.bases.cheby",
        "motsfinder.ndsolve.bases.trig",
        "motsfinder.ndsolve.bcs",
        "motsfinder.ndsolve.solver",
        "motsfinder.ndsolve",
        "motsfinder.ipyutils.plotctx",
        "motsfinder.ipyutils.plotting",
        "motsfinder.ipyutils.plotting3d",
        "motsfinder.ipyutils.printing",
        "motsfinder.ipyutils.reloading",
        "motsfinder.ipyutils",
        "motsfinder.metric.helpers",
        "motsfinder.metric.base",
        "motsfinder.metric.analytical",
        "motsfinder.metric",
        "motsfinder.pickle_helpers",
        "motsfinder.axisym.curve.basecurve",
        "motsfinder.axisym.curve.parametriccurve",
        "motsfinder.axisym.curve.expcalc",
        "motsfinder.axisym.curve.expcurve",
        "motsfinder.axisym.curve.refparamcurve",
        "motsfinder.axisym.curve.starshapedcurve",
        "motsfinder.axisym.curve",
        "motsfinder.axisym.newton",
        "motsfinder.axisym.findmots",
        "motsfinder.axisym",
    )
