r"""@package motsfinder.ipyutils.printing

Printing helpers for IPython/Jupyter notebooks.
"""


from IPython.display import display, Latex
import sympy as sp


__all__ = [
    "disp",
]


def disp(tex, *expr, eq=True):
    r"""Display LaTeX including TeX-able SymPy objects in a notebook.

    The `tex` string should contain LaTeX code and include as many
    placeholders ``%%s`` as there are extra positional arguments.

    @param tex
        String containing the math LaTeX code and `n` placeholders ``%%s``.
    @param *expr
        Further positional arguments are interpreted as SymPy objects to be
        inserted into the placeholders. Should be `n` objects.
    @param eq
        Wehther to enter LaTeX math mode or not. Default is `True`.

    @b Examples

    ```
        from sympy import cos, pi, symbols
        disp("f(x) = %s", cos(pi/2-symbols('x')))
    ```
    """
    delim = '$$' if eq else ''
    display(Latex(delim + (tex % tuple(sp.latex(e) for e in expr)) + delim))
