r"""@package motsfinder

Python MOTS finder project.

Currently, the only implemented finders are located in the motsfinder.axisym
package and assume axisymmetry. The horizon \f$ \sigma \f$ is parameterized
either in the traditional way as \f$ r = h(\theta) \f$ or via coordinates
w.r.t. a reference surface \f$ \sigma_R \f$.

To solve the resulting elliptic ODE for the horizon function, a pseudospectral
solver in the motsfinder.ndsolve module is used.

The analytically implemented metrics in the motsfinder.metric module can serve
as testing cases.
"""
