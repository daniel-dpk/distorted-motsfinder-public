r"""@package motsfinder.exprs._cheby

Evaluate a series of Chebyshev polynomials at a point.
"""

cimport cython
cimport numpy as np
import numpy as np


np.import_array()


DTYPE = np.double
ctypedef np.double_t DTYPE_t
ctypedef size_t uint


@cython.boundscheck(False)
cpdef np.ndarray[DTYPE_t, ndim=1] evaluate_Tn_double(DTYPE_t x, np.ndarray[DTYPE_t, ndim=1] Tn):
    assert Tn.dtype == DTYPE
    cdef uint num = Tn.shape[0]
    cdef uint k, n
    if x == 1.0:
        return np.ones(num)
    if x == -1.0:
        return np.array([(-1.0)**k for k in range(num)])
    Tn[0] = 1.0
    if num > 1:
        Tn[1] = x
    for n in range(2, num):
        Tn[n] = (2.0 * x) * Tn[n-1] - Tn[n-2]
    return Tn
