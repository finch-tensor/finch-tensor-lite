import _operator, builtins
from numba import njit
import numpy


@njit
def dot_product(a: builtins.list, b: builtins.list) -> numpy.float64:
    c: numpy.float64 = 0.0
    a_ = a
    a__arr = a_[0]
    b_ = b
    b__arr = b_[0]
    for i in range(0, len(a__arr)):
        c = _operator.add(c, _operator.mul(a__arr[i], b__arr[i]))
    a_[0] = a__arr
    b_[0] = b__arr
    return c
