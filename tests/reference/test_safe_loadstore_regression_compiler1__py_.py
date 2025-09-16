import _operator, builtins
from numba import njit
import numpy


@njit
def finch_access(a: builtins.list, idx: ctypes.c_ulong) -> numpy.int64:
    a_ = a
    a__arr = a_[0]
    computed = (idx)
    if computed < 0 or computed >= (len(a__arr)):
        raise IndexError()
    val: numpy.int64 = a__arr[computed]
    computed_2 = (idx)
    if computed_2 < 0 or computed_2 >= (len(a__arr)):
        raise IndexError()
    val2: numpy.int64 = a__arr[computed_2]
    return val

@njit
def finch_change(a: builtins.list, idx: ctypes.c_ulong, val: ctypes.c_long) -> numpy.int64:
    a_ = a
    a__arr_2 = a_[0]
    computed_3 = (idx)
    if computed_3 < 0 or computed_3 >= (len(a__arr_2)):
        raise IndexError()
    a__arr_2[computed_3] = val
    return c_long(0)
