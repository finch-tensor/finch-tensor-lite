import _operator, builtins
from numba import njit
import numpy
from numpy import int64, float64


@njit
def finch_access(a: builtins.list, idx: ctypes.c_ulonglong) -> int64:
    a_ = a
    a__arr = a_[0]
    computed = (idx)
    if computed < 0 or computed >= (len(a__arr)):
        raise IndexError()
    val: int64 = a__arr[computed]
    computed_2 = (idx)
    if computed_2 < 0 or computed_2 >= (len(a__arr)):
        raise IndexError()
    val2: int64 = a__arr[computed_2]
    return val

@njit
def finch_change(a: builtins.list, idx: ctypes.c_ulonglong, val: ctypes.c_longlong) -> int64:
    a_ = a
    a__arr_2 = a_[0]
    computed_3 = (idx)
    if computed_3 < 0 or computed_3 >= (len(a__arr_2)):
        raise IndexError()
    a__arr_2[computed_3] = val
    return c_longlong(0)