import builtins
from functools import lru_cache
from typing import Any

import numpy as np

import finchlite as fl
from finchlite.algebra.ftypes import FDTypeNumpy, FType

from .julia import get_jl
from .typing import JLFType

int8: FDTypeNumpy = fl.int8
int16: FDTypeNumpy = fl.int16
int32: FDTypeNumpy = fl.int32
int64: FDTypeNumpy = fl.int64
int_: FDTypeNumpy = fl.intp
uint8: FDTypeNumpy = fl.uint8
uint16: FDTypeNumpy = fl.uint16
uint32: FDTypeNumpy = fl.uint32
uint64: FDTypeNumpy = fl.uint64
uint: FDTypeNumpy = uint32 if np.uintp == np.uint32 else uint64
float16: FDTypeNumpy = fl.float16
float32: FDTypeNumpy = fl.float32
float64: FDTypeNumpy = fl.float64
complex64: FDTypeNumpy = fl.complex64
complex128: FDTypeNumpy = fl.complex128
bool: FDTypeNumpy = fl.bool

number = builtins.int | builtins.float | builtins.complex | builtins.bool

finfo = fl.finfo
iinfo = fl.iinfo

jl_to_np_dtype: dict[FType | None, Any] = {
    int_: int_.dtype,
    int8: int8.dtype,
    int16: int16.dtype,
    int32: int32.dtype,
    int64: int64.dtype,
    uint: uint.dtype,
    uint8: uint8.dtype,
    uint16: uint16.dtype,
    uint32: uint32.dtype,
    uint64: uint64.dtype,
    float16: float16.dtype,
    float32: float32.dtype,
    float64: float64.dtype,
    complex64: complex64.dtype,
    complex128: complex128.dtype,
    bool: bool.dtype,
    None: None,
}


def can_cast(from_, to, /) -> builtins.bool:
    if not isinstance(from_, FType) and hasattr(from_, "dtype"):
        from_ = from_.dtype
    return np.can_cast(jl_to_np_dtype[from_], jl_to_np_dtype[to])


@lru_cache
def _jl_dtype_to_fl() -> dict[Any, FType]:
    jl = get_jl()
    return {
        jl.Int8: int8,
        jl.Int16: int16,
        jl.Int32: int32,
        jl.Int64: int64,
        jl.UInt8: uint8,
        jl.UInt16: uint16,
        jl.UInt32: uint32,
        jl.UInt64: uint64,
        jl.Float16: float16,
        jl.Float32: float32,
        jl.Float64: float64,
        jl.ComplexF32: complex64,
        jl.ComplexF64: complex128,
        jl.Bool: bool,
    }


@lru_cache
def _fl_dtype_to_jl() -> dict[FType, Any]:
    return {v: k for k, v in _jl_dtype_to_fl().items()}


def to_fl_dtype(x) -> FType:
    """Normalize a Julia DataType, numpy dtype/scalar type, Python builtin
    type, or finchlite FType into the corresponding finchlite FType."""
    if isinstance(x, FType):
        return x
    try:
        return fl.ftype(x)
    except NotImplementedError:
        pass
    try:
        return _jl_dtype_to_fl()[x]
    except (KeyError, TypeError):
        raise NotImplementedError(f"Cannot convert {x!r} to a Finch dtype") from None


def to_jl_type(T):
    if isinstance(T, JLFType):
        return T.to_jl_type()
    T = to_fl_dtype(T)
    try:
        return _fl_dtype_to_jl()[T]
    except KeyError:
        raise NotImplementedError(f"Cannot convert {T!r} to a Julia dtype") from None
