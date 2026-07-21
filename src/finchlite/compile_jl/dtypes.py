from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import numpy as np

import finchlite as fl
from finchlite.algebra.ftypes import FDTypeNumpy, FType, TupleFType

from .julia import get_jl, jc

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

finfo = fl.finfo
iinfo = fl.iinfo


class JuliaElementFType(ABC):
    @abstractmethod
    def julia_type(self):
        """
        Return the Julia type used for elements with this ftype.
        """
        ...

    @abstractmethod
    def julia_value(self, value: Any, *, offset: int = 0):
        """
        Convert a Python value with this ftype into a Julia element value.
        """
        ...

    def julia_vector(self, values, *, offset: int = 0):
        jl = get_jl()
        return jc.convert(
            jl.Vector[self.julia_type()],
            [self.julia_value(value, offset=offset) for value in values],
        )


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
    jl = get_jl()
    return {
        **{v: k for k, v in _jl_dtype_to_fl().items()},
        fl.bool_: jl.Bool,
        fl.int_: jl.Int,
        fl.float_: jl.Float64,
        fl.complex_: jl.ComplexF64,
    }


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
    T = to_fl_dtype(T)
    if isinstance(T, JuliaElementFType):
        return T.julia_type()
    if isinstance(T, TupleFType):
        jl = get_jl()
        return jl.Tuple[tuple(to_jl_type(field) for field in T.struct_fieldtypes)]
    try:
        return _fl_dtype_to_jl()[T]
    except KeyError:
        raise NotImplementedError(f"Cannot convert {T!r} to a Julia dtype") from None


def _as_julia_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tuple_field(value, index: int, name: str):
    if isinstance(value, np.void) and value.dtype.fields is not None:
        return value[name]
    return value[index]


def to_jl_value(T, value, *, offset: int = 0):
    T = to_fl_dtype(T)
    if isinstance(T, JuliaElementFType):
        return T.julia_value(value, offset=offset)
    if isinstance(T, TupleFType):
        return tuple(
            to_jl_value(
                field_type,
                _tuple_field(value, index, field_name),
                offset=offset,
            )
            for index, (field_name, field_type) in enumerate(T.struct_fields)
        )
    if offset:
        value = value + offset
    return _as_julia_scalar(T(value))


def to_jl_vector(T, values, *, offset: int = 0):
    T = to_fl_dtype(T)
    if isinstance(T, JuliaElementFType):
        return T.julia_vector(values, offset=offset)
    if isinstance(T, TupleFType) or offset:
        jl = get_jl()
        return jc.convert(
            jl.Vector[to_jl_type(T)],
            [to_jl_value(T, value, offset=offset) for value in values],
        )
    return get_jl().Vector(values)
