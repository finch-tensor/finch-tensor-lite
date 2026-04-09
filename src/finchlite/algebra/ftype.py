# AI modified: 2026-04-08T22:22:21Z 84b3c0ad
import builtins
from abc import abstractmethod
from functools import lru_cache
from typing import Any

import numpy as np

"""
This module defines the FType class and related classes for representing data
types in Finch.  Many design decisions in this file are based on array api
requirements:
https://data-apis.org/array-api/latest/API_specification/data_types.html
"""


class FType:
    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def fisinstance(self, other):
        """
        Check if `other` is an instance of this ftype.
        """
        return ftype(other) == self


# https://data-apis.org/array-api/latest/API_specification/data_types.html#data-type-categories


class FDType(FType):
    def __promote__(self, other):
        """
        Return the result of promoting this type with another type.
        """
        return


def promote_type(T1: FDType, T2: FDType):
    """
    Returns the data type with the smallest size and smallest scalar kind to
    which both type1 and type2 may be safely cast.

    Args:
        *args: The types to promote.

    Returns:
        The common type of the given arguments.
    """
    if T1 == T2:
        return T1
    S1 = T1.__promote__(T2)
    S2 = T2.__promote__(T1)
    if S1 is not None:
        assert S1 == S2, f"Promotion of {T1} and {T2} is not consistent: {S1} vs {S2}"
        return S1
    if S2 is not None:
        return S2
    raise TypeError(f"Cannot promote types {T1} and {T2}")


class FDTypeNumeric(FDType):
    @property
    @abstractmethod
    def type_min(self):
        """
        The minimum value for this type.
        """
        ...

    @property
    @abstractmethod
    def type_max(self):
        """
        The maximum value for this type.
        """
        ...


class FDTypeReal(FDTypeNumeric): ...


class FDTypeComplex(FDTypeNumeric): ...


class FDTypeInteger(FDTypeNumeric):
    @property
    @abstractmethod
    def iinfo(self):
        """
        The iinfo object for this integer type.
        """
        ...


class FDTypeFloat(FDTypeNumeric):
    @property
    @abstractmethod
    def finfo(self):
        """
        The finfo object for this float type.
        """
        ...


class FDTypeBoolean(FDType): ...


class FDTypeBuiltin(FDType):
    @property
    @abstractmethod
    def type(self):
        """
        The corresponding Python built-in type for this ftype.
        """
        ...

    def __call__(self, val):
        """
        Create an instance of this ftype with the given value.
        """
        return self.type(val)


# Ftypes for python built-in datatypes
class _FDTypeBuiltinBool(FDTypeBoolean):
    @property
    def type(self):
        return bool

    def __repr__(self):
        return "finchlite.bool_"


bool_ = _FDTypeBuiltinBool()


class FDTypeNumericBuiltin(FDType):
    def __promote__(self, other):
        if isinstance(other, FDTypeBuiltin):
            return ftype(self.type(False) + other.type(False))
        if isinstance(other, FDTypeNumpy):
            return other.__promote__(self)
        return None


class _FDTypeBuiltinInt(FDTypeNumericBuiltin, FDTypeInteger, FDTypeReal):
    @property
    def type(self):
        return int

    def __repr__(self):
        return "finchlite.int_"

    @property
    def iinfo(self):
        raise NotImplementedError(
            "iinfo is not implemented for Python built-in int type"
        )

    @property
    def type_min(self):
        raise NotImplementedError(
            "type_min is not implemented for Python built-in int type"
        )

    @property
    def type_max(self):
        raise NotImplementedError(
            "type_max is not implemented for Python built-in int type"
        )


int_ = _FDTypeBuiltinInt()


class _FDTypeBuiltinFloat(FDTypeNumericBuiltin, FDTypeFloat, FDTypeReal):
    @property
    def type(self):
        return float

    def __repr__(self):
        return "finchlite.float_"

    @property
    def finfo(self):
        """
        The finfo object for this float type.
        """
        return np.float64.finfo

    @property
    def type_min(self):
        return self.type(-np.inf)

    @property
    def type_max(self):
        return self.type(np.inf)


float_ = _FDTypeBuiltinFloat()


class _FDTypeBuiltinComplex(FDTypeNumericBuiltin, FDTypeFloat, FDTypeComplex):
    @property
    def type(self):
        return complex

    def __repr__(self):
        return "finchlite.complex_"

    @property
    def finfo(self):
        return np.float64.finfo

    @property
    def type_min(self):
        return self.type(complex(-np.inf, -np.inf))

    @property
    def type_max(self):
        return self.type(complex(np.inf, np.inf))


complex_ = _FDTypeBuiltinComplex()


class FDTypeNumpy(FDType):
    @property
    @abstractmethod
    def dtype(self):
        """
        The corresponding numpy dtype for this ftype.
        """
        ...

    def __call__(self, val):
        """
        Create an instance of this ftype with the given value.
        """
        return self.dtype(val)

    def __promote__(self, other):
        if isinstance(other, FDTypeNumpy):
            promoted_dtype = np.promote_types(self.dtype, other.dtype)
            return ftype(promoted_dtype)
        if isinstance(other, FDTypeBuiltin):
            promoted_dtype = np.promote_types(self.dtype, other.type)
            return ftype(promoted_dtype)
        return None


class FDTypeNumpyInteger(FDTypeInteger, FDTypeNumpy):
    @property
    def iinfo(self):
        """
        The iinfo object for this integer type.
        """
        return np.iinfo(self.dtype)

    @property
    def type_min(self):
        """
        The minimum value for this type.
        """
        return self.iinfo.min

    @property
    def type_max(self):
        """
        The maximum value for this type.
        """
        return self.iinfo.max


class FDTypeNumpyFloat(FDTypeFloat, FDTypeNumpy):
    @property
    def finfo(self):
        """
        The finfo object for this float type.
        """
        return np.finfo(self.dtype)

    @property
    def type_min(self):
        """
        The minimum value for this type.
        """
        return self.dtype(-np.inf)

    @property
    def type_max(self):
        """
        The maximum value for this type.
        """
        return self.dtype(np.inf)


# FTypes for numpy built-in datatypes
class _FDTypeBool(FDTypeBoolean, FDTypeNumpy):
    @property
    def dtype(self):
        return np.bool_

    def __repr__(self):
        return "finchlite.bool"


bool = _FDTypeBool()


class _FDTypeInt8(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.int8

    def __repr__(self):
        return "finchlite.int8"


int8 = _FDTypeInt8()


class _FDTypeInt16(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.int16

    def __repr__(self):
        return "finchlite.int16"


int16 = _FDTypeInt16()


class _FDTypeInt32(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.int32

    def __repr__(self):
        return "finchlite.int32"


int32 = _FDTypeInt32()


class _FDTypeInt64(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.int64

    def __repr__(self):
        return "finchlite.int64"


int64 = _FDTypeInt64()


class _FDTypeUInt8(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.uint8

    def __repr__(self):
        return "finchlite.uint8"


uint8 = _FDTypeUInt8()


class _FDTypeUInt16(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.uint16

    def __repr__(self):
        return "finchlite.uint16"


uint16 = _FDTypeUInt16()


class _FDTypeUInt32(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.uint32

    def __repr__(self):
        return "finchlite.uint32"


uint32 = _FDTypeUInt32()


class _FDTypeUInt64(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.uint64

    def __repr__(self):
        return "finchlite.uint64"


uint64 = _FDTypeUInt64()


class _FDTypeFloat32(FDTypeNumpyFloat, FDTypeReal):
    @property
    def dtype(self):
        return np.float32

    def __repr__(self):
        return "finchlite.float32"


float32 = _FDTypeFloat32()


class _FDTypeFloat16(FDTypeNumpyFloat, FDTypeReal):
    @property
    def dtype(self):
        return np.float16

    def __repr__(self):
        return "finchlite.float16"


float16 = _FDTypeFloat16()


class _FDTypeFloat64(FDTypeNumpyFloat, FDTypeReal):
    @property
    def dtype(self):
        return np.float64

    def __repr__(self):
        return "finchlite.float64"


float64 = _FDTypeFloat64()


class _FDTypeComplex64(FDTypeNumpyFloat, FDTypeComplex):
    @property
    def dtype(self):
        return np.complex64

    def __repr__(self):
        return "finchlite.complex64"


complex64 = _FDTypeComplex64()


class _FDTypeComplex128(FDTypeNumpyFloat, FDTypeComplex):
    @property
    def dtype(self):
        return np.complex128

    def __repr__(self):
        return "finchlite.complex128"


complex128 = _FDTypeComplex128()

# alias for default ftypes
int = int32 if np.intp == np.int32 else int64
float = float64
complex = complex64
intp = int32 if np.intp == np.int32 else int64


def finfo(T: FDTypeFloat):
    return T.finfo


def iinfo(T: FDTypeInteger):
    return T.iinfo


class FTyped:
    """
    Abstract base class for objects that can be formatted.
    """

    @property
    @abstractmethod
    def ftype(self):
        """
        The ftype of the object.
        """
        ...


class TupleFType(FType):
    """FType for Python tuples, with a struct-compatible interface."""

    is_mutable = False

    def __init__(self, struct_name, struct_formats):
        self._struct_name = struct_name
        self._struct_formats = struct_formats

    def __eq__(self, other):
        return (
            isinstance(other, TupleFType)
            and self.struct_name == other.struct_name
            and self._struct_formats == other._struct_formats
        )

    def __len__(self):
        return len(self._struct_formats)

    def __hash__(self):
        return hash((self.struct_name, tuple(self.struct_fieldformats)))

    def __str__(self):
        return f"{self.struct_name}({', '.join(map(str, self._struct_formats))})"

    @property
    def struct_name(self):
        return self._struct_name

    @property
    def struct_fields(self):
        return [(f"element_{i}", fmt) for i, fmt in enumerate(self._struct_formats)]

    @property
    def struct_fieldnames(self) -> list[str]:
        return [name for (name, _) in self.struct_fields]

    @property
    def struct_fieldformats(self) -> list[Any]:
        return [type_ for (_, type_) in self.struct_fields]

    def struct_hasattr(self, attr: str) -> bool:
        return attr in dict(self.struct_fields)

    def struct_attrtype(self, attr: str) -> Any:
        return dict(self.struct_fields)[attr]

    def struct_getattr(self, obj, attr):
        index = list(self.struct_fieldnames).index(attr)
        return obj[index]

    def struct_setattr(self, obj, attr, value):
        index = list(self.struct_fieldnames).index(attr)
        obj[index] = value

    def fisinstance(self, other):
        if not isinstance(other, tuple) or len(other) != len(self.struct_fieldformats):
            return False
        return all(
            fisinstance(elt, fmt)
            for elt, fmt in zip(other, self.struct_fieldformats, strict=False)
        )

    def from_fields(self, *args):
        assert all(
            fisinstance(a, f)
            for a, f in zip(args, self.struct_fieldformats, strict=False)
        )
        return tuple(args)

    def __call__(self, **kwargs):
        return self.from_fields(*kwargs.values())

    @staticmethod
    @lru_cache
    def from_tuple(types: tuple[Any, ...]) -> "TupleFType":
        return TupleFType("tuple", types)


def fisinstance(x, f: FType):
    """
    Check if `x` is an instance of `f`.
    """
    return f.fisinstance(x)


def isdtype(x, T: FType):
    """
    Check if `x` is an instance of `T`.
    """
    return fisinstance(x, T)


def ftype(x) -> FType:
    """Return the corresponding FType for a given dtype or object.
    Recognizes numpy, Python builtins, and Python tuples.
    Calls .ftype on the object if type not found.
    """
    if isinstance(x, FType):
        return x
    if isinstance(x, FTyped):
        return x.ftype
    if type(x) is builtins.bool or x is builtins.bool:
        return bool_
    if type(x) is builtins.int or x is builtins.int:
        return int_
    if type(x) is builtins.float or x is builtins.float:
        return float_
    if type(x) is builtins.complex or x is builtins.complex:
        return complex_
    if type(x) is np.bool_ or x is np.bool_:
        return bool
    if type(x) is np.int8 or x is np.int8:
        return int8
    if type(x) is np.int16 or x is np.int16:
        return int16
    if type(x) is np.int32 or x is np.int32:
        return int32
    if type(x) is np.int64 or x is np.int64:
        return int64
    if type(x) is np.uint8 or x is np.uint8:
        return uint8
    if type(x) is np.uint16 or x is np.uint16:
        return uint16
    if type(x) is np.uint32 or x is np.uint32:
        return uint32
    if type(x) is np.uint64 or x is np.uint64:
        return uint64
    if type(x) is np.float32 or x is np.float32:
        return float32
    if type(x) is np.float64 or x is np.float64:
        return float64
    if type(x) is np.complex64 or x is np.complex64:
        return complex64
    if type(x) is np.complex128 or x is np.complex128:
        return complex128
    raise NotImplementedError
