# AI modified: 2026-04-08T22:22:21Z 84b3c0ad
import builtins
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache
from typing import Any

import numpy as np

"""
This module defines the FType class and related classes for representing data
types in Finch.  Many design decisions in this file are based on array api
requirements:
https://data-apis.org/array-api/latest/API_specification/data_types.html
"""


class FType(ABC):

    @abstractmethod
    def __eq__(self, other):
        ...
    
    @abstractmethod
    def __hash__(self):
        ...
    def fisinstance(self, other):
        """
        Check if `other` is an instance of this ftype.
        """
        return ftype(other) == self



# https://data-apis.org/array-api/latest/API_specification/data_types.html#data-type-categories


class FDType(FType):
    @abstractmethod
    def __call__(self, other):
        """
        Create an instance of this ftype with the given value, attempt to cast
        if necessary.
        """

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


class FDTypeOrdered(FDType):
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


class FDTypeNumeric(FDTypeOrdered): ...


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


class FDTypeBoolean(FDTypeOrdered): ...


class FDTypeBuiltin(FDType):
    @property
    @abstractmethod
    def type(self):
        """
        The corresponding Python built-in type for this ftype.
        """
        ...

    def __eq__(self, other):
        return isinstance(other, FDTypeBuiltin) and self.type == other.type
    
    def __hash__(self):
        return hash(self.type)

    def __call__(self, val):
        """
        Create an instance of this ftype with the given value.
        """
        return self.type(val)


class _FDTypeBuiltinStr(FDTypeBuiltin):
    @property
    def type(self):
        return str

    def __repr__(self):
        return "finchlite.str_"


str_ = _FDTypeBuiltinStr()


class _FDTypeBuiltinNone(FDTypeBuiltin):

    @property
    def type(self):
        return type(None)

    def __repr__(self):
        return "none_"

    def __call__(self, val):
        """None type cannot be called, always returns None"""
        if val is not None:
            raise TypeError(f"Cannot convert {val!r} to None type")
        return


none_ = _FDTypeBuiltinNone()


# Ftypes for python built-in datatypes
class _FDTypeBuiltinBool(FDTypeBoolean, FDTypeBuiltin):
    @property
    def type(self):
        return builtins.bool

    def __repr__(self):
        return "finchlite.bool_"

    @property
    def type_min(self):
        return False

    @property
    def type_max(self):
        return True


bool_ = _FDTypeBuiltinBool()


class FDTypeNumericBuiltin(FDTypeBuiltin, FDTypeNumeric):
    def __promote__(self, other):
        if isinstance(other, FDTypeBuiltin):
            return ftype(self.type(False) + other.type(False))
        if isinstance(other, FDTypeNumpy):
            return other.__promote__(self)
        return None


class _FDTypeBuiltinInt(FDTypeNumericBuiltin, FDTypeInteger, FDTypeReal):
    @property
    def type(self):
        return builtins.int

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
        return builtins.float

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
        return builtins.complex

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


    def __eq__(self, other):
        return isinstance(other, FDTypeNumpy) and self.dtype == other.dtype
    
    def __hash__(self):
        return hash(self.dtype)

    def __call__(self, val):
        """
        Create an instance of this ftype with the given value.
        """
        return self.dtype(val)

    def __promote__(self, other):
        if isinstance(other, FDTypeNumpy):
            promoted_dtype = np.promote_types(self.dtype, other.dtype)
            return ftype(promoted_dtype.type)
        if isinstance(other, FDTypeBuiltin):
            promoted_dtype = np.promote_types(self.dtype, other.type)
            return ftype(promoted_dtype.type)
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

    @property
    def type_min(self):
        return np.bool_(False)

    @property
    def type_max(self):
        return np.bool_(True)


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


class StructFType(FType, ABC):
    def __eq__(self, other):
        return (
            type(other) is type(self)
            and self.struct_name == other.struct_name
            and self.struct_fields == other.struct_fields
        )

    def __hash__(self):
        return hash((type(self), self.struct_name, tuple(self.struct_fields)))

    def __repr__(self):
        fields_str = ", ".join(f"{name}: {type_}" for name, type_ in self.struct_fields)
        return f"{self.struct_name}({fields_str})"

    @property
    @abstractmethod
    def struct_name(self) -> str: ...

    @property
    @abstractmethod
    def struct_fields(self) -> list[tuple[str, FType]]: ...

    @abstractmethod
    def from_fields(self, *args): ...

    @property
    @abstractmethod
    def is_mutable(self) -> builtins.bool: ...

    def struct_getattr(self, obj, attr) -> Any:
        return getattr(obj, attr)

    def struct_setattr(self, obj, attr, value) -> None:
        setattr(obj, attr, value)
        return

    @property
    def struct_fieldnames(self) -> list[str]:
        return [name for (name, _) in self.struct_fields]

    @property
    def struct_fieldtypes(self) -> list[FType]:
        return [type_ for (_, type_) in self.struct_fields]

    def struct_hasattr(self, attr: str) -> builtins.bool:
        return attr in dict(self.struct_fields)

    def struct_attrtype(self, attr: str) -> FType:
        return dict(self.struct_fields)[attr]


class ImmutableStructFType(StructFType):
    @property
    def is_mutable(self) -> builtins.bool:
        return False


class MutableStructFType(StructFType):
    """
    Class for a mutable struct type.
    It is currently not used anywhere, but maybe it will be useful in the future?
    """

    @property
    def is_mutable(self) -> builtins.bool:
        return True


class TupleFType(ImmutableStructFType):
    """FType for Python tuples, with a struct-compatible interface."""

    def __init__(self, struct_types):
        self._struct_types = struct_types

    def __repr__(self):
        return f"TupleFType(({', '.join(map(repr, self._struct_types))},))"

    def __str__(self):
        fields_str = ", ".join(f"{name}: {type_}" for name, type_ in self.struct_fields)
        return f"{self.struct_name}({fields_str})"

    @property
    def struct_name(self):
        return "tuple"

    @property
    def struct_fields(self):
        return [(f"element_{i}", fmt) for i, fmt in enumerate(self._struct_types)]

    @property
    def struct_fieldnames(self) -> list[str]:
        return [name for (name, _) in self.struct_fields]

    @property
    def struct_fieldtypes(self) -> list[Any]:
        return [type_ for (_, type_) in self.struct_fields]

    def struct_hasattr(self, attr: str) -> builtins.bool:
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
        if not isinstance(other, tuple) or len(other) != len(self.struct_fieldtypes):
            return False
        return all(
            fisinstance(elt, fmt)
            for elt, fmt in zip(other, self.struct_fieldtypes, strict=False)
        )

    def from_fields(self, *args):
        assert all(
            fisinstance(a, f)
            for a, f in zip(args, self.struct_fieldtypes, strict=False)
        )
        return tuple(args)

    def __call__(self, **kwargs):
        return self.from_fields(*kwargs.values())

    @staticmethod
    @lru_cache
    def from_tuple(types: tuple[FType, ...]) -> "TupleFType":
        if not isinstance(types, tuple):
            raise TypeError("TupleFType.from_tuple expects tuple[FType, ...]")
        if not all(isinstance(type_, FType) for type_ in types):
            raise TypeError("TupleFType.from_tuple expects tuple[FType, ...]")
        return TupleFType(types)


class NamedTupleFType(ImmutableStructFType):
    def __init__(self, struct_name, struct_fields):
        self._struct_name = struct_name
        self._struct_fields = struct_fields

    def __eq__(self, other):
        return (
            isinstance(other, NamedTupleFType)
            and self.struct_name == other.struct_name
            and self.struct_fields == other.struct_fields
        )

    def __len__(self):
        return len(self._struct_fields)

    def __hash__(self):
        return hash((self.struct_name, tuple(self.struct_fields)))

    @property
    def struct_name(self):
        return self._struct_name

    @property
    def struct_fields(self):
        return self._struct_fields

    def fisinstance(self, other):
        if not isinstance(other, tuple) or not hasattr(other, "_fields"):
            return False
        if tuple(other._fields) != tuple(self.struct_fieldnames):
            return False

        return all(
            fisinstance(elt, format)
            for elt, format in zip(other, self.struct_fieldtypes, strict=False)
        )

    def from_fields(self, *args):
        assert all(
            fisinstance(a, f)
            for a, f in zip(args, self.struct_fieldtypes, strict=False)
        )
        return namedtuple(self.struct_name, self.struct_fieldnames)(args)

    def __call__(self, *args):
        return self.from_fields(*args)


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
    """Return the corresponding FType for a given object.  Recognizes numpy,
    Python builtins, and Python tuples.  Sometimes recognizes types.
    Override .ftype to customize the ftype of an object.
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
    if type(x) is builtins.str or x is builtins.str:
        return str_
    if x is None:
        return none_
    if isinstance(x, tuple):
        T = type(x)
        if hasattr(T, "_fields") and all(isinstance(field, str) for field in T._fields):
            return NamedTupleFType(
                T.__name__,
                [(fieldname, ftype(getattr(x, fieldname))) for fieldname in T._fields],
            )
        return TupleFType.from_tuple(tuple(ftype(elem) for elem in x))
    raise NotImplementedError
