# AI modified: 2026-04-08T22:22:21Z 84b3c0ad
import builtins
from abc import abstractmethod

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
class FDTypeBuiltinBool(FDTypeBoolean):
    @property
    def type(self):
        return bool


bool_ = FDTypeBuiltinBool()


class FDTypeNumericBuiltin(FDType):
    def __promote__(self, other):
        if isinstance(other, FDTypeBuiltin):
            return ftype(self.type(False) + other.type(False))
        if isinstance(other, FDTypeNumpy):
            return other.__promote__(self)
        return None


class FDTypeBuiltinInt(FDTypeNumericBuiltin, FDTypeInteger, FDTypeReal):
    @property
    def type(self):
        return int

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


int_ = FDTypeBuiltinInt()


class FDTypeBuiltinFloat(FDTypeNumericBuiltin, FDTypeFloat, FDTypeReal):
    @property
    def type(self):
        return float

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


float_ = FDTypeBuiltinFloat()


class FDTypeBuiltinComplex(FDTypeNumericBuiltin, FDTypeFloat, FDTypeComplex):
    @property
    def type(self):
        return complex

    @property
    def finfo(self):
        return np.float64.finfo

    @property
    def type_min(self):
        return self.type(complex(-np.inf, -np.inf))

    @property
    def type_max(self):
        return self.type(complex(np.inf, np.inf))


complex_ = FDTypeBuiltinComplex()


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
class FDTypeBool(FDTypeBoolean, FDTypeNumpy):
    @property
    def dtype(self):
        return np.bool_


bool = FDTypeBool()


class FDTypeInt8(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.int8


int8 = FDTypeInt8()


class FDTypeInt16(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.int16


int16 = FDTypeInt16()


class FDTypeInt32(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.int32


int32 = FDTypeInt32()


class FDTypeInt64(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.int64


int64 = FDTypeInt64()


class FDTypeUInt8(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.uint8


uint8 = FDTypeUInt8()


class FDTypeUInt16(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.uint16


uint16 = FDTypeUInt16()


class FDTypeUInt32(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.uint32


uint32 = FDTypeUInt32()


class FDTypeUInt64(FDTypeNumpyInteger, FDTypeReal):
    @property
    def dtype(self):
        return np.uint64


uint64 = FDTypeUInt64()


class FDTypeFloat32(FDTypeNumpyFloat, FDTypeReal):
    @property
    def dtype(self):
        return np.float32


float32 = FDTypeFloat32()


class FDTypeFloat16(FDTypeNumpyFloat, FDTypeReal):
    @property
    def dtype(self):
        return np.float16


float16 = FDTypeFloat16()


class FDTypeFloat64(FDTypeNumpyFloat, FDTypeReal):
    @property
    def dtype(self):
        return np.float64


float64 = FDTypeFloat64()


class FDTypeComplex64(FDTypeNumpyFloat, FDTypeComplex):
    @property
    def dtype(self):
        return np.complex64


complex64 = FDTypeComplex64()


class FDTypeComplex128(FDTypeNumpyFloat, FDTypeComplex):
    @property
    def dtype(self):
        return np.complex128


complex128 = FDTypeComplex128()

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


def fisinstance(x, f):
    """
    Check if `x` is an instance of `f`.
    """
    if isinstance(f, type):
        return isinstance(x, f)
    return f.fisinstance(x)


def isdtype(x, T):
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
    if isinstance(x, builtins.bool) or x == builtins.bool:
        return bool_
    if isinstance(x, builtins.int) or x == builtins.int:
        return int_
    if isinstance(x, builtins.float) or x == builtins.float:
        return float_
    if isinstance(x, builtins.complex) or x == builtins.complex:
        return complex_
    if isinstance(x, np.bool_) or x == np.bool_:
        return bool
    if isinstance(x, np.int8) or x == np.int8:
        return int8
    if isinstance(x, np.int16) or x == np.int16:
        return int16
    if isinstance(x, np.int32) or x == np.int32:
        return int32
    if isinstance(x, np.int64) or x == np.int64:
        return int64
    if isinstance(x, np.uint8) or x == np.uint8:
        return uint8
    if isinstance(x, np.uint16) or x == np.uint16:
        return uint16
    if isinstance(x, np.uint32) or x == np.uint32:
        return uint32
    if isinstance(x, np.uint64) or x == np.uint64:
        return uint64
    if isinstance(x, np.float32) or x == np.float32:
        return float32
    if isinstance(x, np.float64) or x == np.float64:
        return float64
    if isinstance(x, np.complex64) or x == np.complex64:
        return complex64
    if isinstance(x, np.complex128) or x == np.complex128:
        return complex128
    raise NotImplementedError
