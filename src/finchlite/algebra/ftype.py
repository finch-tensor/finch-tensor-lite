from abc import ABC, abstractmethod
import numpy as np
from . import query_property
import builtins


class FType(ABC):
    @abstractmethod
    def __eq__(self, other): ...

    @abstractmethod
    def __hash__(self): ...

    def fisinstance(self, other):
        """
        Check if `other` is an instance of this ftype.
        """
        return ftype(other) == self

#https://data-apis.org/array-api/latest/API_specification/data_types.html#data-type-categories

class FDType(FType):...

class FDTypeNumeric(FDType):...
    @property
    @abstractmethod
    def type_max(self):
        """
        The maximum value for this type.
        """
        return self.max

    @property
    @abstractmethod
    def type_max(self):
        """
        The maximum value for this type.
        """
        return self.max


class FDTypeReal(FDTypeNumeric):...

class FDTypeComplex(FDTypeNumeric):...

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

class FDTypeBoolean(FDType):...

#Ftypes for python built-in datatypes
class FDTypeBuiltinBool(FDTypeBoolean):...
bool_ = FDTypeBuiltinBool()

class FDTypeBuiltinInt(FDTypeInteger, FDTypeReal):
    @property
    def iinfo(self):
        raise NotImplementedError("iinfo is not implemented for Python built-in int type")
    ...

int_ = FDTypeBuiltinInt()

class FDTypeBuiltinFloat(FDTypeFloat, FDTypeReal):
    @property
    def finfo(self):
        """
        The finfo object for this float type.
        """
        return np.float64.finfo

float_ = FDTypeBuiltinFloat()

class FDTypeBuiltinComplex(FDTypeFloat, FDTypeComplex):...
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

class FDTypeNumpyInteger(FDTypeInteger, FDTypeNumpy):
    @property
    @abstractmethod
    def iinfo(self):
        """
        The iinfo object for this integer type.
        """
        return self.np_dtype.iinfo

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
        return self.dtype.finfo

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

#FTypes for numpy built-in datatypes
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

class FDTypeFloat64(FDTypeNumpyFloat, FDTypeReal):
    @property
    def dtype(self):
        return np.float64
float64 = FDTypeFloat64()

class FDTypeComplex32(FDTypeNumpyFloat, FDTypeComplex):
    @property
    def dtype(self):
        return np.complex64
complex32 = FDTypeComplex32()

class FDTypeComplex64(FDTypeNumpyFloat, FDTypeComplex):
    @property
    def dtype(self):
        return np.complex64
complex64 = FDTypeComplex64()

#alias for default ftypes
int = int32 if np.intp == np.int32 else int64
float = float64
complex = complex64
index = int32 if np.intp == np.int32 else int64

def finfo(T:FDTypeFloat):
    return T.finfo

def iinfo(T:FDTypeInteger):
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
    """Return the corresponding FType for a given dtype.
    Recognizes numpy, Python builtins, and Python tuples.
    Calls .ftype on the object if type not found.
    """
    if isinstance(x, FTyped):
        return x.ftype
    if isinstance(x, bool):
        return bool_
    if isinstance(x, int):
        return int_
    if isinstance(x, float):
        return float_
    if isinstance(x, complex):
        return complex_
    if isinstance(x, np.bool_):
        return bool
    if isinstance(x, np.int8):
        return int8
    if isinstance(x, np.int16):
        return int16
    if isinstance(x, np.int32):
        return int32
    if isinstance(x, np.int64):
        return int64
    if isinstance(x, np.uint8):
        return uint8
    if isinstance(x, np.uint16):
        return uint16
    if isinstance(x, np.uint32):
        return uint32
    if isinstance(x, np.uint64):
        return uint64
    if isinstance(x, np.float32):
        return float32
    if isinstance(x, np.float64):
        return float64
    if isinstance(x, np.complex64):
        return complex32
    if isinstance(x, np.complex128):
        return complex64
    raise NotImplementedError
