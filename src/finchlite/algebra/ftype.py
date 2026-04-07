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

class FDTypeNumpy(FDType):
    @property
    @abstractmethod
    def dtype(self):
        """
        The corresponding numpy dtype for this ftype.
        """
        ...

class FDTypeNumeric(FDType):...

class FDTypeReal(FDTypeNumeric):...

class FDTypeComplex(FDTypeNumeric):...

class FDTypeInteger(FDTypeNumeric):...

class FDTypeFloat(FDTypeNumeric):...

class FDTypeFloatReal(FDTypeFloat, FDTypeReal):...

class FDTypeFloatComplex(FDTypeFloat, FDTypeComplex):...

class FDTypeBoolean(FDType):...

#Ftypes for python built-in datatypes
class FDTypeBuiltinBool(FDTypeBoolean):
    @property
    def dtype(self):
        return bool
bool_ = FDTypeBuiltinBool()

class FDTypeBuiltinInt(FDTypeInteger):
    @property
    def dtype(self):
        return int
int_ = FDTypeBuiltinInt()

class FDTypeBuiltinFloat(FDTypeFloat):
    @property
    def dtype(self):
        return float
float_ = FDTypeBuiltinFloat()

class FDTypeBuiltinComplex(FDTypeComplex):
    @property
    def dtype(self):
        return complex
complex_ = FDTypeBuiltinComplex()

#FTypes for numpy built-in datatypes
class FDTypeBool(FDTypeBoolean, FDTypeNumpy):
    @property
    def dtype(self):
        return np.bool_

bool = FDTypeBool()

class FDTypeInt8(FDTypeInteger):
    @property
    def dtype(self):
        return np.int8

int8 = FDTypeInt8()

class FDTypeInt16(FDTypeInteger):
    @property
    def dtype(self):
        return np.int16
    
int16 = FDTypeInt16()

class FDTypeInt32(FDTypeInteger):
    @property
    def dtype(self):
        return np.int32
int32 = FDTypeInt32()

class FDTypeInt64(FDTypeInteger):
    @property
    def dtype(self):
        return np.int64
int64 = FDTypeInt64()

class FDTypeUInt8(FDTypeInteger):
    @property
    def dtype(self):
        return np.uint8
uint8 = FDTypeUInt8()

class FDTypeUInt16(FDTypeInteger):
    @property
    def dtype(self):
        return np.uint16
uint16 = FDTypeUInt16()

class FDTypeUInt32(FDTypeInteger):
    @property
    def dtype(self):
        return np.uint32
uint32 = FDTypeUInt32()

class FDTypeUInt64(FDTypeInteger):
    @property
    def dtype(self):
        return np.uint64

uint64 = FDTypeUInt64()

class FDTypeFloat32(FDTypeFloat):
    @property
    def dtype(self):
        return np.float32
float32 = FDTypeFloat32()

class FDTypeFloat64(FDTypeFloat):
    @property
    def dtype(self):
        return np.float64
float64 = FDTypeFloat64()


class FDTypeComplex32(FDTypeComplex):
    @property
    def dtype(self):
        return np.complex64
complex32 = FDTypeComplex32()

class FDTypeComplex64(FDTypeComplex):
    @property
    def dtype(self):
        return np.complex64
complex64 = FDTypeComplex64()

#alias for default ftypes
int = int32 if np.intp == np.int32 else int64
float = float64
complex = complex64
index = int32 if np.intp == np.int32 else int64

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


def ftype(x) -> FType:
    """Return the corresponding FType for a given dtype.
    Recognizes numpy, Python builtins, and Python tuples.
    Calls .ftype on the object if type not found.
    """

    if isinstance(x, FTyped):
        return x.ftype
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