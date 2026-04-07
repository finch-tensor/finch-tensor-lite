from abc import ABC, abstractmethod
import numpy as np
from ..algebra import query_property


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
    

#FTypes for python built-in datatypes
class BoolFType(FType):
    dtype = np.bool_

class Int8FType(FType):
    dtype = np.int8

class Int16FType(FType):
    dtype = np.int16

class Int32FType(FType):
    dtype = np.int32

class Int64FType(FType):
    dtype = np.int64

class UInt8FType(FType):
    dtype = np.uint8

class UInt16FType(FType):
    dtype = np.uint16

class UInt32FType(FType):
    dtype = np.uint32

class UInt64FType(FType):
    dtype = np.uint64

class Float32FType(FType):
    dtype = np.float32

class Float64FType(FType):
    dtype = np.float64

class Complex64FType(FType):
    dtype = np.complex64

class Complex128FType(FType):
    dtype = np.complex128

class StringFType(FType):
    dtype = str


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


def ftype(dtype) -> FType:
    """Return the corresponding FType for a given dtype.
    Recognizes numpy, Python builtins, and Python tuples.
    Calls .ftype on the object if type not found.
    """


    match dtype:
        case np.bool_:
            return BoolFType()
        case np.int8:
            return Int8FType()
        case np.int16:
            return Int16FType()
        case x if x is np.int32 or x is int:
            return Int32FType()
        case np.int64:
            return Int64FType()
        case np.uint8:
            return UInt8FType()
        case np.uint16:
            return UInt16FType()
        case np.uint32:
            return UInt32FType()
        case np.uint64:
            return UInt64FType()
        case np.float32:
            return Float32FType()
        case x if x is np.float64 or x is float:
            return Float64FType()
        case np.complex64:
            return Complex64FType()
        case x if x is np.complex128 or x is complex:
            return Complex128FType()
        case x if x is np.bool_ or x is bool:
            return BoolFType()
        case _ if dtype is str:
            return StringFType()
        case _ if dtype is tuple:
            raise NotImplementedError
        
        #if not one of the above types
        case x if hasattr(x, "ftype"):
            return x.ftype
        case _:
            raise TypeError(f"No ftype registered for {dtype}")
