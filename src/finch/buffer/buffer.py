from abc import ABC, abstractmethod
import numpy as np
from ..codegen.c import CArgument
import ctypes

class AbstractBuffer(ABC):
    """
    Abstract base class for buffer-like data structures. Buffers support random access,
    and can be resized. They are used to store data in a way that allows for efficient
    reading and writing of elements.
    """
    @abstractmethod
    def __init__(self, length: int, dtype: type):
        pass

#    @abstractmethod
#    def get_format(self):
#        """
#        Return the format of the buffer. The format defines how the data is organized
#        and accessed.
#        """
#        pass

    @abstractmethod
    def load(self, index: int):
        pass

    @abstractmethod
    def store(self, index: int, value):
        pass

    @abstractmethod
    def resize(self, new_length: int):
        """
        Resize the buffer to the new length.
        """
        pass

#class AbstractBufferFormat(ABC):
#    """
#    Abstract base class for the format of buffers. The format defines how the data
#    in an AbstractBuffer is organized and accessed.
#    """
#
#    @abstractmethod
#    def make_buffer(self, length):
#        """
#        Create a new buffer of the given length
#        """
#        pass

class NumpyBuffer(AbstractBuffer, CArgument):
    """
    A buffer that uses NumPy arrays to store data. This is a concrete implementation
    of the AbstractBuffer class.
    """
    def __init__(self, np_array: np.ndarray, own: bool = False):
        self.data = np_array
        self.own = own

#    def get_format(self):
#        """
#        Return the format of the buffer. The format defines how the data is organized
#        and accessed.
#        """
#        return NumpyBufferFormat(self._buffer.dtype)

    def load(self, index: int):
        return self.buffer[index]

    def store(self, index: int, value):
        self.data[index] = value

    def resize(self, new_length: int):
        self.data.resize(new_length)
    
    def serialize_to_c(self):
        if not np_array.flags['C_CONTIGUOUS']:
            raise ValueError("NumPy array must be C-contiguous")
        data = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        length = np_array.size
        return CNumpyBuffer(data, length, own)

    def deserialize_from_c(self, obj):
        """
        Update this buffer based on how the C call modified `obj`, the result
        of `serialize_to_c`.
        """
        pass

class CNumpyBuffer(ctypes.Structure):
    """
    A ctypes structure that represents a NumPy-like buffer in C.
    """
    _fields_ = [
    ("data", ctypes.POINTER(ctypes.c_double)),  # Pointer to the data
    ("length", ctypes.c_size_t),                # Length of the buffer
    ("own", ctypes.c_bool),                     # Whether the data is owned by this buffer
    ]


#class NumpyBufferFormat(AbstractBufferFormat, codegen.c.CBufferFormat):
#    """
#    A format for buffers that uses NumPy arrays. This is a concrete implementation
#    of the AbstractBufferFormat class.
#    """
#    def __init__(self, dtype: type):
#        self._dtype = dtype
#
#    def make_buffer(self, length: int):
#        return NumpyBuffer(np.zeros(length, dtype=self._dtype))
#
#    def c_load(self, name: str, index_name: str):
#        return f"""
#        {name}[index]
#        """
#
#    def c_store(self, name: str, value_name: str, index_name: str):
#        return f"""
#        {name}[index] = ({value_type}){value_name};
#        """
#
#    def c_resize(self, name: str, new_length_name: str, new_length_type: str):
#        return f"""
#        {new_length_type} new_length = {new_length_name};
#        {name} = realloc({name}, new_length * sizeof({self._dtype}));
#        """