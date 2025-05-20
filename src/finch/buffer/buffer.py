import ctypes
from abc import ABC, abstractmethod

import numpy as np

from ..codegen.c import CArgument


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
    #        Return the format of the buffer. The format defines how the data is
    #        organized
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


# class AbstractBufferFormat(ABC):
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


@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(ctypes.py_object), ctypes.c_size_t)
def numpy_buffer_resize_callback(arr_ptr, new_length):
    """
    A Python callback function that resizes the NumPy array.
    """
    numpy_array = arr_ptr.contents.value
    resized_array = np.resize(numpy_array, new_length)
    arr_ptr.contents.value = resized_array
    return ctypes.cast(resized_array.ctypes.data, ctypes.c_void_p)


class NumpyBuffer(AbstractBuffer, CArgument):
    """
    A buffer that uses NumPy arrays to store data. This is a concrete implementation
    of the AbstractBuffer class.
    """

    def __init__(self, arr: np.ndarray):
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError("NumPy array must be C-contiguous")
        self.arr = arr

    def load(self, index: int):
        return self.arr[index]

    def store(self, index: int, value):
        self.arr[index] = value

    def resize(self, new_length: int):
        self.arr = np.resize(self.arr, new_length)

    def serialize_to_c(self):
        """
        Serialize the NumPy buffer to a C-compatible structure.
        """
        data = self.arr.ctypes.data_as(
            ctypes.POINTER(np.ctypeslib.as_ctypes_type(self.arr.dtype))
        )
        length = self.arr.size
        arr = ctypes.py_object(self.arr)
        return CNumpyBuffer(arr, data, length, numpy_buffer_resize_callback)

    def deserialize_from_c(self, c_buffer):
        """
        Update this buffer based on how the C call modified the CNumpyBuffer structure.
        """
        self.arr = ctypes.cast(c_buffer.arr, ctypes.py_object).value


# Dynamically define the CNumpyBuffer structure
class CNumpyBuffer(ctypes.Structure):
    _fields_ = [
        ("arr", ctypes.py_object),
        ("data", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        (
            "resize",
            ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_size_t),
        ),
    ]


# class NumpyBufferFormat(AbstractBufferFormat, codegen.c.CBufferFormat):
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
