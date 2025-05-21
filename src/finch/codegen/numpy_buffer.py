import ctypes

import numpy as np

from .abstract_buffer import AbstractBuffer
from .c import CArgument


@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(ctypes.py_object), ctypes.c_size_t)
def numpy_buffer_resize_callback(buf_ptr, new_length):
    """
    A Python callback function that resizes the NumPy array.
    """
    buf = buf_ptr.contents.value
    buf.arr = np.resize(buf.arr, new_length)
    return buf.arr.ctypes.data


class CNumpyBuffer(ctypes.Structure):
    _fields_ = [
        ("arr", ctypes.py_object),
        ("data", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("resize", type(numpy_buffer_resize_callback)),
    ]


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
        data = ctypes.c_void_p(self.arr.ctypes.data)
        length = self.arr.size
        return CNumpyBuffer(self, data, length, numpy_buffer_resize_callback)

    def deserialize_from_c(self, c_buffer):
        """
        Update this buffer based on how the C call modified the CNumpyBuffer structure.
        """


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
