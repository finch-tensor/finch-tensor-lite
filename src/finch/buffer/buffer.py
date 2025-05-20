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


class NumpyBuffer(AbstractBuffer, CArgument):
    """
    A buffer that uses NumPy arrays to store data. This is a concrete implementation
    of the AbstractBuffer class.
    """

    def __init__(self, arr: np.ndarray):
        self.arr = arr

    #    def get_format(self):
    #        """
    #        Return the format of the buffer. The format defines how the data is organized
    #        and accessed.
    #        """
    #        return NumpyBufferFormat(self._buffer.dtype)

    def load(self, index: int):
        return self.arr[index]

    def store(self, index: int, value):
        self.arr[index] = value

    def resize(self, new_length: int):
        self.arr.resize(new_length)

    def get_resize_callback(self):
        """
        Create a ctypes callback that closes over the instance's NumPy array.
        """
        def resize_callback(c_buffer, new_length):
            """
            A Python callback function that resizes the NumPy array.
            """
            # Access the NumPy array from the closure
            numpy_array = ctypes.cast(c_buffer.contents.arr, ctypes.py_object).value
            # Resize the NumPy array
            numpy_array.resize(new_length, refcheck=False)
            # Update the length and data pointer in the CNumpyBuffer structure
            c_buffer.contents.length = new_length
            c_buffer.contents.data = numpy_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            print(f"Resized array to length: {new_length}")

        # Wrap the closure in a ctypes-compatible function pointer
        return ctypes.CFUNCTYPE(None, ctypes.POINTER(CNumpyBuffer), ctypes.c_size_t)(resize_callback)


    def serialize_to_c(self):
        if not self.arr.flags["C_CONTIGUOUS"]:
            raise ValueError("NumPy array must be C-contiguous")
        data = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        length = np_array.size
        arr = ctypes.py_object(self.arr)
        return CNumpyBuffer(arr, data, length)

    def deserialize_from_c(self, obj):
        """
        Update this buffer based on how the C call modified `obj`, the result
        of `serialize_to_c`.
        """


class CNumpyBuffer(ctypes.Structure):
    """
    A ctypes structure that represents a NumPy-like buffer in C.
    """

    _fields_ = [
        ("arr", ctypes.py_object),  # Python object for the NumPy array
        ("data", ctypes.c_void_p),  # Pointer to the data (generic type)
        ("length", ctypes.c_size_t),  # Length of the buffer
    ]


@ctypes.CFUNCTYPE(ctypes.void, ctypes.POINTER(CNumpyBuffer), ctypes.c_size_t)
def resize_callback(c_buffer, new_length):
    """
    A Python callback function that can be called from C to resize the NumPy array.
    """
    # Extract the NumPy array from the CNumpyBuffer structure
    numpy_array = ctypes.cast(c_buffer.contents.arr, ctypes.py_object).value
    # Resize the NumPy array
    numpy_array.resize(new_length, refcheck=False)
    # Update the length in the CNumpyBuffer structure
    c_buffer.contents.length = new_length
    c_buffer.contents.data = numpy_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

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
