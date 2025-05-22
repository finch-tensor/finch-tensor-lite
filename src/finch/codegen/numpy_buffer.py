import ctypes

import numpy as np

from .abstract_buffer import AbstractBuffer
from .c import AbstractCArgument, AbstractCFormat, AbstractSymbolicCBuffer


@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(ctypes.py_object), ctypes.c_size_t)
def numpy_buffer_resize_callback(buf_ptr, new_length):
    """
    A Python callback function that resizes the NumPy array.
    """
    buf = buf_ptr.contents.value
    buf.arr = np.resize(buf.arr, new_length)
    return buf.arr.ctypes.data


class NumpyCBuffer(ctypes.Structure):
    _fields_ = [
        ("arr", ctypes.py_object),
        ("data", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("resize", type(numpy_buffer_resize_callback)),
    ]


class NumpyBuffer(AbstractBuffer, AbstractCArgument):
    """
    A buffer that uses NumPy arrays to store data. This is a concrete implementation
    of the AbstractBuffer class.
    """

    def __init__(self, arr: np.ndarray):
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError("NumPy array must be C-contiguous")
        self.arr = arr

    def length(self):
        return self.arr.size

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
        self._self_obj = ctypes.py_object(self)
        self._c_callback = numpy_buffer_resize_callback
        self._c_buffer = NumpyCBuffer(self._self_obj, data, length, self._c_callback)
        return ctypes.pointer(self._c_buffer)

    def deserialize_from_c(self, c_buffer):
        """
        Update this buffer based on how the C call modified the NumpyCBuffer structure.
        """


class NumpyBufferFormat(AbstractCFormat):
    """
    A format for buffers that uses NumPy arrays. This is a concrete implementation
    of the AbstractBufferFormat class.
    """

    def __init__(self, dtype: type):
        self._dtype = dtype

    def __call__(self, length: int):
        return NumpyBuffer(np.zeros(length, dtype=self._dtype))

    def unpack_c(self, ctx, name: str):
        c_dtype_type = np.ctypeslib.as_ctypes_type(self._dtype).__name__
        data = ctx.freshen("{name}_data")
        length = ctx.freshen("{name}_length")
        ctx.exec(f"""
            {c_dtype_type}* {data} = {name}->data;
            size_t {length} = {name}->length;
        """)
        ctx.post(f"""
            {name}->data = {data};
            {name}->length = {length};
        """)
        return NumpySymbolicCBuffer(self, name, data, length)


class NumpySymbolicCBuffer(AbstractSymbolicCBuffer):
    """
    A symbolic representation of a NumPy buffer.
    """

    def __init__(self, fmt, name, data, length):
        self.fmt = fmt
        self.name = name
        self.data = data
        self.length = length
    
    def c_length(self):
        return self.length

    def c_load(self, index_name: str):
        return f"{self.data}[{index_name}]"

    def c_store(self, index_name: str, value_name: str):
        return f"{self.data}[{index_name}] = {value_name};"

    def c_resize(self, ctx, new_length: str):
        name = self.name
        f"""
        {name}->data = {name}->resize(&({name}->arr), {new_length});
        {self.length} = new_length;
        {self.data} = {name}->data;
        """
