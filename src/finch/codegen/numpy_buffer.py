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

    def get_format(self):
        """
        Returns the format of the buffer, which is a NumpyBufferFormat.
        """
        return NumpyBufferFormat(self.arr.dtype.type)

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
        data = ctx.resolve(name, "data")
        length = ctx.resolve(name, "length")
        t = ctx.ctype_name(np.ctypeslib.as_ctypes_type(self._dtype))
        ctx.exec(
            f"{ctx.feed}{t}* {data} = ({t}*){name}->data;\n"
            f"{ctx.feed}size_t {length} = {name}->length;"
        )
        ctx.post(
            f"{ctx.feed}{name}->data = {data};\n"
            f"{ctx.feed}{name}->length = {length};"
        )
        return NumpySymbolicCBuffer(self, name, data, length)
    
    def c_type(self):
        return NumpyCBuffer

    def c_length(self, ctx, name: str):
        length = ctx.resolve(name, "length")
        return length

    def c_load(self, ctx, name, index: str):
        data = ctx.resolve(name, "data")
        return f"{data}[{index}]"

    def c_store(self, ctx, name, index: str, value: str):
        data = ctx.resolve(name, "data")
        ctx.exec(f"{ctx.feed}{self.data}[{index}] = {value};")

    def c_resize(self, ctx, new_length: str):
        length = ctx.resolve(name, "length")
        data = ctx.resolve(name, "data")
        name = self.name
        ctx.exec(
            f"{ctx.feed}{self.data} = {name}->resize(&({name}->arr), {new_length});\n"
            f"{ctx.feed}{self.length} = new_length;"
        )