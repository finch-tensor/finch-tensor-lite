import ctypes

import numpy as np

from ..finch_assembly.abstract_buffer import Buffer
from .c import CArgument, CBufferFormat, c_type
from .numba_backend import NumbaArgument, NumbaBufferFormat


@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(ctypes.py_object), ctypes.c_size_t)
def numpy_buffer_resize_callback(buf_ptr, new_length):
    """
    A Python callback function that resizes the NumPy array.
    """
    buf = buf_ptr.contents.value
    buf.arr_ref[0] = np.resize(buf.arr_ref[0], new_length)
    return buf.arr_ref[0].ctypes.data


class NumpyCBuffer(ctypes.Structure):
    _fields_ = [
        ("arr_ref", ctypes.py_object),
        ("data", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("resize", type(numpy_buffer_resize_callback)),
    ]


class NumpyBuffer(Buffer, CArgument, NumbaArgument):
    """
    A buffer that uses NumPy arrays to store data. This is a concrete implementation
    of the Buffer class.
    """

    def __init__(self, arr: np.ndarray):
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError("NumPy array must be C-contiguous")
        self.arr_ref = [arr]

    def get_format(self):
        """
        Returns the format of the buffer, which is a NumpyBufferFormat.
        """
        return NumpyBufferFormat(self.arr_ref[0].dtype.type)

    def length(self):
        return self.arr_ref[0].size

    def load(self, index: int):
        return self.arr_ref[0][index]

    def store(self, index: int, value):
        self.arr_ref[0][index] = value

    def resize(self, new_length: int):
        self.arr_ref[0] = np.resize(self.arr_ref[0], new_length)

    def serialize_to_c(self):
        """
        Serialize the NumPy buffer to a C-compatible structure.
        """
        data = ctypes.c_void_p(self.arr_ref[0].ctypes.data)
        length = self.arr_ref[0].size
        self._self_obj = ctypes.py_object(self)
        self._c_callback = numpy_buffer_resize_callback
        self._c_buffer = NumpyCBuffer(self._self_obj, data, length, self._c_callback)
        return ctypes.pointer(self._c_buffer)

    def deserialize_from_c(self, c_buffer):
        """
        Update this buffer based on how the C call modified the NumpyCBuffer structure.
        """

    def serialize_to_numba(self):
        return self.arr_ref

    @classmethod
    def deserialize_from_numba(cls, numba_buffer):
        return cls(numba_buffer[0])


class NumpyBufferFormat(CBufferFormat, NumbaBufferFormat):
    """
    A format for buffers that uses NumPy arrays. This is a concrete implementation
    of the BufferFormat class.
    """

    def __init__(self, dtype: type):
        self._dtype = dtype

    def __eq__(self, other):
        if not isinstance(other, NumpyBufferFormat):
            return False
        return self._dtype == other._dtype

    @property
    def length_type(self):
        """
        Returns the type used for the length of the buffer.
        """
        return int

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        return self._dtype

    def __hash__(self):
        return hash(self._dtype)

    def __call__(self, len_: int):
        return NumpyBuffer(np.zeros(len_, dtype=self._dtype))

    def c_type(self):
        return ctypes.POINTER(NumpyCBuffer)

    def c_length(self, ctx, buf):
        return f"{ctx(buf)}->length"

    def c_load(self, ctx, buf, idx):
        t = ctx.ctype_name(c_type(self._dtype))
        return f"(({t}*){ctx(buf)}->data)[{ctx(idx)}]"

    def c_store(self, ctx, buf, idx, value):
        data = f"{ctx(buf)}->data"
        t = ctx.ctype_name(c_type(self._dtype))
        ctx.exec(f"{ctx.feed}(({t}*){data})[{ctx(idx)}] = {ctx(value)};")

    def c_resize(self, ctx, buf, new_len):
        data = f"{ctx(buf)}->data"
        arr = f"{ctx(buf)}->arr_ref"
        length = f"{ctx(buf)}->length"
        ctx.exec(
            f"{ctx.feed}{data} = {ctx(buf)}->resize(&{arr}, {ctx(new_len)});\n"
            f"{ctx.feed}{length} = {ctx(new_len)};"
        )
