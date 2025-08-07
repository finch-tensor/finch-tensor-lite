import ctypes
from typing import NamedTuple

import numpy as np

from ..finch_assembly import Buffer
from .c import CBufferFType, CStackFType, c_type
from .numba_backend import NumbaBufferFType


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


class NumpyBuffer(Buffer):
    """
    A buffer that uses NumPy arrays to store data. This is a concrete implementation
    of the Buffer class.
    """

    def __init__(self, arr: np.ndarray):
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError("NumPy array must be C-contiguous")
        self.arr = arr

    @property
    def ftype(self):
        """
        Returns the ftype of the buffer, which is a NumpyBufferFType.
        """
        return NumpyBufferFType(self.arr.dtype.type)

    # TODO should be property
    def length(self):
        return self.arr.size

    def load(self, index: int):
        return self.arr[index]

    def store(self, index: int, value):
        self.arr[index] = value

    def resize(self, new_length: int):
        self.arr = np.resize(self.arr, new_length)


class NumpyBufferFType(CBufferFType, NumbaBufferFType, CStackFType):
    """
    A ftype for buffers that uses NumPy arrays. This is a concrete implementation
    of the BufferFType class.
    """

    def __init__(self, dtype: type):
        self._dtype = np.dtype(dtype).type

    def __eq__(self, other):
        if not isinstance(other, NumpyBufferFType):
            return False
        return self._dtype == other._dtype

    @property
    def length_type(self):
        """
        Returns the type used for the length of the buffer.
        """
        return np.intp

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        return self._dtype

    def __hash__(self):
        return hash(self._dtype)

    def __call__(self, len: int = 0, dtype: type | None = None):
        if dtype is None:
            dtype = self._dtype
        return NumpyBuffer(np.zeros(len, dtype=dtype))

    def c_type(self):
        return ctypes.POINTER(CNumpyBuffer)

    def c_length(self, ctx, buf):
        return buf.obj.length

    def c_data(self, ctx, buf):
        return buf.obj.data

    def c_load(self, ctx, buf, idx):
        return f"({buf.obj.data})[{ctx(idx)}]"

    def c_store(self, ctx, buf, idx, value):
        ctx.exec(f"{ctx.feed}({buf.obj.data})[{ctx(idx)}] = {ctx(value)};")

    def c_resize(self, ctx, buf, new_len):
        new_len = ctx(ctx.cache("len", new_len))
        data = buf.obj.data
        length = buf.obj.length
        obj = buf.obj.obj
        t = ctx.ctype_name(c_type(self._dtype))
        ctx.exec(
            f"{ctx.feed}{data} = ({t}*){obj}->resize(&{obj}->arr, {new_len});\n"
            f"{ctx.feed}{length} = {new_len};"
        )
        return

    def c_unpack(self, ctx, var_n, val):
        """
        Unpack the buffer into C context.
        """
        data = ctx.freshen(var_n, "data")
        length = ctx.freshen(var_n, "length")
        t = ctx.ctype_name(c_type(self._dtype))
        ctx.exec(
            f"{ctx.feed}{t}* {data} = ({t}*){ctx(val)}->data;\n"
            f"{ctx.feed}size_t {length} = {ctx(val)}->length;"
        )

        class BufferFields(NamedTuple):
            data: str
            length: str
            obj: str

        return BufferFields(data, length, var_n)

    def c_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from C context.
        """
        ctx.exec(
            f"{ctx.feed}{lhs}->data = (void*){obj.data};\n"
            f"{ctx.feed}{lhs}->length = {obj.length};"
        )
        return

    def serialize_to_c(self, obj):
        """
        Serialize the NumPy buffer to a C-compatible structure.
        """
        data = ctypes.c_void_p(obj.arr.ctypes.data)
        length = obj.arr.size
        obj._self_obj = ctypes.py_object(obj)
        obj._c_callback = numpy_buffer_resize_callback
        obj._c_buffer = CNumpyBuffer(obj._self_obj, data, length, obj._c_callback)
        return ctypes.pointer(obj._c_buffer)

    def deserialize_from_c(self, obj, c_buffer):
        """
        Update this buffer based on how the C call modified the CNumpyBuffer structure.
        """
        # this is handled by the resize callback

    def construct_from_c(self, c_buffer):
        """
        Construct a NumpyBuffer from a C-compatible structure.
        """
        self.arr = c_buffer.contents.arr
        return NumpyBuffer(self.arr)

    def numba_type(self):
        return list[np.ndarray]

    def numba_length(self, ctx, buf):
        arr = buf.obj.arr
        return f"len({arr})"

    def numba_load(self, ctx, buf, idx):
        arr = buf.obj.arr
        return f"{arr}[{ctx(idx)}]"

    def numba_store(self, ctx, buf, idx, val):
        arr = buf.obj.arr
        ctx.exec(f"{ctx.feed}{arr}[{ctx(idx)}] = {ctx(val)}")

    def numba_resize(self, ctx, buf, new_len):
        arr = buf.obj.arr
        ctx.exec(f"{ctx.feed}{arr} = numpy.resize({arr}, {ctx(new_len)})")

    def numba_unpack(self, ctx, var_n, val):
        """
        Unpack the buffer into Numba context.
        """
        arr = ctx.freshen(var_n, "arr")
        ctx.exec(f"{ctx.feed}{arr} = {ctx(val)}[0]")

        class BufferFields(NamedTuple):
            arr: str
            obj: str

        return BufferFields(arr, var_n)

    def numba_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from Numba context.
        """
        ctx.exec(f"{ctx.feed}{lhs}[0] = {obj.arr}")
        return

    def serialize_to_numba(self, obj):
        """
        Serialize the NumPy buffer to a Numba-compatible object.
        """
        return [obj.arr]

    def deserialize_from_numba(self, obj, numba_buffer):
        obj.arr = numba_buffer[0]
        return

    def construct_from_numba(self, numba_buffer):
        """
        Construct a NumpyBuffer from a Numba-compatible object.
        """
        return NumpyBuffer(numba_buffer[0])
