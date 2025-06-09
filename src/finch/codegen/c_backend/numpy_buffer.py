import ctypes

import numpy as np

from ...codegen.numpy_buffer import AbstractNumpyBuffer, AbstractNumpyBufferFormat
from ._c import CArgument, CBufferFormat, c_type


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


class NumpyBuffer(AbstractNumpyBuffer, CArgument):
    """
    A buffer that uses NumPy arrays to store data. This is a concrete implementation
    of the AbstractBuffer class.
    """

    def get_format(self):
        """
        Returns the format of the buffer, which is a NumpyBufferFormat.
        """
        return NumpyBufferFormat(self.arr.dtype.type)

    def finalize(self):
        return self

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


class NumpyBufferFormat(AbstractNumpyBufferFormat, CBufferFormat):
    """
    A format for buffers that uses NumPy arrays. This is a concrete implementation
    of the AbstractNumpyBufferFormat class.
    """

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
        arr = f"{ctx(buf)}->arr"
        length = f"{ctx(buf)}->length"
        ctx.exec(
            f"{ctx.feed}{data} = {ctx(buf)}->resize(&{arr}, {ctx(new_len)});\n"
            f"{ctx.feed}{length} = {ctx(new_len)};"
        )
