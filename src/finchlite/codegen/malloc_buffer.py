import ctypes
from pathlib import Path

import numpy as np

from finchlite.codegen.c import CBufferFType, CStackFType, c_type, load_shared_lib
from finchlite.codegen.numpy_buffer import CBufferFields
from finchlite.finch_assembly import Buffer
from finchlite.util import qual_str

backend_lib = load_shared_lib(
    (Path(__file__).parent / "malloc_buffer_backend.c").read_text(encoding="utf-8")
)


class CMallocBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("datasize", ctypes.c_size_t),
        ("resize", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)),
    ]


class MallocBuffer(Buffer):
    """
    A buffer that uses Malloc buffers to store data.

    To check out the corresponding C code, you should reference
    ./malloc_buffer_backend.c in the same directory as the malloc_buffer.py
    file
    """

    def __init__(self, length: int, dtype, data=None):
        """
        Constructor for the MallocBuffer class.

        length (int): the length of the malloc array.
        dtype (type[ctypes._CData]): the ctype that the buffer will be based on.
        data (optional): a list of data to initialize the buffer with.
        """
        self._dtype = dtype
        self._c_dtype = c_type(dtype)
        self.buffer = ctypes.pointer(CMallocBuffer())
        backend_lib.mallocbuffer_init(
            self.buffer,
            ctypes.c_size_t(ctypes.sizeof(self._c_dtype)),
            ctypes.c_size_t(length),
        )
        if data is None:
            return
        if len(data) > length:
            raise IndexError
        castarray = ctypes.cast(
            self.buffer.contents.data, ctypes.POINTER(self._c_dtype)
        )
        for idx, elt in enumerate(data):
            castarray[idx] = self._c_dtype(elt)

    def __del__(self):
        """
        Frees the mallocbuffer stored inside.
        """
        if hasattr(self, "buffer"):
            backend_lib.mallocbuffer_free(self.buffer)

    @property
    def ftype(self):
        """
        Returns the ftype of the buffer, which is a MallocBufferFType.
        """
        return MallocBufferFType(self._dtype)

    # TODO should be property
    def length(self):
        return np.intp(self.buffer.contents.length)

    def load(self, index: int):
        return self._dtype(
            ctypes.cast(self.buffer.contents.data, ctypes.POINTER(self._c_dtype))[index]
        )

    def store(self, index: int, value):
        ctypes.cast(self.buffer.contents.data, ctypes.POINTER(self._c_dtype))[index] = (
            self._dtype(value)
        )

    def resize(self, new_length: int):
        self.buffer.contents.resize(self.buffer, ctypes.c_size_t(new_length))

    def __str__(self):
        array = ctypes.cast(self.buffer.contents.data, ctypes.POINTER(self._c_dtype))[
            : self.length()
        ]
        return f"malloc_buf({array})"

    def copy(self):
        new_buf = MallocBuffer(self.length(), self._dtype)
        cast_src = ctypes.cast(
            self.buffer.contents.data, ctypes.POINTER(self._c_dtype)
        )
        cast_dst = ctypes.cast(
            new_buf.buffer.contents.data, ctypes.POINTER(self._c_dtype)
        )
        for i in range(self.length()):
            cast_dst[i] = cast_src[i]
        return new_buf


class MallocBufferFType(CBufferFType, CStackFType):
    """
    A ftype for buffers that uses libc-provided malloc functions. This is a
    concrete implementation of the BufferFType class.

    This does not support the numba backend.
    """

    def __init__(self, dtype):
        self._dtype = dtype
        self._c_dtype = c_type(dtype)

    def __eq__(self, other):
        if not isinstance(other, MallocBufferFType):
            return False
        return self._dtype == other._dtype

    def __str__(self):
        return f"malloc_buf_t({qual_str(self._dtype)})"

    def __repr__(self):
        return f"MallocBufferFType({qual_str(self._dtype)})"

    @property
    def length_type(self):
        """
        Returns the type used for the length of the buffer.
        """
        return np.intp

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the buffer. This will be a ctypes array.
        """
        return self._dtype

    def __hash__(self):
        return hash(self._dtype)

    def __call__(self, len: int = 0, dtype: type | None = None):
        if dtype is None:
            dtype = self._dtype
        return MallocBuffer(len, dtype)

    def c_type(self):
        return ctypes.POINTER(CMallocBuffer)

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
        obj = buf.obj.obj
        data = buf.obj.data
        length = buf.obj.length
        t = ctx.ctype_name(c_type(self._dtype))
        ctx.exec(
            f"{ctx.feed}{data} = ({t}*){obj}->resize({obj}, ({new_len}));\n"
            f"{ctx.feed}{length} = {new_len};"
        )
        return

    def c_unpack(self, ctx, var_n, val):
        """
        Unpack the malloc buffer into C context.
        """
        data = ctx.freshen(var_n, "data")
        length = ctx.freshen(var_n, "length")
        t = ctx.ctype_name(c_type(self._dtype))
        ctx.add_header("#include <stddef.h>")
        ctx.exec(
            f"{ctx.feed}{t}* {data} = ({t}*){ctx(val)}->data;\n"
            f"{ctx.feed}size_t {length} = {ctx(val)}->length;"
        )

        return CBufferFields(data, length, var_n)

    def c_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from C context.
        """
        ctx.exec(
            f"{ctx.feed}{lhs}->data = (void*){obj.data};\n"
            f"{ctx.feed}{lhs}->length = {obj.length};"
        )
        return

    def serialize_to_c(self, obj: MallocBuffer):
        """
        Serialize the Malloc buffer to a C-compatible structure.
        This is trivial.
        """
        return obj.buffer

    def deserialize_from_c(self, obj, c_buffer):
        pass
        # this is handled by the resize callback

    def construct_from_c(self, c_buffer):
        """
        Construct a MallocBuffer from a C-compatible structure.
        """
        return c_buffer.contents
