import ctypes
from typing import NamedTuple

import numpy as np

import numba

from finchlite.algebra import FType, TupleFType, ftype, ftypes
from finchlite.codegen.c_codegen import CBufferFType, CContext, CUnpackableFType, c_type
from finchlite.codegen.numba_codegen import NumbaBufferFType, to_numpy_type
from finchlite.finch_assembly import Buffer
from finchlite.finch_assembly.nodes import AssemblyExpression
from finchlite.util import qual_str


class NumbaBufferFields(NamedTuple):
    arr: str
    obj: str


class CBufferFields(NamedTuple):
    data: str
    length: str
    obj: str


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
        return NumpyBufferFType(ftype(self.arr.dtype))

    # TODO should be property
    def length(self):
        return self.arr.size

    def load(self, index: int):
        value = self.arr[index]
        if isinstance(self.ftype.element_type, TupleFType):
            return tuple(
                value[name] for name in self.ftype.element_type.struct_fieldnames
            )
        return value

    def store(self, index: int, value):
        self.arr[index] = value

    def resize(self, new_length: int):
        self.arr = np.resize(self.arr, new_length)

    def __str__(self):
        arr_str = str(self.arr).replace("\n", "")
        return f"np_buf({arr_str})"

    def __repr__(self):
        arr_repr = repr(self.arr).replace("\n", "")
        return f"NumpyBuffer({arr_repr})"


class NumpyBufferFType(CBufferFType, NumbaBufferFType, CUnpackableFType):
    """
    A ftype for buffers that uses NumPy arrays. This is a concrete implementation
    of the BufferFType class.
    """

    def __init__(self, element_type: FType):
        self._element_type = ftype(to_numpy_type(ftype(element_type)))

    @property
    def _dtype(self):
        return to_numpy_type(self._element_type)

    def __eq__(self, other):
        if not isinstance(other, NumpyBufferFType):
            return False
        return self._element_type == other._element_type

    def __str__(self):
        return f"np_buf_t({qual_str(self._dtype.type)})"

    def __repr__(self):
        return f"NumpyBufferFType({repr(self._element_type)})"

    @property
    def length_type(self):
        """
        Returns the type used for the length of the buffer.
        """
        return ftypes.intp

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        return self._element_type

    def __hash__(self):
        return hash(self._element_type)

    def __call__(self, len: int = 0, element_type: FType | None = None):
        return NumpyBuffer(np.zeros(len, dtype=self._dtype))

    def c_type(self):
        return ctypes.POINTER(CNumpyBuffer)

    def c_length(self, ctx: "CContext", buf: CBufferFields):
        return buf.length

    def c_data(self, ctx: "CContext", buf: CBufferFields):
        return buf.data

    def c_load(self, ctx: "CContext", buf: CBufferFields, idx: "AssemblyExpression"):
        return f"({buf.data})[{ctx(idx)}]"

    def c_store(
        self,
        ctx: "CContext",
        buf: CBufferFields,
        idx: "AssemblyExpression",
        value: "AssemblyExpression",
    ):
        ctx.exec(f"{ctx.feed}({buf.data})[{ctx(idx)}] = {ctx(value)};")

    def c_resize(self, ctx, buf: CBufferFields, new_len):
        new_len = ctx(ctx.cache("len", new_len))
        data = buf.data
        length = buf.length
        obj = buf.obj
        t = ctx.ctype_name(c_type(self.element_type))
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
        t = ctx.ctype_name(c_type(self.element_type))
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
        return NumpyBuffer(c_buffer.contents.arr)

    def numba_type(self) -> type:
        return list[np.ndarray]

    def numba_jitclass_type(self) -> numba.types.Type:
        return numba.types.ListType(
            numba.types.Array(numba.from_dtype(self._dtype), 1, "C")
        )

    def numba_length(self, ctx, buf: NumbaBufferFields):
        arr = buf.arr
        return f"len({arr})"

    def numba_load(self, ctx, buf, idx):
        arr = buf.obj.arr
        if isinstance(self.element_type, TupleFType):
            idx = ctx(ctx.cache("idx", idx))
            fields = ", ".join(
                f"{arr}[{idx}]['{name}']"
                for name in self.element_type.struct_fieldnames
            )
            return f"({fields},)"
        return f"{arr}[{ctx(idx)}]"

    def numba_store(self, ctx, buf, idx, val):
        arr = buf.obj.arr
        if isinstance(self.element_type, TupleFType):
            idx = ctx(ctx.cache("idx", idx))
            val = ctx.cache("val", val)
            val_code = ctx(val)
            for i, name in enumerate(self.element_type.struct_fieldnames):
                ctx.exec(f"{ctx.feed}{arr}[{idx}]['{name}'] = {val_code}[{i}]")
            return
        ctx.exec(f"{ctx.feed}{arr}[{ctx(idx)}] = {ctx(val)}")

    def numba_resize(self, ctx, buf: NumbaBufferFields, new_len):
        arr = buf.arr
        ctx.exec(f"{ctx.feed}{arr} = numpy.resize({arr}, {ctx(new_len)})")

    def numba_unpack(self, ctx, var_n, val):
        """
        Unpack the buffer into Numba context.
        """
        arr = ctx.freshen(var_n, "arr")
        ctx.exec(f"{ctx.feed}{arr} = {ctx(val)}[0]")

        return NumbaBufferFields(arr, var_n)

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
        return numba.typed.List([obj.arr])

    def deserialize_from_numba(self, obj, numba_buffer):
        obj.arr = numba_buffer[0]
        return

    def construct_from_numba(self, numba_buffer):
        """
        Construct a NumpyBuffer from a Numba-compatible object.
        """
        return NumpyBuffer(numba_buffer[0])
