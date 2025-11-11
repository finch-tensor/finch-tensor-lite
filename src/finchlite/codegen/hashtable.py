import ctypes
import numpy as np

from finchlite.codegen.c import CBufferFType, CMapFType, CStackFType

from ..finch_assembly.map import Map
from ..util.print import qual_str


def _is_integer_pair(x):
    return (
        isinstance(x, tuple)
        and len(x) == 2
        and isinstance(x[0], int)
        and isinstance(x[1], int)
    )


class HashTable(Map):
    """
    A hashmap that maps (int64, int64) to int64

    Since we are only working with int64, we can afford to not use type generics
    """

    def __init__(self, map: "dict[tuple[int,int],int] | None" = None):
        if map is None:
            map = {}
        self.map = {}
        for key, value in map.items():
            if not _is_integer_pair(key):
                raise TypeError(f"Supplied key {key} is not a tuple of two integers")
            self.map[key] = value

    @property
    def ftype(self):
        """
        Returns the ftype of the buffer, which is a NumpyBufferFType.
        """
        return HashTableFType()

    def exists(self, idx1: int, idx2: int) -> bool:
        return (idx1, idx2) in self.map

    def load(self, idx1: int, idx2: int):
        return self.map[(idx1, idx2)]

    def store(self, idx1: int, idx2: int, val):
        self.map[(idx1,idx2)] = val

    def __str__(self):
        return f"hashtable({self.map})"


class HashTableFType(CMapFType, CStackFType):
    """
    A ftype for buffers that uses NumPy arrays. This is a concrete implementation
    of the BufferFType class.
    """

    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, HashTableFType)

    def __str__(self):
        return f"hashtable_t"

    def __repr__(self):
        return f"HashTableFType()"

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
        """
        I do not understand why this is here now but sure.
        """
        return hash(int)

    def __call__(self):
        return HashTable()

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
            numba.types.Array(numba.from_dtype(self.element_type), 1, "C")
        )

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
