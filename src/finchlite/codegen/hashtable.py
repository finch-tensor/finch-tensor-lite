import ctypes
from typing import NamedTuple

import numba

from finchlite.codegen.c import CContext, CMapFType, CStackFType
from finchlite.codegen.numba_backend import NumbaContext, NumbaMapFType, NumbaStackFType
from finchlite.finch_assembly.nodes import AssemblyExpression, AssemblyNode, Stack
from finchlite.finch_assembly.struct import TupleFType

from ..finch_assembly.map import Map, MapFType


class NumbaMapFields(NamedTuple):
    """
    This is a field that extracts out the map from the obj variable. Its
    purpose is so that we can extract out map from obj in unpack, do
    computations on the map variable, and re-insert that into obj in repack.
    """

    map: str
    obj: str


class CMapFields(NamedTuple):
    """
    TODO: for the C backend, we will pulling in a completely different library
    to do the actual hash function implementation. Should we even try to
    convert back?
    """

    map: str
    obj: str

def _is_integer_tuple(tup, size):
    if not isinstance(tup, tuple) or len(tup) != size:
        return False
    for elt in tup:
        if not isinstance(elt, int):
            return False
    return True


class CHashMap(ctypes.Structure):
    _fields_ = [
        ("map", ctypes.py_object),
        ("data", ctypes.c_void_p),
        # TODO: you need more methods to work with the data.
    ]


class HashTable(Map):
    """
    A Hash Table that maps Z^{in_len} to Z^{out_len}
    """

    def __init__(
        self, key_len, value_len, map: "dict[tuple[int,int],int] | None" = None
    ):
        """
        Constructor for the Hash Table, which maps integer tuples to integer tuples. Takes three arguments:
        in_len: The
        """
        if map is None:
            map = {}
        self.map = {}
        for key, value in map.items():
            if not _is_integer_tuple(key, key_len):
                raise TypeError(
                    f"Supplied key {key} is not a tuple of {key_len} integers"
                )
            if not _is_integer_tuple(value, value_len):
                raise TypeError(
                    f"Supplied value {key} is not a tuple of {value_len} integers"
                )
            self.map[key] = value
        self.key_len = key_len
        self.value_len = value_len

    @property
    def ftype(self):
        """
        Returns the ftype of the buffer, which is a NumpyBufferFType.
        """
        return HashTableFType(self.key_len, self.value_len)

    def exists(self, idx) -> bool:
        assert _is_integer_tuple(idx, self.key_len)
        return idx in self.map

    def load(self, idx):
        assert _is_integer_tuple(idx, self.key_len)
        return self.map[idx]

    def store(self, idx, val):
        assert _is_integer_tuple(idx, self.key_len)
        assert _is_integer_tuple(val, self.value_len)
        self.map[idx] = val

    def __str__(self):
        return f"hashtable({self.map})"


class HashTableFType(MapFType, CMapFType, CStackFType, NumbaMapFType, NumbaStackFType):
    """
    A ftype for buffers that uses NumPy arrays. This is a concrete implementation
    of the BufferFType class.
    """

    def __init__(self, key_len, value_len):
        self.key_len = key_len
        self.value_len = value_len
        self._key_type = TupleFType.from_tuple(tuple(int for _ in range(key_len)))
        self._value_type = TupleFType.from_tuple(tuple(int for _ in range(value_len)))

    def __eq__(self, other):
        if not isinstance(other, HashTableFType):
            return False
        return self.key_len == other.key_len and self.value_len == other.value_len

    def __str__(self):
        return f"hashtable_t({self.key_len}, {self.value_len})"

    def __repr__(self):
        return f"HashTableFType({self.key_len}, {self.value_len})"

    @property
    def key_type(self):
        """
        Returns the type used for the length of the buffer.
        """
        return self._key_type

    @property
    def element_type(self):
        """
        Returns the type of elements stored in the buffer.
        This is typically the same as the dtype used to create the buffer.
        """
        return self._value_type

    def numba_jitclass_type(self) -> numba.types.Type:
        key_t = numba.types.UniTuple(numba.types.int64, self.key_len)
        value_t = numba.types.UniTuple(numba.types.int64, self.value_len)
        return numba.types.ListType(numba.types.DictType(key_t, value_t))

    def numba_existsmap(self, ctx: "NumbaContext", map: "Stack", idx: "AssemblyExpression"):
        assert isinstance(map.obj, NumbaMapFields)
        return f"({ctx(idx)}) in {map.obj.map}"

    def numba_loadmap(self, ctx: "NumbaContext", map: "Stack", idx: "AssemblyExpression"):
        assert isinstance(map.obj, NumbaMapFields)
        return f"{map.obj.map}[{ctx(idx)}]"

    def numba_storemap(
        self,
        ctx: "NumbaContext",
        map: "Stack",
        idx: "AssemblyExpression",
        value: "AssemblyExpression",
    ):
        assert isinstance(map.obj, NumbaMapFields)
        ctx.exec(f"{map.obj.map}[{ctx(idx)}] = {ctx(value)}")

    def numba_unpack(self, ctx: "NumbaContext", var_n: str, val: "AssemblyExpression") -> NumbaMapFields:
        """
        Unpack the buffer into Numba context.
        This is part of a step that will create a new slot for var_n
        that contains variable names corresponding to the unpacked fields.
        """
        # the val field will always be asm.Variable(var_n, var_t)
        map = ctx.freshen(var_n, "map")
        ctx.exec(f"{ctx.feed}{map} = {ctx(val)}[0]")

        return NumbaMapFields(map, var_n)

    def numba_repack(self, ctx: "NumbaContext", lhs: str, obj: "NumbaMapFields"):
        """
        Repack the buffer from Numba context.
        """
        # obj is the fields corresponding to the self.slots[lhs]
        ctx.exec(f"{ctx.feed}{lhs}[0] = {obj.map}")

    def serialize_to_numba(self, obj: "HashTable"):
        """
        Serialize the hashmap to a Numba-compatible object.

        We will supply the input and output length
        """
        return numba.typed.List([obj.map])

    def deserialize_from_numba(self, obj: "HashTable", numba_map: "list[dict]"):
        obj.map = numba_map[0]
        return

    def construct_from_numba(self, numba_buffer):
        """
        Construct a numba buffer from a Numba-compatible object.
        """
        return HashTable(self.key_len, self.value_len, numba_buffer[0])

    def __hash__(self):
        """
        Needs to be here because you are going to be using this type as a key
        in dictionaries.
        """
        return hash(("HashTableFType", self.key_len, self.value_len))

    def __call__(self):
        return HashTable(self.key_len, self.value_len, {})

    def c_type(self):
        return ctypes.POINTER(CHashMap)

    def c_load(self, ctx: "CContext", buf, idx):
        return f"({buf.obj.data})[{ctx(idx)}]"

    def c_store(self, ctx: "CContext", buf, idx, value):
        ctx.exec(f"{ctx.feed}({buf.obj.data})[{ctx(idx)}] = {ctx(value)};")

    def serialize_to_c(self, obj):
        """
        Serialize the Hash Map to a CHashMap structure.
        """
        data = ctypes.c_void_p(obj.arr.ctypes.data)
        obj._c_buffer = CHashMap(obj._self_obj, data)
        return ctypes.pointer(obj._c_buffer)

    def deserialize_from_c(self, obj, c_buffer):
        """
        Update this buffer based on how the C call modified the CNumpyBuffer structure.
        """
        # this is handled by the resize callback

    def construct_from_c(self, c_map):
        """
        Construct a NumpyBuffer from a C-compatible structure.
        """
        return NumpyBuffer(c_map.contents.map)
