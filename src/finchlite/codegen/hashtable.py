import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypeAlias, TypedDict

import numba

from finchlite.codegen.c import (
    CContext,
    CMapFType,
    CStackFType,
    c_type,
    load_shared_lib,
)
from finchlite.codegen.numba_backend import NumbaContext, NumbaMapFType, NumbaStackFType
from finchlite.finch_assembly.map import Map, MapFType
from finchlite.finch_assembly.nodes import AssemblyExpression, Stack
from finchlite.finch_assembly.struct import TupleFType

stcpath = Path(__file__).parent / "stc"
hashmap_h = stcpath / "stc" / "hashmap.h"


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


class CHashMapStruct(ctypes.Structure):
    _fields_ = [
        ("map", ctypes.c_void_p),
        ("obj", ctypes.py_object),
    ]


class CHashMethods(TypedDict):
    init: str
    exists: str
    load: str
    store: str
    cleanup: str


@dataclass
class CHashTableLibrary:
    library: ctypes.CDLL
    methods: CHashMethods
    hmap_t: str


# implement the hash table datastructures
class CHashTable(Map):
    """
    CHashTable class that basically connects up to an STC library.
    """

    libraries: dict[tuple[int, int], CHashTableLibrary] = {}

    @classmethod
    def gen_code(
        cls,
        ctx: "CContext",
        key_type: "TupleFType",
        value_type: "TupleFType",
    ) -> tuple[CHashMethods, str]:

        assert isinstance(key_type, TupleFType)
        assert isinstance(value_type, TupleFType)

        key_len = len(key_type.struct_fields)
        value_len = len(key_type.struct_fields)
        # dereference both key and value types; as given, they are both pointers.
        keytype_c = ctx.ctype_name(c_type(key_type)._type_)
        valuetype_c = ctx.ctype_name(c_type(value_type)._type_)
        hmap_t = ctx.freshen(f"hmap", key_len, value_len)

        ctx.add_header(f"#define T {hmap_t}, {keytype_c}, {valuetype_c}")
        ctx.add_header("#define i_eq c_memcmp_eq")
        ctx.add_header("#include <stdlib.h>")
        ctx.add_header(f'#include "{hashmap_h}"')

        methods: CHashMethods = {
            "init": ctx.freshen("finch_hmap_init", key_len, value_len),
            "exists": ctx.freshen("finch_hmap_exists", key_len, value_len),
            "load": ctx.freshen("finch_hmap_load", key_len, value_len),
            "store": ctx.freshen("finch_hmap_store", key_len, value_len),
            "cleanup": ctx.freshen("finch_hmap_cleanup", key_len, value_len),
        }
        # register these methods in the datastructures.
        ctx.datastructures[HashTableFType(key_len, value_len)] = methods

        # basically for the load functions, you need to provide a variable that
        # can be copied.
        # Yeah, so which API's should we use for load and store?
        lib_code = f"""
#include <stdio.h>
void* {methods['init']}() {{
    void* ptr = malloc(sizeof({hmap_t}));
    memset(ptr, 0, sizeof({hmap_t}));
    return ptr;
}}
bool {methods['exists']}({hmap_t} *map, {keytype_c}* key) {{
    return {hmap_t}_contains(map, *key);
}}
void {methods['load']}({hmap_t} *map, {keytype_c}* key, {valuetype_c}* value) {{
    const {valuetype_c}* internal_val = {hmap_t}_at(map, *key);
    *value = *internal_val;
}}
void {methods['store']}({hmap_t} *map, {keytype_c}* key, {valuetype_c}* value) {{
    {hmap_t}_insert_or_assign(map, *key, *value);
}}
void {methods['cleanup']}(void* ptr) {{
    {hmap_t}* hptr = ptr;
    {hmap_t}_drop(hptr);
    free(hptr);
}}
        """
        ctx.exec(lib_code)

        return methods, hmap_t

    @classmethod
    def compile(cls, key_len: int, value_len: int) -> CHashTableLibrary:
        """
        compile a library to use for the c hash table.
        """
        if (key_len, value_len) in cls.libraries:
            return cls.libraries[(key_len, value_len)]
        key_type = TupleFType.from_tuple(tuple(int for _ in range(key_len)))
        value_type = TupleFType.from_tuple(tuple(int for _ in range(value_len)))

        ctx = CContext()
        methods, hmap_t = cls.gen_code(ctx, key_type, value_type)
        code = ctx.emit_global()
        lib = load_shared_lib(code)

        # get keystruct and value types
        KeyStruct = c_type(key_type)._type_
        ValueStruct = c_type(value_type)._type_

        init_func = getattr(lib, methods["init"])
        init_func.argtypes = []
        init_func.restype = ctypes.c_void_p

        # Exists: Takes (map*, key*) -> returns bool
        exists_func = getattr(lib, methods["exists"])
        exists_func.argtypes = [ctypes.c_void_p, ctypes.POINTER(KeyStruct)]
        exists_func.restype = ctypes.c_bool

        # Load: Takes (map*, key*, out_val*) -> returns void
        load_func = getattr(lib, methods["load"])
        load_func.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(KeyStruct),
            ctypes.POINTER(ValueStruct),
        ]
        load_func.restype = None

        # Store: Takes (map*, key*, val*) -> returns void
        store_func = getattr(lib, methods["store"])
        store_func.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(KeyStruct),
            ctypes.POINTER(ValueStruct),
        ]
        store_func.restype = None

        # Cleanup: Takes (map*) -> returns void
        cleanup_func = getattr(lib, methods["cleanup"])
        cleanup_func.argtypes = [ctypes.c_void_p]
        cleanup_func.restype = None

        cls.libraries[(key_len, value_len)] = CHashTableLibrary(lib, methods, hmap_t)
        return cls.libraries[(key_len, value_len)]

    def __init__(
        self, key_len: int, value_len: int, map: "dict[tuple,tuple] | None" = None
    ):
        """
        Constructor for the C Hash Table
        """
        self.lib = self.__class__.compile(key_len, value_len)

        self.key_len = key_len
        self.value_len = value_len

        if map is None:
            map = {}
        self.map = getattr(self.lib.library, self.lib.methods["init"])()
        for key, value in map.items():
            if not _is_integer_tuple(key, key_len):
                raise TypeError(
                    f"Supplied key {key} is not a tuple of {key_len} integers"
                )
            if not _is_integer_tuple(value, value_len):
                raise TypeError(
                    f"Supplied value {key} is not a tuple of {value_len} integers"
                )
            self.store(key, value)

    def __del__(self):
        getattr(self.lib.library, self.lib.methods["cleanup"])(self.map)

    def exists(self, idx: tuple) -> bool:
        assert _is_integer_tuple(idx, self.key_len)
        KeyStruct = c_type(self.ftype().key_type)._type_
        c_key = KeyStruct(*idx)
        func = getattr(self.lib.library, self.lib.methods["exists"])
        func.restype = ctypes.c_bool
        return func(self.map, ctypes.byref(c_key))

    def load(self, idx):
        assert _is_integer_tuple(idx, self.key_len)
        KeyStruct = c_type(self.ftype().key_type)._type_
        ValueStruct = c_type(self.ftype().value_type)._type_
        c_key = KeyStruct(*idx)
        c_value = ValueStruct()
        getattr(self.lib.library, self.lib.methods["load"])(
            self.map, ctypes.byref(c_key), ctypes.byref(c_value)
        )
        return tuple(getattr(c_value, f) for f, _ in c_value._fields_)

    def store(self, idx, val):
        assert _is_integer_tuple(idx, self.key_len)
        assert _is_integer_tuple(val, self.value_len)
        KeyStruct = c_type(self.ftype().key_type)._type_
        ValueStruct = c_type(self.ftype().value_type)._type_
        c_key = KeyStruct(*idx)
        c_value = ValueStruct()
        getattr(self.lib.library, self.lib.methods["store"])(
            self.map, ctypes.byref(c_key), ctypes.byref(c_value)
        )

    def __str__(self):
        return f"hashtable({self.map})"

    def ftype(self):
        return HashTableFType(self.key_len, self.value_len)


class HashTable(Map):
    """
    A Hash Table that maps Z^{in_len} to Z^{out_len}
    """

    def __init__(self, key_len, value_len, map: "dict[tuple,tuple] | None" = None):
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
        Returns the finch type of this hash table.
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


# class HashTableFType(MapFType, CMapFType, CStackFType, NumbaMapFType, NumbaStackFType):
class HashTableFType(CMapFType, NumbaMapFType, CStackFType, NumbaStackFType):
    """
    A ftype for buffers that uses NumPy arrays. This is a concrete implementation
    of the BufferFType class.
    """

    def __init__(self, key_len: int, value_len: int):
        self.key_len = key_len
        self.value_len = value_len
        self._key_type = TupleFType.from_tuple(tuple(int for _ in range(key_len)))
        self._value_type = TupleFType.from_tuple(tuple(int for _ in range(value_len)))

    def __eq__(self, other):
        if not isinstance(other, HashTableFType):
            return False
        return self.key_len == other.key_len and self.value_len == other.value_len

    def __call__(self):
        return HashTable(self.key_len, self.value_len, {})

    def __str__(self):
        return f"hashtable_t({self.key_len}, {self.value_len})"

    def __repr__(self):
        return f"HashTableFType({self.key_len}, {self.value_len})"

    @property
    def key_type(self):
        """
        Returns the type of elements used as the keys of the hash table.
        (some integer tuple)
        """
        return self._key_type

    @property
    def value_type(self):
        """
        Returns the type of elements used as the value of the hash table.
        (some integer tuple)
        """
        return self._value_type

    def __hash__(self):
        """
        This method needs to be here because you are going to be using this
        type as a key in dictionaries.
        """
        return hash(("HashTableFType", self.key_len, self.value_len))

    """
    Methods for the Numba Backend
    """

    def numba_jitclass_type(self) -> numba.types.Type:
        key_t = numba.types.UniTuple(numba.types.int64, self.key_len)
        value_t = numba.types.UniTuple(numba.types.int64, self.value_len)
        return numba.types.ListType(numba.types.DictType(key_t, value_t))

    def numba_existsmap(
        self, ctx: "NumbaContext", map: "Stack", idx: "AssemblyExpression"
    ):
        assert isinstance(map.obj, NumbaMapFields)
        return f"({ctx(idx)}) in {map.obj.map}"

    def numba_loadmap(
        self, ctx: "NumbaContext", map: "Stack", idx: "AssemblyExpression"
    ):
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

    def numba_unpack(
        self, ctx: "NumbaContext", var_n: str, val: "AssemblyExpression"
    ) -> NumbaMapFields:
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

    def construct_from_numba(self, numba_buffer):
        """
        Construct a numba buffer from a Numba-compatible object.
        """
        return HashTable(self.key_len, self.value_len, numba_buffer[0])

    """
    Methods for the C Backend
    This requires an external library (stc) to work.
    """

    def c_type(self):
        return ctypes.POINTER(CHashMapStruct)

    def c_existsmap(self, ctx: "CContext", map: "Stack", idx: "AssemblyExpression"):
        # TODO: call in the methods from the c library.
        assert isinstance(map.obj, CMapFields)
        methods: CHashMethods = ctx.datastructures[self]
        return (f"{ctx.feed}{methods['store']}({map.obj.map}, {ctx(idx)})")

    def c_storemap(
        self,
        ctx: "CContext",
        map: "Stack",
        idx: "AssemblyExpression",
        value: "AssemblyExpression",
    ):
        assert isinstance(map.obj, CMapFields)
        methods: CHashMethods = ctx.datastructures[self]
        ctx.exec(f"{methods['exists']}({map.obj.map}, {ctx(idx)}, {ctx(value)})")

    def c_loadmap(self, ctx: "CContext", map: "Stack", idx: "AssemblyExpression"):
        """
        Get an expression where we can get the value corresponding to a key.

        TODO: Do we want to use pointers to tuples (standard across everything
        but requires lifecycle management)
        Or do we want to just use tuple values?

        This load is incomplete without this design decision.
        """
        assert isinstance(map.obj, CMapFields)
        methods: CHashMethods = ctx.datastructures[self]

        valuetype_c = ctx.ctype_name(c_type(self.value_type)._type_)
        value = ctx.freshen("value")
        ctx.exec(f"{ctx.feed}{valuetype_c} {value};")
        ctx.exec(f"{ctx.feed}{methods['load']}({map.obj.map}, {ctx(idx)}, &{value})")
        return value

    def c_unpack(self, ctx: "CContext", var_n: str, val: AssemblyExpression):
        """
        Unpack the map into C context.
        """
        assert val.result_format == self
        data = ctx.freshen(var_n, "data")
        # Add all the stupid header stuff from above.
        ctx.add_datastructure(
            self,
            lambda ctx: CHashTable.gen_code(ctx, self.key_type, self.value_type)
        )

        ctx.exec(f"{ctx.feed}void* {data} = {ctx(val)}->map;")
        return CMapFields(data, var_n)

    def c_repack(self, ctx: "CContext", lhs: str, obj: "CMapFields"):
        """
        Repack the map out of C context.
        """
        ctx.exec(f"{ctx.feed}{lhs}->map = {obj.map}")

    def serialize_to_c(self, obj: CHashTable):
        """
        Serialize the Hash Map to a CHashMap structure.
        This datatype will then immediately get turned into a struct.
        """
        assert isinstance(obj, CHashTable)
        map = ctypes.c_void_p(obj.map)
        struct = CHashMapStruct(map, obj)
        return ctypes.pointer(struct)

    def deserialize_from_c(self, obj: CHashTable, res):
        """
        Update this buffer based on how the C call modified the CNumpyBuffer structure.
        """
        assert isinstance(res, ctypes.POINTER(CHashMapStruct))
        assert isinstance(res.contents.obj, CHashTable)
        obj.map = res.contents.map

    def construct_from_c(self, c_map):
        """
        Construct a NumpyBuffer from a C-compatible structure.

        c_map is a pointer to a CHashMapStruct
        """
        raise NotImplementedError


if __name__ == "__main__":
    table = CHashTable(2, 3, {})
    table.store((2, 3), (3, 2, 3))
    print(table.exists((2, 3)))
    print(table.load((2, 3)))
    print(table.exists((2, 1)))
