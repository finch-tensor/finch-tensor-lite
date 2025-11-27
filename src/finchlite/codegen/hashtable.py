import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypedDict

import numba
import numpy as np

from finchlite.codegen.c import (
    CContext,
    CMapFType,
    CStackFType,
    c_type,
    construct_from_c,
    load_shared_lib,
    serialize_to_c,
)
from finchlite.codegen.numba_backend import NumbaContext, NumbaMapFType, NumbaStackFType, numba_type
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


def _int_tuple_ftype(size: int):
    return TupleFType.from_tuple(tuple(int for _ in range(size)))


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
        inline: bool = False,
    ) -> tuple[CHashMethods, str]:

        assert isinstance(key_type, TupleFType)
        assert isinstance(value_type, TupleFType)

        key_len = len(key_type.struct_fields)
        value_len = len(value_type.struct_fields)
        # dereference both key and value types; as given, they are both pointers.
        keytype_c = ctx.ctype_name(c_type(key_type))
        valuetype_c = ctx.ctype_name(c_type(value_type))
        hmap_t = ctx.freshen(f"hmap", key_len, value_len)

        ctx.add_header("#include <stdlib.h>")

        # these headers should just be added to the headers list.
        # deduplication is catastrohpic here.
        ctx.headers.append(f"#define T {hmap_t}, {keytype_c}, {valuetype_c}")
        ctx.headers.append("#define i_eq c_memcmp_eq")
        ctx.headers.append(f'#include "{hashmap_h}"')

        methods: CHashMethods = {
            "init": ctx.freshen("finch_hmap_init", key_len, value_len),
            "exists": ctx.freshen("finch_hmap_exists", key_len, value_len),
            "load": ctx.freshen("finch_hmap_load", key_len, value_len),
            "store": ctx.freshen("finch_hmap_store", key_len, value_len),
            "cleanup": ctx.freshen("finch_hmap_cleanup", key_len, value_len),
        }
        # register these methods in the datastructures.
        ctx.datastructures[CHashTableFType(key_len, value_len)] = methods
        inline_s = "static inline " if inline else ""

        # basically for the load functions, you need to provide a variable that
        # can be copied.
        # Yeah, so which API's should we use for load and store?
        lib_code = f"""
{inline_s}void* {methods['init']}() {{
    void* ptr = malloc(sizeof({hmap_t}));
    memset(ptr, 0, sizeof({hmap_t}));
    return ptr;
}}
{inline_s}bool {methods['exists']}({hmap_t} *map, {keytype_c} key) {{
    return {hmap_t}_contains(map, key);
}}
{inline_s}{valuetype_c} {methods['load']}({hmap_t} *map, {keytype_c} key) {{
    const {valuetype_c}* internal_val = {hmap_t}_at(map, key);
    return *internal_val;
}}
{inline_s}void {methods['store']}({hmap_t} *map, {keytype_c} key, {valuetype_c} value) {{
    {hmap_t}_insert_or_assign(map, key, value);
}}
{inline_s}void {methods['cleanup']}(void* ptr) {{
    {hmap_t}* hptr = ptr;
    {hmap_t}_drop(hptr);
    free(hptr);
}}
        """
        ctx.add_header(lib_code)

        return methods, hmap_t

    @classmethod
    def compile(cls, key_len: int, value_len: int) -> CHashTableLibrary:
        """
        compile a library to use for the c hash table.
        """
        if (key_len, value_len) in cls.libraries:
            return cls.libraries[(key_len, value_len)]
        key_type = _int_tuple_ftype(key_len)
        value_type = _int_tuple_ftype(value_len)

        ctx = CContext()
        methods, hmap_t = cls.gen_code(ctx, key_type, value_type)
        code = ctx.emit_global()
        lib = load_shared_lib(code)

        # get keystruct and value types
        KeyStruct = c_type(key_type)
        ValueStruct = c_type(value_type)

        init_func = getattr(lib, methods["init"])
        init_func.argtypes = []
        init_func.restype = ctypes.c_void_p

        # Exists: Takes (map*, key) -> returns bool
        exists_func = getattr(lib, methods["exists"])
        exists_func.argtypes = [ctypes.c_void_p, KeyStruct]
        exists_func.restype = ctypes.c_bool

        # Load: Takes (map*, key) -> returns value
        load_func = getattr(lib, methods["load"])
        load_func.argtypes = [
            ctypes.c_void_p,
            KeyStruct,
        ]
        load_func.restype = ValueStruct

        # Store: Takes (map*, key, val) -> returns void
        store_func = getattr(lib, methods["store"])
        store_func.argtypes = [
            ctypes.c_void_p,
            KeyStruct,
            ValueStruct,
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
        KeyStruct = c_type(self.ftype.key_type)
        c_key = KeyStruct(*idx)
        func = getattr(self.lib.library, self.lib.methods["exists"])
        return func(self.map, c_key)

    def load(self, idx):
        assert _is_integer_tuple(idx, self.key_len)
        KeyStruct = c_type(self.ftype.key_type)
        c_key = KeyStruct(*idx)
        c_value = getattr(self.lib.library, self.lib.methods["load"])(self.map, c_key)
        return tuple(
            getattr(c_value, f) for f in self.ftype.value_type.struct_fieldnames
        )

    def store(self, idx, val):
        assert _is_integer_tuple(idx, self.key_len)
        assert _is_integer_tuple(val, self.value_len)
        KeyStruct = c_type(self.ftype.key_type)
        ValueStruct = c_type(self.ftype.value_type)
        c_key = KeyStruct(*idx)
        c_value = ValueStruct(*val)
        getattr(self.lib.library, self.lib.methods["store"])(self.map, c_key, c_value)

    def __str__(self):
        return f"c_hashtable({self.map})"

    @property
    def ftype(self):
        return CHashTableFType(self.key_len, self.value_len)


class CHashTableFType(CMapFType, CStackFType):
    """
    An implementation of Hash Tables using the stc library.
    """

    def __init__(self, key_len: int, value_len: int):
        self.key_len = key_len
        self.value_len = value_len
        self._key_type = _int_tuple_ftype(key_len)
        self._value_type = _int_tuple_ftype(value_len)

    def __eq__(self, other):
        if not isinstance(other, CHashTableFType):
            return False
        return self.key_len == other.key_len and self.value_len == other.value_len

    def __call__(self):
        return CHashTable(self.key_len, self.value_len, {})

    def __str__(self):
        return f"chashtable_t({self.key_len}, {self.value_len})"

    def __repr__(self):
        return f"CHashTableFType({self.key_len}, {self.value_len})"

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
        return hash(("CHashTableFType", self.key_len, self.value_len))

    """
    Methods for the C Backend
    This requires an external library (stc) to work.
    """

    def c_type(self):
        return ctypes.POINTER(CHashMapStruct)

    def c_existsmap(self, ctx: "CContext", map: "Stack", idx: "AssemblyExpression"):
        assert isinstance(map.obj, CMapFields)
        methods: CHashMethods = ctx.datastructures[self]
        return f"{ctx.feed}{methods['exists']}({map.obj.map}, {ctx(idx)})"

    def c_storemap(
        self,
        ctx: "CContext",
        map: "Stack",
        idx: "AssemblyExpression",
        value: "AssemblyExpression",
    ):
        assert isinstance(map.obj, CMapFields)
        methods: CHashMethods = ctx.datastructures[self]
        ctx.exec(f"{ctx.feed}{methods['store']}({map.obj.map}, {ctx(idx)}, {ctx(value)});")

    def c_loadmap(self, ctx: "CContext", map: "Stack", idx: "AssemblyExpression"):
        """
        Get an expression where we can get the value corresponding to a key.
        """
        assert isinstance(map.obj, CMapFields)
        methods: CHashMethods = ctx.datastructures[self]

        return f"{methods['load']}({map.obj.map}, {ctx(idx)})"

    def c_unpack(self, ctx: "CContext", var_n: str, val: AssemblyExpression):
        """
        Unpack the map into C context.
        """
        assert val.result_format == self
        data = ctx.freshen(var_n, "data")
        # Add all the stupid header stuff from above.
        ctx.add_datastructure(
            self,
            lambda ctx: CHashTable.gen_code(
                ctx, self.key_type, self.value_type, inline=True
            ),
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
        # We NEED this for stupid ownership reasons.
        obj._self_obj = ctypes.py_object(obj)  # type: ignore
        obj._struct = struct  # type: ignore
        return ctypes.pointer(struct)

    def deserialize_from_c(self, obj: CHashTable, res):
        """
        Update our hash table based on how the C call modified the CHashMapStruct.
        """
        assert isinstance(res, ctypes.POINTER(CHashMapStruct))
        assert isinstance(res.contents.obj, CHashTable)

        obj.map = res.contents.map

    def construct_from_c(self, c_map):
        """
        Construct a CHashTable from a C-compatible structure.

        c_map is a pointer to a CHashMapStruct

        I am going to refrain from doing this because lifecycle management is horrible.
        Should we move?
        """
        raise NotImplementedError


class NumbaHashTable(Map):
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
        return NumbaHashTableFType(self.key_len, self.value_len)

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
        return f"numba_hashtable({self.map})"


class NumbaHashTableFType(NumbaMapFType, NumbaStackFType):
    """
    An implementation of Hash Tables using the stc library.
    """

    def __init__(self, key_len: int, value_len: int):
        self.key_len = key_len
        self.value_len = value_len
        self._key_type = _int_tuple_ftype(key_len)
        self._value_type = _int_tuple_ftype(value_len)

    def __eq__(self, other):
        if not isinstance(other, NumbaHashTableFType):
            return False
        return self.key_len == other.key_len and self.value_len == other.value_len

    def __call__(self):
        return NumbaHashTable(self.key_len, self.value_len, {})

    def __str__(self):
        return f"numba_hashtable_t({self.key_len}, {self.value_len})"

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
        return hash(("NumbaHashTableFType", self.key_len, self.value_len))

    """
    Methods for the Numba Backend
    """

    def numba_jitclass_type(self) -> numba.types.Type:
        key_t = numba.types.UniTuple(numba.types.int64, self.key_len)
        value_t = numba.types.UniTuple(numba.types.int64, self.value_len)
        return numba.types.ListType(numba.types.DictType(key_t, value_t))

    def numba_type(self):
        return self.numba_jitclass_type()
        # return list[dict[key_t, val_t]]

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
        ctx.exec(f"{ctx.feed}{map.obj.map}[{ctx(idx)}] = {ctx(value)}")

    def numba_unpack(
        self, ctx: "NumbaContext", var_n: str, val: "AssemblyExpression"
    ) -> NumbaMapFields:
        """
        Unpack the map into numba context.
        """
        # the val field will always be asm.Variable(var_n, var_t)
        map = ctx.freshen(var_n, "map")
        ctx.exec(f"{ctx.feed}{map} = {ctx(val)}[0]")

        return NumbaMapFields(map, var_n)

    def numba_repack(self, ctx: "NumbaContext", lhs: str, obj: "NumbaMapFields"):
        """
        Repack the map from Numba context.
        """
        # obj is the fields corresponding to the self.slots[lhs]
        ctx.exec(f"{ctx.feed}{lhs}[0] = {obj.map}")

    def serialize_to_numba(self, obj: "NumbaHashTable"):
        """
        Serialize the hashmap to a Numba-compatible object.

        We will supply the input and output length
        """
        return numba.typed.List([obj.map])

    def deserialize_from_numba(self, obj: "NumbaHashTable", numba_map: "list[dict]"):
        obj.map = numba_map[0]

    def construct_from_numba(self, numba_map):
        """
        Construct a numba map from a Numba-compatible object.
        """
        return NumbaHashTable(self.key_len, self.value_len, numba_map[0])


if __name__ == "__main__":
    table = CHashTable(2, 3, {(1, 2): (1, 4, 3)})
    table.store((2, 3), (3, 2, 3))
    print(table.exists((2, 3)))
    print(table.load((2, 3)))
    print(table.exists((2, 1)))
