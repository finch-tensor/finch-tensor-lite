import ctypes
import logging
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Hashable
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from finchlite import algebra
from finchlite import finch_assembly as asm
from finchlite.algebra import (
    COperator,
    FType,
    ImmutableStructFType,
    MutableStructFType,
    NamedTupleFType,
    StructFType,
    TupleFType,
    fisinstance,
    ftype,
    query_property,
    register_property,
)
from finchlite.algebra.algebra import FinchOperator
from finchlite.symbolic import Context, Namespace
from finchlite.util import config, file_cache
from finchlite.util.logging import LOG_BACKEND_C

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_BACKEND_C)

common_h = Path(__file__).parent / "stc" / "include" / "stc" / "common.h"


@file_cache(ext=config.get("shared_library_suffix"), domain="c")
def create_shared_lib(filename, c_code, cc, cflags):
    """
    Compiles a C function into a shared library and returns the path.

    :param c_code: The C code as a string.
    :return: The result of the function call.
    """
    tmp_dir = Path(config.get("data_path")) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Create a temporary directory to store the C file and shared library
    with tempfile.TemporaryDirectory(prefix=str(tmp_dir)) as staging_dir:
        staging_dir = Path(staging_dir)
        c_file_path = staging_dir / "temp.c"
        shared_lib_path = Path(filename)

        # Write the C code to a file
        c_file_path.write_text(c_code)

        # Compile the C code into a shared library
        compile_command = [
            str(cc),
            *cflags,
            "-o",
            str(shared_lib_path),
            str(c_file_path),
        ]
        if not shutil.which(cc):
            raise FileNotFoundError(
                f"Compiler '{cc}' not found. Ensure it is installed and in your PATH."
            )
        try:
            subprocess.run(compile_command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                "Compilation failed with command:\n"
                f"    {compile_command}\n"
                f"on the following code:\n{c_code}"
                f"\nError message: {e}"
            )
            raise RuntimeError("C Compilation failed") from e
        assert shared_lib_path.exists(), f"Compilation failed: {compile_command}"


@lru_cache(maxsize=10_000)
def load_shared_lib(c_code, cc=None, cflags=None):
    """
    :param function_name: The name of the function to call.
    :param c_code: The code to compile
    """
    if cc is None:
        cc = config.get("cc")
    if cflags is None:
        cflags = (
            *config.get("cflags").split(),
            *config.get("shared_cflags").split(),
        )

    shared_lib_path = create_shared_lib(
        c_code,
        cc,
        cflags,
    )

    # Load the shared library using ctypes
    return ctypes.CDLL(str(shared_lib_path))


def c_hash(fmt: FType, ctx: "CContext"):
    """
    Expand to the name of a macro that c hash can use for hashing fmt.

    Args:
        ctx: CContext object
        var_n: name to be supplied. It is a placeholder for a variable with
        type fmt* (so indirection)
    """
    if hasattr(fmt, "c_hash"):
        return fmt.c_hash(ctx)
    return query_property(fmt, "c_hash", "__attr__", ctx)


def c_hash_default(fmt: FType, ctx: "CContext"):
    ctx.add_header(f'#include "{common_h}"')
    return "c_default_hash"


def c_eq(fmt: FType, ctx: "CContext"):
    """
    Expand to the name of a macro that c eq can use for checking equivalence of fmt.

    Args:
        ctx: CContext object
        var_n: name to be supplied. It is a placeholder for a variable with
        type fmt* (so indirection)
    """
    if hasattr(fmt, "c_eq"):
        return fmt.c_eq(ctx)
    return query_property(fmt, "c_eq", "__attr__", ctx)


def c_eq_default(fmt: FType, ctx: "CContext"):
    ctx.add_header(f'#include "{common_h}"')
    return "c_default_eq"


def serialize_to_c(fmt: FType, obj):
    """
    Serialize an object to a C-compatible ftype.

    Args:
        fmt: FType of obj
        obj: The object to serialize.

    Returns:
        A ctypes-compatible struct.
    """
    if hasattr(fmt, "serialize_to_c"):
        return fmt.serialize_to_c(obj)
    return query_property(fmt, "serialize_to_c", "__attr__", obj)


def deserialize_from_c(fmt: FType, obj, c_obj):
    """
    Deserialize a C-compatible object back to the original ftype.

    Args:
        fmt: FType of obj
        obj: The original object to update.
        c_obj: The C-compatible object to deserialize from.

    Returns:
        None
    """
    if hasattr(fmt, "deserialize_from_c"):
        fmt.deserialize_from_c(obj, c_obj)
    else:
        try:
            query_property(fmt, "deserialize_from_c", "__attr__", obj, c_obj)
        except AttributeError:
            return


def construct_from_c(fmt: FType, c_obj):
    """
    Construct an object from a C-compatible ftype.

    Args:
        fmt: The ftype of the object.
        c_obj: The C-compatible object to construct from.

    Returns:
        An instance of the original object type.
    """
    if hasattr(fmt, "construct_from_c"):
        return fmt.construct_from_c(c_obj)
    try:
        return query_property(fmt, "construct_from_c", "__attr__", c_obj)
    except AttributeError:
        return fmt(c_obj)


register_property(
    algebra.ftypes.FDTypeNumpy,
    "serialize_to_c",
    "__attr__",
    lambda fmt, obj: np.ctypeslib.as_ctypes(np.array(obj, dtype=fmt.dtype)),
)

register_property(
    algebra.ftypes.FDTypeNumpy,
    "c_hash",
    "__attr__",
    c_hash_default,
)

register_property(
    algebra.ftypes.FDTypeNumpy,
    "c_eq",
    "__attr__",
    c_eq_default,
)

# pass by value -> no op
register_property(
    algebra.ftypes.FDTypeNumpy,
    "deserialize_from_c",
    "__attr__",
    lambda fmt, obj, c_value: None,
)

register_property(
    algebra.ftypes.FDTypeNumpy,
    "construct_from_c",
    "__attr__",
    lambda fmt, c_value: fmt(c_value.value),
)

# deserialize_to_c should modify in place. TODO: implement

for typ in (algebra.int_, algebra.float_):
    register_property(
        typ,
        "c_hash",
        "__attr__",
        c_hash_default,
    )
    register_property(
        typ,
        "c_eq",
        "__attr__",
        c_eq_default,
    )


def c_function_name(op: FinchOperator, ctx, *args: Any) -> str:
    """Returns the C function name corresponding to the given Python function
    and argument types.

    Args:
        op: The Python function or operator.
        ctx: The context in which the function will be called.
        *args: The argument types.

    Returns:
        The C function name as a string.

    Raises:
        NotImplementedError: If the C function name is not implemented for the
        given function and types.
    """
    if isinstance(op, COperator):
        return op.c_symbol
    raise TypeError(f"{op} has no C representation.")


def c_function_call(op: FinchOperator, ctx, *args: Any) -> str:
    """Returns a call to the C function corresponding to the given Python
    function and argument types.

    Args:
        op: The Python function or operator.
        ctx: The context in which the function will be called.
        *args: The argument types.

    Returns:
        The C function call as a string.
    """
    if not isinstance(op, COperator):
        raise TypeError(f"{op} has no C representation.")
    return op.c_function_call(ctx, *args)


def c_getattr(fmt, ctx, obj, attr):
    if hasattr(fmt, "c_getattr"):
        return fmt.c_getattr(ctx, obj, attr)
    return query_property(fmt, "c_getattr", "__attr__", ctx, obj, attr)


def c_setattr(fmt, ctx, obj, attr, val):
    if hasattr(fmt, "c_setattr"):
        return fmt.c_setattr(ctx, obj, attr, val)
    return query_property(fmt, "c_setattr", "__attr__", ctx, obj, attr, val)


def c_literal(ctx, val):
    """
    Returns the C literal corresponding to the given Python value.

    Args:
        ctx: The context in which the value is used.
        val: The Python value.

    Returns:
        The C literal as a string.
    """
    if hasattr(val, "c_literal"):
        return val.c_literal(ctx)
    try:
        fmt = ftype(val)
    except NotImplementedError:
        return query_property(val, "c_literal", "__attr__", ctx)
    return query_property(fmt, "c_literal", "__attr__", val, ctx)


register_property(algebra.int_, "c_literal", "__attr__", lambda fmt, x, ctx: str(x))
register_property(algebra.float_, "c_literal", "__attr__", lambda fmt, x, ctx: str(x))
register_property(algebra.str_, "c_literal", "__attr__", lambda fmt, x, ctx: f'"{x}"')


def numpy_c_literal(fmt, x, ctx):
    if isinstance(x, np.bool_):
        value = "true" if x else "false"
    else:
        value = str(x.item())
    return f"({ctx.ctype_name(c_type(fmt))}){value}"


register_property(
    algebra.ftypes.FDTypeNumpy,
    "c_literal",
    "__attr__",
    numpy_c_literal,
)


def c_type(t: FType):
    """
    Returns the C type corresponding to the given Finch type.

    Args:
        ctx: The context in which the value is used.
        t: The Finch type.

    Returns:
        The corresponding C type as a ctypes type.
    """
    if t is algebra.none_:
        return None
    if hasattr(t, "c_type"):
        return t.c_type()
    return query_property(t, "c_type", "__attr__")


register_property(algebra.int_, "c_type", "__attr__", lambda x: ctypes.c_int)
register_property(algebra.float_, "c_type", "__attr__", lambda x: ctypes.c_double)
register_property(algebra.bool_, "c_type", "__attr__", lambda x: ctypes.c_bool)
register_property(algebra.str_, "c_type", "__attr__", lambda x: ctypes.c_wchar_p)
register_property(algebra.none_, "c_type", "__attr__", lambda x: None)
register_property(
    algebra.ftypes.FDTypeNumpy,
    "c_type",
    "__attr__",
    lambda x: np.ctypeslib.as_ctypes_type(x.dtype),
)

# ints and floats should be serialized and constructed trivially.
register_property(
    algebra.int_, "serialize_to_c", "__attr__", lambda fmt, x: c_type(fmt)(x)
)
register_property(
    algebra.float_, "serialize_to_c", "__attr__", lambda fmt, x: c_type(fmt)(x)
)
register_property(
    algebra.bool_, "serialize_to_c", "__attr__", lambda fmt, x: c_type(fmt)(x)
)
register_property(
    algebra.int_,
    "construct_from_c",
    "__attr__",
    lambda fmt, x: int(x.value if hasattr(x, "value") else x),
)
register_property(
    algebra.float_,
    "construct_from_c",
    "__attr__",
    lambda fmt, x: float(x.value if hasattr(x, "value") else x),
)
register_property(
    algebra.bool_,
    "construct_from_c",
    "__attr__",
    lambda fmt, x: bool(x.value if hasattr(x, "value") else x),
)
register_property(algebra.none_, "serialize_to_c", "__attr__", lambda fmt, x: None)
register_property(algebra.none_, "construct_from_c", "__attr__", lambda fmt, x: None)
ctype_to_c_name: dict[Any, tuple[str, list[str]]] = {
    ctypes.c_bool: ("bool", ["stdbool.h"]),
    ctypes.c_char: ("char", []),
    ctypes.c_wchar: ("wchar_t", ["wchar.h"]),
    ctypes.c_byte: ("char", []),
    ctypes.c_ubyte: ("unsigned char", []),
    ctypes.c_int8: ("int8_t", ["stdint.h"]),
    ctypes.c_int16: ("int16_t", ["stdint.h"]),
    ctypes.c_int32: ("int32_t", ["stdint.h"]),
    ctypes.c_int64: ("int64_t", ["stdint.h"]),
    ctypes.c_uint8: ("uint8_t", ["stdint.h"]),
    ctypes.c_uint16: ("uint16_t", ["stdint.h"]),
    ctypes.c_uint32: ("uint32_t", ["stdint.h"]),
    ctypes.c_uint64: ("uint64_t", ["stdint.h"]),
    ctypes.c_float: ("float", []),
    ctypes.c_double: ("double", []),
    ctypes.c_char_p: ("char*", []),
    ctypes.c_wchar_p: ("wchar_t*", ["wchar.h"]),
    ctypes.c_void_p: ("void*", []),
    ctypes.py_object: ("void*", []),
    None: ("void", []),
}


class CContext(Context):
    """
    A class to represent a C environment.

    The context has functionality to track which datastructure definitions need
    to get declared via the stc library.
    """

    def __init__(
        self,
        tab="    ",
        indent=0,
        headers=None,
        fptr=None,
        **kwargs,
    ):
        if headers is None:
            headers = []
        super().__init__(**kwargs)
        self.tab = tab
        self.indent = indent
        self.headers = headers
        self._headerset = set(headers)
        if fptr is None:
            fptr = {}
        self.fptr = fptr
        self.datastructures: dict[Hashable, Any] = {}

    def add_header(self, header):
        if header not in self._headerset:
            self.headers.append(header)
            self._headerset.add(header)

    def emit_global(self):
        """
        Emit the headers for the C code.
        """
        return "\n".join([*self.headers, self.emit()])

    def ctype_name(self, t: type) -> str:
        # Mapping from ctypes types to their C type names
        # mypy: ignore-errors for ctypes internals
        if t in ctype_to_c_name:
            (name, libs) = ctype_to_c_name[t]
            for lib in libs:
                self.add_header(f"#include <{lib}>")
            return name
        # The following use of ctypes internals is not type safe, so we ignore mypy
        if (
            hasattr(ctypes, "Structure")
            and isinstance(t, type)
            and issubclass(t, ctypes.Structure)
        ):  # type: ignore[attr-defined]
            name = t.__name__
            args = [
                f"{self.ctype_name(f_type)} {f_name}"
                for (f_name, f_type, *_) in t._fields_
            ]
            header = (
                f"struct {name} {{\n"
                + "\n".join(f"{self.tab}{arg};" for arg in args)
                + "\n};"
            )
            self.add_header(header)
            return f"struct {name}"
        if (
            hasattr(ctypes, "_Pointer")
            and isinstance(t, type)
            and issubclass(t, ctypes._Pointer)
        ):  # type: ignore[attr-defined]
            return f"{self.ctype_name(t._type_)}*"  # type: ignore[attr-defined]
        if (
            hasattr(ctypes, "_CFuncPtr")
            and isinstance(t, type)
            and issubclass(t, ctypes._CFuncPtr)
        ):  # type: ignore[attr-defined]
            arg_types = ", ".join(
                self.ctype_name(arg_type)
                for arg_type in getattr(t, "_argtypes_", [])  # type: ignore[attr-defined]
            )
            # type: ignore[arg-type]
            res_t = self.ctype_name(getattr(t, "_restype_", object))
            key = f"{res_t} (*)( {arg_types} );"
            name = self.fptr.get(key)
            if name is None:
                name = self.freshen("fptr")
                self.add_header(f"typedef {res_t} (*{name})( {arg_types} );")
                self.fptr[key] = name
            return name
        raise NotImplementedError(f"No C type mapping for {t}")

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def block(self) -> "CContext":
        blk = super().block()
        blk.indent = self.indent
        blk.tab = self.tab
        blk.headers = self.headers
        blk._headerset = self._headerset
        blk.fptr = self.fptr
        blk.datastructures = self.datastructures
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        return blk

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])


class CHashableFType(FType):
    @abstractmethod
    def c_hash(self, ctx: CContext) -> str:
        """
        Emit code from CContext that takes an expression and returns the NAME
        of a macro that performs our hashing with STC functions.

        Please reference finch_assembly/struct.py for reference.

        The macro should take one argument (type of fmt*, so there is one layer
        of indirection) and expand to an expression that returns a size_t of
        the final hash.

        The main idea for implementing this is for unpacked structs where you
        want the unused bytes to be set to zero.

        This is important to note for immutable structs because you need to do
        something like &var_n->property if you want to do recursive hashing.
        """
        ...

    @abstractmethod
    def c_eq(self, ctx: CContext) -> str:
        """
        Emit code from CContext that takes an expression and returns the NAME
        of a macro that can be used to check.

        The macro should take two arguments (each a type of fmt*, so there is
        one layer of indirection) and expand to an expression that checks
        equality.

        The main idea for implementing this is for unpacked structs where you
        want the unused bytes to be set to zero.

        This is important to note for immutable structs because you need to do
        something like &var_n->property if you want to do recursive hashing.
        """


class CArgumentFType(ABC):
    @abstractmethod
    def serialize_to_c(self, obj):
        """
        Return a ctypes-compatible struct to be used in place of `obj`
        for the c backend.
        """
        ...

    @abstractmethod
    def deserialize_from_c(self, obj, res):
        """
        Update this `obj` based on how the c call modified `res`, the result
        of `serialize_to_c`.
        """
        ...

    @abstractmethod
    def construct_from_c(self, res):
        """
        Construct a new object based on the return value from c
        """


class CDictFType(CArgumentFType, ABC):
    """
    Abstract base class for the ftype of dictionaries. The ftype defines how
    the data in a Map is organized and accessed.
    """

    @abstractmethod
    def c_existsdict(self, ctx, map_type, map_fields, idx_symbol, idx_type):
        """
        Return C code which checks whether a given key exists in a map.
        """
        ...

    @abstractmethod
    def c_loaddict(self, ctx, map_type, map_fields, idx_symbol, idx_type):
        """
        Return C code which gets a value corresponding to a certain key.
        """
        ...

    @abstractmethod
    def c_storedict(
        self, ctx, map_type, map_fields, idx_symbol, idx_type, value_symbol, value_type
    ):
        """
        Return C code which stores a certain value given a certain integer tuple key.
        """
        ...


class CBufferFType(CArgumentFType, ABC):
    """
    Abstract base class for the ftype of datastructures. The ftype defines how
    the data in an Buffer is organized and accessed.
    """

    @abstractmethod
    def c_length(self, ctx, buffer_type, buffer_fields):
        """
        Return C code which loads a named buffer at the given index.
        """
        ...

    @abstractmethod
    def c_load(self, ctx, buffer_type, buffer_fields, index_symbol, index_type):
        """
        Return C code which loads a named buffer at the given index.
        """
        ...

    @abstractmethod
    def c_store(
        self,
        ctx,
        buffer_type,
        buffer_fields,
        index_symbol,
        index_type,
        value_symbol,
        value_type,
    ):
        """
        Return C code which stores a named buffer to the given index.
        """
        ...

    @abstractmethod
    def c_resize(
        self, ctx, buffer_type, buffer_fields, new_length_symbol, new_length_type
    ):
        """
        Return C code which resizes a named buffer to the given length.
        """
        ...


class CStackFType(ABC):
    """
    Abstract base class for symbolic formats in C. Stack formats must also
    support other functions with symbolic inputs in addition to variable ones.
    """

    @abstractmethod
    def c_unpack(self, ctx, lhs_symbol, lhs_type, rhs_symbol, rhs_type):
        """
        Convert a value to a symbolic representation in C. Returns a NamedTuple
        of unpacked variable names, etc. The `lhs` is the variable namespace to
        assign to.
        """
        ...

    @abstractmethod
    def c_repack(self, ctx, lhs_symbol, lhs_type, rhs):
        """
        Update an object based on a symbolic representation. The `rhs` is the
        symbolic representation to update from, and `lhs` is a variable name referring
        to the original object to update.
        """
        ...


def serialize_struct_to_c(fmt: StructFType, obj) -> Any:
    args = [serialize_to_c(fmt, getattr(obj, name)) for name, fmt in fmt.struct_fields]
    return struct_c_type(fmt)(*args)


register_property(StructFType, "serialize_to_c", "__attr__", serialize_struct_to_c)


def deserialize_struct_from_c(fmt: StructFType, obj, c_struct: Any) -> None:
    if fmt.is_mutable:
        for name in fmt.struct_fieldnames:
            setattr(obj, name, getattr(c_struct, name))
        return


register_property(
    StructFType, "deserialize_from_c", "__attr__", deserialize_struct_from_c
)

c_structs: dict[Any, Any] = {}
c_structnames = Namespace()


def struct_c_type(fmt: StructFType):
    res = c_structs.get(fmt)
    if res:
        return res
    fields = [(name, c_type(fmt)) for name, fmt in fmt.struct_fields]
    new_struct = type(
        c_structnames.freshen("C", fmt.struct_name),
        (ctypes.Structure,),
        {"_fields_": fields},
    )
    c_structs[fmt] = new_struct
    return new_struct


"""
Note: When serializing any struct to C, it will get serialized to a struct with
no indirection.

When you pass a struct into a kernel that expects a struct pointer, ctypes can
intelligently infer whether we are working with a pointer arg type (pass by
reference) or a non-pointer type (in which case it will immediately apply
indirection)
"""

register_property(
    MutableStructFType,
    "c_type",
    "__attr__",
    lambda fmt: ctypes.POINTER(struct_c_type(fmt)),
)

register_property(
    ImmutableStructFType, "c_type", "__attr__", lambda fmt: struct_c_type(fmt)
)


register_property(
    MutableStructFType,
    "c_getattr",
    "__attr__",
    lambda fmt, ctx, obj, attr: f"{obj}->{attr}",
)

register_property(
    ImmutableStructFType,
    "c_getattr",
    "__attr__",
    lambda fmt, ctx, obj, attr: f"{obj}.{attr}",
)


def struct_mutable_setattr(fmt: StructFType, ctx, obj, attr, val):
    ctx.emit(f"{ctx.feed}{obj}->{attr} = {val};")


# the equivalent for immutable is f"{ctx.feed}{obj}.{attr} = {val};"
# but we will not include that because it's bad.

register_property(
    MutableStructFType,
    "c_setattr",
    "__attr__",
    struct_mutable_setattr,
)


def struct_construct_from_c(fmt: StructFType, c_struct):
    args = [getattr(c_struct, name) for name in fmt.struct_fieldnames]
    return fmt.__class__(*args)


register_property(
    StructFType,
    "construct_from_c",
    "__attr__",
    struct_construct_from_c,
)


def serialize_tuple_to_c(fmt, obj):
    x = namedtuple("CTuple", fmt.struct_fieldnames)(*obj)  # noqa: PYI024
    return serialize_to_c(ftype(x), x)


register_property(
    TupleFType,
    "serialize_to_c",
    "__attr__",
    serialize_tuple_to_c,
)


def tuple_construct_from_c(fmt: TupleFType, c_struct):
    args = [getattr(c_struct, name) for name in fmt.struct_fieldnames]
    return tuple(args)


register_property(
    TupleFType,
    "construct_from_c",
    "__attr__",
    tuple_construct_from_c,
)

register_property(
    TupleFType,
    "c_type",
    "__attr__",
    lambda fmt: struct_c_type(NamedTupleFType("CTuple", fmt.struct_fields)),
)

register_property(
    TupleFType,
    "c_getattr",
    "__attr__",
    lambda fmt, ctx, obj, attr: f"{obj}.{attr}",
)

register_property(
    TupleFType,
    "deserialize_from_c",
    "__attr__",
    lambda fmt, obj, c_struct: None,
)


class CHashableProperties(TypedDict):
    eq: str | None
    hash: str | None


def c_hash_struct(fmt: ImmutableStructFType, ctx: "CContext"):
    # this should be true in whatever structs we have.
    assert isinstance(fmt, Hashable) and isinstance(fmt, ImmutableStructFType)
    if fmt in ctx.datastructures:
        properties: CHashableProperties = ctx.datastructures[fmt]
        if properties.get("hash") is not None:
            return properties["hash"]
    else:
        ctx.datastructures[fmt] = {}

    macros = [c_hash(fmt2, ctx) for fmt2 in fmt.struct_fieldtypes]
    name = ctx.freshen("hash")
    ctx.datastructures[fmt]["hash"] = name

    # implement recursion with &{var_n}->{struct_field}
    var_n = ctx.freshen("var")
    args = ",".join(
        f"{macro}(&({var_n})->{field})"
        for macro, field in zip(macros, fmt.struct_fieldnames, strict=False)
    )
    ctx.add_header(f"#define {name}({var_n}) c_hash_mix({args})")
    return name


register_property(
    ImmutableStructFType,
    "c_hash",
    "__attr__",
    c_hash_struct,
)

register_property(
    TupleFType,
    "c_hash",
    "__attr__",
    c_hash_struct,
)


def c_eq_struct(fmt: ImmutableStructFType, ctx: "CContext"):
    # this should be true in whatever structs we have.
    assert isinstance(fmt, Hashable) and isinstance(fmt, ImmutableStructFType)
    if fmt in ctx.datastructures:
        properties: CHashableProperties = ctx.datastructures[fmt]
        if properties.get("eq") is not None:
            return properties["eq"]
    else:
        ctx.datastructures[fmt] = {}

    macros = [c_eq(fmt, ctx) for fmt in fmt.struct_fieldtypes]
    name = ctx.freshen("eq")
    ctx.datastructures[fmt]["eq"] = name

    # implement recursion with &{var_n}->{struct_field}
    var1_n = ctx.freshen("var")
    var2_n = ctx.freshen("var")
    args = " && ".join(
        f"{macro}(&({var1_n})->{field}, &({var2_n})->{field})"
        for macro, field in zip(macros, fmt.struct_fieldnames, strict=False)
    )
    ctx.add_header(f"#define {name}({var1_n}, {var2_n}) ({args})")
    return name


register_property(
    ImmutableStructFType,
    "c_eq",
    "__attr__",
    c_eq_struct,
)

register_property(
    TupleFType,
    "c_eq",
    "__attr__",
    c_eq_struct,
)
