import ctypes
import logging
import operator
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache
from pathlib import Path
from types import NoneType
from typing import Any

import numpy as np

from .. import finch_assembly as asm
from ..algebra import query_property, register_property
from ..finch_assembly import AssemblyStructFType, BufferFType, TupleFType
from ..symbolic import Context, Namespace, ScopedDict, fisinstance, ftype
from ..util import config
from ..util.cache import file_cache

logger = logging.getLogger(__name__)


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


def serialize_to_c(fmt, obj):
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


def deserialize_from_c(fmt, obj, c_obj):
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
        query_property(fmt, "deserialize_from_c", "__attr__", obj, c_obj)


def construct_from_c(fmt, c_obj):
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
        query_property(fmt, "construct_from_c", "__attr__", c_obj)
    except NotImplementedError:
        return fmt(c_obj)


register_property(
    tuple,
    "serialize_to_c",
    "__attr__",
    lambda c_obj: None,
)


register_property(
    NoneType,
    "construct_from_c",
    "__attr__",
    lambda c_obj: None,
)

for t in (
    ctypes.c_bool,
    ctypes.c_char,
    ctypes.c_wchar,
    ctypes.c_byte,
    ctypes.c_ubyte,
    ctypes.c_short,
    ctypes.c_ushort,
    ctypes.c_int,
    ctypes.c_int8,
    ctypes.c_int16,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.c_uint,
    ctypes.c_uint8,
    ctypes.c_uint16,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_long,
    ctypes.c_ulong,
    ctypes.c_longlong,
    ctypes.c_ulonglong,
    ctypes.c_size_t,
    ctypes.c_ssize_t,
    ctypes.c_float,
    ctypes.c_double,
    ctypes.c_wchar_p,
):
    register_property(
        t,
        "serialize_to_c",
        "__attr__",
        lambda obj: obj,
    )


class CKernel:
    """
    A class to represent a C kernel.
    """

    def __init__(self, c_function, ret_type, argtypes):
        self.c_function = c_function
        self.ret_type = ret_type
        self.argtypes = argtypes
        self.c_function.restype = c_type(ret_type)
        self.c_function.argtypes = tuple(c_type(argtype) for argtype in argtypes)

    def __call__(self, *args):
        """
        Calls the C function with the given arguments.
        """
        if len(args) != len(self.argtypes):
            raise ValueError(
                f"Expected {len(self.argtypes)} arguments, got {len(args)}"
            )
        for argtype, arg in zip(self.argtypes, args, strict=False):
            if not fisinstance(arg, argtype):
                raise TypeError(f"Expected argument of type {argtype}, got {type(arg)}")
        serial_args = list(map(serialize_to_c, self.argtypes, args))
        res = self.c_function(*serial_args)
        for type_, arg, serial_arg in zip(
            self.argtypes, args, serial_args, strict=False
        ):
            deserialize_from_c(type_, arg, serial_arg)
        if hasattr(self.ret_type, "construct_from_c"):
            return construct_from_c(res.ftype, res)
        if self.ret_type is type(None):
            return None
        return self.ret_type(res)


class CModule:
    """
    A class to represent a C module.
    """

    def __init__(self, c_module, kernels):
        self.c_module = c_module
        self.kernels = kernels

    def __getattr__(self, name):
        # Allow attribute access to kernels by name
        if name in self.kernels:
            return self.kernels[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class CCompiler:
    """
    A class to compile and run FinchAssembly.
    """

    def __init__(self, ctx=None, cc=None, cflags=None, shared_cflags=None):
        if cc is None:
            cc = config.get("cc")
        if cflags is None:
            cflags = config.get("cflags").split()
        if shared_cflags is None:
            shared_cflags = config.get("shared_cflags").split()
        self.cc = cc
        self.cflags = cflags
        self.shared_cflags = shared_cflags
        self.ctx = CGenerator() if ctx is None else ctx

    def __call__(self, prgm):
        ctx = CContext()
        ctx(prgm)
        c_code = ctx.emit_global()
        logger.info(f"Compiling C code:\n{c_code}")
        lib = load_shared_lib(
            c_code=c_code,
            cc=self.cc,
            cflags=(*self.cflags, *self.shared_cflags),
        )
        kernels = {}
        if prgm.head() != asm.Module:
            raise ValueError(
                "CCompiler expects a Module as the head of the program, "
                f"got {type(prgm.head())}"
            )
        for func in prgm.funcs:
            match func:
                case asm.Function(asm.Variable(func_name, return_t), args, _):
                    return_t = c_type(return_t)
                    arg_ts = [arg.result_format for arg in args]
                    kern = CKernel(getattr(lib, func_name), return_t, arg_ts)
                    kernels[func_name] = kern
                case _:
                    raise NotImplementedError(
                        f"Unrecognized function type: {type(func)}"
                    )
        return CModule(lib, kernels)


def c_function_name(op: Any, ctx, *args: Any) -> str:
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
    return query_property(op, "__call__", "c_function_name", ctx, *args)


def c_function_call(op: Any, ctx, *args: Any) -> str:
    """Returns a call to the C function corresponding to the given Python
    function and argument types.

    Args:
        op: The Python function or operator.
        ctx: The context in which the function will be called.
        *args: The argument types.

    Returns:
        The C function call as a string.
    """
    if hasattr(op, "c_function_call"):
        return op.c_function_call(ctx, *args)
    try:
        return query_property(op, "__call__", "c_function_call", ctx, *args)
    except NotImplementedError:
        return f"{c_function_name(op, ctx, *args)}({', '.join(map(ctx, args))})"


def c_getattr(fmt, ctx, obj, attr):
    if hasattr(fmt, "c_getattr"):
        return fmt.c_getattr(ctx, obj, attr)
    return query_property(fmt, "c_getattr", "__attr__", ctx, obj, attr)


def c_setattr(fmt, ctx, obj, attr, val):
    if hasattr(fmt, "c_setattr"):
        return fmt.c_setattr(ctx, obj, attr, val)
    return query_property(fmt, "c_setattr", "__attr__", ctx, obj, attr, val)


def register_n_ary_c_op_call(op, symbol):
    def property_func(op, ctx, *args):
        assert len(args) > 0
        if len(args) == 1:
            return f"{symbol}{ctx(args[0])}"
        return f" {symbol} ".join(map(ctx, args))

    return property_func


op: Any
symbol: str


for op, symbol in [
    (operator.add, "+"),
    (operator.sub, "-"),
    (operator.mul, "*"),
    (operator.and_, "&"),
    (operator.or_, "|"),
    (operator.xor, "^"),
]:
    register_property(
        op, "__call__", "c_function_call", register_n_ary_c_op_call(op, symbol)
    )


def register_binary_c_op_call(op, symbol):
    def property_func(op, ctx, a, b):
        return f"{ctx(a)} {symbol} {ctx(b)}"

    return property_func


for op, symbol in [
    (operator.eq, "=="),
    (operator.ne, "!="),
    (operator.lt, "<"),
    (operator.le, "<="),
    (operator.gt, ">"),
    (operator.ge, ">="),
    (operator.lshift, "<<"),
    (operator.rshift, ">>"),
    (operator.floordiv, "/"),
    (operator.truediv, "/"),
    (operator.mod, "%"),
    (operator.pow, "**"),
]:
    register_property(
        op, "__call__", "c_function_call", register_binary_c_op_call(op, symbol)
    )


def register_unary_c_op_call(op, symbol):
    def property_func(op, ctx, a):
        return f"{symbol}{ctx(a)}"

    return property_func


for op, symbol in [
    (operator.not_, "!"),
    (operator.invert, "~"),
]:
    register_property(
        op, "__call__", "c_function_call", register_unary_c_op_call(op, symbol)
    )


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
    return query_property(val, "c_literal", "__attr__", ctx)


register_property(int, "c_literal", "__attr__", lambda x, ctx: str(x))
register_property(float, "c_literal", "__attr__", lambda x, ctx: str(x))
register_property(str, "c_literal", "__attr__", lambda x, ctx: f'"{x}"')
register_property(
    np.generic,
    "c_literal",
    "__attr__",
    lambda x, ctx: c_literal(ctx, np.ctypeslib.as_ctypes_type(type(x))(x)),
)
for t in (
    ctypes.c_bool,
    ctypes.c_uint8,
    ctypes.c_uint16,
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.c_int8,
    ctypes.c_int16,
    ctypes.c_int32,
    ctypes.c_int64,
):
    register_property(
        t,
        "c_literal",
        "__attr__",
        lambda x, ctx: f"({ctx.ctype_name(type(x))}){x.value}",
    )

for t in (ctypes.c_float, ctypes.c_double, ctypes.c_longdouble):  # type: ignore[assignment]
    register_property(
        t,
        "c_literal",
        "__attr__",
        lambda x, ctx: f"({ctx.ctype_name(type(x))}){x.value}",
    )


def c_type(t):
    """
    Returns the C type corresponding to the given Python type.

    Args:
        ctx: The context in which the value is used.
        t: The Python type.

    Returns:
        The corresponding C type as a ctypes type.
    """
    if hasattr(t, "c_type"):
        return t.c_type()
    return query_property(t, "c_type", "__attr__")


register_property(int, "c_type", "__attr__", lambda x: ctypes.c_int)
register_property(float, "c_type", "__attr__", lambda x: ctypes.c_double)
register_property(str, "c_type", "__attr__", lambda x: ctypes.c_wchar_p)
register_property(
    np.generic, "c_type", "__attr__", lambda x: np.ctypeslib.as_ctypes_type(x)
)
register_property(ctypes._SimpleCData, "c_type", "__attr__", lambda x: x)
register_property(type(None), "c_type", "__attr__", lambda x: None)


ctype_to_c_name: dict[Any, tuple[str, list[str]]] = {
    ctypes.c_bool: ("bool", ["stdbool.h"]),
    ctypes.c_char: ("char", []),
    ctypes.c_wchar: ("wchar_t", ["wchar.h"]),
    ctypes.c_byte: ("char", []),
    ctypes.c_ubyte: ("unsigned char", []),
    ctypes.c_short: ("short", []),
    ctypes.c_ushort: ("unsigned short", []),
    ctypes.c_int: ("int", []),
    ctypes.c_int8: ("int8_t", ["stdint.h"]),
    ctypes.c_int16: ("int16_t", ["stdint.h"]),
    ctypes.c_int32: ("int32_t", ["stdint.h"]),
    ctypes.c_int64: ("int64_t", ["stdint.h"]),
    ctypes.c_uint: ("unsigned int", []),
    ctypes.c_uint8: ("uint8_t", ["stdint.h"]),
    ctypes.c_uint16: ("uint16_t", ["stdint.h"]),
    ctypes.c_uint32: ("uint32_t", ["stdint.h"]),
    ctypes.c_uint64: ("uint64_t", ["stdint.h"]),
    ctypes.c_long: ("long", []),
    ctypes.c_ulong: ("unsigned long", []),
    ctypes.c_longlong: ("long long", []),
    ctypes.c_ulonglong: ("unsigned long long", []),
    ctypes.c_size_t: ("size_t", ["stddef.h"]),
    ctypes.c_ssize_t: ("ssize_t", ["unistd.h"]),
    ctypes.c_float: ("float", []),
    ctypes.c_double: ("double", []),
    ctypes.c_char_p: ("char*", []),
    ctypes.c_wchar_p: ("wchar_t*", ["wchar.h"]),
    ctypes.c_void_p: ("void*", []),
    ctypes.py_object: ("void*", []),
}


class CGenerator:
    def __call__(self, prgm: asm.AssemblyNode):
        ctx = CContext()
        ctx(prgm)
        return ctx.emit_global()


class CContext(Context):
    """
    A class to represent a C environment.
    """

    def __init__(
        self,
        tab="    ",
        indent=0,
        headers=None,
        types=None,
        slots=None,
        fptr=None,
        **kwargs,
    ):
        if headers is None:
            headers = []
        if types is None:
            types = ScopedDict()
        if slots is None:
            slots = ScopedDict()
        super().__init__(**kwargs)
        self.tab = tab
        self.indent = indent
        self.headers = headers
        self._headerset = set(headers)
        if fptr is None:
            fptr = {}
        self.fptr = fptr
        self.types = types
        self.slots = slots

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
        blk.types = self.types
        blk.slots = self.slots
        blk.fptr = self.fptr
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        blk.types = self.types.scope()
        blk.slots = self.slots.scope()
        return blk

    def resolve(self, node):
        match node:
            case asm.Slot(var_n, var_t):
                if var_n in self.slots:
                    var_o = self.slots[var_n]
                    return asm.Stack(var_o, var_t)
                raise KeyError(f"Slot {var_n} not found in context")
            case asm.Stack(_, _):
                return node
            case _:
                raise ValueError(f"Expected Slot or Stack, got: {type(node)}")

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def cache(self, name, val):
        if isinstance(val, asm.Literal | asm.Variable | asm.Stack):
            return val
        var_n = self.freshen(name)
        var_t = val.result_format
        var_t_code = self.ctype_name(c_type(var_t))
        self.exec(f"{self.feed}{var_t_code} {var_n} = {self(val)};")
        return asm.Variable(var_n, var_t)

    def __call__(self, prgm: asm.AssemblyNode):
        feed = self.feed
        """
        lower the program to C code.
        """
        match prgm:
            case asm.Literal(value):
                # in the future, would be nice to be able to pass in constants that
                # are more complex than C literals, maybe as globals.
                return c_literal(self, value)
            case asm.Variable(name, t):
                return name
            case asm.Assign(asm.Variable(var_n, var_t), val):
                val_code = self(val)
                if val.result_format != var_t:
                    raise TypeError(f"Type mismatch: {val.result_format} != {var_t}")
                if var_n in self.types:
                    assert var_t == self.types[var_n]
                    self.exec(f"{feed}{var_n} = {val_code};")
                else:
                    self.types[var_n] = var_t
                    var_t_code = self.ctype_name(c_type(var_t))
                    self.exec(f"{feed}{var_t_code} {var_n} = {val_code};")
                return None
            case asm.GetAttr(obj, attr):
                if not obj.result_format.struct_hasattr(attr.val):
                    raise ValueError("trying to get missing attr")
                return c_getattr(obj.result_format, self, self(obj), attr.val)
            case asm.SetAttr(obj, attr, val):
                obj = self.cache("obj", obj)
                if not fisinstance(val, obj.result_format.struct_attrtype(attr.val)):
                    raise TypeError(
                        f"Type mismatch: {val.result_format} != "
                        f"{obj.result_format.struct_attrtype(attr.val)}"
                    )
                val_code = self(val)
                c_setattr(obj.result_format, self, self(obj), attr.val, val_code)
                return None
            case asm.Call(f, args):
                assert isinstance(f, asm.Literal)
                return c_function_call(f.val, self, *args)
            # case asm.Slot(var_n, var_t) as ref:
            #    return self(self.deref(ref))
            # case asm.Stack(obj, var_t) as ref:
            #    return var_t.c_lower(self, obj)
            case asm.Unpack(asm.Slot(var_n, var_t), val):
                val_code = self(val)
                if val.result_format != var_t:
                    raise TypeError(f"Type mismatch: {val.result_format} != {var_t}")
                if var_n in self.slots:
                    raise KeyError(
                        f"Slot {var_n} already exists in context, cannot unpack"
                    )
                if var_n in self.types:
                    raise KeyError(
                        f"Variable '{var_n}' is already defined in the current"
                        f" context, cannot overwrite with slot."
                    )
                var_t_code = self.ctype_name(c_type(var_t))
                self.exec(f"{feed}{var_t_code} {var_n} = {val_code};")
                self.types[var_n] = var_t
                self.slots[var_n] = var_t.c_unpack(
                    self, var_n, asm.Variable(var_n, var_t)
                )
                return None
            case asm.Repack(asm.Slot(var_n, var_t)):
                if var_n not in self.slots or var_n not in self.types:
                    raise KeyError(f"Slot {var_n} not found in context, cannot repack")
                if var_t != self.types[var_n]:
                    raise TypeError(f"Type mismatch: {var_t} != {self.types[var_n]}")
                obj = self.slots[var_n]
                var_t.c_repack(self, var_n, obj)
                return None
            case asm.Load(buf, idx):
                buf = self.resolve(buf)
                return buf.result_format.c_load(self, buf, idx)
            case asm.Store(buf, idx, val):
                buf = self.resolve(buf)
                return buf.result_format.c_store(self, buf, idx, val)
            case asm.Resize(buf, len):
                buf = self.resolve(buf)
                return buf.result_format.c_resize(self, buf, len)
            case asm.Length(buf):
                buf = self.resolve(buf)
                return buf.result_format.c_length(self, buf)
            case asm.Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case asm.ForLoop(var, start, end, body):
                var_t = self.ctype_name(c_type(var.result_format))
                var_2 = self(var)
                start = self(start)
                end = self(end)
                ctx_2 = self.subblock()
                ctx_2(body)
                ctx_2.types[var.name] = var.result_format
                body_code = ctx_2.emit()
                self.exec(
                    f"{feed}for ({var_t} {var_2} = {start}; "
                    f"{var_2} < {end}; {var_2}++) {{\n"
                    f"{body_code}"
                    f"\n{feed}}}"
                )
                return None
            case asm.BufferLoop(buf, var, body):
                idx = asm.Variable(
                    self.freshen(var.name + "_i"), buf.result_format.shape_type()
                )
                start = asm.Literal(0)
                stop = asm.Call(
                    asm.Literal(operator.sub), (asm.Length(buf), asm.Literal(1))
                )
                body_2 = asm.Block((asm.Assign(var, asm.Load(buf, idx)), body))
                return self(asm.ForLoop(idx, start, stop, body_2))
            case asm.WhileLoop(cond, body):
                if not isinstance(cond, asm.Literal | asm.Variable):
                    cond_var = asm.Variable(self.freshen("cond"), cond.result_format)
                    new_prgm = asm.Block(
                        (
                            asm.Assign(cond_var, cond),
                            asm.WhileLoop(
                                cond_var,
                                asm.Block(
                                    (
                                        body,
                                        asm.Assign(cond_var, cond),
                                    )
                                ),
                            ),
                        )
                    )
                    return self(new_prgm)
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}while ({cond_code}) {{\n{body_code}\n{feed}}}")
                return None
            case asm.If(cond, body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}if ({cond_code}) {{\n{body_code}\n{feed}}}")
                return None
            case asm.IfElse(cond, body, else_body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                ctx_3 = self.subblock()
                ctx_3(else_body)
                else_body_code = ctx_3.emit()
                self.exec(
                    f"{feed}if ({cond_code}) {{\n{body_code}\n{feed}}} "
                    f"else {{\n{else_body_code}\n{feed}}}"
                )
                return None
            case asm.Function(asm.Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                arg_decls = []
                for arg in args:
                    match arg:
                        case asm.Variable(name, t):
                            t_name = self.ctype_name(c_type(t))
                            arg_decls.append(f"{t_name} {name}")
                            ctx_2.types[name] = t
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )
                ctx_2(body)
                body_code = ctx_2.emit()
                return_t_name = self.ctype_name(c_type(return_t))
                feed = self.feed
                self.exec(
                    f"{feed}{return_t_name} {func_name}({', '.join(arg_decls)}) {{\n"
                    f"{body_code}\n"
                    f"{feed}}}"
                )
                return None
            case asm.Return(value):
                value = self(value)
                self.exec(f"{feed}return {value};")
                return None
            case asm.Break():
                self.exec(f"{feed}break;")
                return None
            case asm.Module(funcs):
                for func in funcs:
                    if not isinstance(func, asm.Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )
                    self(func)
                return None
            case _:
                raise NotImplementedError(
                    f"Unrecognized assembly node type: {type(prgm)}"
                )


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


class CBufferFType(BufferFType, CArgumentFType, ABC):
    """
    Abstract base class for the ftype of datastructures. The ftype defines how
    the data in an Buffer is organized and accessed.
    """

    @abstractmethod
    def c_length(self, ctx, buffer):
        """
        Return C code which loads a named buffer at the given index.
        """
        ...

    @abstractmethod
    def c_load(self, ctx, buffer, index):
        """
        Return C code which loads a named buffer at the given index.
        """
        ...

    @abstractmethod
    def c_store(self, ctx, buffer, index, value):
        """
        Return C code which stores a named buffer to the given index.
        """
        ...

    @abstractmethod
    def c_resize(self, ctx, buffer, new_length):
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
    def c_unpack(self, ctx, lhs, rhs):
        """
        Convert a value to a symbolic representation in C. Returns a NamedTuple
        of unpacked variable names, etc. The `lhs` is the variable namespace to
        assign to.
        """
        ...

    @abstractmethod
    def c_repack(self, ctx, lhs, rhs):
        """
        Update an object based on a symbolic representation. The `rhs` is the
        symbolic representation to update from, and `lhs` is a variable name referring
        to the original object to update.
        """
        ...


def serialize_struct_to_c(fmt: AssemblyStructFType, obj) -> Any:
    args = [getattr(obj, name) for name in fmt.struct_fieldnames]
    return struct_c_type(fmt)(*args)


register_property(
    AssemblyStructFType, "serialize_to_c", "__attr__", serialize_struct_to_c
)


def deserialize_struct_from_c(fmt: AssemblyStructFType, obj, c_struct: Any) -> None:
    if fmt.is_mutable:
        for name in fmt.struct_fieldnames:
            setattr(obj, name, getattr(c_struct, name))
        return


register_property(
    AssemblyStructFType, "deserialize_from_c", "__attr__", deserialize_struct_from_c
)

c_structs: dict[Any, Any] = {}
c_structnames = Namespace()


def struct_c_type(fmt: AssemblyStructFType):
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


register_property(
    AssemblyStructFType,
    "c_type",
    "__attr__",
    lambda fmt: ctypes.POINTER(struct_c_type(fmt)),
)


def struct_c_getattr(fmt: AssemblyStructFType, ctx, obj, attr):
    return f"{obj}->{attr}"


register_property(
    AssemblyStructFType,
    "c_getattr",
    "__attr__",
    struct_c_getattr,
)


def struct_c_setattr(fmt: AssemblyStructFType, ctx, obj, attr, val):
    ctx.emit(f"{ctx.feed}{obj}->{attr} = {val};")
    return


register_property(
    AssemblyStructFType,
    "c_setattr",
    "__attr__",
    struct_c_setattr,
)


def struct_construct_from_c(fmt: AssemblyStructFType, c_struct):
    args = [getattr(c_struct, name) for (name, _) in fmt.struct_fieldnames]
    return fmt.__class__(*args)


register_property(
    AssemblyStructFType,
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
register_property(
    TupleFType,
    "construct_from_c",
    "__attr__",
    lambda fmt, obj, c_tuple: tuple(c_tuple),
)

register_property(
    TupleFType,
    "c_type",
    "__attr__",
    lambda fmt: ctypes.POINTER(
        struct_c_type(asm.NamedTupleFType("CTuple", fmt.struct_fields))
    ),
)
