import ctypes
import logging
import operator
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from functools import lru_cache
from operator import methodcaller
from pathlib import Path
from typing import Any

import numpy as np

from .. import finch_assembly as asm
from ..algebra import query_property, register_property
from ..finch_assembly.abstract_buffer import AbstractFormat, isinstanceorformat
from ..symbolic import AbstractContext, ScopedDict
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


class AbstractCArgument(ABC):
    @abstractmethod
    def serialize_to_c(self, name):
        """
        Return a ctypes-compatible struct to be used in place of this argument
        for the c backend.
        """
        ...

    @abstractmethod
    def deserialize_from_c(self, obj):
        """
        Update this argument based on how the c call modified `obj`, the result
        of `serialize_to_c`.
        """
        ...


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
            if not isinstanceorformat(arg, argtype):
                raise TypeError(f"Expected argument of type {argtype}, got {type(arg)}")
        serial_args = list(map(methodcaller("serialize_to_c"), args))
        res = self.c_function(*serial_args)
        for arg, serial_arg in zip(args, serial_args, strict=False):
            arg.deserialize_from_c(serial_arg)
        if isinstanceorformat(res, self.ret_type):
            return res
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
                    arg_ts = [arg.get_type() for arg in args]
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
    return query_property(val, "__self__", "c_literal", ctx)


register_property(int, "__self__", "c_literal", lambda x, ctx: str(x))
register_property(float, "__self__", "c_literal", lambda x, ctx: str(x))
register_property(
    np.generic,
    "__self__",
    "c_literal",
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
        "__self__",
        "c_literal",
        lambda x, ctx: f"({ctx.ctype_name(type(x))}){x.value}",
    )

for t in (ctypes.c_float, ctypes.c_double, ctypes.c_longdouble):  # type: ignore[assignment]
    register_property(
        t,
        "__self__",
        "c_literal",
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
    return query_property(t, "__self__", "c_type")


register_property(int, "__self__", "c_type", lambda x: ctypes.c_int)
register_property(
    np.generic, "__self__", "c_type", lambda x: np.ctypeslib.as_ctypes_type(x)
)
register_property(ctypes._SimpleCData, "__self__", "c_type", lambda x: x)
register_property(type(None), "__self__", "c_type", lambda x: None)


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


class CContext(AbstractContext):
    """
    A class to represent a C environment.
    """

    def __init__(
        self, tab="    ", indent=0, headers=None, bindings=None, fptr=None, **kwargs
    ):
        if headers is None:
            headers = []
        if bindings is None:
            bindings = ScopedDict()
        super().__init__(**kwargs)
        self.tab = tab
        self.indent = indent
        self.headers = headers
        self._headerset = set(headers)
        self.fptr = {}
        self.bindings = bindings

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
        blk.bindings = self.bindings
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        blk.bindings = self.bindings.scope()
        return blk

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def __call__(self, prgm: asm.AssemblyNode):
        feed = self.feed
        """
        lower the program to C code.
        """
        match prgm:
            case asm.Immediate(value):
                # in the future, would be nice to be able to pass in constants that
                # are more complex than C literals, maybe as globals.
                return c_literal(self, value)
            case asm.Variable(name, t):
                return name
            case asm.Assign(asm.Variable(var_n, var_t), val):
                val_code = self(val)
                if val.get_type() != var_t:
                    raise TypeError(f"Type mismatch: {val.get_type()} != {var_t}")
                if var_n in self.bindings:
                    assert var_t == self.bindings[var_n]
                    self.exec(f"{feed}{var_n} = {val_code};")
                else:
                    self.bindings[var_n] = var_t
                    var_t_code = self.ctype_name(c_type(var_t))
                    self.exec(f"{feed}{var_t_code} {var_n} = {val_code};")
                return None
            case asm.Call(f, args):
                assert isinstance(f, asm.Immediate)
                return c_function_call(f.val, self, *args)
            case asm.Load(buf, idx):
                return buf.get_type().c_load(self, buf, idx)
            case asm.Store(buf, idx, val):
                return buf.get_type().c_store(self, buf, idx, val)
            case asm.Resize(buf, len):
                return buf.get_type().c_resize(self, buf, len)
            case asm.Length(buf):
                return buf.get_type().c_length(self, buf)
            case asm.Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case asm.ForLoop(var, start, end, body):
                var_t = self.ctype_name(c_type(var.get_type()))
                var_2 = self(var)
                start = self(start)
                end = self(end)
                ctx_2 = self.subblock()
                ctx_2(body)
                ctx_2.bindings[var.name] = var.get_type()
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
                    self.freshen(var.name + "_i"), buf.get_type().index_type()
                )
                start = asm.Immediate(0)
                stop = asm.Call(
                    asm.Immediate(operator.sub), (asm.Length(buf), asm.Immediate(1))
                )
                body_2 = asm.Block((asm.Assign(var, asm.Load(buf, idx)), body))
                return self(asm.ForLoop(idx, start, stop, body_2))
            case asm.WhileLoop(cond, body):
                if not isinstance(cond, asm.Immediate | asm.Variable):
                    cond_var = asm.Variable(self.freshen("cond"), cond.get_type())
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
                            ctx_2.bindings[name] = t
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


class AbstractCFormat(AbstractFormat, ABC):
    """
    Abstract base class for the format of datastructures. The format defines how
    the data in an AbstractBuffer is organized and accessed.
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
