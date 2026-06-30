import builtins
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
    FType,
    ImmutableStructFType,
    MutableStructFType,
    NamedTupleFType,
    StructFType,
    TupleFType,
    ffuncs,
    fisinstance,
    ftype,
)
from finchlite.algebra.algebra import FinchOperator
from finchlite.finch_assembly import BufferFType, DictFType
from finchlite.symbolic import Context, Namespace, ScopedDict, UnvalidatedForm
from finchlite.util import config, file_cache
from finchlite.util.logging import LOG_BACKEND_C

from .stages import CCode, CLowerer

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_BACKEND_C)


class COperator(ABC):
    """Abstract base class for C language operators."""

    @property
    @abstractmethod
    def c_symbol(self) -> str:
        """Returns the C symbol for this operator (e.g., '+', '-', '*')."""

    @abstractmethod
    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        """Generates the C function call for this operator."""


class CNAryOperator(COperator):
    """Base class for n-ary C operators (operators that take multiple arguments)."""

    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        return c_nary_function_call(self.c_symbol, ctx, *args)


class CBinaryOperator(COperator):
    """Base class for binary C operators (operators that take exactly two arguments)."""

    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        return c_binary_function_call(self.c_symbol, ctx, *args)


class CUnaryOperator(COperator):
    """Base class for unary C operators (operators that take exactly one argument)."""

    def c_function_call(self, ctx: Any, *args: Any) -> Any:
        return c_unary_function_call(self.c_symbol, ctx, *args)


common_h = Path(__file__).parents[1] / "stc" / "include" / "stc" / "common.h"


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


class CKernel(asm.AssemblyKernel):
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
        if self.ret_type is algebra.none_:
            return None
        return construct_from_c(self.ret_type, res)


class CLibrary(asm.AssemblyLibrary):
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


class CCompiler(UnvalidatedForm, asm.AssemblyLoader):
    """
    A class to compile and run FinchAssembly.
    """

    def __init__(
        self, ctx: CLowerer | None = None, cc=None, cflags=None, shared_cflags=None
    ):
        if cc is None:
            cc = config.get("cc")
        if cflags is None:
            cflags = config.get("cflags").split()
        if shared_cflags is None:
            shared_cflags = config.get("shared_cflags").split()
        self.cc = cc
        self.cflags = cflags
        self.shared_cflags = shared_cflags
        self.ctx: CLowerer = CGenerator() if ctx is None else ctx

    def lower(self, prgm: asm.Module) -> CLibrary:
        c_code = self.ctx(prgm).code
        logger.debug(f"Compiling C code:\n{c_code}")
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
                    # return_t = c_type(return_t)
                    arg_ts = [arg.result_type for arg in args]
                    kern = CKernel(getattr(lib, func_name), return_t, arg_ts)
                    kernels[func_name] = kern
                case _:
                    raise NotImplementedError(
                        f"Unrecognized function type: {type(func)}"
                    )
        return CLibrary(lib, kernels)


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
        TypeError: If the C function name is not implemented for the given function.
    """
    match op:
        case ffuncs.add:
            return "+"
        case ffuncs.mul:
            return "*"
        case ffuncs.sub:
            return "-"
        case ffuncs.truediv | ffuncs.floordiv:
            return "/"
        case ffuncs.mod:
            return "%"
        case ffuncs.lshift:
            return "<<"
        case ffuncs.rshift:
            return ">>"
        case ffuncs.and_:
            return "&"
        case ffuncs.xor:
            return "^"
        case ffuncs.or_:
            return "|"
        case ffuncs.not_:
            return "!"
        case ffuncs.invert:
            return "~"
        case ffuncs.eq:
            return "=="
        case ffuncs.ne:
            return "!="
        case ffuncs.gt:
            return ">"
        case ffuncs.lt:
            return "<"
        case ffuncs.ge:
            return ">="
        case ffuncs.le:
            return "<="
        case ffuncs.pow:
            return "pow"
        case COperator():
            return op.c_symbol
        case _:
            raise TypeError(f"{op} has no C representation.")


def c_nary_function_call(c_symbol: str, ctx: Any, *args: Any) -> str:
    """Generate C code for n-ary operators."""
    assert len(args) > 0
    if len(args) == 1:
        return f"{c_symbol}{ctx(args[0])}"
    return f" {c_symbol} ".join(map(ctx, args))


def c_binary_function_call(c_symbol: str, ctx: Any, *args: Any) -> str:
    """Generate C code for binary operators."""
    a, b = args
    return f"{ctx(a)} {c_symbol} {ctx(b)}"


def c_unary_function_call(c_symbol: str, ctx: Any, *args: Any) -> str:
    """Generate C code for unary operators."""
    return f"{c_symbol}{ctx(args[0])}"


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
    c_symbol = c_function_name(op, ctx, *args)
    match op:
        case ffuncs.add | ffuncs.mul | ffuncs.and_ | ffuncs.xor | ffuncs.or_:
            return c_nary_function_call(c_symbol, ctx, *args)
        case (
            ffuncs.sub
            | ffuncs.truediv
            | ffuncs.floordiv
            | ffuncs.mod
            | ffuncs.lshift
            | ffuncs.rshift
            | ffuncs.eq
            | ffuncs.ne
            | ffuncs.gt
            | ffuncs.lt
            | ffuncs.ge
            | ffuncs.le
        ):
            return c_binary_function_call(c_symbol, ctx, *args)
        case ffuncs.not_ | ffuncs.invert:
            return c_unary_function_call(c_symbol, ctx, *args)
        case ffuncs.pow:
            a, b = args
            return f"{c_symbol}({ctx(a)}, {ctx(b)})"
        case COperator():
            return op.c_function_call(ctx, *args)
        case _:
            raise TypeError(f"{op} has no C representation.")


def c_literal(ctx, val):
    """
    Returns the C literal corresponding to the given Python value.

    Args:
        ctx: The context in which the value is used.
        val: The Python value.

    Returns:
        The C literal as a string.
    """
    fmt = ftype(val)
    match fmt:
        case algebra.int_ | algebra.float_:
            return str(val)
        case algebra.bool_:
            return "true" if val else "false"
        case algebra.str_:
            return f'"{val}"'
        case algebra.ftypes.FDTypeNumpy():
            return numpy_c_literal(fmt, val, ctx)
        case _:
            if hasattr(val, "c_literal"):
                return val.c_literal(ctx)
            raise NotImplementedError(f"No C literal mapping for {fmt}")


def numpy_c_literal(fmt: FType, x, ctx):
    value = ("true" if x else "false") if isinstance(x, np.bool_) else str(x.item())
    return f"({ctx.ctype_name(c_type(fmt))}){value}"


def c_type(t: FType):
    """
    Returns the C type corresponding to the given Finch type.

    Args:
        ctx: The context in which the value is used.
        t: The Finch type.

    Returns:
        The corresponding C type as a ctypes type.
    """
    match t:
        case CArgumentFType():
            return t.c_type()
        case algebra.int_:
            return ctypes.c_int
        case algebra.float_:
            return ctypes.c_double
        case algebra.bool_:
            return ctypes.c_bool
        case algebra.str_:
            return ctypes.c_wchar_p
        case algebra.none_:
            return None
        case algebra.ftypes.FDTypeNumpy():
            return np.ctypeslib.as_ctypes_type(t.dtype)
        case TupleFType():
            return struct_c_type(NamedTupleFType("CTuple", t.struct_fields))
        case MutableStructFType():
            return ctypes.POINTER(struct_c_type(t))
        case ImmutableStructFType():
            return struct_c_type(t)
        case _:
            raise NotImplementedError(f"No C type mapping for {t}")


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


ctype_print_fmt: dict[Any, str] = {
    ctypes.c_bool: "%d",
    ctypes.c_char: "%c",
    ctypes.c_wchar: "%lc",
    ctypes.c_byte: "%d",
    ctypes.c_ubyte: "%d",
    ctypes.c_int8: "%d",
    ctypes.c_int16: "%d",
    ctypes.c_int32: "%d",
    ctypes.c_int64: "%ld",
    ctypes.c_uint8: "%u",
    ctypes.c_uint16: "%u",
    ctypes.c_uint32: "%u",
    ctypes.c_uint64: "%lu",
    ctypes.c_char_p: "%s",
    ctypes.c_wchar_p: "%ls",
    ctypes.c_float: "%f",
    ctypes.c_double: "%f",
}


class PrintableCFType(ABC):
    @abstractmethod
    def c_print(self, ctx, obj) -> str:
        """
        Return a C expression for this value in printf argument position.
        """
        ...


def c_print(fmt: FType, ctx, obj) -> str:
    match fmt:
        case PrintableCFType():
            return fmt.c_print(ctx, obj)
        case (
            algebra.ftypes.FDTypeNumpy()
            | algebra.int_
            | algebra.float_
            | algebra.bool_
            | algebra.str_
        ):
            return c_print_scalar(fmt, ctx, obj)
        case _:
            raise NotImplementedError(f"No C print mapping for {fmt}")


def c_print_scalar(fmt: FType, ctx, obj) -> str:
    ctype = c_type(fmt)
    if ctype not in ctype_print_fmt:
        raise NotImplementedError(f"No C print mapping for {fmt}")
    if isinstance(fmt, algebra.ftypes.FDTypeBoolean):
        return f"(int){obj}"
    return obj


def c_print_string_fallback(fmt: FType, ctx) -> str:
    try:
        fallback = ctx.ctype_name(c_type(fmt))
    except NotImplementedError:
        fallback = str(fmt)
    return f'"{c_string_literal(fallback)}"'


def c_printf_arg(fmt: FType, ctx, obj, conversion: str) -> str:
    if conversion == "s":
        try:
            return c_print(fmt, ctx, obj)
        except NotImplementedError:
            return c_print_string_fallback(fmt, ctx)
    return c_print(fmt, ctx, obj)


def c_string_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def printf_conversions(fmt: str) -> list[str]:
    conversions = []
    idx = 0
    while idx < len(fmt):
        if fmt[idx] != "%":
            idx += 1
            continue
        idx += 1
        if idx < len(fmt) and fmt[idx] == "%":
            idx += 1
            continue
        while idx < len(fmt) and fmt[idx] in "-+ #0":
            idx += 1
        if idx < len(fmt) and fmt[idx] == "*":
            raise NotImplementedError("dynamic printf widths are not supported")
        while idx < len(fmt) and fmt[idx].isdigit():
            idx += 1
        if idx < len(fmt) and fmt[idx] == ".":
            idx += 1
            if idx < len(fmt) and fmt[idx] == "*":
                raise NotImplementedError("dynamic printf precisions are not supported")
            while idx < len(fmt) and fmt[idx].isdigit():
                idx += 1
        if fmt.startswith(("hh", "ll"), idx):
            idx += 2
        elif idx < len(fmt) and fmt[idx] in "hljztL":
            idx += 1
        if idx >= len(fmt):
            raise ValueError("incomplete printf format specifier")
        conversions.append(fmt[idx])
        idx += 1
    return conversions


class CGenerator(UnvalidatedForm, CLowerer):
    def lower(self, prgm: asm.AssemblyNode) -> CCode:
        ctx = CContext()
        ctx(prgm)
        return CCode(ctx.emit_global())


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
        var_t = val.result_type
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
                if val.result_type != var_t:
                    raise TypeError(f"Type mismatch: {val.result_type} != {var_t}")
                if var_n in self.types:
                    assert var_t == self.types[var_n]
                    self.exec(f"{feed}{var_n} = {val_code};")
                else:
                    self.types[var_n] = var_t
                    var_t_code = self.ctype_name(c_type(var_t))
                    self.exec(f"{feed}{var_t_code} {var_n} = {val_code};")
                return None
            case asm.GetAttr(obj, attr):
                obj_t = obj.result_type
                if not isinstance(obj_t, StructFType):
                    raise TypeError(f"Expected struct type, got: {obj_t}")
                if not obj_t.struct_hasattr(attr.val):
                    raise ValueError("trying to get missing attr")
                return c_getattr(obj_t, self, self(obj), attr.val)
            case asm.SetAttr(obj, attr, val):
                obj = self.cache("obj", obj)
                obj_t = obj.result_type
                if not isinstance(obj_t, StructFType):
                    raise TypeError(f"Expected struct type, got: {obj_t}")
                if not fisinstance(val, obj_t.struct_attrtype(attr.val)):
                    raise TypeError(
                        f"Type mismatch: {val.result_type} != "
                        f"{obj_t.struct_attrtype(attr.val)}"
                    )
                val_code = self(val)
                c_setattr(obj_t, self, self(obj), attr.val, val_code)
                return None
            case asm.Call(f, args):
                return c_function_call(f.val, self, *args)
            # case asm.Slot(var_n, var_t) as ref:
            #    return self(self.deref(ref))
            # case asm.Stack(obj, var_t) as ref:
            #    return var_t.c_lower(self, obj)
            case asm.Unpack(asm.Slot(var_n, var_t), val):
                val_code = self(val)
                if val.result_type != var_t:
                    raise TypeError(f"Type mismatch: {val.result_type} != {var_t}")
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
                buf_t = buf.result_type
                if not isinstance(buf_t, CBufferFType):
                    raise TypeError(f"Expected C buffer type, got: {buf_t}")
                return buf_t.c_load(self, buf, idx)
            case asm.Store(buf, idx, val):
                buf = self.resolve(buf)
                buf_t = buf.result_type
                if not isinstance(buf_t, CBufferFType):
                    raise TypeError(f"Expected C buffer type, got: {buf_t}")
                return buf_t.c_store(self, buf, idx, val)
            case asm.Resize(buf, len):
                buf = self.resolve(buf)
                buf_t = buf.result_type
                if not isinstance(buf_t, CBufferFType):
                    raise TypeError(f"Expected C buffer type, got: {buf_t}")
                return buf_t.c_resize(self, buf, len)
            case asm.Length(buf):
                buf = self.resolve(buf)
                buf_t = buf.result_type
                if not isinstance(buf_t, CBufferFType):
                    raise TypeError(f"Expected C buffer type, got: {buf_t}")
                return buf_t.c_length(self, buf)
            case asm.LoadDict(map, idx):
                map = self.resolve(map)
                map_t = map.result_type
                if not isinstance(map_t, CDictFType):
                    raise TypeError(f"Expected C dict type, got: {map_t}")
                return map_t.c_loaddict(self, map, idx)
            case asm.ExistsDict(map, idx):
                map = self.resolve(map)
                map_t = map.result_type
                if not isinstance(map_t, CDictFType):
                    raise TypeError(f"Expected C dict type, got: {map_t}")
                return map_t.c_existsdict(self, map, idx)
            case asm.StoreDict(map, idx, val):
                map = self.resolve(map)
                map_t = map.result_type
                if not isinstance(map_t, CDictFType):
                    raise TypeError(f"Expected C dict type, got: {map_t}")
                return map_t.c_storedict(self, map, idx, val)
            case asm.Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case asm.ForLoop(asm.Variable(_, _) as var, start, end, body):
                var_t = self.ctype_name(c_type(var.result_type))
                var_2 = self(var)
                start = self(start)
                end = self(end)
                ctx_2 = self.subblock()
                ctx_2(body)
                ctx_2.types[var.name] = var.result_type
                body_code = ctx_2.emit()
                self.exec(
                    f"{feed}for ({var_t} {var_2} = {start}; "
                    f"{var_2} < {end}; {var_2}++) {{\n"
                    f"{body_code}"
                    f"\n{feed}}}"
                )
                return None
            case asm.BufferLoop(buf, asm.Variable(_, t) as var, body):
                if not isinstance(buf.result_type, BufferFType):
                    raise TypeError(f"Expected buffer type, got: {buf.result_type}")
                idx = asm.Variable(
                    self.freshen(var.name + "_i"), buf.result_type.length_type
                )
                start = asm.Literal(t(0))
                stop = asm.Call(
                    asm.Literal(ffuncs.sub), (asm.Length(buf), asm.Literal(t(1)))
                )
                body_2 = asm.Block((asm.Assign(var, asm.Load(buf, idx)), body))
                return self(asm.ForLoop(idx, start, stop, body_2))
            case asm.WhileLoop(cond, body):
                if not isinstance(cond, asm.Literal | asm.Variable):
                    cond_var = asm.Variable(self.freshen("cond"), cond.result_type)
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
            case asm.Print(args):
                self.add_header("#include <stdio.h>")
                match args:
                    case (asm.Literal(str() as fmt), *vals):
                        pass
                    case _:
                        raise TypeError("Print expects a literal format string")
                conversions = printf_conversions(fmt)
                if builtins.len(conversions) != builtins.len(vals):
                    raise ValueError(
                        f"printf format expects {builtins.len(conversions)} arguments, "
                        f"got {builtins.len(vals)}"
                    )
                print_args = [
                    c_printf_arg(val.result_type, self, self(val), conversion)
                    for val, conversion in zip(vals, conversions, strict=False)
                ]
                fmt_str = c_string_literal(fmt)
                if print_args:
                    self.exec(
                        f'{feed}printf("{fmt_str}", {", ".join(print_args)});'
                    )
                else:
                    self.exec(f'{feed}printf("{fmt_str}");')
                return None
            case _:
                raise NotImplementedError(
                    f"Unrecognized assembly node type: {type(prgm)}"
                )


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


def c_getattr(fmt: FType, ctx, obj, attr):
    match fmt:
        case _ if hasattr(fmt, "c_getattr"):
            return fmt.c_getattr(ctx, obj, attr)
        case MutableStructFType():
            return f"{obj}->{attr}"
        case ImmutableStructFType() | TupleFType():
            return f"{obj}.{attr}"
        case _:
            raise NotImplementedError(f"No C getattr mapping for {fmt}")


def c_setattr(fmt: FType, ctx, obj, attr, val):
    match fmt:
        case _ if hasattr(fmt, "c_setattr"):
            return fmt.c_setattr(ctx, obj, attr, val)
        case MutableStructFType():
            return struct_mutable_setattr(fmt, ctx, obj, attr, val)
        case _:
            raise NotImplementedError(f"No C setattr mapping for {fmt}")


def struct_mutable_setattr(fmt: StructFType, ctx, obj, attr, val):
    ctx.exec(f"{ctx.feed}{obj}->{attr} = {val};")


# the equivalent for immutable is f"{ctx.feed}{obj}.{attr} = {val};"
# but we will not include that because it's bad.


class CArgumentFType(ABC):
    @abstractmethod
    def c_type(self):
        """
        Return a ctypes type for this ftype.
        """
        ...

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


def serialize_to_c(fmt: FType, obj):
    """
    Serialize an object to a C-compatible ftype.

    Args:
        fmt: FType of obj
        obj: The object to serialize.

    Returns:
        A ctypes-compatible struct.
    """
    match fmt:
        case CArgumentFType():
            return fmt.serialize_to_c(obj)
        case algebra.ftypes.FDTypeNumpy():
            return np.ctypeslib.as_ctypes(np.array(obj, dtype=fmt.dtype))
        case algebra.int_ | algebra.float_ | algebra.bool_:
            return c_type(fmt)(obj)
        case algebra.none_:
            return None
        case TupleFType():
            return serialize_tuple_to_c(fmt, obj)
        case StructFType():
            return serialize_struct_to_c(fmt, obj)
        case _:
            raise NotImplementedError(f"No C serialization mapping for {fmt}")


def serialize_struct_to_c(fmt: StructFType, obj) -> Any:
    args = [serialize_to_c(fmt, getattr(obj, name)) for name, fmt in fmt.struct_fields]
    return struct_c_type(fmt)(*args)


def serialize_tuple_to_c(fmt: TupleFType, obj):
    x = namedtuple("CTuple", fmt.struct_fieldnames)(*obj)  # noqa: PYI024
    return serialize_to_c(ftype(x), x)


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
    match fmt:
        case CArgumentFType():
            fmt.deserialize_from_c(obj, c_obj)
        case algebra.ftypes.FDTypeNumpy() | TupleFType():
            return
        case StructFType():
            deserialize_struct_from_c(fmt, obj, c_obj)
        case _:
            return


def deserialize_struct_from_c(fmt: StructFType, obj, c_struct: Any) -> None:
    if fmt.is_mutable:
        for name in fmt.struct_fieldnames:
            setattr(obj, name, getattr(c_struct, name))
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
    match fmt:
        case CArgumentFType():
            return fmt.construct_from_c(c_obj)
        case algebra.ftypes.FDTypeNumpy():
            return fmt(c_obj.value if hasattr(c_obj, "value") else c_obj)
        case algebra.int_:
            return int(c_obj.value if hasattr(c_obj, "value") else c_obj)
        case algebra.float_:
            return float(c_obj.value if hasattr(c_obj, "value") else c_obj)
        case algebra.bool_:
            return bool(c_obj.value if hasattr(c_obj, "value") else c_obj)
        case algebra.none_:
            return None
        case TupleFType():
            return tuple_construct_from_c(fmt, c_obj)
        case StructFType():
            return struct_construct_from_c(fmt, c_obj)
        case _:
            return fmt(c_obj)


def struct_construct_from_c(fmt: StructFType, c_struct):
    args = [getattr(c_struct, name) for name in fmt.struct_fieldnames]
    return fmt.__class__(*args)


def tuple_construct_from_c(fmt: TupleFType, c_struct):
    args = [getattr(c_struct, name) for name in fmt.struct_fieldnames]
    return tuple(args)


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


def c_hash(fmt: FType, ctx: "CContext"):
    """
    Expand to the name of a macro that c hash can use for hashing fmt.

    Args:
        ctx: CContext object
        var_n: name to be supplied. It is a placeholder for a variable with
        type fmt* (so indirection)
    """
    match fmt:
        case CHashableFType():
            return fmt.c_hash(ctx)
        case algebra.ftypes.FDTypeNumpy() | algebra.int_ | algebra.float_:
            return c_hash_default(fmt, ctx)
        case ImmutableStructFType() | TupleFType():
            return c_hash_struct(fmt, ctx)
        case _:
            raise NotImplementedError(f"No C hash mapping for {fmt}")


def c_hash_default(fmt: FType, ctx: "CContext"):
    ctx.add_header(f'#include "{common_h}"')
    return "c_default_hash"


class CHashableProperties(TypedDict):
    eq: str | None
    hash: str | None


def c_hash_struct(fmt: ImmutableStructFType, ctx: "CContext"):
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


def c_eq(fmt: FType, ctx: "CContext"):
    """
    Expand to the name of a macro that c eq can use for checking equivalence of fmt.

    Args:
        ctx: CContext object
        var_n: name to be supplied. It is a placeholder for a variable with
        type fmt* (so indirection)
    """
    match fmt:
        case CHashableFType():
            return fmt.c_eq(ctx)
        case algebra.ftypes.FDTypeNumpy() | algebra.int_ | algebra.float_:
            return c_eq_default(fmt, ctx)
        case ImmutableStructFType() | TupleFType():
            return c_eq_struct(fmt, ctx)
        case _:
            raise NotImplementedError(f"No C equality mapping for {fmt}")


def c_eq_default(fmt: FType, ctx: "CContext"):
    ctx.add_header(f'#include "{common_h}"')
    return "c_default_eq"


def c_eq_struct(fmt: ImmutableStructFType, ctx: "CContext"):
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


class CDictFType(DictFType, CArgumentFType, ABC):
    """
    Abstract base class for the ftype of dictionaries. The ftype defines how
    the data in a Map is organized and accessed.
    """

    @abstractmethod
    def c_existsdict(self, ctx, map, idx):
        """
        Return C code which checks whether a given key exists in a map.
        """
        ...

    @abstractmethod
    def c_loaddict(self, ctx, map, idx):
        """
        Return C code which gets a value corresponding to a certain key.
        """
        ...

    @abstractmethod
    def c_storedict(self, ctx, buffer, idx, value):
        """
        Return C code which stores a certain value given a certain integer tuple key.
        """
        ...


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
