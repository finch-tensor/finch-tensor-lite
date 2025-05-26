import ctypes
import operator
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from functools import lru_cache
from operator import methodcaller
from pathlib import Path
from typing import Any

from .. import finch_assembly as asm
from ..algebra import query_property, register_property
from ..symbolic import AbstractContext, AbstractSymbolic
from ..util import config
from ..util.cache import file_cache
from .abstract_buffer import AbstractFormat


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
        subprocess.run(compile_command, check=True)
        assert shared_lib_path.exists(), f"Compilation failed: {compile_command}"


@lru_cache(maxsize=10_000)
def get_c_function(function_name, c_code):
    """
    :param function_name: The name of the function to call.
    :param c_code: The code to compile
    """
    shared_lib_path = create_shared_lib(
        c_code,
        config.get("cc"),
        [*config.get("cflags").split(), *config.get("shared_cflags").split()],
    )

    # Load the shared library using ctypes
    shared_lib = ctypes.CDLL(str(shared_lib_path))

    # Get the function from the shared library
    return getattr(shared_lib, function_name)


class AbstractCArgument(ABC):
    @abstractmethod
    def serialize_to_c(self, name):
        """
        Return a ctypes-compatible struct to be used in place of this argument
        for the c backend.
        """

    @abstractmethod
    def deserialize_from_c(self, obj):
        """
        Update this argument based on how the c call modified `obj`, the result
        of `serialize_to_c`.
        """


class CKernel:
    """
    A class to represent a C kernel.
    """

    def __init__(self, function_name, c_code, argtypes):
        self.function_name = function_name
        self.c_code = c_code
        self.c_function = get_c_function(function_name, c_code)
        self.argtypes = argtypes

    def __call__(self, *args):
        """
        Calls the C function with the given arguments.
        """
        if len(args) != len(self.argtypes):
            raise ValueError(
                f"Expected {len(self.argtypes)} arguments, got {len(args)}"
            )
        for argtype, arg in zip(self.argtypes, args, strict=False):
            if not isinstance(arg, argtype):
                raise TypeError(f"Expected argument of type {argtype}, got {type(arg)}")
        serial_args = list(map(methodcaller("serialize_to_c"), args))
        res = self.c_function(*serial_args)
        for arg, serial_arg in zip(args, serial_args, strict=False):
            arg.deserialize_from_c(serial_arg)
        return res


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


ctype_to_c_name = {
    ctypes.c_bool: "bool",
    ctypes.c_char: "char",
    ctypes.c_wchar: "wchar_t",
    ctypes.c_byte: "char",
    ctypes.c_ubyte: "unsigned char",
    ctypes.c_short: "short",
    ctypes.c_ushort: "unsigned short",
    ctypes.c_int: "int",
    ctypes.c_int8: "int8_t",
    ctypes.c_int16: "int16_t",
    ctypes.c_int32: "int32_t",
    ctypes.c_int64: "int64_t",
    ctypes.c_uint: "unsigned int",
    ctypes.c_uint8: "uint8_t",
    ctypes.c_uint16: "uint16_t",
    ctypes.c_uint32: "uint32_t",
    ctypes.c_uint64: "uint64_t",
    ctypes.c_long: "long",
    ctypes.c_ulong: "unsigned long",
    ctypes.c_longlong: "long long",
    ctypes.c_ulonglong: "unsigned long long",
    ctypes.c_size_t: "size_t",
    ctypes.c_ssize_t: "ssize_t",
    ctypes.c_time_t: "time_t",
    ctypes.c_float: "float",
    ctypes.c_double: "double",
    ctypes.c_longdouble: "long double",
    ctypes.c_char_p: "char*",
    ctypes.c_wchar_p: "wchar_t*",
    ctypes.c_void_p: "void*",
}


class CContext(AbstractContext):
    """
    A class to represent a C environment.
    """

    def __init__(self, tab="    ", indent=0, headers=[], **kwargs):
        super().__init__(**kwargs)
        self.tab = tab
        self.indent = indent
        self.headers = headers

    def ctype_name(self, t):
        # Mapping from ctypes types to their C type names
        name = ctype_to_c_name.get(t)
        if name is not None:
            return name
        if isinstance(t, ctypes.Structure):
            name = t.__name__
            args = [
                f"{field.name}: {self.ctype_name(field.type)}" for field in t._fields_
            ]
            header = (
                f"struct {name} {{\n"
                + "\n".join(f"    {arg};" for arg in args)
                + "\n};"
            )
            self.headers.push(header)
            return name
        if isinstance(t, ctypes._Pointer):
            return f"{self.ctype_name(t._type_)}*"
        raise NotImplementedError(f"No C type mapping for {t}")

    @property
    def feed(self):
        return self.tab * self.indent

    def block(self):
        blk = super().block()
        blk.indent = self.indent
        blk.tab = self.tab
        blk.headers = self.headers
        return blk

    def subblock(self):
        blk = super().block()
        blk.indent = self.indent + 1
        blk.tab = self.tab
        blk.headers = self.headers
        return blk

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def __call__(self, prgm: asm.AssemblyNode):
        """
        lower the program to C code.
        """
        match prgm:
            case asm.Immediate(value):
                # in the future, would be nice to be able to pass in constants that
                # are more complex than C literals, maybe as globals.
                return c_literal(self, value)
            case asm.Variable(name, type):
                return name
            case asm.Symbolic():
                return self.to_c_value(self)
            case asm.Assign(var, val):
                t = var.get_type()
                var_t = self.ctype_name(c_type(var.get_type()))
                var = self(var)
                val = self(val)
                self.exec(f"{self.feed}{var_t} {var} = {val};")
            case asm.Call(f, args):
                assert isinstance(f, asm.Immediate)
                return c_function_call(f.val, self, *args)
            case asm.Load(buf, index):
                assert isinstance(buf, asm.Symbolic)
                index = self(index)
                return buf.obj.c_load(self, index)
            case asm.Store(buf, index, value):
                assert isinstance(buf, asm.Symbolic)
                index = self(index)
                value = self(value)
                return buf.obj.c_store(self, index, value)
            case asm.Resize(buf, new_length):
                assert isinstance(buf, asm.Symbolic)
                new_length = self(new_length)
                return buf.obj.c_resize(self, new_length)
            case asm.Length(buf):
                assert isinstance(buf, asm.Symbolic)
                return buf.obj.c_length(self)
            case asm.Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case asm.ForLoop(var, start, end, body):
                var_t = self.ctype_name(c_type(var.get_type()))
                var = self(var)
                start = self(start)
                end = self(end)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(
                    f"{self.feed}for ({var_t} {var} = {start}; {var} < {end}; {var}++) {{\n"
                    + body_code
                    + f"\n{self.feed}}}"
                )
                return None
            case asm.BufferLoop(buf, var, body):
                idx = asm.Variable(self.freshen(var.name + "_i"))
                start = asm.Immediate(0)
                stop = asm.Call(
                    asm.Immediate(operator.sub), asm.Length(buf), asm.Immediate(1)
                )
                body_2 = asm.Block(asm.Assign(var, asm.Load(buf, idx)), body)
                return self(asm.ForLoop(idx, start, stop, body_2))
            case asm.WhileLoop(cond, body):
                if not isinstance(cond, asm.Immediate | asm.Variable):
                    cond_var = asm.Variable(self.freshen("cond"))
                    new_prgm = asm.Block(
                        (
                            asm.Assign(cond_var, cond),
                            asm.While(
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
                self.exec(
                    f"{self.feed}while ({cond_code}) {{\n"
                    + body_code
                    + f"\n{self.feed}}}"
                )
                return None
            case asm.Return(value):
                value = self(value)
                self.exec(f"{self.feed}return {value};")
                return None


class AbstractCFormat(AbstractFormat, ABC):
    """
    Abstract base class for the format of datastructures. The format defines how
    the data in an AbstractBuffer is organized and accessed.
    """

    @abstractmethod
    def unpack_c(self, ctx, name):
        """
        Unpack the C object into a symbolic representation.
        """


class AbstractSymbolicCBuffer(AbstractSymbolic, ABC):
    @abstractmethod
    def c_length(self, ctx):
        """
        Return C code which loads a named buffer at the given index.
        """

    @abstractmethod
    def c_load(self, ctx, index):
        """
        Return C code which loads a named buffer at the given index.
        """

    @abstractmethod
    def c_store(self, ctx, index, value):
        """
        Return C code which stores a named buffer to the given index.
        """

    @abstractmethod
    def c_resize(self, ctx, new_length):
        """
        Return C code which resizes a named buffer to the given length.
        """


def c_function_entrypoint(f, arg_names, args):
    ctx = CContext()
    sym_args = [
        arg.unpack_c(ctx, name) for arg, name in zip(args, arg_names, strict=False)
    ]
    f(ctx, *sym_args)
    return ctx.emit()
