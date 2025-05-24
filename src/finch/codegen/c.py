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

from algebra import query_property, register_property

from .. import finch_assembly as asm
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
    return query_property(op, "__self__", "c_function_name", ctx, *args)


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
        return query_property(op, "__self__", "c_function_call", ctx, *args)
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
    (operator.shift_left, "<<"),
    (operator.shift_right, ">>"),
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


register_property(int, "__self__", "to_c_literal", lambda x: str(x))


class CContext(AbstractContext):
    """
    A class to represent a C environment.
    """

    def __init__(self, tab="    ", indent=0, **kwargs):
        super().__init__(**kwargs)
        self.tab = tab
        self.indent = indent

    def exec(self, thunk):
        super().exec(self.tab * self.indent + str(thunk))

    def post(self, thunk):
        super().post(self.tab * self.indent + str(thunk))

    def make_block(self):
        blk = super().make_block()
        blk.indent = self.indent + 1
        blk.tab = self.tab
        return blk

    def emit(self):
        space = self.tab * self.indent
        return (
            space
            + "{\n"
            + "\n".join(self.preamble)
            + "\n"
            + "\n".join(self.epilogue)
            + space
            + "}\n"
        )

    def __call__(self, prgm: asm.AssemblyNode):
        """
        lower the program to C code.
        """
        match prgm:
            case asm.Immediate(value):
                return c_literal(self, value)
            case asm.Variable(name):
                return name
            case asm.Symbolic():
                return self.to_c_value(self)
            case asm.Assign(var, val):
                var = self(var)
                val = self(val)
                return f"{var} = {val};"
            case asm.Call(f, args):
                return c_function_call(f, self, *args)
            case asm.Load(buf, index):
                buf = self(buf)
                index = self(index)
                return buf.obj.c_load(index)
            case asm.Store(buf, index, value):
                buf = self(buf)
                index = self(index)
                value = self(value)
                return buf.obj.c_store(index, value)
            case asm.Resize(buf, new_length):
                buf = self(buf)
                new_length = self(new_length)
                return buf.obj.c_resize(new_length)
            case asm.Length(buf):
                buf = self(buf)
                return buf.obj.c_length()
            case asm.Block(bodies):
                with self.block() as ctx:
                    for body in bodies:
                        ctx(body)
                return None
            case asm.ForLoop(var, start, end, body):
                var = self(var)
                start = self(start)
                end = self(end)
                with self.block() as ctx:
                    ctx(f"for ({var} = {start}; {var} < {end}; {var}++) {{")
                    ctx(body)
                    ctx("}")
                return None
            case asm.BufferLoop(buf, var, body):
                idx = asm.Variable(self.freshen(var.name + "_i"))
                buf = self(buf)
                start = asm.Immediate(0)
                stop = asm.Call(
                    asm.Immediate(operator.sub), asm.Length(buf), asm.Immediate(1)
                )
                body_2 = asm.Block(asm.Assign(var, asm.Load(buf, idx)), body)
                return self(asm.ForLoop(idx, start, stop, body_2))
            case asm.WhileLoop(cond, body):
                with self.block() as ctx:
                    cond = self(cond)
                    ctx(f"while ({cond}) {{")
                    ctx(body)
                    ctx("}")
                return None
            case asm.Return(value):
                value = self(value)
                self.exec(f"return {value};")
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
    with ctx.block() as ctx_2:
        sym_args = [
            arg.unpack_c(ctx_2, name)
            for arg, name in zip(args, arg_names, strict=False)
        ]
        f(ctx_2, *sym_args)
    return ctx.emit()
