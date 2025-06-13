import logging
from abc import ABC, abstractmethod
from operator import methodcaller
from typing import Any

from .. import finch_assembly as asm
from ..finch_assembly import BufferFormat
from ..symbolic import Context, ScopedDict, has_format

logger = logging.getLogger(__name__)


class NumbaArgument(ABC):
    @abstractmethod
    def serialize_to_numba(self):
        """
        Return a Numba-compatible object to be used in place of this argument
        for the Numba backend.
        """
        ...

    @abstractmethod
    def deserialize_from_numba(self, numba_buffer):
        """
        Return an object from Numba returned value.
        """
        ...


class NumbaBufferFormat(BufferFormat, ABC):
    @abstractmethod
    def numba_length(self, ctx: "NumbaContext", buffer):
        """
        Return a Numba-compatible expression to get the length of the buffer.
        """
        ...

    @abstractmethod
    def numba_resize(self, ctx: "NumbaContext", buffer, size):
        """
        Return a Numba-compatible expression to resize the buffer to the given size.
        """
        ...

    @abstractmethod
    def numba_load(self, ctx: "NumbaContext", buffer, idx):
        """
        Return a Numba-compatible expression to load an element from the buffer
        at the given index.
        """
        ...

    @abstractmethod
    def numba_store(self, ctx: "NumbaContext", buffer, idx, value=None):
        """
        Return a Numba-compatible expression to store an element in the buffer
        at the given index. If value is None, it should store the length of the
        buffer.
        """
        ...

    @staticmethod
    def numba_name():
        return "list[numpy.ndarray]"


class NumbaModule:
    """
    A class to represent a Numba module.
    """

    def __init__(self, kernels):
        self.kernels = kernels

    def __getattr__(self, name):
        # Allow attribute access to kernels by name
        if name in self.kernels:
            return self.kernels[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class NumbaKernel:
    def __init__(self, numba_func, ret_type: type, arg_types):
        self.numba_func = numba_func
        self.ret_type = ret_type
        self.arg_types = arg_types

    def __call__(self, *args):
        for arg_type, arg in zip(self.arg_types, args, strict=False):
            if not has_format(arg, arg_type):
                raise TypeError(
                    f"Expected argument of type {arg_type}, got {type(arg)}"
                )
        serial_args = list(map(methodcaller("serialize_to_numba"), args))
        res = self.numba_func(*serial_args)
        for arg, serial_arg in zip(args, serial_args, strict=False):
            arg.deserialize_from_numba(serial_arg)
        if hasattr(self.ret_type, "construct_from_numba"):
            return res.construct_from_numba(res)
        if self.ret_type is type(None):
            return None
        return self.ret_type(res)


class NumbaCompiler:
    def __call__(self, prgm: asm.Module):
        ctx = NumbaContext()
        ctx(prgm)
        numba_code = ctx.emit_global()
        logger.info(f"Executing Numba code:\n{numba_code}")
        exec(numba_code, globals(), None)

        kernels = {}
        for func in prgm.funcs:
            match func:
                case asm.Function(asm.Variable(func_name, ret_type), args, _):
                    kern = globals()[func_name]
                    arg_ts = [arg.result_format for arg in args]
                    kernels[func_name] = NumbaKernel(kern, ret_type, arg_ts)
                case _:
                    raise NotImplementedError(
                        f"Unrecognized function type: {type(func)}"
                    )

        return NumbaModule(kernels)


class NumbaContext(Context):
    def __init__(self, tab="    ", indent=0, bindings=None):
        if bindings is None:
            bindings = ScopedDict()

        super().__init__()

        self.tab = tab
        self.indent = indent
        self.bindings = bindings

        self.imports = [
            "import _operator, builtins",
            "from numba import njit",
            "import numpy",
            "\n",
        ]

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def emit_global(self):
        """
        Emit the headers for the C code.
        """
        return "\n".join([*self.imports, self.emit()])

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def block(self) -> "NumbaContext":
        blk = super().block()
        blk.indent = self.indent
        blk.tab = self.tab
        blk.bindings = self.bindings
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        blk.bindings = self.bindings.scope()
        return blk

    @staticmethod
    def full_name(val: Any) -> str:
        if hasattr(val, "numba_name"):
            return val.numba_name()
        return f"{val.__module__}.{val.__name__}"

    def __call__(self, prgm: asm.AssemblyNode):
        feed = self.feed
        match prgm:
            case asm.Immediate(value):
                return str(value)
            case asm.Variable(name, _):
                return name
            case asm.Assign(asm.Variable(var_n, var_t), val):
                val_code = self(val)
                if val.result_format != var_t:
                    raise TypeError(f"Type mismatch: {val.result_format} != {var_t}")
                if var_n in self.bindings:
                    assert var_t == self.bindings[var_n]
                    self.exec(f"{feed}{var_n} = {val_code}")
                else:
                    self.bindings[var_n] = var_t
                    self.exec(f"{feed}{var_n}: {self.full_name(var_t)} = {val_code}")
                return None
            case asm.Call(asm.Immediate(val), args):
                return f"{self.full_name(val)}({', '.join(self(arg) for arg in args)})"
            case asm.Load(buffer, idx):
                return buffer.result_format.numba_load(self, buffer, idx)
            case asm.Store(buffer, idx, val):
                buffer.result_format.numba_store(self, buffer, idx, val)
                return None
            case asm.Resize(buffer, size):
                buffer.result_format.numba_resize(self, buffer, size)
                return None
            case asm.Length(buffer):
                return buffer.result_format.numba_length(self, buffer)
            case asm.Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case asm.ForLoop(var, start, end, body):
                var_2 = self(var)
                start = self(start)
                end = self(end)
                ctx_2 = self.subblock()
                ctx_2(body)
                ctx_2.bindings[var.name] = var.result_format
                body_code = ctx_2.emit()
                self.exec(f"{feed}for {var_2} in range({start}, {end}):\n{body_code}\n")
                return None
            case asm.BufferLoop(buffer, var, body):
                raise NotImplementedError
            case asm.WhileLoop(cond, body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}while {cond_code}:\n{body_code}\n")
                return None
            case asm.If(cond, body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}if {cond_code}:\n{body_code}\n")
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
                    f"{feed}if {cond_code}:\n{body_code}\n"
                    f"{feed}else:\n{else_body_code}\n"
                )
                return None
            case asm.Function(asm.Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                arg_decls = []
                for arg in args:
                    match arg:
                        case asm.Variable(name, t):
                            arg_decls.append(f"{name}: {self.full_name(t)}")
                            ctx_2.bindings[name] = t
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )
                ctx_2(body)
                body_code = ctx_2.emit()
                feed = self.feed
                self.exec(
                    f"{feed}@njit\n"
                    f"{feed}def {func_name}({', '.join(arg_decls)}) -> "
                    f"{self.full_name(return_t)}:\n"
                    f"{body_code}\n"
                )
                return None
            case asm.Return(value):
                value = self(value)
                self.exec(f"{feed}return {value}")
                return None
            case asm.Break():
                self.exec(f"{feed}break")
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
                raise NotImplementedError
