import logging
from abc import ABC, abstractmethod
from operator import methodcaller
from typing import Any

import numpy as np

from .. import finch_assembly as asm
from ..symbolic.environment import Context, ScopedDict
from .numpy_buffer import NumpyBuffer, NumpyBufferFormat

logger = logging.getLogger(__name__)


class NumbaArgument(ABC):
    @abstractmethod
    def serialize_to_numba(self):
        """
        Return a Numba-compatible object to be used in place of this argument
        for the Numba backend.
        """
        ...

    @classmethod
    @abstractmethod
    def deserialize_from_numba(cls, numba_buffer):
        """
        Return an object from Numba returned value.
        """
        ...


class NumbaBuffer(NumpyBuffer, NumbaArgument):
    def __init__(self, arr: np.ndarray):
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError("NumPy array must be C-contiguous")
        self.arr_ref = [arr]

    @property
    def arr(self):
        return self.arr_ref[0]

    def get_format(self):
        return NumbaBufferFormat(self.arr.dtype.type)

    def serialize_to_numba(self):
        return self.arr_ref

    @classmethod
    def deserialize_from_numba(cls, numba_buffer):
        return cls(numba_buffer[0])


class NumbaBufferFormat(NumpyBufferFormat):
    @staticmethod
    def numba_name():
        return "list[numpy.ndarray]"

    def __call__(self, len_: int):
        return NumbaBuffer(np.zeros(len_, dtype=self._dtype))


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
    def __init__(self, numba_func):
        self.numba_func = numba_func

    def __call__(self, *args):
        serial_args = list(map(methodcaller("serialize_to_numba"), args))
        results = self.numba_func(*serial_args)
        if np.isscalar(results):
            return results
        if len(results) == 0:
            return ()
        from .numpy_buffer import NumpyBuffer

        if not isinstance(results, tuple):
            return NumpyBuffer.deserialize_from_numba(results)
        return tuple(map(NumpyBuffer.deserialize_from_numba, results))


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
                case asm.Function(asm.Variable(func_name, _), _, _):
                    kern = globals()[func_name]
                    kernels[func_name] = NumbaKernel(kern)
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
                if val.get_type() != var_t:
                    raise TypeError(f"Type mismatch: {val.get_type()} != {var_t}")
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
                return f"{self(buffer)}[0][{self(idx)}]"
            case asm.Store(buffer, idx, val):
                self.exec(f"{self.feed}{self(buffer)}[0][{self(idx)}] = {self(val)}")
                return None
            case asm.Resize(buffer, size):
                self.exec(
                    f"{self.feed}{self(buffer)}[0] = numpy.resize({self(buffer)}[0], "
                    f"{self(size)})"
                )
                return None
            case asm.Length(buffer):
                return f"len({self(buffer)}[0])"
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
                ctx_2.bindings[var.name] = var.get_type()
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
