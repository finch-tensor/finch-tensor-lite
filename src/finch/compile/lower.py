from abc import ABC
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import Tensor, TensorFormat
from ..codegen import NumpyBuffer
from ..symbolic import Context, ScopedDict, has_format


class FinchTensorFormat(TensorFormat, ABC):
    def lower_unwrap(tns):
        """
        Unwrap a tensor view to get the underlying tensor.
        This is used to get the original tensor from a tensor view.
        """

    def lower_increment(tns, val):
        """
        Increment a tensor view with an operation and value.
        This updates the tensor at the specified index with the operation and value.
        """

    def lower_declare(tns, init, op, shape):
        """
        Declare a tensor.
        """

    def lower_freeze(tns, op):
        """
        Freeze a tensor.
        """

    def lower_thaw(tns, op):
        """
        Thaw a tensor.
        """

    def unfurl(ctx, tns, ext, proto): ...


class BufferizedNDArray(Tensor):
    def __init__(self, arr: np.ndarray):
        itemsize = arr.dtype.itemsize
        for stride in arr.strides:
            if stride % itemsize != 0:
                raise ValueError("Array must be aligned to multiple of itemsize")
        self.strides = [stride // itemsize for stride in arr.strides]
        self._shape = arr.shape
        self.buf = NumpyBuffer(
            np.lib.stride_tricks.as_strided(
                arr,
                shape=(np.dot(arr.strides, arr.shape) // itemsize,),
                strides=(itemsize,),
            )
        )

    def to_numpy(self):
        """
        Convert the bufferized NDArray to a NumPy array.
        This is used to get the underlying NumPy array from the bufferized NDArray.
        """
        return self.buf.arr.reshape(self._shape)

    @property
    def format(self):
        """
        Returns the format of the buffer, which is a BufferizedNDArrayFormat.
        """
        return BufferizedNDArrayFormat(format(self.arr), len(self.strides))

    @property
    def shape(self):
        return self._shape

    def declare(self, init, op, shape):
        """
        Declare a bufferized NDArray with the given initialization value,
        operation, and shape.
        """
        for dim in shape:
            if dim.start != 0:
                raise ValueError(
                    f"Invalid dimension start value {dim.start} for ndarray declaration."
                )
        shape = tuple(dim.end for dim in shape)
        self.buf.resize(np.prod(shape))
        for i in range(self.buf.length()):
            self.buf.store(i, init)
        return self

    def freeze(self, op):
        return self
    
    def thaw(self, op):
        return self

    def __getitem__(self, index):
        """
        Get an item from the bufferized NDArray.
        This allows for indexing into the bufferized array.
        """
        if isinstance(index, tuple):
            index = np.ravel_multi_index(index, self._shape)
        return self.buf.load(index)
    
    def __setitem__(self, index, value):
        """
        Set an item in the bufferized NDArray.
        This allows for indexing into the bufferized array.
        """
        if isinstance(index, tuple):
            index = np.ravel_multi_index(index, self._shape)
        self.buf.store(index, value)


class BufferizedNDArrayFormat(FinchTensorFormat):
    """
    A format for bufferized NumPy arrays that provides metadata about the array.
    This includes the fill value, element type, and shape type.
    """

    def __init__(self, buf, ndim: int):
        self.buf = buf
        self._ndim = ndim

    def __eq__(self, other):
        if not isinstance(other, BufferizedNDArrayFormat):
            return False
        return self.buf == other.buf and self._ndim == other._ndim

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def fill_value(self) -> Any:
        return np.zeros((), dtype=self.buf.dtype)[()]

    @property
    def element_type(self):
        return self.buf.dtype.type

    @property
    def shape_type(self) -> tuple:
        return tuple(np.int_ for _ in range(self._ndim))


class BufferizedNDArrayAccessor(Tensor):
    """
    A class representing a tensor view that is bufferized.
    This is used to create a view of a tensor with a specific extent.
    """

    def __init__(self, tns, ndim=None, pos=None, op=None):
        self.tns = tns
        if pos is None:
            pos = format(self.tns).buf.length_type(0)
        self.pos = pos
        self.op = op
        self.ndim = ndim(tns)

    @property
    def format(self):
        return BufferizedNDArrayAccessorFormat(format(self.tns), self.ndim, self.op)


class BufferizedNDArrayAccessorFormat(FinchTensorFormat):
    """
    A format for tensor views that allows unfurling.
    This is used to create a view of a tensor with a specific extent.
    """

    def __init__(self, tns, ndim, op):
        self.tns = tns
        self.ndim = ndim
        self.op = op

    def lower_unwrap(self, ctx, obj): ...

    def lower_increment(self, ctx, obj, val): ...

    def lower_declare(self, ctx, init, op, shape): ...

    def unfurl(self, ctx, tns, ext, mode, proto): ...


@dataclass(eq=True, frozen=True)
class ExtentFormat:
    start: Any
    end: Any

    def lower_loop(self, ctx, idx, ext, body):
        """
        Lower a loop with the given index and body.
        This is used to compile the loop into assembly.
        """

        def rewrite(node):
            match node:
                case ntn.Access(idx, ext, body):
                    return asm.Loop(ctx(idx), ctx(ext.start), ctx(ext.end), ctx(body))
                case _:
                    return node


@dataclass(eq=True)
class HaltState:
    """
    A class to represent the halt state of a notation program.
    These programs can't break, but calling return sets a special return value.
    """

    has_returned: bool = False
    return_var: Any = None


class NotationContext(Context):
    """
    Compiles Finch Notation to Finch Assembly. Holds the state of the
    compilation process.
    """

    def __init__(
        self,
        namespace=None,
        preamble=None,
        epilogue=None,
        bindings=None,
        slots=None,
        func_state=None,
    ):
        super().__init__(namespace=namespace, preamble=preamble, epilogue=epilogue)
        if bindings is None:
            bindings = ScopedDict()
        if slots is None:
            slots = ScopedDict()
        self.bindings = bindings
        self.func_state = func_state

    def scope(self):
        """
        Create a new scoped context that inherits from this one.
        """
        return NotationContext(
            namespace=self.namespace,
            preamble=self.preamble,
            epilogue=self.epilogue,
            bindings=self.bindings.scope(),
            slots=self.slots.scope(),
        )

    def should_halt(self):
        """
        Check if the current function should halt.
        This is used to determine if the function has returned.
        """
        return self.func_state.has_returned

    def __call__(self, prgm):
        """
        Lower Finch Notation to Finch Assembly. First we check for early
        simplifications, then we call the normal lowering for the outermost
        node.
        """
        match prgm:
            case ntn.Literal(value):
                return asm.Literal(value)
            case ntn.Value(expr, _):
                return expr
            case ntn.Call(f, args):
                f_e = self(f)
                args_e = [self(arg) for arg in args]
                return asm.Call(f_e, *args_e)
            case ntn.Assign(var, val):
                self.exec(asm.Assign(self(var), self(val)))
                return None
            case ntn.Variable(var_n, var_t):
                return asm.Variable(var_n, var_t)
            case ntn.Slot(var_n, var_t):
                if var_n in self.types:
                    def_t = self.types[var_n]
                    if def_t != var_t:
                        raise TypeError(
                            f"Slot '{var_n}' is declared as type {def_t}, "
                            f"but used as type {var_t}."
                        )
                if var_n in self.slots:
                    return self.slots[var_n]
                raise KeyError(f"Slot '{var_n}' is not defined in the current context.")
            case ntn.Unpack(ntn.Slot(var_n, var_t), val):
                val_e = self(val)
                if not has_format(val_e, var_t):
                    raise TypeError(
                        f"Assigned value {val_e} is not of type {var_t} for "
                        f"variable '{var_n}'."
                    )
                assert var_n not in self.types, (
                    f"Variable '{var_n}' is already defined in the current"
                    f" context, cannot overwrite with slot."
                )
                self.types[var_n] = var_t
                self.slots[var_n] = val_e
                val_e = self(val)
                return None
            case ntn.Repack(ntn.Slot(var_n, var_t)):
                self.bindings[var_n] = self.slots[var_n]
                return None
            case ntn.Access(_):
                raise NotImplementedError("Access should have been lowered already.")
            case ntn.Unwrap(tns):
                return tns.format.lower_unwrap(self)
            case ntn.Increment(tns, val):
                val_e = self(val)
                return tns.format.lower_increment(self, val_e)
            case ntn.Block(bodies):
                for body in bodies:
                    self(body)
                return None
            case ntn.Loop(idx, ext, body):
                ext.format.lower_loop(self, idx, body)
                return None
            case ntn.Declare(tns, init, op, shape):
                init_e = self(init)
                op_e = self(op)
                shape_e = [self(s) for s in shape]
                return tns.format.lower_declare(init_e, op_e, shape_e)
            case ntn.Freeze(tns, op):
                op_e = self(op)
                return tns.format.lower_freeze(op_e)
            case ntn.Thaw(tns, op):
                op_e = self(op)
                return tns.format.lower_thaw(op_e)
            case ntn.If(cond, body):
                ctx = self.block()
                ctx_2 = ctx.scope()
                ctx_2(body)
                ctx.exec(asm.If(ctx(cond), ctx_2.emit()))
                return None
            case ntn.IfElse(cond, body, else_body):
                ctx = self.block()
                ctx_2 = ctx.scope()
                ctx_2(body)
                ctx_3 = ctx.scope()
                ctx_3(else_body)
                ctx.exec(asm.IfElse(ctx(cond), ctx_2.emit(), ctx_3.emit()))
                return None
            case ntn.Function(ntn.Variable(func_n, ret_t), args, body):
                ctx = self.scope()
                ctx.func_state = HaltState()
                ctx(body)
                exec(
                    asm.Function(
                        asm.Variable(func_n, ret_t),
                        [ctx(arg) for arg in args],
                        ctx.scope()(body),
                    )
                )
                return None
            case ntn.Return(value):
                if self.func_state is None:
                    raise ValueError("Return statement outside of function.")
                self.exec(asm.Assign(self.func_state.return_var, self(value)))
                return None
            case ntn.Module(funcs):
                ctx = self.scope()
                for func in funcs:
                    ctx(func)
                ctx.exec(asm.Module(ctx.emit()))
                return None
