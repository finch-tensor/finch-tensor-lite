from dataclasses import dataclass
from typing import Any

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import query_property
from ..finch_notation import TensorView
from ..symbolic import Context, ScopedDict, has_format
from . import looplets as lpl
from abc import ABC, abstractmethod


class FinchTensorFormat(TensorFormat, ABC):
    def lower_unwrap(tns):
        """
        Unwrap a tensor view to get the underlying tensor.
        This is used to get the original tensor from a tensor view.
        """
        ...

    def lower_increment(tns, val):
        """
        Increment a tensor view with an operation and value.
        This updates the tensor at the specified index with the operation and value.
        """
        ...


    def lower_declare(tns, init, op, shape):
        """
        Declare a tensor.
        """
        ...


    def lower_freeze(tns, op):
        """
        Freeze a tensor.
        """
        ...


    def lower_thaw(tns, op):
        """
        Thaw a tensor.
        """
        ...


    def unfurl(ctx, tns, ext, proto):
        ...


class BufferizedNDArray(Tensor):
    def __init__(self, arr:np.ndarray):
        itemsize = arr.dtype.itemsize
        for stride in arr.strides:
            if mod(stride, itemsize) != 0:
                raise ValueError("Array must be aligned to multiple of itemsize")
        self.strides = [div(stride, itemsize) for stride in arr.strides]
        self.buf = NumpyBuffer(np.stride_tricks.as_strided(x, shape = (np.dot(x.strides, x.shape)/itemsize,), strides=(itemsize,)))
    
    @property
    def format(self):
        """
        Returns the format of the buffer, which is a BufferizedNDArrayFormat.
        """
        return BufferizedNDArrayFormat(self.arr.dtype.type, length(self.strides))
    
    @property
    def shape(self):
        return self.arr.shape
    

class BufferizedNDArrayFormat(FinchTensorFormat):
    """
    A format for bufferized NumPy arrays that provides metadata about the array.
    This includes the fill value, element type, and shape type.
    """

    def __init__(self, dtype: np.dtype, ndim: int):
        self._dtype = dtype
        self._ndim = ndim

    def __eq__(self, other):
        if not isinstance(other, BufferizedNDArrayFormat):
            return False
        return self._dtype == other._dtype and self._ndim == other._ndim

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def fill_value(self) -> Any:
        return np.zeros((), dtype=self._dtype)[()]

    @property
    def element_type(self):
        return self._dtype.type

    @property
    def shape_type(self) -> tuple:
        return tuple(np.int_ for _ in range(self._ndim))

class BufferizedNDArrayAccessor(Tensor):
    """
    A class representing a tensor view that is bufferized.
    This is used to create a view of a tensor with a specific extent.
    """
    def __init__(self, tns, ndim = None, pos=None, op=None):
        self.tns = tns
        if pos is None:
            pos = length_type(tns)(0)
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
    
    def unpack(self, ctx, var_n, val):
        buf = ctx.freshen(var_n, "buf")
        length = ctx.freshen(var_n, "length")
        t = ctx.ctype_name(c_type(self._dtype))
        ctx.exec(
            f"{ctx.feed}{t}* {data} = ({t}*){ctx(val)}->data;\n"
            f"{ctx.feed}size_t {length} = {ctx(val)}->length;"
        )




    def lower_unwrap(self, ctx, obj):
        """
        Unwrap a tensor view to get the underlying tensor.
        This is used to get the original tensor from a tensor view.
        """
        return asm.Load(obj.tns.buf, obj.pos)
    
    def lower_increment(self, ctx, obj, val):
        return asm.Store(
            obj.tns,
            obj.pos,
            asm.Call(
                asm.Literal(obj.op),
                (asm.Load(obj.tns.arr, obj.pos), val),
            ),
        )
    
    def lower_declare(self, ctx, init, op, shape):
        """
        Declare a tensor.
        This creates a new tensor view with the given initialization, operation, and shape.
        """
        return asm.Declare(
            asm.Literal(TensorView),
            init,
            asm.Immediate(op),
            asm.Call(asm.Immediate(tuple), shape),
        )

    
    def unfurl(self, ctx, tns, ext, mode, proto):
        """
        Unfurl the tensor view to get the underlying tensor.
        """
        match mode:
            case ntn.Update(op):
                op_2 = op
            case ntn.Read():
                op_2 = None
            case _:
                raise ValueError(f"Unsupported mode: {mode}")
        assert op_2 == self.op
        return lpl.Lookup(
            body=lambda ctx, idx: ctx(
                ntn.Value(
                    asm.Call(
                        asm.Literal(TensorView),
                        tns,
                        asm.Immediate(op_2),
                        asm.Call(asm.Immediate(tuple), idx),
                    ),
                    TensorViewFormat(self.tns, (*self.idxs, idx.format), self.op),
                )
            ),
        )


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
