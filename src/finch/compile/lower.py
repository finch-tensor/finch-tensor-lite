from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import Tensor, TensorFormat
from ..codegen import NumpyBuffer
from ..symbolic import Context, ScopedDict, has_format, format
from ..symbolic import PostOrderDFS, PostWalk
from typing import NamedTuple


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

    def lower_declare(self, ctx, tns, init, op, shape):
        """
        Declare a tensor.
        """

    def lower_freeze(self, ctx, tns, op):
        """
        Freeze a tensor.
        """

    def lower_thaw(self, ctx, tns, op):
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
        itemsize = self.buf.arr.dtype.itemsize
        return np.lib.stride_tricks.as_strided(
            self.buf.arr,
            shape=self._shape,
            strides=(stride * itemsize for stride in self.strides),
        )

    @property
    def format(self):
        """
        Returns the format of the buffer, which is a BufferizedNDArrayFormat.
        """
        return BufferizedNDArrayFormat(format(self.buf), len(self.strides))

    @property
    def shape(self):
        return self._shape

    def declare(self, init, op, shape):
        """
        Declare a bufferized NDArray with the given initialization value,
        operation, and shape.
        """
        for dim, size in zip(shape, self._shape):
            if dim.start != 0:
                raise ValueError(
                    f"Invalid dimension start value {dim.start} for ndarray declaration."
                )
            if dim.end != size:
                raise ValueError(
                    f"Invalid dimension end value {dim.end} for ndarray declaration."
                )
        shape = tuple(dim.end for dim in shape)
        for i in range(self.buf.length()):
            self.buf.store(i, init)
        return self

    def freeze(self, op):
        return self
    
    def thaw(self, op):
        return self

    def access(self, indices, op):
        return BufferizedNDArrayAccessor(self).access(indices, op)

    def __getitem__(self, index):
        """
        Get an item from the bufferized NDArray.
        This allows for indexing into the bufferized array.
        """
        if isinstance(index, tuple):
            index = np.dot(index, self.strides)
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

    def __hash__(self):
        return hash((self.buf, self._ndim))

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

    def lower_declare(self, ctx, tns, init, op, shape):
        i_var = asm.Variable(f"i", self.buf.length_type)
        buf = asm.Stack(tns.obj.buf, self.buf)
        body = asm.Store(
            buf,
            i_var,
            asm.Literal(init.val),
        )
        ctx.exec(
            asm.ForLoop(
                i_var,
                asm.Literal(0),
                asm.Length(buf),
                body
            )
        )
        return None
        
    def asm_unpack(self, ctx, var_n, val):
        """
        Unpack the into asm context.
        """
        stride = []
        for i in range(self._ndim):
            stride_i = asm.Variable(f"{var_n}_stride_{i}", self.buf.length_type)
            stride.append(stride_i)
            stride_e = asm.GetAttr(val, f"stride")
            stride_i_e = asm.GetAttr(stride_e, f"element_{i}")
            ctx.exec(asm.Assign(stride_i, stride_i_e))
        buf = asm.Variable(f"{var_n}_buf", self.buf)
        buf_e = asm.GetAttr(val, "buf")
        ctx.exec(asm.Assign(buf, buf_e))
        buf_s = asm.Slot(f"{var_n}_buf", self.buf)
        ctx.exec(asm.Unpack(buf_s, buf))
        class BufferizedNDArrayFields(NamedTuple):
            stride: list[asm.Variable]
            buf: asm.Variable
            buf_s: asm.Slot

        return BufferizedNDArrayFields(stride, buf, buf_s)

    def asm_repack(self, ctx, lhs, obj):
        """
        Repack the buffer from C context.
        """
        ctx.exec(asm.Repack(obj.buf))
        return


class BufferizedNDArrayAccessor(Tensor):
    """
    A class representing a tensor view that is bufferized.
    This is used to create a view of a tensor with a specific extent.
    """
    def __init__(self, tns: BufferizedNDArray, nind=None, pos=None, op=None):
        self.tns = tns
        if pos is None:
            pos = format(self.tns).buf.length_type(0)
        self.pos = pos
        self.op = op
        if nind is None:
            nind = 0
        self.nind = nind

    @property
    def format(self):
        return BufferizedNDArrayAccessorFormat(format(self.tns), self.nind, format(self.pos), self.op)

    @property
    def shape(self):
        return self.tns.shape[self.nind:]

    def access(self, indices, op):
        pos = self.pos + np.dot(indices, self.tns.strides[self.nind:self.nind + len(indices)])
        return BufferizedNDArrayAccessor(self.tns, self.nind + len(indices), pos, op)

    def unwrap(self):
        """
        Unwrap the tensor view to get the underlying tensor.
        This is used to get the original tensor from a tensor view.
        """
        assert self.ndim == 0, "Cannot unwrap a tensor view with non-zero dimension."
        return self.tns.buf.load(self.pos)

    def increment(self, val):
        """
        Increment the tensor view with a value.
        This updates the tensor at the specified index with the operation and value.
        """
        if self.op is None:
            raise ValueError("No operation defined for increment.")
        assert self.ndim == 0, "Cannot unwrap a tensor view with non-zero dimension."
        self.tns.buf.store(self.pos, self.op(self.tns.buf.load(self.pos), val))
        return self

    @property
    def format(self):
        return BufferizedNDArrayAccessorFormat(format(self.tns), self.nind, format(self.pos), self.op)


class BufferizedNDArrayAccessorFormat(FinchTensorFormat):
    def __init__(self, tns, nind, pos, op):
        self.tns = tns
        self.nind = nind
        self.pos = pos
        self.op = op

    def __eq__(self, other):
        return (
            isinstance(other, BufferizedNDArrayAccessorFormat)
            and self.tns == other.tns
            and self.nind == other.nind
            and self.pos == other.pos
            and self.op == other.op
        )
    
    def __hash__(self):
        return hash((self.tns, self.nind, self.pos, self.op))

    @property
    def ndim(self) -> int:
        return self.tns.ndim - self.nind

    @property
    def shape_type(self) -> tuple:
        return self.tns.shape_type[self.nind:]

    @property
    def fill_value(self) -> Any:
        return self.tns.fill_value
    
    @property
    def element_type(self):
        return self.tns.element_type
    
    def lower_unwrap(self, ctx, obj): ...

    def lower_increment(self, ctx, obj, val): ...

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


class NotationCompiler():
    def __init__(self, ctx):
        self.ctx = ctx
    
    def __call__(self, prgm):
        ctx_2 = NotationContext()
        
        return self.ctx(ctx_2(prgm))

from pprint import pprint

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
        types=None,
        func_state=None,
    ):
        super().__init__(namespace=namespace, preamble=preamble, epilogue=epilogue)
        if bindings is None:
            bindings = ScopedDict()
        if slots is None:
            slots = ScopedDict()
        if types is None:
            types = ScopedDict()
        self.bindings = bindings
        self.slots = slots
        self.types = types
        self.func_state = func_state

    def block(self):
        """
        Create a new block. Preambles and epilogues will stay within this block.
        This is used to create a new context for compiling a block of code.
        """
        blk = super().block()
        blk.bindings = self.bindings
        blk.slots = self.slots
        blk.types = self.types
        blk.func_state = self.func_state
        return blk

    def scope(self):
        """
        Create a new scoped context that inherits from this one.
        """
        blk = self.block()
        blk.bindings = self.bindings.scope()
        blk.slots = self.slots.scope()
        blk.types = self.types.scope()
        return blk

    def should_halt(self):
        """
        Check if the current function should halt.
        This is used to determine if the function has returned.
        """
        return self.func_state.has_returned

    def emit(self):
        return self.preamble + self.epilogue

    def resolve(self, node):
        match node:
            case ntn.Slot(var_n, var_t):
                if var_n in self.slots:
                    var_o = self.slots[var_n]
                    return ntn.Stack(var_o, var_t)
                raise KeyError(f"Slot {var_n} not found in context")
            case ntn.Stack(_, _):
                return node
            case _:
                raise ValueError(f"Expected Slot or Stack, got: {type(node)}")

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
                return asm.Call(f_e, args_e)
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
                var = asm.Variable(var_n, var_t)
                self.exec(asm.Assign(var, val_code))
                self.types[var_n] = var_t
                self.slots[var_n] = var_t.asm_unpack(
                    self, var_n, ntn.Variable(var_n, var_t)
                )
                return None
            case ntn.Repack(ntn.Slot(var_n, var_t)):
                if var_n not in self.slots or var_n not in self.types:
                    raise KeyError(f"Slot {var_n} not found in context, cannot repack")
                if var_t != self.types[var_n]:
                    raise TypeError(f"Type mismatch: {var_t} != {self.types[var_n]}")
                obj = self.slots[var_n]
                var_t.asm_repack(self, var_n, obj)
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
                #first instantiate tensors
                ext.format.lower_loop(self, idx, body)
                return None
            case ntn.Declare(tns, init, op, shape):
                tns = self.resolve(tns)
                init_e = self(init)
                op_e = self(op)
                shape_e = [self(s) for s in shape]
                return tns.result_format.lower_declare(self, tns, init_e, op_e, shape_e)
            case ntn.Freeze(tns, op):
                tns = self.resolve(tns)
                op_e = self(op)
                return tns.result_format.lower_freeze(tns, op_e)
            case ntn.Thaw(tns, op):
                tns = self.resolve(tns)
                op_e = self(op)
                return tns.result_format.lower_thaw(tns, op_e)
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
                ctx.func_state = HaltState(return_var=asm.Variable(ctx.freshen(f"{func_n}_return"), ret_t))
                blk = ctx.scope()
                blk(body)
                self.exec(
                    asm.Function(
                        asm.Variable(func_n, ret_t),
                        [ctx(arg) for arg in args],
                        asm.Block([
                            *blk.emit(), 
                            asm.Return(ctx.func_state.return_var)
                        ]),
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
                return asm.Module(ctx.emit())

def get_undeclared_slots(prgm):
    undeclared = set()
    for node in PostOrderDFS(prgm):
        match node:
            case ntn.Declare(ntn.Slot(tns_n, _), _, _, _):
                undeclared.add(tns_n)


def instantiate_tns(ctx, tns, mode, undeclared = set()):
    match tns:
        case ntn.Slot(tns_n, tns_t):
            if tns_n in undeclared:
                tns = ctx.resolve(tns_n)
                tns_2 = tns_t.lower_instantiate(ctx, tns, mode)
                return tns_2
    return tns

def instantiate(ctx, prgm):
    undeclared = get_undeclared_slots(prgm)
    def instantiate_node(node):
        match node:
            case ntn.Access(tns, mode, idxs):
                return ntn.Access(
                    instantiate_tns(ctx, tns, mode),
                    mode,
                    idxs,
                )
            case ntn.Increment(tns, val):
                return ntn.Increment(
                    instantiate_tns(ctx, tns, ntn.Update()),
                    val,
                )
            case ntn.Unwrap(tns):
                return ntn.Unwrap(
                    instantiate_tns(ctx, tns, ntn.Read())
                )
            case _:
                return None
    prgm = PostWalk(instantiate_node, prgm)


def lower_looplets(ctx, idx, ext, body):
    body = instantiate(ctx, body)
    ctx_2 = ctx.scope()
    def unfurl_node(node):
        match node:
            case ntn.Access(tns, mode, (j, *idxs)):
                if j == idx:
                    tns = ctx_2.resolve(tns)
                    tns_2 = tns.result_type.unfurl(
                        ctx_2,
                        tns,
                        ext,
                        mode,
                    )
                    return ntn.Access(tns_2, mode, (j, *idxs))
        return None
    body = PostWalk(unfurl_node, body)
    ctx_3 = LoopletContext(ctx, idx)
    ctx_3(ext, body)

class LoopletPass(ABC):
    @property
    @abstractmethod
    def priority(self):
        ...

    def lt(self, other):
        if other is None:
            return False
        assert isinstance(other, LoopletPass)
        return self.priority < other.priority


class DefaultPass(LoopletPass):
    @property
    def priority(self):
        return -Inf

class LookupPass(LoopletPass):
    @property
    def priority(self):
        return 0
    
    def __call__(self, ctx, idx, ext, body):
        idx_2 = asm.Variable(self.freshen("i"), idx.result_type)
        def lookup_node(node):
            match node:
                case ntn.Access(tns, mode, (j, *idxs)):
                    if j == idx:
                        tns = ctx.resolve(tns)
                        tns_2 = tns.result_type.lookup(
                            ctx,
                            tns,
                            idx_2,
                        )
                        return ntn.Access(tns_2, mode, (j, *idxs))
            return None
        body_2 = PostWalk(lookup_node)(body)
        ctx.exec(
            asm.ForLoop(
                idx_2,
                asm.Literal(ext.start),
                asm.Literal(ext.end),
                ctx(body_2)
            )
        )


class LoopletContext(Context):
    def __init__(self, ctx, idx):
        self.ctx = ctx
        self.idx = idx

    def freshen(self, *tags):
        return self.ctx.freshen(*tags)
    
    def resolve(self, *names: str):
        return self.ctx.resolve(*names)
    
    def exec(self, thunk: Any):
        self.ctx.exec(thunk)

    def post(self, thunk: Any):
        self.ctx.post(thunk)

    def scope(self):
        blk = self.ctx.scope()
        return LoopletContext(blk, self.idx)

    def select_pass(self, body):
        def pass_request(node):
            match node:
                case ntn.Access(tns, _, (j, *_)):
                    if j == self.idx:
                        return tns.pass_request()
            return None

        return max(map(pass_request, PostOrderDFS(body)))

    def __call__(self, ext, body):
        """
        Lower a looplet with the given index and body.
        This is used to compile the looplet into assembly.
        """
        pass_ = self.select_pass(body)
        if pass_ is None:
            ctx_2 = self.ctx.scope()
            ctx_2(body)
            return ctx_2.emit()
        else:
            return pass_(self.ctx, self.idx, ext, body)