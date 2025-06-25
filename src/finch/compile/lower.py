from dataclasses import dataclass
from typing import Any

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import query_property
from ..finch_notation import TensorView
from ..symbolic import Context, ScopedDict
from . import looplets as lpl


def lower_unwrap(tns):
    """
    Unwrap a tensor view to get the underlying tensor.
    This is used to get the original tensor from a tensor view.
    """
    if hasattr(tns, "lower_unwrap"):
        return tns.unwrap()
    return query_property(tns, "lower_unwrap", "__attr__")


def lower_increment(tns, val):
    """
    Increment a tensor view with an operation and value.
    This updates the tensor at the specified index with the operation and value.
    """
    if hasattr(tns, "lower_increment"):
        return tns.lower_increment(val)
    return query_property(tns, "lower_increment", "__attr__", val)


def lower_declare(tns, init, op, shape):
    """
    Declare a tensor.
    """
    if hasattr(tns, "lower_declare"):
        return tns.declare(init, op, shape)
    return query_property(tns, "lower_declare", "__attr__", init, op, shape)


def lower_freeze(tns, op):
    """
    Freeze a tensor.
    """
    if hasattr(tns, "lower_freeze"):
        return tns.lower_freeze(op)
    try:
        query_property(tns, "lower_freeze", "__attr__", op)
    except AttributeError:
        return tns


def lower_thaw(tns, op):
    """
    Thaw a tensor.
    """
    if hasattr(tns, "lower_thaw"):
        return tns.lower_thaw(op)
    try:
        return query_property(tns, "lower_thaw", "__attr__", op)
    except AttributeError:
        return tns


def unfurl(ctx, tns, ext, proto):
    fmt = tns.format
    if hasattr(fmt, "unfurl"):
        return fmt.unfurl(ctx, tns, ext, proto)
    return query_property(fmt, "unfurl", "__attr__", ctx, tns, ext, proto)


class TensorViewFormat:
    """
    A format for tensor views that allows unfurling.
    This is used to create a view of a tensor with a specific extent.
    """

    def __init__(self, tns, idxs, op):
        self.tns = tns
        self.idxs = idxs
        self.op = op

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
        func_state=None,
    ):
        super().__init__(namespace=namespace, preamble=preamble, epilogue=epilogue)
        if bindings is None:
            bindings = ScopedDict()
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
