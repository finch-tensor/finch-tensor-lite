from abc import ABC, abstractmethod
from dataclasses import dataclass
from pprint import pprint
from typing import Any

from .. import finch_assembly as asm
from .. import finch_notation as ntn
from ..algebra import TensorFormat
from ..symbolic import Context, PostOrderDFS, PostWalk, Rewrite, ScopedDict


class FinchTensorFormat(TensorFormat, ABC):
    @abstractmethod
    def lower_unwrap(tns):
        """
        Unwrap a tensor view to get the underlying tensor.
        This is used to get the original tensor from a tensor view.
        """

    @abstractmethod
    def lower_increment(tns, val):
        """
        Increment a tensor view with an operation and value.
        This updates the tensor at the specified index with the operation and value.
        """

    @abstractmethod
    def lower_declare(self, ctx, tns, init, op, shape):
        """
        Declare a tensor.
        """

    @abstractmethod
    def lower_freeze(self, ctx, tns, op):
        """
        Freeze a tensor.
        """

    @abstractmethod
    def lower_thaw(self, ctx, tns, op):
        """
        Thaw a tensor.
        """

    @abstractmethod
    def unfurl(ctx, tns, ext, proto): ...


@dataclass(eq=True, frozen=True)
class Extent:
    """
    A class to represent the extent of a loop variable.
    This is used to define the start and end values of a loop.
    """

    start: Any
    end: Any

    def loop(self, ctx, idx, body):
        for idx_e in range(self.start, self.end):
            # Create a new scope for each iteration
            ctx_2 = ctx.scope(loop_state=HaltState())
            # Assign the loop variable
            ctx_2.bindings[idx.name] = idx.type_(idx_e)
            # Execute the body of the loop
            ctx_2(body)


def extent(start, end):
    """
    Create an extent value for a loop.
    """
    return Extent(start, end)


def dimension(tns, mode):
    end = tns.shape[mode]
    return extent(type(end)(0), end)


@dataclass(eq=True, frozen=True)
class ExtentFields:
    start: Any
    end: Any


@dataclass(eq=True, frozen=True)
class SingletonExtent:
    idx: Any

    def loop(self, ctx, idx, body):
        # Create a new scope for each iteration
        ctx_2 = ctx.scope(loop_state=HaltState())
        # Assign the loop variable
        ctx_2.bindings[idx.name] = idx.type_(self.idx)
        # Execute the body of the loop
        ctx_2(body)


class FinchCompileError(Exception):
    """
    Exception raised during Finch compilation.
    This is used to indicate errors in the compilation process.
    """

    def __init__(self, node, message):
        super().__init__(f"{message}:\n{pprint(node)}")
        self.message = message
        self.node = node


@dataclass(eq=True, frozen=True)
class ExtentFormat:
    start: Any
    end: Any

    @classmethod
    def stack(cls, start, end):
        return ntn.Stack(
            ExtentFields(start, end),
            ExtentFormat(start.result_format, end.result_format),
        )

    def get_start(self, ext):
        return asm.GetAttr(ext, "start")

    def get_end(self, ext):
        return asm.GetAttr(ext, "end")

    def lower_loop(self, ctx, idx, ext, body):
        """
        Lower a loop with the given index and body.
        This is used to compile the loop into assembly.
        """
        lower_looplets(ctx, idx, ext, body)
        return

    def default_loop(self, ctx, idx, ext, body):
        def assert_lowered(node):
            match node:
                case ntn.Access(_, _, (j, *_)):
                    if j == idx:
                        raise FinchCompileError(
                            node, f"Access with {j} should have been lowered already"
                        )
            return

        map(assert_lowered, PostOrderDFS(body))

        idx = asm.Variable(ctx.freshen(idx.name), idx.result_format)
        ctx_2 = ctx.scope()
        ctx_2.bindings[idx.name] = idx
        ctx_2(body)
        body_3 = ctx_2.emit()
        ctx.exec(
            asm.ForLoop(
                idx,
                self.get_start(ext),
                self.get_end(ext),
                body_3,
            )
        )
        return


@dataclass(eq=True, frozen=True)
class SingletonExtentFields:
    idx: Any


@dataclass(eq=True, frozen=True)
class SingletonExtentFormat:
    idx: Any

    @classmethod
    def stack(cls, idx):
        return ntn.Stack(
            SingletonExtentFields(idx),
            SingletonExtentFormat(idx.result_format),
        )

    def get_start(self, ext):
        return asm.GetAttr(ext, "idx")

    def get_end(self, ext):
        return asm.GetAttr(ext, "idx")

    def lower_loop(self, ctx, idx, ext, body):
        lower_looplets(ctx, idx, ext, body)
        return

    def default_loop(self, ctx, idx, ext, body):
        def assert_lowered(node):
            match node:
                case ntn.Access(_, _, (j, *_)):
                    if j == idx:
                        raise FinchCompileError(
                            node, f"Access with {j} should have been lowered already"
                        )
            return

        map(assert_lowered, PostOrderDFS(body))

        ctx_2 = ctx.scope()
        ctx_2.bindings[idx.name] = self.get_start(ext)
        ctx_2(body)
        return ctx_2.emit()


@dataclass(eq=True)
class HaltState:
    """
    A class to represent the halt state of a notation program.
    These programs can't break, but calling return sets a special return value.
    """

    has_returned: bool = False
    return_var: Any = None


class NotationCompiler:
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, prgm):
        ctx_2 = NotationContext()

        return self.ctx(ctx_2(prgm))


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
            case ntn.Repack(ntn.Slot(var_n, var_t), _):
                if var_n not in self.slots or var_n not in self.types:
                    raise KeyError(f"Slot {var_n} not found in context, cannot repack")
                if var_t != self.types[var_n]:
                    raise TypeError(f"Type mismatch: {var_t} != {self.types[var_n]}")
                obj = self.slots[var_n]
                var_t.asm_repack(self, var_n, obj)
                return None
            case ntn.Unwrap(ntn.Access(tns, mode, _)):
                assert isinstance(mode, ntn.Read)
                # assert len(idxs) == 0
                tns = self.resolve(tns)
                return tns.result_format.lower_unwrap(self, tns.obj)
            case ntn.Increment(ntn.Access(tns, mode, _), val):
                assert isinstance(mode, ntn.Update)
                # assert len(idxs) == 0
                tns = self.resolve(tns)
                val_e = self(val)
                return tns.result_format.lower_increment(self, tns.obj, val_e)
            case ntn.Block(bodies):
                for body in bodies:
                    self(body)
                return None
            case ntn.Loop(idx, ext, body):
                # first instantiate tensors
                ext.result_format.lower_loop(self, idx, ext, body)
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
                return tns.result_format.lower_freeze(self, tns, op_e)
            case ntn.Thaw(tns, op):
                tns = self.resolve(tns)
                op_e = self(op)
                return tns.result_format.lower_thaw(self, tns, op_e)
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
                ctx.func_state = HaltState(
                    return_var=asm.Variable(ctx.freshen(f"{func_n}_return"), ret_t)
                )
                blk = ctx.scope()
                blk(body)
                self.exec(
                    asm.Function(
                        asm.Variable(func_n, ret_t),
                        [ctx(arg) for arg in args],
                        asm.Block([*blk.emit(), asm.Return(ctx.func_state.return_var)]),
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
    return undeclared


def instantiate_tns(ctx, tns, mode, undeclared=None):
    if undeclared is None:
        undeclared = set()
    match tns:
        case ntn.Slot(tns_n, tns_t):
            if tns_n in undeclared:
                tns = ctx.resolve(tns_n)
                return tns_t.lower_instantiate(ctx, tns, mode)
    return tns


def instantiate(ctx, prgm):
    undeclared = get_undeclared_slots(prgm)

    def instantiate_node(node):
        match node:
            case ntn.Access(tns, mode, idxs):
                if tns not in undeclared:
                    return ntn.Access(
                        instantiate_tns(ctx, tns, mode),
                        mode,
                        idxs,
                    )
        return None

    return Rewrite(PostWalk(instantiate_node))(prgm)


def lower_looplets(ctx, idx, ext, body):
    body = instantiate(ctx, body)
    ctx_2 = ctx.scope()

    def unfurl_node(node):
        match node:
            case ntn.Access(tns, mode, (j, *idxs)):
                if j == idx:
                    tns = ctx_2.resolve(tns)
                    tns_2 = tns.result_format.unfurl(ctx_2, tns, ext, mode, None)
                    return ntn.Access(tns_2, mode, (j, *idxs))
        return None

    body = Rewrite(PostWalk(unfurl_node))(body)
    ctx_3 = LoopletContext(ctx, idx)
    ctx_3(ext, body)


class LoopletPass(ABC):
    @property
    @abstractmethod
    def priority(self): ...

    def __lt__(self, other):
        assert isinstance(other, LoopletPass)
        return self.priority < other.priority


class DefaultPass(LoopletPass):
    @property
    def priority(self):
        return float("-inf")

    def __call__(self, ctx, idx, ext, body):
        """
        Default pass that does nothing. This is used when no other pass is selected.
        """
        ext.result_format.default_loop(ctx, idx, ext, body)


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

    def emit(self):
        return self.ctx.emit()

    def select_pass(self, body):
        def pass_request(node):
            match node:
                case ntn.Access(tns, _, (j, *_)):
                    if j == self.idx:
                        return tns.pass_request
            return DefaultPass()

        return max(map(pass_request, PostOrderDFS(body)))

    def __call__(self, ext, body):
        pass_ = self.select_pass(body)
        if pass_ is None:
            ctx_2 = self.ctx.scope()
            ctx_2(body)
            return ctx_2.emit()
        pass_(self, self.idx, ext, body)
        return None
