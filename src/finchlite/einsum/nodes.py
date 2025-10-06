import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self, cast

from finchlite.algebra import (
    overwrite,
    promote_max,
    promote_min,
)
from finchlite.symbolic import Context, Term, TermTree
from finchlite.util.print import qual_str


class EinsumNode(Term):
    @classmethod
    def head(cls):
        """Returns the head of the node."""
        return cls

    @classmethod
    def make_term(cls, head, *children: Term) -> Self:
        return head.from_children(*children)

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        return cls(*children)

    def __str__(self):
        """Returns a string representation of the node."""
        ctx = EinsumPrinterContext()
        res = ctx(self)
        return res if res is not None else ctx.emit()


class EinsumTree(EinsumNode, TermTree):
    """
    EinsumExpr

    Represents a pointwise expression in the Einsum IR
    """

    @property
    @abstractmethod
    def children(self) -> list[EinsumNode]:  # type: ignore[override]
        ...


class EinsumExpr(EinsumNode, ABC):
    @abstractmethod
    def get_idxs(self) -> set["Index"]:
        pass



@dataclass(eq=True, frozen=True)
class Literal(EinsumExpr):
    """
    Literal
    """

    val: Any

    def __hash__(self):
        return hash(self.val)

    def __eq__(self, other):
        return isinstance(other, Literal) and self.val == other.val

    def get_idxs(self) -> set["Index"]:
        return set()

@dataclass(eq=True, frozen=True)
class Index(EinsumExpr):
    """
    Represents a  AST expression for an index named `name`.

    Attributes:
        name: The name of the index.
    """

    name: str

    def get_idxs(self) -> set["Index"]:
        return {self}


@dataclass(eq=True, frozen=True)
class Alias(EinsumExpr):
    """
    Represents a  AST expression for an index named `name`.

    Attributes:
        name: The name of the index.
    """

    name: str

    def get_idxs(self) -> set["Index"]:
        return {self}


@dataclass(eq=True, frozen=True)
class Access(EinsumExpr, EinsumTree):
    """
    Access

    Tensor access like a[i, j].

    Attributes:
        tensor: The tensor to access.
        idxs: The indices at which to access the tensor.
    """

    tns: EinsumExpr
    idxs: tuple[EinsumExpr, ...]  # (Field('i'), Field('j'))
    # Children: None (leaf)

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # First child is tns, rest are indices
        if len(children) < 1:
            raise ValueError("Access expects at least 1 child")
        tns = cast(EinsumExpr, children[0])
        idxs = cast(tuple[EinsumExpr, ...], children[1:])
        return cls(tns, tuple(idxs))

    @property
    def children(self):
        return [self.tns, *self.idxs]

    def get_idxs(self) -> set["Index"]:
        idxs = set()
        for idx in self.idxs:
            idxs.update(idx.get_idxs())
        return idxs


@dataclass(eq=True, frozen=True)
class Call(EinsumExpr, EinsumTree):
    """
    Call

    Represents an operation like + or * on pointwise expressions for multiple operands.
    If operation is not commutative, pointwise node must be binary, with 2 args at most.

    Attributes:
        op: The function to apply e.g.,
            operator.add, operator.mul, operator.subtract, operator.div, etc...
            Must be a callable.
        args: The arguments to the operation.
    """

    op: Literal  # the function to apply e.g., operator.add
    args: tuple[EinsumExpr, ...]  # Subtrees
    # input_fields: tuple[tuple[Field, ...], ...]
    # Children: The args

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # First child is op, rest are args
        if len(children) < 2:
            raise ValueError("Call expects at least 2 children (op + 1 arg)")
        op = cast(Literal, children[0])
        args = cast(tuple[EinsumExpr, ...], children[1:])
        return cls(op, tuple(args))

    @property
    def children(self):
        return [self.op, *self.args]

    def get_idxs(self) -> set["Index"]:
        idxs = set()
        for arg in self.args:
            idxs.update(arg.get_idxs())
        return idxs


@dataclass(eq=True, frozen=True)
class Einsum(EinsumTree):
    """
    Einsum

    A einsum operation that maps pointwise expressions and aggregates them.

    Attributes:
        op: The function to apply to the pointwise expressions
                    (e.g. +=, f=, max=, etc...). Must be a callable.

        idxs: The indices that are used in the output
                    (i.e. i, j).

        arg: The pointwise expression that
                    is mapped and aggregated.
    """

    # technically a reduce operation, much akin to the one in aggregate
    op: Callable
    tns: Alias
    idxs: tuple[EinsumExpr, ...]
    arg: EinsumExpr

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # Expecting exactly 4 children
        if len(children) != 4:
            raise ValueError(f"Einsum expects 4 children, got {len(children)}")
        op = cast(Callable, children[0])
        tns = cast(Alias, children[1])
        idxs = cast(tuple[EinsumExpr, ...], children[2])
        arg = cast(EinsumExpr, children[3])
        return cls(op, tns, idxs, arg)

    @property
    def children(self):
        return [self.op, self.tns, self.idxs, self.arg]


@dataclass(eq=True, frozen=True)
class Plan(EinsumTree):
    """
    Plan

    A plan that contains einsum operations.
    Basically a list of einsums and some return values.
    """

    bodies: tuple[Einsum, ...] = ()
    returnValues: tuple[EinsumExpr, ...] = ()

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        # The last child is the returnValues tuple, all others are bodies
        if len(children) < 1:
            raise ValueError("Plan expects at least 1 child")
        *bodies, returnValues = children

        return cls(
            tuple(cast(Einsum, b) for b in bodies),
            cast(tuple[EinsumExpr, ...], returnValues),
        )

    @property
    def children(self):
        return [*self.bodies, self.returnValues]

    def __str__(self):
        ctx = EinsumPrinterContext()
        return ctx(self)


@dataclass(eq=True, frozen=True)
class Produces(EinsumTree):
    """
    Represents a logical AST statement that returns `args...` from the current plan.
    Halts execution of the program.

    Attributes:
        args: The arguments to return.
    """

    args: tuple[EinsumNode, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [*self.args]

    @classmethod
    def from_children(cls, *args):
        return cls(args)


infix_strs = {
    overwrite: "",
    operator.add: "+",
    operator.sub: "-",
    operator.mul: "*",
    operator.truediv: "/",
    operator.mod: "%",
    operator.pow: "**",
    operator.and_: "&",
    operator.or_: "|",
    operator.xor: "^",
    operator.floordiv: "//",
    operator.mod: "%",
    operator.pow: "**",
    promote_max: "max",
    promote_min: "min",
}


unary_strs = {
    operator.add: "+",
    operator.pos: "+",
    operator.sub: "-",
    operator.neg: "-",
    operator.invert: "~",
}


class EinsumPrinterContext(Context):
    def __init__(self, tab="    ", indent=0):
        super().__init__()
        self.tab = tab
        self.indent = indent

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def block(self):
        blk = super().block()
        blk.indent = self.indent
        blk.tab = self.tab
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        return blk

    def __call__(self, prgm: EinsumNode):
        feed = self.feed
        match prgm:
            case Literal(value):
                return qual_str(value).replace("\n", "")
            case Alias(name):
                return str(name)
            case Access(tns, idxs):
                return f"{self(tns)}[{', '.join(self(idx) for idx in idxs)}]"
            case Call(fn, args):
                args_e = tuple(self(arg) for arg in args)
                if len(args) == 2 and fn.val in infix_strs:
                    return f"({args_e[0]} {infix_strs[fn.val]} {args_e[1]})"
                if len(args) == 1 and fn.val in unary_strs:
                    return f"{unary_strs[fn.val]}{args_e[0]}"
                return f"{self(fn)}({', '.join(args_e)})"
            case Einsum(op, tns, idxs, arg):
                op_str = infix_strs.get(op, op.__name__)
                self.exec(
                    f"{self.feed}{self(tns)}["
                    f"{', '.join(self(idx) for idx in idxs)}] "
                    f"{op_str}= {self(arg)}"
                )
                return None
            case Plan(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case Produces(args):
                args = tuple(self(arg) for arg in args)
                self.exec(f"{feed}return {args}\n")
                return None
            case str(label):
                return label
            case _:
                raise ValueError(f"Unknown expression type: {type(prgm)}")
