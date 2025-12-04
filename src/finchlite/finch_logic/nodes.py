from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Self, TypeVar

from finchlite.algebra.algebra import fixpoint_type, return_type

from ..symbolic import (
    Context,
    FType,
    FTyped,
    NamedTerm,
    Term,
    TermTree,
    ftype,
    literal_repr,
)
from ..util import qual_str

"""
Notes on Finch Logic IR:
Finch Logic IR is an intermediate representation (IR) used in the Finch Tensor
Lite framework. It is designed to represent logical operations and transformations
on tensors in a structured and abstract manner. The Logic IR is built around the
concept of "tables," which are tensors with named dimensions, allowing for more
intuitive manipulation and reasoning about tensor operations.

Dimensions in Finch Logic IR are represented using "fields," which are named
entities that index the dimensions of a tensor. This naming convention helps
clarify the relationships between different tensors and their dimensions during
operations such as mapping, aggregation, reordering, and relabeling.

Fields may not be used to represent different dimension sizes within the same
logic program.

Tables may be referenced using "aliases," which are symbolic names that refer to
specific tables within the program. Evaluators for Finch logic may accept a list
of bindings from aliases to tables, allowing the logic program to modify the
state of tensors. Queries in Finch Logic IR can bind the result of an expression
to an alias for later use, or update the value of an existing alias.
"""


@dataclass(eq=True, frozen=True)
class TableValueFType(FType):
    tns: Any
    idxs: tuple[Field, ...]

    def __eq__(self, other):
        if not isinstance(other, TableValueFType):
            return False
        return self.tns == other.tns and self.idxs == other.idxs

    def __hash__(self):
        return hash((self.tns, self.idxs))


@dataclass(frozen=True)
class TableValue(FTyped):
    tns: Any
    idxs: tuple[Field, ...]

    @property
    def ftype(self):
        return TableValueFType(ftype(self.tns), self.idxs)

    def __post_init__(self):
        if isinstance(self.tns, TableValue):
            raise ValueError("The tensor (tns) cannot be a TableValue")

    def __eq__(self, other):
        if not isinstance(other, TableValue):
            return False
        return (self.tns == other.tns).all() and self.idxs == other.idxs


@dataclass(eq=True, frozen=True)
class LogicNode(Term, ABC):
    """
    LogicNode

    Represents a Finch Logic IR node. Finch uses a variant of Concrete Field Notation
    as an intermediate representation.

    The LogicNode struct represents many different Finch IR nodes. The nodes are
    differentiated by a `FinchLogic.LogicNodeKind` enum.
    """

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
        ctx = LogicPrinterContext()
        res = ctx(self)
        return res if res is not None else ctx.emit()


@dataclass(eq=True, frozen=True)
class LogicTree(LogicNode, TermTree, ABC):
    @property
    @abstractmethod
    def children(self) -> list[LogicNode]:  # type: ignore[override]
        ...


T = TypeVar("T")


class LogicExpression(LogicNode):
    """
    Logic AST expression base class.

    A Logic expression is a program node which evaluates to a TableValue, a
    tensor with named dimensions.
    """

    @abstractmethod
    def fields(
        self, bindings: dict[Alias, tuple[Field, ...]] | None = None
    ) -> tuple[Field, ...]:
        """Returns fields of the node."""
        ...

    @abstractmethod
    def mapdims(
        self,
        op: Callable,
        dim_bindings: dict[Alias, tuple[T | None, ...]],
        field_bindings: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> tuple[T | None, ...]:
        """Compute per-dimension values, combining using `op`. When dimensions
        are expanded, None is used.  When dimensions are contracted, the value
        is combined with None."""
        ...

    @abstractmethod
    def element_type(
        self, bindings: dict[Alias, Any] | None = None
    ) -> Any:  # In the future should be FType
        """Returns element type of the node."""
        ...

    @abstractmethod
    def fill_value(self, bindings: dict[Alias, Any] | None = None) -> Any:
        """Returns fill value of the node."""
        ...


class LogicStatement(LogicNode):
    """
    Logic AST statement base class.

    A Logic statement may modify the state of the machine by assigning table
    values to Aliases. Logic statements evaluate to a tuple of table values.
    """


@dataclass(eq=True, frozen=True)
class Literal(LogicNode):
    """
    Represents a logical AST expression for the literal value `val`.

    Attributes:
        val: The literal value.
    """

    val: Any

    def __hash__(self):
        try:
            return hash(self.val)
        except TypeError:
            return hash(id(self.val))

    def __eq__(self, value):
        if not isinstance(value, Literal):
            return False
        # For consistency with __hash__, we fall back to pointer equality
        # when the value is unhashable
        try:
            hash(value.val)
            hash(self.val)
            return self.val == value.val
        except TypeError:
            return id(self.val) == id(value.val)

    def __repr__(self) -> str:
        return literal_repr(type(self).__name__, asdict(self))


@dataclass(eq=True, frozen=True)
class Value(LogicNode):
    """
    Represents a logical AST expression for an expression `ex` of type `type`,
    yet to be evaluated.

    Attributes:
        ex: The expression to be evaluated.
        type_: The type of the expression.
    """

    ex: Any
    type_: Any


@dataclass(eq=True, frozen=True)
class Field(LogicNode, NamedTerm):
    """
    Represents a logical AST expression for a field named `name`.
    Fields are used to name the dimensions of a tensor. The named
    tensor is referred to as a "table".

    Attributes:
        name: The name of the field.
    """

    name: str

    @property
    def symbol(self) -> str:
        return self.name


@dataclass(eq=True, frozen=True)
class Alias(LogicExpression, NamedTerm):
    """
    Represents a logical AST expression for an alias named `name`. Aliases are used to
    refer to tables in the program.

    Attributes:
        name: The name of the alias.
    """

    name: str

    @property
    def symbol(self) -> str:
        return self.name

    def fields(
        self, bindings: dict[Alias, tuple[Field, ...]] | None = None
    ) -> tuple[Field, ...]:
        """Returns fields of the node."""
        if bindings is None or self not in bindings:
            raise NotImplementedError("Cannot resolve fields of Alias {self.name}")
        return bindings[self]

    def mapdims(
        self,
        op: Callable,
        dim_bindings: dict[Alias, tuple[T | None, ...]],
        field_bindings: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> tuple[T | None, ...]:
        if dim_bindings is None or self not in dim_bindings:
            raise NotImplementedError("Cannot resolve dims of Alias {self.name}")
        return dim_bindings[self]

    def element_type(self, bindings: dict[Alias, Any] | None = None) -> Any:
        """Returns element type of the node."""
        if bindings is None or self not in bindings:
            raise NotImplementedError(
                "Cannot resolve element_type of Alias {self.name}"
            )
        return bindings[self]

    def fill_value(self, bindings: dict[Alias, Any] | None = None) -> Any:
        """Returns fill value of the node."""
        if bindings is None or self not in bindings:
            raise NotImplementedError("Cannot resolve fill_value of Alias {self.name}")
        return bindings[self]


@dataclass(eq=True, frozen=True)
class Table(LogicTree, LogicExpression):
    """
    Represents a logical AST expression for a tensor object `tns`, indexed by fields
    `idxs...`. A table is a tensor with named dimensions.

    Attributes:
        tns: The tensor object.
        idxs: The fields indexing the tensor.
    """

    tns: Literal | Value
    idxs: tuple[Field, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.tns, *self.idxs]

    def fields(
        self, bindings: dict[Alias, tuple[Field, ...]] | None = None
    ) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return self.idxs

    def mapdims(
        self,
        op: Callable,
        dim_bindings: dict[Alias, tuple[T | None, ...]],
        field_bindings: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> tuple[T | None, ...]:
        raise NotImplementedError("Cannot resolve dims of Tables")

    def element_type(self, bindings: dict[Alias, Any] | None = None) -> Any:
        """Returns element type of the node."""
        if isinstance(self.tns, Literal):
            return ftype(self.tns.val.element_type)
        if isinstance(self.tns, Value):
            return self.tns.type_.element_type
        raise ValueError(f"Unknown tensor type: {type(self.tns)}")

    def fill_value(self, bindings: dict[Alias, Any] | None = None) -> Any:
        """Returns fill value of the node."""
        if isinstance(self.tns, Literal):
            return self.tns.val.fill_value
        if isinstance(self.tns, Value):
            return self.tns.type_.fill_value
        raise ValueError(f"Unknown tensor type: {type(self.tns)}")

    @classmethod
    def from_children(cls, tns, *idxs):
        return cls(tns, idxs)


@dataclass(eq=True, frozen=True)
class MapJoin(LogicTree, LogicExpression):
    """
    Represents a logical AST expression for mapping the function `op` across `args...`.
    Dimensions which are not present are broadcasted. Dimensions which are
    present must match.  The order of fields in the mapjoin is
    `unique(vcat(map(getfields, args)...))`

    Attributes:
        op: The function to map.
        args: The arguments to map the function across.
    """

    op: Literal
    args: tuple[LogicExpression, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.op, *self.args]

    def fields(
        self, bindings: dict[Alias, tuple[Field, ...]] | None = None
    ) -> tuple[Field, ...]:
        """Returns fields of the node."""
        args_fields = [x.fields(bindings) for x in self.args]
        return tuple(dict.fromkeys([f for fs in args_fields for f in fs]))

    def mapdims(
        self,
        op: Callable,
        dim_bindings: dict[Alias, tuple[T | None, ...]],
        field_bindings: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> tuple[T | None, ...]:
        arg_dims: dict[Field, T | None] = {}
        for arg in self.args:
            dims = arg.mapdims(op, dim_bindings, field_bindings)
            fields = arg.fields(field_bindings)
            for idx, dim in zip(fields, dims, strict=True):
                if idx in arg_dims:
                    arg_dims[idx] = op(arg_dims[idx], dim)
                else:
                    arg_dims[idx] = dim
        return tuple(arg_dims[f] for f in self.fields(field_bindings))

    def element_type(self, bindings: dict[Alias, Any] | None = None) -> Any:
        return return_type(
            self.op.val, [arg.element_type(bindings) for arg in self.args]
        )

    def fill_value(self, bindings: dict[Alias, Any] | None = None) -> Any:
        return self.op.val(*[arg.fill_value(bindings) for arg in self.args])

    @classmethod
    def from_children(cls, op, *args):
        return cls(op, args)


@dataclass(eq=True, frozen=True)
class Aggregate(LogicTree, LogicExpression):
    """
    Represents a logical AST statement that reduces `arg` using `op`, starting
    with `init`.  `idxs` are the dimensions to reduce. May happen in any order.

    Attributes:
        op: The reduction operation.
        init: The initial value for the reduction.
        arg: The argument to reduce.
        idxs: The dimensions to reduce.
    """

    op: Literal
    init: Literal
    arg: LogicExpression
    idxs: tuple[Field, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.op, self.init, self.arg, *self.idxs]

    def fields(
        self, bindings: dict[Alias, tuple[Field, ...]] | None = None
    ) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return tuple(
            field for field in self.arg.fields(bindings) if field not in self.idxs
        )

    def mapdims(
        self,
        op: Callable,
        dim_bindings: dict[Alias, tuple[T | None, ...]],
        field_bindings: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> tuple[T | None, ...]:
        idxs = self.arg.fields(field_bindings)
        dims = self.arg.mapdims(op, dim_bindings, field_bindings)
        return tuple(
            val for idx, val in zip(idxs, dims, strict=True) if idx not in self.idxs
        )

    def element_type(self, bindings: dict[Alias, Any] | None = None) -> Any:
        return fixpoint_type(self.op, self.init, self.arg.element_type(bindings))

    def fill_value(self, bindings: dict[Alias, Any] | None = None) -> Any:
        return self.init.val

    @classmethod
    def from_children(cls, op, init, arg, *idxs):
        return cls(op, init, arg, idxs)


@dataclass(eq=True, frozen=True)
class Reorder(LogicTree, LogicExpression):
    """
    Represents a logical AST statement that reorders the dimensions of `arg` to be
    `idxs...`. Dimensions known to be length 1 may be dropped. Dimensions that do not
    exist in `arg` may be added.

    Attributes:
        arg: The argument to reorder.
        idxs: The new order of dimensions.
    """

    arg: LogicExpression
    idxs: tuple[Field, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.arg, *self.idxs]

    def fields(
        self, bindings: dict[Alias, tuple[Field, ...]] | None = None
    ) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return self.idxs

    def mapdims(
        self,
        op: Callable,
        dim_bindings: dict[Alias, tuple[T | None, ...]],
        field_bindings: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> tuple[T | None, ...]:
        idxs = self.arg.fields(field_bindings)
        dims = self.arg.mapdims(op, dim_bindings, field_bindings)
        idx_dims = dict(zip(idxs, dims, strict=True))
        for idx in self.idxs:
            if idx not in idxs:
                # when squeezing a dimension, we combine with None
                op(idx_dims[idx], None)
        return tuple(idx_dims.get(f) for f in self.idxs)

    def element_type(self, bindings: dict[Alias, Any] | None = None) -> Any:
        return self.arg.element_type(bindings)

    def fill_value(self, bindings: dict[Alias, Any] | None = None) -> Any:
        return self.arg.fill_value(bindings)

    @classmethod
    def from_children(cls, arg, *idxs):
        return cls(arg, idxs)


@dataclass(eq=True, frozen=True)
class Relabel(LogicTree, LogicExpression):
    """
    Represents a logical AST statement that relabels the dimensions of `arg` to be
    `idxs...`.

    Attributes:
        arg: The argument to relabel.
        idxs: The new labels for dimensions.
    """

    arg: LogicExpression
    idxs: tuple[Field, ...]

    def fields(
        self, bindings: dict[Alias, tuple[Field, ...]] | None = None
    ) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return self.idxs

    def mapdims(
        self,
        op: Callable,
        dim_bindings: dict[Alias, tuple[T | None, ...]],
        field_bindings: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> tuple[T | None, ...]:
        return self.arg.mapdims(op, dim_bindings, field_bindings)

    def element_type(self, bindings: dict[Alias, Any] | None = None) -> Any:
        """Returns element type of the node."""
        return self.arg.element_type(bindings)

    def fill_value(self, bindings: dict[Alias, Any] | None = None) -> Any:
        """Returns fill value of the node."""
        return self.arg.fill_value(bindings)

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.arg, *self.idxs]

    @classmethod
    def from_children(cls, arg, *idxs):
        return cls(arg, idxs)


@dataclass(eq=True, frozen=True)
class Reformat(LogicTree, LogicExpression):
    """
    Represents a logical AST statement that reformats `arg` into the tensor `tns`.

    Attributes:
        tns: The target tensor.
        arg: The argument to reformat.
    """

    tns: LogicNode
    arg: LogicExpression

    def fields(
        self, bindings: dict[Alias, tuple[Field, ...]] | None = None
    ) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return self.arg.fields(bindings)

    def mapdims(
        self,
        op: Callable,
        dim_bindings: dict[Alias, tuple[T | None, ...]],
        field_bindings: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> tuple[T | None, ...]:
        return self.arg.mapdims(op, dim_bindings, field_bindings)

    def element_type(self, bindings: dict[Alias, Any] | None = None) -> Any:
        """Returns element type of the node."""
        return self.arg.element_type(bindings)

    def fill_value(self, bindings: dict[Alias, Any] | None = None) -> Any:
        """Returns fill value of the node."""
        return self.arg.fill_value(bindings)

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.tns, self.arg]


@dataclass(eq=True, frozen=True)
class Subquery(LogicTree, LogicExpression):
    """
    Represents a logical AST statement that evaluates `rhs`, binding the result to
    `lhs`, and returns `rhs`.

    Attributes:
        lhs: The left-hand side of the binding.
        arg: The argument to evaluate.
    """

    lhs: Alias
    arg: LogicExpression

    def fields(
        self, bindings: dict[Alias, tuple[Field, ...]] | None = None
    ) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return self.arg.fields(bindings)

    def mapdims(
        self,
        op: Callable,
        dim_bindings: dict[Alias, tuple[T | None, ...]],
        field_bindings: dict[Alias, tuple[Field, ...]] | None = None,
    ) -> tuple[T | None, ...]:
        return self.arg.mapdims(op, dim_bindings, field_bindings)

    def element_type(self, bindings: dict[Alias, Any] | None = None) -> Any:
        """Returns element type of the node."""
        return self.arg.element_type(bindings)

    def fill_value(self, bindings: dict[Alias, Any] | None = None) -> Any:
        """Returns fill value of the node."""
        return self.arg.fill_value(bindings)

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.lhs, self.arg]


@dataclass(eq=True, frozen=True)
class Query(LogicTree, LogicStatement):
    """
    Represents a logical AST statement that evaluates `rhs`, binding the result to
    `lhs`.

    Attributes:
        lhs: The left-hand side of the binding.
        rhs: The right-hand side to evaluate.
    """

    lhs: Alias
    rhs: LogicExpression

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.lhs, self.rhs]


@dataclass(eq=True, frozen=True)
class Produces(LogicTree, LogicStatement):
    """
    Represents a logical AST statement that returns `args...` from the current plan.
    Halts execution of the program.

    Attributes:
        args: The arguments to return.
    """

    args: tuple[LogicExpression, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [*self.args]

    @classmethod
    def from_children(cls, *args):
        return cls(args)


@dataclass(eq=True, frozen=True)
class Plan(LogicTree, LogicStatement):
    """
    Represents a logical AST statement that executes a sequence of statements
    `bodies...`. Returns the last statement.

    Attributes:
        bodies: The sequence of statements to execute.
    """

    bodies: tuple[LogicStatement, ...] = ()

    @property
    def children(self):
        """Returns the children of the node."""
        return [*self.bodies]

    @classmethod
    def from_children(cls, *bodies):
        return cls(bodies)


class LogicPrinterContext(Context):
    def __init__(self, tab="    ", indent=0):
        super().__init__()
        self.tab = tab
        self.indent = indent

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def block(self) -> LogicPrinterContext:
        blk = super().block()
        blk.indent = self.indent
        blk.tab = self.tab
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        return blk

    def __call__(self, prgm: LogicNode):
        feed = self.feed
        match prgm:
            case Literal(value):
                return qual_str(value).replace("\n", "")
            case Value(ex):
                return self(ex)
            case Field(name):
                return str(name)
            case Alias(name):
                return str(name)
            case Table(tns, idxs):
                idxs_e = [self(idx) for idx in idxs]
                return f"Table({self(tns)}, {idxs_e})"
            case MapJoin(op, args):
                args_e = tuple(self(arg) for arg in args)
                return f"MapJoin({self(op)}, {args_e})"
            case Aggregate(op, init, arg, idxs):
                idxs_e = [self(idx) for idx in idxs]
                return f"Aggregate({self(op)}, {self(init)}, {self(arg)}, {idxs_e})"
            case Relabel(arg, idxs):
                idxs_e = [self(idx) for idx in idxs]
                arg = self(arg)
                return f"Relabel({arg}, {idxs_e})"
            case Reorder(arg, idxs):
                idxs_e = [self(idx) for idx in idxs]
                arg = self(arg)
                return f"Reorder({self(arg)}, {idxs_e})"
            case Query(lhs, rhs):
                self.exec(f"{feed}{self(lhs)} = {self(rhs)}")
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
            case Subquery(lhs, arg):
                self.exec(f"{feed}{self(lhs)} = {self(arg)}")
                return self(lhs)
            case str(label):
                return label
            case _:
                raise ValueError(f"Unknown expression type: {type(prgm)}")
