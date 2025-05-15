from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Never, Self, TypeVar

from ..symbolic import Term

__all__ = [
    "LogicNode",
    "LogicExpression",
    "Immediate",
    "Deferred",
    "Field",
    "Alias",
    "Table",
    "MapJoin",
    "Aggregate",
    "Reorder",
    "Relabel",
    "Reformat",
    "Subquery",
    "Query",
    "Produces",
    "Plan",
]


T = TypeVar("T", bound="LogicNode")


@dataclass(eq=True, frozen=True)
class LogicNode(Term):
    """
    LogicNode

    Represents a Finch Logic IR node. Finch uses a variant of Concrete Field Notation
    as an intermediate representation.

    The LogicNode struct represents many different Finch IR nodes. The nodes are
    differentiated by a `FinchLogic.LogicNodeKind` enum.
    """

    @staticmethod
    @abstractmethod
    def is_stateful():
        """Determines if the node is stateful."""

    @classmethod
    def head(cls) -> type[Self]:
        """Returns the head of the node."""
        return cls

    @classmethod
    def make_term(cls, head: Callable[..., Self], *args: Term) -> Self:
        """Creates a term with the given head and arguments."""
        return head(*args)


@dataclass(eq=True, frozen=True)
class LogicExpression(LogicNode):
    @abstractmethod
    def get_fields(self) -> list[Field]:
        """Get this node's fields."""

    @abstractmethod
    def children(self) -> list[LogicNode]:
        """Get this node's children."""


@dataclass(eq=True, frozen=True)
class LogicLeaf(LogicNode):
    pass


@dataclass(eq=True, frozen=True)
class Immediate(LogicLeaf):
    """
    Represents a logical AST expression for the literal value `val`.

    Attributes:
        val: The literal value.
    """

    val: Any

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    @property
    def fill_value(self):
        from ..algebra import fill_value

        return fill_value(self)

    def children(self) -> Never:
        raise TypeError(f"`{type(self).__name__}` doesn't support `.children()`.")

    def get_fields(self) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return ()


@dataclass(eq=True, frozen=True)
class Deferred(LogicLeaf):
    """
    Represents a logical AST expression for an expression `ex` of type `type`,
    yet to be evaluated.

    Attributes:
        ex: The expression to be evaluated.
        type_: The type of the expression.
    """

    ex: Any
    type_: Any

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self) -> list[Any]:
        """Returns the children of the node."""
        return [self.ex, self.type_]


@dataclass(eq=True, frozen=True)
class Field(LogicLeaf):
    """
    Represents a logical AST expression for a field named `name`.
    Fields are used to name the dimensions of a tensor. The named
    tensor is referred to as a "table".

    Attributes:
        name: The name of the field.
    """

    name: str

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self) -> list[str]:
        """Returns the children of the node."""
        return [self.name]


@dataclass(eq=True, frozen=True)
class Alias(LogicLeaf):
    """
    Represents a logical AST expression for an alias named `name`. Aliases are used to
    refer to tables in the program.

    Attributes:
        name: The name of the alias.
    """

    name: str

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self) -> list[str]:
        """Returns the children of the node."""
        return [self.name]


@dataclass(eq=True, frozen=True)
class Table(LogicExpression):
    """
    Represents a logical AST expression for a tensor object `tns`, indexed by fields
    `idxs...`. A table is a tensor with named dimensions.

    Attributes:
        tns: The tensor object.
        idxs: The fields indexing the tensor.
    """

    tns: Immediate
    idxs: tuple[Field, ...]

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self) -> list[LogicLeaf]:  # type: ignore[override]
        """Returns the children of the node."""
        return [self.tns, *self.idxs]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        return [*self.idxs]

    @classmethod
    def make_term(  # type: ignore[override]
        cls,
        head: Callable[[Immediate, tuple[Field, ...]], Self],
        tns: Immediate,
        *idxs: Field,
    ) -> Self:
        return head(tns, idxs)


@dataclass(eq=True, frozen=True)
class MapJoin(LogicExpression):
    """
    Represents a logical AST expression for mapping the function `op` across `args...`.
    Dimensions which are not present are broadcasted. Dimensions which are
    present must match.  The order of fields in the mapjoin is
    `unique(vcat(map(getfields, args)...))`

    Attributes:
        op: The function to map.
        args: The arguments to map the function across.
    """

    op: Immediate
    args: tuple[LogicExpression, ...]

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self) -> list[LogicNode]:
        """Returns the children of the node."""
        return [self.op, *self.args]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        # (mtsokol) I'm not sure if this comment still applies - the order is preserved.
        # TODO: this is wrong here: the overall order should at least be concordant with
        # the args if the args are concordant
        fs: list[Field] = []
        for arg in self.args:
            fs.extend(arg.get_fields())

        return list(dict.fromkeys(fs))

    @classmethod
    def make_term(  # type: ignore[override]
        cls,
        head: Callable[[Immediate, tuple[LogicExpression, ...]], Self],
        op: Immediate,
        *args: LogicExpression,
    ) -> Self:
        return head(op, tuple(args))


@dataclass(eq=True, frozen=True)
class Aggregate(LogicExpression):
    """
    Represents a logical AST statement that reduces `arg` using `op`, starting
    with `init`.  `idxs` are the dimensions to reduce. May happen in any order.

    Attributes:
        op: The reduction operation.
        init: The initial value for the reduction.
        arg: The argument to reduce.
        idxs: The dimensions to reduce.
    """

    op: Immediate
    init: Immediate
    arg: LogicExpression
    idxs: tuple[Field, ...]

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self) -> list[LogicNode]:
        """Returns the children of the node."""
        return [self.op, self.init, self.arg, *self.idxs]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        return [field for field in self.arg.get_fields() if field not in self.idxs]

    @classmethod
    def make_term(  # type: ignore[override]
        cls,
        head: Callable[
            [Immediate, Immediate, LogicExpression, tuple[Field, ...]], Self
        ],
        op: Immediate,
        init: Immediate,
        arg: LogicExpression,
        *idxs: Field,
    ) -> Self:
        return head(op, init, arg, idxs)


@dataclass(eq=True, frozen=True)
class Reorder(LogicExpression):
    """
    Represents a logical AST statement that reorders the dimensions of `arg` to be
    `idxs...`. Dimensions known to be length 1 may be dropped. Dimensions that do not
    exist in `arg` may be added.

    Attributes:
        arg: The argument to reorder.
        idxs: The new order of dimensions.
    """

    arg: LogicNode
    idxs: tuple[Field, ...]

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self) -> list[LogicNode]:
        """Returns the children of the node."""
        return [self.arg, *self.idxs]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        return [*self.idxs]

    @classmethod
    def make_term(  # type: ignore[override]
        cls,
        head: Callable[[LogicNode, tuple[Field, ...]], Self],
        arg: LogicNode,
        *idxs: Field,
    ) -> Self:
        return head(arg, tuple(idxs))


@dataclass(eq=True, frozen=True)
class Relabel(LogicExpression):
    """
    Represents a logical AST statement that relabels the dimensions of `arg` to be
    `idxs...`.

    Attributes:
        arg: The argument to relabel.
        idxs: The new labels for dimensions.
    """

    arg: LogicNode
    idxs: tuple[Field, ...]

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self) -> list[LogicNode]:
        """Returns the children of the node."""
        return [self.arg, *self.idxs]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        return [*self.idxs]


@dataclass(eq=True, frozen=True)
class Reformat(LogicExpression):
    """
    Represents a logical AST statement that reformats `arg` into the tensor `tns`.

    Attributes:
        tns: The target tensor.
        arg: The argument to reformat.
    """

    tns: Immediate
    arg: LogicExpression

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self) -> list[LogicNode]:
        """Returns the children of the node."""
        return [self.tns, self.arg]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        return self.arg.get_fields()


@dataclass(eq=True, frozen=True)
class Subquery(LogicExpression):
    """
    Represents a logical AST statement that evaluates `rhs`, binding the result to
    `lhs`, and returns `rhs`.

    Attributes:
        lhs: The left-hand side of the binding.
        arg: The argument to evaluate.
    """

    lhs: LogicNode
    arg: LogicExpression

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self) -> list[LogicNode]:
        """Returns the children of the node."""
        return [self.lhs, self.arg]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        return self.arg.get_fields()


@dataclass(eq=True, frozen=True)
class Query(LogicExpression):
    """
    Represents a logical AST statement that evaluates `rhs`, binding the result to
    `lhs`.

    Attributes:
        lhs: The left-hand side of the binding.
        rhs: The right-hand side to evaluate.
    """

    lhs: LogicNode
    rhs: LogicNode

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self) -> list[LogicNode]:
        """Returns the children of the node."""
        return [self.lhs, self.rhs]

    def get_fields(self) -> list[Field]:
        raise NotImplementedError


@dataclass(eq=True, frozen=True)
class Produces(LogicExpression):
    """
    Represents a logical AST statement that returns `args...` from the current plan.
    Halts execution of the program.

    Attributes:
        args: The arguments to return.
    """

    args: tuple[LogicNode, ...]

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self) -> list[LogicNode]:
        """Returns the children of the node."""
        return [*self.args]

    def get_fields(self) -> list[Field]:
        raise NotImplementedError

    @classmethod
    def make_term(  # type: ignore[override]
        cls, head: Callable[[tuple[LogicNode, ...]], Self], *args: LogicNode
    ) -> Self:
        return head(tuple(args))


@dataclass(eq=True, frozen=True)
class Plan(LogicExpression):
    """
    Represents a logical AST statement that executes a sequence of statements
    `bodies...`. Returns the last statement.

    Attributes:
        bodies: The sequence of statements to execute.
    """

    bodies: tuple[LogicNode, ...] = ()

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self) -> list[LogicNode]:
        """Returns the children of the node."""
        return [*self.bodies]

    def get_fields(self) -> list[Field]:
        raise NotImplementedError

    @classmethod
    def make_term(  # type: ignore[override]
        cls, head: Callable[[tuple[LogicNode, ...]], Self], *bodies: LogicNode
    ) -> Self:
        return head(tuple(bodies))
