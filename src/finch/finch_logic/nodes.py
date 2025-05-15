from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Self, TypeVar

from ..symbolic import Term

__all__ = [
    "LogicNode",
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
    def is_expr():
        """Determines if the node is expresion."""
        ...

    @staticmethod
    @abstractmethod
    def is_stateful():
        """Determines if the node is stateful."""
        ...

    @classmethod
    def head(cls):
        """Returns the head of the node."""
        return cls

    @abstractmethod
    def children(self) -> list[LogicNode]:
        """Returns the children of the node."""

    @classmethod
    def make_term(cls, *args: Term) -> Self:
        """Creates a term with the given head and arguments."""
        return cls(*args)


@dataclass(eq=True, frozen=True)
class NodeWithFields(LogicNode):
    @abstractmethod
    def get_fields(self) -> tuple[Field, ...]:
        """Get this node's fields."""


@dataclass(eq=True, frozen=True)
class Immediate(NodeWithFields):
    """
    Represents a logical AST expression for the literal value `val`.

    Attributes:
        val: The literal value.
    """

    val: Any

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return False

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    @property
    def fill_value(self):
        from ..algebra import fill_value

        return fill_value(self)

    def children(self):
        raise TypeError(f"`{type(self).__name__}` doesn't support `.children()`.")

    def get_fields(self) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return ()


@dataclass(eq=True, frozen=True)
class Deferred(LogicNode):
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
    def is_expr():
        """Determines if the node is an expression."""
        return False

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.ex, self.type_]


@dataclass(eq=True, frozen=True)
class Field(LogicNode):
    """
    Represents a logical AST expression for a field named `name`.
    Fields are used to name the dimensions of a tensor. The named
    tensor is referred to as a "table".

    Attributes:
        name: The name of the field.
    """

    name: str

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return False

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.name]


@dataclass(eq=True, frozen=True)
class Alias(LogicNode):
    """
    Represents a logical AST expression for an alias named `name`. Aliases are used to
    refer to tables in the program.

    Attributes:
        name: The name of the alias.
    """

    name: str

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return False

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.name]


@dataclass(eq=True, frozen=True)
class Table(NodeWithFields):
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
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.tns, *self.idxs]

    def get_fields(self) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return self.idxs

    @classmethod
    def make_term(cls, tns: Immediate, *idxs: Field) -> Self:  # type: ignore[override]
        return cls(tns, idxs)


@dataclass(eq=True, frozen=True)
class MapJoin(NodeWithFields):
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
    args: tuple[NodeWithFields, ...]

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.op, *self.args]

    def get_fields(self) -> tuple[Field, ...]:
        """Returns fields of the node."""
        # (mtsokol) I'm not sure if this comment still applies - the order is preserved.
        # TODO: this is wrong here: the overall order should at least be concordant with
        # the args if the args are concordant
        fs: list[Field] = []
        for arg in self.args:
            fs.extend(arg.get_fields())

        return tuple(dict.fromkeys(fs))

    @classmethod
    def make_term(cls, op: Immediate, *args: NodeWithFields) -> Self:  # type: ignore[override]
        return cls(op, tuple(args))


@dataclass(eq=True, frozen=True)
class Aggregate(LogicNode):
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
    arg: NodeWithFields
    idxs: tuple[Field, ...]

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.op, self.init, self.arg, *self.idxs]

    def get_fields(self) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return tuple(field for field in self.arg.get_fields() if field not in self.idxs)

    @classmethod
    def make_term(  # type: ignore[override]
        cls, op: Immediate, init: Immediate, arg: NodeWithFields, *idxs: Field
    ) -> Self:
        return cls(op, init, arg, idxs)


@dataclass(eq=True, frozen=True)
class Reorder(NodeWithFields):
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
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.arg, *self.idxs]

    def get_fields(self) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return self.idxs

    @classmethod
    def make_term(cls, arg: LogicNode, *idxs: Field) -> Self:  # type: ignore[override]
        return cls(arg, idxs)


@dataclass(eq=True, frozen=True)
class Relabel(NodeWithFields):
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
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.arg, *self.idxs]

    def get_fields(self) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return self.idxs


@dataclass(eq=True, frozen=True)
class Reformat(NodeWithFields):
    """
    Represents a logical AST statement that reformats `arg` into the tensor `tns`.

    Attributes:
        tns: The target tensor.
        arg: The argument to reformat.
    """

    tns: Immediate
    arg: NodeWithFields

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.tns, self.arg]

    def get_fields(self) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return self.arg.get_fields()


@dataclass(eq=True, frozen=True)
class Subquery(NodeWithFields):
    """
    Represents a logical AST statement that evaluates `rhs`, binding the result to
    `lhs`, and returns `rhs`.

    Attributes:
        lhs: The left-hand side of the binding.
        arg: The argument to evaluate.
    """

    lhs: LogicNode
    arg: NodeWithFields

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.lhs, self.arg]

    def get_fields(self) -> tuple[Field, ...]:
        """Returns fields of the node."""
        return self.arg.get_fields()


@dataclass(eq=True, frozen=True)
class Query(LogicNode):
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
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self):
        """Returns the children of the node."""
        return [self.lhs, self.rhs]


@dataclass(eq=True, frozen=True)
class Produces(LogicNode):
    """
    Represents a logical AST statement that returns `args...` from the current plan.
    Halts execution of the program.

    Attributes:
        args: The arguments to return.
    """

    args: tuple[LogicNode, ...]

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self):
        """Returns the children of the node."""
        return [*self.args]

    @classmethod
    def make_term(cls, *args: LogicNode) -> Self:  # type: ignore[override]
        return cls(tuple(args))


@dataclass(eq=True, frozen=True)
class Plan(LogicNode):
    """
    Represents a logical AST statement that executes a sequence of statements
    `bodies...`. Returns the last statement.

    Attributes:
        bodies: The sequence of statements to execute.
    """

    bodies: tuple[LogicNode, ...] = ()

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self):
        """Returns the children of the node."""
        return tuple(self.bodies)

    @classmethod
    def make_term(cls, *bodies: LogicNode) -> Self:  # type: ignore[override]
        return cls(tuple(bodies))
