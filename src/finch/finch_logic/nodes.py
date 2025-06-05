from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from ..symbolic import Term, TermTree


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


@dataclass(eq=True, frozen=True)
class LogicTree(LogicNode, TermTree, ABC):
    @abstractmethod
    def children(self) -> list[LogicNode]:  # type: ignore[override]
        ...


class LogicExpression(LogicNode):
    @abstractmethod
    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        ...


@dataclass(eq=True, frozen=True)
class Immediate(LogicNode):
    """
    Represents a logical AST expression for the literal value `val`.

    Attributes:
        val: The literal value.
    """

    val: Any

    def __hash__(self):
        val = self.val
        return id(val) if isinstance(val, np.ndarray) else hash(val)

    def __eq__(self, other):
        if not isinstance(other, Immediate):
            return False
        res = self.val == other.val
        return res.all() if isinstance(res, np.ndarray) else res


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


@dataclass(eq=True, frozen=True)
class Alias(LogicNode):
    """
    Represents a logical AST expression for an alias named `name`. Aliases are used to
    refer to tables in the program.

    Attributes:
        name: The name of the alias.
    """

    name: str


@dataclass(eq=True, frozen=True)
class Table(LogicTree, LogicExpression):
    """
    Represents a logical AST expression for a tensor object `tns`, indexed by fields
    `idxs...`. A table is a tensor with named dimensions.

    Attributes:
        tns: The tensor object.
        idxs: The fields indexing the tensor.
    """

    tns: Immediate | Deferred
    idxs: tuple[Field, ...]

    def children(self):
        """Returns the children of the node."""
        return [self.tns, *self.idxs]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        return [*self.idxs]

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

    op: LogicNode
    args: tuple[LogicExpression, ...]

    def children(self):
        """Returns the children of the node."""
        return [self.op, *self.args]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        # (mtsokol) I'm not sure if this comment still applies - the order is preserved.
        # TODO: this is wrong here: the overall order should at least be concordant with
        # the args if the args are concordant
        fields = [f for fs in (x.get_fields() for x in self.args) for f in fs]
        return list(dict.fromkeys(fields))

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

    op: LogicNode
    init: LogicNode
    arg: LogicExpression
    idxs: tuple[LogicNode, ...]

    def children(self):
        """Returns the children of the node."""
        return [self.op, self.init, self.arg, *self.idxs]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        assert isinstance(self.arg, LogicExpression)
        return [field for field in self.arg.get_fields() if field not in self.idxs]

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

    arg: LogicNode
    idxs: tuple[Field, ...]

    def children(self):
        """Returns the children of the node."""
        return [self.arg, *self.idxs]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        return [*self.idxs]

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

    arg: LogicNode
    idxs: tuple[Field, ...]

    def children(self):
        """Returns the children of the node."""
        return [self.arg, *self.idxs]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        return [*self.idxs]

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

    def children(self):
        """Returns the children of the node."""
        return [self.tns, self.arg]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        assert isinstance(self.arg, LogicExpression)
        return self.arg.get_fields()


@dataclass(eq=True, frozen=True)
class Subquery(LogicTree, LogicExpression):
    """
    Represents a logical AST statement that evaluates `rhs`, binding the result to
    `lhs`, and returns `rhs`.

    Attributes:
        lhs: The left-hand side of the binding.
        arg: The argument to evaluate.
    """

    lhs: LogicNode
    arg: LogicNode

    def children(self):
        """Returns the children of the node."""
        return [self.lhs, self.arg]

    def get_fields(self) -> list[Field]:
        """Returns fields of the node."""
        assert isinstance(self.arg, LogicExpression)
        return self.arg.get_fields()


@dataclass(eq=True, frozen=True)
class Query(LogicTree):
    """
    Represents a logical AST statement that evaluates `rhs`, binding the result to
    `lhs`.

    Attributes:
        lhs: The left-hand side of the binding.
        rhs: The right-hand side to evaluate.
    """

    lhs: LogicNode
    rhs: LogicNode

    def children(self):
        """Returns the children of the node."""
        return [self.lhs, self.rhs]


@dataclass(eq=True, frozen=True)
class Produces(LogicTree):
    """
    Represents a logical AST statement that returns `args...` from the current plan.
    Halts execution of the program.

    Attributes:
        args: The arguments to return.
    """

    args: tuple[LogicNode, ...]

    def children(self):
        """Returns the children of the node."""
        return [*self.args]

    @classmethod
    def from_children(cls, *args):
        return cls(args)


@dataclass(eq=True, frozen=True)
class Plan(LogicTree):
    """
    Represents a logical AST statement that executes a sequence of statements
    `bodies...`. Returns the last statement.

    Attributes:
        bodies: The sequence of statements to execute.
    """

    bodies: tuple[LogicNode, ...] = ()

    def children(self):
        """Returns the children of the node."""
        return [*self.bodies]

    @classmethod
    def from_children(cls, *bodies):
        return cls(bodies)
