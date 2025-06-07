from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..algebra import element_type, return_type
from ..symbolic import Term, TermTree


# Base class for all Finch Notation nodes
class FinchNode(Term):
    pass

class FinchTree(FinchNode, TermTree):
    @classmethod
    def make_term(cls, head, *children):
        return head.from_children(*children)

    @classmethod
    def from_children(cls, *children):
        return cls(*children)

class FinchExpression:
    """
    Finch AST expression base class.
    """

    def get_type(self) -> Any:
        """
        Get the type of the expression.
        """


@dataclass(eq=True, frozen=True)
class Literal(FinchNode, FinchExpression):
    """
    Finch AST expression for the literal value `val`.
    """

    val: Any

    def get_type(self):
        return type(self.val)


@dataclass(eq=True, frozen=True)
class Value(FinchNode, FinchExpression):
    """
    Finch AST expression for host code `val` expected to evaluate to a value of
    type `type_`.
    """

    val: Any
    type_: Any

    def get_type(self):
        return self.type_


@dataclass(eq=True, frozen=True)
class Index(FinchNode, FinchExpression):
    """
    Finch AST expression for an index named `name`.
    """

    name: str
    type_: Any = None

    def get_type(self):
        return self.type_


@dataclass(eq=True, frozen=True)
class Variable(FinchNode, FinchExpression):
    """
    Finch AST expression for a variable named `name`.
    """

    name: str
    type_: Any = None

    def get_type(self):
        return self.type_


@dataclass(eq=True, frozen=True)
class Call(FinchTree, FinchExpression):
    """
    Finch AST expression for the result of calling the function `op` on
    `args...`.
    """

    op: Literal
    args: tuple[FinchNode, ...]

    def get_type(self):
        arg_types = [a.get_type() for a in self.args]
        return return_type(self.op.val, *arg_types)

    @classmethod
    def from_children(cls, op, *args):
        return cls(op, args)

    def children(self):
        return [self.op, *self.args]


@dataclass(eq=True, frozen=True)
class Access(FinchTree, FinchExpression):
    """
    Finch AST expression representing the value of tensor `tns` at the indices
    `idx...`.
    """

    tns: FinchNode
    mode: FinchNode
    idxs: tuple[FinchNode, ...]

    def get_type(self):
        # Placeholder: in a real system, would use tns/type system
        return element_type(self.tns.get_type())

    @classmethod
    def from_children(cls, tns, mode, *idxs):
        return cls(tns, mode, idxs)

    def children(self):
        return [self.tns, self.mode, *self.idxs]


@dataclass(eq=True, frozen=True)
class Cached(FinchTree, FinchExpression):
    """
    Finch AST expression `val`, equivalent to the quoted expression `ref`.
    """

    arg: FinchNode
    ref: FinchNode

    def get_type(self):
        return self.arg.get_type()

    def children(self):
        return [self.arg, self.ref]


@dataclass(eq=True, frozen=True)
class Loop(FinchTree):
    """
    Finch AST statement that runs `body` for each value of `idx` in `ext`.
    """

    idx: FinchNode
    ext: FinchNode
    body: FinchNode

    def children(self):
        return [self.idx, self.ext, self.body]


@dataclass(eq=True, frozen=True)
class If(FinchTree):
    """
    Finch AST statement that only executes `body` if `cond` is true.
    """

    cond: FinchNode
    body: FinchNode

    def children(self):
        return [self.cond, self.body]


@dataclass(eq=True, frozen=True)
class IfElse(FinchTree):
    """
    Finch AST statement that executes `then_body` if `cond` is true, otherwise
    executes `else_body`.
    """

    cond: FinchNode
    then_body: FinchNode
    else_body: FinchNode

    def children(self):
        return [self.cond, self.then_body, self.else_body]


@dataclass(eq=True, frozen=True)
class Assign(FinchTree):
    """
    Finch AST statement that updates the value of `lhs` to `op(lhs, rhs)`.
    """

    lhs: FinchNode
    op: FinchNode
    rhs: FinchNode

    def children(self):
        return [self.lhs, self.op, self.rhs]


@dataclass(eq=True, frozen=True)
class Define(FinchTree):
    """
    Finch AST statement that defines `lhs` as having the value `rhs` in `body`.
    """

    lhs: FinchNode
    rhs: FinchNode
    body: FinchNode

    def children(self):
        return [self.lhs, self.rhs, self.body]


@dataclass(eq=True, frozen=True)
class Declare(FinchTree):
    """
    Finch AST statement that declares `tns` with an initial value `init` reduced
    with `op` in the current scope.
    """

    tns: FinchNode
    init: FinchNode
    op: FinchNode

    def children(self):
        return [self.tns, self.init, self.op]


@dataclass(eq=True, frozen=True)
class Freeze(FinchTree):
    """
    Finch AST statement that freezes `tns` in the current scope after
    modifications with `op`.
    """

    tns: FinchNode
    op: FinchNode

    def children(self):
        return [self.tns, self.op]


@dataclass(eq=True, frozen=True)
class Thaw(FinchTree):
    """
    Finch AST statement that thaws `tns` in the current scope, moving the tensor
    from read-only mode to update-only mode with a reduction operator `op`.
    """

    tns: FinchNode
    op: FinchNode

    def children(self):
        return [self.tns, self.op]


@dataclass(eq=True, frozen=True)
class Block(FinchTree):
    """
    Finch AST statement that executes each of its arguments in turn.
    """

    bodies: tuple[FinchNode, ...]

    @classmethod
    def from_children(cls, *bodies):
        return cls(bodies)

    def children(self):
        return list(self.bodies)
