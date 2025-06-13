from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from ..algebra import element_type, query_property, return_type
from ..symbolic import Term, TermTree


# Base class for all Finch Notation nodes
class NotationNode(Term):
    pass


class NotationTree(NotationNode, TermTree):
    @classmethod
    def make_term(cls, head, *children):
        return head.from_children(*children)

    @classmethod
    def from_children(cls, *children):
        return cls(*children)


class NotationExpression(NotationNode):
    """
    Notation AST expression base class.
    """

    @property
    @abstractmethod
    def result_format(self) -> Any:
        """
        Get the type of the expression.
        """
        ...


@dataclass(eq=True, frozen=True)
class Literal(NotationExpression):
    """
    Notation AST expression for the literal value `val`.
    """

    val: Any

    @property
    def result_format(self):
        return type(self.val)


@dataclass(eq=True, frozen=True)
class Value(NotationExpression):
    """
    Notation AST expression for host code `val` expected to evaluate to a value of
    type `type_`.
    """

    val: Any
    type_: Any

    @property
    def result_format(self):
        return self.type_


@dataclass(eq=True, frozen=True)
class Variable(NotationExpression):
    """
    Notation AST expression for a variable named `name`.
    """

    name: str
    type_: Any = None

    @property
    def result_format(self):
        return self.type_


@dataclass(eq=True, frozen=True)
class Call(NotationTree, NotationExpression):
    """
    Notation AST expression for the result of calling the function `op` on
    `args...`.
    """

    op: Literal
    args: tuple[NotationNode, ...]

    @property
    def result_format(self):
        arg_types = [a.result_format for a in self.args]
        return return_type(self.op.val, *arg_types)

    @classmethod
    def from_children(cls, op, *args):
        return cls(op, args)

    @property
    def children(self):
        return [self.op, *self.args]


class AccessMode(NotationNode):
    """
    Notation AST node representing the access mode of a tensor.
    """


@dataclass(eq=True, frozen=True)
class Access(NotationTree, NotationExpression):
    """
    Notation AST expression representing the value of tensor `tns` at the indices
    `idx...`.
    """

    tns: NotationNode
    mode: AccessMode
    idxs: tuple[NotationNode, ...]

    @property
    def result_format(self):
        # Placeholder: in a real system, would use tns/type system
        return element_type(self.tns.result_format)

    @classmethod
    def from_children(cls, tns, mode, *idxs):
        return cls(tns, mode, idxs)

    @property
    def children(self):
        return [self.tns, self.mode, *self.idxs]


@dataclass(eq=True, frozen=True)
class Read(AccessMode):
    """
    Notation AST node representing a read-only access mode for a tensor.
    This mode allows reading the value of a tensor without modifying it.
    """

    @property
    def children(self):
        return []


@dataclass(eq=True, frozen=True)
class Update(AccessMode, NotationTree):
    """
    Notation AST node representing an update access mode for a tensor.  This
    mode allows reading and modifying the value of a tensor.  Increment
    operations are allowed in this mode, and will use the update operation `op`
    to increment `ref` with `val` as `ref = op(ref, val)`.  To overwrite the
    value of a tensor, use the ops `algebra.overwrite` or `algebra.InitWrite`.
    Attributes:
        op: The operation used to update the value of the tensor.
    """

    op: NotationNode

    @property
    def children(self):
        return [self.op]


@dataclass(eq=True, frozen=True)
class Increment(NotationTree):
    """
    Notation AST statement that updates the value `lhs` using `rhs`.
    """

    lhs: NotationNode
    rhs: NotationNode

    @property
    def children(self):
        return [self.lhs, self.rhs]


@dataclass(eq=True, frozen=True)
class Unwrap(NotationTree):
    """
    Notation AST statement that unwraps the scalar value from a 0-dimensional
    tensor `arg`.
    """

    arg: NotationNode

    @property
    def children(self):
        return [self.arg]


@dataclass(eq=True, frozen=True)
class Cached(NotationTree, NotationExpression):
    """
    Notation AST expression `arg`, equivalent to the quoted expression `ref`.

    Often used after the compiler caches the computation `ref` into a variable
    `arg`, but we still wish to refer to the original expression to prove
    properties about it.
    """

    arg: NotationNode
    ref: NotationNode

    @property
    def result_format(self):
        return self.arg.result_format

    @property
    def children(self):
        return [self.arg, self.ref]


@dataclass(eq=True, frozen=True)
class Loop(NotationTree):
    """
    Notation AST statement that runs `body` for each value of `idx` in `ext`.
    """

    idx: NotationNode
    ext: NotationNode
    body: NotationNode

    @property
    def children(self):
        return [self.idx, self.ext, self.body]


@dataclass(eq=True, frozen=True)
class If(NotationTree):
    """
    Notation AST statement that only executes `body` if `cond` is true.
    """

    cond: NotationNode
    body: NotationNode

    @property
    def children(self):
        return [self.cond, self.body]


@dataclass(eq=True, frozen=True)
class IfElse(NotationTree):
    """
    Notation AST statement that executes `then_body` if `cond` is true, otherwise
    executes `else_body`.
    """

    cond: NotationNode
    then_body: NotationNode
    else_body: NotationNode

    @property
    def children(self):
        return [self.cond, self.then_body, self.else_body]


@dataclass(eq=True, frozen=True)
class Assign(NotationTree):
    """
    Notation AST statement that defines `lhs` as having the value `rhs`.
    """

    lhs: NotationNode
    rhs: NotationNode

    @property
    def children(self):
        return [self.lhs, self.rhs]


@dataclass(eq=True, frozen=True)
class Declare(NotationTree, NotationExpression):
    """
    Notation AST statement that declares `tns` with an initial value `init` reduced
    with `op` in the current scope.
    """

    tns: NotationNode
    init: NotationNode
    op: NotationNode
    shape: tuple[NotationNode, ...]

    @property
    def children(self):
        return [self.tns, self.init, self.op, *self.shape]

    def from_children(cls, tns, init, op, *shape):
        """
        Creates a Declare node from its children.
        """
        return cls(tns, init, op, shape)

    @property
    def result_format(self):
        """
        Returns the type of the declared tensor.
        """
        return query_property(
            self.tns.result_format,
            "declare",
            "return_type",
            self.op.result_format,
            *[s.result_format for s in self.shape],
        )


@dataclass(eq=True, frozen=True)
class Freeze(NotationTree, NotationExpression):
    """
    Notation AST statement that freezes `tns` in the current scope after
    modifications with `op`.
    """

    tns: NotationNode
    op: NotationNode

    @property
    def children(self):
        return [self.tns, self.op]

    @property
    def result_format(self):
        """
        Returns the type of the frozen tensor.
        """
        return query_property(
            self.tns.result_format,
            "freeze",
            "return_type",
            self.op.result_format,
        )


@dataclass(eq=True, frozen=True)
class Thaw(NotationTree, NotationExpression):
    """
    Notation AST statement that thaws `tns` in the current scope, moving the tensor
    from read-only mode to update-only mode with a reduction operator `op`.
    """

    tns: NotationNode
    op: NotationNode

    @property
    def children(self):
        return [self.tns, self.op]

    @property
    def result_format(self):
        """
        Returns the type of the thawed tensor.
        """
        return query_property(
            self.tns.result_format,
            "thaw",
            "return_type",
            self.op.result_format,
        )


@dataclass(eq=True, frozen=True)
class Block(NotationTree):
    """
    Notation AST statement that executes each of its arguments in turn.
    """

    bodies: tuple[NotationNode, ...]

    @classmethod
    def from_children(cls, *bodies):
        return cls(bodies)

    @property
    def children(self):
        return list(self.bodies)


@dataclass(eq=True, frozen=True)
class Function(NotationTree):
    """
    Represents a logical AST statement that defines a function `fun` on the
    arguments `args...`.

    Attributes:
        name: The name of the function to define as a variable typed with the
            return type of this function.
        args: The arguments to the function.
        body: The body of the function. If it does not contain a return statement,
            the function returns the value of `body`.
    """

    name: Variable
    args: tuple[Variable, ...]
    body: NotationNode

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.name, *self.args, self.body]

    @classmethod
    def from_children(cls, name, *args, body):
        """Creates a term with the given head and arguments."""
        return cls(name, args, body)


@dataclass(eq=True, frozen=True)
class Return(NotationTree):
    """
    Notation AST statement that returns the value of `val` from the current
    function.
    """

    val: NotationNode

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.val]

    @classmethod
    def from_children(cls, val):
        return cls(val)


@dataclass(eq=True, frozen=True)
class Module(NotationTree):
    """
    Represents a group of functions. This is the toplevel translation unit for
    FinchNotation.

    Attributes:
        funcs: The functions defined in the module.
    """

    funcs: tuple[NotationNode, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [*self.funcs]

    @classmethod
    def from_children(cls, *funcs):
        return cls(funcs)
