from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, List, Optional
from ..algebra import return_type, element_type
from ..symbolic import Term


# Base class for all Finch Notation nodes
class FinchNode:
    pass

class FinchExpression:
    """
    Finch AST expression base class.
    
    This class is used to represent expressions in the Finch Abstract Syntax Tree (AST).
    It is a marker class that can be extended by specific expression types.
    """
    def get_type(self) -> Any:
        """
        Get the type of the expression.
        
        This method should be overridden by subclasses to return the specific type of the expression.
        """
        ...

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
    Finch AST expression for host code `val` expected to evaluate to a value of type `type_`.
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
class Call(FinchNode, FinchExpression):
    """
    Finch AST expression for the result of calling the function `op` on `args...`.
    """
    op: Literal 
    args: Tuple[FinchNode, ...]
    def get_type(self):
        arg_types = [a.get_type() for a in self.args]
        return return_type(self.op.val, *arg_types)

@dataclass(eq=True, frozen=True)
class Access(FinchNode, FinchExpression):
    """
    Finch AST expression representing the value of tensor `tns` at the indices `idx...`.
    """
    tns: FinchNode
    mode: FinchNode
    idxs: Tuple[FinchNode, ...]
    def get_type(self):
        # Placeholder: in a real system, would use tns/type system
        return element_type(self.tns.get_type())

@dataclass(eq=True, frozen=True)
class Cached(FinchNode, FinchExpression):
    """
    Finch AST expression `val`, equivalent to the quoted expression `ref`.
    """
    arg: FinchNode
    ref: FinchNode
    def get_type(self):
        return self.arg.get_type()

@dataclass(eq=True, frozen=True)
class Loop(FinchNode):
    """
    Finch AST statement that runs `body` for each value of `idx` in `ext`.
    """
    idx: FinchNode
    ext: FinchNode
    body: FinchNode

@dataclass(eq=True, frozen=True)
class If(FinchNode):
    """
    Finch AST statement that only executes `body` if `cond` is true.
    """
    cond: FinchNode
    body: FinchNode

@dataclass(eq=True, frozen=True)
class IfElse(FinchNode):
    """
    Finch AST statement that executes `then_body` if `cond` is true, otherwise executes `else_body`.
    """
    cond: FinchNode
    then_body: FinchNode
    else_body: FinchNode

@dataclass(eq=True, frozen=True)
class Assign(FinchNode):
    """
    Finch AST statement that updates the value of `lhs` to `op(lhs, rhs)`.
    """
    lhs: FinchNode
    op: FinchNode
    rhs: FinchNode

@dataclass(eq=True, frozen=True)
class Define(FinchNode):
    """
    Finch AST statement that defines `lhs` as having the value `rhs` in `body`.
    """
    lhs: FinchNode
    rhs: FinchNode
    body: FinchNode

@dataclass(eq=True, frozen=True)
class Declare(FinchNode):
    """
    Finch AST statement that declares `tns` with an initial value `init` reduced with `op` in the current scope.
    """
    tns: FinchNode
    init: FinchNode
    op: FinchNode

@dataclass(eq=True, frozen=True)
class Freeze(FinchNode):
    """
    Finch AST statement that freezes `tns` in the current scope after modifications with `op`.
    """
    tns: FinchNode
    op: FinchNode

@dataclass(eq=True, frozen=True)
class Thaw(FinchNode):
    """
    Finch AST statement that thaws `tns` in the current scope, moving the tensor from read-only mode to update-only mode with a reduction operator `op`.
    """
    tns: FinchNode
    op: FinchNode

@dataclass(eq=True, frozen=True)
class Block(FinchNode):
    """
    Finch AST statement that executes each of its arguments in turn.
    """
    bodies: Tuple[FinchNode, ...]