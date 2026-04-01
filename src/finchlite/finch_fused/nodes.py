"""AI modified: 2026-03-16T14:23:28Z parent=18fbb013175241f5081102d7b13f81f0e6c3de04"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

from ..algebra import return_type
from ..symbolic import Context, NamedTerm, Term, TermTree, ftype, literal_repr
from ..util import qual_str

"""
Finch Fused IR draft.

This language models simple Python functions with imperative control flow:
- Assignment
- Calls
- If / else
- While
- For

Nested function declarations are intentionally not representable in this IR.
"""


@dataclass(eq=True, frozen=True)
class FusedNode(Term, ABC):
    @classmethod
    def head(cls):
        return cls

    @classmethod
    def make_term(cls, head: Any, *children: Term) -> Self:
        return head.from_children(*children)

    @classmethod
    def from_children(cls, *children: Term) -> Self:
        return cls(*children)

    def __str__(self):
        ctx = FusedPrinterContext()
        res = ctx(self)
        return res if res is not None else ctx.emit()


@dataclass(eq=True, frozen=True)
class FusedTree(FusedNode, TermTree, ABC):
    @property
    @abstractmethod
    def children(self) -> list[FusedNode]:  # type: ignore[override]
        ...


class FusedExpression(FusedNode, ABC):
    @property
    @abstractmethod
    def result_format(self) -> Any: ...


class FusedStatement(FusedNode, ABC):
    pass


@dataclass(eq=True, frozen=True)
class Literal(FusedExpression):
    val: Any

    @property
    def result_format(self):
        return ftype(self.val)

    def __repr__(self) -> str:
        return literal_repr(type(self).__name__, {"val": self.val})


@dataclass(eq=True, frozen=True)
class Variable(FusedExpression, NamedTerm):
    name: str
    type_: Any = None

    @property
    def result_format(self):
        return self.type_

    @property
    def symbol(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return literal_repr(
            type(self).__name__, {"name": self.name, "type_": self.type_}
        )


@dataclass(eq=True, frozen=True)
class Call(FusedTree, FusedExpression):
    fn: FusedExpression
    args: tuple[FusedExpression, ...]

    @property
    def children(self):
        return [self.fn, *self.args]

    @classmethod
    def from_children(cls, fn, *args):
        return cls(fn, tuple(args))

    @property
    def result_format(self):
        arg_types = [arg.result_format for arg in self.args]
        if isinstance(self.fn, Literal):
            return return_type(self.fn.val, *arg_types)
        return None


@dataclass(eq=True, frozen=True)
class Compare(FusedTree, FusedExpression):
    left: FusedExpression
    op: Literal
    right: FusedExpression

    @property
    def children(self):
        return [self.left, self.op, self.right]

    @property
    def result_format(self):
        return bool


@dataclass(eq=True, frozen=True)
class BinaryOp(FusedTree, FusedExpression):
    left: FusedExpression
    op: Literal
    right: FusedExpression

    @property
    def children(self):
        return [self.left, self.op, self.right]

    @property
    def result_format(self):
        if isinstance(self.op, Literal):
            return return_type(
                self.op.val, self.left.result_format, self.right.result_format
            )
        return None


@dataclass(eq=True, frozen=True)
class Block(FusedTree, FusedStatement):
    body: tuple[FusedStatement, ...] = ()

    @property
    def children(self):
        return [*self.body]

    @classmethod
    def from_children(cls, *children):
        return cls(tuple(children))


@dataclass(eq=True, frozen=True)
class Assign(FusedTree, FusedStatement):
    lhs: Variable
    rhs: FusedExpression

    @property
    def children(self):
        return [self.lhs, self.rhs]


@dataclass(eq=True, frozen=True)
class ExprStmt(FusedTree, FusedStatement):
    value: FusedExpression

    @property
    def children(self):
        return [self.value]


@dataclass(eq=True, frozen=True)
class If(FusedTree, FusedStatement):
    cond: FusedExpression
    then_body: Block
    else_body: Block | None = None

    @property
    def children(self):
        if self.else_body is None:
            return [self.cond, self.then_body]
        return [self.cond, self.then_body, self.else_body]

    @classmethod
    def from_children(cls, cond, then_body, else_body=None):
        return cls(cond, then_body, else_body)


@dataclass(eq=True, frozen=True)
class While(FusedTree, FusedStatement):
    cond: FusedExpression
    body: Block

    @property
    def children(self):
        return [self.cond, self.body]


@dataclass(eq=True, frozen=True)
class For(FusedTree, FusedStatement):
    target: Variable
    iterable: FusedExpression
    body: Block

    @property
    def children(self):
        return [self.target, self.iterable, self.body]


@dataclass(eq=True, frozen=True)
class Return(FusedTree, FusedStatement):
    values: tuple[FusedExpression, ...] = ()

    @property
    def children(self):
        return [*self.values]

    @classmethod
    def from_children(cls, *values):
        return cls(tuple(values))


@dataclass(eq=True, frozen=True)
class Break(FusedTree, FusedStatement):
    @property
    def children(self):
        return []


@dataclass(eq=True, frozen=True)
class Function(FusedTree):
    name: Literal
    params: tuple[Variable, ...]
    body: Block

    @property
    def children(self):
        return [self.name, *self.params, self.body]

    @classmethod
    def from_children(cls, name, *children):
        if len(children) == 0:
            raise ValueError("Function requires at least a body")
        *params, body = children
        return cls(name, tuple(params), body)


@dataclass(eq=True, frozen=True)
class Module(FusedTree):
    functions: tuple[Function, ...]

    @property
    def children(self):
        return [*self.functions]

    @classmethod
    def from_children(cls, *functions):
        return cls(tuple(functions))


class FusedPrinterContext(Context):
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

    def __call__(self, prgm: FusedNode):
        feed = self.feed
        match prgm:
            case Literal(value):
                return qual_str(value).replace("\n", "")
            case Variable(name, _):
                return name
            case Call(fn, args):
                return f"{self(fn)}({', '.join(self(arg) for arg in args)})"
            case Compare(left, op, right):
                return f"({self(left)} {self(op)} {self(right)})"
            case BinaryOp(left, op, right):
                return f"({self(left)} {self(op)} {self(right)})"
            case Block(body):
                for stmt in body:
                    self(stmt)
                return None
            case Assign(lhs, rhs):
                self.exec(f"{feed}{self(lhs)} = {self(rhs)}")
                return None
            case ExprStmt(value):
                self.exec(f"{feed}{self(value)}")
                return None
            case If(cond, then_body, else_body):
                self.exec(f"{feed}if {self(cond)}:")
                then_ctx = self.subblock()
                then_ctx(then_body)
                if len(then_ctx.preamble) == 0:
                    then_ctx.exec(f"{then_ctx.feed}pass")
                self.exec(then_ctx.emit())
                if else_body is not None:
                    self.exec(f"{feed}else:")
                    else_ctx = self.subblock()
                    else_ctx(else_body)
                    if len(else_ctx.preamble) == 0:
                        else_ctx.exec(f"{else_ctx.feed}pass")
                    self.exec(else_ctx.emit())
                return None
            case While(cond, body):
                self.exec(f"{feed}while {self(cond)}:")
                body_ctx = self.subblock()
                body_ctx(body)
                if len(body_ctx.preamble) == 0:
                    body_ctx.exec(f"{body_ctx.feed}pass")
                self.exec(body_ctx.emit())
                return None
            case For(target, iterable, body):
                self.exec(f"{feed}for {self(target)} in {self(iterable)}:")
                body_ctx = self.subblock()
                body_ctx(body)
                if len(body_ctx.preamble) == 0:
                    body_ctx.exec(f"{body_ctx.feed}pass")
                self.exec(body_ctx.emit())
                return None
            case Return(values):
                if len(values) == 0:
                    self.exec(f"{feed}return")
                elif len(values) == 1:
                    self.exec(f"{feed}return {self(values[0])}")
                else:
                    self.exec(f"{feed}return ({', '.join(self(v) for v in values)})")
                return None
            case Function(name, params, body):
                param_str = ", ".join(self(param) for param in params)
                self.exec(f"{feed}def {name.val}({param_str}):")
                body_ctx = self.subblock()
                body_ctx(body)
                if len(body_ctx.preamble) == 0:
                    body_ctx.exec(f"{body_ctx.feed}pass")
                self.exec(body_ctx.emit())
                return None
            case Module(functions):
                for i, fn in enumerate(functions):
                    if i > 0:
                        self.exec("")
                    self(fn)
                return None
            case prgm:
                self.exec(f"{self.feed}{str(prgm)}")
                return None
