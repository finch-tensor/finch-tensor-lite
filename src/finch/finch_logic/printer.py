from inspect import isbuiltin, isclass, isfunction
from typing import Any

from ..symbolic import Context
from .nodes import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicNode,
    MapJoin,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    Subquery,
    Table,
    Value,
)


def _get_str(val: Any) -> str:
    if isbuiltin(val) or isclass(val) or isfunction(val):
        return str(val.__qualname__)
    return str(val)


class PrinterCompiler:
    def __call__(self, prgm: LogicNode):
        ctx = PrinterContext()
        ctx(prgm)
        return ctx.emit()


class PrinterContext(Context):
    def __init__(self, tab="    ", indent=0):
        super().__init__()
        self.tab = tab
        self.indent = indent

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def block(self) -> "PrinterContext":
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
                return _get_str(value).replace("\n", "")
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
            case _:
                raise ValueError(f"Unknown expression type: {type(prgm)}")
