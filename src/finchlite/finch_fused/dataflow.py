from finchlite.interface.fuse import compute

from ..interface import lazy
from .cfg_builder import (
    fused_desugar,
)
from .nodes import (
    Assign,
    Block,
    Call,
    For,
    Function,
    FusedNode,
    If,
    Literal,
    Variable,
    While,
)


class LazyAndComputeInsertion:
    """Rewriting pass to automatically insert explicit lazy and compute calls."""

    def __init__(self, active_vars: set[Variable] = set()):
        self.active_vars = active_vars

    def create_lazy(self) -> FusedNode:
        return Block(Assign(var, Call(Literal(lazy), var)) for var in self.active_vars)

    def create_compute(self) -> FusedNode:
        return Block(
            Assign(var, Call(Literal(compute), var)) for var in self.active_vars
        )

    def __call__(self, node: FusedNode) -> FusedNode:
        match node:
            case Function(name, args, body):
                return self(body)
            case Block(bodies):
                return Block(tuple(self(b) for b in bodies))
            case If(cond, body, else_body):
                return Block(
                    (
                        self.create_compute(),
                        If(
                            cond,
                            Block((self.create_lazy(), self(body))),
                            Block((self.create_lazy(), self(else_body))),
                        ),
                    )
                )
            case While(cond, body):
                return Block(
                    (self.create_compute(), Block((self.create_lazy(), self(body))))
                )
            case For(target, iter, body):
                return Block(
                    (
                        self.create_compute(),
                        For(target, iter, Block((self.create_lazy(), self(body)))),
                    )
                )
            case Assign(var, value):
                self.active_vars.add(var)
                return node
            case node:
                return node


def insert_lazy_and_compute(node: FusedNode) -> FusedNode:
    node, sid = fused_desugar(node, -1)
    return LazyAndComputeInsertion()(node)
