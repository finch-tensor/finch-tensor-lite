from ..symbolic.dataflow import DataFlowAnalysis
from ..symbolic.rewriters import PostWalk, Rewrite
from ..interface import lazy, compute
from .cfg_builder import (
    NumberedStatement,
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


class LivenessAnalysis(DataFlowAnalysis):

    def get_variables_in_stmt(self, stmt: FusedNode) -> set[Variable]:
        var_set = set()
        def _var_gatherer(node: FusedNode) -> FusedNode:
            match node:
                case Variable() as var:
                    var_set.add(var.name)
                    return node
                case node:
                    return node
        Rewrite(PostWalk(_var_gatherer))(stmt)
        print("Variables in statement:", stmt,  var_set)
        return var_set

    def stmt_str(self, stmt: FusedNode) -> str:
        return str(stmt)

    def transfer(self, stmts, state: dict) -> dict:
        """
        Transfer function for the data flow analysis.
        This should be implemented by subclasses.
        """
        ...
        # Walk through the statements in reverse to compute 
        # liveness
        new_state = state.copy()
        for stmt in reversed(stmts):
            if isinstance(stmt, Assign) and stmt.lhs in new_state:
                del new_state[stmt.lhs]
                for var in self.get_variables_in_stmt(stmt.rhs):
                    new_state[var] = True
            else:
                for var in self.get_variables_in_stmt(stmt):
                    new_state[var] = True
        return new_state

    def join(self, state_1: dict, state_2: dict) -> dict:
        """
        Join function for the data flow analysis.
        This should be implemented by subclasses.
        """
        return state_1 | state_2

    def direction(self) -> str:
        return "backward"


class LazyAndComputeInsertion:
    """Rewriting pass to automatically insert explicit lazy and compute calls."""

    def __init__(self, active_vars: set[Variable] = set()):
        self.active_vars = active_vars

    def create_lazy(self) -> FusedNode:
        return Block(
            tuple(Assign(var, Call(Literal(lazy), (var,))) for var in self.active_vars)
        )

    def create_compute(self) -> FusedNode:
        return Block(
            tuple(Assign(var, Call(Literal(compute), (var,))) for var in self.active_vars)
        )

    def __call__(self, node: FusedNode) -> FusedNode:
        match node:
            case Function(name, args, body):
                return Function(name, args, self(body))
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
