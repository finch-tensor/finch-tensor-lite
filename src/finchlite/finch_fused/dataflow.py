from typing import cast

from ..interface import compute, lazy
from ..symbolic.dataflow import DataFlowAnalysis
from ..symbolic.rewriters import Chain, PostWalk, Rewrite
from .cfg_builder import (
    NumberedStatement,
    fused_build_cfg,
    fused_desugar,
    number_statements,
)
from .nodes import (
    Assign,
    Block,
    Break,
    Call,
    FusedNode,
    FusedStatement,
    Literal,
    Return,
    Variable,
)


class LivenessAnalysis(DataFlowAnalysis):
    def get_variables_in_stmt(self, stmt: FusedNode) -> set[Variable]:
        var_set = set()

        def _var_gatherer(node: FusedNode) -> FusedNode:
            match node:
                case Variable() as var:
                    var_set.add(var)
                    return node
                case node:
                    return node

        Rewrite(PostWalk(_var_gatherer))(stmt)
        return var_set

    def stmt_str(self, stmt: FusedNode, state: dict) -> str:
        str_state = ", ".join(f"{var}" for var in state)
        return f"Live vars: {{{str_state}}} | Stmt: {stmt}"

    def transfer(self, stmts, state: dict) -> dict:
        # Walk through the statements in reverse to compute
        # liveness
        new_state = state.copy()
        for stmt in reversed(stmts):
            match stmt:
                case NumberedStatement(Assign(lhs, rhs), _):
                    if lhs in new_state:
                        del new_state[lhs]
                    for var in self.get_variables_in_stmt(rhs):
                        new_state[var] = True
                case Assign(lhs, rhs):
                    if lhs in new_state:
                        del new_state[lhs]
                    for var in self.get_variables_in_stmt(rhs):
                        new_state[var] = True
                case stmt:
                    for var in self.get_variables_in_stmt(stmt):
                        new_state[var] = True
        return new_state

    def join(self, state_1: dict, state_2: dict) -> dict:
        return state_1 | state_2

    def direction(self) -> str:
        return "backward"


# Find the first and last statement in the block to determine
# where to insert lazy and compute calls
def _get_stmt_bounds(stmts: list[FusedNode]) -> tuple[int, int]:
    if not stmts:
        return -1, -1

    top_id = -1
    bot_id = -1
    for stmt in stmts:
        if isinstance(stmt, NumberedStatement):
            if top_id == -1:
                top_id = stmt.sid
            bot_id = stmt.sid
    return top_id, bot_id


def _insert_compute(prgm: FusedNode, compute_sid, vars: set[Variable]) -> FusedNode:
    def _visitor(node):
        match node:
            # In the case of returns, we should just compute the retuned values
            case NumberedStatement(Return(ret_vars), sid) if sid == compute_sid:
                computes = tuple(
                    Assign(var, Call(Literal(compute), (var,))) for var in ret_vars
                )
                return Block(computes + (node,))
            case NumberedStatement(stmt, sid) if sid == compute_sid:
                computes = tuple(
                    Assign(var, Call(Literal(compute), (var,))) for var in vars
                )
                match stmt:
                    case Return():
                        return Block(computes + (node,))
                    case Break():
                        return Block(computes + (node,))
                    case stmt:
                        return Block((node,) + computes)
            case node:
                return node

    return Rewrite(PostWalk(_visitor))(prgm)


def _insert_lazy(prgm: FusedNode, lazy_sid: int, vars: set[Variable]) -> FusedNode:
    def _visitor(node):
        match node:
            case NumberedStatement(stmt, sid) if sid == lazy_sid and not isinstance(
                stmt, (Return, Break)
            ):
                lazies = tuple(Assign(var, Call(Literal(lazy), (var,))) for var in vars)
                return Block(lazies + (node,))
            case node:
                return node

    return Rewrite(PostWalk(_visitor))(prgm)


def _unwrap_numbered_stmt(node: FusedNode) -> FusedNode:
    match node:
        case NumberedStatement(stmt, _):
            return stmt
        case node:
            return node


def _unnest_block(node: FusedNode) -> FusedNode:
    match node:
        case Block(bodies):
            new_bodies: list[FusedStatement] = []
            for b in bodies:
                b2 = _unnest_block(b)
                if isinstance(b2, Block):
                    new_bodies.extend(b2.body)
                else:
                    new_bodies.append(cast(FusedStatement, b2))
            return Block(tuple(new_bodies))
        case node:
            return node


def insert_lazy_and_compute(prgm: FusedNode) -> FusedNode:
    # desugar the input name and number additional statements for CFG construction
    numbered_prgm, _ = number_statements(prgm)
    desugared_prgm = fused_desugar(numbered_prgm)
    cfg = fused_build_cfg(desugared_prgm)
    liveness = LivenessAnalysis(cfg)
    liveness.analyze()
    print("Liveness analysis results:")
    print(liveness)
    for block in cfg.blocks.values():
        live_outputs = set(liveness.input_states[
            block.id
        ].keys())  # Backwards analysis, so live outputs are the input state of the block
        live_inputs = set(liveness.output_states[
            block.id
        ].keys())  # Backwards analysis, so live inputs are the output state of the block
        min_id, max_id = _get_stmt_bounds(block.statements)
        print(
            "insert lazy for live inputs",
            live_inputs,
            " at block",
            block,
            "with min stmt id",
            min_id,
        )
        print(
            "insert compute for live outputs",
            live_outputs,
            " at block",
            block,
            "with max stmt id",
            max_id,
        )
        numbered_prgm = _insert_lazy(numbered_prgm, min_id, live_inputs)
        numbered_prgm = _insert_compute(numbered_prgm, max_id, live_outputs)
    return Rewrite(PostWalk(Chain([_unwrap_numbered_stmt, _unnest_block])))(
        numbered_prgm
    )
