import logging
from functools import reduce
from itertools import chain as join_chains

from finchlite.algebra import ffuncs
from finchlite.algebra.tensor import TensorFType
from finchlite.algebra.utils import intersect, is_subsequence
from finchlite.autoschedule.galley.logical_optimizer import insert_statistics
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    LogicExpression,
    LogicLoader,
    LogicNode,
    LogicStatement,
    MockLogicLoader,
    Plan,
    Produces,
    Query,
    Reorder,
    StatsFactory,
    Table,
    TensorStats,
)
from finchlite.finch_logic.nodes import MapJoin
from finchlite.symbolic import PostOrderDFS, PostWalk, Rewrite
from finchlite.util.logging import LOG_LOGIC_POST_OPT

from .optimize import propagate_copy_queries, with_unique_lhs
from .stages import LogicLoopOrderOptimizer
from .standardize import concordize, flatten_plans, push_fields

logger = logging.LoggerAdapter(logging.getLogger(__name__), extra=LOG_LOGIC_POST_OPT)

"""Julia port """


def _transpose_penalty(
    expr: LogicExpression,
    loop_prefix: tuple[Field, ...],
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
) -> float:
    penalty = 0.0
    for node in PostOrderDFS(expr):
        match node:
            case Table(Alias() as tns, idxs):
                base = stats_bindings.get(tns)
                st = stats_factory.relabel(base, tuple(idxs))
                if not is_subsequence(tuple(idxs), loop_prefix):
                    penalty += st.estimate_non_fill_values()
            case _:
                pass
    return penalty


def loop_order_cost(
    expr: LogicExpression,
    loop_order: tuple[Field, ...],
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
) -> float:
    full_stats = insert_statistics(
        stats_factory, expr, stats_bindings.copy(), replace=False, cache={}
    )
    cost = 0.0
    for j in range(1, len(loop_order) + 1):
        prefix = set(loop_order[:j])
        reduce_idxs = tuple(f for f in full_stats.index_order if f not in prefix)
        projected = stats_factory.aggregate(ffuncs.or_, False, reduce_idxs, full_stats)
        cost += projected.estimate_non_fill_values()
    cost += _transpose_penalty(expr, loop_order, stats_factory, stats_bindings)
    return cost


""" End Julia port """


def add_output_orders(prgm: LogicStatement) -> LogicStatement:
    produced_aliases: set[Alias] = set()
    for stmt in PostOrderDFS(prgm):
        match stmt:
            case Produces(vars):
                produced_aliases.update(vars)

    def rule_1(node: LogicNode) -> LogicNode | None:
        match node:
            case Query(lhs, Reorder()):
                return node
            case Query(lhs, rhs):
                if lhs in produced_aliases:
                    return Query(lhs, Reorder(rhs, rhs.fields()))
                return node
        return None

    return Rewrite(PostWalk(rule_1))(prgm)


def drop_internal_reorders(
    root: LogicStatement, keep_loop_orders: bool
) -> LogicStatement:
    def reorder_remover(ex):
        match ex:
            case Reorder(arg_2, _):
                return arg_2

    def rule_1(stmt):
        match stmt:
            case Query(lhs, Aggregate(op, init, arg, idxs_2)):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return Query(lhs, Aggregate(op, init, arg_1, idxs_2))
            case Query(lhs, Reorder(Aggregate(op, init, arg, ag_idxs), idxs)):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return Query(lhs, Reorder(Aggregate(op, init, arg_1, ag_idxs), idxs))
            case Query(
                lhs,
                Reorder(MapJoin(op, (tbl, Aggregate(op1, init, arg, ag_idxs))), idxs),
            ):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return Query(
                    lhs,
                    Reorder(
                        MapJoin(op, (tbl, Aggregate(op1, init, arg_1, ag_idxs))), idxs
                    ),
                )

    def rule_2(stmt):
        match stmt:
            case Query(lhs, Aggregate(op, init, Reorder(arg, idxs_1), idxs_2)):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return Query(lhs, Aggregate(op, init, Reorder(arg_1, idxs_1), idxs_2))
            case Query(
                lhs, Reorder(Aggregate(op, init, Reorder(arg, idxs_1), ag_idxs), idxs)
            ):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return Query(
                    lhs,
                    Reorder(Aggregate(op, init, Reorder(arg_1, idxs_1), ag_idxs), idxs),
                )
            case Query(
                lhs,
                Reorder(
                    MapJoin(
                        op, (tbl, Aggregate(op1, init, Reorder(arg, idxs_1), ag_idxs))
                    ),
                    idxs,
                ),
            ):
                arg_1 = Rewrite(PostWalk(reorder_remover))(arg)
                return Query(
                    lhs,
                    Reorder(
                        MapJoin(
                            op,
                            tbl,
                            Aggregate(op1, init, Reorder(arg_1, idxs_1), ag_idxs),
                        ),
                        idxs,
                    ),
                )

    if keep_loop_orders:
        return Rewrite(PostWalk(rule_2))(root)
    return Rewrite(PostWalk(rule_1))(root)


class CycleInFields(Exception): ...


def toposort(chains: list[list[Field]]) -> tuple[Field, ...]:
    chains = [c for c in chains if len(c) > 0]
    parents = {chain[0]: 0 for chain in chains}
    for chain in chains:
        for f in chain[1:]:
            parents[f] = parents.get(f, 0) + 1
    roots = [f for f in parents if parents[f] == 0]
    perm = []
    while len(parents) > 0:
        if len(roots) == 0:
            raise CycleInFields("Cycle detected in fields' orders")
        perm.append(roots.pop())
        for chain in chains:
            if len(chain) > 0 and chain[0] == perm[-1]:
                chain.pop(0)
                if len(chain) > 0:
                    parents[chain[0]] -= 1
                    if parents[chain[0]] == 0:
                        roots.append(chain[0])
        parents.pop(perm[-1])
    return tuple(perm)


def _heuristic_loop_order(root: LogicExpression) -> tuple[Field, ...]:
    chains = []
    for node in PostOrderDFS(root):
        match node:
            case Table(_, idxs_1):
                chains.append(list(intersect(idxs_1, root.fields())))
    chains.extend([f] for f in root.fields())

    need_fix = False
    try:
        result = toposort(chains)
    except CycleInFields:
        logger.warning("Cycle in fields detected, need to permute.")
        need_fix = True
        result = root.fields()

    if need_fix or reduce(max, [len(c) for c in chains], 0) < len(
        set(join_chains(*chains))
    ):
        counts: dict[Field, int] = {}
        for chain in chains:
            for f in chain:
                counts[f] = counts.get(f, 0) + 1
        result = tuple(sorted(result, key=lambda x: counts[x] == 1))
    return result


def set_loop_order(plan: Plan) -> Plan:
    new_queries = []
    for query in plan.bodies[:-1]:

        def rule_1(query):
            match query:
                case Query(lhs, Aggregate(op, init, arg, idxs)):
                    assert isinstance(arg, LogicExpression)
                    idxs_2 = _heuristic_loop_order(arg)
                    rhs_2 = Aggregate(op, init, Reorder(arg, idxs_2), idxs)
                    return Query(lhs, rhs_2)
                case Query(lhs, Reorder(Aggregate(op, init, arg, ag_idxs), idxs)):
                    idxs_2 = _heuristic_loop_order(arg)
                    rhs_2 = Reorder(
                        Aggregate(op, init, Reorder(arg, idxs_2), ag_idxs), idxs
                    )
                    return Query(lhs, rhs_2)
                case Query(lhs, Reorder(Table(Alias(), _), idxs)) as q:
                    return q
                case _:
                    raise Exception(f"Invalid node: {query} in set_loop_order")

        new_queries.append(rule_1(query))
    return Plan(tuple(new_queries + [plan.bodies[-1]]))


class DefaultLoopOrderer(LogicLoopOrderOptimizer):
    def __init__(self, ctx: LogicLoader | None = None):
        if ctx is None:
            ctx = MockLogicLoader()
        self.ctx = ctx

    def lower(
        self,
        prgm: LogicStatement,
        bindings: dict[Alias, TensorFType],
        stats: dict[Alias, TensorStats],
        stats_factory: StatsFactory,
    ):
        def loop_order_transform(prgm, bindings):
            prgm = add_output_orders(prgm)
            prgm = drop_internal_reorders(prgm, keep_loop_orders=False)
            prgm = set_loop_order(prgm)
            prgm = push_fields(prgm)
            prgm = concordize(prgm, bindings)
            prgm = drop_internal_reorders(prgm, keep_loop_orders=True)
            prgm = propagate_copy_queries(prgm, bindings)
            prgm = flatten_plans(prgm)
            return prgm, bindings

        prgm, bindings = with_unique_lhs(loop_order_transform, prgm, bindings)
        prgm = flatten_plans(prgm)
        return self.ctx(prgm, bindings, stats, stats_factory)
