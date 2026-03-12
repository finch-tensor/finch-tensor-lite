# AI modified: 2026-02-17 ada0c7d
from collections import OrderedDict

from finchlite.autoschedule.optimize import (
    concordize,
    propagate_copy_queries,
    propagate_transpose_queries,
    push_fields,
    set_loop_order,
)
from finchlite.finch_logic import Alias, LogicNode, Plan, Produces, Query
from finchlite.galley.TensorStats import DCStats, TensorStats

from .annotated_query import (
    AnnotatedQuery,
    get_cost_of_reduce,
    get_reducible_idxs,
    get_remaining_query,
    reduce_idx,
)
from .logic_to_stats import insert_statistics


def greedy_query_optimizer(aq: AnnotatedQuery) -> list[Query]:
    queries: list[Query] = []
    reducible_idxs = get_reducible_idxs(aq)
    while reducible_idxs:
        cheapest_idx = reducible_idxs[0]
        cost = float("inf")
        for idx in reducible_idxs:
            idx_cost = get_cost_of_reduce(idx, aq)
            if idx_cost < cost:
                cheapest_idx = idx
                cost = idx_cost

        query = reduce_idx(cheapest_idx, aq)
        queries.append(query)
        reducible_idxs = get_reducible_idxs(aq)

    remaining_q = get_remaining_query(aq)
    if remaining_q is not None:
        queries.append(remaining_q)
    else:
        queries[-1] = Query(aq.output_name, queries[-1].rhs)
    return queries


def greedy_optimizer(ST: type[TensorStats], plan: Plan):
    bindings: OrderedDict[Alias, TensorStats] = OrderedDict()
    stats_cache: OrderedDict[LogicNode, TensorStats] = OrderedDict()
    new_bodies: list[Query | Produces] = []
    print("Plan bodies to optimize:", plan.bodies)
    for body in plan.bodies:
        match body:
            case Query() as query:
                print("MATCHED QUERY:", query)
                aq = AnnotatedQuery(ST, query)
                optimized_queries = greedy_query_optimizer(aq)
                for sub_query in optimized_queries:
                    bindings[sub_query.lhs] = insert_statistics(
                        ST, sub_query.rhs, bindings, False, stats_cache
                    )
                new_bodies.extend(optimized_queries)
            case Produces() as produce:
                new_bodies.append(produce)
    return Plan(tuple(new_bodies))


def galley_greedy_optimizer(prgm: LogicNode) -> LogicNode:
    prgm = greedy_optimizer(DCStats, prgm)
    print("After greedy optimization:\n", prgm)
    prgm = propagate_transpose_queries(prgm)
    prgm = set_loop_order(prgm)
    prgm = push_fields(prgm)

    prgm = concordize(prgm)

    prgm = propagate_copy_queries(prgm)

    prgm = propagate_copy_queries(prgm)
    print("After final optimizations:\n", prgm)
    return prgm


class GalleyLogicOptimizer:
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, prgm: LogicNode):
        self.ctx(prgm)
        return galley_greedy_optimizer(prgm)
