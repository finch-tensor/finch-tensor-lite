from collections import OrderedDict

from finchlite.autoschedule.optimize import concordize, flatten_plans, isolate_reformats, isolate_tables, \
                                        lift_fields, lift_subqueries, materialize_squeeze_expand_productions,\
                                        normalize_names, pretty_labels, propagate_copy_queries, \
                                        propagate_fields, propagate_into_reformats, propagate_map_queries, \
                                        propagate_map_queries_backward, propagate_transpose_queries, \
                                        push_fields, set_loop_order

from finchlite.finch_logic import Alias, LogicNode, Plan, Produces, Query
from finchlite.galley.TensorStats import TensorStats, DCStats

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
    prgm = lift_subqueries(prgm)

    prgm = propagate_map_queries_backward(prgm)

    prgm = isolate_reformats(prgm)
    prgm = isolate_tables(prgm)
    prgm = lift_subqueries(prgm)

    prgm = pretty_labels(prgm)

    prgm = propagate_fields(prgm)

    prgm = propagate_copy_queries(prgm)
    prgm = propagate_transpose_queries(prgm)
    prgm = propagate_map_queries(prgm)

    prgm = propagate_fields(prgm)
    prgm = push_fields(prgm)
    prgm = lift_fields(prgm)
    prgm = push_fields(prgm)
    prgm = flatten_plans(prgm)
    
    print("Before greedy optimization:\n", prgm)
    prgm = greedy_optimizer(DCStats, prgm)
    print("After greedy optimization:\n", prgm)
    prgm = propagate_transpose_queries(prgm)
    prgm = set_loop_order(prgm)
    prgm = push_fields(prgm)

    prgm = concordize(prgm)

    prgm = materialize_squeeze_expand_productions(prgm)
    prgm = propagate_copy_queries(prgm)

    prgm = propagate_into_reformats(prgm)
    prgm = propagate_copy_queries(prgm)
    prgm = normalize_names(prgm)
    print("After final optimizations:\n", prgm)
    return prgm


class GalleyLogicOptimizer:
    def __init__(self, ctx):
        self.ctx = ctx

    def __call__(self, prgm: LogicNode):
        prgm = galley_greedy_optimizer(prgm)
        return self.ctx(prgm)
