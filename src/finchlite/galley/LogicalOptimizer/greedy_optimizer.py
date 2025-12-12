from collections import OrderedDict

from finchlite.finch_logic import Alias, LogicNode, Plan, Produces, Query
from finchlite.galley.TensorStats import TensorStats

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
    for body in plan.bodies:
        match body:
            case Query():
                aq = AnnotatedQuery(ST, body)
                optimized_queries = greedy_query_optimizer(aq)
                for sub_query in optimized_queries:
                    bindings[sub_query.lhs] = insert_statistics(
                        ST, sub_query.rhs, bindings, False, stats_cache
                    )
                new_bodies.extend(optimized_queries)
            case Produces():
                new_bodies.append(body)
    return Plan(tuple(new_bodies))
