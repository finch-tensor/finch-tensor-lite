from ...finch_logic import Query
from .annotated_query import AnnotatedQuery


def greedy_query(input_aq: AnnotatedQuery) -> list[Query]:
    aq = input_aq
    queries: list[Query] = []
    reducible_idxs = aq.get_reducible_idxs()
    while len(reducible_idxs) > 0:
        best_idx = reducible_idxs[0]
        cheapest_cost = float("inf")
        for idx in reducible_idxs:
            cost = aq.get_cost_of_reduce_idx(idx)
            print("[COST HERE]", cost, "for idx", idx)
            if cost < cheapest_cost:
                cheapest_cost = cost
                best_idx = idx
        query = aq.reduce_idx(best_idx)
        queries.append(query)
        reducible_idxs = aq.get_reducible_idxs()

    remaining_q = aq.get_remaining_query()
    if remaining_q is not None:
        queries.append(remaining_q)

    if queries:
        last_query = queries[-1]
        if last_query.lhs != aq.output_name:
            queries[-1] = Query(aq.output_name, last_query.rhs)

    return queries
