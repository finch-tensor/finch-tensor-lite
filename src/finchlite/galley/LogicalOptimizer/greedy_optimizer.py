from ...finch_logic import Query
from .annotated_query import (
    AnnotatedQuery,
    get_reducible_idxs,
    get_remaining_query,
    reduce_idx,
)


def greedy_query(input_aq: AnnotatedQuery) -> list[Query]:
    aq = input_aq
    queries: list[Query] = []
    reducible_idxs = get_reducible_idxs(aq)
    while reducible_idxs:
        next_idx = reducible_idxs[0]
        query = reduce_idx(next_idx, aq)
        queries.append(query)
        reducible_idxs = get_reducible_idxs(aq)

    remaining_q = get_remaining_query(aq)
    if remaining_q is not None:
        queries.append(remaining_q)

    if queries:
        last_query = queries[-1]
        if last_query.lhs != aq.output_name:
            queries[-1] = Query(aq.output_name, last_query.rhs)

    return queries
