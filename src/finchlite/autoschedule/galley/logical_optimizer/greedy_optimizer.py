from functools import reduce

from ....finch_logic import Field, Query
from .annotated_query import AnnotatedQuery


def greedy_query(input_aq: AnnotatedQuery, use_components: bool = True) -> list[Query]:
    aq = input_aq
    queries: list[Query] = []

    # Using branch and bound component code to avoid indexes
    while aq.get_reducible_idxs():
        if not use_components:
            component = reduce(
                lambda a, b: a + b, aq.connected_components, list[Field]()
            )
        else:
            component = aq.connected_components[0]

        reducible_idxs = aq.get_reducible_idxs_for_component(component)
        best_idx = min(reducible_idxs, key=lambda idx: aq.get_cost_of_reduce_idx(idx))
        query = aq.reduce_idx(best_idx)
        queries.append(query)

    remaining_q = aq.get_remaining_query()
    queries.append(remaining_q)
    return queries
