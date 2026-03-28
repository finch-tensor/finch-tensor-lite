from functools import reduce

from ....finch_logic import Query
from .annotated_query import AnnotatedQuery


def greedy_query(input_aq: AnnotatedQuery, use_components: bool = True) -> list[Query]:
    aq = input_aq
    queries: list[Query] = []

    component_idx = 0
    while True:
        components = aq.connected_components
        if not use_components:
            # merges all the components into one list
            component = reduce(lambda a, b: a + b, components, [])
        else:
            if component_idx >= len(components):
                break
            component = components[component_idx]

        reducible_idxs = aq.get_reducible_idxs_for_component(component)
        if not reducible_idxs:
            if use_components:
                component_idx += 1
            else:
                break
            continue

        best_idx = min(
            reducible_idxs, key=lambda idx: aq.get_cost_of_reduce_idx(idx)
        )
        query = aq.reduce_idx(best_idx)
        queries.append(query)
        # connected_components are recomputed in reduce_idx; stay on the same
        # component index until it has no reducible indices left.

    remaining_q = aq.get_remaining_query()
    if remaining_q is not None:
        queries.append(remaining_q)

    if queries:
        last_query = queries[-1]
        if last_query.lhs != aq.output_name:
            queries[-1] = Query(aq.output_name, last_query.rhs)

    return queries
