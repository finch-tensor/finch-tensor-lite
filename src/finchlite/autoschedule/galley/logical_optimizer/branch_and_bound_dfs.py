"""
Depth-first branch-and-bound for Galley reduction ordering.
"""

from collections import OrderedDict, deque

from ....finch_logic import Query
from .annotated_query import AnnotatedQuery


def _cost_of_reduce(idx, aq: AnnotatedQuery) -> tuple[float, list]:
    """
    Return (cost, reduced_vars) for reducing idx in aq.
    Used to score each candidate reduction and to know which indices
    get eliminated (reduced_vars) for the branch-and-bound state key.
    """
    _, _, _, reduced_idxs = aq.get_reduce_query(idx)
    cost = aq.get_cost_of_reduce_idx(idx)
    return cost, list(reduced_idxs)


def branch_and_bound_dfs(
    input_aq: AnnotatedQuery,
    component: list,
    max_subquery_costs: OrderedDict,
) -> tuple:
    """
    Depth-first branch-and-bound (full expansion at each node: no beam width).

    Prunes with the same greedy ``max_subquery_costs`` bound, and additionally:
    if some set ``S`` already seen has ``S >= new_vars`` (i.e. ``new_vars`` ⊆ ``S``)
    and ``cost(S) < total_cost`` for reaching ``new_vars``, the branch is discarded.
    Frozenset comparison is used to check if a set is a superset of another set.

    Returns ``((order, queries, aq, cost), optimal_subquery_costs)`` like
    ``branch_and_bound``. ``optimal_subquery_costs`` maps every visited state to its
    best cost (for use as pruning bounds after a greedy pass).
    """
    component_set = frozenset(component)
    optimal_orders: OrderedDict[frozenset, tuple] = OrderedDict(
        [(frozenset(), ([], [], input_aq, 0.0))]
    )
    # Minimum cost observed for each reduced-var set (for superset domination).
    seen_min_cost: dict[frozenset, float] = {frozenset(): 0.0}

    queue: deque[tuple[frozenset, AnnotatedQuery, list, list, float]] = deque()
    queue.appendleft((frozenset(), input_aq.copy(), [], [], 0.0))

    while queue:
        vars_key, aq, order, queries_list, prev_cost = queue.popleft()
        if vars_key == component_set:
            continue

        reducible_in_comp = aq.get_reducible_idxs_for_component(component)
        candidates: list[tuple[float, int, list]] = []
        for idx in reducible_in_comp:
            cost, reduced_vars = _cost_of_reduce(idx, aq)
            total_cost = cost + prev_cost
            new_vars = vars_key | frozenset(reduced_vars)
            bound = float("inf")
            for vars2 in max_subquery_costs:
                if vars2 >= new_vars:
                    bound = min(bound, max_subquery_costs[vars2])
            best_known = optimal_orders.get(
                new_vars, (None, None, None, float("inf"))
            )[3]
            cheapest_here = min(best_known, bound)
            if total_cost > cheapest_here:
                continue
            for superset, sup_cost in seen_min_cost.items():
                if superset >= new_vars and sup_cost < total_cost:
                    break
            else:
                candidates.append((total_cost, idx, reduced_vars))

        for total_cost, idx, reduced_vars in candidates:
            new_vars = vars_key | frozenset(reduced_vars)
            best_known = optimal_orders.get(
                new_vars, (None, None, None, float("inf"))
            )[3]
            if total_cost > best_known:
                continue
            for superset, sup_cost in seen_min_cost.items():
                if superset >= new_vars and sup_cost < total_cost:
                    break
            else:
                new_aq = aq.copy()
                reduce_query = new_aq.reduce_idx(idx)
                new_queries = list(queries_list) + [reduce_query]
                new_order = list(order) + [idx]
                optimal_orders[new_vars] = (new_order, new_queries, new_aq, total_cost)
                seen_min_cost[new_vars] = min(
                    seen_min_cost.get(new_vars, float("inf")), total_cost
                )
                queue.appendleft((new_vars, new_aq, new_order, new_queries, total_cost))

    optimal_subquery_costs = OrderedDict()
    for vars_key in optimal_orders:
        optimal_subquery_costs[vars_key] = optimal_orders[vars_key][3]

    if component_set not in optimal_orders:
        raise RuntimeError(
            "branch_and_bound_dfs: no complete reduction order for component "
            f"{component!r}; reducible idxs on input: "
            f"{input_aq.get_reducible_idxs()!r}"
        )
    return (optimal_orders[component_set], optimal_subquery_costs)


bnb_dfs = branch_and_bound_dfs


def pruned_query_to_plan_dfs(
    input_aq: AnnotatedQuery,
    use_components: bool = True,
    use_greedy: bool = False,
) -> tuple[list[Query], float]:
    """
    Same as ``pruned_query_to_plan`` but uses depth-first branch-and-bound
    for the exact phase.
    """
    total_cost = 0.0
    elimination_order: list = []
    queries: list[Query] = []
    cur_aq = input_aq.copy()

    while cur_aq.get_reducible_idxs():
        component = cur_aq.connected_components[0]

        if not use_components:
            component = list(
                set().union(*(set(c) for c in cur_aq.connected_components))
            )
        greedy_result = branch_and_bound_dfs(cur_aq, component, OrderedDict())
        (
            (greedy_order, _, _, greedy_cost),
            greedy_subquery_costs,
        ) = greedy_result

        if (len(component) > 12) or use_greedy:
            elimination_order.extend(greedy_order)
            for idx in greedy_order:
                reduce_query = cur_aq.reduce_idx(idx)
                queries.append(reduce_query)
            total_cost += greedy_cost
            continue

        (exact_order, _, _, exact_cost), _ = branch_and_bound_dfs(
            cur_aq, component, greedy_subquery_costs
        )
        elimination_order.extend(exact_order)
        for idx in exact_order:
            reduce_query = cur_aq.reduce_idx(idx)
            queries.append(reduce_query)
        total_cost += exact_cost

    remaining_q = cur_aq.get_remaining_query()
    queries.append(remaining_q)
    return queries, total_cost
