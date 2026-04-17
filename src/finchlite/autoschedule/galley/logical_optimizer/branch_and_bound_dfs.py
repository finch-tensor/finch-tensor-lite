"""
Depth-first branch-and-bound for Galley reduction order (same optimum as layered
``branch_and_bound`` with ``k=`` ``float('inf')``).

Pruning uses the minimum cost among all **known** supersets of the child state's
variable set: greedy ``k=1`` bounds, entries in the DFS ``memo``, and the best
complete cost found so far.
"""

from __future__ import annotations

from collections import OrderedDict, deque

from ....finch_logic import Query
from .annotated_query import AnnotatedQuery
from .branch_and_bound import branch_and_bound


def _cost_of_reduce(idx, aq: AnnotatedQuery) -> tuple[float, list]:
    """Return (cost, reduced_vars) for reducing ``idx`` in ``aq``."""
    _, _, _, reduced_idxs = aq.get_reduce_query(idx)
    cost = aq.get_cost_of_reduce_idx(idx)
    return cost, list(reduced_idxs)


def branch_and_bound_dfs(
    input_aq: AnnotatedQuery,
    component: list,
    max_subquery_costs: OrderedDict,
) -> tuple:
    """
    Exact branch-and-bound via iterative DFS (cheapest child expanded first).

    Returns ``((order, queries, aq, cost), optimal_subquery_costs)`` where the
    second element is an empty ``OrderedDict`` (greedy bounds use layered BnB).

    Raises
    ------
    RuntimeError
        If no complete elimination order exists for ``component``.
    """
    component_set = frozenset(component)
    memo: dict[frozenset, float] = {frozenset(): 0.0}
    best_complete = float("inf")
    best_state: tuple[list, list, AnnotatedQuery, float] | None = None

    stack: deque[tuple[frozenset, list, list, AnnotatedQuery, float]] = deque()
    stack.append((frozenset(), [], [], input_aq, 0.0))

    while stack:
        vars_key, order, queries, aq, cost = stack.popleft()

        if cost > memo.get(vars_key, float("inf")):
            continue

        if vars_key == component_set:
            if cost < best_complete:
                best_complete = cost
                best_state = (order, queries, aq, cost)
            continue

        if cost >= best_complete:
            continue

        reducible_in_comp = aq.get_reducible_idxs_for_component(component)
        children: list[tuple[frozenset, list, list, AnnotatedQuery, float]] = []
        for idx in reducible_in_comp:
            step_cost, reduced_vars = _cost_of_reduce(idx, aq)
            new_cost = cost + step_cost
            new_vars = vars_key | frozenset(reduced_vars)

            # Getting all supersets and getting min cost
            sup_best = float("inf")
            for v, c in max_subquery_costs.items():
                if v >= new_vars:
                    sup_best = min(sup_best, c)
            for v, c in memo.items():
                if v >= new_vars:
                    sup_best = min(sup_best, c)
            if best_complete < float("inf"):
                sup_best = min(sup_best, best_complete)
            prev_best = memo.get(new_vars, float("inf"))

            if new_cost > sup_best:
                continue
            if new_cost >= prev_best:
                continue

            memo[new_vars] = new_cost
            new_aq = aq.copy()
            reduce_query = new_aq.reduce_idx(idx)
            new_order = list(order) + [idx]
            new_queries = list(queries) + [reduce_query]
            children.append((new_vars, new_order, new_queries, new_aq, new_cost))

        children.sort(key=lambda x: x[4])
        for ch in reversed(children):
            stack.appendleft(ch)

    if best_state is None:
        raise RuntimeError(
            "branch_and_bound_dfs: no complete reduction order for component "
            f"{component!r}; reducible idxs on input: "
            f"{input_aq.get_reducible_idxs()!r}"
        )
    return (best_state, OrderedDict())


def pruned_query_to_plan_dfs(
    input_aq: AnnotatedQuery,
    use_components: bool = True,
    use_greedy: bool = False,
) -> tuple[list[Query], float]:
    """
    Like ``pruned_query_to_plan``, but the exact phase uses ``branch_and_bound_dfs``.
    Greedy bounds still use layered ``branch_and_bound`` with ``k=1``.
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
        greedy_result = branch_and_bound(cur_aq, component, 1, OrderedDict())
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