"""
Branch-and-bound optimizer for Galley query planning.
Converted from Julia.
Removed general cache. Both alias_hash and cost_cache

TODO: add dfs
"""

from collections import OrderedDict, deque

from ....finch_logic import Query
from .annotated_query import AnnotatedQuery


def _aq_with_stats(aq: AnnotatedQuery) -> AnnotatedQuery:
    """
    Return a copy of aq with bindings/cache/cache_point preserved.
    Saves the stats for cost estimation and make sure we dont mutate the original aq
    as we might use it later.
    """
    c = aq.copy()
    for attr in ("bindings", "cache", "cache_point"):
        if hasattr(aq, attr):
            setattr(c, attr, getattr(aq, attr))
    return c


def _reducible_idxs_for_component(
    aq: AnnotatedQuery,
    component: list,
) -> list:
    """Make sure it is a list, precommit throws error."""
    comp_set = set(component)
    return [idx for idx in aq.get_reducible_idxs() if idx in comp_set]


def _cost_of_reduce(idx, aq: AnnotatedQuery) -> tuple[float, list]:
    """
    Return (cost, reduced_vars) for reducing idx in aq.
    Used to score each candidate reduction and to know which indices
    get eliminated (reduced_vars) for the branch-and-bound state key.
    """
    _, _, _, reduced_idxs = aq.get_reduce_query(idx)
    cost = aq.get_cost_of_reduce_idx(idx)
    return cost, list(reduced_idxs)


def branch_and_bound(
    input_aq: AnnotatedQuery,
    component: list,
    k: float,
    max_subquery_costs: OrderedDict,
) -> tuple:
    """
    Branch-and-bound search for optimal reduction order within a component.

    Returns ``((order, queries, aq, cost), optimal_subquery_costs)``.

    Raises
    ------
    RuntimeError
        If no complete elimination order exists for ``component``.
    """
    # --- Initialize ---
    # optimal_orders[vars_key] = (order, queries, aq, cost) for each state.
    # vars_key = frozenset of indices already reduced. Start with empty set.
    optimal_orders: OrderedDict[frozenset, tuple] = OrderedDict(
        [(frozenset(), ([], [], input_aq, 0.0))]
    )
    prev_new_optimal_orders = optimal_orders

    # --- Main loop: one iteration per reduction step ---
    for _ in range(len(component)):
        # best_idx_ext[new_vars] = (aq, idx, old_order, old_queries, total_cost)
        # Tracks best way to reach each new state in this step.
        best_idx_ext: OrderedDict[frozenset, tuple] = OrderedDict()

        # --- Extend each current state by trying every possible next reduction ---
        for vars_key, pc in prev_new_optimal_orders.items():
            old_order, old_queries, aq, prev_cost = pc
            reducible_in_comp = _reducible_idxs_for_component(aq, component)
            for idx in reducible_in_comp:
                # Cost of this reduction and which vars it eliminates
                cost, reduced_vars = _cost_of_reduce(idx, aq)
                total_cost = cost + prev_cost
                new_vars = vars_key | frozenset(reduced_vars)

                # --- Pruning bound: if we have a cheaper cost for a superset
                # (vars2 >= new_vars), that gives an upper bound; we can prune
                # if our total_cost exceeds it.
                bound = float("inf")
                for vars2 in max_subquery_costs:
                    if vars2 >= new_vars:
                        bound = min(bound, max_subquery_costs[vars2])

                # cheapest_cost = best known cost to reach new_vars (from any path)
                best_val = best_idx_ext.get(
                    new_vars, (None, None, None, None, float("inf"))
                )
                cheapest_cost = min(
                    best_val[4],
                    optimal_orders.get(new_vars, (None, None, None, float("inf")))[3],
                    bound,
                )
                # Keep this candidate only if it is at least as good as cheapest
                if total_cost <= cheapest_cost:
                    best_idx_ext[new_vars] = (
                        aq,
                        idx,
                        old_order,
                        old_queries,
                        total_cost,
                    )

        if len(best_idx_ext) == 0:
            break

        # --- Keep only top-k by cost (greedy k=1 keeps single best per state) ---
        if k == float("inf"):
            num_to_keep = len(best_idx_ext)
        else:
            num_to_keep = int(min(k, len(best_idx_ext)))

        sorted_items = sorted(best_idx_ext.items(), key=lambda x: x[1][4])
        top_k_idx_ext = OrderedDict(sorted_items[:num_to_keep])

        # --- Apply reductions and build next round of states ---
        new_optimal_orders = OrderedDict()
        for new_vars, idx_ext_info in top_k_idx_ext.items():
            aq, idx, old_order, old_queries, cost = idx_ext_info
            new_aq = _aq_with_stats(aq)
            reduce_query = new_aq.reduce_idx(idx)
            new_queries = list(old_queries) + [reduce_query]
            new_order = list(old_order) + [idx]
            new_optimal_orders[new_vars] = (new_order, new_queries, new_aq, cost)

        optimal_orders.update(new_optimal_orders)
        prev_new_optimal_orders = new_optimal_orders

    # --- Build subquery cost map for pruning (only when k=1 greedy) ---
    optimal_subquery_costs = OrderedDict()
    if k == 1:
        for vars_key in optimal_orders:
            optimal_subquery_costs[vars_key] = optimal_orders[vars_key][3]

    # --- Return result if we found a full order for the component ---
    component_set = frozenset(component)
    if component_set not in optimal_orders:
        raise RuntimeError(
            "branch_and_bound: no complete reduction order for component "
            f"{component!r}; reducible idxs on input: "
            f"{input_aq.get_reducible_idxs()!r}"
        )
    return (optimal_orders[component_set], optimal_subquery_costs)


def branch_and_bound_dfs(
    input_aq: AnnotatedQuery,
    component: list,
    k: float,
    max_subquery_costs: OrderedDict,
) -> tuple:
    """
    Same contract as :func:`branch_and_bound`.

    For ``k != float("inf")``, delegates to :func:`branch_and_bound` so layered
    semantics (including greedy ``k=1``) match exactly.

    For ``k == float("inf")``, runs iterative DFS: ``deque`` stack (``appendleft`` /
    ``popleft`` for LIFO on the left), ``memo[vars_key]`` minimum cumulative cost
    to reach each eliminated-index set, the same ``max_subquery_costs`` superset
    bound as the layered code, and optional incumbent pruning when a partial
    state's cost is already at or above the best complete cost found so far.
    """
    if k != float("inf"):
        return branch_and_bound(input_aq, component, k, max_subquery_costs)

    component_set = frozenset(component)
    memo: dict[frozenset, float] = {frozenset(): 0.0}
    best_complete: tuple | None = None
    best_complete_cost = float("inf")

    stack: deque = deque()
    stack.appendleft(
        (frozenset(), input_aq, [], [], 0.0)
    )

    while stack:
        vars_key, aq, order, queries, cost = stack.popleft()
        if cost > memo.get(vars_key, float("inf")):
            continue

        if vars_key == component_set:
            if cost < best_complete_cost:
                best_complete_cost = cost
                best_complete = (order, queries, aq, cost)
            continue

        if cost >= best_complete_cost:
            continue

        reducible_in_comp = _reducible_idxs_for_component(aq, component)
        for idx in reducible_in_comp:
            step_cost, reduced_vars = _cost_of_reduce(idx, aq)
            total_cost = cost + step_cost
            new_vars = vars_key | frozenset(reduced_vars)

            bound = float("inf")
            for vars2 in max_subquery_costs:
                if vars2 >= new_vars:
                    bound = min(bound, max_subquery_costs[vars2])

            if total_cost > bound:
                continue

            prev_best = memo.get(new_vars, float("inf"))
            if total_cost >= prev_best:
                continue

            memo[new_vars] = total_cost
            new_aq = _aq_with_stats(aq)
            reduce_query = new_aq.reduce_idx(idx)
            new_queries = list(queries) + [reduce_query]
            new_order = list(order) + [idx]
            stack.appendleft((new_vars, new_aq, new_order, new_queries, total_cost))

    if best_complete is None:
        raise RuntimeError(
            "This should not happen! Maybe delete this line because its impossible."
        )

    optimal_subquery_costs = OrderedDict()
    return (best_complete, optimal_subquery_costs)


def pruned_query_to_plan(
    input_aq: AnnotatedQuery,
    use_greedy: bool = False,
) -> tuple[list[Query], float]:
    """
    Pruned optimizer: greedy first for bounds, then exact with pruning.
    Returns (queries, total_cost).
    """
    total_cost = 0.0
    elimination_order: list = []
    queries: list[Query] = []
    cur_aq = _aq_with_stats(input_aq)

    # --- Process components until no more reducible indices ---
    # Do this instead of for comp in components because components
    # are recomputed in reduce_idx same as julia code
    while cur_aq.get_reducible_idxs():
        component = cur_aq.connected_components[0]

        # --- Run greedy (k=1) to get subquery costs for pruning bounds ---
        greedy_result = branch_and_bound(cur_aq, component, 1, OrderedDict())
        (
            (greedy_order, _, _, greedy_cost),
            greedy_subquery_costs,
        ) = greedy_result

        # --- Large components or use_greedy: use greedy order directly ---
        if len(component) >= 10 or use_greedy:
            elimination_order.extend(greedy_order)
            for idx in greedy_order:
                reduce_query = cur_aq.reduce_idx(idx)
                queries.append(reduce_query)
            total_cost += greedy_cost
            continue

        # --- Exact search with pruning: greedy_subquery_costs bounds the search ---
        (exact_order, _, _, exact_cost), _ = branch_and_bound(
            cur_aq, component, float("inf"), greedy_subquery_costs
        )
        elimination_order.extend(exact_order)
        for idx in exact_order:
            reduce_query = cur_aq.reduce_idx(idx)
            queries.append(reduce_query)
        total_cost += exact_cost

    # --- Append remaining (non-reducible) query and fix output name ---
    remaining_q = cur_aq.get_remaining_query()
    if remaining_q is not None:
        queries.append(remaining_q)
    if queries:
        last_query = queries[-1]
        if last_query.lhs != cur_aq.output_name:
            queries[-1] = Query(cur_aq.output_name, last_query.rhs)
    return queries, total_cost


def pruned_query_to_plan_dfs(
    input_aq: AnnotatedQuery,
    use_greedy: bool = False,
) -> tuple[list[Query], float]:
    """
    Pruned optimizer like :func:`pruned_query_to_plan`, using :func:`branch_and_bound_dfs`
    for the greedy and exact component passes (iterative DFS for ``k=inf``; other ``k``
    delegate to the layered implementation).
    Returns (queries, total_cost).
    """
    total_cost = 0.0
    elimination_order: list = []
    queries: list[Query] = []
    cur_aq = _aq_with_stats(input_aq)

    while cur_aq.get_reducible_idxs():
        component = cur_aq.connected_components[0]

        # Maybe remove, let dfs run
        greedy_result = branch_and_bound_dfs(cur_aq, component, 1, OrderedDict())
        (
            (greedy_order, _, _, greedy_cost),
            greedy_subquery_costs,
        ) = greedy_result

        if len(component) >= 10 or use_greedy:
            elimination_order.extend(greedy_order)
            for idx in greedy_order:
                reduce_query = cur_aq.reduce_idx(idx)
                queries.append(reduce_query)
            total_cost += greedy_cost
            continue

        (exact_order, _, _, exact_cost), _ = branch_and_bound_dfs(
            cur_aq, component, float("inf"), greedy_subquery_costs
        )
        elimination_order.extend(exact_order)
        for idx in exact_order:
            reduce_query = cur_aq.reduce_idx(idx)
            queries.append(reduce_query)
        total_cost += exact_cost

    remaining_q = cur_aq.get_remaining_query()
    if remaining_q is not None:
        queries.append(remaining_q)
    if queries:
        last_query = queries[-1]
        if last_query.lhs != cur_aq.output_name:
            queries[-1] = Query(cur_aq.output_name, last_query.rhs)
    return queries, total_cost


def exact_query_to_plan(input_aq: AnnotatedQuery) -> tuple[list[Query], float]:
    """
    Exact optimizer: branch-and-bound with k=Inf for each component.
    Returns (queries, total_cost).
    """
    total_cost = 0.0
    elimination_order: list = []
    cur_aq = _aq_with_stats(input_aq)

    # --- Run exact branch-and-bound on each component, collect order ---
    for component in input_aq.connected_components:
        (exact_order, _, _, exact_cost), _ = branch_and_bound(
            cur_aq, component, float("inf"), OrderedDict()
        )
        elimination_order.extend(exact_order)
        total_cost += exact_cost

    # --- Rebuild queries by applying reductions in elimination order ---
    queries = []
    cur_aq = _aq_with_stats(input_aq)
    for idx in elimination_order:
        if idx in cur_aq.get_reducible_idxs():
            q = cur_aq.reduce_idx(idx)
            queries.append(q)
    remaining_q = cur_aq.get_remaining_query()
    if remaining_q is not None:
        queries.append(remaining_q)
    if queries and queries[-1].lhs != cur_aq.output_name:
        queries[-1] = Query(cur_aq.output_name, queries[-1].rhs)
    return queries, total_cost
