"""
Branch-and-bound optimizer for Galley query planning.
Converted from Julia.
Removed general cache. Both alias_hash and cost_cache

``branch_and_bound`` is layered BFS-style; ``branch_and_bound_dfs`` uses an
iterative DFS (stack + memo) for ``k == float("inf")``, storing
``memo[vars_key] = (cost, elimination_order)`` per set (suffix relaxation
disabled).
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


def _vars_key_sort_key(vk: frozenset) -> tuple:
    """Deterministic ordering for OrderedDict keys (subset size, then repr)."""
    return (len(vk), tuple(sorted(repr(x) for x in vk)))


def _dfs_memo_cost(memo: dict[frozenset, tuple[float, list]], vk: frozenset) -> float:
    """Best cumulative cost for ``vk``, or inf if unseen."""
    ent = memo.get(vk)
    return ent[0] if ent is not None else float("inf")


def _print_dfs_memo_snapshot(stage: str, n: int, memo: dict[frozenset, tuple[float, list]]) -> None:
    """Print ``memo`` after each stack pop / memo assign (DFS ``k=inf``)."""
    print(f"\n--- branch_and_bound_dfs memo {stage} #{n} ---")
    for vk in sorted(memo.keys(), key=_vars_key_sort_key):
        c, ord_ = memo[vk]
        label = "∅" if not vk else "{" + ", ".join(sorted(repr(x) for x in vk)) + "}"
        print(f"  {label} -> (cost={c}, order={[repr(x) for x in ord_]})")
    print(f"  [n_entries={len(memo)}]")


# def _relax_suffix_induced_sets(
#     memo: dict[frozenset, tuple[float, list]],
#     order: list,
#     total_cost: float,
# ) -> None:
#     """
#     Suffix-induced subset entries (optional tighten after DFS, not part of search).
#
#     For each non-empty suffix of the **best complete** ``order``, build
#     ``S = frozenset(suffix)`` (indices appearing in that suffix). If the **full**
#     plan cost ``total_cost`` strictly improves ``memo[S]``, set
#     ``memo[S] = (total_cost, suffix)``. Example: order [a,b,c,d] also touches
#     keys for ``{b,c,d}``, ``{c,d}``, ``{d}`` with cost ``total_cost`` — not the
#     same as “minimum cumulative cost to reach S” from the main DFS loop.
#     """
#     n = len(order)
#     for k in range(n):
#         suffix = order[k:]
#         if not suffix:
#             continue
#         # Subset S = indices in this suffix; may duplicate keys already filled by search.
#         s_set = frozenset(suffix)
#         prev = _dfs_memo_cost(memo, s_set)
#         if total_cost < prev:
#             # Full-plan total and suffix order; only applied when better than prior memo[S].
#             memo[s_set] = (total_cost, list(suffix))


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
    # Maybe delete below since it should never happen
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
    Same contract as :func:`branch_and_bound`, plus a third return when
    ``k == float("inf")``.

    For ``k != float("inf")``, delegates to :func:`branch_and_bound` and returns
    ``(result, None)`` for the third element.

    For ``k == float("inf")``, runs iterative DFS: ``deque`` stack,
    ``memo[vars_key] = (cumulative_cost, elimination_order)`` for each eliminated-
    index set (single dict — cost and permutation updated together),
    ``max_subquery_costs`` superset bound, incumbent pruning.

    Prints ``memo`` after each stack ``pop`` and each ``memo`` assignment.
    Returns ``(best_complete, optimal_subquery_costs, optimal_perm_by_vars)``.
    """
    if k != float("inf"):
        res = branch_and_bound(input_aq, component, k, max_subquery_costs)
        return (*res, None)

    component_set = frozenset(component)
    memo: dict[frozenset, tuple[float, list]] = {frozenset(): (0.0, [])}
    best_complete: tuple | None = None
    best_complete_cost = float("inf")

    stack: deque = deque()
    stack.appendleft(
        (frozenset(), input_aq, [], [], 0.0)
    )

    pop_idx = 0
    assign_idx = 0
    while stack:
        vars_key, aq, order, queries, cost = stack.popleft()
        pop_idx += 1
        if __debug__:
            _print_dfs_memo_snapshot("pop", pop_idx, dict(memo))
        # Stale frame: a cheaper path to this vars_key already updated memo[vars_key].
        if cost > _dfs_memo_cost(memo, vars_key):
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

            prev_best = _dfs_memo_cost(memo, new_vars)
            if total_cost >= prev_best:
                continue

            # Search path: best cost so far to eliminated-index set ``new_vars`` (prefix walk).
            memo[new_vars] = (total_cost, list(order) + [idx])
            assign_idx += 1
            _print_dfs_memo_snapshot("assign", assign_idx, dict(memo))
            new_aq = _aq_with_stats(aq)
            reduce_query = new_aq.reduce_idx(idx)
            new_queries = list(queries) + [reduce_query]
            new_order = list(order) + [idx]
            stack.appendleft((new_vars, new_aq, new_order, new_queries, total_cost))

    if best_complete is None:
        raise RuntimeError(
            "branch_and_bound_dfs: no complete reduction order for component "
            f"{component!r}; reducible idxs on input: "
            f"{input_aq.get_reducible_idxs()!r}"
        )

    # best_order, _, _, best_cost = best_complete
    # Extra memo keys: suffix-induced subsets of best_order get (best_cost, suffix) if
    # that improves memo[S] — separate from assignments in the DFS loop above.
    # _relax_suffix_induced_sets(memo, best_order, best_cost)

    sorted_keys = sorted(memo.keys(), key=_vars_key_sort_key)
    # --- optimal_subquery_costs and optimal_perm_by_vars only used for testing ---
    optimal_subquery_costs = OrderedDict((vk, memo[vk][0]) for vk in sorted_keys)
    optimal_perm_by_vars = OrderedDict((vk, memo[vk][1]) for vk in sorted_keys)
    return (best_complete, optimal_subquery_costs, optimal_perm_by_vars)


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
            _,
        ) = greedy_result

        if len(component) >= 10 or use_greedy:
            elimination_order.extend(greedy_order)
            for idx in greedy_order:
                reduce_query = cur_aq.reduce_idx(idx)
                queries.append(reduce_query)
            total_cost += greedy_cost
            continue

        (exact_order, _, _, exact_cost), _, _ = branch_and_bound_dfs(
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
