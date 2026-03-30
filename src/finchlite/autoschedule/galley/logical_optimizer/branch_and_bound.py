"""
Branch-and-bound optimizer for Galley query planning.
Converted from Julia.
Removed general cache. Both alias_hash and cost_cache
"""

from collections import OrderedDict

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
    query, _, _, reduced_idxs = aq.get_reduce_query(idx)
    cost = aq.get_cost_of_reduce_idx(idx)
    return cost, list(reduced_idxs)


def branch_and_bound(
    input_aq: AnnotatedQuery,
    component: list,
    k: float,
    max_subquery_costs: OrderedDict,
) -> tuple | None:
    """
    Branch-and-bound search for optimal reduction order within a component.
    Returns (order, queries, aq, cost) and optimal_subquery_costs;
    or None if no complete order found.
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
        num_to_keep = min(k, len(best_idx_ext))
        if k == float("inf"):
            num_to_keep = len(best_idx_ext)

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
    if component_set in optimal_orders:
        print(f"Component: {component}")
        print(f"Optimal Orders: {list(optimal_orders.keys())}")
        print(f"Reducible Idxs: {input_aq.get_reducible_idxs()}")
        return (optimal_orders[component_set], optimal_subquery_costs)
    else:
        # No complete order; debug output
        # TODO: rasie error
        #print(f"Component: {component}")
        #print(f"Optimal Orders: {list(optimal_orders.keys())}")
        #print(f"Reducible Idxs: {input_aq.get_reducible_idxs()}")
        return None


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
    # Do this instead of for comp in components because components are recomputed in reduce_idx
    # same as julia code
    while cur_aq.get_reducible_idxs():
        component = cur_aq.connected_components[0]

        # --- Run greedy (k=1) to get subquery costs for pruning bounds ---
        greedy_result = branch_and_bound(cur_aq, component, 1, OrderedDict())
        if greedy_result is None:
            break
        (
            (
                greedy_order,
                greedy_queries,
                greedy_aq,
                greedy_cost,
            ),
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
        exact_result = branch_and_bound(
            cur_aq, component, float("inf"), greedy_subquery_costs
        )
        if exact_result is not None:
            # Use exact order
            (exact_order, exact_queries, exact_aq, exact_cost), _ = exact_result
            elimination_order.extend(exact_order)
            for idx in exact_order:
                reduce_query = cur_aq.reduce_idx(idx)
                queries.append(reduce_query)
            total_cost += exact_cost
        else:
            # Fallback to greedy if exact search fails
            print("WARNING: Pruned Optimizer Failed. Falling Back to Greedy Plan.")
            elimination_order.extend(greedy_order)
            for idx in greedy_order:
                reduce_query = cur_aq.reduce_idx(idx)
                queries.append(reduce_query)
            total_cost += greedy_cost

    # --- Append remaining (non-reducible) query and fix output name ---
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
        result = branch_and_bound(cur_aq, component, float("inf"), OrderedDict())
        if result is None:
            continue
        (exact_order, _, _, exact_cost), _ = result
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
