from typing import Any, Optional, List, Tuple, Set
from dataclasses import dataclass
from collections import OrderedDict

from finchlite.galley.TensorStats.tensor_stats import TensorStats
from finchlite.galley.TensorStats.dc_stats import estimate_nnz

@dataclass
class AnnotatedQuery:
    ST: type
    output_name: Any
    reduce_idxs: List[str]
    idx_lowest_root: OrderedDict[str, int]
    idx_op: OrderedDict[str, Any]
    idx_init: OrderedDict[str, Any]
    parent_idxs: OrderedDict[str, List[str]]
    original_idx: OrderedDict[str, str]
    connected_components: List[List[str]]
    connected_idxs: OrderedDict[str, set[str]]

def copy_aq(aq: AnnotatedQuery) -> AnnotatedQuery:
    return AnnotatedQuery(
        ST=aq.ST,
        output_name=aq.output_name,
        reduce_idxs=list(aq.reduce_idxs),
        idx_lowest_root=OrderedDict(aq.idx_lowest_root.items()),
        idx_op=OrderedDict(aq.idx_op.items()),
        idx_init=OrderedDict(aq.idx_init.items()),
        parent_idxs=OrderedDict((m, list(n)) for m, n in aq.parent_idxs.items()),
        original_idx=OrderedDict(aq.original_idx.items()),
        connected_components=[list(n) for n in aq.connected_components],
        connected_idxs=OrderedDict((m, set(n)) for m, n in aq.connected_idxs.items()),
    )

def get_reducible_idxs(aq) -> List[str]:
    return [idx for idx in aq.reduce_idxs if not aq.parent_idxs[idx]]

def is_dense(mat_stats: Any, mat_size: float) -> bool:
    sparsity = mat_size / TensorStats.get_dim_space_size(mat_stats, TensorStats.index_set(mat_stats))
    return sparsity > 0.5

def cost_of_reduce(
    reduce_idx: str,
    aq: AnnotatedQuery,
    cache: Optional[OrderedDict[int, float]] = None,
    alias_hash: Optional[OrderedDict[str, int]] = None,
) -> Tuple[float, Set[str]]:
    """
    Compute the estimated cost of materializing the reduction for `reduce_idx`.

    Returns:
        cost: total estimated cost including any forced-transpose penalty.
        reduced_idxs: the (possibly larger) set of indices that this reduction would eliminate.
    """
    if cache is None:
        cache = OrderedDict()
    if alias_hash is None:
        alias_hash = OrderedDict()

    query, _, _, reduced_idxs = get_reduce_query(reduce_idx, aq)
    cache_key = id(query.expr)

    if cache_key not in cache:
        comp_stats = query.expr.arg.stats
        mat_stats  = query.expr.stats
        mat_size   = estimate_nnz(mat_stats)

        mat_factor = DenseAllocateCost if is_dense(mat_stats, mat_size) else SparseAllocateCost
        comp_factor = len(TensorStats.index_set(comp_stats)) * ComputeCost
        cost = estimate_nnz(comp_stats) * comp_factor + mat_size * mat_factor

        forced_transpose_cost = get_forced_transpose_cost(query.expr)

        total = cost + forced_transpose_cost

        if total == float("inf"):
            print(f"INFINITE QUERY FOR: {reduce_idx}")
            print(query)
            print(f"COMP STATS: {estimate_nnz(comp_stats)}")
            print(comp_stats)
            print(f"MAT STATS: {estimate_nnz(mat_stats)}")
            print(mat_stats)

        cache[cache_key] = total

    return cache[cache_key], reduced_idxs