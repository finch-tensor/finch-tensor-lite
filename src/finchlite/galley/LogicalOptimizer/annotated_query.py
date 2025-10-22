from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from finchlite.galley.TensorStats.tensor_stats import TensorStats


@dataclass
class AnnotatedQuery:
    ST: type
    output_name: Any
    reduce_idxs: list[str]
    idx_lowest_root: OrderedDict[str, int]
    idx_op: OrderedDict[str, Any]
    idx_init: OrderedDict[str, Any]
    parent_idxs: OrderedDict[str, list[str]]
    original_idx: OrderedDict[str, str]
    connected_components: list[list[str]]
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


def get_reducible_idxs(aq: AnnotatedQuery) -> list[str]:
    return [idx for idx in aq.reduce_idxs if len(aq.parent_idxs.get(idx, [])) == 0]


def is_dense(mat_stats: Any, mat_size: float) -> bool:
    sparsity = mat_size / TensorStats.get_dim_space_size(
        mat_stats, TensorStats.index_set(mat_stats)
    )
    return sparsity > 0.5
