from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from finchlite.finch_logic import LogicNode


@dataclass
class AnnotatedQuery:
    ST: type
    output_name: Any
    reduce_idxs: list[str]
    point_expr: "LogicNode"
    idx_lowest_root: OrderedDict[str, int]
    idx_op: OrderedDict[str, Any]
    idx_init: OrderedDict[str, Any]
    id_to_node: OrderedDict[int, "LogicNode"]
    parent_idxs: OrderedDict[str, list[str]]
    original_idx: OrderedDict[str, str]
    connected_components: list[list[str]]
    connected_idxs: OrderedDict[str, set[str]]
    output_order: list[str] | None = None
    output_format: list[Any] | None = None


def copy_aq(aq: AnnotatedQuery) -> AnnotatedQuery:
    return AnnotatedQuery(
        ST=aq.ST,
        output_name=aq.output_name,
        reduce_idxs=list(aq.reduce_idxs),
        point_expr=aq.point_expr,
        idx_lowest_root=OrderedDict(aq.idx_lowest_root.items()),
        idx_op=OrderedDict(aq.idx_op.items()),
        idx_init=OrderedDict(aq.idx_init.items()),
        id_to_node=OrderedDict(aq.id_to_node.items()),
        parent_idxs=OrderedDict((m, list(n)) for m, n in aq.parent_idxs.items()),
        original_idx=OrderedDict(aq.original_idx.items()),
        connected_components=[list(n) for n in aq.connected_components],
        connected_idxs=OrderedDict((m, set(n)) for m, n in aq.connected_idxs.items()),
        output_order=None if aq.output_order is None else list(aq.output_order),
        output_format=None if aq.output_format is None else list(aq.output_format),
    )


def get_reducible_idxs(aq: AnnotatedQuery) -> list[str]:
    return [idx for idx in aq.reduce_idxs if len(aq.parent_idxs.get(idx, [])) == 0]
