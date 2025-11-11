from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from utility import PostOrderDFS

from finchlite.finch_logic import (
    Aggregate,
    Alias,
    LogicNode,
    MapJoin,
    Plan,
    Query,
)


@dataclass
class AnnotatedQuery:
    ST: type
    output_name: Alias | None
    reduce_idxs: list[str]
    point_expr: "LogicNode"
    idx_lowest_root: OrderedDict[str, int]
    idx_op: OrderedDict[str, Any]
    idx_init: OrderedDict[str, Any]
    hash_to_node: OrderedDict[int, "LogicNode"]
    parent_idxs: OrderedDict[str, list[str]]
    original_idx: OrderedDict[str, str]
    connected_components: list[list[str]]
    connected_idxs: OrderedDict[str, set[str]]
    output_order: list[str] | None = None
    output_format: list[Any] | None = None


def copy_aq(aq: AnnotatedQuery) -> AnnotatedQuery:
    """
    Make a structured copy of an AnnotatedQuery.
    """
    return AnnotatedQuery(
        ST=aq.ST,
        output_name=aq.output_name,
        reduce_idxs=list(aq.reduce_idxs),
        point_expr=aq.point_expr,
        idx_lowest_root=OrderedDict(aq.idx_lowest_root.items()),
        idx_op=OrderedDict(aq.idx_op.items()),
        idx_init=OrderedDict(aq.idx_init.items()),
        hash_to_node=OrderedDict(aq.hash_to_node.items()),
        parent_idxs=OrderedDict((m, list(n)) for m, n in aq.parent_idxs.items()),
        original_idx=OrderedDict(aq.original_idx.items()),
        connected_components=[list(n) for n in aq.connected_components],
        connected_idxs=OrderedDict((m, set(n)) for m, n in aq.connected_idxs.items()),
        output_order=None if aq.output_order is None else list(aq.output_order),
        output_format=None if aq.output_format is None else list(aq.output_format),
    )


def get_reducible_idxs(aq: AnnotatedQuery) -> list[str]:
    """
    Indices eligible to be reduced immediately (no parents).

    Parameters
    ----------
    aq : AnnotatedQuery
        Query containing the candidate reduction indices and their parent map.

    Returns
    -------
    list[str]
        Indices in `aq.reduce_idxs` with zero parents.
    """
    return [idx for idx in aq.reduce_idxs if len(aq.parent_idxs.get(idx, [])) == 0]

def get_idx_connected_components(
    parent_idxs: dict[str, Iterable[str]],
    connected_idxs: dict[str, Iterable[str]],
) -> list[list[str]]:
    """
    Compute connected components of indices and order those components by
    parent/child constraints.

    Parameters
    ----------
    parent_idxs : Dict[str, Iterable[str]]
        Mapping from an index to the set/iterable of its parent indices.
    connected_idxs : Dict[str, Iterable[str]]
        Mapping from an index to the set/iterable of indices considered
        "connected" to it (undirected neighbors). Only connections between
        non-parent pairs are used to form components.

    Returns
    -------
    List[List[str]]
        A list of components, each a list of index names. Components are
        ordered so that any component containing a parent appears before any
        component containing its child. Within a component, indices are kept
        in the insertion order of `connected_idxs`.
    """
    parent_map = {k: set(v) for k, v in parent_idxs.items()}
    conn_map: OrderedDict[str, set[str]] = OrderedDict(
        (k, set(v)) for k, v in connected_idxs.items()
    )

    component_ids: OrderedDict[str, int] = OrderedDict(
        (x, i) for i, x in enumerate(conn_map.keys())
    )
    finished = False
    while not finished:
        finished = True
        for idx1, neighbours in conn_map.items():
            for idx2 in neighbours:
                if idx2 in parent_map.get(idx1, set()) or idx1 in parent_map.get(
                    idx2, set()
                ):
                    continue
                if component_ids[idx2] != component_ids[idx1]:
                    finished = False
                component_ids[idx2] = min(component_ids[idx2], component_ids[idx1])
                component_ids[idx1] = min(component_ids[idx2], component_ids[idx1])

    unique_ids = list(OrderedDict.fromkeys(component_ids[idx] for idx in conn_map))
    components: list[list[str]] = []
    for id in unique_ids:
        members = [idx for idx in conn_map if component_ids[idx] == id]
        components.append(members)

    component_order: OrderedDict[tuple, int] = OrderedDict(
        (tuple(c), i) for i, c in enumerate(components)
    )

    finished = False
    while not finished:
        finished = True
        for component1 in components:
            for component2 in components:
                is_parent_of_1 = False
                for idx1 in component1:
                    for idx2 in component2:
                        if idx2 in parent_map.get(idx1, set()):
                            is_parent_of_1 = True
                            break
                    if is_parent_of_1:
                        break

                if (
                    is_parent_of_1
                    and component_order[tuple(component2)]
                    > component_order[tuple(component1)]
                ):
                    max_pos = max(
                        component_order[tuple(component1)],
                        component_order[tuple(component2)],
                    )
                    min_pos = min(
                        component_order[tuple(component1)],
                        component_order[tuple(component2)],
                    )
                    component_order[tuple(component1)] = max_pos
                    component_order[tuple(component2)] = min_pos
                    finished = False

    components.sort(key=lambda c: component_order[tuple(c)])
    return components