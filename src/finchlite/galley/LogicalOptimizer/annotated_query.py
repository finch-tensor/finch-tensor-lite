from collections import OrderedDict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from finchlite.algebra import (
    cansplitpush,
    is_distributive,
)
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicExpression,
    LogicNode,
    MapJoin,
    Plan,
    Query,
    Table,
)
from finchlite.galley.TensorStats import TensorStats
from finchlite.symbolic import PostOrderDFS


@dataclass
class AnnotatedQuery:
    ST: type[TensorStats]
    output_name: Alias | None
    reduce_idxs: list[Field]
    point_expr: LogicNode
    idx_lowest_root: OrderedDict[Field, LogicExpression]
    idx_op: OrderedDict[Field, Any]
    idx_init: OrderedDict[Field, Any]
    parent_idxs: OrderedDict[Field, list[Field]]
    original_idx: OrderedDict[Field, Field]
    connected_components: list[list[Field]]
    connected_idxs: OrderedDict[Field, set[Field]]
    output_order: list[Field] | None = None
    output_format: list[Any] | None = None


def copy_aq(aq: AnnotatedQuery) -> AnnotatedQuery:
    """
    Make a structured copy of an AnnotatedQuery.
    """
    new = object.__new__(AnnotatedQuery)
    new.ST = aq.ST
    new.output_name = aq.output_name
    new.point_expr = aq.point_expr
    new.reduce_idxs = list(aq.reduce_idxs)
    new.idx_lowest_root = OrderedDict(aq.idx_lowest_root.items())
    new.idx_op = OrderedDict(aq.idx_op.items())
    new.idx_init = OrderedDict(aq.idx_init.items())
    new.parent_idxs = OrderedDict((m, list(n)) for m, n in aq.parent_idxs.items())
    new.original_idx = OrderedDict(aq.original_idx.items())
    new.connected_components = [list(n) for n in aq.connected_components]
    new.connected_idxs = OrderedDict((m, set(n)) for m, n in aq.connected_idxs.items())
    new.output_order = None if aq.output_order is None else list(aq.output_order)
    new.output_format = None if aq.output_format is None else list(aq.output_format)

    return new


def get_reducible_idxs(aq: AnnotatedQuery) -> list[Field]:
    """
    Indices eligible to be reduced immediately (no parents).

    Parameters
    ----------
    aq : AnnotatedQuery
        Query containing the candidate reduction indices and their parent map.

    Returns
    -------
    list[Field]
        Field objects in `aq.reduce_idxs` with zero parents.
    """
    return [idx for idx in aq.reduce_idxs if len(aq.parent_idxs.get(idx, [])) == 0]


def get_idx_connected_components(
    parent_idxs: Mapping[Field, Iterable[Field]],
    connected_idxs: Mapping[Field, Iterable[Field]],
) -> list[list[Field]]:
    """
    Compute connected components of indices (Field objects) and order those
    components by parent/child constraints.

    Parameters
    ----------
    parent_idxs : Dict[Field, Iterable[Field]]
        Mapping from an index to the iterable of its parent indices.
    connected_idxs : Dict[Field, Iterable[Field]]
        Mapping from an index to the iterable of indices considered
        "connected" to it (undirected neighbors). Only connections between
        non-parent pairs are used to form components.

    Returns
    -------
    List[List[Field]]
        A list of components, each a list of Field objects. Components are
        ordered so that any component containing a parent appears before any
        component containing its child.
    """
    parent_map: dict[Field, set[Field]] = {k: set(v) for k, v in parent_idxs.items()}
    conn_map: OrderedDict[Field, set[Field]] = OrderedDict(
        (k, set(v)) for k, v in connected_idxs.items()
    )

    component_ids: OrderedDict[Field, int] = OrderedDict(
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
    components: list[list[Field]] = []
    for id in unique_ids:
        members = [idx for idx in conn_map if component_ids[idx] == id]
        components.append(members)

    component_order: OrderedDict[tuple[Field, ...], int] = OrderedDict(
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


def replace_and_remove_nodes(
    expr: LogicExpression,
    node_to_replace: LogicExpression,
    new_node: LogicExpression,
    nodes_to_remove: Collection[LogicExpression],
) -> LogicExpression:
    """
    Replace and/or remove arguments of a pointwise MapJoin expression.

    Parameters
    ----------
    expr : LogicExpression
        The expression to transform. Typically a `MapJoin` in a pointwise
        subexpression.
    node_to_replace : LogicExpression
        The node to replace when it appears as an argument to `expr`, or as
        `expr` itself.
    new_node : LogicExpression
        The node that replaces `node_to_replace` wherever it is found.
    nodes_to_remove : Collection[LogicExpression]
        A collection of nodes that, if present as arguments to a `MapJoin`,
        should be removed from its argument list.

    Returns
    -------
    LogicExpression
        A `MapJoin` node with updated arguments if `expr` is a `MapJoin`,
        `new_node` if `expr == node_to_replace`, or the original `expr`
        otherwise.
    """
    if expr == node_to_replace:
        return new_node

    nodes_to_remove = set(nodes_to_remove)

    for node in PostOrderDFS(expr):
        if isinstance(node, (Plan, Query, Aggregate)):
            raise ValueError(
                f"There should be no {type(node).__name__} "
                "nodes in a pointwise expression."
            )

        if isinstance(node, MapJoin) and any(
            (arg == node_to_replace) or (arg in nodes_to_remove) for arg in node.args
        ):
            new_args: list[LogicExpression] = [
                arg for arg in node.args if arg not in nodes_to_remove
            ]
            for i, arg in enumerate(new_args):
                if arg == node_to_replace:
                    new_args[i] = new_node
            object.__setattr__(node, "args", tuple(new_args))
    return expr


def find_lowest_roots(
    op: Literal, idx: Field, root: LogicExpression
) -> list[LogicExpression]:
    """
        Compute the lowest MapJoin / leaf nodes that a reduction over `idx` can be
        safely pushed down to in a logical expression.

        Parameters
        ----------
        op : Literal
            The reduction operator node (e.g., Literal(operator.add))
            that we are trying to push down.
        idx : Field
            The index (dimension) being reduced over.
        root : LogicExpression
            The root logical expression under which we search for the lowest
            pushdown positions for the reduction.

        Returns
        -------
        list[LogicExpression]
    `        A list of expression nodes representing the lowest positions in
            the expression tree where the reduction over `idx` with operator
            `op` can be safely pushed down.
    """

    if isinstance(root, MapJoin):
        if not isinstance(root.op, Literal):
            raise TypeError(
                f"Expected MapJoin.op to be a Literal, got {type(root.op).__name__}"
            )
        args_with = [arg for arg in root.args if idx in arg.fields()]
        args_without = [arg for arg in root.args if idx not in arg.fields()]

        if len(args_with) == 1 and is_distributive(root.op.val, op.val):
            return find_lowest_roots(op, idx, args_with[0])

        if cansplitpush(op.val, root.op.val):
            roots_without: list[LogicExpression] = list(args_without)
            roots_with: list[LogicExpression] = []
            for arg in args_with:
                roots_with.extend(find_lowest_roots(op, idx, arg))
            return roots_without + roots_with
        return [root]

    if isinstance(root, (Alias, Table)):
        return [root]

    raise ValueError(
        f"There shouldn't be nodes of type {type(root).__name__} during root pushdown."
    )
