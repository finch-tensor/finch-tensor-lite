from collections import OrderedDict
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any

from finchlite.algebra import (
    InitWrite,
    cansplitpush,
    is_associative,
    is_commutative,
    is_distributive,
    repeat_operator,
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
    Value,
)
from finchlite.galley.TensorStats import TensorStats
from finchlite.symbolic import Chain, PostWalk, Rewrite, intree, isdescendant

from .logic_to_stats import insert_statistics


@dataclass
class AnnotatedQuery:
    ST: type
    output_name: Alias | None
    reduce_idxs: list[str]
    point_expr: "LogicNode"
    idx_lowest_root: OrderedDict[str, LogicNode]
    idx_op: OrderedDict[str, Any]
    idx_init: OrderedDict[str, Any]
    parent_idxs: OrderedDict[str, list[str]]
    original_idx: OrderedDict[str, str]
    connected_components: list[list[str]]
    connected_idxs: OrderedDict[str, set[str]]
    output_order: list[str] | None = None
    output_format: list[Any] | None = None

    # @classmethod
    # def from_query(cls, q: LogicNode, ST: type) -> "AnnotatedQuery":
    #     if not isinstance(q, Query):
    #         raise ValueError(
    #             "Annotated Queries can only be built from queries of the form: "
    #             "Query(name, Materialize(formats, index_order, agg_map_expr)) or "
    #             "Query(name, agg_map_expr)"
    #         )
    #     bindings: OrderedDict[Alias, TensorStats] = OrderedDict()
    #     cache: dict[object, TensorStats] = {}
    #     insert_statistics(ST, q, bindings=bindings, replace=False, cache=cache)
    #     output_name = q.lhs
    #     expr = q.rhs
    #     output_format = None
    #     output_order = None
    #     starting_reduce_idxs: list[Field] = []
    #     idx_starting_root: OrderedDict[Field, LogicExpression] = OrderedDict()
    #     idx_top_order: OrderedDict[Field, int] = OrderedDict()
    #     top_counter = 1
    #     idx_op: OrderedDict[Field, Any] = OrderedDict()
    #     idx_init: OrderedDict[Field, Any] = OrderedDict()

    #     def aggregate_annotation_rule(node: LogicNode) -> LogicNode:
    #         nonlocal top_counter
    #         match node:
    #             case Aggregate(Literal() as op, Literal() as init, arg, idxs):
    #                 for idx in idxs:
    #                     idx_starting_root[idx] = arg
    #                     idx_top_order[idx] = top_counter
    #                     top_counter += 1

    #                     if op.val == None:
    #                         idx_op[idx] = InitWrite(cache[arg].fill_value)
    #                         idx_init[idx] = cache[arg].fill_value
    #                     else:
    #                         idx_op[idx] = op.val
    #                         idx_init[idx] = init.val

    #                     starting_reduce_idxs.append(idx)
    #                 return arg

    #             case _:
    #                 return node

    #     point_expr = Rewrite(PostWalk(Chain([aggregate_annotation_rule])))(expr)

    #     bindings_point: OrderedDict[Alias, TensorStats] = OrderedDict()
    #     cache_point: dict[object, TensorStats] = {}
    #     insert_statistics(
    #         ST, point_expr, bindings=bindings_point, replace=False, cache=cache_point
    #     )

    #     reduce_idxs: list[Field] = []
    #     original_idx: OrderedDict[Field, Field] = OrderedDict(
    #         (Field(idx), Field(idx)) for idx in cache[q.rhs].index_set
    #     )
    #     idx_lowest_root: OrderedDict[Field, LogicNode] = OrderedDict()
    #     for idx in starting_reduce_idxs:
    #         agg_op = idx_op[idx]
    #         stats_point = cache_point[point_expr]
    #         idx_dim_size = stats_point.dim_sizes[idx.name]
    #         lowest_roots = find_lowest_roots(agg_op, idx, idx_starting_root[idx])
    #         original_idx[idx] = idx
    #         if len(lowest_roots) == 1:
    #             idx_lowest_root[idx] = lowest_roots[0]
    #             reduce_idxs.append(idx)
    #         else:
    #             new_idxs = [
    #                 Field(f"{idx.name}_{i}")
    #                 for i, _ in enumerate(lowest_roots, start=1)
    #             ]
    #             for i, node in enumerate(lowest_roots):
    #                 if idx.name not in cache_point[node].index_set:
    #                     # If the lowest root doesn't contain the reduction index, we attempt
    #                     # to remove the reduction via a repeat_operator, i.e.
    #                     # ∑_i B_j = B_j*|Dom(i)|
    #                     if repeat_operator(agg_op) == None:
    #                         continue
    #                     f = repeat_operator(agg_op)
    #                     dim_val = Value(idx_dim_size, type(idx_dim_size))
    #                     cache_point[dim_val] = ST(idx_dim_size)
    #                     new_node = MapJoin(Literal(f), (node, dim_val))
    #                     cache_point[new_node] = ST.mapjoin(
    #                         f, cache_point[node], cache_point[dim_val]
    #                     )
    #                     point_expr = replace_and_remove_nodes(
    #                         point_expr,
    #                         node_to_replace=node,
    #                         new_node=new_node,
    #                         nodes_to_remove=set(),
    #                     )
    #                     continue
    #                 new_idx = new_idxs[i]
    #                 idx_op[new_idx] = agg_op
    #                 idx_init[new_idx] = idx_init[idx]
    #                 idx_lowest_root[new_idx] = node
    #                 idx_starting_root[new_idx] = idx_starting_root[idx]
    #                 original_idx[new_idx] = idx
    #                 reduce_idxs.append(new_idx)
    #     parent_idxs: OrderedDict[Field, list[Field]] = OrderedDict(
    #         (i, []) for i in reduce_idxs
    #     )
    #     connected_idxs: OrderedDict[Field, set[Field]] = OrderedDict(
    #         (i, set()) for i in reduce_idxs
    #     )
    #     for idx1 in reduce_idxs:
    #         idx1_op = idx_op[idx1]
    #         idx1_bottom_root = idx_lowest_root[idx1]
    #         for idx2 in reduce_idxs:
    #             idx2_op = idx_op[idx2]
    #             idx2_top_root = idx_starting_root[idx2]
    #             idx2_bottom_root = idx_lowest_root[idx2]
    #             if intree(idx2_bottom_root, idx1_bottom_root):
    #                 connected_idxs[idx1].add(idx2)
    #             mergeable_agg_op = (
    #                 idx1_op == idx2_op
    #                 and is_associative(idx1_op)
    #                 and is_commutative(idx1_op)
    #             )
    #             # If idx1 isn't a parent of idx2, then idx2 can't restrict the summation of idx1
    #             if isdescendant(idx2_top_root, idx1_bottom_root) or (
    #                 not mergeable_agg_op
    #                 and idx_top_order[original_idx[idx2]]
    #                 < idx_top_order[original_idx[idx1]]
    #             ):
    #                 parent_idxs[idx1].append(idx2)

    #     connected_components = get_idx_connected_components(parent_idxs, connected_idxs)

    #     return cls(
    #         ST=ST,
    #         output_name=output_name,
    #         reduce_idxs=reduce_idxs,
    #         point_expr=point_expr,
    #         idx_lowest_root=idx_lowest_root,
    #         idx_op=idx_op,
    #         idx_init=idx_init,
    #         parent_idxs=parent_idxs,
    #         original_idx=original_idx,
    #         connected_components=connected_components,
    #         connected_idxs=connected_idxs,
    #         output_order=output_order,
    #         output_format=output_format,
    #     )


def copy_aq(aq: AnnotatedQuery) -> AnnotatedQuery:
    """
    Make a structured copy of an AnnotatedQuery.
    """
    return AnnotatedQuery(
        ST=aq.ST,
        output_name=aq.output_name,
        reduce_idxs=list(aq.reduce_idxs),
        point_expr=aq.point_expr,
        idx_lowest_root=aq.idx_lowest_root.copy(),
        idx_op=OrderedDict(aq.idx_op.items()),
        idx_init=OrderedDict(aq.idx_init.items()),
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
        component containing its child.
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


def replace_and_remove_nodes(
    expr: LogicNode,
    node_to_replace: LogicExpression,
    new_node: LogicExpression,
    nodes_to_remove: Collection[LogicExpression],
) -> LogicNode:
    """
    Replace and/or remove arguments of a pointwise MapJoin expression.

    Parameters
    ----------
    expr : LogicNode
        The expression to transform. Typically a `MapJoin` in a pointwise
        subexpression.
    node_to_replace : LogicNode
        The node to replace when it appears as an argument to `expr`, or as
        `expr` itself.
    new_node : LogicNode
        The node that replaces `node_to_replace` wherever it is found.
    nodes_to_remove : Collection[LogicNode]
        A collection of nodes that, if present as arguments to a `MapJoin`,
        should be removed from its argument list.

    Returns
    -------
    LogicNode
        A new `MapJoin` node with updated arguments if `expr` is a `MapJoin`,
        `new_node` if `expr == node_to_replace`, or the original `expr`
        otherwise.
    """
    if expr == node_to_replace:
        return new_node

    if isinstance(expr, (Plan, Query, Aggregate)):
        raise ValueError(
            f"There should be no {type(expr).__name__} nodes in a pointwise expression."
        )

    if isinstance(expr, MapJoin):
        nodes_to_remove = set(nodes_to_remove)
        new_args: list[LogicExpression] = []

        for arg in expr.args:
            if arg in nodes_to_remove:
                continue
            if arg == node_to_replace:
                arg = new_node
            new_args.append(arg)

        return MapJoin(expr.op, tuple(new_args))
    return expr


def find_lowest_roots(
    op: Literal, idx: Field, root: LogicExpression
) -> list[LogicExpression]:
    """
    Compute the lowest MapJoin / leaf nodes that a reduction over `idx` can be
    safely pushed down to in a logical expression.

    Parameters
    ----------
    op : LogicNode
        The reduction operator node (e.g., a Literal wrapping `operator.add`)
        that we are trying to push down.
    idx : Field
        The index (dimension) being reduced over.
    root : LogicExpression
        The root logical expression under which we search for the lowest
        pushdown positions for the reduction.

    Returns
    -------
    list[LogicExpression]
        A list of expression nodes representing the lowest positions in
        the expression tree where the reduction over `idx` with operator
    """

    if isinstance(root, MapJoin):
        if not isinstance(root.op, Literal):
            raise TypeError(
                f"Expected MapJoin.op to be a Literal, got {type(root.op).__name__}"
            )
        args_with = [arg for arg in root.args if idx in arg.fields]
        args_without = [arg for arg in root.args if idx not in arg.fields]

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
