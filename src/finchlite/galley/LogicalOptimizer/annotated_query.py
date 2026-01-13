from collections import OrderedDict
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from ...algebra import (
    InitWrite,
    cansplitpush,
    is_associative,
    is_commutative,
    is_distributive,
    repeat_operator,
)
from ...finch_logic import (
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
from ...symbolic import (
    Chain,
    PostWalk,
    PreOrderDFS,
    Rewrite,
    gensym,
    intree,
    isdescendant,
)
from ..TensorStats import TensorStats
from .logic_to_stats import insert_statistics


@dataclass
class AnnotatedQuery:
    ST: type[TensorStats]
    output_name: Alias
    reduce_idxs: list[Field]
    point_expr: LogicNode
    idx_lowest_root: OrderedDict[Field, LogicExpression]
    idx_op: OrderedDict[Field, Any]
    idx_init: OrderedDict[Field, Any]
    parent_idxs: OrderedDict[Field, list[Field]]
    original_idx: OrderedDict[Field, Field]
    connected_components: list[list[Field]]
    connected_idxs: OrderedDict[Field, set[Field]]
    bindings: OrderedDict[Alias, TensorStats]
    output_order: list[Field] | None = None
    output_format: list[Any] | None = None

    def __init__(
        self,
        ST: type[TensorStats],
        q: Query,
        bindings: OrderedDict[Alias, TensorStats] | None = None,
    ):
        """
        Build an `AnnotatedQuery` from a logical `Query`, extracting reduction
        structure and precomputing tensor statistics.

        Parameters
        ----------
        ST : type[TensorStats]
            Concrete `TensorStats` implementation used to hold statistics.
        q : Query
            Logical query of the form `Query(name, rhs)` whose `rhs` may contain
            `Aggregate` nodes.
        bindings : OrderedDict[Alias, TensorStats], optional
            Existing alias→stats environment to seed the analysis.
        """
        assert isinstance(q, Query), (
            "Annotated Queries can only be built from queries of the form: "
            "Query(lhs, rhs)"
        )
        self.ST = ST
        if bindings is None:
            bindings = OrderedDict()
        self.bindings = bindings
        cache: dict[object, TensorStats] = {}
        insert_statistics(ST, q, bindings=bindings, replace=False, cache=cache)
        self.cache = cache
        output_name = q.lhs
        expr = q.rhs
        output_format: list[Any] | None = None
        output_order: list[Field] | None = None
        starting_reduce_idxs: list[Field] = []
        idx_starting_root: OrderedDict[Field, LogicExpression] = OrderedDict()
        idx_top_order: OrderedDict[Field, int] = OrderedDict()
        top_counter = 1
        idx_op: OrderedDict[Field, Any] = OrderedDict()
        idx_init: OrderedDict[Field, Any] = OrderedDict()

        def aggregate_annotation_rule(node: LogicNode) -> LogicNode:
            nonlocal top_counter
            match node:
                case Aggregate(Literal() as op, Literal() as init, arg, idxs):
                    for idx in idxs:
                        idx_starting_root[idx] = arg
                        idx_top_order[idx] = top_counter
                        top_counter += 1

                        if op.val is None:
                            idx_op[idx] = InitWrite(cache[arg].fill_value)
                            idx_init[idx] = cache[arg].fill_value
                        else:
                            idx_op[idx] = op.val
                            idx_init[idx] = init.val

                        starting_reduce_idxs.append(idx)
                    return arg

                case _:
                    return node

        point_expr = Rewrite(PostWalk(Chain([aggregate_annotation_rule])))(expr)
        cache_point: dict[object, TensorStats] = {}
        insert_statistics(
            ST, point_expr, bindings=bindings, replace=False, cache=cache_point
        )
        self.cache_point = cache_point

        reduce_idxs: list[Field] = []
        original_idx: OrderedDict[Field, Field] = OrderedDict(
            (Field(idx), Field(idx)) for idx in cache[q.rhs].index_set
        )
        idx_lowest_root: OrderedDict[Field, LogicExpression] = OrderedDict()
        for idx in starting_reduce_idxs:
            agg_op = idx_op[idx]
            stats_point = cache_point[point_expr]
            idx_dim_size = stats_point.dim_sizes[idx.name]
            lowest_roots = find_lowest_roots(
                Literal(agg_op), idx, idx_starting_root[idx]
            )
            original_idx[idx] = idx
            if len(lowest_roots) == 1:
                idx_lowest_root[idx] = cast(LogicExpression, lowest_roots[0])
                reduce_idxs.append(idx)
            else:
                new_idxs = [
                    Field(f"{idx.name}_{i}")
                    for i, _ in enumerate(lowest_roots, start=1)
                ]
                for i, node in enumerate(lowest_roots):
                    if idx.name not in cache_point[node].index_set:
                        # If the lowest root doesn't contain the reduction index, we
                        # attempt to remove the reduction via a repeat_operator, i.e.
                        # ∑_i B_j = B_j*|Dom(i)|
                        f = repeat_operator(agg_op)
                        if f is None:
                            continue
                        dim_val = Table(
                            Literal(idx_dim_size),
                            (),
                        )
                        cache_point[dim_val] = ST(idx_dim_size, ())
                        new_node = MapJoin(Literal(f), (node, dim_val))
                        cache_point[new_node] = ST.mapjoin(
                            f, cache_point[node], cache_point[dim_val]
                        )
                        point_expr = cast(
                            LogicExpression,
                            replace_and_remove_nodes(
                                point_expr,
                                node_to_replace=node,
                                new_node=new_node,
                                nodes_to_remove=set(),
                            ),
                        )
                        continue
                    new_idx = new_idxs[i]
                    idx_op[new_idx] = agg_op
                    idx_init[new_idx] = idx_init[idx]
                    idx_lowest_root[new_idx] = cast(LogicExpression, node)
                    idx_starting_root[new_idx] = idx_starting_root[idx]
                    original_idx[new_idx] = idx
                    reduce_idxs.append(new_idx)
        parent_idxs: OrderedDict[Field, list[Field]] = OrderedDict(
            (i, []) for i in reduce_idxs
        )
        connected_idxs: OrderedDict[Field, set[Field]] = OrderedDict(
            (i, set()) for i in reduce_idxs
        )
        for idx1 in reduce_idxs:
            idx1_op = idx_op[idx1]
            idx1_bottom_root = idx_lowest_root[idx1]
            for idx2 in reduce_idxs:
                idx2_op = idx_op[idx2]
                idx2_top_root = idx_starting_root[idx2]
                idx2_bottom_root = idx_lowest_root[idx2]
                if intree(idx2_bottom_root, idx1_bottom_root):
                    connected_idxs[idx1].add(idx2)
                mergeable_agg_op = (
                    idx1_op == idx2_op
                    and is_associative(idx1_op)
                    and is_commutative(idx1_op)
                )
                # If idx1 isn't a parent of idx2, then idx2 can't restrict the
                # summation of idx1
                if isdescendant(idx2_top_root, idx1_bottom_root) or (
                    not mergeable_agg_op
                    and idx_top_order[original_idx[idx2]]
                    < idx_top_order[original_idx[idx1]]
                ):
                    parent_idxs[idx1].append(idx2)

        connected_components = get_idx_connected_components(parent_idxs, connected_idxs)

        self.output_name = output_name
        self.reduce_idxs = reduce_idxs
        self.point_expr = point_expr
        self.idx_lowest_root = idx_lowest_root
        self.idx_op = idx_op
        self.idx_init = idx_init
        self.parent_idxs = parent_idxs
        self.original_idx = original_idx
        self.connected_components = connected_components
        self.connected_idxs = connected_idxs
        self.output_order = output_order
        self.output_format = output_format


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
    nodes_to_remove_set = set(nodes_to_remove)

    def replace_remove_rule(node: LogicExpression) -> LogicExpression | None:
        match node:
            case Plan(_) | Query(_, _) | Aggregate(_, _, _, _) as illegal:
                raise ValueError(
                    f"There should be no {type(illegal).__name__} "
                    "nodes in a pointwise expression."
                )
            case node if node == node_to_replace and node not in nodes_to_remove_set:
                return new_node
            case MapJoin(op, args) if any(
                (arg == node_to_replace) or (arg in nodes_to_remove) for arg in args
            ):
                new_args = []
                for arg in args:
                    if arg in nodes_to_remove_set:
                        continue
                    if arg == node_to_replace:
                        new_args.append(new_node)
                    else:
                        new_args.append(arg)
                return MapJoin(op, tuple(new_args))
            case _:
                return None

    return Rewrite(PostWalk(Chain([replace_remove_rule])))(expr)


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
        A list of expression nodes representing the lowest positions in
        the expression tree where the reduction over `idx` with operator
        `op` can be safely pushed down.
    """
    match root:
        case MapJoin(Literal(mj_op), args) as mj:
            args_with = [arg for arg in args if idx in arg.fields()]
            args_without = [arg for arg in args if idx not in arg.fields()]

            if len(args_with) == 1 and is_distributive(mj_op, op.val):
                return find_lowest_roots(op, idx, args_with[0])

            if cansplitpush(op.val, mj_op):
                roots_without: list[LogicExpression] = list(args_without)
                roots_with: list[LogicExpression] = []
                for arg in args_with:
                    roots_with.extend(find_lowest_roots(op, idx, arg))
                return roots_without + roots_with

            return [mj]
        case Alias(_) | Table(_, _) as root:
            return [root]
        case _:
            raise ValueError(
                f"There shouldn't be nodes of type {type(root).__name__} "
                "during root pushdown."
            )


def get_reduce_query(
    reduce_idx: Field, aq: AnnotatedQuery
) -> tuple[Query, LogicExpression, set[LogicExpression], list[Field]]:
    """
    Extract the maximal kernel that depends on `reduce_idx` into a standalone
    reduction query, and return the information needed to splice the result
    back into the main expression.

    Parameters
    ----------
    reduce_idx : Field
        The index being reduced.
    aq : AnnotatedQuery
        The annotated query class

    Returns
    -------
    query : Query
        A new Query whose RHS is an Aggregate over the kernel that depends on
        `reduce_idx`.
    node_to_replace : LogicExpression
        The subexpression in `aq.point_expr` that will be replaced with the
        alias produced by `query`.
    nodes_to_remove : set[LogicExpression]
        Child nodes that become redundant after replacing `node_to_replace`.
    reduced_idxs : list[Field]
        The list of indices actually reduced in `query`.
    """
    original_idx = aq.original_idx[reduce_idx]
    reduce_op = aq.idx_op[reduce_idx]
    root_node: LogicExpression = aq.idx_lowest_root[reduce_idx]
    query_expr: LogicExpression
    idxs_to_be_reduced: set[Field] = set()
    nodes_to_remove: set[LogicExpression] = set()
    node_to_replace: LogicExpression = root_node
    reducible_idxs = get_reducible_idxs(aq)
    stats_cache = aq.cache_point

    use_root = False

    match root_node:
        case MapJoin(Literal(op), args) as mj if is_distributive(op, reduce_op):
            # If you're already reducing one index, then it may
            # make sense to reduce others as well.
            # E.g. when you reduce one vertex of a triangle, you should
            # do the other two as well.
            args_with_reduce_idx = [
                arg for arg in args if original_idx.name in stats_cache[arg].index_set
            ]
            kernel_idxs = set().union(
                *(stats_cache[arg].index_set for arg in args_with_reduce_idx)
            )
            relevant_args = [
                arg for arg in args if stats_cache[arg].index_set.issubset(kernel_idxs)
            ]
            if len(relevant_args) == len(args):
                node_to_replace = mj
                for node in PreOrderDFS(mj):
                    if node != node_to_replace:
                        nodes_to_remove.add(cast(LogicExpression, node))
            else:
                node_to_replace = relevant_args[0]
                for arg in relevant_args[1:]:
                    for node in PreOrderDFS(arg):
                        nodes_to_remove.add(cast(LogicExpression, node))
            query_expr = MapJoin(Literal(op), tuple(relevant_args))
            stats_cache[query_expr] = aq.ST.mapjoin(
                op, *[stats_cache[arg] for arg in relevant_args]
            )
            relevant_args_set = set(relevant_args)
            for idx in reducible_idxs:
                if aq.idx_op[idx] != aq.idx_op[reduce_idx]:
                    continue

                args_with_idx = [
                    arg
                    for arg in args
                    if aq.original_idx[idx].name in stats_cache[arg].index_set
                ]

                if idx in aq.connected_idxs[
                    reduce_idx
                ] and relevant_args_set.issuperset(args_with_idx):
                    idxs_to_be_reduced.add(idx)
        case _:
            use_root = True

    if use_root:
        query_expr = root_node
        node_to_replace = root_node
        reducible_idxs = get_reducible_idxs(aq)
        for idx in reducible_idxs:
            if aq.idx_op[idx] != aq.idx_op[reduce_idx]:
                continue
            if (
                idx in aq.connected_idxs[reduce_idx]
                or aq.idx_lowest_root[idx] == node_to_replace
            ):
                idxs_to_be_reduced.add(idx)

    final_idxs_to_be_reduced: list[Field] = []
    for idx in idxs_to_be_reduced:
        orig = aq.original_idx[idx]
        if orig not in final_idxs_to_be_reduced:
            final_idxs_to_be_reduced.append(orig)
    reduced_idxs = list(idxs_to_be_reduced)
    final_idxs_to_be_reduced.sort(key=lambda f: f.name)

    agg_op = aq.idx_op[aq.original_idx[reduce_idx]]
    agg_init = aq.idx_init[aq.original_idx[reduce_idx]]

    query_expr = Aggregate(
        Literal(agg_op),
        Literal(agg_init),
        query_expr,
        tuple(final_idxs_to_be_reduced),
    )

    stats_cache[query_expr] = aq.ST.aggregate(
        agg_op,
        agg_init,
        [i.name for i in final_idxs_to_be_reduced],
        stats_cache[query_expr.arg],
    )

    query = Query(Alias(gensym("A")), query_expr)

    return query, node_to_replace, nodes_to_remove, reduced_idxs


def reduce_idx(
    reduce_idx: Field, aq: AnnotatedQuery, do_condense: bool = False
) -> Query:
    """
    Perform a single reduction rewrite over `reduce_idx`, restructuring `aq`
    so that the portion of the expression dependent on `reduce_idx` becomes
    a standalone subquery.

    Steps:
      1. Use `get_reduce_query` to extract the maximal subexpression that
         depends on `reduce_idx` and package it into a new `Query`.
      2. Create a fresh `Alias` for this subquery and register its statistics.
      3. Replace the extracted kernel in `aq.point_expr` with that alias and
         remove any nodes that are no longer reachable.
      4. Update all index-related metadata in `aq`—roots, ops, inits, parent
         structure, connectivity, components, and the remaining reduction set.

    Parameters
    ----------
    reduce_idx : Field
        The index being reduced.
    aq : AnnotatedQuery
        The annotated query to rewrite in place.
    do_condense : bool.

    Returns
    -------
    Query
        The newly created `Query` whose RHS computes the reduced kernel; its
        alias is used in the updated `aq.point_expr`.
    """
    query, node_to_replace, nodes_to_remove, reduced_idxs = get_reduce_query(
        reduce_idx, aq
    )

    alias_expr = Alias(query.lhs.name)
    stats_cache = aq.cache_point
    insert_statistics(aq.ST, query, aq.bindings, replace=False, cache=stats_cache)
    alias_idxs = [Field(idx) for idx in aq.bindings[alias_expr].index_set]

    new_point_expr: LogicExpression = replace_and_remove_nodes(
        expr=cast(LogicExpression, aq.point_expr),
        node_to_replace=node_to_replace,
        new_node=Table(alias_expr, tuple(alias_idxs)),
        nodes_to_remove=nodes_to_remove,
    )
    new_reduce_idxs = [x for x in aq.reduce_idxs if x not in reduced_idxs]
    new_idx_lowest_root: OrderedDict[Field, LogicExpression] = OrderedDict()
    new_idx_op: OrderedDict[Field, Any] = OrderedDict()
    new_idx_init: OrderedDict[Field, Any] = OrderedDict()
    new_parent_idxs: OrderedDict[Field, list[Field]] = OrderedDict()
    new_connected_idxs: OrderedDict[Field, set[Field]] = OrderedDict()
    for idx in aq.idx_lowest_root:
        if idx in reduced_idxs:
            continue
        root = aq.idx_lowest_root[idx]
        if root == node_to_replace or root in nodes_to_remove:
            root = Table(alias_expr, tuple(reduced_idxs))

        new_idx_lowest_root[idx] = root
        new_idx_op[idx] = aq.idx_op[idx]
        new_idx_init[idx] = aq.idx_init[idx]
        new_idx_op[aq.original_idx[idx]] = aq.idx_op[idx]
        new_idx_init[aq.original_idx[idx]] = aq.idx_init[idx]
        new_parent_idxs[idx] = [
            x for x in aq.parent_idxs.get(idx, []) if x not in reduced_idxs
        ]
        new_connected_idxs[idx] = {
            x for x in aq.connected_idxs.get(idx, set()) if x not in reduced_idxs
        }

    for idx in new_idx_lowest_root:
        for idx2 in new_idx_lowest_root:
            if new_idx_lowest_root[idx] is new_idx_lowest_root[idx2]:
                new_connected_idxs[idx].add(idx2)
                new_connected_idxs[idx2].add(idx)

    new_components = get_idx_connected_components(new_parent_idxs, new_connected_idxs)

    insert_statistics(
        aq.ST,
        new_point_expr,
        aq.bindings,
        replace=True,
        cache=stats_cache,
    )

    aq.reduce_idxs = new_reduce_idxs
    aq.point_expr = new_point_expr
    aq.idx_lowest_root = new_idx_lowest_root
    aq.idx_op = new_idx_op
    aq.idx_init = new_idx_init
    aq.parent_idxs = new_parent_idxs
    aq.connected_idxs = new_connected_idxs
    aq.connected_components = new_components
    return query


def get_remaining_query(aq: AnnotatedQuery) -> Query | None:
    """
    Build a final `Query` from the remaining pointwise expression in `aq`.

    Returns
    -------
    Query | None
        The constructed final `Query`, or `None` if no expression remains.
    """
    expr = aq.point_expr
    insert_statistics(
        aq.ST, expr, bindings=aq.bindings, replace=True, cache=aq.cache_point
    )
    match expr:
        case Table(Alias(_), _):
            return None
    query = Query(aq.output_name, cast(LogicExpression, expr))
    remaining_cache: dict[object, TensorStats] = {}
    insert_statistics(
        aq.ST, query.rhs, bindings=aq.bindings, replace=True, cache=remaining_cache
    )
    return query
