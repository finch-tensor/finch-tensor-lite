from __future__ import annotations

from ....algebra import is_associative, is_commutative, is_distributive
from ....finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicExpression,
    LogicNode,
    LogicStatement,
    MapJoin,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    Table,
)
from ....symbolic import Fixpoint, PostWalk, Rewrite, gensym

"""
Query merging and reorder normalization for Galley.
Rewrites Plan:

The transformation has two steps:

1. Alias-based query merging:
   - Replace Table(Alias, ...) uses with the RHS of the defining Query
     for that alias, with fields renamed to match the table's indices.
   - Rebuild the plan so that, for each alias mentioned in the final
     Produces, there is only one merged Query computing it.
   - Ensure we never drop a Query whose lhs alias appears in a
     Produces statement.

2. Reorder normalization:
   - For each Query, record rhs.fields() as the final output field order.
   - Recursively strip all interior Reorder nodes from the RHS.
   - Wrap the resulting expression in a single outer `Reorder` that restores
     the original fields() order.
"""


def _rename_field(idx: Field, mapping: dict[Field, Field]) -> Field:
    """Return the renamed field if present, otherwise the original."""
    return mapping.get(idx, idx)


def alpha_rename_expr(
    expr: LogicExpression, mapping: dict[Field, Field]
) -> LogicExpression:
    """
    Apply a simple alpha-renaming to expr, replacing any Field that appears
    in mapping with its mapped value.

    This traverses only the LogicExpression nodes that actually store fields
    and leaves all other nodes unchanged.

    The goal is to rewrite the dimension variables of an alias
    definition to match the index variables at a particular call site.
    """

    match expr:
        case Table(tns, idxs):
            new_idxs = tuple(_rename_field(idx, mapping) for idx in idxs)
            return Table(tns, new_idxs)
        case MapJoin(op, args):
            new_args = tuple(alpha_rename_expr(arg, mapping) for arg in args)
            return MapJoin(op, new_args)
        case Aggregate(op, init, arg, idxs):
            new_arg = alpha_rename_expr(arg, mapping)
            new_idxs = tuple(_rename_field(idx, mapping) for idx in idxs)
            return Aggregate(op, init, new_arg, new_idxs)
        case Reorder(arg, idxs):
            new_arg = alpha_rename_expr(arg, mapping)
            new_idxs = tuple(_rename_field(idx, mapping) for idx in idxs)
            return Reorder(new_arg, new_idxs)
        case Relabel(arg, idxs):
            new_arg = alpha_rename_expr(arg, mapping)
            new_idxs = tuple(_rename_field(idx, mapping) for idx in idxs)
            return Relabel(new_arg, new_idxs)
        case _:
            return expr


def _collect_all_fields(expr: LogicExpression) -> set[Field]:
    """
    Recursively collect all Field objects appearing in an expression
    """
    result: set[Field] = set()
    match expr:
        case Table(_, idxs):
            result.update(idxs)
        case MapJoin(_, args):
            for arg in args:
                result.update(_collect_all_fields(arg))
        case Aggregate(_, _, arg, idxs):
            result.update(idxs)
            result.update(_collect_all_fields(arg))
        case Reorder(arg, idxs):
            result.update(idxs)
            result.update(_collect_all_fields(arg))
        case Relabel(arg, idxs):
            result.update(idxs)
            result.update(_collect_all_fields(arg))
        case _:
            pass
    return result


def _push_aggregate_up(expr: LogicExpression) -> LogicExpression | None:
    """
    Rewrite: MapJoin(op, args_1, Aggregate(agg_op, agg_init, agg_arg), args_2)
         -> Aggregate(agg_op, agg_init, MapJoin(op, args_1, agg_arg, args_2))
    when op distributes over agg_op. Pushes aggregates outward so reduction
    indices are no longer nested.
    """
    match expr:
        case MapJoin(Literal(mj_op), args):
            for i, arg in enumerate(args):
                if (
                    isinstance(arg, Aggregate)
                    and isinstance(arg.op, Literal)
                    and isinstance(arg.init, Literal)
                    and is_distributive(mj_op, arg.op.val)
                ):
                    new_args = args[:i] + (arg.arg,) + args[i + 1 :]
                    return Aggregate(
                        arg.op, arg.init, MapJoin(expr.op, new_args), arg.idxs
                    )
            return None
        case _:
            return None


def _merge_adjacent_aggregates(expr: LogicExpression) -> LogicExpression | None:
    """
    Rewrite: Aggregate(op, init, Aggregate(op, init, arg, idxs1), idxs2)
         -> Aggregate(op, init, arg, idxs1 + idxs2)
    when op is associative and commutative. Flattens nested same-op aggregates
    so all reduction indices share the same root; neither is a parent of the other.
    """
    match expr:
        case Aggregate(
            Literal(op),
            Literal(init),
            Aggregate(Literal(inner_op), Literal(inner_init), arg, idxs1),
            idxs2,
        ) if op == inner_op and init == inner_init:
            if is_associative(op) and is_commutative(op):
                return Aggregate(expr.op, expr.init, arg, idxs1 + idxs2)
            return None
        case _:
            return None


def push_aggregates_up(expr: LogicExpression) -> LogicExpression:
    """
    Apply push-aggregate and merge-aggregate rewrites to fixpoint.
    Produces an expression where reduction indices are not parent-related
    """

    def rule(node: LogicNode) -> LogicNode | None:
        if isinstance(node, LogicExpression):
            r = _merge_adjacent_aggregates(node)
            if r is not None:
                return r
            r = _push_aggregate_up(node)
            if r is not None:
                return r
        return None

    result = Rewrite(Fixpoint(PostWalk(rule)))(expr)
    return result if result is not None else expr


def _inline_tables_in_expr(
    expr: LogicExpression,
    alias_to_query: dict[Alias, Query],
    produced_aliases: set[Alias] | None = None,
) -> LogicExpression:
    """
    Recursively inline Table(Alias, ...) nodes using alias_to_query.

    Whenever we see Table(Alias X, idxs), and X has a defining Query
    (and X is not in ``produced_aliases``), we:
      1. Take the RHS of that query.
      2. Compute its natural field order via rhs.fields().
      3. Build a mapping from those fields to idxs.
      4. Alpha-rename the RHS under that mapping.
      5. Recursively inline within the renamed RHS.

    Aliases listed in produced_aliases are not inlined because they
    will already be computed as separate queries; reusing them is more
    efficient than reexpanding their definitions.
    """
    if produced_aliases is None:
        produced_aliases = set()

    match expr:
        case Table(Alias() as alias, idxs):
            if alias not in alias_to_query or alias in produced_aliases:
                return expr

            rhs = alias_to_query[alias].rhs
            orig_fields = rhs.fields()
            if len(orig_fields) != len(idxs):
                raise ValueError(
                    f"Alias {alias} has {len(orig_fields)} output fields, "
                    f"but Table call site requires {len(idxs)} indices"
                )

            mapping: dict[Field, Field] = dict(zip(orig_fields, idxs, strict=True))

            # Assign fresh names to internal non-output fields so that when the same
            # alias is inlined multiple times, internal indices do not collide.
            # e.g matmul: A @ A @ A @ A @ A @ A
            orig_fields_set = set(orig_fields)
            all_fields = _collect_all_fields(rhs)
            for f in all_fields:
                if f not in orig_fields_set:
                    mapping[f] = Field(gensym("i"))

            renamed_rhs = alpha_rename_expr(rhs, mapping)
            return _inline_tables_in_expr(renamed_rhs, alias_to_query, produced_aliases)

        case MapJoin(op, args):
            new_args = tuple(
                _inline_tables_in_expr(arg, alias_to_query, produced_aliases)
                for arg in args
            )
            return MapJoin(op, new_args)

        case Aggregate(op, init, arg, idxs):
            new_arg = _inline_tables_in_expr(arg, alias_to_query, produced_aliases)
            return Aggregate(op, init, new_arg, idxs)

        case Reorder(arg, idxs):
            new_arg = _inline_tables_in_expr(arg, alias_to_query, produced_aliases)
            return Reorder(new_arg, idxs)

        case Relabel(arg, idxs):
            new_arg = _inline_tables_in_expr(arg, alias_to_query, produced_aliases)
            return Relabel(new_arg, idxs)

        case _:
            return expr


def merge_queries(plan: Plan) -> Plan:
    """
    Merge per-alias queries into larger queries, one per produced tensor.

    - Builds a mapping from Alias to its defining Query in the original plan.
    - For each alias mentioned in the final Produces statement, constructs a
      new Query(lhs, merged_rhs) where merged_rhs inlines any
      Table(Alias, ...) references using _inline_tables_in_expr.
    - Ensures that we never drop a Query whose lhs alias appears in a
      Produces statement
    """
    if not isinstance(plan, Plan):
        return plan

    bodies: tuple[LogicNode, ...] = plan.bodies  # type: ignore[assignment]
    if not bodies:
        return plan

    # Find the last Produces statement, if any.
    produces_stmt: Produces | None = None
    for body in reversed(bodies):
        if isinstance(body, Produces):
            produces_stmt = body
            break

    if produces_stmt is None:
        return plan

    # Map each alias to the last Query that defines it in the original plan.
    alias_to_query: dict[Alias, Query] = {}
    for body in bodies:
        if isinstance(body, Query):
            alias_to_query[body.lhs] = body

    produced_aliases: tuple[Alias, ...] = produces_stmt.args
    produced_alias_set: set[Alias] = set(produced_aliases)

    new_queries: list[Query] = []
    for alias in produced_aliases:
        defining_query = alias_to_query.get(alias)
        if defining_query is None:
            continue
        # Don't inline other produced aliases, resue them
        other_produced = produced_alias_set - {alias}
        merged_rhs = _inline_tables_in_expr(
            defining_query.rhs, alias_to_query, other_produced
        )
        new_queries.append(Query(alias, merged_rhs))

    # Rebuild the plan as: [merged queries..., original Produces]
    if not new_queries:
        raise ValueError("Plan should not be empty after merging queries")

    new_bodies: list[LogicStatement] = []
    new_bodies.extend(new_queries)
    new_bodies.append(produces_stmt)
    return Plan(tuple(new_bodies))


def strip_reorders(expr: LogicExpression) -> LogicExpression:
    """
    Recursively remove Reorder nodes that are pure permutations.

    A Reorder(arg, idxs) is considered a pure permutation if it preserves the
    set of fields of arg (i.e., set(arg.fields()) == set(idxs)). Such nodes
    can be safely removed without changing the logical dimensionality. Reorders
    that add or drop dimensions are preserved to avoid changing semantics or
    conflicting with tensor statistics.
    """
    match expr:
        case Reorder(arg, idxs):
            new_arg = strip_reorders(arg)
            # Collapse consecutive Reorders: Reorder(Reorder(X, A), B) -> Reorder(X, B)
            while isinstance(new_arg, Reorder):
                new_arg = new_arg.arg
            child_fields = new_arg.fields()
            # Only strip Reorder nodes that are pure permutations of the same
            # field set. If the Reorder adds or drops dimensions relative to
            # its child, we must keep it.
            if set(child_fields) == set(idxs):
                return new_arg
            return Reorder(new_arg, idxs)
        case MapJoin(op, args):
            new_args = tuple(strip_reorders(arg) for arg in args)
            return MapJoin(op, new_args)
        case Aggregate(op, init, arg, idxs):
            new_arg = strip_reorders(arg)
            return Aggregate(op, init, new_arg, idxs)
        case Relabel(arg, idxs):
            new_arg = strip_reorders(arg)
            return Relabel(new_arg, idxs)
        case _:
            return expr


def normalize_reorders_in_query(query: Query) -> Query:
    """
    Normalize Reorder usage in a single query.

    For a given Query(lhs, rhs):
      1. Record out_fields = rhs.fields().
      2. Strip all interior Reorder nodes from rhs.
      3. Wrap the result in a single outer Reorder(inner_rhs, *out_fields).
    """
    rhs = query.rhs
    out_fields = rhs.fields()
    inner_rhs = strip_reorders(rhs)
    inner_fields = inner_rhs.fields()

    # If the inner expression already has the desired field order, we can avoid
    # inserting a redundant Reorder.
    if inner_fields == out_fields:
        return Query(query.lhs, inner_rhs)

    # Otherwise, insert a single outer Reorder to enforce the original output
    # ordering while keeping the same field set.
    return Query(query.lhs, Reorder(inner_rhs, out_fields))


def normalize_reorders_in_plan(plan: Plan) -> Plan:
    """Apply `normalize_reorders_in_query` to each Query body in the plan."""
    if not isinstance(plan, Plan):
        return plan

    new_bodies: list[LogicStatement] = []
    for body in plan.bodies:
        if isinstance(body, Query):
            new_bodies.append(normalize_reorders_in_query(body))
        else:
            new_bodies.append(body)

    return Plan(tuple(new_bodies))


def merge_mapjoin_rule(node: LogicNode) -> LogicNode:
    match node:
        case MapJoin(Literal(op1), (MapJoin(Literal(op2), args2), *args1)) if (
            op1 == op2 and is_associative(op1)
        ):
            return MapJoin(Literal(op1), tuple(args2) + tuple(args1))
        case _:
            return node


def preprocess_plan_for_galley(plan: Plan) -> Plan:
    """
    End-to-end preprocessing used before running Galley greedy optimization.

    - First merges alias-based queries to produce roughly one query per
      produced alias.
    - Pushes aggregates up and merges adjacent same-op aggregates so reduction
      indices are not parent-related (enables cost-optimal reduction order).
    - Then normalizes Reorder usage so that each query RHS has a single
      outer Reorder and no interior Reorder nodes.
    """
    merged = merge_queries(plan)
    merged = normalize_reorders_in_plan(merged)
    new_bodies: list[LogicStatement] = []
    for body in merged.bodies:
        if isinstance(body, Query):
            pushed_rhs = push_aggregates_up(body.rhs)
            merged_rhs = Rewrite(PostWalk(merge_mapjoin_rule))(pushed_rhs)
            new_bodies.append(Query(body.lhs, merged_rhs))
        else:
            new_bodies.append(body)
    merged = Plan(tuple(new_bodies))
    return normalize_reorders_in_plan(merged)


def split_mapjoin(node: LogicExpression) -> LogicExpression:
    match node:
        case MapJoin(Literal(mj_op), args) if is_associative(mj_op) and len(args) > 2:
            return MapJoin(Literal(mj_op), (args[0], MapJoin(Literal(mj_op), args[1:])))
        case _:
            return node


def postprocess_plan_after_galley(plan: Plan) -> Plan:
    """
    Postprocessing used after running Galley greedy optimization.

    - Rebuilds the plan so that, for each alias mentioned in the final
      Produces, there is only one Query computing it. This is necessary
      because the greedy optimizer may produce multiple queries with the same
      lhs alias as it reduces different indices, but downstream components
      expect at most one query per alias.
    """
    new_bodies: list[LogicStatement] = []
    for body in plan.bodies:
        if isinstance(body, Query):
            new_rhs = Rewrite(PostWalk(split_mapjoin))(body.rhs)
            new_bodies.append(Query(body.lhs, new_rhs))
        else:
            new_bodies.append(body)
    return Plan(tuple(new_bodies))
