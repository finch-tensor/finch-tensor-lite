from finchlite.algebra import ffuncs
from finchlite.algebra.utils import is_subsequence
from finchlite.autoschedule.galley.logical_optimizer import insert_statistics
from finchlite.autoschedule.tensor_stats.numeric_stats import NumericStats
from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicExpression,
    Reorder,
    StatsFactory,
    Table,
    TensorStats,
)
from finchlite.finch_logic.nodes import MapJoin
from finchlite.symbolic import PostOrderDFS


def stats_with_false_fill(
    stats_factory: StatsFactory,
    stats: TensorStats,
) -> TensorStats:
    stats_copy = stats_factory.copy_stats(stats)
    stats_copy.fill_value = False
    return stats_copy


def merge_prefix_stats(
    stats_factory: StatsFactory,
    op,
    args: list[TensorStats],
) -> TensorStats | None:
    if not args:
        return None
    if len(args) == 1:
        return args[0]
    return stats_factory.mapjoin(op, *args)


def get_conjunctive_and_disjunctive_inputs(
    expr: LogicExpression,
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
    cache: dict[object, TensorStats],
    disjunct_branch: bool = False,
) -> tuple[list[TensorStats], list[TensorStats]]:
    match expr:
        case Aggregate(arg=arg):
            return get_conjunctive_and_disjunctive_inputs(
                arg, stats_factory, stats_bindings, cache, disjunct_branch
            )
        case Reorder(arg=arg):
            return get_conjunctive_and_disjunctive_inputs(
                arg, stats_factory, stats_bindings, cache, disjunct_branch
            )
        case MapJoin(Literal() as op_node, args):
            conjuncts = []
            disjuncts = []
            for arg in args:
                arg_stats = insert_statistics(
                    stats_factory, arg, stats_bindings, replace=False, cache=cache
                )
                arg_is_conjunct = op_node.val.is_annihilator(arg_stats.fill_value)
                arg_conjuncts, arg_disjuncts = get_conjunctive_and_disjunctive_inputs(
                    arg,
                    stats_factory,
                    stats_bindings,
                    cache,
                    disjunct_branch or not arg_is_conjunct,
                )
                conjuncts.extend(arg_conjuncts)
                disjuncts.extend(arg_disjuncts)
            return conjuncts, disjuncts
        case Table():
            stats = insert_statistics(
                stats_factory, expr, stats_bindings, replace=False, cache=cache
            )
            if disjunct_branch:
                return [], [stats]
            return [stats], []
        case _:
            return [], []


def get_loop_lookups(
    prefix: tuple[Field, ...],
    conjunct_stats: list[TensorStats],
    disjunct_stats: list[TensorStats],
    stats_factory: StatsFactory,
) -> float:
    prefix_set = set(prefix)
    rel_conjuncts = [
        stats_with_false_fill(stats_factory, stat)
        for stat in conjunct_stats
        if prefix_set.intersection(stat.index_order)
    ]
    rel_disjuncts = [
        stats_with_false_fill(stats_factory, stat)
        for stat in disjunct_stats
        if prefix_set.intersection(stat.index_order)
    ]

    conjunct_index_set = set().union(*(stat.index_order for stat in rel_conjuncts))
    if not rel_disjuncts or prefix_set.issubset(conjunct_index_set):
        loop_stats = merge_prefix_stats(stats_factory, ffuncs.and_, rel_conjuncts)
    elif not rel_conjuncts:
        loop_stats = merge_prefix_stats(stats_factory, ffuncs.or_, rel_disjuncts)
    else:
        disjunct_union = merge_prefix_stats(stats_factory, ffuncs.or_, rel_disjuncts)
        loop_stats = merge_prefix_stats(
            stats_factory,
            ffuncs.and_,
            rel_conjuncts + ([disjunct_union] if disjunct_union is not None else []),
        )

    if loop_stats is None:
        return 0.0

    reduce_idxs = tuple(f for f in loop_stats.index_order if f not in prefix_set)
    projected = stats_factory.aggregate(ffuncs.or_, False, reduce_idxs, loop_stats)
    return projected.estimate_non_fill_values()


def transpose_penalty(
    expr: LogicExpression,
    loop_prefix: tuple[Field, ...],
    stats_bindings: dict[Alias, TensorStats],
) -> float:
    penalty = 0.0
    for node in PostOrderDFS(expr):
        match node:
            case Table(Alias() as tns, idxs):
                base = stats_bindings.get(tns)
                # test requires isinstance(base, NumericStats)
                if isinstance(base, NumericStats) and not is_subsequence(
                    tuple(idxs), loop_prefix
                ):
                    penalty += base.estimate_non_fill_values()
            case _:
                pass
    return penalty


def loop_order_cost(
    expr: LogicExpression,
    loop_order: tuple[Field, ...],
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
) -> float:
    stats_bindings_2 = stats_bindings.copy()
    cache: dict[object, TensorStats] = {}
    conjunct_stats, disjunct_stats = get_conjunctive_and_disjunctive_inputs(
        expr, stats_factory, stats_bindings_2, cache
    )
    cost = 0.0
    for j in range(1, len(loop_order) + 1):
        cost += get_loop_lookups(
            loop_order[:j], conjunct_stats, disjunct_stats, stats_factory
        )
    cost += transpose_penalty(expr, loop_order, stats_bindings)
    return cost
