from finchlite.algebra import ffuncs
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

# Cost parameters ported from Julia
# TODO: Julia comments says these need to be adjusted.
# confirm that these values are OK
SEQ_READ_COST = 1
SEQ_WRITE_COST = 5
RANDOM_READ_COST = 5
RANDOM_WRITE_COST = 10
DENSE_ALLOCATE_COST = 0.5
SPARSE_ALLOCATE_COST = 60


def stats_with_false_fill(
    stats_factory: StatsFactory,
    stats: TensorStats,
) -> TensorStats:
    stats_copy = stats_factory.copy(stats)
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


# Transpose penalty funcs more like Julia version
# TODO: used in bnb
def cost_of_reformat(stats: TensorStats) -> float:
    if not isinstance(stats, NumericStats):
        return 0.0
    nnz = stats.estimate_non_fill_values()
    space = stats.get_dim_space_size(tuple(stats.index_order))
    if space == 0 or space == float("inf"):
        return nnz * SPARSE_ALLOCATE_COST
    if nnz / space > 0.1:
        return nnz * DENSE_ALLOCATE_COST * 0.01
    return nnz * SPARSE_ALLOCATE_COST


def get_reformat_set(
    input_stats: list[TensorStats], prefix: tuple[Field, ...]
) -> frozenset[int]:
    return frozenset(
        i for i, stats in enumerate(input_stats) if needs_reformat(stats, prefix)
    )


# End transpose penalty funcs


def needs_reformat(stats: TensorStats, prefix: tuple[Field, ...]) -> bool:
    prefix_pos = {idx: pos for pos, idx in enumerate(prefix)}
    current_loop = -1.0
    reformat = False
    for idx in stats.index_order:
        idx_loop = float(prefix_pos.get(idx, float("inf")))
        if idx_loop < current_loop:
            reformat = True
        current_loop = idx_loop
    return reformat


# TODO: Julia's estimate used conditional indices for per-axis nnz fractions.
# For now use prefix density.
def _approx_axis_density(stats: TensorStats, rel_vars: set[Field]) -> float:
    if not isinstance(stats, NumericStats):
        return 1.0
    nnz = stats.estimate_non_fill_values()
    space = stats.get_dim_space_size(tuple(rel_vars))
    if space == 0 or space == float("inf"):
        return 0.0
    return nnz / space


def get_prefix_cost(
    new_prefix: tuple[Field, ...],
    conjunct_stats: list[TensorStats],
    disjunct_stats: list[TensorStats],
    stats_factory: StatsFactory,
    output_vars: tuple[Field, ...] | None = None,
) -> float:
    if not new_prefix:
        return 0.0

    new_var = new_prefix[-1]
    prefix_set = set(new_prefix)

    rel_conjuncts = [
        stat for stat in conjunct_stats if prefix_set.intersection(stat.index_order)
    ]
    rel_disjuncts = [
        stat for stat in disjunct_stats if prefix_set.intersection(stat.index_order)
    ]

    lookups = get_loop_lookups(new_prefix, rel_conjuncts, rel_disjuncts, stats_factory)

    lookup_factor = 0.0
    seen: list[TensorStats] = []
    for stat in rel_conjuncts + rel_disjuncts:
        if any(stat is s for s in seen):
            continue
        seen.append(stat)

        if new_var not in set(stat.index_order):
            continue

        rel_vars = set(stat.index_order) & prefix_set
        approx_sparsity = _approx_axis_density(stat, rel_vars)
        is_dense = approx_sparsity > 0.05

        if needs_reformat(stat, new_prefix):
            lookup_factor += SEQ_READ_COST / 5 if is_dense else SEQ_READ_COST
            continue

        lookup_factor += SEQ_READ_COST / 5 if is_dense else SEQ_READ_COST

    if output_vars is not None and new_var in output_vars:
        output_list = list(output_vars)
        new_var_idx = output_list.index(new_var)
        prefix_positions = [
            output_list.index(v) for v in new_prefix if v in output_list
        ]
        max_var_idx = max(prefix_positions)
        is_rand_write = new_var_idx != max_var_idx
        lookup_factor += RANDOM_WRITE_COST if is_rand_write else SEQ_WRITE_COST
    else:
        lookup_factor += SEQ_WRITE_COST

    return lookups * lookup_factor


def loop_order_cost(
    expr: LogicExpression,
    loop_order: tuple[Field, ...],
    stats_factory: StatsFactory,
    stats_bindings: dict[Alias, TensorStats],
    output_vars: tuple[Field, ...] | None = None,
) -> float:
    stats_bindings_2 = stats_bindings.copy()
    cache: dict[object, TensorStats] = {}
    conjunct_stats, disjunct_stats = get_conjunctive_and_disjunctive_inputs(
        expr, stats_factory, stats_bindings_2, cache
    )
    cost = 0.0
    for j in range(1, len(loop_order) + 1):
        cost += get_prefix_cost(
            loop_order[:j],
            conjunct_stats,
            disjunct_stats,
            stats_factory,
            output_vars,
        )
    return cost
