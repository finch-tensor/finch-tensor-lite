from ..finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicExpression,
    Reorder,
    Table,
)


def is_inplace_expr(
    query_lhs: Alias,
    mapjoin_op: Literal,
    mapjoin_idxs: tuple[Field, ...],
    mapjoin_args: tuple[LogicExpression, ...],
):
    lhs_arg, *non_lhs_arg = mapjoin_args

    # Return false if rhs does not contain lhs (i.e. not inplace)
    if not (
        isinstance(lhs_arg, Reorder)
        and isinstance(lhs_arg.arg, Table)
        and lhs_arg.arg.tns == query_lhs
        and lhs_arg.idxs == mapjoin_idxs
    ):
        return False

    # Return false if the inner argument is not one of the following
    # 1. Queries that perform a Reorder of a single argument.
    # 2. Queries that perform an Aggregate (with mapjoin_op) over a Reorder
    #    of a series of map-joins.
    return all(
        isinstance(arg, Reorder)
        or (
            isinstance(arg, Aggregate)
            and isinstance(arg.arg, Reorder)
            and arg.op.val == mapjoin_op.val
        )
        for arg in non_lhs_arg
    )
