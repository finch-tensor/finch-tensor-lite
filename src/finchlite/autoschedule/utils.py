from ..finch_logic import (
    Aggregate,
    Alias,
    Literal,
    LogicExpression,
    Reorder,
    Table,
)


def is_inplace_expr(
    query_lhs: Alias,
    mapjoin_op: Literal,
    rhs: tuple[LogicExpression, ...],
):
    # TODO: This is a temporary measure till we figure out
    # how to handle the case where args = [lhs_arg, arg1, arg2].
    if len(rhs) > 2:
        return False

    # Return false if rhs does not contain lhs (i.e. not inplace)
    if not any(
        isinstance(arg, Reorder)
        and isinstance(arg.arg, Table)
        and arg.arg.tns == query_lhs
        for arg in rhs
    ):
        return False

    # Return false if the inner argument is not one of the following
    # 1. Queries that perform a Reorder of a single argument.
    # 2. Queries that perform an Aggregate (with mapjoin_op) over a Reorder
    #    of a series of map-joins.
    return all(
        (isinstance(arg, Reorder) and isinstance(arg.arg, Table))
        or (
            isinstance(arg, Aggregate)
            and isinstance(arg.arg, Reorder)
            and arg.op.val == mapjoin_op.val
        )
        for arg in rhs
    )
