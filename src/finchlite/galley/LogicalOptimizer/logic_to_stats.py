from __future__ import annotations

from collections import OrderedDict

from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicNode,
    MapJoin,
    Reformat,
    Reorder,
    Table,
    Value,
)
from finchlite.galley.TensorStats.tensor_stats import TensorStats


def insert_statistics(
    ST,
    node: LogicNode,
    bindings: OrderedDict[Alias, TensorStats],
    replace: bool,
    cache: dict[object, TensorStats],
) -> TensorStats:
    if node in cache:
        return cache[node]

    if isinstance(node, MapJoin):
        if not isinstance(node.op, Literal):
            raise TypeError("MapJoin.op must be Literal(...).")
        op = node.op.val

        args = [insert_statistics(ST, a, bindings, replace, cache) for a in node.args]
        if not args:
            raise ValueError("MapJoin expects at least one argument with stats.")

        st = ST.mapjoin(op, *args)
        cache[node] = st
        return st

    if isinstance(node, Aggregate):
        if not isinstance(node.op, Literal):
            raise TypeError("Aggregate.op must be Literal(...).")
        op = node.op.val
        init = node.init.val if isinstance(node.init, Literal) else None
        arg = insert_statistics(ST, node.arg, bindings, replace, cache)
        reduce_indices = list(
            dict.fromkeys(
                [i.name if isinstance(i, Field) else str(i) for i in node.idxs]
            )
        )
        st = ST.aggregate(op, init, reduce_indices, arg)
        cache[node] = st
        return st

    if isinstance(node, Alias):
        st = bindings.get(node)
        cache[node] = st
        return st

    if isinstance(node, (Reformat, Reorder)):
        child = insert_statistics(ST, node.arg, bindings, replace, cache)
        cache[node] = child
        return child

    if isinstance(node, Table):
        if not isinstance(node.tns, Literal):
            raise TypeError("Table.tns must be Literal(...).")

        tensor = node.tns.val
        idxs = [f.name for f in node.idxs]

        if (node not in cache) or replace:
            cache[node] = ST(tensor, idxs)
        return cache[node]

    if isinstance(node, (Value, Literal)):
        val = node.val if isinstance(node, Literal) else node.ex
        st = ST(val)
        cache[node] = st
        return st

    raise TypeError(f"Unsupported node type: {type(node).__name__}")
