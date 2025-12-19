from __future__ import annotations

from collections import OrderedDict

from finchlite.finch_logic import (
    Aggregate,
    Alias,
    Field,
    Literal,
    LogicNode,
    MapJoin,
    Query,
    Reorder,
    Table,
    Value,
)
from finchlite.galley.TensorStats import TensorStats
from finchlite.interface import LazyTensor


def insert_statistics(
    ST,
    node: LogicNode,
    bindings: OrderedDict[Alias, TensorStats],
    replace: bool,
    cache: dict[object, TensorStats],
) -> TensorStats:
    if node in cache:
        return cache[node]

    match node:
        case MapJoin():
            if not isinstance(node.op, Literal):
                raise TypeError("MapJoin.op must be Literal(...).")
            op = node.op.val
            args = [
                insert_statistics(ST, a, bindings, replace, cache) for a in node.args
            ]
            if not args:
                raise ValueError("MapJoin expects at least one argument with stats.")
            st = ST.mapjoin(op, *args)
            cache[node] = st
            return st
        case Query():
            stats = insert_statistics(ST, node.rhs, bindings, replace, cache)
            if isinstance(node.lhs, Alias):
                bindings[node.lhs] = stats
            cache[node] = stats
            return stats

        case Aggregate():
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

        case Alias():
            st = bindings.get(node)
            if st is None:
                raise ValueError(f"No TensorStats bound to alias {node}")
            cache[node] = st
            return st

        case Reorder():
            child = insert_statistics(ST, node.arg, bindings, replace, cache)
            cache[node] = child
            return child

        # We need implementation for reformat and relabel

        case Table():
            if not isinstance(node.tns, Literal):
                raise TypeError("Table.tns must be Literal(...).")

            tensor = node.tns.val
            idxs = [f.name for f in node.idxs]

            if (node not in cache) or replace:
                cache[node] = ST(tensor, idxs)
            return cache[node]

        case Value() | Literal():
            val = node.val if isinstance(node, Literal) else node.ex
            st = ST(val)
            cache[node] = st
            return st

        case _:
            raise TypeError(f"Unsupported node type: {type(node).__name__}")


def get_lazy_tensor_stats(
    lazy_tensor: LazyTensor, StatsImpl: TensorStats
) -> TensorStats:
    root_node = lazy_tensor.data
    cache: dict[object, TensorStats] = {}
    bindings: OrderedDict[Alias, TensorStats] = OrderedDict()
    replace = False
    return insert_statistics(
        ST=StatsImpl, node=root_node, bindings=bindings, replace=replace, cache=cache
    )
