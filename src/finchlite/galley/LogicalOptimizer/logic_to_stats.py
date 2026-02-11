from __future__ import annotations

from collections import OrderedDict

from finchlite.galley.TensorStats.stats_interpreter import StatsInterpreter
from finchlite.interface import LazyTensor

from ...finch_logic import (
    Aggregate,
    Alias,
    Literal,
    LogicNode,
    MapJoin,
    Query,
    Reorder,
    Table,
)
from ..TensorStats import TensorStats


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
            reduce_indices = list(node.idxs)
            st = ST.aggregate(op, init, reduce_indices, arg)
            cache[node] = st
            return st

        case Reorder():
            child = insert_statistics(ST, node.arg, bindings, replace, cache)
            cache[node] = child
            return child

        case Table():
            if isinstance(node.tns, Literal):
                idxs = list(node.idxs)
                tensor = ST(node.tns.val, idxs)
            elif isinstance(node.tns, Alias):
                base_stats = bindings.get(node.tns)
                if base_stats is None:
                    raise ValueError(f"No TensorStats bound to alias {node.tns}")

                new_indices = tuple(f for f in node.idxs)
                tensor = ST.relabel(base_stats, new_indices)

            if (node not in cache) or replace:
                cache[node] = tensor
            return cache[node]

        case _:
            raise TypeError(f"Unhandled node type: {type(node)}")


def get_lazy_tensor_stats(
    lazy_tensor: LazyTensor, StatsImpl: type[TensorStats]
) -> TensorStats:
    trace = lazy_tensor.ctx.trace()
    interpreter = StatsInterpreter(StatsImpl=StatsImpl)
    bindings: OrderedDict[Alias, TensorStats] = OrderedDict()
    last_stats: TensorStats | tuple[TensorStats, ...]
    for stmt in trace:
        last_stats = interpreter(stmt, bindings)

    if last_stats is None:
        raise ValueError("Trace was empty or no stats produced")
    if isinstance(last_stats, tuple):
        return last_stats[0]

    return last_stats
