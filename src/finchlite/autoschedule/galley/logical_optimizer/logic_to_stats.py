# AI modified: 2026-04-03T00:55:25Z 38d789f35f1c9ba5c8ed00178371222826773dbe
# AI modified: 2026-04-03T01:08:06Z 38d789f35f1c9ba5c8ed00178371222826773dbe
# AI modified: 2026-04-03T01:33:01Z 38d789f35f1c9ba5c8ed00178371222826773dbe
from __future__ import annotations

from collections import OrderedDict

from ....finch_logic import (
    Aggregate,
    Alias,
    Literal,
    LogicNode,
    MapJoin,
    Query,
    Reorder,
    StatsFactory,
    Table,
    TensorStats,
)


def insert_statistics(
    stats_factory: StatsFactory,
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
                insert_statistics(stats_factory, a, bindings, replace, cache)
                for a in node.args
            ]
            if not args:
                raise ValueError("MapJoin expects at least one argument with stats.")
            if len(args) == 1:
                cache[node] = args[0]
                return args[0]
            st = stats_factory.mapjoin(op, *args)
            cache[node] = st
            return st
        case Query():
            stats = insert_statistics(stats_factory, node.rhs, bindings, replace, cache)
            if isinstance(node.lhs, Alias):
                bindings[node.lhs] = stats
            cache[node] = stats
            return stats

        case Aggregate():
            if not isinstance(node.op, Literal):
                raise TypeError("Aggregate.op must be Literal(...).")
            op = node.op.val
            init = node.init.val if isinstance(node.init, Literal) else None
            arg = insert_statistics(stats_factory, node.arg, bindings, replace, cache)
            reduce_indices = list(node.idxs)
            st = stats_factory.aggregate(op, init, tuple(reduce_indices), arg)
            cache[node] = st
            return st

        case Reorder():
            child = insert_statistics(stats_factory, node.arg, bindings, replace, cache)
            st = stats_factory.reorder(child, node.idxs)
            cache[node] = st
            return st

        case Table():
            if isinstance(node.tns, Literal):
                idxs = list(node.idxs)
                tensor = stats_factory(node.tns.val, tuple(idxs))
            elif isinstance(node.tns, Alias):
                base_stats = bindings.get(node.tns)
                if base_stats is None:
                    raise ValueError(f"No TensorStats bound to alias {node.tns}")

                new_indices = tuple(f for f in node.idxs)
                tensor = stats_factory.relabel(base_stats, new_indices)

            if (node not in cache) or replace:
                cache[node] = tensor
            return cache[node]

        case _:
            raise TypeError(f"Unhandled node type: {type(node)}")
