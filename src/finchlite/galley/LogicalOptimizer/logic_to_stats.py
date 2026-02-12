from __future__ import annotations

from collections import OrderedDict

from ...finch_logic import (
    Plan,
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
from finchlite.interface import LazyTensor
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
            reduce_indices = [idx.name for idx in node.idxs]
            st = ST.aggregate(op, init, reduce_indices, arg)
            cache[node] = st
            return st
        

        case Reorder():
            child = insert_statistics(ST, node.arg, bindings, replace, cache)
            cache[node] = child
            return child
        
        case Table():
            if isinstance(node.tns, Literal):
                idxs = [f.name for f in node.idxs]
                tensor = ST(node.tns.val, idxs)
            elif isinstance(node.tns, Alias):
                base_stats = bindings.get(node.tns)
                if base_stats is None :
                    raise ValueError(f"No TensorStats bound to alias {node.tns}")
                
                new_indices = tuple(f.name for f in node.idxs)
                tensor = ST.relabel(base_stats,new_indices)

            if (node not in cache) or replace:
                cache[node] = tensor
            return cache[node]
        
        case Plan():
            last_result = () 
            for body in node.bodies:
                last_result = insert_statistics(ST,body,bindings,replace,cache)
            return last_result

        case _:
            raise TypeError(f"Unhandled node type: {type(node)}")

'''
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
'''

def get_lazy_tensor_stats(
        lazy_tensor : LazyTensor, StatsImpl : TensorStats
)   -> TensorStats:
    trace = lazy_tensor.ctx.trace()
    cache: dict[object, TensorStats] = {}
    bindings: OrderedDict[Alias, TensorStats] = OrderedDict()
    replace = False

    last_stats = None

    for stmt in trace:
        last_stats = insert_statistics(ST=StatsImpl,node=stmt,bindings=bindings,replace=replace,cache=cache)

    return last_stats


