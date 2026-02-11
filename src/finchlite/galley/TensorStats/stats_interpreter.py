from __future__ import annotations

from collections import OrderedDict
from typing import overload

import numpy as np

from finchlite.algebra.tensor import TensorFType
from finchlite.finch_logic import LogicInterpreter
from finchlite.finch_logic.nodes import LogicExpression, LogicStatement

from ...finch_logic import (
    Aggregate,
    Alias,
    Literal,
    LogicNode,
    MapJoin,
    Plan,
    Produces,
    Query,
    Relabel,
    Reorder,
    Table,
)
from ..TensorStats import TensorStats


class StatsInterpreter:
    def __init__(self, StatsImpl: type[TensorStats], verbose=False):
        self.ST = StatsImpl
        self.verbose = verbose

    def __call__(
        self, node: LogicNode, bindings: OrderedDict[Alias, TensorStats]
    ) -> TensorStats | tuple[TensorStats, ...]:
        machine = StatsMachine(
            StatsImpl=self.ST, bindings=bindings, verbose=self.verbose
        )
        return machine(node)


class StatsMachine:
    def __init__(self, StatsImpl: type[TensorStats], bindings=None, verbose=False):
        self.ST = StatsImpl
        if bindings is None:
            bindings = OrderedDict()
        self.bindings = bindings
        self.verbose = verbose

    @overload
    def __call__(self, node: LogicExpression) -> TensorStats: ...

    @overload
    def __call__(self, node: Alias) -> TensorStats: ...

    @overload
    def __call__(self, node: LogicStatement) -> tuple[TensorStats, ...]: ...

    @overload
    def __call__(self, node: LogicNode) -> TensorStats | tuple[TensorStats, ...]: ...

    def __call__(self, node) -> TensorStats | tuple[TensorStats, ...]:
        if self.verbose:
            print(f"Evaluating: {node}")
        match node:
            case Plan():
                last_result : TensorStats | tuple[TensorStats, ...] = ()
                for body in node.bodies:
                    last_result = self(body)
                return last_result

            case Query():
                rhs_stats = self(node.rhs)
                self.bindings[node.lhs] = rhs_stats
                return rhs_stats

            case Alias():
                stats = self.bindings.get(node)
                if stats is None:
                    raise ValueError(f"undefined tensor alias {node}")
                return stats

            case Table(tns, idxs):
                if isinstance(tns, Literal):
                    tensor = self.ST(tns.val, idxs)
                elif isinstance(tns, Alias):
                    base_stats = self.bindings.get(tns)
                    if base_stats is None:
                        raise ValueError(f"No TensorStats bound to alias {node.tns}")

                    new_indices = tuple(f for f in node.idxs)
                    tensor = self.ST.relabel(base_stats, new_indices)
                return tensor

            case MapJoin():
                if not isinstance(node.op, Literal):
                    raise TypeError("MapJoin.op must be Literal(...).")
                child_stats_list: list[TensorStats] = []
                for arg in node.args:
                    res = self(arg)
                    child_stats_list.append(res)

                """"
                child_stats = [self(arg) for arg in node.args]
                if not child_stats:
                    raise ValueError("MapJoin expects at least one argument with stats.")
                """
                return self.ST.mapjoin(node.op.val, *child_stats_list)

            case Aggregate():
                if not isinstance(node.op, Literal):
                    raise TypeError("Aggregate.op must be Literal(...).")
                op = node.op.val
                init = node.init.val if isinstance(node.init, Literal) else None
                arg2 = self(node.arg)
                reduce_indices = node.idxs
                return self.ST.aggregate(op, init, reduce_indices, arg2)

            case Reorder():
                return self(node.arg)

            case Relabel():
                base_stats = self(node.arg)
                new_indices = tuple(f for f in node.idxs)
                return self.ST.relabel(base_stats, new_indices)

            case Produces(args):
                return tuple(self(arg) for arg in args)

            case _:
                raise TypeError(f"Unhandled node type {type(node)}")


def calculate_estimated_error(
    node: LogicNode,
    StatsImpl: type[TensorStats],
    logic_bindings: OrderedDict[Alias, TensorFType],
    stats_bindings: OrderedDict[Alias, TensorStats],
) -> tuple[float, ...]:
    if logic_bindings is None:
        logic_bindings = OrderedDict()

    if stats_bindings is None:
        stats_bindings = OrderedDict()

    logic_interpreter = LogicInterpreter()
    actual_result = logic_interpreter(node, logic_bindings)

    if not isinstance(actual_result, tuple):
        actual_result = (actual_result,)

    stats_interpreter = StatsInterpreter(StatsImpl=StatsImpl)
    stats_result = stats_interpreter(node, stats_bindings)

    if not isinstance(stats_result, tuple):
        stats_result = (stats_result,)

    errors = []
    for actual_tns, stats_obj in zip(actual_result, stats_result):
        actual_nnz = float(np.count_nonzero(actual_tns.to_numpy()))
        est_nnz = float(stats_obj.estimate_non_fill_values())

        if actual_nnz == 0.0:
            if est_nnz == 0.0:
                rel_err = 0.0
            else:
                rel_err = float("inf")

        else:
            rel_err = abs(actual_nnz - est_nnz) / actual_nnz

        errors.append(rel_err)
    return tuple(errors)
