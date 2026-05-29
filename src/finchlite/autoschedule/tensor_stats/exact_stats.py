from __future__ import annotations

from typing import Any

import numpy as np

import finchlite
from finchlite.algebra import FinchOperator
from finchlite.autoschedule.default_schedulers import get_default_scheduler
from finchlite.finch_logic import (
    Aggregate,
    Field,
    Literal,
    LogicExpression,
    MapJoin,
    Relabel,
    Reorder,
    Table,
)
from finchlite.finch_logic.nodes import TableValue

from .numeric_stats import NumericStats
from .tensor_def import TensorDef
from .tensor_stats import BaseTensorStatsFactory


class ExactStatsFactory(BaseTensorStatsFactory["ExactStats"]):
    def __init__(self):
        super().__init__(ExactStats)

    def copy_stats(self, stat: ExactStats) -> ExactStats:
        if not isinstance(stat, ExactStats):
            raise TypeError("copy_stats expected a ExactStats instance")
        return ExactStats.from_def(stat.tensordef.copy(), stat.expr)

    def _mapjoin_join(
        self, new_def: TensorDef, op: FinchOperator, join_args: list[ExactStats]
    ) -> ExactStats:
        if len(join_args) == 0:
            return ExactStats.from_def(new_def, None)
        expr = MapJoin(Literal(op), tuple(s.expr for s in join_args))
        return ExactStats.from_def(new_def, expr)

    def _mapjoin_union(
        self, new_def: TensorDef, op: FinchOperator, union_args: list[ExactStats]
    ) -> ExactStats:
        expr = MapJoin(Literal(op), tuple(s.expr for s in union_args))
        return ExactStats.from_def(new_def, expr)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: ExactStats,
    ) -> ExactStats:
        new_def = TensorDef.aggregate(op, init, reduce_indices, stats.tensordef)
        expr = Aggregate(Literal(op), Literal(init), stats.expr, reduce_indices)
        return ExactStats.from_def(new_def, expr)

    def relabel(
        self, stats: ExactStats, relabel_indices: tuple[Field, ...]
    ) -> ExactStats:
        new_def = TensorDef.relabel(stats.tensordef, relabel_indices)
        expr = Relabel(stats.expr, relabel_indices)
        return ExactStats.from_def(new_def, expr)

    def reorder(
        self, stats: ExactStats, reorder_indices: tuple[Field, ...]
    ) -> ExactStats:
        new_def = TensorDef.reorder(stats.tensordef, reorder_indices)
        expr = Reorder(stats.expr, reorder_indices)
        return ExactStats.from_def(new_def, expr)


class ExactStats(NumericStats):
    def __init__(self, tensor, fields):
        self.tensordef = TensorDef.from_tensor(tensor, fields)
        self.expr = Table(Literal(tensor), fields)
        self.nnz = self.estimate_non_fill_values()

    @classmethod
    def from_def(cls, tensordef: TensorDef, expr: LogicExpression | None) -> ExactStats:
        obj = object.__new__(cls)
        obj.tensordef = tensordef
        obj.expr = expr
        obj.nnz = obj.estimate_non_fill_values()
        return obj

    def estimate_non_fill_values(self) -> float:
        if self.expr is None:
            return 0.0

        result = get_default_scheduler()(self.expr)
        if not isinstance(result, TableValue):
            raise TypeError("estimate_non_fill_value expected a TableValue instance")

        return float(finchlite.count_nonfill_values(result.tns, self.fill_value))

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        return np.log2(np.array(sizes + [self.nnz + 1]))
