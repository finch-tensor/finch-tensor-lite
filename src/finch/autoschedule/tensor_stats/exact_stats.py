from __future__ import annotations

from typing import Any

import numpy as np

import finch
from finch.algebra import FinchOperator, ffuncs, is_annihilator, is_identity
from finch.autoschedule.default_schedulers import get_default_scheduler
from finch.finch_logic import (
    Aggregate,
    Field,
    Literal,
    MapJoin,
    Relabel,
    Reorder,
    StatsFactory,
    Table,
)
from finch.finch_logic.nodes import TableValue

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory


class ExactStatsFactory(
    BaseTensorStatsFactory["ExactStats"],
    StatsFactory["ExactStats"],
):
    def __init__(self):
        super().__init__(ExactStats)

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> ExactStats:
        base = super().__call__(tensor, fields)
        expr = Table(Literal(tensor != tensor.fill_value), fields)
        return ExactStats(base, expr=expr)

    def _mapjoin_join(self, op: FinchOperator, *join_args: ExactStats) -> ExactStats:
        base = super()._mapjoin_defs(op, *join_args)
        expr = MapJoin(Literal(ffuncs.and_), tuple(s.expr for s in join_args))
        return ExactStats(base, expr=expr)

    def _mapjoin_union(self, op: FinchOperator, *union_args: ExactStats) -> ExactStats:
        base = super()._mapjoin_defs(op, *union_args)
        expr = MapJoin(Literal(ffuncs.or_), tuple(s.expr for s in union_args))
        return ExactStats(base, expr=expr)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: ExactStats,
    ) -> ExactStats:
        f = stats.fill_value
        bool_op: FinchOperator

        if is_identity(op, f):
            bool_op, bool_init = ffuncs.or_, False
        elif not is_annihilator(op, f) and not is_identity(op, f):
            bool_op, bool_init = ffuncs.or_, True
        else:
            bool_op, bool_init = ffuncs.and_, True

        base = self.aggregate_def(op, init, reduce_indices, stats)
        expr = Aggregate(
            Literal(bool_op), Literal(bool_init), stats.expr, reduce_indices
        )
        return ExactStats(base, expr=expr)

    def relabel(
        self, stats: ExactStats, relabel_indices: tuple[Field, ...]
    ) -> ExactStats:
        base = self.relabel_def(stats, relabel_indices)
        expr = Relabel(stats.expr, relabel_indices)
        return ExactStats(base, expr=expr)

    def reorder(
        self, stats: ExactStats, reorder_indices: tuple[Field, ...]
    ) -> ExactStats:
        base = self.reorder_def(stats, reorder_indices)
        expr = Reorder(stats.expr, reorder_indices)
        return ExactStats(base, expr=expr)


class ExactStats(NumericStats):
    def __init__(
        self,
        base: BaseTensorStats,
        expr: Any,
    ):
        super().__init__(base)
        self.expr = expr
        self.nnz = self.estimate_non_fill_values()

    def estimate_non_fill_values(self) -> float:
        if self.expr is None:
            return 0.0

        result = get_default_scheduler()(self.expr)
        if not isinstance(result, TableValue):
            raise TypeError("estimate_non_fill_value expected a TableValue instance")

        return float(finch.reduce(ffuncs.add, result.tns, init=np.uint64(0)))

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]
        return np.log2(np.array(sizes + [self.nnz + 1]))
