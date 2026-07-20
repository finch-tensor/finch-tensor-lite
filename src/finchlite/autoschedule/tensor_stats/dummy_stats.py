from __future__ import annotations

from typing import Any

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field, StatsFactory

from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory


class DummyStatsFactory(
    BaseTensorStatsFactory["DummyStats"], StatsFactory["DummyStats"]
):
    def __init__(self):
        super().__init__(DummyStats)

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> DummyStats:
        return DummyStats(super().__call__(tensor, fields))

    def _mapjoin_join(self, op: FinchOperator, *join_args: DummyStats) -> DummyStats:
        base = super()._mapjoin_defs(op, *join_args)
        return DummyStats(base)

    def _mapjoin_union(self, op: FinchOperator, *union_args: DummyStats) -> DummyStats:
        base = super()._mapjoin_defs(op, *union_args)
        return DummyStats(base)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DummyStats,
    ) -> DummyStats:
        base = self.aggregate_def(op, init, reduce_indices, stats)
        return DummyStats(base)

    def relabel(
        self, stats: DummyStats, relabel_indices: tuple[Field, ...]
    ) -> DummyStats:
        base = self.relabel_def(stats, relabel_indices)
        return DummyStats(base)

    def reorder(
        self, stats: DummyStats, reorder_indices: tuple[Field, ...]
    ) -> DummyStats:
        base = self.reorder_def(stats, reorder_indices)
        return DummyStats(base)


class DummyStats(BaseTensorStats):
    pass
