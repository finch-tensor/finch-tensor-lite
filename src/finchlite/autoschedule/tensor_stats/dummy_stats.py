from __future__ import annotations

from typing import Any

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field

from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory


class DummyStatsFactory(BaseTensorStatsFactory["DummyStats"]):
    def __init__(self):
        super().__init__(DummyStats)

    def _mapjoin_join(self, op: FinchOperator, *join_args: DummyStats) -> DummyStats:
        base_stats = super()._mapjoin_defs(op, *join_args)
        return DummyStats.from_base_stats(base_stats)

    def _mapjoin_union(self, op: FinchOperator, *union_args: DummyStats) -> DummyStats:
        base_stats = super()._mapjoin_defs(op, *union_args)
        return DummyStats.from_base_stats(base_stats)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DummyStats,
    ) -> DummyStats:
        base_stats = self.aggregate_def(op, init, reduce_indices, stats)
        return DummyStats.from_base_stats(base_stats)

    def relabel(
        self, stats: DummyStats, relabel_indices: tuple[Field, ...]
    ) -> DummyStats:
        base_stats = self.relabel_def(stats, relabel_indices)
        return DummyStats.from_base_stats(base_stats)

    def reorder(
        self, stats: DummyStats, reorder_indices: tuple[Field, ...]
    ) -> DummyStats:
        base_stats = self.reorder_def(stats, reorder_indices)
        return DummyStats.from_base_stats(base_stats)


class DummyStats(BaseTensorStats):
    pass
