from __future__ import annotations

from typing import Any

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field

from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory


class DummyStatsFactory(BaseTensorStatsFactory["DummyStats"]):
    def __init__(self):
        super().__init__(DummyStats)

    def _mapjoin_join(
        self, new_def: BaseTensorStats, op: FinchOperator, join_args: list[DummyStats]
    ) -> DummyStats:
        return DummyStats.from_def(new_def)

    def _mapjoin_union(
        self, new_def: BaseTensorStats, op: FinchOperator, union_args: list[DummyStats]
    ) -> DummyStats:
        return DummyStats.from_def(new_def)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DummyStats,
    ) -> DummyStats:
        new_def = super().aggregate(op, init, reduce_indices, stats)
        return DummyStats.from_def(new_def)

    def relabel(
        self, stats: DummyStats, relabel_indices: tuple[Field, ...]
    ) -> DummyStats:
        new_def = super().relabel(stats, relabel_indices)
        return DummyStats.from_def(new_def)

    def reorder(
        self, stats: DummyStats, reorder_indices: tuple[Field, ...]
    ) -> DummyStats:
        new_def = super().reorder(stats, reorder_indices)
        return DummyStats.from_def(new_def)


class DummyStats(BaseTensorStats):
    pass
