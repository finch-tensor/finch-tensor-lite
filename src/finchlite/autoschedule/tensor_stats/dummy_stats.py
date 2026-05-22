from __future__ import annotations

from typing import Any, Self

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field

from .tensor_def import TensorDef
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory


class DummyStatsFactory(BaseTensorStatsFactory["DummyStats"]):
    def __init__(self):
        super().__init__(DummyStats)

    def copy_stats(self, stat: DummyStats) -> DummyStats:
        if not isinstance(stat, DummyStats):
            raise TypeError("copy_stats expected a DummyStats instance")
        return DummyStats.from_def(stat.tensordef.copy())

    def _mapjoin_join(
        self, new_def: TensorDef, op: FinchOperator, join_args: list[DummyStats]
    ) -> DummyStats:
        axes_set = [set(s.index_order) for s in join_args]
        same_axes = all(axes_set[0] == axes for axes in axes_set)

        if not same_axes:
            new_def.fill_value = 0.0

        return DummyStats.from_def(new_def)

    def _mapjoin_union(
        self, new_def: TensorDef, op: FinchOperator, union_args: list[DummyStats]
    ) -> DummyStats:
        axes_set = [set(s.index_order) for s in union_args]
        same_axes = all(axes_set[0] == axes for axes in axes_set)

        if not same_axes:
            new_def.fill_value = 0.0

        return DummyStats.from_def(new_def)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DummyStats,
    ) -> DummyStats:
        d = stats.tensordef
        new_def = TensorDef.aggregate(op, init, reduce_indices, d)
        return DummyStats.from_def(new_def)

    def relabel(
        self, stats: DummyStats, relabel_indices: tuple[Field, ...]
    ) -> DummyStats:
        d = stats.tensordef
        new_def = TensorDef.relabel(d, relabel_indices)
        return DummyStats.from_def(new_def)

    def reorder(
        self, stats: DummyStats, reorder_indices: tuple[Field, ...]
    ) -> DummyStats:
        d = stats.tensordef
        new_def = TensorDef.reorder(d, reorder_indices)
        return DummyStats.from_def(new_def)


class DummyStats(BaseTensorStats):
    @classmethod
    def from_def(cls, d: TensorDef) -> Self:
        ds = object.__new__(cls)
        ds.tensordef = d.copy()
        return ds
