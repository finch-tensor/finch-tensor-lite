# AI modified: 2026-04-03T01:53:09Z 6877aca3b7b141666a6b9c061af7f26a4f65c0dd
# AI modified: 2026-04-03T02:16:03Z 6877aca3b7b141666a6b9c061af7f26a4f65c0dd
# AI modified: 2026-04-03T02:34:01Z 6877aca3b7b141666a6b9c061af7f26a4f65c0dd
from __future__ import annotations

from typing import Any, Self

from finchlite.finch_logic import Field

from ..algebra import FinchOperator
from .tensor_def import TensorDef
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory


class DummyStatsFactory(BaseTensorStatsFactory["DummyStats"]):
    def __init__(self):
        super().__init__(DummyStats)

    def copy_stats(self, stat: DummyStats) -> DummyStats:
        if not isinstance(stat, DummyStats):
            raise TypeError("copy_stats expected a DummyStats instance")
        return DummyStats.from_def(stat.tensordef.copy())

    def mapjoin(self, op: FinchOperator, *args: DummyStats) -> DummyStats:
        axes_set = [set(s.index_order) for s in args]
        same_axes = all(axes_set[0] == axes for axes in axes_set)

        def_args = [stat.tensordef for stat in args]
        new_def = TensorDef.mapjoin(op, *def_args)

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

    def issimilar(self, a: DummyStats, b: DummyStats) -> bool:
        return (
            isinstance(a, DummyStats)
            and isinstance(b, DummyStats)
            and a.dim_sizes == b.dim_sizes
            and a.tensordef.index_order == b.tensordef.index_order
        )

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
