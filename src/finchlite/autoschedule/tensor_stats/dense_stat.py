# AI modified: 2026-04-03T01:35:32Z 38d789f35f1c9ba5c8ed00178371222826773dbe
# AI modified: 2026-04-03T01:53:09Z 6877aca3b7b141666a6b9c061af7f26a4f65c0dd
# AI modified: 2026-04-03T02:16:03Z 6877aca3b7b141666a6b9c061af7f26a4f65c0dd
# AI modified: 2026-04-03T02:34:01Z 6877aca3b7b141666a6b9c061af7f26a4f65c0dd
from __future__ import annotations

from typing import Any, Self

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field

from .numeric_stats import NumericStats
from .tensor_def import TensorDef
from .tensor_stats import BaseTensorStatsFactory


class DenseStatsFactory(BaseTensorStatsFactory["DenseStats"]):
    def __init__(self):
        super().__init__(DenseStats)

    def copy_stats(self, stat: DenseStats) -> DenseStats:
        if not isinstance(stat, DenseStats):
            raise TypeError("copy_stats expected a DenseStats instance")
        return DenseStats.from_def(stat.tensordef.copy())

    def mapjoin(self, op: FinchOperator, *args: DenseStats) -> DenseStats:
        axes_set = [set(s.index_order) for s in args]
        same_axes = all(axes_set[0] == axes for axes in axes_set)

        def_args = [stat.tensordef for stat in args]
        new_def = TensorDef.mapjoin(op, *def_args)

        if not same_axes:
            # Additional check needed for the case when dimesions do not match
            new_def.fill_value = 0.0

        return DenseStats.from_def(new_def)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DenseStats,
    ) -> DenseStats:
        d = stats.tensordef
        new_def = TensorDef.aggregate(op, init, reduce_indices, d)
        return DenseStats.from_def(new_def)

    def issimilar(self, a: DenseStats, b: DenseStats) -> bool:
        return (
            isinstance(a, DenseStats)
            and isinstance(b, DenseStats)
            and a.dim_sizes == b.dim_sizes
            and a.fill_value == b.fill_value
        )

    def relabel(
        self, stats: DenseStats, relabel_indices: tuple[Field, ...]
    ) -> DenseStats:
        d = stats.tensordef
        new_def = TensorDef.relabel(d, relabel_indices)
        return DenseStats.from_def(new_def)

    def reorder(
        self, stats: DenseStats, reorder_indices: tuple[Field, ...]
    ) -> DenseStats:
        d = stats.tensordef
        new_def = TensorDef.reorder(d, reorder_indices)
        return DenseStats.from_def(new_def)


class DenseStats(NumericStats):
    @classmethod
    def from_def(cls, d: TensorDef) -> Self:
        ds = object.__new__(cls)
        ds.tensordef = d.copy()
        return ds

    def estimate_non_fill_values(self) -> float:
        total = 1.0
        for size in self.dim_sizes.values():
            total *= size
        return total
