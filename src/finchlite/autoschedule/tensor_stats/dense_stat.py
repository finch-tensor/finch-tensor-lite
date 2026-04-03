# AI modified: 2026-04-03T01:35:32Z 38d789f35f1c9ba5c8ed00178371222826773dbe
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


class DenseStats(NumericStats):
    @classmethod
    def from_def(cls, d: TensorDef) -> Self:
        ds = object.__new__(cls)
        ds.tensordef = d.copy()
        return ds

    @classmethod
    def copy_stats(cls, stat: DenseStats) -> DenseStats:
        """
        Deep copy of a DenseStats object.
        """
        if not isinstance(stat, DenseStats):
            raise TypeError("copy_stats expected a DenseStats instance")
        return cls.from_def(stat.tensordef.copy())

    def estimate_non_fill_values(self) -> float:
        total = 1.0
        for size in self.dim_sizes.values():
            total *= size
        return total

    @classmethod
    def mapjoin(cls, op: FinchOperator, *args: DenseStats) -> DenseStats:
        axes_set = [set(s.index_order) for s in args]
        same_axes = all(axes_set[0] == axes for axes in axes_set)

        def_args = [stat.tensordef for stat in args]
        new_def = TensorDef.mapjoin(op, *def_args)

        if not same_axes:
            # Additional check needed for the case when dimesions do not match
            new_def.fill_value = 0.0

        return cls.from_def(new_def)

    @classmethod
    def aggregate(
        cls,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DenseStats,
    ) -> DenseStats:
        d = stats.tensordef
        new_def = TensorDef.aggregate(op, init, reduce_indices, d)
        return cls.from_def(new_def)

    @classmethod
    def issimilar(cls, a: DenseStats, b: DenseStats) -> bool:
        return (
            isinstance(a, DenseStats)
            and isinstance(b, DenseStats)
            and a.dim_sizes == b.dim_sizes
            and a.fill_value == b.fill_value
        )

    @classmethod
    def relabel(
        cls, stats: DenseStats, relabel_indices: tuple[Field, ...]
    ) -> DenseStats:
        d = stats.tensordef
        new_def = TensorDef.relabel(d, relabel_indices)
        return cls.from_def(new_def)

    @classmethod
    def reorder(
        cls, stats: DenseStats, reorder_indices: tuple[Field, ...]
    ) -> DenseStats:

        d = stats.tensordef
        new_def = TensorDef.reorder(d, reorder_indices)
        return cls.from_def(new_def)
