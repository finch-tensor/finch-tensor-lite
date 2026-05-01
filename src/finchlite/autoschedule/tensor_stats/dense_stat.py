from __future__ import annotations

from typing import Any, Self

import numpy as np

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

    def _mapjoin_union(
        self, new_def: TensorDef, op: FinchOperator, union_args: list[DenseStats]
    ) -> DenseStats:
        axes_set = [set(s.index_order) for s in union_args]
        same_axes = all(axes_set[0] == axes for axes in axes_set)

        if not same_axes:
            new_def.fill_value = 0.0
        return DenseStats.from_def(new_def)

    def _mapjoin_join(
        self, new_def: TensorDef, op: FinchOperator, join_args: list[DenseStats]
    ) -> DenseStats:
        axes_set = [set(s.index_order) for s in join_args]
        same_axes = all(axes_set[0] == axes for axes in axes_set)

        if not same_axes:
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

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]

        return np.array(np.log2(sizes))
