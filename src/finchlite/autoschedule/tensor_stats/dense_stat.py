from __future__ import annotations

from typing import Any

import numpy as np

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStatsFactory


class DenseStatsFactory(BaseTensorStatsFactory["DenseStats"]):
    def __init__(self):
        super().__init__(DenseStats)

    def _mapjoin_union(self, op: FinchOperator, *union_args: DenseStats) -> DenseStats:
        base_stats = super()._mapjoin_defs(op, *union_args)
        return DenseStats.from_def(base_stats)

    def _mapjoin_join(self, op: FinchOperator, *join_args: DenseStats) -> DenseStats:
        base_stats = super()._mapjoin_defs(op, *join_args)
        return DenseStats.from_def(base_stats)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DenseStats,
    ) -> DenseStats:
        new_def = self.aggregate_def(op, init, reduce_indices, stats)
        return DenseStats.from_def(new_def)

    def relabel(
        self, stats: DenseStats, relabel_indices: tuple[Field, ...]
    ) -> DenseStats:
        new_def = self.relabel_def(stats, relabel_indices)
        return DenseStats.from_def(new_def)

    def reorder(
        self, stats: DenseStats, reorder_indices: tuple[Field, ...]
    ) -> DenseStats:
        new_def = self.reorder_def(stats, reorder_indices)
        return DenseStats.from_def(new_def)


class DenseStats(NumericStats):
    def estimate_non_fill_values(self) -> float:
        total = 1.0
        for size in self.dim_sizes.values():
            total *= size
        return total

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]

        return np.array(np.log2(sizes))
