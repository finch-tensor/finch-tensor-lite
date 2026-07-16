from __future__ import annotations

from typing import Any

import numpy as np

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field, StatsFactory

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStatsFactory


class DenseStatsFactory(BaseTensorStatsFactory["DenseStats"], StatsFactory["DenseStats"]):
    def __init__(self):
        super().__init__(DenseStats)

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> DenseStats:
        return DenseStats(super().__call__(tensor, fields))

    def _mapjoin_union(self, op: FinchOperator, *union_args: DenseStats) -> DenseStats:
        base = super()._mapjoin_defs(op, *union_args)
        return DenseStats(base)

    def _mapjoin_join(self, op: FinchOperator, *join_args: DenseStats) -> DenseStats:
        base = super()._mapjoin_defs(op, *join_args)
        return DenseStats(base)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DenseStats,
    ) -> DenseStats:
        base = self.aggregate_def(op, init, reduce_indices, stats)
        return DenseStats(base)

    def relabel(
        self, stats: DenseStats, relabel_indices: tuple[Field, ...]
    ) -> DenseStats:
        base = self.relabel_def(stats, relabel_indices)
        return DenseStats(base)

    def reorder(
        self, stats: DenseStats, reorder_indices: tuple[Field, ...]
    ) -> DenseStats:
        base = self.reorder_def(stats, reorder_indices)
        return DenseStats(base)


class DenseStats(NumericStats):
    def estimate_non_fill_values(self) -> float:
        total = 1.0
        for size in self.dim_sizes.values():
            total *= size
        return total

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]

        return np.array(np.log2(sizes))
