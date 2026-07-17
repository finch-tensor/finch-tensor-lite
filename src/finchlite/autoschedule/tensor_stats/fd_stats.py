from __future__ import annotations

from typing import Any

import numpy as np

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field, StatsFactory
from abc import ABC, abstractmethod

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStatsFactory

class FDStatsFactory(BaseTensorStatsFactory["FDStats"], StatsFactory["FDStats"]):
    def __init__(self):
        super().__init__(FDStats)

    def __call__(self, tensor: Tensor, fields: tuple[Field, ...]) -> FDStats:
        base = super().__call__(tensor, fields)
        if isinstance(tensor, BufferizedNDArray)
        return FDStats(base)

    def _mapjoin_union(self, op: FinchOperator, *union_args: FDStats) -> FDStats:
        base = super()._mapjoin_defs(op, *union_args)
        return FDStats(base)

    def _mapjoin_join(self, op: FinchOperator, *join_args: FDStats) -> FDStats:
        base = super()._mapjoin_defs(op, *join_args)
        return FDStats(base)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: FDStats,
    ) -> FDStats:
        base = self.aggregate_def(op, init, reduce_indices, stats)
        return FDStats(base)

    def relabel(
        self, stats: FDStats, relabel_indices: tuple[Field, ...]
    ) -> FDStats:
        base = self.relabel_def(stats, relabel_indices)
        return FDStats(base)

    def reorder(
        self, stats: FDStats, reorder_indices: tuple[Field, ...]
    ) -> FDStats:
        base = self.reorder_def(stats, reorder_indices)
        return FDStats(base)


class FDStats(NumericStats):
    def estimate_non_fill_values(self) -> float:
        total = 1.0
        for size in self.dim_sizes.values():
            total *= size
        return total

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]

        return np.array(np.log2(sizes))
