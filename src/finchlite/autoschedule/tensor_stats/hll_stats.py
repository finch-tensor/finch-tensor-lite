from __future__ import annotations

from typing import Any

import numpy as np

import hyperloglog

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field

from .numeric_stats import NumericStats
from .tensor_def import TensorDef
from .tensor_stats import BaseTensorStatsFactory


class HLLStatsFactory(BaseTensorStatsFactory["HLLStats"]):
    def __init__(self, error: float = 0.05):
        super().__init__(HLLStats)
        self.error = error

    def copy_stats(self, stat: HLLStats) -> HLLStats:
        if not isinstance(stat, HLLStats):
            raise TypeError("copy_stats expected a ExactStats instance")
        return HLLStats.from_def(stat.tensordef.copy(), stat.sketches.copy())

    def _mapjoin_join(
        self, new_def: TensorDef, op: FinchOperator, join_args: list[HLLStats]
    ) -> HLLStats: ...

    def _mapjoin_union(
        self, new_def: TensorDef, op: FinchOperator, union_args: list[HLLStats]
    ) -> HLLStats: ...

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: HLLStats,
    ) -> HLLStats: ...

    def relabel(
        self, stats: HLLStats, relabel_indices: tuple[Field, ...]
    ) -> HLLStats: ...

    def reorder(
        self, stats: HLLStats, reorder_indices: tuple[Field, ...]
    ) -> HLLStats: ...


class HLLStats(NumericStats):
    def __init__(self, tensor, fields):
        self.tensordef = TensorDef.from_tensor(tensor, fields)

        self.sketches: dict[Field, dict[int, hyperloglog.HyperLogLog]]

    @classmethod
    def from_def(
        cls,
        tensordef: TensorDef,
        sketches: dict[Field, dict[int, hyperloglog.HyperLogLog]],
    ) -> HLLStats:
        obj = object.__new__(cls)
        obj.tensordef = tensordef
        obj.sketches = sketches
        return obj

    def estimate_non_fill_values(self) -> float:
        for m in self.sketches.values():
            if m:
                return sum(len(h) for h in m.values())
        return 0.0

    def get_embedding(self) -> np.ndarray: ...
