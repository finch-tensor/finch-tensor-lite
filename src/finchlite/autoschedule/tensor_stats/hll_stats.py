from __future__ import annotations

from typing import Any

import numpy as np

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import (
    Field,
    LogicExpression,
)

from .numeric_stats import NumericStats
from .tensor_def import TensorDef
from .tensor_stats import BaseTensorStatsFactory


class HLLStatsFactory(BaseTensorStatsFactory["HLLStats"]):
    def __init__(self):
        super().__init__(HLLStats)

    def copy_stats(self, stat: HLLStats) -> HLLStats: ...

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
    def __init__(self, tensor, fields): ...

    @classmethod
    def from_def(
        cls, tensordef: TensorDef, expr: LogicExpression | None
    ) -> HLLStats: ...

    def estimate_non_fill_values(self) -> float: ...

    def get_embedding(self) -> np.ndarray: ...
