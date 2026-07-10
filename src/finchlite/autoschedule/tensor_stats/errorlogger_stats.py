from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field
from finchlite.finch_logic.tensor_stats import StatsFactory

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats


class ErrorLoggerStatsFactory(StatsFactory["ErrorLoggerStats"]):
    def __init__(
        self,
        stats_factory: StatsFactory[NumericStats],
    ):
        self.inner_factory = stats_factory

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> ErrorLoggerStats: ...

    def copy(self, stat: ErrorLoggerStats) -> ErrorLoggerStats: ...

    def mapjoin(
        self, op: FinchOperator, *args: ErrorLoggerStats
    ) -> ErrorLoggerStats: ...

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: ErrorLoggerStats,
    ) -> ErrorLoggerStats: ...

    def relabel(
        self, stats: ErrorLoggerStats, relabel_indices: tuple[Field, ...]
    ) -> ErrorLoggerStats: ...

    def reorder(
        self, stats: ErrorLoggerStats, reorder_indices: tuple[Field, ...]
    ) -> ErrorLoggerStats: ...


class ErrorLoggerStats(NumericStats):
    def __init__(
        self,
        tensordef: BaseTensorStats,
        stats_factory: StatsFactory[NumericStats],
    ):
        self.index_order = tensordef.index_order
        self.dim_sizes = tensordef.dim_sizes
        self.fill_value = tensordef.fill_value
        self.stats_factory = stats_factory

    @classmethod
    def from_tensor(
        cls,
        tensor: Any,
        fields: tuple[Field, ...],
        blocks_per_dim: Mapping[Field, int],
        stats_factory: StatsFactory[NumericStats],
    ) -> ErrorLoggerStats: ...

    def estimate_non_fill_values(self): ...

    def get_embedding(self) -> np.ndarray: ...
