from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

import matplotlib.pyplot as plt

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field, LogicExpression
from finchlite.finch_logic.tensor_stats import StatsFactory

from .exact_stats import ExactStatsFactory
from .numeric_stats import NumericStats


class ErrorLoggerStatsFactory(StatsFactory["ErrorLoggerStats"]):
    def __init__(
        self,
        estimate_factory: StatsFactory[NumericStats],
        exact_factory: ExactStatsFactory | None = None,
        error_log: ErrorLog | None = None,
    ):
        self.estimate_factory = estimate_factory
        if exact_factory is None:
            self.exact_factory = ExactStatsFactory()
        else:
            self.exact_factory = exact_factory

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> ErrorLoggerStats: ...

    def copy(self, stat: ErrorLoggerStats) -> ErrorLoggerStats: ...

    def mapjoin(self, op: FinchOperator, *stats: ErrorLoggerStats) -> ErrorLoggerStats:
        estimate = self.estimate_factory.mapjoin(op, *(stat.estimate for stat in stats))
        exact = self.exact_factory.mapjoin(op, *(stat.estimate for stat in stats))
        return ErrorLoggerStats(exact=exact, estmate=estimate)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: ErrorLoggerStats,
    ) -> ErrorLoggerStats:
        estimate = self.estimate_factory.aggregate(
            op, init, reduce_indices, *(stat.estimate for stat in stats)
        )
        exact = self.exact_factory.aggregate(
            op, init, reduce_indices, *(stat.estimate for stat in stats)
        )
        return ErrorLoggerStats(exact=exact, estmate=estimate)

    def relabel(
        self, stats: ErrorLoggerStats, relabel_indices: tuple[Field, ...]
    ) -> ErrorLoggerStats:
        estimate = self.estimate_factory.relabel(stats, relabel_indices)
        exact = self.exact_factory.relabel(stats, relabel_indices)
        return ErrorLoggerStats(exact=exact, estimate=estimate)

    def reorder(
        self, stats: ErrorLoggerStats, reorder_indices: tuple[Field, ...]
    ) -> ErrorLoggerStats:
        estimate = self.estimate_factory.reorder(stats, reorder_indices)
        exact = self.exact_factory.reorder(stats, reorder_indices)
        return ErrorLoggerStats(exact=exact, estiamte=estimate)


class ErrorLoggerStats(NumericStats):
    def __init__(
        self,
        exact: ExactStatsFactory,
        estimate: NumericStats,
    ):
        self.exact = exact
        self.estimate = estimate

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


class ErrorLog:
    """Record difference between estimated and exact stats."""

    def __init__(self):
        self.samples = {}

    def record(
        self,
        operation: type[LogicExpression],
        operator: FinchOperator,
        exact: NumericStats,
        estimate: NumericStats,
    ) -> None:
        label = f"{operation.__name__}({operator})"

        if label not in self.samples:
            self.samples[label] = []

        self.samples[label].append(
            (exact.estimate_non_fill_values(), estimate.estimate_non_fill_values())
        )


class ErrorReport:
    """Create box and whisker plots from errors."""

    def __init__(self, error_log: ErrorLog):
        self.error_log = error_log

    def plot(self, filename: str) -> None:
        labels = []
        errors = []
        for operation, samples in self.error_log.samples.items():
            operation_errors = [abs(estimate - exact) for exact, estimate in samples]
            print(f"{operation}: {operation_errors}")
            labels.append(operation)
            errors.append(operation_errors)

        figure, axis = plt.subplots()
        axis.boxplot(errors, tick_labels=labels, orientation="horizontal")
        axis.set_xlabel("Absolute NNZ error")
        axis.set_ylabel("Operation")
        axis.set_title("NNZ Estimation Error")
        figure.savefig(filename, bbox_inches="tight")
        plt.close(figure)
