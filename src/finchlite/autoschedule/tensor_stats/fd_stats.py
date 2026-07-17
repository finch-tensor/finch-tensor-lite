from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field, StatsFactory
from finchlite.tensor.traits import Blocked, Dense, Extruded, FormatProperty, Repeated

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory


class FDStatsFactory(BaseTensorStatsFactory["FDStats"], StatsFactory["FDStats"]):
    def __init__(self):
        super().__init__(FDStats)

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> FDStats:
        base = super().__call__(tensor, fields)
        props = getattr(tensor.ftype, "level_format_properties", ())
        return FDStats(base, props)

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
        return FDStats(base, stats.format_properties)

    def reorder(
        self, stats: FDStats, reorder_indices: tuple[Field, ...]
    ) -> FDStats:
        base = self.reorder_def(stats, reorder_indices)
        return FDStats(base)


class FDStats(NumericStats):
    def __init__(
        self,
        base: BaseTensorStats,
        props: Iterable[FormatProperty] = (),
    ):
        super().__init__(base)
        self.format_properties = tuple(props)
        self.dense_props: dict[Field, list[tuple[Field, ...]]] = {}
        self.blocked_props: dict[Field, list[tuple[Field, ...]]] = {}
        self.repeated_props: dict[Field, list[tuple[Field, ...]]] = {}
        self.extruded_props: dict[Field, list[tuple[Field, ...]]] = {}

        for prop in self.format_properties:
            match prop:
                case Dense():
                    self._add_property(self.dense_props, prop)
                case Blocked():
                    self._add_property(self.blocked_props, prop)
                case Repeated():
                    self._add_property(self.repeated_props, prop)
                case Extruded():
                    self._add_property(self.extruded_props, prop)

    def _fields_for_dims(self, dims: tuple[int, ...]) -> tuple[Field, ...]:
        try:
            return tuple(self.index_order[dim] for dim in dims)
        except IndexError as exc:
            raise ValueError(
                f"Format property dimensions {dims} do not fit tensor "
                f"with {len(self.index_order)} dimensions."
            ) from exc

    def _add_property(
        self,
        props_by_conclusion: dict[Field, list[tuple[Field, ...]]],
        prop: FormatProperty,
    ):
        hypotheses = self._fields_for_dims(prop.hypothesis_dims)
        for conclusion in self._fields_for_dims(prop.conclusion_dims):
            props_by_conclusion.setdefault(conclusion, []).append(hypotheses)

    def estimate_non_fill_values(self) -> float:
        total = 1.0
        for size in self.dim_sizes.values():
            total *= size
        return total

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]

        return np.array(np.log2(sizes))
