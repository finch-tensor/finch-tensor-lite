from __future__ import annotations

from typing import Any

import numpy as np

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field, StatsFactory
from finchlite.tensor.traits import Blocked, Dense, Extruded, FormatProperty, Repeated

from .numeric_stats import NumericStats
from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory

PropertyMap = dict[Field, set[frozenset[Field]]]


class FDStatsFactory(BaseTensorStatsFactory["FDStats"], StatsFactory["FDStats"]):
    def __init__(self):
        super().__init__(FDStats)

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> FDStats:
        base = super().__call__(tensor, fields)
        props = getattr(tensor.ftype, "level_format_properties", ())
        dense_props: PropertyMap = {}
        blocked_props: PropertyMap = {}
        repeated_props: PropertyMap = {}
        extruded_props: PropertyMap = {}

        for prop in props:
            match prop:
                case Dense():
                    self._add_property(base, dense_props, prop)
                case Blocked():
                    self._add_property(base, blocked_props, prop)
                case Repeated():
                    self._add_property(base, repeated_props, prop)
                case Extruded():
                    self._add_property(base, extruded_props, prop)

        return FDStats(
            base,
            self._chase(dense_props),
            blocked_props,
            repeated_props,
            self._chase(extruded_props),
        )

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
        relabel_map = dict(zip(stats.index_order, relabel_indices, strict=True))
        return FDStats(
            base,
            self._relabel_property_map(stats.dense_props, relabel_map),
            self._relabel_property_map(stats.blocked_props, relabel_map),
            self._relabel_property_map(stats.repeated_props, relabel_map),
            self._relabel_property_map(stats.extruded_props, relabel_map),
        )

    def reorder(
        self, stats: FDStats, reorder_indices: tuple[Field, ...]
    ) -> FDStats:
        base = self.reorder_def(stats, reorder_indices)
        return FDStats(base)

    def _chase(
        self,
        props: PropertyMap,
    ) -> PropertyMap:
        changed = True
        while changed:
            changed = False
            for conclusion, hypotheses in tuple(props.items()):
                for hypothesis in tuple(hypotheses):
                    for head in hypothesis:
                        tail = hypothesis - {head}
                        for replacement in tuple(props.get(head, ())):
                            chased = tail | replacement
                            if conclusion in chased:
                                continue
                            if chased not in hypotheses:
                                hypotheses.add(chased)
                                changed = True
        return props

    def _fields_for_dims(
        self,
        base: BaseTensorStats,
        dims: tuple[int, ...],
    ) -> tuple[Field, ...]:
        try:
            return tuple(base.index_order[dim] for dim in dims)
        except IndexError as exc:
            raise ValueError(
                f"Format property dimensions {dims} do not fit tensor "
                f"with {len(base.index_order)} dimensions."
            ) from exc

    def _add_property(
        self,
        base: BaseTensorStats,
        props_by_conclusion: PropertyMap,
        prop: FormatProperty,
    ):
        hypotheses = frozenset(self._fields_for_dims(base, prop.hypothesis_dims))
        for conclusion in self._fields_for_dims(base, prop.conclusion_dims):
            props_by_conclusion.setdefault(conclusion, set()).add(hypotheses)

    def _relabel_property_map(
        self,
        props: PropertyMap,
        relabel_map: dict[Field, Field],
    ) -> PropertyMap:
        return {
            relabel_map[conclusion]: {
                frozenset(relabel_map[field] for field in hypothesis)
                for hypothesis in hypotheses
            }
            for conclusion, hypotheses in props.items()
        }


class FDStats(NumericStats):
    def __init__(
        self,
        base: BaseTensorStats,
        dense_props: PropertyMap | None = None,
        blocked_props: PropertyMap | None = None,
        repeated_props: PropertyMap | None = None,
        extruded_props: PropertyMap | None = None,
    ):
        super().__init__(base)
        self.dense_props = dense_props or {}
        self.blocked_props = blocked_props or {}
        self.repeated_props = repeated_props or {}
        self.extruded_props = extruded_props or {}

    def estimate_non_fill_values(self) -> float:
        total = 1.0
        for size in self.dim_sizes.values():
            total *= size
        return total

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]

        return np.array(np.log2(sizes))
