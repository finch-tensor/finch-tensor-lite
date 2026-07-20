from __future__ import annotations

from copy import deepcopy
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
            dense_props,
            blocked_props,
            repeated_props,
            extruded_props,
        )

    def _mapjoin_union(self, op: FinchOperator, *union_args: FDStats) -> FDStats:
        base = super()._mapjoin_defs(op, *union_args)
        dense_props: PropertyMap = {}
        blocked_props: PropertyMap = {}
        repeated_props: PropertyMap = {}
        extruded_props: PropertyMap = {}

        for arg in union_args:
            self._union_properties(dense_props, arg.dense_props)
            self._union_properties(blocked_props, arg.blocked_props)
            self._union_properties(repeated_props, arg.repeated_props)
            self._union_properties(extruded_props, arg.extruded_props)

        return FDStats(
            base,
            dense_props,
            blocked_props,
            repeated_props,
            extruded_props,
        )

    @staticmethod
    def _union_properties(a: PropertyMap, b: PropertyMap):
        for conclusion, hypotheses in b.items():
            a.setdefault(conclusion, set()).update(hypotheses)

    def _mapjoin_join(self, op: FinchOperator, *join_args: FDStats) -> FDStats:
        base = super()._mapjoin_defs(op, *join_args)
        if not join_args:
            return FDStats(base)

        dense_props = deepcopy(join_args[0].dense_props)
        blocked_props = deepcopy(join_args[0].blocked_props)
        repeated_props = deepcopy(join_args[0].repeated_props)
        extruded_props = deepcopy(join_args[0].extruded_props)

        for arg in join_args[1:]:
            dense_props = self._join_properties(dense_props, arg.dense_props)
            blocked_props = self._join_properties(blocked_props, arg.blocked_props)
            repeated_props = self._join_properties(repeated_props, arg.repeated_props)
            extruded_props = self._join_properties(extruded_props, arg.extruded_props)

        return FDStats(
            base,
            dense_props,
            blocked_props,
            repeated_props,
            extruded_props,
        )

    @staticmethod
    def _join_properties(a: PropertyMap, b: PropertyMap) -> PropertyMap:
        out: PropertyMap = {}
        for conclusion, a_hypotheses in a.items():
            if conclusion not in b:
                continue
            hypotheses: set[frozenset[Field]] = set()
            for a_hypothesis in a_hypotheses:
                for b_hypothesis in b[conclusion]:
                    hypotheses.add(a_hypothesis | b_hypothesis)
            out[conclusion] = hypotheses
        return out

    @staticmethod
    def _drop_property_indices(
        props: PropertyMap,
        dropped_indices: frozenset[Field],
    ) -> PropertyMap:
        return {
            conclusion: {hypothesis - dropped_indices for hypothesis in hypotheses}
            for conclusion, hypotheses in props.items()
            if conclusion not in dropped_indices
        }

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: FDStats,
    ) -> FDStats:
        base = self.aggregate_def(op, init, reduce_indices, stats)
        dropped_indices = frozenset(reduce_indices)
        return FDStats(
            base,
            self._drop_property_indices(stats.dense_props, dropped_indices),
            self._drop_property_indices(stats.blocked_props, dropped_indices),
            self._drop_property_indices(stats.repeated_props, dropped_indices),
            self._drop_property_indices(stats.extruded_props, dropped_indices),
        )

    def relabel(self, stats: FDStats, relabel_indices: tuple[Field, ...]) -> FDStats:
        base = self.relabel_def(stats, relabel_indices)
        relabel_map = dict(zip(stats.index_order, relabel_indices, strict=True))
        return FDStats(
            base,
            self._relabel_property_map(stats.dense_props, relabel_map),
            self._relabel_property_map(stats.blocked_props, relabel_map),
            self._relabel_property_map(stats.repeated_props, relabel_map),
            self._relabel_property_map(stats.extruded_props, relabel_map),
        )

    def reorder(self, stats: FDStats, reorder_indices: tuple[Field, ...]) -> FDStats:
        base = self.reorder_def(stats, reorder_indices)
        return FDStats(
            base,
            stats.dense_props,
            stats.blocked_props,
            stats.repeated_props,
            stats.extruded_props,
        )

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
        self.dense_props = self._chase(deepcopy(dense_props) if dense_props else {})
        self.blocked_props = deepcopy(blocked_props) if blocked_props else {}
        self.repeated_props = deepcopy(repeated_props) if repeated_props else {}
        self.extruded_props = self._chase(
            deepcopy(extruded_props) if extruded_props else {}
        )

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

    def estimate_non_fill_values(self) -> float:
        total = 1.0
        for size in self.dim_sizes.values():
            total *= size
        return total

    def get_embedding(self) -> np.ndarray:
        sizes = [float(self.dim_sizes[field]) for field in self.index_order]

        return np.array(np.log2(sizes))
