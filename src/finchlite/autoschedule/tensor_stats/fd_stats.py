from __future__ import annotations

from copy import deepcopy
from typing import Any

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field, StatsFactory
from finchlite.tensor.traits import Blocked, Dense, Repeated

from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory

PropertyMap = dict[Field, set[frozenset[Field]]]
DensePropertySet = set[frozenset[Field]]


class FDStatsFactory(BaseTensorStatsFactory["FDStats"], StatsFactory["FDStats"]):
    def __init__(self):
        super().__init__(FDStats)

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> FDStats:
        base = super().__call__(tensor, fields)
        props = getattr(tensor.ftype, "level_format_properties", ())
        dense_props: DensePropertySet = set()
        blocked_props: PropertyMap = {}
        repeated_props: PropertyMap = {}

        for prop in props:
            match prop:
                case Dense():
                    dense_props.add(frozenset(self._fields_for_dims(base, prop.dims)))
                case Blocked():
                    self._add_property(base, blocked_props, prop)
                case Repeated():
                    self._add_property(base, repeated_props, prop)

        return FDStats(
            base,
            dense_props,
            blocked_props,
            repeated_props,
        )

    def _mapjoin_union(self, op: FinchOperator, *union_args: FDStats) -> FDStats:
        result = union_args[0]
        for arg in union_args[1:]:
            result = self._mapjoin_union_binary(op, result, arg)
        return result

    def _mapjoin_union_binary(
        self, op: FinchOperator, a: FDStats, b: FDStats
    ) -> FDStats:
        base = super()._mapjoin_defs(op, a, b)
        dense_props = self._union_dense_properties(base, a, b)
        blocked_props = deepcopy(a.blocked_props)
        self._union_properties(blocked_props, b.blocked_props)
        repeated_props = self._join_properties(
            a.repeated_props, a.index_order, b.repeated_props, b.index_order
        )
        return FDStats(base, dense_props, blocked_props, repeated_props)

    @staticmethod
    def _union_dense_properties(
        base: BaseTensorStats, a: FDStats, b: FDStats
    ) -> DensePropertySet:
        output_fields = frozenset(base.index_order)
        a_broadcast = output_fields - frozenset(a.index_order)
        b_broadcast = output_fields - frozenset(b.index_order)
        return {
            *(dims | a_broadcast for dims in a.dense_props),
            *(dims | b_broadcast for dims in b.dense_props),
        }

    @staticmethod
    def _union_properties(a: PropertyMap, b: PropertyMap):
        for conclusion, hypotheses in b.items():
            a.setdefault(conclusion, set()).update(hypotheses)

    def _mapjoin_join(self, op: FinchOperator, *join_args: FDStats) -> FDStats:
        result = join_args[0]
        for arg in join_args[1:]:
            result = self._mapjoin_join_binary(op, result, arg)
        return result

    def _mapjoin_join_binary(
        self, op: FinchOperator, a: FDStats, b: FDStats
    ) -> FDStats:
        base = super()._mapjoin_defs(op, a, b)
        dense_props = self._join_dense_properties(base, a, b)
        blocked_props = self._join_properties(
            a.blocked_props, a.index_order, b.blocked_props, b.index_order
        )
        repeated_props = self._join_properties(
            a.repeated_props, a.index_order, b.repeated_props, b.index_order
        )
        return FDStats(base, dense_props, blocked_props, repeated_props)

    @staticmethod
    def _join_dense_properties(
        base: BaseTensorStats, a: FDStats, b: FDStats
    ) -> DensePropertySet:
        a_fields = frozenset(a.index_order)
        b_fields = frozenset(b.index_order)
        if not any(a_fields.issubset(dims) for dims in a.dense_props):
            return set()
        if not any(b_fields.issubset(dims) for dims in b.dense_props):
            return set()
        return {frozenset(base.index_order)}

    @staticmethod
    def _join_properties(a: PropertyMap, a_idxs, b: PropertyMap, b_idxs) -> PropertyMap:
        a_idxs = frozenset(a_idxs)
        b_idxs = frozenset(b_idxs)
        out: PropertyMap = {}
        for conclusion, a_hypotheses in a.items():
            if conclusion not in b:
                continue
            hypotheses: set[frozenset[Field]] = set()
            for a_hypothesis in a_hypotheses:
                for b_hypothesis in b[conclusion]:
                    hypotheses.add(
                        a_hypothesis.union(b_idxs - a_idxs)
                        & b_hypothesis.union(a_idxs - b_idxs)
                    )
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
            {
                dims - dropped_indices
                for dims in stats.dense_props
                if dims - dropped_indices
            },
            self._drop_property_indices(stats.blocked_props, dropped_indices),
            self._drop_property_indices(stats.repeated_props, dropped_indices),
        )

    def relabel(self, stats: FDStats, relabel_indices: tuple[Field, ...]) -> FDStats:
        base = self.relabel_def(stats, relabel_indices)
        relabel_map = dict(zip(stats.index_order, relabel_indices, strict=True))
        return FDStats(
            base,
            {
                frozenset(relabel_map[field] for field in dims)
                for dims in stats.dense_props
            },
            self._relabel_property_map(stats.blocked_props, relabel_map),
            self._relabel_property_map(stats.repeated_props, relabel_map),
        )

    def reorder(self, stats: FDStats, reorder_indices: tuple[Field, ...]) -> FDStats:
        base = self.reorder_def(stats, reorder_indices)
        return FDStats(
            base,
            stats.dense_props,
            stats.blocked_props,
            stats.repeated_props,
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
        prop: Blocked | Repeated,
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


class FDStats(BaseTensorStats):
    def __init__(
        self,
        base: BaseTensorStats,
        dense_props: DensePropertySet | None = None,
        blocked_props: PropertyMap | None = None,
        repeated_props: PropertyMap | None = None,
    ):
        super().__init__(base)
        self.dense_props = deepcopy(dense_props) if dense_props else set()
        self.blocked_props = deepcopy(blocked_props) if blocked_props else {}
        self.repeated_props = deepcopy(repeated_props) if repeated_props else {}
