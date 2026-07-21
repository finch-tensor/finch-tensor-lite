from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field, StatsFactory
from finchlite.tensor.traits import Blocked, Dense, Repeated

from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory

PropertyMap = dict[Field, set[frozenset[Field]]]
PropertySet = set[frozenset[Field]]


class FDStatsFactory(BaseTensorStatsFactory["FDStats"], StatsFactory["FDStats"]):
    def __init__(self):
        super().__init__(FDStats)

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> FDStats:
        base = super().__call__(tensor, fields)
        props = getattr(tensor.ftype, "level_format_properties", ())
        dense_props: PropertySet = set()
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
        base = super()._mapjoin_defs(op, *union_args)
        output_fields = frozenset(base.index_order)
        dense_props = {
            dims | (output_fields - frozenset(arg.index_order))
            for arg in union_args
            for dims in arg.dense_props
        }

        unioned_props: list[PropertyMap] = []
        for attr in ("blocked_props", "repeated_props"):
            properties = tuple(
                (getattr(arg, attr), arg.index_order) for arg in union_args
            )
            conclusions = set.intersection(*(set(prop) for prop, _ in properties))
            out: PropertyMap = {}
            for conclusion in conclusions:
                out[conclusion] = {
                    frozenset().union(
                        *(
                            hypothesis | (output_fields - frozenset(fields))
                            for hypothesis, (_, fields) in zip(
                                hypotheses, properties, strict=True
                            )
                        )
                    )
                    for hypotheses in product(
                        *(prop[conclusion] for prop, _ in properties)
                    )
                }
            unioned_props.append(out)

        blocked_props, repeated_props = unioned_props
        return FDStats(base, dense_props, blocked_props, repeated_props)

    def _mapjoin_join(self, op: FinchOperator, *join_args: FDStats) -> FDStats:
        base = super()._mapjoin_defs(op, *join_args)
        output_fields = frozenset(base.index_order)
        sparse_args = [
            arg
            for arg in join_args
            if not any(
                frozenset(arg.index_order).issubset(dims) for dims in arg.dense_props
            )
        ]
        if not sparse_args:
            dense_props = {output_fields}
        elif len(sparse_args) == 1:
            sparse_arg = sparse_args[0]
            dense_props = {
                dims | (output_fields - frozenset(sparse_arg.index_order))
                for dims in sparse_arg.dense_props
            }
        else:
            dense_props = set()

        joined_props: list[PropertyMap] = []
        for attr in ("blocked_props", "repeated_props"):
            properties = tuple(
                (getattr(arg, attr), arg.index_order) for arg in join_args
            )
            conclusions = set.intersection(*(set(prop) for prop, _ in properties))
            out: PropertyMap = {}
            for conclusion in conclusions:
                out[conclusion] = {
                    frozenset().union(
                        *(
                            hypothesis | (output_fields - frozenset(fields))
                            for hypothesis, (_, fields) in zip(
                                hypotheses, properties, strict=True
                            )
                        )
                    )
                    for hypotheses in product(
                        *(prop[conclusion] for prop, _ in properties)
                    )
                }
            joined_props.append(out)

        blocked_props, repeated_props = joined_props
        return FDStats(base, dense_props, blocked_props, repeated_props)

    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: FDStats,
    ) -> FDStats:
        base = self.aggregate_def(op, init, reduce_indices, stats)
        dropped_indices = frozenset(reduce_indices)
        projected_props = [
            {
                conclusion: {hypothesis - dropped_indices for hypothesis in hypotheses}
                for conclusion, hypotheses in props.items()
                if conclusion not in dropped_indices
            }
            for props in (stats.blocked_props, stats.repeated_props)
        ]
        blocked_props, repeated_props = projected_props
        return FDStats(
            base,
            {
                dims - dropped_indices
                for dims in stats.dense_props
                if dims - dropped_indices
            },
            blocked_props,
            repeated_props,
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
        dense_props: PropertySet | None = None,
        blocked_props: PropertyMap | None = None,
        repeated_props: PropertyMap | None = None,
    ):
        super().__init__(base)
        self.dense_props = deepcopy(dense_props) if dense_props else set()
        self.blocked_props = deepcopy(blocked_props) if blocked_props else {}
        self.repeated_props = deepcopy(repeated_props) if repeated_props else {}
