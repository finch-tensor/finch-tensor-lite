from __future__ import annotations

from copy import deepcopy
from typing import Any

from finch.algebra import FinchOperator
from finch.finch_logic import Field, StatsFactory
from finch.tensor.traits import Dense

from .tensor_stats import BaseTensorStats, BaseTensorStatsFactory

PropertySet = set[frozenset[Field]]


class FDStatsFactory(BaseTensorStatsFactory["FDStats"], StatsFactory["FDStats"]):
    def __init__(self):
        super().__init__(FDStats)

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> FDStats:
        base = super().__call__(tensor, fields)
        props = getattr(tensor.ftype, "level_format_properties", ())
        dense_props: PropertySet = set()

        for prop in props:
            match prop:
                case Dense():
                    dense_props.add(
                        frozenset(base.index_order[dim] for dim in prop.dims)
                    )

        return FDStats(base, dense_props)

    def _mapjoin_union(self, op: FinchOperator, *union_args: FDStats) -> FDStats:
        base = super()._mapjoin_defs(op, *union_args)
        output_fields = frozenset(base.index_order)
        dense_props = {
            dims | (output_fields - frozenset(arg.index_order))
            for arg in union_args
            for dims in arg.dense_props
        }

        return FDStats(base, dense_props)

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
        match sparse_args:
            case []:
                dense_props = {output_fields}
            case [sparse_arg]:
                broadcast_fields = output_fields - frozenset(sparse_arg.index_order)
                dense_props = {
                    dims | broadcast_fields for dims in sparse_arg.dense_props
                }
            case _:
                dense_props = set()

        return FDStats(base, dense_props)

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
        )

    def reorder(self, stats: FDStats, reorder_indices: tuple[Field, ...]) -> FDStats:
        base = self.reorder_def(stats, reorder_indices)
        return FDStats(base, stats.dense_props)


class FDStats(BaseTensorStats):
    def __init__(
        self,
        base: BaseTensorStats,
        dense_props: PropertySet | None = None,
    ):
        super().__init__(base)
        self.dense_props = deepcopy(dense_props) if dense_props else set()
