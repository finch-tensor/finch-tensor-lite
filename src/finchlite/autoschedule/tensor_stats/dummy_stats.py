from __future__ import annotations

from typing import Any, Self

from finchlite.finch_logic import Field

from ..algebra import FinchOperator
from .tensor_def import TensorDef
from .tensor_stats import BaseTensorStats


class DummyStats(BaseTensorStats):
    @classmethod
    def from_def(cls, d: TensorDef) -> Self:
        ds = object.__new__(cls)
        ds.tensordef = d.copy()
        return ds

    @classmethod
    def copy_stats(cls, stat: DummyStats) -> DummyStats:
        if not isinstance(stat, DummyStats):
            raise TypeError("copy_stats expected a DummyStats instance")
        return cls.from_def(stat.tensordef.copy())

    @classmethod
    def mapjoin(cls, op: FinchOperator, *args: DummyStats) -> DummyStats:
        axes_set = [set(s.index_order) for s in args]
        same_axes = all(axes_set[0] == axes for axes in axes_set)

        def_args = [stat.tensordef for stat in args]
        new_def = TensorDef.mapjoin(op, *def_args)

        if not same_axes:
            new_def.fill_value = 0.0

        return cls.from_def(new_def)

    @classmethod
    def aggregate(
        cls,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: DummyStats,
    ) -> DummyStats:
        d = stats.tensordef
        new_def = TensorDef.aggregate(op, init, reduce_indices, d)
        return cls.from_def(new_def)

    @classmethod
    def issimilar(cls, a: DummyStats, b: DummyStats) -> bool:
        return (
            isinstance(a, DummyStats)
            and isinstance(b, DummyStats)
            and a.dim_sizes == b.dim_sizes
            and a.tensordef.index_order == b.tensordef.index_order
        )

    @classmethod
    def relabel(
        cls, stats: DummyStats, relabel_indices: tuple[Field, ...]
    ) -> DummyStats:
        d = stats.tensordef
        new_def = TensorDef.relabel(d, relabel_indices)
        return cls.from_def(new_def)

    @classmethod
    def reorder(
        cls, stats: DummyStats, reorder_indices: tuple[Field, ...]
    ) -> DummyStats:

        d = stats.tensordef
        new_def = TensorDef.reorder(d, reorder_indices)
        return cls.from_def(new_def)
