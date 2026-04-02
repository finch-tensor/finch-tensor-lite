from collections.abc import Callable
from typing import Any, Self

from finchlite.finch_logic import Field

from .tensor_def import TensorDef
from .tensor_stats import TensorStats


class DummyStats(TensorStats):
    @classmethod
    def from_def(cls, d: TensorDef) -> Self:
        ds = object.__new__(cls)
        ds.tensordef = d.copy()
        return ds

    @staticmethod
    def copy_stats(stat: TensorStats) -> TensorStats:
        if not isinstance(stat, DummyStats):
            raise TypeError("copy_stats expected a DummyStats instance")
        return DummyStats.from_def(stat.tensordef.copy())

    @staticmethod
    def mapjoin(op: Callable, *args: TensorStats) -> TensorStats:
        axes_set = [set(s.index_order) for s in args]
        same_axes = all(axes_set[0] == axes for axes in axes_set)

        def_args = [stat.tensordef for stat in args]
        new_def = TensorDef.mapjoin(op, *def_args)

        if not same_axes:
            new_def.fill_value = 0.0

        return DummyStats.from_def(new_def)

    @staticmethod
    def aggregate(
        op: Callable[..., Any],
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: "TensorStats",
    ) -> "DummyStats":
        d = stats.tensordef
        new_def = TensorDef.aggregate(op, init, reduce_indices, d)
        return DummyStats.from_def(new_def)

    @staticmethod
    def issimilar(a: TensorStats, b: TensorStats) -> bool:
        return (
            isinstance(a, DummyStats)
            and isinstance(b, DummyStats)
            and a.dim_sizes == b.dim_sizes
            and a.tensordef.index_order == b.tensordef.index_order
        )

    @staticmethod
    def relabel(
        stats: "TensorStats", relabel_indices: tuple[Field, ...]
    ) -> "DummyStats":
        d = stats.tensordef
        new_def = TensorDef.relabel(d, relabel_indices)
        return DummyStats.from_def(new_def)

    @staticmethod
    def reorder(
        stats: "TensorStats", reorder_indices: tuple[Field, ...]
    ) -> "DummyStats":

        d = stats.tensordef
        new_def = TensorDef.reorder(d, reorder_indices)
        return DummyStats.from_def(new_def)
