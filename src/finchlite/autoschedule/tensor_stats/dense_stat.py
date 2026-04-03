from typing import Any, Self

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field

from .numeric_stats import NumericStats
from .tensor_def import TensorDef
from .tensor_stats import DenseStats


class DenseStats(NumericStats):
    @classmethod
    def from_def(cls, d: TensorDef) -> Self:
        ds = object.__new__(cls)
        ds.tensordef = d.copy()
        return ds

    @staticmethod
    def copy_stats(stat: DenseStats) -> DenseStats:
        """
        Deep copy of a DenseStats object.
        """
        if not isinstance(stat, DenseStats):
            raise TypeError("copy_stats expected a DenseStats instance")
        return DenseStats.from_def(stat.tensordef.copy())

    def estimate_non_fill_values(self) -> float:
        total = 1.0
        for size in self.dim_sizes.values():
            total *= size
        return total

    @staticmethod
    def mapjoin(op: FinchOperator, *args: DenseStats) -> DenseStats:
        axes_set = [set(s.index_order) for s in args]
        same_axes = all(axes_set[0] == axes for axes in axes_set)

        def_args = [stat.tensordef for stat in args]
        new_def = TensorDef.mapjoin(op, *def_args)

        if not same_axes:
            # Additional check needed for the case when dimesions do not match
            new_def.fill_value = 0.0

        return DenseStats.from_def(new_def)

    @staticmethod
    def aggregate(
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: "DenseStats",
    ) -> "DenseStats":
        d = stats.tensordef
        new_def = TensorDef.aggregate(op, init, reduce_indices, d)
        return DenseStats.from_def(new_def)

    @staticmethod
    def issimilar(a: DenseStats, b: DenseStats) -> bool:
        return (
            isinstance(a, DenseStats)
            and isinstance(b, DenseStats)
            and a.dim_sizes == b.dim_sizes
            and a.fill_value == b.fill_value
        )

    @staticmethod
    def relabel(
        stats: "DenseStats", relabel_indices: tuple[Field, ...]
    ) -> "DenseStats":
        d = stats.tensordef
        new_def = TensorDef.relabel(d, relabel_indices)
        return DenseStats.from_def(new_def)

    @staticmethod
    def reorder(
        stats: "DenseStats", reorder_indices: tuple[Field, ...]
    ) -> "DenseStats":

        d = stats.tensordef
        new_def = TensorDef.reorder(d, reorder_indices)
        return DenseStats.from_def(new_def)
