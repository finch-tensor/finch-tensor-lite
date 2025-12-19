from collections.abc import Callable, Iterable
from typing import Any, Self

from .tensor_def import TensorDef
from .tensor_stats import TensorStats


class DenseStats(TensorStats):
    @classmethod
    def from_def(cls, d: TensorDef) -> Self:
        ds = object.__new__(cls)
        ds.tensordef = d.copy()
        return ds

    @staticmethod
    def copy_stats(stat: TensorStats) -> TensorStats:
        """
        Deep copy of a DenseStats object.
        """
        if not isinstance(stat, DenseStats):
            raise TypeError("copy_stats expected a DenseStats instance")
        return DenseStats.from_def(stat.tensordef.copy())

    @staticmethod
    def relabel_fields(stats, field_mapping):
        stats = DenseStats.copy_stats(stats)
        stats.tensordef = stats.tensordef.relabel_fields(field_mapping)
        return stats

    def estimate_non_fill_values(self) -> float:
        total = 1.0
        for size in self.dim_sizes.values():
            total *= size
        return total

    @staticmethod
    def mapjoin(op: Callable, *args: TensorStats) -> TensorStats:
        new_axes = set().union(*(s.index_set for s in args))

        new_dims = {
            ax: next(s.get_dim_size(ax) for s in args if ax in s.index_set)
            for ax in new_axes
        }

        axes_sets = [set(s.index_set) for s in args]

        if len(args) == 1:
            new_fill = args[0].fill_value
        else:
            same_axes = all(axes_sets[0] == axes for axes in axes_sets)
            new_fill = op(*(s.fill_value for s in args)) if same_axes else 0.0

        new_def = TensorDef(new_axes, new_dims, new_fill)
        return DenseStats.from_def(new_def)

    @staticmethod
    def aggregate(
        op: Callable[..., Any],
        init: Any | None,
        reduce_indices: Iterable[str],
        stats: "TensorStats",
    ) -> "DenseStats":
        new_axes = set(stats.index_set) - set(reduce_indices)
        new_dims = {m: stats.get_dim_size(m) for m in new_axes}
        new_fill = stats.fill_value

        new_def = TensorDef(new_axes, new_dims, new_fill)
        return DenseStats.from_def(new_def)

    @staticmethod
    def issimilar(a: TensorStats, b: TensorStats) -> bool:
        return (
            isinstance(a, DenseStats)
            and isinstance(b, DenseStats)
            and a.dim_sizes == b.dim_sizes
            and a.fill_value == b.fill_value
        )
