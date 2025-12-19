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
        same_axes = all(axes_sets[0] == axes for axes in axes_sets)
        new_fill = op(*[s.fill_value for s in args]) if same_axes else 0.0

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

    # Defining relabel method which just relabels the indices of the tensorstats object and the stats remain same
    @staticmethod
    def relabel(stats: "TensorStats", relabel_indices: Iterable[str]) -> "DenseStats":
        new_axes = set(relabel_indices)
        new_dims = {m: stats.get_dim_size(m) for m in new_axes}
        new_fill = stats.fill_value
        new_def = TensorDef(new_axes, new_dims, new_fill)
        return DenseStats.from_def(new_def)

    # Defining reorder method which reorder the indices and returns stats for the new tensor
    @staticmethod
    def reorder(stats: "TensorStats", reorder_indices: Iterable[str]) -> "DenseStats":
        new_axes = set(reorder_indices)
        for old_idx in stats.index_set:
            if old_idx not in new_axes:
                if stats.get_dim_size(old_idx) != 1:
                    raise ValueError(
                        f"Trying to drop dimension '{old_idx}' of size {stats.get_dim_size(old_idx)}. Only size 1 dimensions can be dropped."
                    )

        new_dims = {}
        for idx in reorder_indices:
            if idx in stats.index_set:
                new_dims[idx] = stats.get_dim_size(idx)
            else:
                new_dims[idx] = 1

        new_def = TensorDef(reorder_indices, new_dims, stats.fill_value)
        return DenseStats.from_def(new_def)
