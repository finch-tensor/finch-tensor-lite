from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...finch_logic import Field
from .dc_stats import DCStats
from .tensor_def import TensorDef
from .tensor_stats import TensorStats


class DatabaseStats(TensorStats):
    def __init__(self, tensor: Any, fields: tuple[Field, ...]):
        self.tensordef = TensorDef.from_tensor(tensor, fields)
        val = tensor

        if hasattr(val, "tns"):
            val = val.tns.val
        if hasattr(val, "val") and not hasattr(val, "to_numpy"):
            val = val.val

        dc = DCStats(tensor, fields)
        for d in dc.dcs:
            if d.from_indices == frozenset() and d.to_indices == frozenset(fields):
                self.nnz = d.value
                break

        self.V: dict[Field, float] = {}
        for idx in fields:
            for d in dc.dcs:
                if d.from_indices == frozenset() and d.to_indices == frozenset({idx}):
                    self.V[idx] = d.value

    @staticmethod
    def mapjoin(op: Callable[..., Any], *all_stats: TensorStats) -> TensorStats: ...

    @staticmethod
    def aggregate(
        op: Callable[..., Any],
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: TensorStats,
    ) -> TensorStats: ...

    @staticmethod
    def issimilar(a: TensorStats, b: TensorStats) -> bool:
        if not (isinstance(a, DatabaseStats) and isinstance(b, DatabaseStats)):
            return False
        return a.fill_value == b.fill_value and a.dim_sizes == b.dim_sizes

    @staticmethod
    def relabel(
        stats: TensorStats, relabel_indices: tuple[Field, ...]
    ) -> DatabaseStats:
        d = stats.tensordef
        new_def = TensorDef.relabel(d, relabel_indices)
        return DatabaseStats.from_def(new_def, stats.estimate_non_fill_values())

    @staticmethod
    def reorder(
        stats: TensorStats, reorder_indices: tuple[Field, ...]
    ) -> DatabaseStats:
        d = stats.tensordef
        new_def = TensorDef.reorder(d, reorder_indices)
        return DatabaseStats.from_def(new_def, stats.estimate_non_fill_values())

    @staticmethod
    def copy_stats(stat: TensorStats) -> DatabaseStats:
        if not isinstance(stat, DatabaseStats):
            raise TypeError("copy_stats expected a DatabaseStats instance")
        return DatabaseStats.from_def(stat.tensordef.copy(), stat.nnz)

    def estimate_non_fill_values(self) -> float:
        return self.nnz
