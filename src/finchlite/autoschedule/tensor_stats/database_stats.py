from typing import Any

from ...finch_logic import Field
from .tensor_def import TensorDef
from .tensor_stats import TensorStats


class DatabaseStats(TensorStats):
    def __init__(self, tensor: Any, fields: tuple[Field, ...]):
        self.tensordef = TensorDef.from_tensor(tensor, fields)

    @staticmethod
    def aggregate(): ...

    @staticmethod
    def mapjoin(): ...

    @staticmethod
    def issimilar(a: TensorStats, b: TensorStats) -> bool:
        if not (isinstance(a, "DatabaseStats") and isinstance(b, "DatabaseStats")):
            return False
        return a.fill_value == b.fill_value and a.dim_sizes == b.dim_sizes

    @staticmethod
    def relabel(
        stats: TensorStats, relabel_indices: tuple[Field, ...]
    ) -> "DatabaseStats":
        d = stats.tensordef
        new_def = TensorDef.relabel(d, relabel_indices)
        return DatabaseStats.from_def(new_def, stats.estimate_non_fill_values())

    @staticmethod
    def reorder(
        stats: TensorStats, reorder_indices: tuple[Field, ...]
    ) -> "DatabaseStats":
        d = stats.tensordef
        new_def = TensorDef.reorder(d, reorder_indices)
        return DatabaseStats.from_def(new_def, stats.estimate_non_fill_values())

    @staticmethod
    def copy_stats(stat: TensorStats) -> "DatabaseStats":
        if not isinstance(stat, DatabaseStats):
            raise TypeError("copy_stats expected a DatabaseStats instance")
        return DatabaseStats.from_def(stat.tensordef.copy(), stat.nnz)

    def estimate_non_fill_values(self) -> float:
        return self.nnz
