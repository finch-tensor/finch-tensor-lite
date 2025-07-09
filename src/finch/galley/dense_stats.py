from typing import Any, Iterable
from .tensor_stats import TensorStats, TensorDef

class DenseStats(TensorStats):
    def __init__(self, tensor: Any, fields: Iterable[str]):
        return 0

    def estimate_non_fill_values(self) -> float:
        return 0

    def mapjoin(self, op, *args: "DenseStats") -> "DenseStats":
        return 0

    def aggregate(self, op, fields: Iterable[str], arg: "DenseStats") -> "DenseStats":
        return 0

    def issimilar(self, other: "TensorStats") -> bool:
        return 0