from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .tensor_def import TensorDef


class TensorStats(ABC):
    tensordef: TensorDef

    def __init__(self, tensor: Any, fields: Iterable[str]):
        self.tensordef = TensorDef.from_tensor(tensor, fields)
        self.from_tensor(tensor, fields)

    @classmethod
    @abstractmethod
    def from_tensor(self, tensor: Any, fields: Iterable[str]) -> None:
        """
        Populate this instance’s state from (tensor, fields).
        """
        ...

    @abstractmethod
    def estimate_non_fill_values(arg: "TensorStats") -> float:
        """
        Return an estimate on the number of non-fill values.
        """
        ...

    @staticmethod
    @abstractmethod
    def mapjoin(op: Callable, *args: "TensorStats") -> "TensorStats":
        """
        Return a new statistic representing the tensor resulting
        from calling op on args... in an elementwise fashion
        """
        ...

    @staticmethod
    @abstractmethod
    def aggregate(
        op: Callable, fields: Iterable[str], arg: "TensorStats"
    ) -> "TensorStats":
        """
        Return a new statistic representing the tensor resulting
        from aggregating arg over fields with the op aggregation function
        """
        ...

    @staticmethod
    @abstractmethod
    def issimilar(a: "TensorStats", b: "TensorStats") -> bool:
        """
        Returns whether two statistics objects represent similarly distributed tensors,
        and only returns true if the tensors have the same dimensions and fill value
        """
        ...

    @property
    def dim_sizes(self) -> Mapping[str, float]:
        return self.tensordef.dim_sizes

    @dim_sizes.setter
    def dim_sizes(self, value: Mapping[str, float]):
        self.tensordef.dim_sizes = value

    def get_dim_size(self, idx: str) -> float:
        return self.tensordef.get_dim_size(idx)

    @property
    def index_set(self) -> set[str]:
        return self.tensordef.index_set

    @index_set.setter
    def index_set(self, value: set[str]):
        self.tensordef.index_set = value

    @property
    def fill_value(self) -> Any:
        return self.tensordef.fill_value

    @fill_value.setter
    def fill_value(self, value: Any):
        self.tensordef.fill_value = value
