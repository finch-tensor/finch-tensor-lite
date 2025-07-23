from collections import OrderedDict
from typing import Any, Iterable, Mapping, Set, Callable, Type
from abc import ABC, abstractmethod

class TensorStats(ABC):

    def __init__(self, tensor:Any, fields: Iterable[str]):
        self = from_tensor(tensor, fields)

    @abstractmethod
    def from_tensor(self, tensor: Any, fields: Iterable[str]) -> "TensorStats":
        ...

    @staticmethod
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
        Return a new statistic representing the tensor resulting from calling op on args... in an elementwise fashion
        """
        ...

    @staticmethod
    @abstractmethod
    def aggregate(op: Callable,
                  fields: Iterable[str],
                  arg: "TensorStats"
                 ) -> "TensorStats":
        """
        Return a new statistic representing the tensor resulting from aggregating arg over fields with the op aggregation function
        """
        ...

    @staticmethod
    @abstractmethod
    def issimilar(
        a: "TensorStats",
        b: "TensorStats"
    ) -> bool:
        """
        Returns whether two statistics objects represent similarly distributed tensors,
        and only returns true if the tensors have the same dimensions and fill value
        """
        ...
