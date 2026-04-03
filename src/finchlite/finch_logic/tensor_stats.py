from abc import ABC, abstractmethod
from typing import Any
from ..algebra import FinchOperator

from .nodes import Field



class TensorStats(ABC):
    @staticmethod
    @abstractmethod
    def copy_stats(stat: "TensorStats") -> "TensorStats":
        """
        Return a copy of a TensorStats object.
        """
        ...

    @staticmethod
    @abstractmethod
    def mapjoin(op: FinchOperator, *args: "TensorStats") -> "TensorStats":
        """
        Return a new statistic representing the tensor resulting
        from calling op on args... in an elementwise fashion
        """
        ...

    @staticmethod
    @abstractmethod
    def aggregate(
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: "TensorStats",
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

    @staticmethod
    @abstractmethod
    def relabel(
        stats: "TensorStats", relabel_indices: tuple[Field, ...]
    ) -> "TensorStats":
        """ """
        ...

    @staticmethod
    @abstractmethod
    def reorder(
        stats: "TensorStats", reorder_indices: tuple[Field, ...]
    ) -> "TensorStats":
        """ """
        ...

    @property
    @abstractmethod
    def idxs(self) -> tuple[Field, ...]: ...

    @property
    @abstractmethod
    def fill_value(self) -> Any: ...
