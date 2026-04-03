# AI modified: 2026-04-03T00:24:22Z 7e517b16f3803378be07f55bd66f95bd09981f0c
from abc import ABC, abstractmethod
from typing import Any, Self

from ..algebra import FinchOperator
from .nodes import Field


class TensorStats(ABC):
    @abstractmethod
    def __init__(self, tensor: Any, fields: tuple[Field, ...]): ...

    @classmethod
    @abstractmethod
    def copy_stats(cls, stat: Self) -> Self:
        """
        Return a copy of a TensorStats object.
        """
        ...

    @classmethod
    @abstractmethod
    def mapjoin(cls, op: FinchOperator, *args: Self) -> Self:
        """
        Return a new statistic representing the tensor resulting
        from calling op on args... in an elementwise fashion
        """
        ...

    @classmethod
    @abstractmethod
    def aggregate(
        cls,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: Self,
    ) -> Self:
        """
        Return a new statistic representing the tensor resulting
        from aggregating arg over fields with the op aggregation function
        """
        ...

    @classmethod
    @abstractmethod
    def issimilar(cls, a: Self, b: Self) -> bool:
        """
        Returns whether two statistics objects represent similarly distributed tensors,
        and only returns true if the tensors have the same dimensions and fill value
        """
        ...

    @classmethod
    @abstractmethod
    def relabel(cls, stats: Self, relabel_indices: tuple[Field, ...]) -> Self:
        """ """
        ...

    @classmethod
    @abstractmethod
    def reorder(cls, stats: Self, reorder_indices: tuple[Field, ...]) -> Self:
        """ """
        ...

    @property
    @abstractmethod
    def idxs(self) -> tuple[Field, ...]: ...

    @property
    @abstractmethod
    def fill_value(self) -> Any: ...
