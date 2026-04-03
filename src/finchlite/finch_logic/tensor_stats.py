# AI modified: 2026-04-03T00:24:22Z 7e517b16f3803378be07f55bd66f95bd09981f0c
# AI modified: 2026-04-03T00:55:25Z 38d789f35f1c9ba5c8ed00178371222826773dbe
# AI modified: 2026-04-03T01:08:06Z 38d789f35f1c9ba5c8ed00178371222826773dbe
# AI modified: 2026-04-03T01:33:01Z 38d789f35f1c9ba5c8ed00178371222826773dbe
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, Self, TypeVar

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
    def index_order(self) -> tuple[Field, ...]:
        return self.idxs

    @property
    @abstractmethod
    def dim_sizes(self) -> Mapping[Field, float]: ...

    @property
    @abstractmethod
    def fill_value(self) -> Any: ...

    @classmethod
    @abstractmethod
    def factory(cls) -> StatsFactory[Self]: ...

    def statsfactory(self) -> StatsFactory[Self]:
        return type(self).factory()


T = TypeVar("T", bound=TensorStats)


class StatsFactory(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> T: ...

    @abstractmethod
    def copy_stats(self, stat: T) -> T: ...

    @abstractmethod
    def mapjoin(self, op: FinchOperator, *args: T) -> T: ...

    @abstractmethod
    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: T,
    ) -> T: ...

    @abstractmethod
    def issimilar(self, a: T, b: T) -> bool: ...

    @abstractmethod
    def relabel(self, stats: T, relabel_indices: tuple[Field, ...]) -> T: ...

    @abstractmethod
    def reorder(self, stats: T, reorder_indices: tuple[Field, ...]) -> T: ...