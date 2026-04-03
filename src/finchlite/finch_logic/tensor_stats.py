from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, TypeVar

from ..algebra import FinchOperator
from .nodes import Field


class TensorStats(ABC):
    @abstractmethod
    def __init__(self, tensor: Any, fields: tuple[Field, ...]): ...

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
