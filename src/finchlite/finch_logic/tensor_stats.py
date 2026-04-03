# AI modified: 2026-04-03T00:24:22Z 7e517b16f3803378be07f55bd66f95bd09981f0c
# AI modified: 2026-04-03T00:55:25Z 38d789f35f1c9ba5c8ed00178371222826773dbe
# AI modified: 2026-04-03T01:08:06Z 38d789f35f1c9ba5c8ed00178371222826773dbe
# AI modified: 2026-04-03T01:33:01Z 38d789f35f1c9ba5c8ed00178371222826773dbe
# AI modified: 2026-04-03T02:16:03Z 6877aca3b7b141666a6b9c061af7f26a4f65c0dd
# AI modified: 2026-04-03T02:16:03Z 6877aca3b7b141666a6b9c061af7f26a4f65c0dd
# AI modified: 2026-04-03T02:34:01Z 6877aca3b7b141666a6b9c061af7f26a4f65c0dd
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