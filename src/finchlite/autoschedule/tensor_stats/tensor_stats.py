from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Generic, TypeVar

from finchlite.finch_logic import Field, StatsFactory, TensorStats

from ...algebra import FinchOperator
from .tensor_def import TensorDef

TS = TypeVar("TS", bound="BaseTensorStats")


class BaseTensorStatsFactory(StatsFactory[TS], Generic[TS]):
    def __init__(self, stats_cls: type[TS]):
        self.stats_cls = stats_cls

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> TS:
        return self.stats_cls(tensor, fields)

    @abstractmethod
    def copy_stats(self, stat: TS) -> TS: ...

    @abstractmethod
    def mapjoin(self, op: FinchOperator, *args: TS) -> TS: ...

    @abstractmethod
    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: TS,
    ) -> TS: ...

    @abstractmethod
    def issimilar(self, a: TS, b: TS) -> bool: ...

    @abstractmethod
    def relabel(self, stats: TS, relabel_indices: tuple[Field, ...]) -> TS: ...

    @abstractmethod
    def reorder(self, stats: TS, reorder_indices: tuple[Field, ...]) -> TS: ...


class BaseTensorStats(TensorStats):
    tensordef: TensorDef

    def __init__(self, tensor: Any, fields: tuple[Field, ...]):
        self.tensordef = TensorDef.from_tensor(tensor, fields)

    @property
    def dim_sizes(self) -> Mapping[Field, float]:
        return self.tensordef.dim_sizes

    @dim_sizes.setter
    def dim_sizes(self, value: Mapping[Field, float]):
        self.tensordef.dim_sizes = value

    def get_dim_size(self, idx: Field) -> float:
        return self.tensordef.get_dim_size(idx)

    @property
    def index_order(self) -> tuple[Field, ...]:
        return self.tensordef.index_order

    @index_order.setter
    def index_order(self, value: tuple[Field, ...]):
        self.tensordef.index_order = value

    @property
    def idxs(self) -> tuple[Field, ...]:
        return self.tensordef.index_order

    @property
    def fill_value(self) -> Any:
        return self.tensordef.fill_value

    @fill_value.setter
    def fill_value(self, value: Any):
        self.tensordef.fill_value = value

    def get_dim_space_size(self, idx: Iterable[Field]):
        return self.tensordef.get_dim_space_size(idx)
