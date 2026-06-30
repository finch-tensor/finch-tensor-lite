from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Generic, TypeVar

from finchlite.algebra import FinchOperator
from finchlite.finch_logic import Field, StatsFactory, TensorStats

from .tensor_def import TensorDef

TS = TypeVar("TS", bound="BaseTensorStats")


class BaseTensorStatsFactory(StatsFactory[TS], Generic[TS]):
    def __init__(self, stats_cls: type[TS]):
        self.stats_cls = stats_cls

    def __call__(self, tensor: Any, fields: tuple[Field, ...]) -> TS:
        return self.stats_cls(tensor, fields)

    @abstractmethod
    def copy_stats(self, stat: TS) -> TS: ...

    def mapjoin(self, op: FinchOperator, *args: TS) -> TS:
        def_args = [stat.tensordef for stat in args]
        new_def = TensorDef.mapjoin(op, *def_args)

        join_args: list[TS] = []
        union_args: list[TS] = []
        for s in args:
            if op.is_annihilator(s.fill_value):
                join_args.append(s)
            else:
                union_args.append(s)

        if union_args:
            join_args.append(self._mapjoin_union(new_def, op, union_args))
            # Add test cases - To test both join and union

        return self._mapjoin_join(new_def, op, join_args)

    @abstractmethod
    def _mapjoin_union(
        self, new_def: TensorDef, op: FinchOperator, union_args: list[TS]
    ) -> TS: ...

    @abstractmethod
    def _mapjoin_join(
        self, new_def: TensorDef, op: FinchOperator, join_args: list[TS]
    ) -> TS: ...

    @abstractmethod
    def aggregate(
        self,
        op: FinchOperator,
        init: Any | None,
        reduce_indices: tuple[Field, ...],
        stats: TS,
    ) -> TS: ...

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
